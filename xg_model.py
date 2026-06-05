"""
WSL Expected Goals (xG) Model
==============================
Single-stage XGBoost model: P(goal | shot context)

Key design choices
------------------
- Single-stage direct xG avoids two-stage error propagation
- XGBoost captures feature interactions automatically
- GroupKFold CV ensures no match leaks between train/test
- Isotonic calibration on a held-out set prevents leakage
- Symmetrised y-coordinate treats left/right shots identically
- Opta pre-computed distance (Q103) and angle (Q230) used alongside
  geometry re-derived from x,y to cross-validate and fill gaps
- Penalties handled separately (empirical mean)
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve, auc, brier_score_loss
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mplsoccer import Pitch, VerticalPitch

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not installed — pip install xgboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'xg_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PITCH_WIDTH    = 100
PITCH_HEIGHT   = 100
GOAL_Y_CENTRE  = 50.0
GOAL_X         = 100.0
GOAL_Y_LEFT    = 44.0
GOAL_Y_RIGHT   = 56.0
GOAL_WIDTH     = GOAL_Y_RIGHT - GOAL_Y_LEFT   # 12 Opta units

# Opta qualifier IDs found in this dataset
Q_PENALTY      = 9
Q_HEADER       = 15
Q_RIGHT_FOOT   = 20
Q_LEFT_FOOT    = 72
Q_REGULAR_PLAY = 22
Q_FAST_BREAK   = 23
Q_SET_PIECE    = 24
Q_FROM_CORNER  = 25
Q_FREE_KICK    = 26
Q_DIRECT_FK    = 28        # direct free kick attempt
Q_VOLLEY       = 108
Q_DEFLECTION   = 133
Q_ASSIST       = 210
Q_CROSS_ASSIST = 2
Q_BIG_CHANCE   = 233
Q_FIRST_TIME   = 200
Q_PULL_BACK    = 195
Q_DISTANCE     = 103       # Opta-computed distance (yards / Opta units)
Q_ANGLE        = 230       # Opta-computed angle  (degrees)
Q_END_X        = 102       # shot end x (where on goal)
Q_END_Y        = 231       # shot end y
Q_FOOT_SIDE    = 56        # 'Left','Right','Center'

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def open_angle(x: float, y: float) -> float:
    """Angle (degrees) subtended by goal mouth at shot position."""
    shot  = np.array([x, y], dtype=float)
    left  = np.array([GOAL_X, GOAL_Y_LEFT],  dtype=float)
    right = np.array([GOAL_X, GOAL_Y_RIGHT], dtype=float)
    v1, v2 = left - shot, right - shot
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def distance_to_goal(x: float, y: float) -> float:
    return float(np.sqrt((GOAL_X - x) ** 2 + (y - GOAL_Y_CENTRE) ** 2))


def in_six_yard_box(x: float, y: float) -> int:
    return int(x >= 94.2 and 36.8 <= y <= 63.2)


def in_penalty_area(x: float, y: float) -> int:
    return int(x >= 83.0 and 21.1 <= y <= 78.9)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def get_q_val(event: dict, qid: int):
    for q in event.get('qualifier', []):
        if q['qualifierId'] == qid:
            return q.get('value', 1)
    return None


def has_q(event: dict, qid: int) -> bool:
    return any(q['qualifierId'] == qid for q in event.get('qualifier', []))


def safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load_match(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    events = data.get('liveData', {}).get('event', data.get('event', []))

    rows = []
    # Track running score for match-state feature
    home_goals = away_goals = 0
    # Determine home/away contestant IDs from first events
    contestants = data.get('matchDetails', {})

    for e in events:
        tid = str(e.get('typeId'))

        # Update running score from goals (typeId 16)
        if tid == '16':
            # We'll update score AFTER recording the shot (score before shot)
            pass

        if tid not in ('13', '14', '15', '16'):
            continue

        x = safe_float(e.get('x'))
        y = safe_float(e.get('y'))
        if np.isnan(x) or np.isnan(y):
            continue

        # Opta pre-computed geometry
        opta_dist  = safe_float(get_q_val(e, Q_DISTANCE))
        opta_angle = safe_float(get_q_val(e, Q_ANGLE))

        # Re-derived geometry (symmetrised around goal centre)
        y_sym  = abs(y - GOAL_Y_CENTRE)
        dist   = distance_to_goal(x, y)
        angle  = open_angle(x, y)

        row = {
            'match_file':    os.path.basename(path),
            'season':        _season_from_path(path),
            'event_id':      e.get('id'),
            'player_id':     str(e.get('playerId', '')),
            'player_name':   e.get('playerName', ''),
            'contestant_id': str(e.get('contestantId', '')),
            'type_id':       int(tid),
            'period_id':     int(e.get('periodId') or 0),
            'time_min':      int(e.get('timeMin') or 0),
            'x':             x,
            'y':             y,
            'y_sym':         y_sym,

            # Geometry
            'distance':      dist,
            'log_distance':  np.log(max(dist, 0.5)),
            'angle':         angle,
            'angle_sin':     np.sin(np.radians(angle)),
            'opta_distance': opta_dist,
            'opta_angle':    opta_angle,

            # Shot zones
            'in_six_yard':   in_six_yard_box(x, y),
            'in_penalty_box': in_penalty_area(x, y),
            'central_y':     int(y_sym < GOAL_WIDTH / 2),  # between posts
            'dist_to_post':  abs(y_sym - GOAL_WIDTH / 2),

            # Body part
            'is_header':     int(has_q(e, Q_HEADER)),
            'is_right_foot': int(has_q(e, Q_RIGHT_FOOT)),
            'is_left_foot':  int(has_q(e, Q_LEFT_FOOT)),

            # Shot technique / context
            'is_volley':     int(has_q(e, Q_VOLLEY)),
            'is_deflected':  int(has_q(e, Q_DEFLECTION)),
            'is_first_time': int(has_q(e, Q_FIRST_TIME)),
            'is_big_chance': int(has_q(e, Q_BIG_CHANCE)),
            'is_fast_break': int(has_q(e, Q_FAST_BREAK)),
            'is_from_corner': int(has_q(e, Q_FROM_CORNER)),
            'is_free_kick':  int(has_q(e, Q_DIRECT_FK)),
            'is_penalty':    int(has_q(e, Q_PENALTY)),
            'is_set_piece':  int(has_q(e, Q_SET_PIECE)),
            'is_open_play':  int(has_q(e, Q_REGULAR_PLAY)),

            # Outcome flags
            'is_goal':       int(tid == '16'),
            'is_on_target':  int(tid in ('15', '16')),
            'is_blocked':    int(tid == '15'),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _season_from_path(path: str) -> str:
    parts = path.replace('\\', '/').split('/')
    for part in parts:
        if part.startswith('WSL'):
            return part
    return 'Unknown'


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL MATCHES
# ─────────────────────────────────────────────────────────────────────────────
json_files = glob.glob(os.path.join(PROJECT_ROOT, '**', '*.json'), recursive=True)
print(f"Found {len(json_files)} JSON files.")

frames = []
for path in json_files:
    try:
        frames.append(load_match(path))
    except Exception as exc:
        print(f"  Skipping {os.path.basename(path)}: {exc}")

shots = pd.concat(frames, ignore_index=True)
print(f"Loaded {len(shots):,} shots  |  Goals: {shots['is_goal'].sum():,}  "
      f"({shots['is_goal'].mean()*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    # Geometry
    'x', 'y_sym',
    'distance', 'log_distance',
    'angle', 'angle_sin',
    'dist_to_post',
    # Zones
    'in_six_yard', 'in_penalty_box', 'central_y',
    # Body part
    'is_header', 'is_right_foot', 'is_left_foot',
    # Context
    'is_volley', 'is_deflected', 'is_first_time',
    'is_big_chance', 'is_fast_break', 'is_from_corner',
    'is_free_kick', 'is_penalty', 'is_set_piece', 'is_open_play',
]

# Separate penalties (xG = empirical mean, position fixed)
pen_mask  = shots['is_penalty'] == 1
nonpen    = shots[~pen_mask].copy().reset_index(drop=True)

pen_xg = shots.loc[pen_mask, 'is_goal'].mean() if pen_mask.sum() > 0 else 0.76
shots.loc[pen_mask, 'xg'] = pen_xg
print(f"Penalty xG (empirical): {pen_xg:.3f}  n={pen_mask.sum()}")

X      = nonpen[FEATURES].fillna(0).astype(float)
y      = nonpen['is_goal'].astype(int)
groups = nonpen['match_file']

print(f"Feature matrix: {X.shape[0]:,} shots × {X.shape[1]} features")

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / CALIBRATION / TEST SPLIT  (60 / 20 / 20, match-grouped)
# ─────────────────────────────────────────────────────────────────────────────
gss_outer = GroupShuffleSplit(1, test_size=0.20, random_state=42)
dev_idx, test_idx = next(gss_outer.split(X, y, groups=groups))

X_dev, X_test = X.iloc[dev_idx], X.iloc[test_idx]
y_dev, y_test = y.iloc[dev_idx], y.iloc[test_idx]
g_dev         = groups.iloc[dev_idx]

gss_inner = GroupShuffleSplit(1, test_size=0.25, random_state=0)
tr_idx, cal_idx = next(gss_inner.split(X_dev, y_dev, groups=g_dev))

X_tr,  X_cal  = X_dev.iloc[tr_idx],  X_dev.iloc[cal_idx]
y_tr,  y_cal  = y_dev.iloc[tr_idx],  y_dev.iloc[cal_idx]
g_tr          = g_dev.iloc[tr_idx]

print(f"Train: {len(X_tr):,}  |  Cal: {len(X_cal):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL — XGBoost (fallback: Logistic Regression)
# ─────────────────────────────────────────────────────────────────────────────
group_cv = GroupKFold(n_splits=5)

if HAS_XGB:
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    param_grid = {
        'max_depth':        [4, 6],
        'learning_rate':    [0.05, 0.1],
        'n_estimators':     [300, 500],
        'subsample':        [0.8],
        'colsample_bytree': [0.7, 0.9],
        'min_child_weight': [5, 10],
        'scale_pos_weight': [spw],
    }
    base_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    grid = GridSearchCV(base_clf, param_grid,
                        scoring='roc_auc', cv=group_cv,
                        n_jobs=-1, verbose=1)
    grid.fit(X_tr, y_tr, groups=g_tr)
    print(f"Best params:  {grid.best_params_}")
    print(f"CV AUC:       {grid.best_score_:.4f}")

    best_clf = grid.best_estimator_

else:
    print("Falling back to Logistic Regression (install xgboost for better results)")
    base_clf  = LogisticRegression(max_iter=5000, class_weight='balanced', solver='lbfgs')
    grid      = GridSearchCV(base_clf, {'C': [0.1, 0.3, 1, 3, 10]},
                             scoring='roc_auc', cv=group_cv, n_jobs=-1)
    grid.fit(X_tr, y_tr, groups=g_tr)
    best_clf  = LogisticRegression(
        max_iter=5000, class_weight='balanced',
        C=grid.best_params_['C'], solver='lbfgs'
    )
    best_clf.fit(X_tr, y_tr)

# Isotonic calibration on held-out calibration set (no leakage)
raw_cal_probs = best_clf.predict_proba(X_cal)[:, 1]
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(raw_cal_probs, y_cal)

class CalibratedModel:
    """Thin wrapper: XGBoost + isotonic calibration."""
    def __init__(self, clf, iso):
        self.clf = clf
        self.iso = iso
    def predict_proba(self, X):
        raw = self.clf.predict_proba(X)[:, 1]
        cal = self.iso.predict(raw)
        return np.column_stack([1 - cal, cal])

cal_model = CalibratedModel(best_clf, iso_reg)

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────
xg_test  = cal_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, xg_test)
auc_score   = auc(fpr, tpr)
brier       = brier_score_loss(y_test, xg_test)
print(f"\nTest AUC:    {auc_score:.4f}")
print(f"Brier score: {brier:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT ON ALL NON-PENALTY SHOTS
# ─────────────────────────────────────────────────────────────────────────────
xg_all = cal_model.predict_proba(X)[:, 1]
nonpen['xg'] = xg_all
shots.loc[~pen_mask, 'xg'] = nonpen['xg'].values

# Scaled 0-100 index
mn, mx = shots['xg'].min(), shots['xg'].max()
shots['xg_index'] = (shots['xg'] - mn) / max(mx - mn, 1e-9) * 100

# ─────────────────────────────────────────────────────────────────────────────
# SAVE ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
shots.to_csv(os.path.join(OUTPUT_DIR, 'all_shots_xg.csv'), index=False)
joblib.dump(cal_model, os.path.join(OUTPUT_DIR, 'xg_model.pkl'))
print(f"\nSaved shots CSV and model to {OUTPUT_DIR}/")

# Per-player xG summary
player_xg = (
    shots.groupby(['player_name', 'season'])
    .agg(
        shots_total=('xg', 'count'),
        goals=('is_goal', 'sum'),
        xg_total=('xg', 'sum'),
        xg_per_shot=('xg', 'mean'),
        on_target=('is_on_target', 'sum'),
    )
    .reset_index()
)
player_xg['goals_minus_xg']  = player_xg['goals'] - player_xg['xg_total']
player_xg['conversion_rate'] = player_xg['goals'] / player_xg['shots_total'].clip(lower=1)
player_xg.to_csv(os.path.join(OUTPUT_DIR, 'player_xg_summary.csv'), index=False)

# Beth Mead specific
beth = player_xg[player_xg['player_name'].str.contains('Mead', na=False)].sort_values('season')
print("\n── Beth Mead xG by Season ──")
print(beth[['season', 'shots_total', 'goals', 'xg_total', 'goals_minus_xg']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
PITCH_BG   = '#0d1117'
PITCH_LINE = '#c9d1d9'
FIG_BG     = '#0e1117'
C_BLUE     = '#58a6ff'
C_ORANGE   = '#f78166'
C_GREEN    = '#3fb950'
C_YELLOW   = '#e3b341'

plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 14), facecolor=FIG_BG)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. ROC curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(fpr, tpr, lw=2, color=C_BLUE,
         label=f'xG model  AUC={auc_score:.3f}')
ax1.plot([0, 1], [0, 1], 'w--', lw=1, alpha=0.4)
ax1.set_xlabel('FPR', fontsize=9)
ax1.set_ylabel('TPR', fontsize=9)
ax1.set_title('ROC — Test Set', fontsize=10)
ax1.legend(fontsize=8)
ax1.set_facecolor(FIG_BG)

# 2. Calibration curve
ax2 = fig.add_subplot(gs[0, 1])
prob_true, prob_pred = calibration_curve(y_test, xg_test, n_bins=10, strategy='quantile')
ax2.plot(prob_pred, prob_true, marker='o', lw=2, color=C_GREEN, label='Calibration')
ax2.plot([0, 1], [0, 1], 'w--', lw=1, alpha=0.4, label='Perfect')
ax2.set_xlabel('Mean predicted xG', fontsize=9)
ax2.set_ylabel('Actual goal rate', fontsize=9)
ax2.set_title(f'Calibration  (Brier={brier:.4f})', fontsize=10)
ax2.legend(fontsize=8)
ax2.set_facecolor(FIG_BG)

# 3. xG distribution: goals vs non-goals (test set)
ax3 = fig.add_subplot(gs[0, 2])
xg_test_df = pd.DataFrame({'xg': xg_test, 'is_goal': y_test.values})
bins = np.linspace(0, 1, 25)
ax3.hist(xg_test_df.loc[xg_test_df['is_goal'] == 0, 'xg'],
         bins=bins, alpha=0.55, color=C_ORANGE, density=True, label='No goal')
ax3.hist(xg_test_df.loc[xg_test_df['is_goal'] == 1, 'xg'],
         bins=bins, alpha=0.55, color=C_BLUE,   density=True, label='Goal')
ax3.set_xlabel('xG', fontsize=9)
ax3.set_title('xG Distribution', fontsize=10)
ax3.legend(fontsize=8)
ax3.set_facecolor(FIG_BG)

# 4. Feature importance
ax4 = fig.add_subplot(gs[1, :2])
if HAS_XGB and isinstance(best_clf, xgb.XGBClassifier):
    imp = pd.Series(best_clf.feature_importances_, index=FEATURES)
else:
    imp = pd.Series(np.abs(best_clf.coef_[0]), index=FEATURES)
imp = imp.sort_values(ascending=True).tail(15)
bars = ax4.barh(imp.index, imp.values, color=C_BLUE, alpha=0.8)
ax4.set_title('Feature Importance (gain)', fontsize=10)
ax4.set_facecolor(FIG_BG)
for bar, val in zip(bars, imp.values):
    ax4.text(val + imp.values.max() * 0.01, bar.get_y() + bar.get_height() / 2,
             f'{val:.3f}', va='center', fontsize=7, color='white')

# 5. Pitch heatmap of average xG
ax5 = fig.add_subplot(gs[1, 2])
pitch = Pitch(pitch_type='opta', pitch_color=PITCH_BG,
              line_color=PITCH_LINE, linewidth=0.8)
pitch.draw(ax=ax5)
bin_stat = pitch.bin_statistic(
    shots['x'], shots['y'],
    values=shots['xg'], statistic='mean', bins=(12, 8)
)
pitch.heatmap(bin_stat, ax=ax5, cmap='magma', vmin=0, vmax=0.25)
pitch.label_heatmap(bin_stat, ax=ax5, color='white', fontsize=5)
ax5.set_title('Mean xG by Zone', fontsize=10)

# 6. Beth Mead: Goals vs xG by season
ax6 = fig.add_subplot(gs[2, :])
if len(beth) > 0:
    x_pos  = np.arange(len(beth))
    width  = 0.35
    ax6.bar(x_pos - width/2, beth['goals'].values,    width, color=C_YELLOW, alpha=0.85, label='Actual Goals')
    ax6.bar(x_pos + width/2, beth['xg_total'].values, width, color=C_BLUE,   alpha=0.85, label='xG')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(beth['season'].values, rotation=30, ha='right', fontsize=8)
    ax6.set_title('Beth Mead — Goals vs xG by Season', fontsize=11)
    ax6.legend(fontsize=9)
    ax6.set_facecolor(FIG_BG)
    for i, row in enumerate(beth.itertuples()):
        diff  = row.goals - row.xg_total
        color = C_GREEN if diff > 0 else C_ORANGE
        ax6.text(i, max(row.goals, row.xg_total) + 0.2,
                 f'{diff:+.1f}', ha='center', fontsize=8, color=color)
else:
    ax6.text(0.5, 0.5, 'No Beth Mead shots found', ha='center', va='center',
             fontsize=12, transform=ax6.transAxes)
    ax6.set_facecolor(FIG_BG)

fig.suptitle('WSL xG Model — Evaluation Dashboard', fontsize=14,
             fontweight='bold', color='white', y=0.98)

dashboard_path = os.path.join(OUTPUT_DIR, 'xg_dashboard.png')
plt.savefig(dashboard_path, dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close()
print(f"Dashboard saved to {dashboard_path}")

# ─────────────────────────────────────────────────────────────────────────────
# SHAP (if available)
# ─────────────────────────────────────────────────────────────────────────────
if HAS_SHAP and HAS_XGB:
    explainer   = shap.TreeExplainer(best_clf)
    sample      = X_test.sample(min(2000, len(X_test)), random_state=0)
    shap_vals   = explainer.shap_values(sample)
    shap_fig, _ = plt.subplots(figsize=(10, 7), facecolor=FIG_BG)
    shap.summary_plot(shap_vals, sample, feature_names=FEATURES,
                      show=False, max_display=15)
    plt.savefig(os.path.join(OUTPUT_DIR, 'xg_shap.png'),
                dpi=150, bbox_inches='tight', facecolor=FIG_BG)
    plt.close()
    print("SHAP plot saved.")

print("\nDone.")
