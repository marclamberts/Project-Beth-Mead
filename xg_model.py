"""
WSL Three-Model xG System
==========================

Model 1 — Pre-Shot xG
  P(goal | shot position + body part + technique + situation)
  Trained on all non-penalty shots.

Model 2 — Post-Shot xG (psxG)
  P(goal | everything above + WHERE the ball was aimed in the goal frame)
  Uses Opta qualifiers for goal-frame placement:
    Q102 = lateral position within goal mouth (y-axis, 44–56 Opta units)
    Q231 = shot height (z-axis, 0–100 scale)
    Q146/Q147 = end-position for saves (pitch coords)
  Post-shot xG is available only for shots with placement data.
  It is a better measure of shot quality than xG because it accounts
  for WHERE the shooter aimed, not just the shot context.

Model 3 — Shot Situation Danger
  P(goal | build-up context only — NO shot geometry)
  Answers: "How dangerous was this attack BEFORE the shot location
  was known?" High danger = counter-attack / pull-back / big chance.
  Useful for chance-creation analysis independent of execution.

All models
  • XGBoost with GroupKFold CV (no match leakage)
  • Isotonic calibration on a held-out fold
  • Penalties handled separately (empirical mean)
"""

import os, glob, json, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve, auc, brier_score_loss
import joblib
import matplotlib
matplotlib.use('Agg')
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

GOAL_Y_CENTRE  = 50.0
GOAL_X         = 100.0
GOAL_Y_LEFT    = 44.0
GOAL_Y_RIGHT   = 56.0
GOAL_WIDTH     = GOAL_Y_RIGHT - GOAL_Y_LEFT   # 12 Opta units

# Opta qualifier IDs
Q_PENALTY      = 9
Q_HEADER       = 15
Q_RIGHT_FOOT   = 20
Q_LEFT_FOOT    = 72
Q_REGULAR_PLAY = 22
Q_FAST_BREAK   = 23
Q_SET_PIECE    = 24
Q_FROM_CORNER  = 25
Q_FREE_KICK    = 26
Q_DIRECT_FK    = 28
Q_VOLLEY       = 108
Q_DEFLECTION   = 133
Q_PULL_BACK    = 195
Q_BIG_CHANCE   = 233
Q_FIRST_TIME   = 200
Q_DISTANCE     = 103
Q_ANGLE        = 230   # Opta percentage: 100 = on-target centre
Q_GOAL_Y       = 102   # lateral goal-frame position (y ≈ 44–56)
Q_GOAL_HEIGHT  = 231   # shot height (z, 0–100)
Q_SAVE_END_X   = 146   # end pitch-x for saves/off-target
Q_SAVE_END_Y   = 147   # end pitch-y for saves/off-target

# Appearance colours (matches app.py palette)
FIG_BG   = '#0e1117'
PITCH_BG = '#0d1117'
LINE_COL = '#c9d1d9'
C_BLUE   = '#58a6ff'
C_ORANGE = '#f78166'
C_GREEN  = '#3fb950'
C_YELLOW = '#e3b341'
C_PURPLE = '#bc8cff'

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def open_angle(x: float, y: float) -> float:
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

def get_q(event: dict, qid: int):
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
    for e in events:
        tid = str(e.get('typeId'))
        if tid not in ('13', '14', '15', '16'):
            continue
        x = safe_float(e.get('x'))
        y = safe_float(e.get('y'))
        if np.isnan(x) or np.isnan(y):
            continue

        # ── Post-shot placement ───────────────────────────────────────────────
        # Goal-frame lateral (y): Q102 available for goals + posts + most saves
        goal_y_raw    = safe_float(get_q(e, Q_GOAL_Y))    # 44–56 scale
        goal_h_raw    = safe_float(get_q(e, Q_GOAL_HEIGHT)) # 0–100 scale
        save_end_x    = safe_float(get_q(e, Q_SAVE_END_X))
        save_end_y    = safe_float(get_q(e, Q_SAVE_END_Y))

        # Normalise goal_y_raw → -1…+1 (–1 = far post, +1 = far post other side)
        if not np.isnan(goal_y_raw):
            goal_y_norm = (goal_y_raw - GOAL_Y_CENTRE) / (GOAL_WIDTH / 2)  # –1…+1
        else:
            goal_y_norm = np.nan

        # Height: 0 = ground, 100 = top. Normalise to 0–1.
        goal_h_norm = goal_h_raw / 100.0 if not np.isnan(goal_h_raw) else np.nan

        # Is the ball aimed within the goal frame?
        in_frame = int(
            (not np.isnan(goal_y_raw) and GOAL_Y_LEFT <= goal_y_raw <= GOAL_Y_RIGHT)
            or tid in ('15', '16')
        )

        # Distance from goal-frame centre (lateral + height combined)
        if not np.isnan(goal_y_norm) and not np.isnan(goal_h_norm):
            goal_frame_dist = np.sqrt(goal_y_norm ** 2 + (goal_h_norm - 0.35) ** 2)
        elif not np.isnan(goal_y_norm):
            goal_frame_dist = abs(goal_y_norm)
        else:
            goal_frame_dist = np.nan

        # Corner zone: high + wide = harder for goalkeeper
        if not np.isnan(goal_y_norm) and not np.isnan(goal_h_norm):
            corner_zone = int(abs(goal_y_norm) > 0.55 or goal_h_norm > 0.6)
        else:
            corner_zone = np.nan

        dist  = distance_to_goal(x, y)
        angle = open_angle(x, y)
        y_sym = abs(y - GOAL_Y_CENTRE)

        row = {
            'match_file':    os.path.basename(path),
            'season':        _season_from_path(path),
            'player_id':     str(e.get('playerId', '')),
            'player_name':   e.get('playerName', ''),
            'contestant_id': str(e.get('contestantId', '')),
            'type_id':       int(tid),
            'period_id':     int(e.get('periodId') or 0),
            'time_min':      int(e.get('timeMin') or 0),
            'x':             x,
            'y':             y,
            'y_sym':         y_sym,

            # Pre-shot geometry
            'distance':      dist,
            'log_distance':  np.log(max(dist, 0.5)),
            'angle':         angle,
            'angle_sin':     np.sin(np.radians(angle)),
            'in_six_yard':   in_six_yard_box(x, y),
            'in_penalty_box': in_penalty_area(x, y),
            'central_y':     int(y_sym < GOAL_WIDTH / 2),
            'dist_to_post':  abs(y_sym - GOAL_WIDTH / 2),

            # Post-shot placement
            'goal_y_raw':    goal_y_raw,
            'goal_y_norm':   goal_y_norm,
            'goal_h_raw':    goal_h_raw,
            'goal_h_norm':   goal_h_norm,
            'in_frame':      in_frame,
            'goal_frame_dist': goal_frame_dist,
            'corner_zone':   corner_zone,
            'save_end_x':    save_end_x,
            'save_end_y':    save_end_y,

            # Body part
            'is_header':     int(has_q(e, Q_HEADER)),
            'is_right_foot': int(has_q(e, Q_RIGHT_FOOT)),
            'is_left_foot':  int(has_q(e, Q_LEFT_FOOT)),

            # Shot technique / situation
            'is_volley':      int(has_q(e, Q_VOLLEY)),
            'is_deflected':   int(has_q(e, Q_DEFLECTION)),
            'is_first_time':  int(has_q(e, Q_FIRST_TIME)),
            'is_big_chance':  int(has_q(e, Q_BIG_CHANCE)),
            'is_fast_break':  int(has_q(e, Q_FAST_BREAK)),
            'is_from_corner': int(has_q(e, Q_FROM_CORNER)),
            'is_free_kick':   int(has_q(e, Q_DIRECT_FK)),
            'is_penalty':     int(has_q(e, Q_PENALTY)),
            'is_set_piece':   int(has_q(e, Q_SET_PIECE)),
            'is_open_play':   int(has_q(e, Q_REGULAR_PLAY)),
            'is_pull_back':   int(has_q(e, Q_PULL_BACK)),

            # Outcomes
            'is_goal':       int(tid == '16'),
            'is_on_target':  int(tid in ('15', '16')),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _season_from_path(path: str) -> str:
    for part in path.replace('\\', '/').split('/'):
        if part.startswith('WSL'):
            return part
    return 'Unknown'


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATED MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class CalibratedXGB:
    """XGBoost + isotonic regression calibration, fitted on a held-out fold."""
    def __init__(self, clf, iso: IsotonicRegression):
        self.clf = clf
        self.iso = iso

    def predict_proba(self, X):
        raw = self.clf.predict_proba(X)[:, 1]
        cal = self.iso.predict(raw)
        return np.column_stack([1 - cal, cal])


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN + EVALUATE ONE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_tr, y_tr, g_tr, X_cal, y_cal,
                X_test, y_test, label: str):
    """Fit XGBoost (or LR fallback), calibrate, evaluate, return model."""
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
        base = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            use_label_encoder=False, random_state=42, verbosity=0,
        )
        grid = GridSearchCV(base, param_grid, scoring='roc_auc',
                            cv=group_cv, n_jobs=-1, verbose=0)
        grid.fit(X_tr, y_tr, groups=g_tr)
        print(f"  [{label}] Best params: {grid.best_params_}")
        print(f"  [{label}] CV AUC: {grid.best_score_:.4f}")
        clf = grid.best_estimator_
    else:
        base = LogisticRegression(max_iter=5000, class_weight='balanced', solver='lbfgs')
        grid = GridSearchCV(base, {'C': [0.1, 1, 10]},
                            scoring='roc_auc', cv=group_cv, n_jobs=-1)
        grid.fit(X_tr, y_tr, groups=g_tr)
        clf = LogisticRegression(
            max_iter=5000, class_weight='balanced',
            C=grid.best_params_['C'], solver='lbfgs'
        ).fit(X_tr, y_tr)

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(clf.predict_proba(X_cal)[:, 1], y_cal)
    model = CalibratedXGB(clf, iso)

    proba_test = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    score_auc   = auc(fpr, tpr)
    score_brier = brier_score_loss(y_test, proba_test)
    print(f"  [{label}] Test AUC: {score_auc:.4f}  Brier: {score_brier:.4f}")
    return model, proba_test, fpr, tpr, score_auc, score_brier


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
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
# PENALTIES — empirical mean only
# ─────────────────────────────────────────────────────────────────────────────
pen_mask  = shots['is_penalty'] == 1
nonpen    = shots[~pen_mask].copy().reset_index(drop=True)

pen_xg = shots.loc[pen_mask, 'is_goal'].mean() if pen_mask.sum() > 0 else 0.76
shots.loc[pen_mask, 'xg'] = pen_xg
shots.loc[pen_mask, 'psxg'] = pen_xg
shots.loc[pen_mask, 'situation_danger'] = pen_xg
print(f"Penalty xG (empirical): {pen_xg:.3f}  n={pen_mask.sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED SPLIT  (all three models use the same train/cal/test indices)
# ─────────────────────────────────────────────────────────────────────────────
XG_FEATURES = [
    'x', 'y_sym', 'distance', 'log_distance', 'angle', 'angle_sin',
    'dist_to_post', 'in_six_yard', 'in_penalty_box', 'central_y',
    'is_header', 'is_right_foot', 'is_left_foot',
    'is_volley', 'is_deflected', 'is_first_time', 'is_big_chance',
    'is_fast_break', 'is_from_corner', 'is_free_kick', 'is_set_piece',
    'is_open_play', 'is_pull_back',
]

SITUATION_FEATURES = [
    # Build-up context — no shot geometry
    'is_big_chance', 'is_fast_break', 'is_from_corner',
    'is_free_kick', 'is_set_piece', 'is_open_play', 'is_pull_back',
    'is_first_time', 'is_header',
    'period_id', 'time_min',
]

X_base   = nonpen[XG_FEATURES].fillna(0).astype(float)
X_sit    = nonpen[SITUATION_FEATURES].fillna(0).astype(float)
y        = nonpen['is_goal'].astype(int)
groups   = nonpen['match_file']

# Group-aware train/cal/test split
gss_outer = GroupShuffleSplit(1, test_size=0.20, random_state=42)
dev_idx, test_idx = next(gss_outer.split(X_base, y, groups=groups))

X_base_dev,  X_base_test  = X_base.iloc[dev_idx],  X_base.iloc[test_idx]
X_sit_dev,   X_sit_test   = X_sit.iloc[dev_idx],   X_sit.iloc[test_idx]
y_dev, y_test              = y.iloc[dev_idx],        y.iloc[test_idx]
g_dev                      = groups.iloc[dev_idx]

gss_inner = GroupShuffleSplit(1, test_size=0.25, random_state=0)
tr_idx, cal_idx = next(gss_inner.split(X_base_dev, y_dev, groups=g_dev))

y_tr,  y_cal  = y_dev.iloc[tr_idx],  y_dev.iloc[cal_idx]
g_tr          = g_dev.iloc[tr_idx]

print(f"Train: {len(y_tr):,}  Cal: {len(y_cal):,}  Test: {len(y_test):,}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — Pre-Shot xG
# ─────────────────────────────────────────────────────────────────────────────
print("── Model 1: Pre-Shot xG ──")
xg_model, xg_test_proba, xg_fpr, xg_tpr, xg_auc, xg_brier = train_model(
    X_base_dev.iloc[tr_idx],  y_tr, g_tr,
    X_base_dev.iloc[cal_idx], y_cal,
    X_base_test,              y_test,
    label='xG',
)
nonpen['xg'] = xg_model.predict_proba(X_base)[:, 1]
shots.loc[~pen_mask, 'xg'] = nonpen['xg'].values
joblib.dump(xg_model, os.path.join(OUTPUT_DIR, 'model_xg.pkl'))

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — Post-Shot xG
#
# Post-shot xG (psxG) adds goal-frame placement to refine the prediction.
# It requires knowing WHERE the ball was aimed → only shots with placement
# data are used to train the model.
# For shots missing placement we fall back to pre-shot xG.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Model 2: Post-Shot xG ──")

PSXG_EXTRA = [
    'goal_y_norm',     # lateral position in goal mouth (–1…+1)
    'goal_h_norm',     # height (0–1)
    'goal_frame_dist', # distance from goal-frame centre (lateral + height)
    'corner_zone',     # 1 if aimed at a difficult corner
    # NOTE: is_on_target and in_frame intentionally excluded — target leakage
]
PSXG_FEATURES = XG_FEATURES + PSXG_EXTRA

# Rows with placement data
has_placement = nonpen[['goal_y_norm', 'goal_h_norm']].notna().any(axis=1)
nonpen_ps     = nonpen[has_placement].copy().reset_index(drop=True)
X_ps          = nonpen_ps[PSXG_FEATURES].fillna(0).astype(float)
y_ps          = nonpen_ps['is_goal'].astype(int)
g_ps          = nonpen_ps['match_file']

print(f"  Shots with placement data: {len(nonpen_ps):,} / {len(nonpen):,}")

gss_ps = GroupShuffleSplit(1, test_size=0.20, random_state=42)
dev_ps, test_ps = next(gss_ps.split(X_ps, y_ps, groups=g_ps))

X_ps_dev,  X_ps_test  = X_ps.iloc[dev_ps],  X_ps.iloc[test_ps]
y_ps_dev,  y_ps_test  = y_ps.iloc[dev_ps],  y_ps.iloc[test_ps]
g_ps_dev              = g_ps.iloc[dev_ps]

gss_ps_in = GroupShuffleSplit(1, test_size=0.25, random_state=0)
tr_ps, cal_ps = next(gss_ps_in.split(X_ps_dev, y_ps_dev, groups=g_ps_dev))

psxg_model, psxg_test_proba, psxg_fpr, psxg_tpr, psxg_auc, psxg_brier = train_model(
    X_ps_dev.iloc[tr_ps],  y_ps_dev.iloc[tr_ps],  g_ps_dev.iloc[tr_ps],
    X_ps_dev.iloc[cal_ps], y_ps_dev.iloc[cal_ps],
    X_ps_test,             y_ps_test,
    label='psxG',
)

# Predict on all shots — fill missing placement with pre-shot xG
X_all_ps            = nonpen[PSXG_FEATURES].fillna(0).astype(float)
nonpen['psxg']      = np.nan
nonpen.loc[has_placement, 'psxg'] = psxg_model.predict_proba(
    X_all_ps[has_placement]
)[:, 1]
# Fallback for shots without placement data
nonpen['psxg'] = nonpen['psxg'].fillna(nonpen['xg'])
shots.loc[~pen_mask, 'psxg'] = nonpen['psxg'].values
joblib.dump(psxg_model, os.path.join(OUTPUT_DIR, 'model_psxg.pkl'))

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — Shot Situation Danger
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Model 3: Shot Situation Danger ──")
sit_model, sit_test_proba, sit_fpr, sit_tpr, sit_auc, sit_brier = train_model(
    X_sit_dev.iloc[tr_idx],  y_tr, g_tr,
    X_sit_dev.iloc[cal_idx], y_cal,
    X_sit_test,              y_test,
    label='Situation',
)
nonpen['situation_danger'] = sit_model.predict_proba(X_sit)[:, 1]
shots.loc[~pen_mask, 'situation_danger'] = nonpen['situation_danger'].values
joblib.dump(sit_model, os.path.join(OUTPUT_DIR, 'model_situation.pkl'))

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED METRICS
# ─────────────────────────────────────────────────────────────────────────────
# Shot quality over expectation: psxG – xG
# Positive = aimed better than the average shot from that position
shots['placement_quality'] = shots['psxg'] - shots['xg']

# Situation vs Execution gap: xG – situation_danger
# Positive = shooter found a better position than the average attack would yield
shots['execution_quality'] = shots['xg'] - shots['situation_danger']

# Scaled 0–100 indices for dashboard use
for col in ['xg', 'psxg', 'situation_danger']:
    mn, mx = shots[col].min(), shots[col].max()
    shots[f'{col}_index'] = (shots[col] - mn) / max(mx - mn, 1e-9) * 100

# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
shots.to_csv(os.path.join(OUTPUT_DIR, 'all_shots_xg.csv'), index=False)

player_summary = (
    shots.groupby(['player_name', 'season'])
    .agg(
        shots_n=('xg', 'count'),
        goals=('is_goal', 'sum'),
        xg=('xg', 'sum'),
        psxg=('psxg', 'sum'),
        sit_danger=('situation_danger', 'sum'),
        on_target=('is_on_target', 'sum'),
    )
    .reset_index()
)
player_summary['goals_minus_xg']   = player_summary['goals'] - player_summary['xg']
player_summary['goals_minus_psxg'] = player_summary['goals'] - player_summary['psxg']
player_summary['xg_per_shot']      = player_summary['xg'] / player_summary['shots_n'].clip(lower=1)
player_summary['shot_on_target_%'] = player_summary['on_target'] / player_summary['shots_n'].clip(lower=1)
player_summary.to_csv(os.path.join(OUTPUT_DIR, 'player_xg_summary.csv'), index=False)

beth = player_summary[
    player_summary['player_name'].str.contains('Mead', na=False)
].sort_values('season')

print("\n── Beth Mead — Three-Model Summary ──")
print(beth[['season','shots_n','goals','xg','psxg','sit_danger',
            'goals_minus_xg','goals_minus_psxg']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 16), facecolor=FIG_BG)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

# ── Row 0: ROC, Calibration, xG Distribution, psxG Distribution ──────────────
ax_roc = fig.add_subplot(gs[0, 0])
ax_roc.plot(xg_fpr,  xg_tpr,  lw=2, color=C_BLUE,   label=f'Pre-Shot xG   AUC={xg_auc:.3f}')
ax_roc.plot(psxg_fpr, psxg_tpr, lw=2, color=C_GREEN, label=f'Post-Shot psxG AUC={psxg_auc:.3f}')
ax_roc.plot(sit_fpr, sit_tpr,  lw=2, color=C_ORANGE, linestyle='--',
            label=f'Situation Danger AUC={sit_auc:.3f}')
ax_roc.plot([0,1],[0,1],'w--',lw=1,alpha=0.3)
ax_roc.set(xlabel='FPR', ylabel='TPR', title='ROC — All Three Models', facecolor=FIG_BG)
ax_roc.legend(fontsize=7)

ax_cal = fig.add_subplot(gs[0, 1])
for proba, label, color in [
    (xg_test_proba,   'Pre-Shot xG',    C_BLUE),
    (psxg_test_proba, 'Post-Shot psxG', C_GREEN),
    (sit_test_proba,  'Situation',      C_ORANGE),
]:
    y_ref = y_ps_test if label == 'Post-Shot psxG' else y_test
    pt, pp = calibration_curve(y_ref, proba, n_bins=10, strategy='quantile')
    ax_cal.plot(pp, pt, marker='o', lw=2, color=color, label=label, markersize=4)
ax_cal.plot([0,1],[0,1],'w--',lw=1,alpha=0.3)
ax_cal.set(xlabel='Mean predicted', ylabel='Actual rate',
           title='Calibration Curves', facecolor=FIG_BG)
ax_cal.legend(fontsize=7)

ax_xg_dist = fig.add_subplot(gs[0, 2])
bins = np.linspace(0, 1, 25)
xg_test_df = pd.DataFrame({'xg': xg_test_proba, 'goal': y_test.values})
ax_xg_dist.hist(xg_test_df[xg_test_df['goal']==0]['xg'],
                bins=bins, alpha=0.5, color=C_ORANGE, density=True, label='No goal')
ax_xg_dist.hist(xg_test_df[xg_test_df['goal']==1]['xg'],
                bins=bins, alpha=0.5, color=C_BLUE,   density=True, label='Goal')
ax_xg_dist.set(xlabel='Pre-Shot xG', title='xG Distribution', facecolor=FIG_BG)
ax_xg_dist.legend(fontsize=7)

ax_psxg_dist = fig.add_subplot(gs[0, 3])
ps_df = pd.DataFrame({'psxg': psxg_test_proba, 'goal': y_ps_test.values})
ax_psxg_dist.hist(ps_df[ps_df['goal']==0]['psxg'],
                  bins=bins, alpha=0.5, color=C_ORANGE, density=True, label='No goal')
ax_psxg_dist.hist(ps_df[ps_df['goal']==1]['psxg'],
                  bins=bins, alpha=0.5, color=C_GREEN,  density=True, label='Goal')
ax_psxg_dist.set(xlabel='Post-Shot psxG', title='psxG Distribution', facecolor=FIG_BG)
ax_psxg_dist.legend(fontsize=7)

# ── Row 1: Feature importances (xG, psxG, Situation) + Pitch heatmap ─────────
for ax_col, (model, feat_names, title, color) in enumerate([
    (xg_model,   XG_FEATURES,       'xG Feature Importance',       C_BLUE),
    (psxg_model, PSXG_FEATURES,     'psxG Feature Importance',     C_GREEN),
    (sit_model,  SITUATION_FEATURES,'Situation Danger Importance',  C_ORANGE),
]):
    ax = fig.add_subplot(gs[1, ax_col])
    if HAS_XGB and hasattr(model.clf, 'feature_importances_'):
        imp = pd.Series(model.clf.feature_importances_, index=feat_names)
    else:
        imp = pd.Series(np.abs(model.clf.coef_[0]), index=feat_names)
    imp = imp.sort_values(ascending=True).tail(10)
    ax.barh(imp.index, imp.values, color=color, alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_facecolor(FIG_BG)
    ax.tick_params(labelsize=7)

ax_heatmap = fig.add_subplot(gs[1, 3])
pitch = Pitch(pitch_type='opta', pitch_color=PITCH_BG,
              line_color=LINE_COL, linewidth=0.8)
pitch.draw(ax=ax_heatmap)
bin_stat = pitch.bin_statistic(
    shots['x'], shots['y'], values=shots['psxg'],
    statistic='mean', bins=(12, 8)
)
pitch.heatmap(bin_stat, ax=ax_heatmap, cmap='magma', vmin=0, vmax=0.3)
ax_heatmap.set_title('Mean psxG by Zone', fontsize=9)

# ── Row 2: Beth Mead — Goals vs xG vs psxG, and Situation Danger ─────────────
ax_beth = fig.add_subplot(gs[2, :3])
if len(beth) > 0:
    x_pos = np.arange(len(beth))
    w = 0.25
    ax_beth.bar(x_pos - w,   beth['goals'].values,      w, color=C_YELLOW, alpha=0.9, label='Goals')
    ax_beth.bar(x_pos,       beth['xg'].values,         w, color=C_BLUE,   alpha=0.8, label='Pre-Shot xG')
    ax_beth.bar(x_pos + w,   beth['psxg'].values,       w, color=C_GREEN,  alpha=0.8, label='Post-Shot psxG')
    ax_beth.set_xticks(x_pos)
    ax_beth.set_xticklabels(beth['season'].values, rotation=30, ha='right', fontsize=8)
    ax_beth.set_title('Beth Mead — Goals vs xG vs psxG by Season', fontsize=11)
    ax_beth.legend(fontsize=9)
    ax_beth.set_facecolor(FIG_BG)
    for i, row in enumerate(beth.itertuples()):
        diff = row.goals - row.psxg
        clr  = C_GREEN if diff >= 0 else C_ORANGE
        ax_beth.text(i + w, max(row.goals, row.psxg) + 0.2,
                     f'{diff:+.1f}', ha='center', fontsize=7, color=clr)

ax_sit = fig.add_subplot(gs[2, 3])
if len(beth) > 0:
    ax_sit.barh(beth['season'].values,
                beth['sit_danger'].values, color=C_PURPLE, alpha=0.8, label='Situation Danger')
    ax_sit.barh(beth['season'].values,
                beth['xg'].values, color=C_BLUE, alpha=0.6, label='xG', left=0)
    ax_sit.set_title('Sit. Danger vs xG', fontsize=9)
    ax_sit.legend(fontsize=7)
    ax_sit.set_facecolor(FIG_BG)

fig.suptitle('WSL Three-Model xG System — Evaluation Dashboard',
             fontsize=15, fontweight='bold', color='white', y=0.99)
plt.savefig(os.path.join(OUTPUT_DIR, 'xg_dashboard.png'),
            dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close()
print(f"\nDashboard saved → {OUTPUT_DIR}/xg_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# SHAP — one plot per model
# ─────────────────────────────────────────────────────────────────────────────
if HAS_SHAP and HAS_XGB:
    for model, feat_names, tag, X_sample in [
        (xg_model,   XG_FEATURES,        'xg',        X_base_test.sample(min(2000,len(X_base_test)), random_state=0)),
        (psxg_model, PSXG_FEATURES,      'psxg',      X_ps_test.sample(min(2000,len(X_ps_test)), random_state=0)),
        (sit_model,  SITUATION_FEATURES, 'situation', X_sit_test.sample(min(2000,len(X_sit_test)), random_state=0)),
    ]:
        explainer = shap.TreeExplainer(model.clf)
        sv        = explainer.shap_values(X_sample)
        shap.summary_plot(sv, X_sample, feature_names=feat_names,
                          show=False, max_display=12)
        plt.savefig(os.path.join(OUTPUT_DIR, f'shap_{tag}.png'),
                    dpi=150, bbox_inches='tight', facecolor=FIG_BG)
        plt.close()
    print("SHAP plots saved.")

# ─────────────────────────────────────────────────────────────────────────────
# SHOT-MAP: Beth Mead psxG (vertical half-pitch)
# ─────────────────────────────────────────────────────────────────────────────
beth_shots = shots[shots['player_name'].str.contains('Mead', na=False)].copy()
if len(beth_shots) > 0:
    fig_b, ax_b = plt.subplots(figsize=(7, 9), facecolor=FIG_BG)
    vpitch = VerticalPitch(pitch_type='opta', pitch_color=PITCH_BG,
                           line_color=LINE_COL, linewidth=0.8, half=True)
    vpitch.draw(ax=ax_b)

    size_scale = 300
    for _, row in beth_shots.iterrows():
        psxg_val = row.get('psxg', row.get('xg', 0.05))
        C_MUTED  = '#484f58'
        color    = C_YELLOW if row['is_goal'] else (C_BLUE if row['is_on_target'] else C_MUTED)
        marker   = '*' if row['is_goal'] else ('o' if row['is_on_target'] else 'x')
        ax_b.scatter(row['y'], row['x'],
                     s=max(psxg_val, 0.03) * size_scale,
                     c=color, marker=marker, alpha=0.75, linewidths=0.5,
                     edgecolors='white' if row['is_goal'] else 'none', zorder=3)

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0],[0], marker='*', color='w', markerfacecolor=C_YELLOW, markersize=12, label='Goal', linestyle='None'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_BLUE,   markersize=8,  label='On Target', linestyle='None'),
        Line2D([0],[0], marker='x', color='w', markerfacecolor='#484f58', markersize=8, label='Off Target', linestyle='None'),
    ]
    ax_b.legend(handles=legend_items, loc='lower center', fontsize=9,
                facecolor=FIG_BG, edgecolor='#30363d', labelcolor='white')
    ax_b.set_title('Beth Mead — Shot Map\n(size = psxG)', fontsize=12, color='white', pad=10)
    ax_b.set_facecolor(PITCH_BG)
    fig_b.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'beth_mead_shotmap.png'),
                dpi=150, bbox_inches='tight', facecolor=FIG_BG)
    plt.close()
    print(f"Beth Mead shot map saved → {OUTPUT_DIR}/beth_mead_shotmap.png")

print("\nAll done.")
