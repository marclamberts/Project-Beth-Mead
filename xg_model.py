"""
WSL Shot Analytics — Five-Model System
========================================

Model 1  Pre-Shot xG          P(goal | position + technique + situation)
Model 2  Post-Shot psxG       P(goal | xG features + goal-frame placement)
Model 3  Shot Situation Danger P(goal | build-up context only, no geometry)
Model 4  P(On Target)          P(shot on target | pre-shot features)
Model 5  xGOT                  P(goal | shot ON TARGET + placement)

Derived shot metrics
────────────────────
placement_quality   psxG − xG               How well the shot was aimed
execution_quality   xG − situation_danger   How well the position was found
save_difficulty     psxG of saves            How hard the keeper worked
finishing_luck      goals − psxG            Conversion above shot quality
on_target_over_exp  is_on_target − p_ontarget  Accuracy vs expectation
technique_index     composite difficulty    Header/weak-foot/first-time penalty
goal_zone           6-zone classification   Top/bottom × left/centre/right
under_pressure      Q18 flag               Opta's difficulty/pressure marker

All models
  XGBoost with GroupKFold CV (no match leakage)
  Isotonic calibration on a held-out fold
  Penalties handled separately (empirical mean)
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
import matplotlib.patches as mpatches
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

GOAL_Y_CENTRE = 50.0
GOAL_X        = 100.0
GOAL_Y_LEFT   = 44.0
GOAL_Y_RIGHT  = 56.0
GOAL_WIDTH    = GOAL_Y_RIGHT - GOAL_Y_LEFT   # 12 Opta units

# Goal-zone boundaries (from decoded Q102/Q231 data)
GOAL_Y_LEFT_POST  = 45.5   # left post (from Q102 distributions)
GOAL_Y_RIGHT_POST = 54.5   # right post
GOAL_H_HIGH       = 62.0   # top of goal zone (Q231 scale)

# Opta qualifier IDs (verified against this dataset)
Q_PENALTY       = 9
Q_HEADER        = 15
Q_RIGHT_FOOT    = 20
Q_LEFT_FOOT     = 72
Q_REGULAR_PLAY  = 22
Q_FAST_BREAK    = 23
Q_SET_PIECE     = 24
Q_FROM_CORNER   = 25
Q_FREE_KICK     = 26
Q_DIRECT_FK     = 28
Q_VOLLEY        = 108
Q_DEFLECTION    = 133
Q_PULL_BACK     = 195
Q_BIG_CHANCE    = 233
Q_FIRST_TIME    = 200
Q_DISTANCE      = 103
Q_GOAL_Y        = 102   # lateral goal-frame position (44–56 Opta units)
Q_GOAL_HEIGHT   = 231   # shot height (0–100 scale)
Q_SAVE_END_X    = 146   # end pitch-x for saves/off-target
Q_SAVE_END_Y    = 147   # end pitch-y for saves/off-target
Q_UNDER_PRESSURE = 18   # Opta difficulty marker (disproportionately on misses/saves)
Q_INTENTIONAL   = 154   # Intentional assist
Q_BODY_SIDE     = 56    # 'Left' / 'Right' / 'Center' body position

# Visual palette (matches app.py)
FIG_BG   = '#0e1117'
PITCH_BG = '#0d1117'
LINE_COL = '#c9d1d9'
C_BLUE   = '#58a6ff'
C_ORANGE = '#f78166'
C_GREEN  = '#3fb950'
C_YELLOW = '#e3b341'
C_PURPLE = '#bc8cff'
C_MUTED  = '#484f58'

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def open_angle(x: float, y: float) -> float:
    shot  = np.array([x, y], dtype=float)
    left  = np.array([GOAL_X, GOAL_Y_LEFT],  dtype=float)
    right = np.array([GOAL_X, GOAL_Y_RIGHT], dtype=float)
    v1, v2 = left - shot, right - shot
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.degrees(np.arccos(
        np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    )))


def distance_to_goal(x: float, y: float) -> float:
    return float(np.sqrt((GOAL_X - x) ** 2 + (y - GOAL_Y_CENTRE) ** 2))


def in_six_yard_box(x: float, y: float) -> int:
    return int(x >= 94.2 and 36.8 <= y <= 63.2)


def in_penalty_area(x: float, y: float) -> int:
    return int(x >= 83.0 and 21.1 <= y <= 78.9)


def goal_zone(goal_y: float, goal_h: float) -> str:
    """
    6-zone goal frame classification using Q102 (lateral) and Q231 (height).
    Zones: {Top|Bottom} × {Left|Centre|Right}
    """
    if np.isnan(goal_y) or np.isnan(goal_h):
        return 'Unknown'
    height_label = 'Top'    if goal_h >= GOAL_H_HIGH else 'Bottom'
    if goal_y < GOAL_Y_LEFT_POST:
        side_label = 'Left'
    elif goal_y > GOAL_Y_RIGHT_POST:
        side_label = 'Right'
    else:
        side_label = 'Centre'
    return f'{height_label} {side_label}'


def placement_score(goal_y_norm: float, goal_h_norm: float) -> float:
    """
    0–100 index: how difficult the shot placement was for the goalkeeper.
    High = aimed at corner / top of goal (hardest to save).
    Low  = aimed at keeper's body / centre (easiest to save).
    Keeper's natural reach zone: y_norm ≈ 0, height ≈ 35–45%.
    """
    if np.isnan(goal_y_norm) or np.isnan(goal_h_norm):
        return np.nan
    keeper_h = 0.40
    lateral_difficulty  = abs(goal_y_norm)                      # 0 = straight, 1 = post
    vertical_difficulty = abs(goal_h_norm - keeper_h) / max(keeper_h, 1 - keeper_h)
    raw = np.sqrt(0.6 * lateral_difficulty ** 2 + 0.4 * vertical_difficulty ** 2)
    # Normalise to 0–100 (max possible ≈ sqrt(0.6 + 0.4) = 1.0)
    return float(np.clip(raw * 100, 0, 100))


def technique_index(row: dict) -> float:
    """
    0–100 composite technique difficulty.
    Penalises headers (harder to place accurately),
    first-time shots, volleys, under-pressure attempts.
    """
    score = 0.0
    if row.get('is_header'):        score += 20
    if row.get('is_volley'):        score += 15
    if row.get('is_first_time'):    score += 10
    if row.get('under_pressure'):   score += 25
    if row.get('is_deflected'):     score += 10
    # Weak foot: player is right-footed (Q_RIGHT_FOOT) but shot from left side,
    # or left-footed (Q_LEFT_FOOT) but shot from right side
    if row.get('weak_foot'):        score += 20
    return float(min(score, 100))


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

        # ── Post-shot placement ──────────────────────────────────────────────
        goal_y_raw = safe_float(get_q(e, Q_GOAL_Y))
        goal_h_raw = safe_float(get_q(e, Q_GOAL_HEIGHT))

        goal_y_norm = (
            (goal_y_raw - GOAL_Y_CENTRE) / (GOAL_WIDTH / 2)
            if not np.isnan(goal_y_raw) else np.nan
        )
        goal_h_norm = goal_h_raw / 100.0 if not np.isnan(goal_h_raw) else np.nan

        if not np.isnan(goal_y_norm) and not np.isnan(goal_h_norm):
            gf_dist    = np.sqrt(goal_y_norm ** 2 + (goal_h_norm - 0.35) ** 2)
            corner_zone = int(abs(goal_y_norm) > 0.55 or goal_h_norm > 0.60)
            ps_score   = placement_score(goal_y_norm, goal_h_norm)
        elif not np.isnan(goal_y_norm):
            gf_dist    = abs(goal_y_norm)
            corner_zone = int(abs(goal_y_norm) > 0.55)
            ps_score   = abs(goal_y_norm) * 100
        else:
            gf_dist = corner_zone = ps_score = np.nan

        # ── Body-side / weak-foot detection ─────────────────────────────────
        body_side = str(get_q(e, Q_BODY_SIDE) or '').strip()
        is_rf     = has_q(e, Q_RIGHT_FOOT)
        is_lf     = has_q(e, Q_LEFT_FOOT)
        # Weak foot: right-footer shooting from left side, or vice versa
        weak_foot = int(
            (is_rf and body_side == 'Left') or
            (is_lf and body_side == 'Right')
        )

        dist  = distance_to_goal(x, y)
        angle = open_angle(x, y)
        y_sym = abs(y - GOAL_Y_CENTRE)

        row = {
            'match_file':     os.path.basename(path),
            'season':         _season_from_path(path),
            'player_id':      str(e.get('playerId', '')),
            'player_name':    e.get('playerName', ''),
            'contestant_id':  str(e.get('contestantId', '')),
            'type_id':        int(tid),
            'period_id':      int(e.get('periodId') or 0),
            'time_min':       int(e.get('timeMin') or 0),
            'x': x, 'y': y, 'y_sym': y_sym,

            # Pre-shot geometry
            'distance':        dist,
            'log_distance':    np.log(max(dist, 0.5)),
            'angle':           angle,
            'angle_sin':       np.sin(np.radians(angle)),
            'in_six_yard':     in_six_yard_box(x, y),
            'in_penalty_box':  in_penalty_area(x, y),
            'central_y':       int(y_sym < GOAL_WIDTH / 2),
            'dist_to_post':    abs(y_sym - GOAL_WIDTH / 2),

            # Post-shot placement
            'goal_y_raw':      goal_y_raw,
            'goal_y_norm':     goal_y_norm,
            'goal_h_raw':      goal_h_raw,
            'goal_h_norm':     goal_h_norm,
            'goal_frame_dist': gf_dist,
            'corner_zone':     corner_zone,
            'placement_score': ps_score,
            'goal_zone':       goal_zone(goal_y_raw, goal_h_raw),

            # Body part
            'is_header':       int(has_q(e, Q_HEADER)),
            'is_right_foot':   int(is_rf),
            'is_left_foot':    int(is_lf),
            'weak_foot':       weak_foot,

            # Technique & situation
            'is_volley':        int(has_q(e, Q_VOLLEY)),
            'is_deflected':     int(has_q(e, Q_DEFLECTION)),
            'is_first_time':    int(has_q(e, Q_FIRST_TIME)),
            'is_big_chance':    int(has_q(e, Q_BIG_CHANCE)),
            'is_fast_break':    int(has_q(e, Q_FAST_BREAK)),
            'is_from_corner':   int(has_q(e, Q_FROM_CORNER)),
            'is_free_kick':     int(has_q(e, Q_DIRECT_FK)),
            'is_penalty':       int(has_q(e, Q_PENALTY)),
            'is_set_piece':     int(has_q(e, Q_SET_PIECE)),
            'is_open_play':     int(has_q(e, Q_REGULAR_PLAY)),
            'is_pull_back':     int(has_q(e, Q_PULL_BACK)),
            'under_pressure':   int(has_q(e, Q_UNDER_PRESSURE)),
            'is_intentional':   int(has_q(e, Q_INTENTIONAL)),

            # Outcomes
            'is_goal':          int(tid == '16'),
            'is_on_target':     int(tid in ('15', '16')),
            'is_post':          int(tid == '14'),
            'is_blocked':       int(tid == '15'),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _season_from_path(path: str) -> str:
    for part in path.replace('\\', '/').split('/'):
        if part.startswith('WSL'):
            return part
    return 'Unknown'


# ─────────────────────────────────────────────────────────────────────────────
# MODEL INFRASTRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

class CalibratedXGB:
    """XGBoost + held-out isotonic calibration (no leakage)."""
    def __init__(self, clf, iso: IsotonicRegression):
        self.clf = clf
        self.iso = iso

    def predict_proba(self, X):
        raw = self.clf.predict_proba(X)[:, 1]
        cal = self.iso.predict(raw)
        return np.column_stack([1 - cal, cal])


def train_model(X_tr, y_tr, g_tr, X_cal, y_cal,
                X_test, y_test, label: str,
                extra_params: dict | None = None):
    group_cv = GroupKFold(n_splits=5)
    if HAS_XGB:
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        base_grid = {
            'max_depth':        [4, 6],
            'learning_rate':    [0.05, 0.1],
            'n_estimators':     [300, 500],
            'subsample':        [0.8],
            'colsample_bytree': [0.7, 0.9],
            'min_child_weight': [5, 10],
            'scale_pos_weight': [spw],
        }
        if extra_params:
            base_grid.update(extra_params)
        base = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            use_label_encoder=False, random_state=42, verbosity=0,
        )
        grid = GridSearchCV(base, base_grid, scoring='roc_auc',
                            cv=group_cv, n_jobs=-1, verbose=0)
        grid.fit(X_tr, y_tr, groups=g_tr)
        print(f"  [{label}] CV AUC: {grid.best_score_:.4f}  params: {grid.best_params_}")
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

    p_test      = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p_test)
    m_auc       = auc(fpr, tpr)
    m_brier     = brier_score_loss(y_test, p_test)
    print(f"  [{label}] Test AUC: {m_auc:.4f}  Brier: {m_brier:.4f}")
    return model, p_test, fpr, tpr, m_auc, m_brier


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

# Compute technique_index (needs full row dict)
shots['technique_index'] = shots.apply(technique_index, axis=1)

print(f"Loaded {len(shots):,} shots  |  Goals: {shots['is_goal'].sum():,}  "
      f"({shots['is_goal'].mean()*100:.1f}%)")
print(f"Under pressure: {shots['under_pressure'].sum():,}  "
      f"Weak foot: {shots['weak_foot'].sum():,}  "
      f"With placement: {shots[['goal_y_norm','goal_h_norm']].notna().any(axis=1).sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# PENALTIES
# ─────────────────────────────────────────────────────────────────────────────
pen_mask = shots['is_penalty'] == 1
nonpen   = shots[~pen_mask].copy().reset_index(drop=True)

pen_xg = shots.loc[pen_mask, 'is_goal'].mean() if pen_mask.sum() > 0 else 0.76
for col in ['xg', 'psxg', 'situation_danger', 'p_ontarget', 'xgot']:
    shots.loc[pen_mask, col] = pen_xg
print(f"Penalty xG (empirical): {pen_xg:.3f}  n={pen_mask.sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SETS
# ─────────────────────────────────────────────────────────────────────────────
XG_FEATURES = [
    'x', 'y_sym', 'distance', 'log_distance', 'angle', 'angle_sin',
    'dist_to_post', 'in_six_yard', 'in_penalty_box', 'central_y',
    'is_header', 'is_right_foot', 'is_left_foot', 'weak_foot',
    'is_volley', 'is_deflected', 'is_first_time', 'is_big_chance',
    'is_fast_break', 'is_from_corner', 'is_free_kick', 'is_set_piece',
    'is_open_play', 'is_pull_back', 'under_pressure', 'is_intentional',
]

PSXG_EXTRA = [
    'goal_y_norm',     # lateral (-1…+1)
    'goal_h_norm',     # height (0–1)
    'goal_frame_dist', # distance from goal-frame centre
    'corner_zone',     # 1 = aimed at hard corner
    'placement_score', # 0–100 placement difficulty index
    # NOTE: is_on_target / in_frame excluded — target leakage
]
PSXG_FEATURES = XG_FEATURES + PSXG_EXTRA

SITUATION_FEATURES = [
    'is_big_chance', 'is_fast_break', 'is_from_corner',
    'is_free_kick', 'is_set_piece', 'is_open_play', 'is_pull_back',
    'is_first_time', 'is_header', 'under_pressure', 'is_intentional',
    'period_id', 'time_min',
]

# P(on target) uses only pre-shot context (no goal-frame placement)
POT_FEATURES = [
    'x', 'y_sym', 'distance', 'log_distance', 'angle', 'angle_sin',
    'dist_to_post', 'in_six_yard', 'in_penalty_box', 'central_y',
    'is_header', 'is_right_foot', 'is_left_foot', 'weak_foot',
    'is_volley', 'is_deflected', 'is_first_time', 'is_big_chance',
    'is_fast_break', 'is_from_corner', 'is_open_play', 'under_pressure',
]

# xGOT: expected goal given ON-TARGET shot — only uses on-target shots
XGOT_FEATURES = PSXG_FEATURES  # same feature set as psxG

# ─────────────────────────────────────────────────────────────────────────────
# SHARED TRAIN / CAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
X_base = nonpen[XG_FEATURES].fillna(0).astype(float)
X_sit  = nonpen[SITUATION_FEATURES].fillna(0).astype(float)
X_pot  = nonpen[POT_FEATURES].fillna(0).astype(float)
y_goal = nonpen['is_goal'].astype(int)
y_ot   = nonpen['is_on_target'].astype(int)
groups = nonpen['match_file']

gss_outer = GroupShuffleSplit(1, test_size=0.20, random_state=42)
dev_idx, test_idx = next(gss_outer.split(X_base, y_goal, groups=groups))

gss_inner = GroupShuffleSplit(1, test_size=0.25, random_state=0)
tr_idx, cal_idx = next(gss_inner.split(
    X_base.iloc[dev_idx], y_goal.iloc[dev_idx], groups=groups.iloc[dev_idx]
))

def split(X):
    Xd = X.iloc[dev_idx]
    return Xd.iloc[tr_idx], Xd.iloc[cal_idx], X.iloc[test_idx]

X_base_tr, X_base_cal, X_base_test = split(X_base)
X_sit_tr,  X_sit_cal,  X_sit_test  = split(X_sit)
X_pot_tr,  X_pot_cal,  X_pot_test  = split(X_pot)

y_goal_tr  = y_goal.iloc[dev_idx].iloc[tr_idx]
y_goal_cal = y_goal.iloc[dev_idx].iloc[cal_idx]
y_goal_test= y_goal.iloc[test_idx]
y_ot_tr    = y_ot.iloc[dev_idx].iloc[tr_idx]
y_ot_cal   = y_ot.iloc[dev_idx].iloc[cal_idx]
y_ot_test  = y_ot.iloc[test_idx]
g_tr       = groups.iloc[dev_idx].iloc[tr_idx]

print(f"Train: {len(y_goal_tr):,}  Cal: {len(y_goal_cal):,}  Test: {len(y_goal_test):,}\n")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — Pre-Shot xG
# ─────────────────────────────────────────────────────────────────────────────
print("── Model 1: Pre-Shot xG ──")
xg_model, xg_p, xg_fpr, xg_tpr, xg_auc, xg_brier = train_model(
    X_base_tr, y_goal_tr, g_tr,
    X_base_cal, y_goal_cal,
    X_base_test, y_goal_test, 'xG'
)
nonpen['xg'] = xg_model.predict_proba(X_base)[:, 1]
shots.loc[~pen_mask, 'xg'] = nonpen['xg'].values
joblib.dump(xg_model, os.path.join(OUTPUT_DIR, 'model_xg.pkl'))

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — Post-Shot xG  (requires placement data)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Model 2: Post-Shot psxG ──")
has_placement = nonpen[['goal_y_norm', 'goal_h_norm']].notna().any(axis=1)
nonpen_ps     = nonpen[has_placement].reset_index(drop=True)
X_ps          = nonpen_ps[PSXG_FEATURES].fillna(0).astype(float)
y_ps          = nonpen_ps['is_goal'].astype(int)
g_ps          = nonpen_ps['match_file']
print(f"  Shots with placement data: {len(nonpen_ps):,} / {len(nonpen):,}")

gss_ps = GroupShuffleSplit(1, test_size=0.20, random_state=42)
dev_ps, test_ps = next(gss_ps.split(X_ps, y_ps, groups=g_ps))
gss_ps_in = GroupShuffleSplit(1, test_size=0.25, random_state=0)
tr_ps, cal_ps = next(gss_ps_in.split(X_ps.iloc[dev_ps], y_ps.iloc[dev_ps],
                                      groups=g_ps.iloc[dev_ps]))

psxg_model, psxg_p, psxg_fpr, psxg_tpr, psxg_auc, psxg_brier = train_model(
    X_ps.iloc[dev_ps].iloc[tr_ps],  y_ps.iloc[dev_ps].iloc[tr_ps],  g_ps.iloc[dev_ps].iloc[tr_ps],
    X_ps.iloc[dev_ps].iloc[cal_ps], y_ps.iloc[dev_ps].iloc[cal_ps],
    X_ps.iloc[test_ps],             y_ps.iloc[test_ps],
    'psxG'
)
X_all_ps       = nonpen[PSXG_FEATURES].fillna(0).astype(float)
nonpen['psxg'] = nonpen['xg'].copy()  # default = xG (no placement available)
nonpen.loc[has_placement, 'psxg'] = psxg_model.predict_proba(X_all_ps[has_placement])[:, 1]
shots.loc[~pen_mask, 'psxg'] = nonpen['psxg'].values
joblib.dump(psxg_model, os.path.join(OUTPUT_DIR, 'model_psxg.pkl'))

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — Shot Situation Danger
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Model 3: Situation Danger ──")
sit_model, sit_p, sit_fpr, sit_tpr, sit_auc, sit_brier = train_model(
    X_sit_tr, y_goal_tr, g_tr,
    X_sit_cal, y_goal_cal,
    X_sit_test, y_goal_test, 'Situation'
)
nonpen['situation_danger'] = sit_model.predict_proba(X_sit)[:, 1]
shots.loc[~pen_mask, 'situation_danger'] = nonpen['situation_danger'].values
joblib.dump(sit_model, os.path.join(OUTPUT_DIR, 'model_situation.pkl'))

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4 — P(On Target)
# Predicts whether a shot will hit the target (type 15 or 16).
# Uses only pre-shot features — no goal-frame placement.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Model 4: P(On Target) ──")
pot_model, pot_p, pot_fpr, pot_tpr, pot_auc, pot_brier = train_model(
    X_pot_tr, y_ot_tr, g_tr,
    X_pot_cal, y_ot_cal,
    X_pot_test, y_ot_test, 'P(OT)'
)
nonpen['p_ontarget'] = pot_model.predict_proba(X_pot)[:, 1]
shots.loc[~pen_mask, 'p_ontarget'] = nonpen['p_ontarget'].values
joblib.dump(pot_model, os.path.join(OUTPUT_DIR, 'model_pontarget.pkl'))

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 5 — xGOT (Expected Goals on Target)
# Trained only on shots that were ON TARGET (saved or goal).
# P(goal | on target + placement)
# More discriminative than psxG for analysing keeper performance
# because it conditions on the shot reaching the goal.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Model 5: xGOT (Expected Goals on Target) ──")
ot_mask    = nonpen['is_on_target'] == 1
nonpen_ot  = nonpen[ot_mask].reset_index(drop=True)
X_ot       = nonpen_ot[XGOT_FEATURES].fillna(0).astype(float)
y_ot_goal  = nonpen_ot['is_goal'].astype(int)
g_ot       = nonpen_ot['match_file']
print(f"  On-target shots: {len(nonpen_ot):,}  Goals: {y_ot_goal.sum():,}  "
      f"({y_ot_goal.mean()*100:.1f}%)")

gss_ot = GroupShuffleSplit(1, test_size=0.20, random_state=42)
dev_ot, test_ot = next(gss_ot.split(X_ot, y_ot_goal, groups=g_ot))
gss_ot_in = GroupShuffleSplit(1, test_size=0.25, random_state=0)
tr_ot, cal_ot = next(gss_ot_in.split(X_ot.iloc[dev_ot], y_ot_goal.iloc[dev_ot],
                                      groups=g_ot.iloc[dev_ot]))

xgot_model, xgot_p, xgot_fpr, xgot_tpr, xgot_auc, xgot_brier = train_model(
    X_ot.iloc[dev_ot].iloc[tr_ot],  y_ot_goal.iloc[dev_ot].iloc[tr_ot],  g_ot.iloc[dev_ot].iloc[tr_ot],
    X_ot.iloc[dev_ot].iloc[cal_ot], y_ot_goal.iloc[dev_ot].iloc[cal_ot],
    X_ot.iloc[test_ot],             y_ot_goal.iloc[test_ot],
    'xGOT'
)
# Predict xGOT only for on-target shots; off-target shots get 0
nonpen['xgot']        = 0.0
nonpen.loc[ot_mask, 'xgot'] = xgot_model.predict_proba(X_ot)[:, 1]
shots.loc[~pen_mask, 'xgot'] = nonpen['xgot'].values
shots.loc[pen_mask,  'xgot'] = pen_xg   # penalty stays
joblib.dump(xgot_model, os.path.join(OUTPUT_DIR, 'model_xgot.pkl'))

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED SHOT METRICS
# ─────────────────────────────────────────────────────────────────────────────
# Placement quality: psxG − xG
# Positive = aimed better than average for that position + technique
shots['placement_quality'] = shots['psxg'] - shots['xg']

# Execution quality: xG − situation_danger
# Positive = shooter found a better position than the build-up typically yields
shots['execution_quality'] = shots['xg'] - shots['situation_danger']

# On-target over-expectation: actual on-target minus expected
shots['ot_over_exp'] = shots['is_on_target'] - shots.get('p_ontarget', 0)

# Save difficulty (for keeper analysis): psxG of saved shots
# A save with psxG=0.7 was very difficult; psxG=0.05 was routine
shots['save_difficulty'] = np.where(
    shots['is_blocked'] == 1,
    shots['psxg'],
    np.nan
)

# Finishing luck: goals − psxG (close to 0 = true skill; large = luck/variance)
shots['finishing_luck'] = shots['is_goal'] - shots['psxg']

# 0–100 indices for each model output
for col in ['xg', 'psxg', 'situation_danger', 'p_ontarget', 'xgot']:
    if col in shots.columns and shots[col].notna().any():
        mn, mx = shots[col].min(), shots[col].max()
        shots[f'{col}_index'] = (shots[col] - mn) / max(mx - mn, 1e-9) * 100

# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
shots.to_csv(os.path.join(OUTPUT_DIR, 'all_shots_xg.csv'), index=False)

player_summary = (
    shots.groupby(['player_name', 'season'])
    .agg(
        shots_n        = ('xg',            'count'),
        goals          = ('is_goal',        'sum'),
        xg             = ('xg',             'sum'),
        psxg           = ('psxg',           'sum'),
        xgot_sum       = ('xgot',           'sum'),
        sit_danger     = ('situation_danger','sum'),
        on_target      = ('is_on_target',   'sum'),
        p_ontarget_sum = ('p_ontarget',     'sum'),
        placement_q    = ('placement_quality','mean'),
        execution_q    = ('execution_quality','mean'),
        technique_idx  = ('technique_index', 'mean'),
        save_diff_mean = ('save_difficulty', 'mean'),
    )
    .reset_index()
)
player_summary['goals_minus_xg']   = player_summary['goals'] - player_summary['xg']
player_summary['goals_minus_psxg'] = player_summary['goals'] - player_summary['psxg']
player_summary['xg_per_shot']      = player_summary['xg'] / player_summary['shots_n'].clip(1)
player_summary['shot_ot_pct']      = player_summary['on_target'] / player_summary['shots_n'].clip(1)
player_summary['ot_over_exp']      = player_summary['on_target'] - player_summary['p_ontarget_sum']
player_summary.to_csv(os.path.join(OUTPUT_DIR, 'player_xg_summary.csv'), index=False)

# Keeper summary: save difficulty and goals-above-average
keeper_saves = shots[shots['is_blocked'] == 1].copy()
keeper_summary = (
    keeper_saves.groupby('contestant_id')
    .agg(
        saves_n       = ('psxg',           'count'),
        psxg_faced    = ('psxg',           'sum'),
        goals_conceded= ('is_goal',        'sum'),  # 0 for saves by definition
        avg_difficulty= ('save_difficulty', 'mean'),
    )
    .reset_index()
)
keeper_summary['goals_prevented'] = keeper_summary['psxg_faced']   # all faced were saved
keeper_summary.to_csv(os.path.join(OUTPUT_DIR, 'team_keeper_summary.csv'), index=False)

beth = player_summary[
    player_summary['player_name'].str.contains('Mead', na=False)
].sort_values('season')

print("\n── Beth Mead — Five-Model Summary ──")
print(beth[['season','shots_n','goals','xg','psxg','xgot_sum',
            'goals_minus_xg','goals_minus_psxg',
            'shot_ot_pct','placement_q','execution_q']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD  (3 rows × 4 cols)
# ─────────────────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
fig = plt.figure(figsize=(22, 18), facecolor=FIG_BG)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38)

# ── Row 0: ROC curves, Calibration, Goal Zone dist, Technique distribution ───
ax_roc = fig.add_subplot(gs[0, 0])
for fpr_c, tpr_c, score, label, color, ls in [
    (xg_fpr,   xg_tpr,   xg_auc,   'Pre-Shot xG',   C_BLUE,   '-'),
    (psxg_fpr, psxg_tpr, psxg_auc, 'Post-Shot psxG', C_GREEN,  '-'),
    (pot_fpr,  pot_tpr,  pot_auc,  'P(On Target)',   C_PURPLE, '--'),
    (xgot_fpr, xgot_tpr, xgot_auc, 'xGOT',           C_YELLOW, '-'),
    (sit_fpr,  sit_tpr,  sit_auc,  'Sit. Danger',    C_ORANGE, ':'),
]:
    ax_roc.plot(fpr_c, tpr_c, lw=2, color=color, ls=ls,
                label=f'{label} {score:.3f}')
ax_roc.plot([0,1],[0,1],'w--',lw=1,alpha=0.3)
ax_roc.set(xlabel='FPR', ylabel='TPR', title='ROC — All Five Models', facecolor=FIG_BG)
ax_roc.legend(fontsize=6.5, loc='lower right')

ax_cal = fig.add_subplot(gs[0, 1])
for proba, label, color, y_ref in [
    (xg_p,   'Pre-Shot xG',   C_BLUE,   y_goal_test),
    (psxg_p, 'Post-Shot psxG',C_GREEN,  y_ps.iloc[test_ps]),
    (pot_p,  'P(On Target)',   C_PURPLE, y_ot_test),
    (xgot_p, 'xGOT',          C_YELLOW, y_ot_goal.iloc[test_ot]),
]:
    pt, pp = calibration_curve(y_ref, proba, n_bins=10, strategy='quantile')
    ax_cal.plot(pp, pt, marker='o', lw=2, color=color, label=label, markersize=3)
ax_cal.plot([0,1],[0,1],'w--',lw=1,alpha=0.3)
ax_cal.set(xlabel='Predicted', ylabel='Actual', title='Calibration', facecolor=FIG_BG)
ax_cal.legend(fontsize=7)

# Goal Zone distribution (shots with placement data)
ax_gz = fig.add_subplot(gs[0, 2])
gz_shots = shots[shots['goal_zone'] != 'Unknown']
zone_order = ['Top Left','Top Centre','Top Right','Bottom Left','Bottom Centre','Bottom Right']
zone_goals = gz_shots.groupby('goal_zone')['is_goal'].agg(['sum','count'])
zone_goals['conv'] = zone_goals['sum'] / zone_goals['count'].clip(1)
zone_goals = zone_goals.reindex([z for z in zone_order if z in zone_goals.index])
colors_gz  = [C_ORANGE if 'Top' in z else C_BLUE for z in zone_goals.index]
ax_gz.bar(range(len(zone_goals)), zone_goals['conv'].values, color=colors_gz, alpha=0.8)
ax_gz.set_xticks(range(len(zone_goals)))
ax_gz.set_xticklabels([z.replace(' ','\n') for z in zone_goals.index], fontsize=7)
ax_gz.set_ylabel('Conversion Rate', fontsize=8)
ax_gz.set_title('Goal Conversion by Zone', fontsize=9)
ax_gz.set_facecolor(FIG_BG)
for i, (_, row) in enumerate(zone_goals.iterrows()):
    ax_gz.text(i, row['conv']+0.005, f"{row['sum']:.0f}g\n{row['count']:.0f}s",
               ha='center', fontsize=6, color='white')

# Shot technique index distribution
ax_tech = fig.add_subplot(gs[0, 3])
tech_goals = shots[shots['is_goal']==1]['technique_index']
tech_ngoals= shots[shots['is_goal']==0]['technique_index']
bins_t = np.arange(0, 105, 10)
ax_tech.hist(tech_ngoals, bins=bins_t, alpha=0.5, color=C_ORANGE, density=True, label='No Goal')
ax_tech.hist(tech_goals,  bins=bins_t, alpha=0.5, color=C_BLUE,   density=True, label='Goal')
ax_tech.set(xlabel='Technique Index', title='Technique Difficulty Distribution', facecolor=FIG_BG)
ax_tech.legend(fontsize=8)

# ── Row 1: Feature importances (xG, psxG, P(OT), xGOT) ──────────────────────
for ax_col, (model, feat_names, title, color) in enumerate([
    (xg_model,   XG_FEATURES,       'xG Features',    C_BLUE),
    (psxg_model, PSXG_FEATURES,     'psxG Features',  C_GREEN),
    (pot_model,  POT_FEATURES,      'P(OT) Features', C_PURPLE),
    (xgot_model, XGOT_FEATURES,     'xGOT Features',  C_YELLOW),
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

# ── Row 2: Beth Mead — multi-metric bar, shot map, save difficulty pitch ─────
ax_beth = fig.add_subplot(gs[2, :2])
if len(beth) > 0:
    x_pos = np.arange(len(beth))
    w = 0.2
    ax_beth.bar(x_pos - 1.5*w, beth['goals'].values,      w, color=C_YELLOW, alpha=0.9, label='Goals')
    ax_beth.bar(x_pos - 0.5*w, beth['xg'].values,         w, color=C_BLUE,   alpha=0.8, label='xG')
    ax_beth.bar(x_pos + 0.5*w, beth['psxg'].values,       w, color=C_GREEN,  alpha=0.8, label='psxG')
    ax_beth.bar(x_pos + 1.5*w, beth['xgot_sum'].values,   w, color=C_YELLOW, alpha=0.5, label='xGOT')
    ax_beth.set_xticks(x_pos)
    ax_beth.set_xticklabels(beth['season'].values, rotation=35, ha='right', fontsize=7)
    ax_beth.set_title('Beth Mead — Goals vs xG vs psxG vs xGOT', fontsize=10)
    ax_beth.legend(fontsize=7, ncol=4)
    ax_beth.set_facecolor(FIG_BG)
    for i, row in enumerate(beth.itertuples()):
        diff = row.goals - row.psxg
        clr  = C_GREEN if diff >= 0 else C_ORANGE
        ax_beth.text(i + 0.5*w, max(row.goals, row.psxg) + 0.15,
                     f'{diff:+.1f}', ha='center', fontsize=6.5, color=clr)

# Shot map: Beth Mead, sized by psxG, coloured by goal zone
ax_map = fig.add_subplot(gs[2, 2])
beth_shots = shots[shots['player_name'].str.contains('Mead', na=False)].copy()
if len(beth_shots) > 0:
    vpitch = VerticalPitch(pitch_type='opta', pitch_color=PITCH_BG,
                           line_color=LINE_COL, linewidth=0.8, half=True)
    vpitch.draw(ax=ax_map)
    for _, row in beth_shots.iterrows():
        pv = max(float(row.get('psxg', row.get('xg', 0.05))), 0.03)
        c  = C_YELLOW if row['is_goal'] else (C_GREEN if row['is_on_target'] else C_MUTED)
        mk = '*' if row['is_goal'] else ('o' if row['is_on_target'] else 'x')
        ax_map.scatter(row['y'], row['x'], s=pv*350, c=c, marker=mk,
                       alpha=0.75, linewidths=0.5,
                       edgecolors='white' if row['is_goal'] else 'none', zorder=3)
    ax_map.set_title("Beth Mead Shot Map\n(size=psxG)", fontsize=9)

# Placement quality scatter: psxG vs placement_score (on-target shots)
ax_ps = fig.add_subplot(gs[2, 3])
ot_shots = shots[(shots['is_on_target']==1) & shots['placement_score'].notna()].copy()
scatter_c = [C_YELLOW if g else C_BLUE for g in ot_shots['is_goal']]
ax_ps.scatter(ot_shots['placement_score'], ot_shots['psxg'],
              c=scatter_c, alpha=0.3, s=8)
ax_ps.set_xlabel('Placement Score (0=easy, 100=corner)', fontsize=8)
ax_ps.set_ylabel('Post-Shot xG', fontsize=8)
ax_ps.set_title('Shot Placement vs psxG\n(yellow=goal, blue=save)', fontsize=9)
ax_ps.set_facecolor(FIG_BG)

fig.suptitle('WSL Five-Model Shot Analytics Dashboard', fontsize=15,
             fontweight='bold', color='white', y=0.995)
plt.savefig(os.path.join(OUTPUT_DIR, 'xg_dashboard.png'),
            dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close()
print(f"\nDashboard saved → {OUTPUT_DIR}/xg_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# GOAL FRAME HEATMAP (where goals are scored vs saves)
# ─────────────────────────────────────────────────────────────────────────────
fig_gf, axes_gf = plt.subplots(1, 2, figsize=(14, 5), facecolor=FIG_BG)
for ax, mask, title, cmap in [
    (axes_gf[0], shots['is_goal']==1,    'Goal Placement Heatmap',   'YlOrRd'),
    (axes_gf[1], shots['is_blocked']==1, 'Save Placement Heatmap',   'Blues'),
]:
    sub = shots[mask & shots['goal_y_raw'].notna() & shots['goal_h_raw'].notna()]
    if len(sub) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, color='white')
        continue
    # Draw goal frame
    rect = plt.Rectangle((GOAL_Y_LEFT, 0), GOAL_WIDTH, 100,
                          fill=False, edgecolor='white', lw=2)
    ax.add_patch(rect)
    ax.hist2d(sub['goal_y_raw'], sub['goal_h_raw'], bins=(20, 15),
              range=[[GOAL_Y_LEFT-2, GOAL_Y_RIGHT+2], [0, 100]],
              cmap=cmap, density=True)
    ax.axvline(GOAL_Y_CENTRE, color='white', ls='--', alpha=0.3, lw=1)
    ax.axhline(GOAL_H_HIGH, color='white', ls=':', alpha=0.3, lw=1)
    ax.set_xlabel('Goal Frame Lateral (Opta y)', fontsize=9)
    ax.set_ylabel('Shot Height (0–100)', fontsize=9)
    ax.set_title(title, fontsize=11, color='white')
    ax.set_facecolor(PITCH_BG)
    # Mark zones
    ax.text(GOAL_Y_CENTRE, 80, 'Top Centre', ha='center', color='white', fontsize=7, alpha=0.6)
    ax.text(GOAL_Y_LEFT+0.5, 80, 'Top\nLeft',   ha='left',   color='white', fontsize=7, alpha=0.6)
    ax.text(GOAL_Y_RIGHT-0.5,80, 'Top\nRight',  ha='right',  color='white', fontsize=7, alpha=0.6)

fig_gf.suptitle('Goal Frame Placement — Goals vs Saves', fontsize=13,
                color='white', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'goal_frame_heatmap.png'),
            dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close()
print(f"Goal frame heatmap saved.")

# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────
if HAS_SHAP and HAS_XGB:
    for model, feat_names, tag, X_samp in [
        (xg_model,   XG_FEATURES,   'xg',         X_base_test.sample(min(2000, len(X_base_test)), random_state=0)),
        (psxg_model, PSXG_FEATURES, 'psxg',       X_ps.iloc[test_ps].sample(min(2000, len(test_ps)), random_state=0)),
        (pot_model,  POT_FEATURES,  'p_ontarget', X_pot_test.sample(min(2000, len(X_pot_test)), random_state=0)),
        (xgot_model, XGOT_FEATURES, 'xgot',       X_ot.iloc[test_ot].sample(min(2000, len(test_ot)), random_state=0)),
        (sit_model,  SITUATION_FEATURES,'situation',X_sit_test.sample(min(2000, len(X_sit_test)), random_state=0)),
    ]:
        sv = shap.TreeExplainer(model.clf).shap_values(X_samp)
        shap.summary_plot(sv, X_samp, feature_names=feat_names, show=False, max_display=12)
        plt.savefig(os.path.join(OUTPUT_DIR, f'shap_{tag}.png'),
                    dpi=150, bbox_inches='tight', facecolor=FIG_BG)
        plt.close()
    print("SHAP plots saved.")

print("\nAll done.")
print(f"\nModel performance summary:")
print(f"  Pre-Shot xG        AUC={xg_auc:.4f}   Brier={xg_brier:.4f}")
print(f"  Post-Shot psxG     AUC={psxg_auc:.4f}   Brier={psxg_brier:.4f}")
print(f"  Situation Danger   AUC={sit_auc:.4f}   Brier={sit_brier:.4f}")
print(f"  P(On Target)       AUC={pot_auc:.4f}   Brier={pot_brier:.4f}")
print(f"  xGOT               AUC={xgot_auc:.4f}   Brier={xgot_brier:.4f}")
