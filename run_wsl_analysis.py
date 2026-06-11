"""
WSL Shot Analytics — Full Pipeline Runner
Produces: all_shots_full.csv, player_summary.csv, career_summary.csv,
          bayesian_finishing.csv, and PNG charts saved to xg_output/
"""
import os, glob, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve, auc, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.special import betaln
from scipy.optimize import minimize
import xgboost as xgb

plt.rcParams['figure.dpi'] = 120

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_ROOT  = '/home/user/Project-Beth-Mead'
OUTPUT_DIR = '/home/user/Project-Beth-Mead/xg_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIG_BG   = '#0e1117'; PITCH_BG = '#0d1117'; LINE_COL = '#c9d1d9'
C_BLUE   = '#58a6ff'; C_ORANGE = '#f78166'; C_GREEN  = '#3fb950'
C_YELLOW = '#e3b341'; C_PURPLE = '#bc8cff'; C_MUTED  = '#484f58'

GOAL_X = 100.0; GOAL_Y_LEFT = 44.0; GOAL_Y_RIGHT = 56.0
GOAL_Y_CENTRE = 50.0; GOAL_WIDTH = 12.0; GOAL_H_HIGH = 62.0

Q = dict(
    PENALTY=9, HEADER=15, RIGHT_FOOT=20, LEFT_FOOT=72,
    REGULAR_PLAY=22, FAST_BREAK=23, SET_PIECE=24, FROM_CORNER=25,
    FREE_KICK=26, DIRECT_FK=28, VOLLEY=108, DEFLECTION=133,
    PULL_BACK=195, BIG_CHANCE=233, FIRST_TIME=200,
    GOAL_Y=102, GOAL_HEIGHT=231, SAVE_END_X=146, SAVE_END_Y=147,
    UNDER_PRESSURE=18, INTENTIONAL=154, BODY_SIDE=56,
    BLOCKED=82,
)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def open_angle(x, y):
    shot  = np.array([x, y], dtype=float)
    left  = np.array([GOAL_X, GOAL_Y_LEFT]);  right = np.array([GOAL_X, GOAL_Y_RIGHT])
    v1, v2 = left - shot, right - shot
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))))

def dist_to_goal(x, y):
    return float(np.sqrt((GOAL_X-x)**2 + (y-GOAL_Y_CENTRE)**2))

def placement_score(gy_norm, gh_norm):
    if pd.isna(gy_norm) or pd.isna(gh_norm): return np.nan
    gy = float(np.clip(gy_norm,-1,1)); gh = float(np.clip(gh_norm,0,1))
    keeper_h = 0.40; lat = abs(gy); vert = abs(gh - keeper_h) / max(keeper_h, 1-keeper_h)
    return float(np.clip(np.sqrt(0.6*lat**2 + 0.4*vert**2)*100, 0, 100))

def goal_zone(goal_y, goal_h):
    if pd.isna(goal_y) or pd.isna(goal_h): return 'Unknown'
    h = 'Top' if goal_h >= GOAL_H_HIGH else 'Bottom'
    s = 'Left' if goal_y < 47.5 else ('Right' if goal_y > 52.5 else 'Centre')
    return f'{h} {s}'

def get_q(event, qid):
    for q in event.get('qualifier', []):
        if q['qualifierId'] == qid: return q.get('value', 1)
    return None

def has_q(event, qid):
    return any(q['qualifierId'] == qid for q in event.get('qualifier', []))

def safe_float(v):
    try: return float(v)
    except: return np.nan

def season_from_path(path):
    for p in path.replace('\\','/').split('/'):
        if p.startswith('WSL'): return p
    return 'Unknown'

# ── LOAD MATCHES ──────────────────────────────────────────────────────────────
def load_match(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    events = data.get('liveData', {}).get('event', data.get('event', []))
    rows = []
    for e in events:
        tid = str(e.get('typeId'))
        if tid not in ('13','14','15','16'): continue
        x = safe_float(e.get('x')); y = safe_float(e.get('y'))
        if np.isnan(x) or np.isnan(y): continue
        gy_raw  = safe_float(get_q(e, Q['GOAL_Y']))
        gh_raw  = safe_float(get_q(e, Q['GOAL_HEIGHT']))
        gy_norm = (gy_raw - GOAL_Y_CENTRE)/(GOAL_WIDTH/2) if not np.isnan(gy_raw) else np.nan
        gh_norm = gh_raw/100.0 if not np.isnan(gh_raw) else np.nan
        if not np.isnan(gy_norm) and not np.isnan(gh_norm):
            gf_dist = np.sqrt(gy_norm**2 + (gh_norm-0.35)**2)
            corner  = int(abs(gy_norm)>0.55 or gh_norm>0.60)
            ps      = placement_score(gy_norm, gh_norm)
        elif not np.isnan(gy_norm):
            gf_dist = abs(gy_norm); corner = int(abs(gy_norm)>0.55); ps = abs(gy_norm)*100
        else:
            gf_dist = corner = ps = np.nan
        body_side = str(get_q(e, Q['BODY_SIDE']) or '').strip()
        is_rf = has_q(e, Q['RIGHT_FOOT']); is_lf = has_q(e, Q['LEFT_FOOT'])
        weak_foot = int((is_rf and body_side=='Left') or (is_lf and body_side=='Right'))
        dist = dist_to_goal(x, y); angle = open_angle(x, y); y_sym = abs(y - GOAL_Y_CENTRE)
        rows.append({
            'match_file': os.path.basename(path), 'season': season_from_path(path),
            'player_id': str(e.get('playerId','')), 'player_name': e.get('playerName',''),
            'contestant_id': str(e.get('contestantId','')),
            'type_id': int(tid), 'period_id': int(e.get('periodId') or 0),
            'time_min': int(e.get('timeMin') or 0),
            'x': x, 'y': y, 'y_sym': y_sym,
            'distance': dist, 'log_distance': np.log(max(dist,0.5)),
            'angle': angle, 'angle_sin': np.sin(np.radians(angle)),
            'in_six_yard': int(x>=94.2 and 36.8<=y<=63.2),
            'in_penalty_box': int(x>=83.0 and 21.1<=y<=78.9),
            'central_y': int(y_sym < GOAL_WIDTH/2),
            'dist_to_post': abs(y_sym - GOAL_WIDTH/2),
            'goal_y_raw': gy_raw, 'goal_y_norm': gy_norm,
            'goal_h_raw': gh_raw, 'goal_h_norm': gh_norm,
            'goal_frame_dist': gf_dist, 'corner_zone': corner,
            'placement_score': ps, 'goal_zone': goal_zone(gy_raw, gh_raw),
            'is_header': int(has_q(e,Q['HEADER'])),
            'is_right_foot': int(is_rf), 'is_left_foot': int(is_lf), 'weak_foot': weak_foot,
            'is_volley': int(has_q(e,Q['VOLLEY'])), 'is_deflected': int(has_q(e,Q['DEFLECTION'])),
            'is_first_time': int(has_q(e,Q['FIRST_TIME'])), 'is_big_chance': int(has_q(e,Q['BIG_CHANCE'])),
            'is_fast_break': int(has_q(e,Q['FAST_BREAK'])), 'is_from_corner': int(has_q(e,Q['FROM_CORNER'])),
            'is_free_kick': int(has_q(e,Q['DIRECT_FK'])), 'is_penalty': int(has_q(e,Q['PENALTY'])),
            'is_set_piece': int(has_q(e,Q['SET_PIECE'])), 'is_open_play': int(has_q(e,Q['REGULAR_PLAY'])),
            'is_pull_back': int(has_q(e,Q['PULL_BACK'])), 'under_pressure': int(has_q(e,Q['UNDER_PRESSURE'])),
            'is_intentional': int(has_q(e,Q['INTENTIONAL'])),
            'is_goal': int(tid=='16'), 'is_on_target': int(tid in ('15','16')),
            'is_post': int(tid=='14'), 'is_blocked': int(tid=='15'),
            'is_outfield_block': int(tid=='15' and has_q(e, Q['BLOCKED'])),
            'is_keeper_save':    int(tid=='15' and not has_q(e, Q['BLOCKED'])),
        })
    return pd.DataFrame(rows)

print('Loading all WSL matches...')
json_files = glob.glob(os.path.join(DATA_ROOT, 'WSL*', '**', '*.json'), recursive=True)
json_files += glob.glob(os.path.join(DATA_ROOT, 'WSL*', '*.json'))
json_files = list(set(json_files))
print(f'Found {len(json_files)} JSON files')

frames, errors = [], []
for path in json_files:
    try: frames.append(load_match(path))
    except Exception as exc: errors.append((os.path.basename(path), str(exc)))

if errors: print(f'Skipped {len(errors)} files: {errors[:3]}')
shots = pd.concat(frames, ignore_index=True)

def technique_index(row):
    s = 0
    if row['is_header']:      s += 20
    if row['is_volley']:      s += 15
    if row['is_first_time']:  s += 10
    if row['under_pressure']: s += 25
    if row['is_deflected']:   s += 10
    if row['weak_foot']:      s += 20
    return float(min(s, 100))

shots['technique_index'] = shots.apply(technique_index, axis=1)
print(f"Loaded {len(shots):,} shots | Goals: {shots['is_goal'].sum():,} ({shots['is_goal'].mean()*100:.1f}%)")
print(f"Outfield blocks: {shots['is_outfield_block'].sum():,} | Keeper saves: {shots['is_keeper_save'].sum():,}")
print(f"Seasons: {sorted(shots['season'].unique())}")

# ── FEATURE SETS ──────────────────────────────────────────────────────────────
XG_FEATURES = [
    'x','y_sym','distance','log_distance','angle','angle_sin',
    'dist_to_post','in_six_yard','in_penalty_box','central_y',
    'is_header','is_right_foot','is_left_foot','weak_foot',
    'is_volley','is_deflected','is_first_time','is_big_chance',
    'is_fast_break','is_from_corner','is_free_kick','is_set_piece',
    'is_open_play','is_pull_back','under_pressure','is_intentional',
]
PSXG_FEATURES = XG_FEATURES + ['goal_y_norm','goal_h_norm','goal_frame_dist','corner_zone','placement_score']

pen_mask = shots['is_penalty'] == 1
nonpen   = shots[~pen_mask].copy().reset_index(drop=True)
pen_xg   = shots.loc[pen_mask, 'is_goal'].mean() if pen_mask.sum() > 0 else 0.76

X_base = nonpen[XG_FEATURES].fillna(0).astype(float)
y_goal = nonpen['is_goal'].astype(int)
groups = nonpen['match_file']

gss_outer = GroupShuffleSplit(1, test_size=0.20, random_state=42)
dev_idx, test_idx = next(gss_outer.split(X_base, y_goal, groups=groups))
gss_inner = GroupShuffleSplit(1, test_size=0.25, random_state=0)
tr_idx, cal_idx = next(gss_inner.split(X_base.iloc[dev_idx], y_goal.iloc[dev_idx], groups=groups.iloc[dev_idx]))

def split(X):
    Xd = X.iloc[dev_idx]
    return Xd.iloc[tr_idx], Xd.iloc[cal_idx], X.iloc[test_idx]

X_base_tr, X_base_cal, X_base_test = split(X_base)
y_tr   = y_goal.iloc[dev_idx].iloc[tr_idx]
y_cal  = y_goal.iloc[dev_idx].iloc[cal_idx]
y_test = y_goal.iloc[test_idx]
g_tr   = groups.iloc[dev_idx].iloc[tr_idx]

class CalibratedXGB:
    def __init__(self, clf, iso): self.clf = clf; self.iso = iso
    def predict_proba(self, X):
        raw = self.clf.predict_proba(X)[:, 1]
        cal = self.iso.predict(raw)
        return np.column_stack([1-cal, cal])

def train_model(X_tr, y_tr, g_tr, X_cal, y_cal, X_test, y_test, label):
    group_cv = GroupKFold(n_splits=5)
    spw = (y_tr==0).sum() / max((y_tr==1).sum(), 1)
    grid = GridSearchCV(
        xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc',
                          use_label_encoder=False, random_state=42, verbosity=0),
        {'max_depth': [4,6], 'learning_rate': [0.05,0.1], 'n_estimators': [300,500],
         'subsample': [0.8], 'colsample_bytree': [0.7,0.9],
         'min_child_weight': [5,10], 'scale_pos_weight': [spw]},
        scoring='roc_auc', cv=group_cv, n_jobs=-1, verbose=0,
    )
    grid.fit(X_tr, y_tr, groups=g_tr)
    clf = grid.best_estimator_
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(clf.predict_proba(X_cal)[:, 1], y_cal)
    model = CalibratedXGB(clf, iso)
    p_test = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p_test)
    m_auc   = auc(fpr, tpr)
    m_brier = brier_score_loss(y_test, p_test)
    print(f'  [{label}]  CV AUC: {grid.best_score_:.4f}  |  Test AUC: {m_auc:.4f}  Brier: {m_brier:.4f}')
    return model, p_test, fpr, tpr, m_auc, m_brier

print('\n=== Training Models ===')
print('── Model 1: Pre-Shot xG ──')
xg_model, xg_p, xg_fpr, xg_tpr, xg_auc, xg_brier = train_model(
    X_base_tr, y_tr, g_tr, X_base_cal, y_cal, X_base_test, y_test, 'xG')
nonpen['xg'] = xg_model.predict_proba(X_base)[:, 1]
shots.loc[~pen_mask, 'xg'] = nonpen['xg'].values
shots.loc[pen_mask,  'xg'] = float(pen_xg)

print('\n── Model 2: Post-Shot psxG ──')
has_pl = nonpen[['goal_y_norm','goal_h_norm']].notna().any(axis=1)
nps    = nonpen[has_pl].reset_index(drop=True)
X_ps   = nps[PSXG_FEATURES].fillna(0).astype(float)
y_ps   = nps['is_goal'].astype(int); g_ps = nps['match_file']
print(f'  Shots with placement: {len(nps):,}/{len(nonpen):,}')
g1 = GroupShuffleSplit(1,test_size=0.20,random_state=42)
dp,tp = next(g1.split(X_ps,y_ps,groups=g_ps))
g2 = GroupShuffleSplit(1,test_size=0.25,random_state=0)
trp,cap = next(g2.split(X_ps.iloc[dp],y_ps.iloc[dp],groups=g_ps.iloc[dp]))
psxg_model,psxg_p,psxg_fpr,psxg_tpr,psxg_auc,psxg_brier = train_model(
    X_ps.iloc[dp].iloc[trp],y_ps.iloc[dp].iloc[trp],g_ps.iloc[dp].iloc[trp],
    X_ps.iloc[dp].iloc[cap],y_ps.iloc[dp].iloc[cap],X_ps.iloc[tp],y_ps.iloc[tp],'psxG')
X_all_ps = nonpen[PSXG_FEATURES].fillna(0).astype(float)
nonpen['psxg'] = nonpen['xg'].copy()
nonpen.loc[has_pl,'psxg'] = psxg_model.predict_proba(X_all_ps[has_pl])[:,1]
shots.loc[~pen_mask,'psxg'] = nonpen['psxg'].values
shots.loc[pen_mask, 'psxg'] = float(pen_xg)

print('\n── Derived metrics ──')
shots['placement_quality'] = shots['psxg'] - shots['xg']

def shot_power(row):
    p = 50.0
    if row['is_volley']:      p += 25
    if row['is_first_time']:  p += 12
    if row['is_header']:      p -= 8
    if row['weak_foot']:      p -= 12
    if row['under_pressure']: p -= 8
    if row['is_deflected']:   p += 5
    return float(np.clip(p, 0, 100))

shots['shot_power']      = shots.apply(shot_power, axis=1)
shots['finishing_luck']  = shots['is_goal'] - shots.get('psxg', shots['xg'])

# ── EMPIRICAL BAYES ───────────────────────────────────────────────────────────
print('\n=== Empirical Bayes Finishing ===')
eb_df = (
    shots[shots['is_penalty'] == 0]
    .groupby('player_name')
    .agg(n=('is_goal','count'), goals=('is_goal','sum'), xg_sum=('xg','sum'))
    .reset_index()
)
eb_df = eb_df[eb_df['n'] >= 5].copy()
eb_df['raw_rate'] = eb_df['goals'] / eb_df['n']
eb_df['xg_rate']  = eb_df['xg_sum'] / eb_df['n']

def bb_nll(params, n_arr, k_arr):
    a, b = np.exp(params)
    ll = 0.0
    for ni, ki in zip(n_arr, k_arr):
        ll += (betaln(ki+a, ni-ki+b) - betaln(a, b))
    return -ll

res = minimize(bb_nll, x0=[0.0, 2.5], args=(eb_df['n'].values, eb_df['goals'].values),
               method='Nelder-Mead', options={'maxiter':5000,'xatol':1e-6})
alpha_hat, beta_hat = np.exp(res.x)
league_mean = alpha_hat / (alpha_hat + beta_hat)
print(f'Beta prior: α={alpha_hat:.3f}  β={beta_hat:.3f}  league mean={league_mean*100:.2f}%')

eb_df['eb_alpha_post'] = alpha_hat + eb_df['goals']
eb_df['eb_beta_post']  = beta_hat  + (eb_df['n'] - eb_df['goals'])
eb_df['eb_mean']       = eb_df['eb_alpha_post'] / (eb_df['eb_alpha_post'] + eb_df['eb_beta_post'])
eb_df['eb_ci_lo']      = stats.beta.ppf(0.05, eb_df['eb_alpha_post'], eb_df['eb_beta_post'])
eb_df['eb_ci_hi']      = stats.beta.ppf(0.95, eb_df['eb_alpha_post'], eb_df['eb_beta_post'])
eb_df['eb_vs_xg']      = eb_df['eb_mean'] - eb_df['xg_rate']
eb_df['shrinkage']     = 1 - (eb_df['eb_mean'] - league_mean) / \
                         (eb_df['raw_rate'] - league_mean).replace(0, np.nan)
eb_df = eb_df.sort_values('eb_mean', ascending=False)

print('\nTop 20 WSL finishers (EB posterior mean, min 5 shots):')
print(eb_df[['player_name','n','goals','raw_rate','xg_rate','eb_mean','eb_ci_lo','eb_ci_hi','eb_vs_xg']]
      .head(20).round(4).to_string(index=False))

# ── PLAYER SUMMARY ────────────────────────────────────────────────────────────
player_summary = (
    shots.groupby(['player_name','season'])
    .agg(shots_n=('is_goal','count'), goals=('is_goal','sum'),
         xg=('xg','sum'), psxg=('psxg','sum'),
         placement_mean=('placement_score','mean'), power_mean=('shot_power','mean'))
    .reset_index()
)
player_summary['conv_rate']        = player_summary['goals'] / player_summary['shots_n']
player_summary['goals_minus_xg']   = player_summary['goals'] - player_summary['xg']
player_summary['goals_minus_psxg'] = player_summary['goals'] - player_summary['psxg']

career = (
    shots.groupby('player_name')
    .agg(shots_n=('is_goal','count'), goals=('is_goal','sum'),
         xg=('xg','sum'), psxg=('psxg','sum'), seasons=('season','nunique'))
    .reset_index()
)
career['conv_rate']        = career['goals'] / career['shots_n']
career['goals_minus_xg']   = career['goals'] - career['xg']
career['goals_minus_psxg'] = career['goals'] - career['psxg']

print('\n\n=== Top 20 Career Finishers (Goals − psxG, min 20 shots) ===')
top_finishers = career[career['shots_n']>=20].nlargest(20,'goals_minus_psxg')
print(top_finishers[['player_name','seasons','shots_n','goals','conv_rate','xg','psxg','goals_minus_psxg']]
      .round(2).to_string(index=False))

# ── OUTFIELD BLOCK STATS ──────────────────────────────────────────────────────
print('\n=== Block Type Analysis (Q82) ===')
total_ot = shots['is_blocked'].sum()
ob_n     = shots['is_outfield_block'].sum()
ks_n     = shots['is_keeper_save'].sum()
print(f'Total on-target shots (typeId=15): {total_ot:,}')
print(f'  Outfield blocks (Q82 present):   {ob_n:,}  ({ob_n/total_ot*100:.1f}%)')
print(f'  Keeper saves   (Q82 absent):     {ks_n:,}  ({ks_n/total_ot*100:.1f}%)')

# Team block stats
team_blocks = shots.groupby('contestant_id').agg(
    shots_faced=('xg','count'),
    goals_conceded=('is_goal','sum'),
    xg_conceded=('xg','sum'),
    outfield_blocks=('is_outfield_block','sum'),
    keeper_saves=('is_keeper_save','sum'),
    total_blocked=('is_blocked','sum'),
).reset_index()
team_blocks['ob_rate'] = team_blocks['outfield_blocks'] / team_blocks['shots_faced']
team_blocks['ks_rate'] = team_blocks['keeper_saves'] / team_blocks['shots_faced']
team_blocks = team_blocks.sort_values('outfield_blocks', ascending=False)
print('\nTop 10 teams by outfield blocks:')
print(team_blocks[['contestant_id','shots_faced','outfield_blocks','keeper_saves','ob_rate','ks_rate','goals_conceded']].head(10).round(3).to_string(index=False))

# ── SAVE FILES ────────────────────────────────────────────────────────────────
shots.to_csv(os.path.join(OUTPUT_DIR, 'all_shots_full.csv'), index=False)
player_summary.to_csv(os.path.join(OUTPUT_DIR, 'player_summary.csv'), index=False)
career.to_csv(os.path.join(OUTPUT_DIR, 'career_summary.csv'), index=False)
eb_df.to_csv(os.path.join(OUTPUT_DIR, 'bayesian_finishing.csv'), index=False)
team_blocks.to_csv(os.path.join(OUTPUT_DIR, 'team_block_stats.csv'), index=False)

# ── VISUALISATIONS ────────────────────────────────────────────────────────────
print('\nGenerating charts...')

# 1. Top finishers EB chart
fig, ax = plt.subplots(figsize=(12, 10), facecolor=FIG_BG)
top_eb = eb_df.nlargest(25, 'eb_vs_xg').sort_values('eb_vs_xg')
y_pos  = np.arange(len(top_eb))
ax.barh(y_pos, top_eb['eb_vs_xg'], color=C_BLUE, alpha=0.7, height=0.7)
ax.errorbar(top_eb['eb_mean']-top_eb['xg_rate'], y_pos,
            xerr=[top_eb['eb_mean']-top_eb['eb_ci_lo'], top_eb['eb_ci_hi']-top_eb['eb_mean']],
            fmt='none', color='white', capsize=3, lw=1, alpha=0.5)
ax.axvline(0, color='white', lw=0.8, alpha=0.4)
ax.set_yticks(y_pos); ax.set_yticklabels(top_eb['player_name'].values, fontsize=8)
ax.set(xlabel='Bayesian finishing edge (EB mean − avg xG/shot)',
       title='Top 25 WSL Finishers — Empirical Bayes Edge over Expected\n(90% credible intervals, min 5 shots)',
       facecolor=FIG_BG)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'top_finishers_eb.png'), dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close()

# 2. Outfield block rate by team
fig, ax = plt.subplots(figsize=(11, 8), facecolor=FIG_BG)
tb_sorted = team_blocks.sort_values('ob_rate')
w = 0.35
yp = np.arange(len(tb_sorted))
ax.barh(yp - w/2, tb_sorted['ob_rate'], w, color=C_BLUE,   alpha=0.8, label='Outfield block rate')
ax.barh(yp + w/2, tb_sorted['ks_rate'], w, color=C_ORANGE, alpha=0.8, label='Keeper save rate')
ax.set_yticks(yp); ax.set_yticklabels(tb_sorted['contestant_id'].astype(str), fontsize=7)
ax.set(xlabel='Rate (per shot faced)', title='Outfield Block vs Keeper Save Rate by Team (Q82 split)',
       facecolor=FIG_BG)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'team_block_rates.png'), dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close()

# 3. Career top finishers G-psxG
min20 = career[career['shots_n']>=20].nlargest(25,'goals_minus_psxg').sort_values('goals_minus_psxg')
fig, ax = plt.subplots(figsize=(12, 10), facecolor=FIG_BG)
yp2 = np.arange(len(min20))
cols = [C_GREEN if v>=0 else C_ORANGE for v in min20['goals_minus_psxg']]
bars = ax.barh(yp2, min20['goals_minus_psxg'], color=cols, alpha=0.85, height=0.7)
for i, (bar, row) in enumerate(zip(bars, min20.itertuples())):
    ax.text(bar.get_width()+0.1, i, f'{row.shots_n}sh', va='center', fontsize=7, color='white', alpha=0.6)
ax.axvline(0, color='white', lw=1, alpha=0.4)
ax.set_yticks(yp2); ax.set_yticklabels(min20['player_name'].values, fontsize=8)
ax.set(xlabel='Career Goals − psxG', title='Top 25 WSL Career Finishers (Goals − psxG, min 20 shots)',
       facecolor=FIG_BG)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'career_finishers.png'), dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close()

# 4. xG by season
season_totals = shots.groupby('season').agg(
    shots_n=('is_goal','count'), goals=('is_goal','sum'), xg=('xg','sum'), psxg=('psxg','sum')
).reset_index().sort_values('season')
fig, axes = plt.subplots(1,2, figsize=(16,6), facecolor=FIG_BG)
xs = np.arange(len(season_totals)); w=0.22
axes[0].bar(xs-w, season_totals['goals'], w*2, color=C_YELLOW, alpha=0.9, label='Goals')
axes[0].bar(xs+w, season_totals['xg'],    w*2, color=C_BLUE,   alpha=0.8, label='xG')
axes[0].set_xticks(xs); axes[0].set_xticklabels(season_totals['season'].values, rotation=30, ha='right', fontsize=7)
axes[0].set(title='WSL Goals vs xG by Season', facecolor=FIG_BG); axes[0].legend()
conv = season_totals['goals']/season_totals['shots_n']
xgps = season_totals['xg']/season_totals['shots_n']
axes[1].plot(xs, conv.values, 'o-', color=C_YELLOW, lw=2, ms=7, label='Conv rate')
axes[1].plot(xs, xgps.values, 's-', color=C_BLUE,   lw=2, ms=7, label='xG/shot')
axes[1].set_xticks(xs); axes[1].set_xticklabels(season_totals['season'].values, rotation=30, ha='right', fontsize=7)
axes[1].set(title='Conv Rate & xG/shot by Season', facecolor=FIG_BG); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'xg_by_season.png'), dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close()

print('\n=== Summary ===')
print(f'Total shots:         {len(shots):,}')
print(f'Total goals:         {shots["is_goal"].sum():,}')
print(f'Seasons covered:     {shots["season"].nunique()}  ({", ".join(sorted(shots["season"].unique())[:3])} ...)')
print(f'Players tracked:     {shots["player_name"].nunique():,}')
print(f'Teams tracked:       {shots["contestant_id"].nunique():,}')
print(f'xG model AUC:        {xg_auc:.4f}')
print(f'psxG model AUC:      {psxg_auc:.4f}')
print(f'League conv rate:    {shots["is_goal"].mean()*100:.2f}%')
print(f'EB league mean:      {league_mean*100:.2f}%')
print(f'Outfield block rate: {shots["is_outfield_block"].mean()*100:.1f}%')
print(f'Keeper save rate:    {shots["is_keeper_save"].mean()*100:.1f}%')
print(f'\nOutput files: {OUTPUT_DIR}/')
for f in sorted(os.listdir(OUTPUT_DIR)):
    sz = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
    print(f'  {f:40s}  {sz:7.1f} KB')
