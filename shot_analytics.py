"""
WSL Shot Analytics — Extended Shot Metrics
==========================================
Run AFTER xg_model.py (reads xg_output/all_shots_xg.csv).

New metrics added
─────────────────
1. Multi-outcome model   P(goal|save|miss|post) — 4-class XGBoost
2. Optimal placement     Did the shooter pick the best zone for their position?
3. Shot power index      Velocity proxy: volley/first-time/weak-foot-adjusted
4. Rolling shot quality  Per-player rolling xG, psxG, goals (last-10-shots window)
5. Shot archetypes       K-means (k=6) on shot profile features
6. Pressure efficiency   Conversion rate and xG delta: under pressure vs open
7. Contact quality index psxG per unit of technique difficulty

Also fixes placement_score overflow from off-target shots where Q102 was
outside the 44–56 goal-frame range.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, log_loss
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mplsoccer import Pitch, VerticalPitch

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not installed — install it for the multi-outcome model")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'xg_output')
IN_CSV       = os.path.join(OUTPUT_DIR, 'all_shots_xg.csv')
OUT_CSV      = os.path.join(OUTPUT_DIR, 'all_shots_full.csv')

FIG_BG   = '#0e1117'
PITCH_BG = '#0d1117'
LINE_COL = '#c9d1d9'
C_BLUE   = '#58a6ff'
C_ORANGE = '#f78166'
C_GREEN  = '#3fb950'
C_YELLOW = '#e3b341'
C_PURPLE = '#bc8cff'
C_MUTED  = '#484f58'

GOAL_Y_LEFT   = 44.0
GOAL_Y_RIGHT  = 56.0
GOAL_Y_CENTRE = 50.0
GOAL_WIDTH    = 12.0

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
shots = pd.read_csv(IN_CSV)
print(f"Loaded {len(shots):,} shots from {IN_CSV}")

# ─────────────────────────────────────────────────────────────────────────────
# FIX: placement_score — clip goal_y_norm to [-1, 1] before scoring
# Off-target shots have Q102 outside 44–56 → goal_y_norm > 1 → overflow
# ─────────────────────────────────────────────────────────────────────────────
def placement_score_fixed(goal_y_norm, goal_h_norm):
    if pd.isna(goal_y_norm) or pd.isna(goal_h_norm):
        return np.nan
    gy = float(np.clip(goal_y_norm, -1.0, 1.0))
    gh = float(np.clip(goal_h_norm,  0.0, 1.0))
    keeper_h = 0.40
    lat  = abs(gy)
    vert = abs(gh - keeper_h) / max(keeper_h, 1.0 - keeper_h)
    raw  = np.sqrt(0.6 * lat**2 + 0.4 * vert**2)
    return float(np.clip(raw * 100, 0, 100))

shots['placement_score'] = shots.apply(
    lambda r: placement_score_fixed(r['goal_y_norm'], r['goal_h_norm']), axis=1
)
print(f"placement_score fixed — max: {shots['placement_score'].max():.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# METRIC 1 — Shot Power Index (0–100)
# Proxy for ball velocity based on technique and body-part difficulty.
# Volley / first-time = high-pace shot; placed / weak-foot = controlled / slower.
# ─────────────────────────────────────────────────────────────────────────────
def shot_power(row) -> float:
    p = 50.0
    if row['is_volley']:      p += 25
    if row['is_first_time']:  p += 12
    if row['is_header']:      p -= 8   # headers have less pace
    if row['weak_foot']:      p -= 12  # weaker contact
    if row['under_pressure']: p -= 8   # rushed
    if row['is_deflected']:   p += 5   # unpredictable deflection pace
    return float(np.clip(p, 0, 100))

shots['shot_power'] = shots.apply(shot_power, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# METRIC 2 — Contact Quality Index (0–100)
# Measures how cleanly the shot was executed:
#   high psxG + high technique_index → elite contact (hard situation, great result)
#   high psxG + low technique_index  → good placement but easy technique
# CQI rewards difficult, well-placed shots.
# ─────────────────────────────────────────────────────────────────────────────
def contact_quality(psxg, tech_idx, ps_score) -> float:
    if pd.isna(psxg) or pd.isna(ps_score):
        return float(psxg * 100) if not pd.isna(psxg) else np.nan
    # Weighted combination: 50% placement, 30% psxG, 20% technique difficulty
    norm_tech = tech_idx / 60.0   # max technique_index = 60
    norm_ps   = ps_score / 100.0
    return float(np.clip(
        (0.50 * norm_ps + 0.30 * psxg + 0.20 * norm_tech) * 100, 0, 100
    ))

shots['contact_quality'] = shots.apply(
    lambda r: contact_quality(r['psxg'], r['technique_index'], r['placement_score']), axis=1
)

# ─────────────────────────────────────────────────────────────────────────────
# METRIC 3 — Optimal Placement Score
# For each shot origin bin (x × y_sym × body_part), find the goal zone with
# the highest historical conversion. Score actual placement:
#   100 = shot went to the optimal zone
#    50 = shot went to an adjacent zone
#     0 = shot went to the worst zone or off target
# ─────────────────────────────────────────────────────────────────────────────
def make_origin_bin(x, y_sym):
    """Coarse 3×3 origin grid: {close/mid/long} × {central/wide}."""
    dist_bin = 'close' if x >= 88 else ('mid' if x >= 75 else 'long')
    wide_bin = 'central' if y_sym < 8 else 'wide'
    return f'{dist_bin}_{wide_bin}'

def make_body_bin(row):
    if row['is_header']:   return 'header'
    if row['weak_foot']:   return 'weak_foot'
    return 'foot'

shots['origin_bin'] = shots.apply(
    lambda r: make_origin_bin(r['x'], r['y_sym']), axis=1
)
shots['body_bin'] = shots.apply(make_body_bin, axis=1)
shots['origin_key'] = shots['origin_bin'] + '|' + shots['body_bin']

# Build optimal zone lookup: per origin_key, which goal_zone converts best?
zone_order = ['Top Left', 'Top Centre', 'Top Right',
              'Bottom Left', 'Bottom Centre', 'Bottom Right']

known_zones = shots[shots['goal_zone'].isin(zone_order)].copy()
zone_conv = (
    known_zones.groupby(['origin_key', 'goal_zone'])['is_goal']
    .agg(['sum', 'count'])
    .reset_index()
)
zone_conv['conv_rate'] = zone_conv['sum'] / zone_conv['count'].clip(lower=3)
zone_conv.loc[zone_conv['count'] < 3, 'conv_rate'] = np.nan

# Best zone per origin_key
optimal_zone = (
    zone_conv.dropna(subset=['conv_rate'])
    .sort_values('conv_rate', ascending=False)
    .groupby('origin_key')
    .first()
    .reset_index()
    [['origin_key', 'goal_zone']]
    .rename(columns={'goal_zone': 'optimal_zone'})
)

shots = shots.merge(optimal_zone, on='origin_key', how='left')

# Score: 100 = optimal, 50 = same height row, 0 = different height + wrong side
def optimal_placement_score(actual_zone, optimal_zone_val) -> float:
    if pd.isna(actual_zone) or pd.isna(optimal_zone_val):
        return np.nan
    if actual_zone == 'Unknown':
        return np.nan
    if actual_zone == optimal_zone_val:
        return 100.0
    actual_h,   actual_side   = actual_zone.split()
    optimal_h,  optimal_side  = optimal_zone_val.split()
    if actual_h == optimal_h:
        return 50.0   # right height, wrong lateral position
    if actual_side == optimal_side:
        return 30.0   # right lateral, wrong height
    return 0.0

shots['optimal_placement'] = shots.apply(
    lambda r: optimal_placement_score(r['goal_zone'], r.get('optimal_zone')), axis=1
)

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-OUTCOME MODEL (Section 4)
# 4-class XGBoost: P(miss) P(post) P(save) P(goal) per shot
# Classes: 0=miss, 1=post, 2=save/blocked, 3=goal
# Uses pre-shot features only (no placement — those aren't known before shot)
# ─────────────────────────────────────────────────────────────────────────────
MO_FEATURES = [
    'x', 'y_sym', 'distance', 'log_distance', 'angle', 'angle_sin',
    'dist_to_post', 'in_six_yard', 'in_penalty_box', 'central_y',
    'is_header', 'is_right_foot', 'is_left_foot', 'weak_foot',
    'is_volley', 'is_deflected', 'is_first_time', 'is_big_chance',
    'is_fast_break', 'is_from_corner', 'is_set_piece', 'is_open_play',
    'is_pull_back', 'under_pressure',
]

OUTCOME_MAP  = {13: 0, 14: 1, 15: 2, 16: 3}
OUTCOME_NAME = {0: 'Miss', 1: 'Post', 2: 'Save', 3: 'Goal'}

nonpen = shots[shots['is_penalty'] == 0].copy().reset_index(drop=True)
X_mo   = nonpen[MO_FEATURES].fillna(0).astype(float)
y_mo   = nonpen['type_id'].map(OUTCOME_MAP).astype(int)
groups = nonpen['match_file']

gss = GroupShuffleSplit(1, test_size=0.20, random_state=42)
dev_idx, test_idx = next(gss.split(X_mo, y_mo, groups=groups))
gss2 = GroupShuffleSplit(1, test_size=0.25, random_state=0)
tr_idx, cal_idx = next(gss2.split(
    X_mo.iloc[dev_idx], y_mo.iloc[dev_idx], groups=groups.iloc[dev_idx]
))

X_tr,  y_tr  = X_mo.iloc[dev_idx].iloc[tr_idx],  y_mo.iloc[dev_idx].iloc[tr_idx]
X_cal, y_cal = X_mo.iloc[dev_idx].iloc[cal_idx],  y_mo.iloc[dev_idx].iloc[cal_idx]
X_test,y_test= X_mo.iloc[test_idx],               y_mo.iloc[test_idx]
g_tr         = groups.iloc[dev_idx].iloc[tr_idx]

print("\n── Multi-Outcome Model (4-class) ──")
if HAS_XGB:
    mo_clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        eval_metric='mlogloss',
        use_label_encoder=False,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    mo_clf.fit(X_tr, y_tr)
    test_proba = mo_clf.predict_proba(X_test)
    test_logloss = log_loss(y_test, test_proba)
    # Per-class AUC (one-vs-rest)
    for c, name in OUTCOME_NAME.items():
        y_binary = (y_test == c).astype(int)
        if y_binary.sum() > 0:
            try:
                auc_c = roc_auc_score(y_binary, test_proba[:, c])
                print(f"  {name}: AUC={auc_c:.4f}")
            except Exception:
                pass
    print(f"  Log-loss: {test_logloss:.4f}")

    # Predict on all non-penalty shots
    all_proba = mo_clf.predict_proba(X_mo)
    nonpen['p_miss'] = all_proba[:, 0]
    nonpen['p_post'] = all_proba[:, 1]
    nonpen['p_save'] = all_proba[:, 2]
    nonpen['p_goal_mo'] = all_proba[:, 3]  # independent estimate from multi-outcome

    for col in ['p_miss', 'p_post', 'p_save', 'p_goal_mo']:
        shots.loc[shots['is_penalty'] == 0, col] = nonpen[col].values

    joblib.dump(mo_clf, os.path.join(OUTPUT_DIR, 'model_multi_outcome.pkl'))
    print("  Multi-outcome model saved.")
else:
    print("  Skipped (xgboost not available)")

# ─────────────────────────────────────────────────────────────────────────────
# METRIC 4 — Rolling Shot Quality (last 10 shots per player)
# Sorted by season + time_min within season (approximate chronological order).
# rolling_xg10, rolling_psxg10, rolling_goals10
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Rolling Shot Quality ──")

shots_sorted = shots.sort_values(['player_name', 'season', 'time_min']).copy()

for col, src in [('rolling_xg10', 'xg'), ('rolling_psxg10', 'psxg'), ('rolling_goals10', 'is_goal')]:
    shots_sorted[col] = (
        shots_sorted.groupby('player_name')[src]
        .transform(lambda s: s.rolling(10, min_periods=3).mean())
    )

shots = shots_sorted.sort_index()
print("  Rolling metrics computed.")

# ─────────────────────────────────────────────────────────────────────────────
# METRIC 5 — Shot Archetypes (K-Means, k=6)
# Clusters shots into 6 types based on position, technique, and placement.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Shot Archetypes (k=6) ──")

CLUSTER_FEATURES = [
    'x', 'y_sym', 'log_distance', 'angle_sin',
    'is_header', 'is_first_time', 'is_volley', 'under_pressure',
    'goal_y_norm', 'goal_h_norm',
]

cluster_data = shots[CLUSTER_FEATURES].copy()
# Fill placement NaN with neutral values (goal centre, medium height)
cluster_data['goal_y_norm'] = cluster_data['goal_y_norm'].clip(-1, 1).fillna(0.0)
cluster_data['goal_h_norm'] = cluster_data['goal_h_norm'].clip(0, 1).fillna(0.35)

scaler  = StandardScaler()
X_clust = scaler.fit_transform(cluster_data.fillna(0).values)

km = KMeans(n_clusters=6, random_state=42, n_init=20)
shots['archetype_id'] = km.fit_predict(X_clust)

# Name clusters from their centroids
centres = scaler.inverse_transform(km.cluster_centers_)
centre_df = pd.DataFrame(centres, columns=CLUSTER_FEATURES)

# Auto-name based on dominant characteristics
def name_archetype(row) -> str:
    if row['is_header'] > 0.4:
        return 'Aerial Header'
    if row['x'] >= 88 and row['y_sym'] < 6:
        return 'Close-Range Central'
    if row['log_distance'] > np.log(22):
        return 'Long-Range Effort'
    if row['under_pressure'] > 0.5:
        return 'Pressured Attempt'
    if row['is_first_time'] > 0.25 or row['is_volley'] > 0.25:
        return 'First-Time / Volley'
    return 'Placed Box Finish'

archetype_names = {
    i: name_archetype(centre_df.iloc[i]) for i in range(6)
}
# Ensure unique names
seen = {}
for k, v in archetype_names.items():
    n = seen.get(v, 0)
    if n > 0:
        archetype_names[k] = f'{v} ({n+1})'
    seen[v] = n + 1

shots['archetype'] = shots['archetype_id'].map(archetype_names)

arch_stats = (
    shots.groupby('archetype')
    .agg(
        n=('is_goal','count'),
        goals=('is_goal','sum'),
        xg_mean=('xg','mean'),
        psxg_mean=('psxg','mean'),
        ot_pct=('is_on_target','mean'),
        ps_mean=('placement_score','mean'),
        power_mean=('shot_power','mean'),
    )
    .reset_index()
)
arch_stats['conv_rate'] = arch_stats['goals'] / arch_stats['n']
print(arch_stats[['archetype','n','conv_rate','xg_mean','psxg_mean','power_mean']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# METRIC 6 — Pressure Efficiency
# For each player: conversion rate and mean xG under pressure vs not
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Pressure Efficiency ──")

pressure_eff = (
    shots[~shots['is_penalty'].fillna(0).astype(bool)]
    .groupby(['player_name', 'under_pressure'])
    .agg(shots_n=('xg','count'), goals=('is_goal','sum'),
         xg_mean=('xg','mean'), psxg_mean=('psxg','mean'))
    .reset_index()
)
pressure_eff['conv_rate'] = pressure_eff['goals'] / pressure_eff['shots_n'].clip(1)

# Pivot to get pressure vs open side by side
peff_pivot = pressure_eff.pivot_table(
    index='player_name', columns='under_pressure',
    values=['conv_rate', 'xg_mean', 'psxg_mean', 'shots_n'],
    aggfunc='first'
).reset_index()
peff_pivot.columns = ['_'.join(str(c) for c in col).strip('_')
                      if col[1] != '' else col[0]
                      for col in peff_pivot.columns]
peff_pivot = peff_pivot.rename(columns={
    'conv_rate_0': 'conv_open', 'conv_rate_1': 'conv_pressure',
    'xg_mean_0':  'xg_open',   'xg_mean_1':  'xg_pressure',
    'shots_n_0':  'n_open',     'shots_n_1':  'n_pressure',
})
peff_pivot['pressure_conv_drop'] = (
    peff_pivot['conv_open'].fillna(0) - peff_pivot['conv_pressure'].fillna(0)
)
peff_pivot.to_csv(os.path.join(OUTPUT_DIR, 'pressure_efficiency.csv'), index=False)

# ─────────────────────────────────────────────────────────────────────────────
# PLAYER SUMMARIES — extended
# ─────────────────────────────────────────────────────────────────────────────
player_ext = (
    shots.groupby(['player_name', 'season'])
    .agg(
        shots_n=('xg','count'),
        goals=('is_goal','sum'),
        xg=('xg','sum'),
        psxg=('psxg','sum'),
        on_target=('is_on_target','sum'),
        placement_q_mean=('placement_score','mean'),
        shot_power_mean=('shot_power','mean'),
        contact_quality_mean=('contact_quality','mean'),
        technique_mean=('technique_index','mean'),
        optimal_placement_mean=('optimal_placement','mean'),
        finishing_luck=('finishing_luck','sum'),
    )
    .reset_index()
)
player_ext['conv_rate']   = player_ext['goals'] / player_ext['shots_n'].clip(1)
player_ext['ot_pct']      = player_ext['on_target'] / player_ext['shots_n'].clip(1)
player_ext['goals_minus_psxg'] = player_ext['goals'] - player_ext['psxg']
player_ext.to_csv(os.path.join(OUTPUT_DIR, 'player_extended_summary.csv'), index=False)

beth = player_ext[
    player_ext['player_name'].str.contains('Mead', na=False)
].sort_values('season')
print("\n── Beth Mead Extended Summary ──")
print(beth[['season','shots_n','goals','conv_rate','xg','psxg',
            'placement_q_mean','shot_power_mean','contact_quality_mean',
            'optimal_placement_mean','goals_minus_psxg']].to_string(index=False))

# Save enriched shot file
shots.to_csv(OUT_CSV, index=False)
print(f"\nEnriched shot file saved → {OUT_CSV}")

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
plt.style.use('dark_background')

# ── DASHBOARD 1: Shot analytics overview ─────────────────────────────────────
fig = plt.figure(figsize=(22, 18), facecolor=FIG_BG)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.40)

# 1a. Multi-outcome probability distributions
ax1 = fig.add_subplot(gs[0, 0])
if 'p_goal_mo' in shots.columns:
    bins = np.linspace(0, 1, 30)
    for col, label, color in [
        ('p_goal_mo', 'P(Goal)',  C_YELLOW),
        ('p_save',    'P(Save)',  C_BLUE),
        ('p_miss',    'P(Miss)',  C_ORANGE),
        ('p_post',    'P(Post)',  C_PURPLE),
    ]:
        if col in shots.columns:
            ax1.hist(shots[col].dropna(), bins=bins, alpha=0.5,
                     color=color, density=True, label=label)
    ax1.set(xlabel='Probability', title='Multi-Outcome Distributions', facecolor=FIG_BG)
    ax1.legend(fontsize=7)

# 1b. Shot Power by body part
ax2 = fig.add_subplot(gs[0, 1])
bp_labels = {
    (1,0,0): 'Header',
    (0,1,0): 'Right Foot',
    (0,0,1): 'Left Foot',
}
for (h,r,l), label, color in [
    ((1,0,0), 'Header',     C_PURPLE),
    ((0,1,0), 'Right Foot', C_BLUE),
    ((0,0,1), 'Left Foot',  C_GREEN),
]:
    sub = shots[(shots['is_header']==h)&(shots['is_right_foot']==r)&(shots['is_left_foot']==l)]
    if len(sub) > 0:
        ax2.hist(sub['shot_power'].dropna(), bins=20, alpha=0.55,
                 color=color, density=True, label=f'{label} (n={len(sub):,})')
ax2.set(xlabel='Shot Power Index', title='Shot Power by Body Part', facecolor=FIG_BG)
ax2.legend(fontsize=7)

# 1c. Archetype breakdown: conversion rate bar
ax3 = fig.add_subplot(gs[0, 2])
arch_sorted = arch_stats.sort_values('conv_rate', ascending=True)
colors_arch = plt.cm.plasma(np.linspace(0.2, 0.9, len(arch_sorted)))
bars = ax3.barh(arch_sorted['archetype'], arch_sorted['conv_rate'],
                color=colors_arch, alpha=0.85)
ax3.set_xlabel('Conversion Rate', fontsize=8)
ax3.set_title('Shot Archetypes — Conversion Rate', fontsize=9)
ax3.set_facecolor(FIG_BG)
ax3.tick_params(labelsize=7)
for bar, (_, row) in zip(bars, arch_sorted.iterrows()):
    ax3.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f"n={row['n']:,}", va='center', fontsize=6, color='white')

# 1d. Optimal placement: hit rate per origin bin
ax4 = fig.add_subplot(gs[0, 3])
opt_hit = (
    shots[shots['optimal_placement'].notna()]
    .groupby('origin_bin')
    .agg(
        pct_optimal=('optimal_placement', lambda s: (s==100).mean()),
        n=('optimal_placement','count'),
    )
    .reset_index()
    .sort_values('pct_optimal', ascending=True)
)
ax4.barh(opt_hit['origin_bin'], opt_hit['pct_optimal'],
         color=C_GREEN, alpha=0.8)
ax4.set_xlabel('% Shots in Optimal Zone', fontsize=8)
ax4.set_title('Optimal Zone Hit Rate\nby Shot Origin', fontsize=9)
ax4.set_facecolor(FIG_BG)
ax4.tick_params(labelsize=7)
for i, (_, row) in enumerate(opt_hit.iterrows()):
    ax4.text(row['pct_optimal']+0.005, i, f"n={row['n']:,}",
             va='center', fontsize=6, color='white')

# 1e. Contact quality vs xG scatter (sample)
ax5 = fig.add_subplot(gs[1, 0])
sample = shots[shots['contact_quality'].notna()].sample(min(3000, len(shots)), random_state=0)
c_vals = [C_YELLOW if g else C_MUTED for g in sample['is_goal']]
ax5.scatter(sample['xg'], sample['contact_quality'],
            c=c_vals, alpha=0.25, s=5)
ax5.set(xlabel='Pre-Shot xG', ylabel='Contact Quality Index',
        title='Contact Quality vs xG\n(yellow=goal)', facecolor=FIG_BG)

# 1f. Pressure efficiency: top 20 players by shot volume
ax6 = fig.add_subplot(gs[1, 1])
peff_top = peff_pivot.dropna(subset=['conv_open','conv_pressure'])
peff_top = peff_top[
    peff_top['n_open'].fillna(0) + peff_top['n_pressure'].fillna(0) >= 50
].sort_values('pressure_conv_drop', ascending=False).head(15)
y_pos = np.arange(len(peff_top))
ax6.barh(y_pos, peff_top['conv_open'].values,     0.4, color=C_BLUE,   alpha=0.8, label='Open')
ax6.barh(y_pos+0.4, peff_top['conv_pressure'].values, 0.4, color=C_ORANGE, alpha=0.8, label='Pressure')
ax6.set_yticks(y_pos+0.2)
ax6.set_yticklabels(peff_top['player_name'].values, fontsize=7)
ax6.set_xlabel('Conversion Rate', fontsize=8)
ax6.set_title('Pressure vs Open Conversion\n(top 15 by drop-off)', fontsize=9)
ax6.legend(fontsize=7)
ax6.set_facecolor(FIG_BG)

# 1g. Placement score vs psxG heatmap
ax7 = fig.add_subplot(gs[1, 2])
ot = shots[(shots['is_on_target']==1) & shots['placement_score'].notna()].copy()
h2d = ax7.hist2d(
    ot['placement_score'].clip(0,100),
    ot['psxg'],
    bins=(20, 20), cmap='magma',
    range=[[0,100],[0,1]]
)
plt.colorbar(h2d[3], ax=ax7)
ax7.set(xlabel='Placement Score', ylabel='Post-Shot xG',
        title='Placement Score vs psxG\n(on-target shots)', facecolor=FIG_BG)

# 1h. P(Goal) from multi-outcome vs pre-shot xG comparison
ax8 = fig.add_subplot(gs[1, 3])
if 'p_goal_mo' in shots.columns:
    nonpen_sample = shots[(shots['is_penalty']==0) & shots['p_goal_mo'].notna()].sample(
        min(3000, len(shots)), random_state=1
    )
    ax8.scatter(nonpen_sample['xg'], nonpen_sample['p_goal_mo'],
                c=C_BLUE, alpha=0.15, s=5)
    ax8.plot([0,1],[0,1],'w--',lw=1,alpha=0.4)
    ax8.set(xlabel='Pre-Shot xG (single-stage)',
            ylabel='P(Goal) from Multi-Outcome',
            title='xG vs Multi-Outcome P(Goal)', facecolor=FIG_BG)

# ── Row 2: Beth Mead deep-dive ────────────────────────────────────────────────
beth_shots = shots[shots['player_name'].str.contains('Mead', na=False)].copy()
beth_shots_s = beth_shots.sort_values(['season', 'time_min'])

# Rolling form curve
ax9 = fig.add_subplot(gs[2, :2])
if len(beth_shots_s) > 0 and 'rolling_xg10' in beth_shots_s.columns:
    idx = np.arange(len(beth_shots_s))
    ax9.fill_between(idx, beth_shots_s['rolling_xg10'].fillna(0),
                     alpha=0.3, color=C_BLUE, label='Rolling xG (last 10)')
    ax9.fill_between(idx, beth_shots_s['rolling_psxg10'].fillna(0),
                     alpha=0.3, color=C_GREEN, label='Rolling psxG (last 10)')
    ax9.fill_between(idx, beth_shots_s['rolling_goals10'].fillna(0),
                     alpha=0.4, color=C_YELLOW, label='Rolling Goals (last 10)')
    # Season boundary lines
    season_starts = beth_shots_s.reset_index(drop=True).groupby('season').apply(
        lambda df: df.index[0]
    )
    for s_idx, season_name in zip(season_starts.values, season_starts.index):
        ax9.axvline(s_idx, color='white', alpha=0.2, lw=0.8)
        ax9.text(s_idx+1, ax9.get_ylim()[1]*0.85 if ax9.get_ylim()[1] > 0 else 0.8,
                 season_name.replace('WSL ',''), color='white', fontsize=6, alpha=0.6)
    ax9.set(xlabel='Shot number (career)', ylabel='Rolling mean',
            title='Beth Mead — Career Rolling Shot Quality (10-shot window)', facecolor=FIG_BG)
    ax9.legend(fontsize=8)

# Archetype distribution: Beth Mead vs all players
ax10 = fig.add_subplot(gs[2, 2])
if 'archetype' in shots.columns and len(beth_shots) > 0:
    all_arch  = shots[shots['is_penalty']==0]['archetype'].value_counts(normalize=True)
    beth_arch = beth_shots['archetype'].value_counts(normalize=True)
    arch_keys = sorted(set(all_arch.index) | set(beth_arch.index))
    x_a = np.arange(len(arch_keys))
    ax10.bar(x_a - 0.2, [all_arch.get(k,0) for k in arch_keys],  0.4,
             color=C_MUTED, alpha=0.8, label='All WSL')
    ax10.bar(x_a + 0.2, [beth_arch.get(k,0) for k in arch_keys], 0.4,
             color=C_YELLOW, alpha=0.8, label='Beth Mead')
    ax10.set_xticks(x_a)
    ax10.set_xticklabels([k.replace(' ','\n') for k in arch_keys], fontsize=6)
    ax10.set_ylabel('Share of shots', fontsize=8)
    ax10.set_title('Shot Archetype Mix\nBeth Mead vs All WSL', fontsize=9)
    ax10.legend(fontsize=7)
    ax10.set_facecolor(FIG_BG)

# Shot map coloured by archetype
ax11 = fig.add_subplot(gs[2, 3])
if len(beth_shots) > 0 and 'archetype_id' in beth_shots.columns:
    vpitch = VerticalPitch(pitch_type='opta', pitch_color=PITCH_BG,
                           line_color=LINE_COL, linewidth=0.8, half=True)
    vpitch.draw(ax=ax11)
    arch_colors = plt.cm.tab10(np.linspace(0, 1, 6))
    arch_color_map = {v: arch_colors[i] for i, (k, v) in enumerate(archetype_names.items())}
    for _, row in beth_shots.iterrows():
        pv  = max(float(row.get('psxg', 0.03)), 0.03)
        col = arch_color_map.get(row['archetype'], (0.5,0.5,0.5,1))
        mk  = '*' if row['is_goal'] else 'o'
        ax11.scatter(row['y'], row['x'], s=pv*280, color=[col],
                     marker=mk, alpha=0.75, linewidths=0.4,
                     edgecolors='white' if row['is_goal'] else 'none', zorder=3)
    legend_handles = [
        mpatches.Patch(color=arch_color_map.get(v, 'gray'), label=v, alpha=0.8)
        for v in archetype_names.values()
    ]
    ax11.legend(handles=legend_handles, fontsize=5.5, loc='lower center',
                facecolor=FIG_BG, edgecolor='#30363d', labelcolor='white')
    ax11.set_title('Beth Mead Shots\nby Archetype (size=psxG)', fontsize=9)

fig.suptitle('WSL Extended Shot Analytics Dashboard', fontsize=15,
             fontweight='bold', color='white', y=0.997)
plt.savefig(os.path.join(OUTPUT_DIR, 'shot_analytics_dashboard.png'),
            dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close()
print(f"Shot analytics dashboard saved.")

# ── DASHBOARD 2: Multi-outcome pitch maps ─────────────────────────────────────
if 'p_goal_mo' in shots.columns:
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12), facecolor=FIG_BG)
    for ax, col, title, cmap in [
        (axes2[0,0], 'p_goal_mo', 'P(Goal) — Pitch Heatmap',  'YlOrRd'),
        (axes2[0,1], 'p_save',    'P(Save) — Pitch Heatmap',  'Blues'),
        (axes2[1,0], 'p_miss',    'P(Miss) — Pitch Heatmap',  'Oranges'),
        (axes2[1,1], 'p_post',    'P(Post Hit) — Pitch Heatmap','Purples'),
    ]:
        pitch = Pitch(pitch_type='opta', pitch_color=PITCH_BG,
                      line_color=LINE_COL, linewidth=0.8)
        pitch.draw(ax=ax)
        bin_stat = pitch.bin_statistic(
            shots['x'], shots['y'],
            values=shots[col].fillna(0),
            statistic='mean', bins=(14, 9)
        )
        pitch.heatmap(bin_stat, ax=ax, cmap=cmap)
        ax.set_title(title, fontsize=10, color='white')

    fig2.suptitle('Multi-Outcome Probability Maps', fontsize=14,
                  fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'multi_outcome_pitch_maps.png'),
                dpi=150, bbox_inches='tight', facecolor=FIG_BG)
    plt.close()
    print("Multi-outcome pitch maps saved.")

# ── DASHBOARD 3: Goal frame analysis ─────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10), facecolor=FIG_BG)

def draw_goal_frame(ax):
    ax.add_patch(plt.Rectangle(
        (GOAL_Y_LEFT, 0), GOAL_WIDTH, 100,
        fill=False, edgecolor='white', lw=2, zorder=5
    ))
    ax.axvline(GOAL_Y_CENTRE, color='white', ls='--', alpha=0.3, lw=1)
    ax.set_xlim(GOAL_Y_LEFT-1, GOAL_Y_RIGHT+1)
    ax.set_ylim(-5, 105)
    ax.set_xlabel('Lateral Position', fontsize=8)
    ax.set_ylabel('Height (0–100)', fontsize=8)

# Goals: where they went
ax_gl = axes3[0, 0]
goals_with_placement = shots[
    (shots['is_goal']==1) &
    shots['goal_y_raw'].notna() &
    shots['goal_h_raw'].notna()
]
ax_gl.hist2d(goals_with_placement['goal_y_raw'],
             goals_with_placement['goal_h_raw'],
             bins=(12,10), cmap='YlOrRd',
             range=[[GOAL_Y_LEFT-1,GOAL_Y_RIGHT+1],[0,100]])
draw_goal_frame(ax_gl)
ax_gl.set_title(f'Goals ({len(goals_with_placement):,})', color='white', fontsize=10)
ax_gl.set_facecolor(PITCH_BG)

# Saves: where they went
ax_sv = axes3[0, 1]
saves_with_placement = shots[
    (shots['is_blocked']==1) &
    shots['goal_y_raw'].notna() &
    shots['goal_h_raw'].notna()
]
ax_sv.hist2d(saves_with_placement['goal_y_raw'],
             saves_with_placement['goal_h_raw'],
             bins=(12,10), cmap='Blues',
             range=[[GOAL_Y_LEFT-1,GOAL_Y_RIGHT+1],[0,100]])
draw_goal_frame(ax_sv)
ax_sv.set_title(f'Saves ({len(saves_with_placement):,})', color='white', fontsize=10)
ax_sv.set_facecolor(PITCH_BG)

# Conversion rate by goal zone
ax_gz = axes3[0, 2]
gz_conv = (
    shots[shots['goal_zone'].isin([
        'Top Left','Top Centre','Top Right',
        'Bottom Left','Bottom Centre','Bottom Right'
    ])]
    .groupby('goal_zone')['is_goal']
    .agg(['sum','count'])
)
gz_conv['rate'] = gz_conv['sum'] / gz_conv['count']
# Plot as 2x3 grid
grid = np.zeros((2, 3))
row_map = {'Top': 1, 'Bottom': 0}
col_map = {'Left': 0, 'Centre': 1, 'Right': 2}
for zone, row in gz_conv.iterrows():
    h, s = zone.split()
    grid[row_map[h], col_map[s]] = row['rate']
im = ax_gz.imshow(grid, cmap='RdYlGn', vmin=0, vmax=0.5, aspect='auto')
ax_gz.set_xticks([0,1,2]); ax_gz.set_xticklabels(['Left','Centre','Right'])
ax_gz.set_yticks([0,1]); ax_gz.set_yticklabels(['Bottom','Top'])
for i in range(2):
    for j in range(3):
        z = list(gz_conv.index)[i*3+j] if i*3+j < len(gz_conv) else ''
        rate = grid[i,j]
        n = int(gz_conv.iloc[i*3+j]['count']) if i*3+j < len(gz_conv) else 0
        ax_gz.text(j, i, f'{rate:.2f}\n(n={n})',
                   ha='center', va='center', fontsize=9, color='black', fontweight='bold')
ax_gz.set_title('Conversion Rate by Goal Zone', color='white', fontsize=10)
plt.colorbar(im, ax=ax_gz)

# Placement score distribution by outcome
ax_ps = axes3[1, 0]
for tid, label, color in [(16,'Goal',C_YELLOW),(15,'Save',C_BLUE),(13,'Miss',C_ORANGE)]:
    sub = shots[(shots['type_id']==tid) & shots['placement_score'].notna()]
    sub_clipped = sub['placement_score'].clip(0,100)
    ax_ps.hist(sub_clipped, bins=20, alpha=0.55, color=color,
               density=True, label=f'{label} (n={len(sub):,})', range=(0,100))
ax_ps.set(xlabel='Placement Score', title='Placement Score by Outcome', facecolor=FIG_BG)
ax_ps.legend(fontsize=7)

# Contact quality distribution by outcome
ax_cq = axes3[1, 1]
for tid, label, color in [(16,'Goal',C_YELLOW),(15,'Save',C_BLUE),(13,'Miss',C_ORANGE)]:
    sub = shots[(shots['type_id']==tid) & shots['contact_quality'].notna()]
    ax_cq.hist(sub['contact_quality'].clip(0,100), bins=20, alpha=0.55,
               color=color, density=True, label=label, range=(0,100))
ax_cq.set(xlabel='Contact Quality Index', title='Contact Quality by Outcome', facecolor=FIG_BG)
ax_cq.legend(fontsize=7)

# Shot power by outcome
ax_sp = axes3[1, 2]
for tid, label, color in [(16,'Goal',C_YELLOW),(15,'Save',C_BLUE),(13,'Miss',C_ORANGE)]:
    sub = shots[shots['type_id']==tid]
    ax_sp.hist(sub['shot_power'].clip(0,100), bins=20, alpha=0.55,
               color=color, density=True, label=label)
ax_sp.set(xlabel='Shot Power Index', title='Shot Power by Outcome', facecolor=FIG_BG)
ax_sp.legend(fontsize=7)

fig3.suptitle('Goal Frame & Shot Quality Analysis', fontsize=14,
              fontweight='bold', color='white')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'goal_frame_analysis.png'),
            dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close()
print("Goal frame analysis saved.")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Model & Metric Summary ──")
print(f"  Multi-outcome model: P(miss|post|save|goal) per shot")
print(f"  placement_score fixed (max={shots['placement_score'].max():.1f})")
print(f"  shot_power range: {shots['shot_power'].min():.0f}–{shots['shot_power'].max():.0f}")
print(f"  contact_quality range: {shots['contact_quality'].min():.1f}–{shots['contact_quality'].max():.1f}")
print(f"  optimal_placement coverage: {shots['optimal_placement'].notna().sum():,} / {len(shots):,}")
print(f"  rolling_xg10 coverage: {shots['rolling_xg10'].notna().sum():,} / {len(shots):,}")
print(f"  archetypes: {shots['archetype'].value_counts().to_dict()}")
print("\nAll done.")
