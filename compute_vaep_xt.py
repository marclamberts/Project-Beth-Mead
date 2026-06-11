"""
compute_vaep_xt.py
Calculates xT (Expected Threat) and VAEP (Valuing Actions by Estimating Probabilities)
for every player in the WSL dataset.
"""

import json
import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = "/home/user/Project-Beth-Mead"
OUTPUT_CSV = os.path.join(DATA_ROOT, "vaep_xt_results.csv")
SUMMARY_CSV = os.path.join(DATA_ROOT, "player_vaep_xt_summary.csv")

# xT grid
XT_COLS = 16   # x-axis (0-100)
XT_ROWS = 12   # y-axis (0-100)

# Action type IDs
PASS_IDS    = {1}
DRIBBLE_IDS = {3}
TACKLE_IDS  = {7}
INTERC_IDS  = {8}
CLEAR_IDS   = {12}
MISS_IDS    = {13, 14}
SAVED_IDS   = {15}
GOAL_IDS    = {16}
SHOT_IDS    = MISS_IDS | SAVED_IDS | GOAL_IDS

ACTION_TYPE_MAP = {
    1:  "pass",
    3:  "dribble",
    7:  "tackle",
    8:  "interception",
    12: "clearance",
    13: "miss",
    14: "post",
    15: "attempt_saved",
    16: "goal",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_qualifiers(event):
    return {q['qualifierId']: q.get('value') for q in event.get('qualifier', [])}


def safe_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def coord_to_cell(x, y, n_cols=XT_COLS, n_rows=XT_ROWS):
    col = int(np.clip(x / 100.0 * n_cols, 0, n_cols - 1))
    row = int(np.clip(y / 100.0 * n_rows, 0, n_rows - 1))
    return col, row


def cell_idx(col, row, n_cols=XT_COLS):
    return row * n_cols + col


# ---------------------------------------------------------------------------
# Step 1: Load all events
# ---------------------------------------------------------------------------

def load_all_events():
    pattern1 = os.path.join(DATA_ROOT, "WSL*/**/*.json")
    pattern2 = os.path.join(DATA_ROOT, "WSL*/*.json")
    all_files = sorted(set(glob.glob(pattern1, recursive=True) + glob.glob(pattern2)))
    print(f"Found {len(all_files)} JSON files")

    all_events = []
    for file_idx, fpath in enumerate(all_files):
        if file_idx > 0 and file_idx % 100 == 0:
            print(f"  Processed {file_idx}/{len(all_files)} files, {len(all_events):,} events so far")
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  WARNING: Could not read {fpath}: {e}")
            continue

        parts = os.path.normpath(fpath).split(os.sep)
        season_dir = next((p for p in parts if p.startswith("WSL")), "Unknown")
        season = season_dir.replace("WSL ", "WSL_")
        match_id = os.path.splitext(os.path.basename(fpath))[0]

        events = data.get('event', [])
        if not isinstance(events, list):
            continue

        for ev in events:
            type_id = ev.get('typeId')
            if type_id is None:
                continue
            quals = get_qualifiers(ev)
            all_events.append({
                'season':       season,
                'match_id':     match_id,
                'typeId':       type_id,
                'periodId':     ev.get('periodId', 0),
                'timeMin':      ev.get('timeMin', 0),
                'timeSec':      ev.get('timeSec', 0),
                'contestantId': ev.get('contestantId', ''),
                'playerId':     ev.get('playerId', ''),
                'playerName':   ev.get('playerName', ''),
                'outcome':      ev.get('outcome', 0),
                'x':            safe_float(ev.get('x'), 0.0),
                'y':            safe_float(ev.get('y'), 0.0),
                'end_x':        safe_float(quals.get(140)),
                'end_y':        safe_float(quals.get(141)),
                'eventId':      ev.get('eventId', 0),
            })

    print(f"  Processed {len(all_files)}/{len(all_files)} files, {len(all_events):,} events total")
    return all_events, all_files


# ---------------------------------------------------------------------------
# Step 2: xT Model
# ---------------------------------------------------------------------------

def build_xt_model(df):
    print("\nBuilding xT model...")
    n_cells = XT_ROWS * XT_COLS
    shot_count  = np.zeros(n_cells, dtype=float)
    goal_count  = np.zeros(n_cells, dtype=float)
    move_count  = np.zeros(n_cells, dtype=float)
    move_matrix = np.zeros((n_cells, n_cells), dtype=float)

    # Vectorised cell index computation
    def cells_for(sub_df, use_end=False):
        xs = sub_df['end_x'].values if use_end else sub_df['x'].values
        ys = sub_df['end_y'].values if use_end else sub_df['y'].values
        cols = np.clip((xs / 100.0 * XT_COLS).astype(int), 0, XT_COLS - 1)
        rows = np.clip((ys / 100.0 * XT_ROWS).astype(int), 0, XT_ROWS - 1)
        return rows * XT_COLS + cols

    shots = df[df['typeId'].isin(SHOT_IDS)]
    goals = df[df['typeId'].isin(GOAL_IDS)]
    moves = df[df['typeId'].isin(PASS_IDS | DRIBBLE_IDS) & (df['outcome'] == 1) &
               df['end_x'].notna() & df['end_y'].notna()]

    np.add.at(shot_count, cells_for(shots), 1)
    np.add.at(goal_count, cells_for(goals), 1)

    start_cells = cells_for(moves)
    end_cells   = cells_for(moves, use_end=True)
    np.add.at(move_count, start_cells, 1)
    for sc, ec in zip(start_cells, end_cells):
        move_matrix[sc, ec] += 1

    total = shot_count + move_count
    shot_rate = np.where(total > 0, shot_count / total, 0.0)
    goal_rate = np.where(shot_count > 0, goal_count / shot_count, 0.0)

    row_sums  = move_matrix.sum(axis=1, keepdims=True)
    move_prob = np.where(row_sums > 0, move_matrix / row_sums, 0.0)

    xT = np.zeros(n_cells, dtype=float)
    for _ in range(10):
        xT = shot_rate * goal_rate + (1 - shot_rate) * (move_prob @ xT)

    xT_grid = xT.reshape(XT_ROWS, XT_COLS)
    print(f"  xT model built. Max={xT_grid.max():.4f}, Mean={xT_grid.mean():.6f}")
    return xT_grid


def compute_xt_values(df, xT_grid):
    xt_values = np.full(len(df), np.nan)
    mask = (df['typeId'].isin(PASS_IDS | DRIBBLE_IDS) & (df['outcome'] == 1) &
            df['end_x'].notna() & df['end_y'].notna()).values
    sub = df[mask]

    xs1 = sub['x'].values;   ys1 = sub['y'].values
    xs2 = sub['end_x'].values; ys2 = sub['end_y'].values

    cols1 = np.clip((xs1 / 100.0 * XT_COLS).astype(int), 0, XT_COLS - 1)
    rows1 = np.clip((ys1 / 100.0 * XT_ROWS).astype(int), 0, XT_ROWS - 1)
    cols2 = np.clip((xs2 / 100.0 * XT_COLS).astype(int), 0, XT_COLS - 1)
    rows2 = np.clip((ys2 / 100.0 * XT_ROWS).astype(int), 0, XT_ROWS - 1)

    start_xt = xT_grid[rows1, cols1]
    end_xt   = xT_grid[rows2, cols2]
    vals     = end_xt - start_xt

    xt_values[np.where(mask)[0]] = vals
    return xt_values


# ---------------------------------------------------------------------------
# Step 3: VAEP Model
# ---------------------------------------------------------------------------

N_ACTION_FEATURES = 10
N_CONTEXT = 2
FEATURE_DIM = N_ACTION_FEATURES * (1 + N_CONTEXT)


def action_feature_matrix(df):
    x  = np.clip(df['x'].fillna(50).values / 100.0,  0, 1)
    y  = np.clip(df['y'].fillna(50).values / 100.0,  0, 1)
    ex = np.where(df['end_x'].notna(), df['end_x'].fillna(50).values / 100.0, x)
    ey = np.where(df['end_y'].notna(), df['end_y'].fillna(50).values / 100.0, y)
    ex = np.clip(ex, 0, 1); ey = np.clip(ey, 0, 1)

    tid = df['typeId'].values
    is_pass  = np.isin(tid, list(PASS_IDS)).astype(float)
    is_shot  = np.isin(tid, list(SHOT_IDS)).astype(float)
    is_drib  = np.isin(tid, list(DRIBBLE_IDS)).astype(float)
    is_other = (1 - is_pass - is_shot - is_drib).clip(0, 1)
    outcome  = (df['outcome'].fillna(0).values == 1).astype(float)
    time_n   = np.clip(df['timeMin'].fillna(45).values / 90.0, 0, 2)

    feat = np.column_stack([x, y, ex, ey, is_pass, is_shot, is_drib, is_other, outcome, time_n])
    return feat


def build_vaep_features(df):
    print("\nBuilding VAEP features...")
    df = df.sort_values(['match_id', 'contestantId', 'periodId', 'timeMin', 'timeSec']).reset_index(drop=True)
    feat_matrix = action_feature_matrix(df)

    goal_mask  = df['typeId'].isin(GOAL_IDS).values
    n = len(df)

    X              = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    labels_score   = np.zeros(n, dtype=np.float32)
    labels_concede = np.zeros(n, dtype=np.float32)
    valid_mask     = np.zeros(n, dtype=bool)

    # Build per-match lookup: list of row indices sorted by time
    match_all_indices = {}
    for match_id, grp in df.groupby('match_id', sort=False):
        match_all_indices[match_id] = grp.index.values

    groups = df.groupby(['match_id', 'contestantId'], sort=False)
    print(f"  Processing {len(groups)} match-team groups...")

    gc = 0
    for (match_id, contestant_id), grp in groups:
        grp_idx = grp.index.values
        if len(grp_idx) == 0:
            continue

        all_match_idx = match_all_indices[match_id]
        match_df = df.loc[all_match_idx]

        own_goal_set = set(all_match_idx[
            (match_df['contestantId'] == contestant_id).values & goal_mask[all_match_idx]
        ])
        opp_goal_set = set(all_match_idx[
            (match_df['contestantId'] != contestant_id).values & goal_mask[all_match_idx]
        ])

        for pos, i in enumerate(grp_idx):
            feats = []
            for c in range(N_CONTEXT, 0, -1):
                if pos - c >= 0:
                    feats.extend(feat_matrix[grp_idx[pos - c]])
                else:
                    feats.extend([0.0] * N_ACTION_FEATURES)
            feats.extend(feat_matrix[i])
            X[i] = feats
            valid_mask[i] = True

            # Future actions in same match (up to 10 ahead by row order)
            pos_in_match = np.searchsorted(all_match_idx, i)
            future = all_match_idx[pos_in_match + 1: pos_in_match + 11]
            labels_score[i]   = 1.0 if any(fi in own_goal_set for fi in future) else 0.0
            labels_concede[i] = 1.0 if any(fi in opp_goal_set for fi in future) else 0.0

        gc += 1
        if gc % 500 == 0:
            print(f"    {gc}/{len(groups)} groups processed")

    print(f"  Valid actions for VAEP: {valid_mask.sum():,}")
    return df, X, labels_score, labels_concede, valid_mask


def train_vaep_models(X, labels_score, labels_concede, valid_mask):
    print("\nTraining VAEP models...")
    X_v = X[valid_mask]
    y_s = labels_score[valid_mask]
    y_c = labels_concede[valid_mask]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_v)

    clf_s = LogisticRegression(max_iter=500, C=0.1, solver='lbfgs', random_state=42)
    clf_s.fit(X_s, y_s)
    print(f"  P_score  model: score_rate={y_s.mean():.4f}")

    clf_c = LogisticRegression(max_iter=500, C=0.1, solver='lbfgs', random_state=42)
    clf_c.fit(X_s, y_c)
    print(f"  P_concede model: concede_rate={y_c.mean():.4f}")

    return clf_s, clf_c, scaler


def compute_vaep_values(df, X, valid_mask, clf_s, clf_c, scaler):
    print("\nComputing VAEP values...")
    X_scaled = scaler.transform(X[valid_mask])
    p_s = clf_s.predict_proba(X_scaled)[:, 1]
    p_c = clf_c.predict_proba(X_scaled)[:, 1]

    valid_idx = np.where(valid_mask)[0]
    full_ps = np.full(len(df), np.nan)
    full_pc = np.full(len(df), np.nan)
    full_ps[valid_idx] = p_s
    full_pc[valid_idx] = p_c

    vaep_values = np.full(len(df), np.nan)

    for (match_id, contestant_id), grp in df.groupby(['match_id', 'contestantId'], sort=False):
        grp_idx = grp.index.values
        if len(grp_idx) == 0:
            continue
        for pos, i in enumerate(grp_idx):
            if not valid_mask[i]:
                continue
            ps_after = full_ps[i]; pc_after = full_pc[i]
            if np.isnan(ps_after) or np.isnan(pc_after):
                continue
            if pos > 0:
                pi = grp_idx[pos - 1]
                ps_before = full_ps[pi] if not np.isnan(full_ps[pi]) else 0.0
                pc_before = full_pc[pi] if not np.isnan(full_pc[pi]) else 0.0
            else:
                ps_before = 0.0; pc_before = 0.0
            vaep_values[i] = (ps_after - ps_before) - (pc_after - pc_before)

    print(f"  VAEP computed for {(~np.isnan(vaep_values)).sum():,} actions")
    return vaep_values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("WSL xT and VAEP Computation")
    print("=" * 60)

    raw_events, all_files = load_all_events()
    df_all = pd.DataFrame(raw_events)
    print(f"\nLoaded {len(df_all):,} total events from {len(all_files)} files")

    df = df_all[df_all['playerName'].notna() & (df_all['playerName'] != '')].copy().reset_index(drop=True)
    print(f"Events with player attribution: {len(df):,}")

    df['action_type'] = df['typeId'].map(ACTION_TYPE_MAP).fillna('other')

    # xT
    xT_grid = build_xt_model(df)
    df['xt_value'] = compute_xt_values(df, xT_grid)

    # VAEP
    df, X, labels_score, labels_concede, valid_mask = build_vaep_features(df)
    clf_s, clf_c, scaler = train_vaep_models(X, labels_score, labels_concede, valid_mask)
    df['vaep_value'] = compute_vaep_values(df, X, valid_mask, clf_s, clf_c, scaler)

    # Team name lookup
    print("\nBuilding team name lookup...")
    team_name_map = {}
    for match_id, grp in df.groupby('match_id'):
        if '_' in match_id:
            teams_part = match_id.split('_', 1)[1]
            teams = [t.strip() for t in teams_part.split(' - ', 1)]
        else:
            continue
        if len(teams) != 2:
            continue
        first_two = grp.drop_duplicates('contestantId').sort_values('eventId')['contestantId'].values[:2]
        if len(first_two) >= 2:
            team_name_map[first_two[0]] = teams[0]
            team_name_map[first_two[1]] = teams[1]
        elif len(first_two) == 1:
            contestants = grp['contestantId'].unique()
            if len(contestants) >= 2:
                team_name_map[contestants[0]] = teams[0]
                team_name_map[contestants[1]] = teams[1]

    df['team'] = df['contestantId'].map(team_name_map).fillna(df['contestantId'])

    # Write action CSV
    print("\nWriting action-level CSV...")
    out_cols = ['season', 'match_id', 'playerName', 'team', 'typeId', 'action_type',
                'timeMin', 'x', 'y', 'end_x', 'end_y', 'xt_value', 'vaep_value']
    df_out = df[out_cols].copy()
    df_out.columns = ['season', 'match_id', 'player', 'team', 'typeId', 'action_type',
                      'timeMin', 'x', 'y', 'end_x', 'end_y', 'xt_value', 'vaep_value']
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"  Written: {OUTPUT_CSV}  ({len(df_out):,} rows)")

    # Player summary
    print("\nBuilding player summary...")
    summary_rows = []
    for (season, player, cid), grp in df.groupby(['season', 'playerName', 'contestantId']):
        n_actions    = len(grp)
        xt_valid     = grp['xt_value'].dropna()
        total_xt     = xt_valid.sum()
        xt_per_act   = xt_valid.mean() if len(xt_valid) > 0 else np.nan

        vaep_valid   = grp['vaep_value'].dropna()
        total_vaep   = vaep_valid.sum()
        vaep_per_act = vaep_valid.mean() if len(vaep_valid) > 0 else np.nan

        mins = grp['timeMin'].dropna()
        total_mins = max(mins.max() - mins.min(), 1.0) if len(mins) > 0 else 90.0
        vaep_per90 = total_vaep / total_mins * 90.0 if total_mins > 0 else np.nan

        summary_rows.append({
            'season': season, 'player': player,
            'team': team_name_map.get(cid, cid),
            'n_actions': n_actions,
            'total_xt': round(total_xt, 6),
            'xt_per_action': round(xt_per_act, 6) if not np.isnan(xt_per_act) else np.nan,
            'total_vaep': round(total_vaep, 6),
            'vaep_per_action': round(vaep_per_act, 6) if not np.isnan(vaep_per_act) else np.nan,
            'vaep_per90': round(vaep_per90, 6) if not np.isnan(vaep_per90) else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"  Written: {SUMMARY_CSV}  ({len(summary_df):,} rows)")

    # Top 10 by xT
    print("\n" + "=" * 60)
    print("TOP 10 PLAYERS BY TOTAL xT (all seasons)")
    print("=" * 60)
    xt_top = (
        summary_df.groupby('player')
        .agg(total_xt=('total_xt', 'sum'), n_actions=('n_actions', 'sum'))
        .assign(xt_per_action=lambda d: d['total_xt'] / d['n_actions'])
        .sort_values('total_xt', ascending=False)
        .head(10)
    )
    print(xt_top.to_string())

    # Top 10 by VAEP per 90 (min 500 actions)
    print("\n" + "=" * 60)
    print("TOP 10 PLAYERS BY VAEP PER 90 (min 500 actions)")
    print("=" * 60)

    def weighted_v90(grp):
        tv = grp['total_vaep'].sum()
        valid = grp[grp['vaep_per90'].notna() & (grp['vaep_per90'] != 0)]
        if len(valid) == 0:
            return np.nan
        est_mins = (valid['total_vaep'] / valid['vaep_per90'] * 90).sum()
        return tv / est_mins * 90 if est_mins > 0 else np.nan

    player_agg = (
        summary_df.groupby('player')
        .agg(total_vaep=('total_vaep', 'sum'), n_actions=('n_actions', 'sum'))
        .reset_index()
    )
    v90_series = summary_df.groupby('player').apply(weighted_v90).rename('vaep_per90_combined')
    player_agg = player_agg.join(v90_series, on='player')

    vaep_top = (
        player_agg[player_agg['n_actions'] >= 500]
        .sort_values('vaep_per90_combined', ascending=False)
        .head(10)[['player', 'total_vaep', 'n_actions', 'vaep_per90_combined']]
    )
    print(vaep_top.to_string(index=False))

    print("\nDone.")


if __name__ == '__main__':
    main()
