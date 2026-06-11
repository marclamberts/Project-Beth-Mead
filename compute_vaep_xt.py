"""
compute_vaep_xt.py
------------------
Calculates xT (Expected Threat) and VAEP (Valuing Actions by Estimating
Probabilities) for every player in the WSL dataset.

Outputs:
  vaep_xt_results.csv          - per-action values
  player_vaep_xt_summary.csv   - per-player per-season aggregates
"""

import json
import os
import glob
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
XT_COLS = 16   # pitch divided into 16 columns (x-axis)
XT_ROWS = 12   # pitch divided into 12 rows (y-axis)
XT_ITERS = 10

# typeId constants
TYPE_PASS      = 1
TYPE_DRIBBLE   = 3
TYPE_TACKLE    = 7
TYPE_INTERCEPT = 8
TYPE_CLEAR     = 12
TYPE_MISS      = 13
TYPE_POST      = 14
TYPE_SAVED     = 15
TYPE_GOAL      = 16

SHOT_TYPES = {TYPE_MISS, TYPE_POST, TYPE_SAVED, TYPE_GOAL}
MOVE_TYPES = {TYPE_PASS, TYPE_DRIBBLE}

# qualifier IDs
QUAL_END_X = 140
QUAL_END_Y = 141

# ---------------------------------------------------------------------------
# Helper: extract qualifier value
# ---------------------------------------------------------------------------
def get_qualifier(qualifiers, qid):
    for q in qualifiers:
        if q.get("qualifierId") == qid:
            val = q.get("value")
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
    return None


# ---------------------------------------------------------------------------
# Step 1: Load all JSON files
# ---------------------------------------------------------------------------
def load_all_events(base_dir):
    pattern = os.path.join(base_dir, "WSL*", "**", "*.json")
    files = glob.glob(pattern, recursive=True)
    # also try direct children
    pattern2 = os.path.join(base_dir, "WSL*", "*.json")
    files += glob.glob(pattern2)
    files = sorted(set(files))

    all_events = []
    print(f"Found {len(files)} JSON files. Loading...")

    for i, fpath in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{len(files)} files...")

        # derive season from directory name
        parts = fpath.replace(base_dir, "").lstrip(os.sep).split(os.sep)
        season = parts[0] if parts else "unknown"

        # derive match_id from filename
        fname = os.path.basename(fpath)
        match_id = fname.replace(".json", "")

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        events = data.get("event", [])
        for ev in events:
            tid = ev.get("typeId")
            # skip non-action events (e.g. period start typeId 32, 34, 43 …)
            # we keep passes, shots, dribbles, tackles, interceptions, clearances
            if tid not in (TYPE_PASS, TYPE_DRIBBLE, TYPE_TACKLE, TYPE_INTERCEPT,
                           TYPE_CLEAR, TYPE_MISS, TYPE_POST, TYPE_SAVED, TYPE_GOAL):
                continue

            qualifiers = ev.get("qualifier", []) or []
            end_x = get_qualifier(qualifiers, QUAL_END_X)
            end_y = get_qualifier(qualifiers, QUAL_END_Y)

            all_events.append({
                "season":       season,
                "match_id":     match_id,
                "player":       ev.get("playerName", "Unknown"),
                "playerId":     ev.get("playerId", ""),
                "team":         ev.get("contestantId", ""),
                "typeId":       tid,
                "periodId":     ev.get("periodId", 1),
                "timeMin":      ev.get("timeMin", 0),
                "timeSec":      ev.get("timeSec", 0),
                "outcome":      int(ev.get("outcome", 0)),
                "x":            float(ev.get("x", 0) or 0),
                "y":            float(ev.get("y", 0) or 0),
                "end_x":        end_x,
                "end_y":        end_y,
            })

    print(f"  Loaded {len(files)}/{len(files)} files. Total relevant events: {len(all_events)}")
    return pd.DataFrame(all_events)


# ---------------------------------------------------------------------------
# Step 2: Build xT model
# ---------------------------------------------------------------------------
def coord_to_cell(x, y):
    """Convert pitch percentage coordinates to (col, row) grid indices."""
    col = int(np.clip(x / 100.0 * XT_COLS, 0, XT_COLS - 1))
    row = int(np.clip(y / 100.0 * XT_ROWS, 0, XT_ROWS - 1))
    return col, row


def build_xt_model(df):
    print("\nBuilding xT model...")

    shot_count  = np.zeros((XT_COLS, XT_ROWS))
    pass_drib   = np.zeros((XT_COLS, XT_ROWS))
    goal_count  = np.zeros((XT_COLS, XT_ROWS))
    move_matrix = np.zeros((XT_COLS, XT_ROWS, XT_COLS, XT_ROWS))

    for row in df.itertuples(index=False):
        sx, sy = coord_to_cell(row.x, row.y)

        if row.typeId in SHOT_TYPES:
            shot_count[sx, sy] += 1
            if row.typeId == TYPE_GOAL:
                goal_count[sx, sy] += 1

        elif row.typeId in MOVE_TYPES:
            pass_drib[sx, sy] += 1
            if row.outcome == 1 and row.end_x is not None and row.end_y is not None:
                ex, ey = coord_to_cell(row.end_x, row.end_y)
                move_matrix[sx, sy, ex, ey] += 1

    # shot_rate: P(shot | ball in zone)
    total = shot_count + pass_drib
    shot_rate = np.where(total > 0, shot_count / total, 0.0)

    # goal_rate: P(goal | shot from zone)
    goal_rate = np.where(shot_count > 0, goal_count / shot_count, 0.0)

    # move probability: P(move to z2 | move from z1)
    move_totals = move_matrix.sum(axis=(2, 3), keepdims=True)
    move_prob = np.where(move_totals > 0, move_matrix / move_totals, 0.0)

    # iterative xT solve
    xT = np.zeros((XT_COLS, XT_ROWS))
    for iteration in range(XT_ITERS):
        new_xT = (shot_rate * goal_rate +
                  (1 - shot_rate) * (move_prob * xT[np.newaxis, np.newaxis, :, :]).sum(axis=(2, 3)))
        xT = new_xT

    print(f"  xT model built. Max xT value: {xT.max():.4f}")
    return xT


def compute_xt_values(df, xT):
    """Assign xT value to each pass/dribble action."""
    xt_values = np.zeros(len(df))

    mask = df["typeId"].isin(MOVE_TYPES) & (df["outcome"] == 1)
    idxs = df.index[mask]

    for idx in idxs:
        row = df.loc[idx]
        if pd.isna(row.end_x) or pd.isna(row.end_y):
            continue
        sx, sy = coord_to_cell(row.x, row.y)
        ex, ey = coord_to_cell(row.end_x, row.end_y)
        val = xT[ex, ey] - xT[sx, sy]
        xt_values[df.index.get_loc(idx)] = val  # only positive = threat increase

    return xt_values


# ---------------------------------------------------------------------------
# Step 3: VAEP Model
# ---------------------------------------------------------------------------
ACTION_TYPE_MAP = {
    TYPE_PASS:      0,
    TYPE_DRIBBLE:   1,
    TYPE_MISS:      2,
    TYPE_POST:      2,
    TYPE_SAVED:     2,
    TYPE_GOAL:      2,
    TYPE_TACKLE:    3,
    TYPE_INTERCEPT: 3,
    TYPE_CLEAR:     3,
}

def action_features(row_dict, prefix=""):
    """Return feature dict for one action."""
    tid = row_dict.get("typeId", 0)
    atype = ACTION_TYPE_MAP.get(tid, 3)
    one_hot = [1 if atype == i else 0 for i in range(4)]

    end_x = row_dict.get("end_x")
    end_y = row_dict.get("end_y")
    end_x_n = float(end_x) / 100.0 if end_x is not None else float(row_dict.get("x", 0)) / 100.0
    end_y_n = float(end_y) / 100.0 if end_y is not None else float(row_dict.get("y", 0)) / 100.0

    feats = {
        f"{prefix}start_x":   float(row_dict.get("x", 0)) / 100.0,
        f"{prefix}start_y":   float(row_dict.get("y", 0)) / 100.0,
        f"{prefix}end_x":     end_x_n,
        f"{prefix}end_y":     end_y_n,
        f"{prefix}type_pass": one_hot[0],
        f"{prefix}type_drib": one_hot[1],
        f"{prefix}type_shot": one_hot[2],
        f"{prefix}type_other":one_hot[3],
        f"{prefix}outcome":   float(row_dict.get("outcome", 0)),
        f"{prefix}time_norm": float(row_dict.get("timeMin", 0)) / 90.0,
    }
    return feats


NULL_ACTION = {
    "typeId": 0, "x": 0, "y": 0, "end_x": None, "end_y": None,
    "outcome": 0, "timeMin": 0
}


def build_vaep_dataset(df):
    """Build feature matrix and labels for VAEP training."""
    print("\nBuilding VAEP training dataset...")

    # sort events
    df = df.sort_values(["match_id", "team", "periodId", "timeMin", "timeSec"]).reset_index(drop=True)

    rows_feat = []
    rows_label = []

    groups = df.groupby(["match_id", "team"])
    total_groups = len(groups)

    for gi, ((match_id, team), grp) in enumerate(groups):
        grp = grp.reset_index(drop=True)
        n = len(grp)

        # find goal events for this match (by either team)
        match_events = df[df["match_id"] == match_id]

        # goals by this team and by opponent
        team_goals = match_events[
            (match_events["team"] == team) & (match_events["typeId"] == TYPE_GOAL)
        ][["timeMin", "timeSec"]].values

        opp_goals = match_events[
            (match_events["team"] != team) & (match_events["typeId"] == TYPE_GOAL)
        ][["timeMin", "timeSec"]].values

        for i in range(n):
            cur = grp.iloc[i].to_dict()
            c1  = grp.iloc[i-1].to_dict() if i >= 1 else NULL_ACTION
            c2  = grp.iloc[i-2].to_dict() if i >= 2 else NULL_ACTION

            feats = {}
            feats.update(action_features(cur))
            feats.update(action_features(c1, "c1_"))
            feats.update(action_features(c2, "c2_"))

            # look at next 10 actions in the group for labelling
            next_10 = grp.iloc[i+1 : i+11]

            scores_in_next_10 = 0
            concedes_in_next_10 = 0
            if len(next_10) > 0:
                # any goal by same team in next 10 actions
                scores_in_next_10 = int(
                    (next_10["typeId"] == TYPE_GOAL).any()
                )
                # any goal by opp: look at match-level events in that time window
                last_action = next_10.iloc[-1]
                last_min = last_action["timeMin"]
                last_sec = last_action["timeSec"]
                cur_min  = cur["timeMin"]
                cur_sec  = cur["timeSec"]
                cur_time = cur_min * 60 + cur_sec
                last_time= last_min * 60 + last_sec
                for g in opp_goals:
                    gt = int(g[0]) * 60 + int(g[1])
                    if cur_time < gt <= last_time + 10:
                        concedes_in_next_10 = 1
                        break

            rows_feat.append(feats)
            rows_label.append({"scores": scores_in_next_10, "concedes": concedes_in_next_10})

    feat_df  = pd.DataFrame(rows_feat).fillna(0)
    label_df = pd.DataFrame(rows_label)

    print(f"  VAEP dataset: {len(feat_df)} rows, {feat_df.shape[1]} features")
    return feat_df, label_df


def train_vaep_models(feat_df, label_df):
    print("\nTraining VAEP logistic regression models...")

    X = feat_df.values.astype(float)

    models = {}
    for target in ["scores", "concedes"]:
        y = label_df[target].values
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, C=0.1, solver="lbfgs"))
        ])
        pipe.fit(X, y)
        pred = pipe.predict_proba(X)[:, 1]
        print(f"  {target}: mean prob={pred.mean():.4f}, positive rate={y.mean():.4f}")
        models[target] = pipe

    return models


def compute_vaep_values(feat_df, models):
    """Compute VAEP values: delta_score - delta_concede."""
    X = feat_df.values.astype(float)
    n = len(X)

    p_score   = models["scores"].predict_proba(X)[:, 1]
    p_concede = models["concedes"].predict_proba(X)[:, 1]

    # "before" state = previous action's probabilities (shift by 1, fill first with 0)
    p_score_before   = np.concatenate([[0.0], p_score[:-1]])
    p_concede_before = np.concatenate([[0.0], p_concede[:-1]])

    vaep = (p_score - p_score_before) - (p_concede - p_concede_before)
    return vaep


# ---------------------------------------------------------------------------
# Step 4: Derive action type label
# ---------------------------------------------------------------------------
TYPE_LABEL = {
    TYPE_PASS:      "pass",
    TYPE_DRIBBLE:   "dribble",
    TYPE_TACKLE:    "tackle",
    TYPE_INTERCEPT: "interception",
    TYPE_CLEAR:     "clearance",
    TYPE_MISS:      "shot_miss",
    TYPE_POST:      "shot_post",
    TYPE_SAVED:     "shot_saved",
    TYPE_GOAL:      "goal",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    base_dir = "/home/user/Project-Beth-Mead"

    # --- Load data ---
    df = load_all_events(base_dir)

    if df.empty:
        print("No events loaded. Exiting.")
        return

    df = df.reset_index(drop=True)

    # --- xT ---
    xT = build_xt_model(df)
    xt_values = compute_xt_values(df, xT)
    df["xt_value"] = xt_values

    # --- VAEP ---
    feat_df, label_df = build_vaep_dataset(df)
    models = train_vaep_models(feat_df, label_df)
    vaep_values = compute_vaep_values(feat_df, models)

    # vaep_values aligns with feat_df which was built in the same order
    # We need to re-align with df. Re-sort df the same way.
    df_sorted = df.sort_values(
        ["match_id", "team", "periodId", "timeMin", "timeSec"]
    ).reset_index(drop=False)  # keep original index in 'index' column

    df_sorted["vaep_value"] = vaep_values

    # map back to original df index
    df = df.copy()
    df["vaep_value"] = 0.0
    for _, row in df_sorted.iterrows():
        df.at[row["index"], "vaep_value"] = row["vaep_value"]

    # --- Build results CSV ---
    print("\nBuilding output CSVs...")
    df["action_type"] = df["typeId"].map(TYPE_LABEL).fillna("other")

    results = df[[
        "season", "match_id", "player", "team", "typeId", "action_type",
        "timeMin", "x", "y", "end_x", "end_y", "xt_value", "vaep_value"
    ]].copy()

    out_path = os.path.join(base_dir, "vaep_xt_results.csv")
    results.to_csv(out_path, index=False)
    print(f"  Saved {out_path} ({len(results)} rows)")

    # --- Player summary ---
    # Estimate total minutes: max timeMin - min timeMin per player per season
    player_groups = df.groupby(["season", "player", "team"])

    summary_rows = []
    for (season, player, team), grp in player_groups:
        if player in ("Unknown", "", None):
            continue
        n_actions   = len(grp)
        total_xt    = grp["xt_value"].sum()
        total_vaep  = grp["vaep_value"].sum()
        xt_per      = total_xt / n_actions if n_actions else 0.0
        vaep_per    = total_vaep / n_actions if n_actions else 0.0

        min_min = grp["timeMin"].min()
        max_min = grp["timeMin"].max()
        total_minutes = max(max_min - min_min, 1)
        vaep_per90 = total_vaep / total_minutes * 90.0

        summary_rows.append({
            "season":          season,
            "player":          player,
            "team":            team,
            "n_actions":       n_actions,
            "total_xt":        round(total_xt, 4),
            "xt_per_action":   round(xt_per, 5),
            "total_vaep":      round(total_vaep, 4),
            "vaep_per_action": round(vaep_per, 5),
            "vaep_per90":      round(vaep_per90, 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(base_dir, "player_vaep_xt_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved {summary_path} ({len(summary_df)} rows)")

    # --- Top 10 by total xT ---
    top_xt = (
        summary_df.groupby("player")[["total_xt", "n_actions"]]
        .sum()
        .assign(xt_per_action=lambda d: d["total_xt"] / d["n_actions"])
        .sort_values("total_xt", ascending=False)
        .head(10)
    )
    print("\n=== Top 10 Players by Total xT ===")
    print(top_xt.to_string())

    # --- Top 10 by VAEP per 90 (min 90 actions to filter noise) ---
    top_vaep = (
        summary_df[summary_df["n_actions"] >= 90]
        .groupby("player")
        .apply(lambda g: pd.Series({
            "total_vaep":  g["total_vaep"].sum(),
            "n_actions":   g["n_actions"].sum(),
            "vaep_per90":  g["total_vaep"].sum() / max(g["n_actions"].sum(), 1) * 90,
        }))
        .sort_values("vaep_per90", ascending=False)
        .head(10)
    )
    print("\n=== Top 10 Players by VAEP per 90 (min 90 actions) ===")
    print(top_vaep.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
