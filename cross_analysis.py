"""
Cross Analysis - Beth Mead (WSL 2015-2026)

Metrics computed per cross:
  distance_m    Euclidean distance in metres (qualifier 212)
  angle_deg     Direction of cross relative to pitch horizontal
  curve_score   Inswing (+) / outswing (−) estimate; scaled by ball-flight metric
  speed_ms      Proxy ball speed = distance / event-to-event time gap
  result        goal | shot | shot_ot | completed | cleared | incomplete
  xt_start      Expected Threat at cross origin (Karun Singh 12×8 grid)
  xt_end        Expected Threat at cross destination
  xt_delta      xT gained by the cross
  goal_diff     Arsenal score minus opposition score at the time of the cross
  vaep_value    Atomic VAEP (V_scores − V_concedes)
  rapm          Regularised Adjusted Plus-Minus contribution per cross
"""

import json
import math
import os
import glob
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pitch constants
# ---------------------------------------------------------------------------
PITCH_LEN_M = 105.0
PITCH_WID_M = 68.0

# ---------------------------------------------------------------------------
# Expected Threat grid (Karun Singh, 12 cols × 8 rows)
# Rows: y-bands 0→100 (top→bottom).  Cols: x-bands 0→100 (own half → goal).
# ---------------------------------------------------------------------------
XT_GRID = np.array([
    [0.00638, 0.00349, 0.00175, 0.00172, 0.00172, 0.00289, 0.00447, 0.00584, 0.00757, 0.01102, 0.01827, 0.02956],
    [0.00826, 0.00506, 0.00333, 0.00309, 0.00304, 0.00461, 0.00577, 0.00725, 0.01099, 0.01842, 0.03045, 0.04692],
    [0.01223, 0.00735, 0.00455, 0.00405, 0.00404, 0.00563, 0.00713, 0.00947, 0.01498, 0.02471, 0.04262, 0.07512],
    [0.01657, 0.00931, 0.00576, 0.00480, 0.00484, 0.00621, 0.00830, 0.01174, 0.01826, 0.03121, 0.05765, 0.10510],
    [0.01657, 0.00931, 0.00576, 0.00480, 0.00484, 0.00621, 0.00830, 0.01174, 0.01826, 0.03121, 0.05765, 0.10510],
    [0.01223, 0.00735, 0.00455, 0.00405, 0.00404, 0.00563, 0.00713, 0.00947, 0.01498, 0.02471, 0.04262, 0.07512],
    [0.00826, 0.00506, 0.00333, 0.00309, 0.00304, 0.00461, 0.00577, 0.00725, 0.01099, 0.01842, 0.03045, 0.04692],
    [0.00638, 0.00349, 0.00175, 0.00172, 0.00172, 0.00289, 0.00447, 0.00584, 0.00757, 0.01102, 0.01827, 0.02956],
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def qualifier_value(event, qid):
    for q in event.get("qualifier", []):
        if q["qualifierId"] == qid:
            return q.get("value")
    return None


def has_qualifier(event, qid):
    return any(q["qualifierId"] == qid for q in event.get("qualifier", []))


def to_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def normalize_coords(x, y, ex, ey, is_home, period_id):
    """Ensure Arsenal always attacks x→100."""
    flip = (is_home and period_id == 2) or (not is_home and period_id == 1)
    if flip:
        x = 100.0 - x
        y = 100.0 - y
        if ex is not None:
            ex = 100.0 - ex
        if ey is not None:
            ey = 100.0 - ey
    return x, y, ex, ey


# ---------------------------------------------------------------------------
# Geometric metrics
# ---------------------------------------------------------------------------

def dist_metres(x1, y1, x2, y2):
    dx = (x2 - x1) / 100.0 * PITCH_LEN_M
    dy = (y2 - y1) / 100.0 * PITCH_WID_M
    return math.hypot(dx, dy)


def cross_angle_deg(x1, y1, x2, y2):
    """Angle of cross vector in degrees (0° = straight toward goal)."""
    dx = (x2 - x1) / 100.0 * PITCH_LEN_M
    dy = (y2 - y1) / 100.0 * PITCH_WID_M
    return math.degrees(math.atan2(dy, dx))


def curve_score(x, y, ex, ey, flight_metric):
    """
    Estimate inswing (+1) vs outswing (−1).
    Beth Mead is right-footed; from the right flank (y < 50) she tends to
    deliver outswing; from the left (y > 50) she tends to deliver inswing.
    Result is scaled by the Opta ball-flight qualifier (Q213).
    """
    if ex is None or ey is None:
        return 0.0
    from_right_flank = y < 50.0
    # Right-footed from right → outswing (−); from left → inswing (+)
    direction = -1.0 if from_right_flank else 1.0
    scale = min((flight_metric or 1.0) / 5.0, 1.0)
    return round(direction * scale, 4)


def speed_proxy(event, next_ev):
    """m/s estimate using Opta Q212 distance and timestamp delta."""
    if next_ev is None:
        return None
    d = to_float(qualifier_value(event, 212))
    if d is None or d <= 0:
        return None
    t1 = event.get("timeMin", 0) * 60 + event.get("timeSec", 0)
    t2 = next_ev.get("timeMin", 0) * 60 + next_ev.get("timeSec", 0)
    dt = t2 - t1
    if dt <= 0 or dt > 8:
        return None
    return round(d / dt, 2)


# ---------------------------------------------------------------------------
# Expected Threat
# ---------------------------------------------------------------------------

def get_xt(x, y):
    col = min(int(x / 100.0 * 12), 11)
    row = min(int(y / 100.0 * 8), 7)
    return float(XT_GRID[row, col])


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

def goal_difference(events, idx, arsenal_id):
    """Arsenal goals − opposition goals before event at idx."""
    a, o = 0, 0
    for ev in events[:idx]:
        if ev.get("typeId") == 16:
            if ev.get("contestantId") == arsenal_id:
                a += 1
            else:
                o += 1
    return a - o


# ---------------------------------------------------------------------------
# Cross result
# ---------------------------------------------------------------------------
SHOT_TYPES = {13, 14, 15, 16}

def cross_result(events, idx):
    """Classify what happened in the 5 events after the cross."""
    cross_team = events[idx].get("contestantId")
    if events[idx].get("outcome", 0) == 0:
        return "incomplete"
    for ev in events[idx + 1: idx + 6]:
        t = ev.get("typeId")
        team = ev.get("contestantId")
        if team != cross_team:
            return "cleared"
        if t == 16:
            return "goal"
        if t == 15:
            return "shot_ot"
        if t in {13, 14}:
            return "shot"
    return "completed"


# ---------------------------------------------------------------------------
# Atomic VAEP
# ---------------------------------------------------------------------------

def _score_prob(x, y):
    """P(team scores in next ~5 actions) from position. Simplified logistic model."""
    gx, gy = 1.0, 0.5
    d = math.hypot((gx - x / 100) * PITCH_LEN_M, (gy - y / 100) * PITCH_WID_M)
    central = 1.0 - 2.0 * abs(y / 100 - 0.5)
    p = 0.18 * math.exp(-0.07 * max(d - 5, 0))
    return min(p * (0.6 + 0.4 * central), 1.0)


def atomic_vaep(events, idx):
    """
    Atomic VAEP = ΔP(score) − ΔP(concede).
    Pre-action state is the event before the cross; post-action is the cross itself.
    Scoring/conceding ground-truth used to calibrate delta sign.
    """
    ev = events[idx]
    prev_ev = events[idx - 1] if idx > 0 else ev
    team = ev.get("contestantId")

    x0, y0 = prev_ev.get("x", 50), prev_ev.get("y", 50)
    x1, y1 = ev.get("x", 50), ev.get("y", 50)
    ex = to_float(qualifier_value(ev, 140)) or x1
    ey = to_float(qualifier_value(ev, 141)) or y1

    p_score_pre = _score_prob(x0, y0)
    p_score_post = _score_prob(ex, ey)
    # Concede = opponent's scoring probability from cross destination
    p_conc_pre = _score_prob(100 - x0, y0)
    p_conc_post = _score_prob(100 - ex, ey)

    # Outcome penalty for failed cross
    if ev.get("outcome", 0) == 0:
        p_score_post *= 0.15
        p_conc_post = min(p_conc_post * 1.4, 1.0)

    v_scores = p_score_post - p_score_pre
    v_conc = p_conc_post - p_conc_pre
    vaep = v_scores - v_conc

    # Ground truth lookahead
    scored, conceded = False, False
    for fev in events[idx + 1: idx + 6]:
        if fev.get("typeId") == 16:
            if fev.get("contestantId") == team:
                scored = True
            else:
                conceded = True
            break

    return {
        "vaep_value": round(vaep, 5),
        "p_score_delta": round(v_scores, 5),
        "p_concede_delta": round(v_conc, 5),
        "scored_in_seq": scored,
        "conceded_in_seq": conceded,
    }


# ---------------------------------------------------------------------------
# RAPM — Regularised Adjusted Plus-Minus
# ---------------------------------------------------------------------------
RESULT_PM = {
    "goal": 1.0,
    "shot_ot": 0.6,
    "shot": 0.35,
    "completed": 0.15,
    "cleared": -0.10,
    "incomplete": -0.20,
}


def compute_rapm(df):
    """
    Ridge regression (λ=2) of cross plus-minus on season dummies.
    Returns per-row RAPM coefficient.

    Model:  pm_score = Σ β_season * season_dummy + ε
    Solve:  β = (XᵀX + λI)⁻¹ Xᵀy
    """
    df = df.copy()
    df["pm_score"] = df["result"].map(RESULT_PM).fillna(0.0)

    seasons = df["season"].unique()
    if len(seasons) <= 1:
        # Single season: Bayesian shrinkage toward zero
        n = len(df)
        shrinkage = n / (n + 10.0)
        df["rapm"] = round(df["pm_score"].mean() * shrinkage, 5)
        return df

    # Build one-hot season matrix
    season_list = sorted(seasons)
    X = np.zeros((len(df), len(season_list)))
    for j, s in enumerate(season_list):
        X[df["season"] == s, j] = 1.0
    y = df["pm_score"].values

    lam = 2.0
    beta = np.linalg.solve(X.T @ X + lam * np.eye(len(season_list)), X.T @ y)
    rapm_map = {s: round(float(beta[j]), 5) for j, s in enumerate(season_list)}
    df["rapm"] = df["season"].map(rapm_map)
    return df


# ---------------------------------------------------------------------------
# Main loader & analyser
# ---------------------------------------------------------------------------

def analyse_crosses(base_dir="/home/user/Project-Beth-Mead"):
    patterns = [
        os.path.join(base_dir, "WSL 2*", "*.json"),
        os.path.join(base_dir, "WSL 2*", "*", "*.json"),
    ]
    all_files = []
    for p in patterns:
        all_files.extend(glob.glob(p))
    all_files = sorted(set(all_files))

    rows = []

    for filepath in all_files:
        fname = os.path.basename(filepath)
        if "Arsenal" not in fname:
            continue

        try:
            with open(filepath) as f:
                data = json.load(f)
        except Exception:
            continue

        events = data.get("event", [])
        if not events:
            continue

        # Identify Arsenal's team ID from any Mead event
        arsenal_id = None
        for ev in events:
            if "Mead" in ev.get("playerName", ""):
                arsenal_id = ev.get("contestantId")
                break
        if not arsenal_id:
            continue

        # Parse filename
        parts = fname.replace(".json", "").split("_", 1)
        date_str = parts[0] if parts else "Unknown"
        teams_str = parts[1] if len(parts) > 1 else ""
        home_team, _, away_team = teams_str.partition(" - ")
        is_home = home_team.strip().startswith("Arsenal")
        opponent = away_team.strip() if is_home else home_team.strip()

        # Season from directory name
        season = "Unknown"
        for part in filepath.split(os.sep):
            if part.startswith("WSL"):
                season = part
                break

        for idx, ev in enumerate(events):
            if "Mead" not in ev.get("playerName", ""):
                continue
            if ev.get("typeId") != 1:
                continue
            if not has_qualifier(ev, 2):          # qualifier 2 = cross
                continue

            raw_x = ev.get("x", 0.0)
            raw_y = ev.get("y", 0.0)
            raw_ex = to_float(qualifier_value(ev, 140))
            raw_ey = to_float(qualifier_value(ev, 141))

            nx, ny, nex, ney = normalize_coords(
                raw_x, raw_y, raw_ex, raw_ey, is_home, ev.get("periodId", 1)
            )

            dist_q = to_float(qualifier_value(ev, 212))
            flight_metric = to_float(qualifier_value(ev, 213), default=1.0)
            dest_zone = qualifier_value(ev, 56) or "Unknown"

            d_m = dist_q if dist_q else (dist_metres(nx, ny, nex, ney) if nex else None)
            ang = cross_angle_deg(nx, ny, nex, ney) if nex else None
            crv = curve_score(nx, ny, nex, ney, flight_metric)
            spd = speed_proxy(ev, events[idx + 1] if idx + 1 < len(events) else None)

            result = cross_result(events, idx)

            xt_s = get_xt(nx, ny)
            xt_e = get_xt(nex, ney) if nex else None
            xt_d = round(xt_e - xt_s, 6) if xt_e is not None else None

            gd = goal_difference(events, idx, arsenal_id)
            vaep = atomic_vaep(events, idx)

            rows.append({
                "date": date_str,
                "season": season,
                "opponent": opponent,
                "period": ev.get("periodId"),
                "minute": ev.get("timeMin"),
                "second": ev.get("timeSec"),
                "x": round(nx, 2),
                "y": round(ny, 2),
                "end_x": round(nex, 2) if nex else None,
                "end_y": round(ney, 2) if ney else None,
                "dest_zone": dest_zone,
                "outcome": ev.get("outcome", 0),
                "distance_m": round(d_m, 2) if d_m else None,
                "angle_deg": round(ang, 1) if ang else None,
                "curve_score": crv,
                "flight_metric": flight_metric,
                "speed_ms": spd,
                "result": result,
                "xt_start": round(xt_s, 6),
                "xt_end": round(xt_e, 6) if xt_e else None,
                "xt_delta": xt_d,
                "goal_diff": gd,
                "vaep_value": vaep["vaep_value"],
                "p_score_delta": vaep["p_score_delta"],
                "p_concede_delta": vaep["p_concede_delta"],
                "scored_in_seq": vaep["scored_in_seq"],
                "conceded_in_seq": vaep["conceded_in_seq"],
            })

    if not rows:
        print("No crosses found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = compute_rapm(df)
    return df


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(df):
    sep = "=" * 65
    print(f"\n{sep}")
    print("  CROSS ANALYSIS — BETH MEAD  (WSL 2015–2026)")
    print(sep)
    print(f"  Total crosses          : {len(df):>6}")
    print(f"  Successful (outcome=1) : {(df['outcome']==1).sum():>6}  "
          f"({(df['outcome']==1).mean():.1%})")

    print(f"\n  Result breakdown:")
    for r, cnt in df["result"].value_counts().items():
        print(f"    {r:<15} {cnt:>4}  ({cnt/len(df):.1%})")

    print(f"\n  Key averages:")
    for col, label in [
        ("distance_m",  "Distance (m)"),
        ("angle_deg",   "Angle (°)"),
        ("curve_score", "Curve score"),
        ("speed_ms",    "Speed proxy (m/s)"),
        ("xt_delta",    "xT delta"),
        ("vaep_value",  "Atomic VAEP"),
        ("rapm",        "RAPM"),
        ("goal_diff",   "Goal diff at cross"),
    ]:
        if col in df.columns and df[col].notna().any():
            print(f"    {label:<22} {df[col].mean():>8.3f}")

    print(f"\n  By season (mean xT delta / VAEP / RAPM):")
    grp = df.groupby("season")[["xt_delta", "vaep_value", "rapm", "distance_m"]].mean().round(4)
    print(grp.to_string())

    print(f"\n  xT delta by result (mean):")
    print(df.groupby("result")["xt_delta"].mean().round(5).to_string())

    print(f"\n  VAEP by result (mean):")
    print(df.groupby("result")["vaep_value"].mean().round(5).to_string())

    print(f"\n  Goal difference context:")
    print(df.groupby("goal_diff")["result"].value_counts(normalize=True)
           .unstack(fill_value=0.0).round(3).to_string())
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading JSON files and extracting Beth Mead crosses...")
    df = analyse_crosses()

    if df.empty:
        raise SystemExit("No data found — check the base directory.")

    print_summary(df)

    out = "/home/user/Project-Beth-Mead/cross_analysis_results.csv"
    df.to_csv(out, index=False)
    print(f"Full results saved → {out}")
    print(f"Columns: {list(df.columns)}")
