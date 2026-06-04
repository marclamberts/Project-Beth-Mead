import json
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from mplsoccer import Pitch

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="B. Mead – Pass Map",
    layout="wide",
    page_icon="⚽",
)

st.title("⚽ B. Mead — Pass Map")

BG = "#1a1a1a"
LINE_COLOR = "white"

# ── data loading ──────────────────────────────────────────────────────────────

def get_qualifier(event, qid):
    for q in event.get("qualifier", []):
        if q["qualifierId"] == qid:
            return q.get("value")
    return None

def has_qualifier(event, qid):
    return any(q["qualifierId"] == qid for q in event.get("qualifier", []))

def normalize(x, y, flip):
    return (100 - x, 100 - y) if flip else (x, y)

@st.cache_data(show_spinner="Loading pass data…")
def load_all_passes():
    base = os.path.dirname(__file__)
    files = sorted(glob.glob(os.path.join(base, "**/*.json"), recursive=True))
    all_passes = []

    for fpath in files:
        fname = os.path.basename(fpath)
        season = fpath.split(os.sep)[-2]

        with open(fpath) as fp:
            data = json.load(fp)

        if "event" not in data:
            continue

        events = data["event"]

        # Find Arsenal contestant ID
        arsenal_id = None
        for e in events:
            if e.get("playerName") == "B. Mead":
                arsenal_id = e["contestantId"]
                break
        if arsenal_id is None:
            continue

        # Home/away from filename: "DATE_HomeTeam - AwayTeam.json"
        teams_part = fname.replace(".json", "").split("_", 1)[1] if "_" in fname else fname
        home_team = teams_part.split(" - ")[0].strip()
        arsenal_is_home = "Arsenal" in home_team

        for e in events:
            if e.get("playerName") != "B. Mead":
                continue
            if e.get("typeId") != 1:
                continue

            x = e.get("x")
            y = e.get("y")
            end_x_str = get_qualifier(e, 140)
            end_y_str = get_qualifier(e, 141)

            if None in (x, y, end_x_str, end_y_str):
                continue

            end_x = float(end_x_str)
            end_y = float(end_y_str)
            period  = e.get("periodId", 1)
            outcome = e.get("outcome", 0)

            # Normalise: Arsenal always attacks left→right
            flip = (not arsenal_is_home and period == 1) or (arsenal_is_home and period == 2)
            x, y       = normalize(x, y, flip)
            end_x, end_y = normalize(end_x, end_y, flip)

            is_key_pass    = has_qualifier(e, 210)
            is_progressive = (outcome == 1 and end_x - x >= 10 and end_x > 50)

            all_passes.append({
                "x": x, "y": y,
                "end_x": end_x, "end_y": end_y,
                "outcome": outcome,
                "key_pass": is_key_pass,
                "progressive": is_progressive,
                "season": season,
            })

    return all_passes

all_passes = load_all_passes()
seasons_available = sorted(set(p["season"] for p in all_passes))

# ── filters (top of page, horizontal columns) ────────────────────────────────

st.markdown("### Filters")
fcol1, fcol2, fcol3, fcol4, fcol5, fcol6 = st.columns(6)

with fcol1:
    season_choice = st.selectbox("Season", options=["All Seasons"] + seasons_available)

with fcol2:
    show_successful = st.selectbox("Successful", options=["Show", "Hide"])

with fcol3:
    show_unsuccessful = st.selectbox("Unsuccessful", options=["Show", "Hide"])

with fcol4:
    show_progressive = st.selectbox("Progressive", options=["Show", "Hide"])

with fcol5:
    show_key_pass = st.selectbox("Shot Assist / Key Pass", options=["Show", "Hide"])

with fcol6:
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Data: Opta / WSL\nViz: mplsoccer")

# Build selected seasons + pass types from dropdowns
if season_choice == "All Seasons":
    selected_seasons = seasons_available
else:
    selected_seasons = [season_choice]

pass_types = []
if show_successful   == "Show": pass_types.append("Successful")
if show_unsuccessful == "Show": pass_types.append("Unsuccessful")
if show_progressive  == "Show": pass_types.append("Progressive")
if show_key_pass     == "Show": pass_types.append("Shot Assist / Key Pass")

st.markdown("---")

# ── filter passes ─────────────────────────────────────────────────────────────

def pass_type_label(p):
    if p["key_pass"]:
        return "Shot Assist / Key Pass"
    if p["progressive"]:
        return "Progressive"
    if p["outcome"] == 1:
        return "Successful"
    return "Unsuccessful"

COLOR_MAP = {
    "Successful":              "#f5a623",
    "Unsuccessful":            "#e63946",
    "Progressive":             "#4ecb71",
    "Shot Assist / Key Pass":  "#5bc8f5",
}

def pass_color(p):
    return COLOR_MAP[pass_type_label(p)]

filtered = [
    p for p in all_passes
    if p["season"] in selected_seasons
    and pass_type_label(p) in pass_types
]

# Sort so key passes render on top
SORT_ORDER = {"Unsuccessful": 0, "Successful": 1, "Progressive": 2, "Shot Assist / Key Pass": 3}
filtered_sorted = sorted(filtered, key=lambda p: SORT_ORDER[pass_type_label(p)])

# ── summary stats ─────────────────────────────────────────────────────────────

total   = len(filtered)
succ    = sum(1 for p in filtered if p["outcome"] == 1)
prog    = sum(1 for p in filtered if p["progressive"])
key     = sum(1 for p in filtered if p["key_pass"])
acc_pct = round(succ / total * 100, 1) if total else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Passes",     f"{total:,}")
col2.metric("Successful",       f"{succ:,}")
col3.metric("Accuracy",         f"{acc_pct}%")
col4.metric("Progressive",      f"{prog:,}")
col5.metric("Shot Assists",     f"{key:,}")

st.markdown("---")

# ── draw pitch ────────────────────────────────────────────────────────────────

pitch = Pitch(
    pitch_type="opta",
    pitch_color=BG,
    line_color=LINE_COLOR,
    linewidth=1.2,
    goal_type="box",
)

fig, ax = pitch.draw(figsize=(16, 10))
fig.patch.set_facecolor(BG)

for p in filtered_sorted:
    alpha = 0.30 if p["outcome"] == 0 else 0.50
    if p["key_pass"] or p["progressive"]:
        alpha = 0.80

    pitch.arrows(
        p["x"], p["y"],
        p["end_x"], p["end_y"],
        ax=ax,
        color=pass_color(p),
        alpha=alpha,
        width=1.2,
        headwidth=4,
        headlength=4,
    )

# legend
legend_items = [
    mpatches.Patch(color=COLOR_MAP[t], label=t) for t in pass_types
]
if legend_items:
    ax.legend(
        handles=legend_items,
        loc="lower left",
        fontsize=9,
        framealpha=0.25,
        facecolor=BG,
        edgecolor="white",
        labelcolor="white",
        handlelength=1.5,
        borderpad=0.7,
    )

# season label
season_label = ", ".join(s.replace("WSL ", "") for s in selected_seasons) if selected_seasons else "—"
fig.text(
    0.5, 0.98,
    "B. Mead — Pass Map  |  WSL",
    ha="center", va="top",
    fontsize=16, fontweight="bold",
    color="white",
)
fig.text(
    0.5, 0.95,
    f"Seasons: {season_label}  ·  {total:,} passes shown",
    ha="center", va="top",
    fontsize=9,
    color="#cccccc",
)

plt.tight_layout(rect=[0, 0, 1, 0.94])
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── per-season breakdown ──────────────────────────────────────────────────────

if len(selected_seasons) > 1:
    st.markdown("### Season Breakdown")
    rows = []
    for s in selected_seasons:
        sp = [p for p in all_passes if p["season"] == s]
        t  = len(sp)
        sc = sum(1 for p in sp if p["outcome"] == 1)
        pr = sum(1 for p in sp if p["progressive"])
        kp = sum(1 for p in sp if p["key_pass"])
        rows.append({
            "Season": s,
            "Total": t,
            "Successful": sc,
            "Accuracy %": round(sc / t * 100, 1) if t else 0,
            "Progressive": pr,
            "Shot Assists": kp,
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
