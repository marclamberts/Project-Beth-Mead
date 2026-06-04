import json
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
from mplsoccer import Pitch, VerticalPitch

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="B. Mead | Analytics",
    layout="wide",
    page_icon="⚽",
)

BG          = "#1a1a1a"
DARK        = "#111111"
LINE_COLOR  = "white"
ACCENT      = "#f5a623"

# ── helpers ───────────────────────────────────────────────────────────────────

def get_qualifier(event, qid):
    for q in event.get("qualifier", []):
        if q["qualifierId"] == qid:
            return q.get("value")
    return None

def has_qualifier(event, qid):
    return any(q["qualifierId"] == qid for q in event.get("qualifier", []))

def normalize(x, y, flip):
    return (100 - x, 100 - y) if flip else (x, y)

# ── data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data…")
def load_data():
    base  = os.path.dirname(__file__)
    files = sorted(glob.glob(os.path.join(base, "**/*.json"), recursive=True))

    passes   = []
    shots    = []
    touches  = []
    def_acts = []

    for fpath in files:
        fname  = os.path.basename(fpath)
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

        teams_part  = fname.replace(".json", "").split("_", 1)[1] if "_" in fname else fname
        home_team   = teams_part.split(" - ")[0].strip()
        is_home     = "Arsenal" in home_team

        for e in events:
            if e.get("playerName") != "B. Mead":
                continue

            tid     = e.get("typeId")
            x       = e.get("x")
            y       = e.get("y")
            outcome = e.get("outcome", 0)
            period  = e.get("periodId", 1)

            if x is None or y is None:
                continue

            flip       = (not is_home and period == 1) or (is_home and period == 2)
            nx, ny     = normalize(x, y, flip)

            base_rec = dict(x=nx, y=ny, season=season, outcome=outcome, period=period)

            # ── passes ────────────────────────────────────────────────────────
            if tid == 1:
                ex = get_qualifier(e, 140)
                ey = get_qualifier(e, 141)
                if ex is None or ey is None:
                    continue
                end_x, end_y = normalize(float(ex), float(ey), flip)
                is_key  = has_qualifier(e, 210)
                is_prog = outcome == 1 and end_x - nx >= 10 and end_x > 50
                passes.append({**base_rec,
                    "end_x": end_x, "end_y": end_y,
                    "key_pass": is_key, "progressive": is_prog})

            # ── shots (miss=13, post=14, saved=15, goal=16) ───────────────────
            elif tid in (13, 14, 15, 16):
                is_goal   = tid == 16
                on_target = tid in (15, 16)
                shots.append({**base_rec,
                    "is_goal": is_goal, "on_target": on_target, "type_id": tid})

            # ── all touches / heat ────────────────────────────────────────────
            elif tid in (1, 2, 3, 4, 7, 8, 12, 13, 14, 15, 16, 43, 44, 45, 49, 50, 61, 74):
                touches.append(base_rec)

            # ── defensive actions ─────────────────────────────────────────────
            if tid in (7, 8, 12, 74):
                label_map = {7: "Tackle", 8: "Interception", 12: "Clearance", 74: "Block"}
                def_acts.append({**base_rec, "action": label_map[tid]})

    return passes, shots, touches, def_acts


passes, shots, touches, def_acts = load_data()
seasons_available = sorted(set(p["season"] for p in passes))

# ── sidebar filters ───────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
        width=60,
    )
    st.title("B. Mead")
    st.caption("WSL Analytics Dashboard")
    st.markdown("---")

    st.subheader("🗓 Season")
    selected_seasons = st.multiselect(
        "Select season(s)",
        options=seasons_available,
        default=seasons_available,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.subheader("🎯 Pass Filters")
    show_succ  = st.checkbox("Successful",            value=True)
    show_unsucc= st.checkbox("Unsuccessful",          value=True)
    show_prog  = st.checkbox("Progressive",           value=True)
    show_key   = st.checkbox("Shot Assist / Key Pass",value=True)

    st.markdown("---")
    st.subheader("🥅 Shot Filters")
    show_goals    = st.checkbox("Goals",         value=True)
    show_on_tgt   = st.checkbox("On Target",     value=True)
    show_off_tgt  = st.checkbox("Off Target",    value=True)

    st.markdown("---")
    st.subheader("🛡 Defensive Filters")
    all_def_types = ["Tackle", "Interception", "Clearance", "Block"]
    sel_def = st.multiselect("Action type(s)", all_def_types, default=all_def_types,
                             label_visibility="collapsed")

    st.markdown("---")
    st.caption("Data: Opta / WSL  |  Viz: mplsoccer")

# ── filter helpers ────────────────────────────────────────────────────────────

def f_passes(data):
    out = []
    for p in data:
        if p["season"] not in selected_seasons:
            continue
        if p["key_pass"]    and not show_key:   continue
        if p["progressive"] and not show_prog:  continue
        if p["outcome"] == 1 and not p["key_pass"] and not p["progressive"] and not show_succ:  continue
        if p["outcome"] == 0 and not show_unsucc: continue
        out.append(p)
    return out

def f_shots(data):
    out = []
    for s in data:
        if s["season"] not in selected_seasons: continue
        if s["is_goal"]                   and not show_goals:   continue
        if s["on_target"] and not s["is_goal"] and not show_on_tgt:  continue
        if not s["on_target"]             and not show_off_tgt: continue
        out.append(s)
    return out

def f_touches(data):
    return [t for t in data if t["season"] in selected_seasons]

def f_def(data):
    return [d for d in data if d["season"] in selected_seasons and d["action"] in sel_def]

# ── colour maps ───────────────────────────────────────────────────────────────

PASS_COLOR = {
    "key":        "#5bc8f5",
    "progressive":"#4ecb71",
    "successful": "#f5a623",
    "unsuccessful":"#e63946",
}

DEF_COLOR = {
    "Tackle":       "#ff6b6b",
    "Interception": "#ffd93d",
    "Clearance":    "#6bcb77",
    "Block":        "#4d96ff",
}

def pass_color(p):
    if p["key_pass"]:    return PASS_COLOR["key"]
    if p["progressive"]: return PASS_COLOR["progressive"]
    if p["outcome"] == 1: return PASS_COLOR["successful"]
    return PASS_COLOR["unsuccessful"]

def pass_sort(p):
    if p["key_pass"]:    return 3
    if p["progressive"]: return 2
    if p["outcome"] == 1: return 1
    return 0

# ── metric bar ────────────────────────────────────────────────────────────────

def metric_row(fp, fs, ft, fd):
    total_p = len(fp)
    succ_p  = sum(1 for p in fp if p["outcome"] == 1)
    prog_p  = sum(1 for p in fp if p["progressive"])
    key_p   = sum(1 for p in fp if p["key_pass"])
    goals   = sum(1 for s in fs if s["is_goal"])
    shots_t = len(fs)
    tackles = sum(1 for d in fd if d["action"] == "Tackle")
    intercepts = sum(1 for d in fd if d["action"] == "Interception")

    cols = st.columns(8)
    cols[0].metric("Passes",        f"{total_p:,}")
    cols[1].metric("Pass Acc %",    f"{round(succ_p/total_p*100,1) if total_p else 0}%")
    cols[2].metric("Progressive",   f"{prog_p:,}")
    cols[3].metric("Shot Assists",  f"{key_p:,}")
    cols[4].metric("Shots",         f"{shots_t:,}")
    cols[5].metric("Goals",         f"{goals:,}")
    cols[6].metric("Tackles",       f"{tackles:,}")
    cols[7].metric("Interceptions", f"{intercepts:,}")

# ── pitch factory ─────────────────────────────────────────────────────────────

def make_pitch(vertical=False, figsize=(16, 10)):
    cls = VerticalPitch if vertical else Pitch
    pitch = cls(
        pitch_type="opta",
        pitch_color=BG,
        line_color=LINE_COLOR,
        linewidth=1.2,
        goal_type="box",
    )
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(BG)
    return pitch, fig, ax

def add_title(fig, title, subtitle=""):
    fig.text(0.5, 0.97, title, ha="center", va="top",
             fontsize=15, fontweight="bold", color="white")
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", va="top",
                 fontsize=9, color="#aaaaaa")

def season_label():
    if len(selected_seasons) == len(seasons_available):
        return "All Seasons"
    return ", ".join(s.replace("WSL ", "") for s in selected_seasons)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<h2 style='color:white; margin-bottom:0'>⚽ Beth Mead — WSL Analytics</h2>",
    unsafe_allow_html=True,
)
st.markdown(f"<p style='color:#aaaaaa; margin-top:2px'>{season_label()}</p>",
            unsafe_allow_html=True)

tab_pass, tab_shot, tab_heat, tab_def, tab_perc = st.tabs([
    "🎯  Pass Map",
    "🥅  Shot Map",
    "🔥  Heat Map",
    "🛡  Defensive Actions",
    "📊  Percentile Chart",
])

# ── TAB 1: PASS MAP ────────────────────────────────────────────────────────────

with tab_pass:
    fp = f_passes(passes)
    fs = f_shots(shots)
    ft = f_touches(touches)
    fd = f_def(def_acts)
    metric_row(fp, fs, ft, fd)
    st.markdown("---")

    pitch, fig, ax = make_pitch(figsize=(16, 10))

    for p in sorted(fp, key=pass_sort):
        alpha = 0.30 if p["outcome"] == 0 else 0.50
        if p["key_pass"] or p["progressive"]: alpha = 0.80
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"],
                     ax=ax, color=pass_color(p), alpha=alpha,
                     width=1.2, headwidth=4, headlength=4)

    legend_items = []
    if show_succ:   legend_items.append(mpatches.Patch(color=PASS_COLOR["successful"],  label="Successful"))
    if show_unsucc: legend_items.append(mpatches.Patch(color=PASS_COLOR["unsuccessful"],label="Unsuccessful"))
    if show_prog:   legend_items.append(mpatches.Patch(color=PASS_COLOR["progressive"], label="Progressive"))
    if show_key:    legend_items.append(mpatches.Patch(color=PASS_COLOR["key"],         label="Shot Assist"))
    if legend_items:
        ax.legend(handles=legend_items, loc="lower left", fontsize=9,
                  framealpha=0.25, facecolor=BG, edgecolor="white",
                  labelcolor="white", handlelength=1.5, borderpad=0.7)

    add_title(fig, "Pass Map — B. Mead",
              f"{season_label()}  ·  {len(fp):,} passes shown")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── TAB 2: SHOT MAP ────────────────────────────────────────────────────────────

with tab_shot:
    fp = f_passes(passes)
    fs = f_shots(shots)
    ft = f_touches(touches)
    fd = f_def(def_acts)
    metric_row(fp, fs, ft, fd)
    st.markdown("---")

    pitch, fig, ax = make_pitch(vertical=True, figsize=(9, 13))

    for s in fs:
        if s["is_goal"]:
            color, marker, size, alpha, zorder = "#f5a623", "*", 220, 0.95, 4
        elif s["on_target"]:
            color, marker, size, alpha, zorder = "#5bc8f5", "o", 100, 0.80, 3
        else:
            color, marker, size, alpha, zorder = "#e63946", "X", 80, 0.55, 2

        ax.scatter(s["x"], s["y"], c=color, marker=marker, s=size,
                   alpha=alpha, zorder=zorder, edgecolors="white", linewidths=0.4)

    legend_items = [
        mpatches.Patch(color="#f5a623", label=f"Goal ({sum(1 for s in fs if s['is_goal'])})"),
        mpatches.Patch(color="#5bc8f5", label=f"On Target ({sum(1 for s in fs if s['on_target'] and not s['is_goal'])})"),
        mpatches.Patch(color="#e63946", label=f"Off Target ({sum(1 for s in fs if not s['on_target'])})"),
    ]
    ax.legend(handles=legend_items, loc="lower center", fontsize=9,
              framealpha=0.25, facecolor=BG, edgecolor="white",
              labelcolor="white", handlelength=1.5, borderpad=0.7)

    add_title(fig, "Shot Map — B. Mead",
              f"{season_label()}  ·  {len(fs)} shots  ·  {sum(1 for s in fs if s['is_goal'])} goals")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── TAB 3: HEAT MAP ────────────────────────────────────────────────────────────

with tab_heat:
    fp = f_passes(passes)
    fs = f_shots(shots)
    ft = f_touches(touches)
    fd = f_def(def_acts)
    metric_row(fp, fs, ft, fd)
    st.markdown("---")

    pitch, fig, ax = make_pitch(figsize=(16, 10))

    xs = [t["x"] for t in ft]
    ys = [t["y"] for t in ft]

    if xs:
        # Custom dark-to-orange colormap
        cmap = LinearSegmentedColormap.from_list(
            "mead_heat", [BG, "#3d1c00", "#f5a623", "#ffffff"], N=256
        )
        pitch.kdeplot(xs, ys, ax=ax, cmap=cmap, fill=True, levels=100,
                      alpha=0.85, bw_adjust=0.7, zorder=1)

    add_title(fig, "Touch Heat Map — B. Mead",
              f"{season_label()}  ·  {len(ft):,} touches")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── TAB 4: DEFENSIVE ACTIONS ───────────────────────────────────────────────────

with tab_def:
    fp = f_passes(passes)
    fs = f_shots(shots)
    ft = f_touches(touches)
    fd = f_def(def_acts)
    metric_row(fp, fs, ft, fd)
    st.markdown("---")

    pitch, fig, ax = make_pitch(figsize=(16, 10))

    MARKER = {"Tackle": "s", "Interception": "D", "Clearance": "o", "Block": "^"}

    for d in fd:
        ax.scatter(d["x"], d["y"],
                   c=DEF_COLOR[d["action"]],
                   marker=MARKER[d["action"]],
                   s=90, alpha=0.75, zorder=3,
                   edgecolors="white", linewidths=0.4)

    legend_items = [
        mpatches.Patch(color=DEF_COLOR[a], label=f"{a} ({sum(1 for d in fd if d['action']==a)})")
        for a in sel_def
    ]
    if legend_items:
        ax.legend(handles=legend_items, loc="lower left", fontsize=9,
                  framealpha=0.25, facecolor=BG, edgecolor="white",
                  labelcolor="white", handlelength=1.5, borderpad=0.7)

    add_title(fig, "Defensive Actions — B. Mead",
              f"{season_label()}  ·  {len(fd)} actions")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── TAB 5: PERCENTILE CHART ────────────────────────────────────────────────────

with tab_perc:
    st.markdown("Per-season stats shown as percentile bars "
                "(100 = Mead's personal best season for that metric).")
    st.markdown("---")

    # Build per-season stats
    season_stats = {}
    for s in seasons_available:
        sp = [p for p in passes  if p["season"] == s]
        ss = [sh for sh in shots  if sh["season"] == s]
        sd = [d  for d  in def_acts if d["season"] == s]

        total_p = len(sp)
        succ_p  = sum(1 for p in sp if p["outcome"] == 1)
        season_stats[s] = {
            "Passes":        total_p,
            "Pass Acc %":    round(succ_p / total_p * 100, 1) if total_p else 0,
            "Progressive":   sum(1 for p in sp if p["progressive"]),
            "Shot Assists":  sum(1 for p in sp if p["key_pass"]),
            "Shots":         len(ss),
            "Goals":         sum(1 for sh in ss if sh["is_goal"]),
            "Shot Acc %":    round(sum(1 for sh in ss if sh["on_target"]) / len(ss) * 100, 1) if ss else 0,
            "Tackles":       sum(1 for d in sd if d["action"] == "Tackle"),
            "Interceptions": sum(1 for d in sd if d["action"] == "Interception"),
        }

    metrics   = list(next(iter(season_stats.values())).keys())
    max_vals  = {m: max(season_stats[s][m] for s in season_stats) or 1 for m in metrics}

    # One chart per selected season, side by side (max 4 per row)
    display_seasons = [s for s in seasons_available if s in selected_seasons]
    cols_per_row    = min(len(display_seasons), 3)
    rows            = [display_seasons[i:i+cols_per_row]
                       for i in range(0, len(display_seasons), cols_per_row)]

    BAR_COLORS = [
        "#f5a623","#5bc8f5","#4ecb71","#e63946",
        "#c084fc","#fb923c","#a3e635","#38bdf8","#f472b6",
    ]

    for row_seasons in rows:
        cols = st.columns(len(row_seasons))
        for col, s in zip(cols, row_seasons):
            with col:
                fig, ax = plt.subplots(figsize=(5, 4.5))
                fig.patch.set_facecolor(BG)
                ax.set_facecolor(BG)

                vals       = [season_stats[s][m] for m in metrics]
                pcts       = [v / max_vals[m] * 100 for v, m in zip(vals, metrics)]
                y_pos      = np.arange(len(metrics))
                bar_colors = BAR_COLORS[:len(metrics)]

                bars = ax.barh(y_pos, pcts, color=bar_colors, height=0.6,
                               alpha=0.85, edgecolor="none")

                # value labels
                for bar, val, pct in zip(bars, vals, pcts):
                    ax.text(min(pct + 2, 97), bar.get_y() + bar.get_height() / 2,
                            f"{val:g}", va="center", ha="left",
                            color="white", fontsize=7.5, fontweight="bold")

                ax.set_yticks(y_pos)
                ax.set_yticklabels(metrics, color="white", fontsize=8)
                ax.set_xlim(0, 110)
                ax.set_xlabel("% of personal best", color="#aaaaaa", fontsize=7)
                ax.tick_params(colors="white", labelsize=7)
                ax.xaxis.label.set_color("#aaaaaa")
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.axvline(100, color="#444444", linewidth=0.8, linestyle="--")
                ax.set_title(s.replace("WSL ", ""), color=ACCENT,
                             fontsize=10, fontweight="bold", pad=8)

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
