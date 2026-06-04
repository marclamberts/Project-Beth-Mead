import json
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
import streamlit as st
from mplsoccer import Pitch, VerticalPitch

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Beth Mead | WSL Analytics",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded",
)

# ── global CSS (StatsBomb-inspired) ───────────────────────────────────────────

st.markdown("""
<style>
/* ── base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: #e6edf3;
    font-family: 'Inter', 'Helvetica Neue', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* ── tabs ── */
[data-baseweb="tab-list"] {
    background-color: #161b22 !important;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #30363d;
}
[data-baseweb="tab"] {
    background-color: transparent !important;
    color: #8b949e !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    border: none !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background-color: #21262d !important;
    color: #58a6ff !important;
}
[data-baseweb="tab"]:hover {
    color: #e6edf3 !important;
    background-color: #21262d !important;
}

/* ── metrics ── */
[data-testid="stMetric"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 22px !important; font-weight: 700 !important; }

/* ── divider ── */
hr { border-color: #30363d !important; }

/* ── checkboxes / multiselect ── */
[data-testid="stCheckbox"] label { color: #c9d1d9 !important; font-size: 13px !important; }
[data-baseweb="select"] { background-color: #21262d !important; border-color: #30363d !important; }

/* ── section headers ── */
h3 { color: #58a6ff !important; font-size: 13px !important; text-transform: uppercase;
     letter-spacing: 0.08em; font-weight: 700 !important; margin-bottom: 6px !important; }

/* ── dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── colour palette ────────────────────────────────────────────────────────────

PITCH_BG    = "#0d1117"
PITCH_LINE  = "#c9d1d9"
FIG_BG      = "#0e1117"
ACCENT_BLUE = "#58a6ff"
ACCENT_ORG  = "#f78166"
ACCENT_GRN  = "#3fb950"
ACCENT_PRP  = "#bc8cff"
ACCENT_YLW  = "#e3b341"

PASS_COLOR = {
    "key":         "#58a6ff",
    "progressive": "#3fb950",
    "successful":  "#e3b341",
    "unsuccessful":"#f78166",
}

SHOT_STYLE = {
    16: ("#e3b341", "*", 300, 0.95, 4),  # goal
    15: ("#58a6ff", "o", 130, 0.85, 3),  # on target
    14: ("#bc8cff", "D", 110, 0.85, 3),  # post
    13: ("#f78166", "X",  90, 0.50, 2),  # off target
}

DEF_COLOR = {
    "Tackle":       "#f78166",
    "Interception": "#e3b341",
    "Clearance":    "#3fb950",
    "Block":        "#58a6ff",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def get_qualifier(event, qid):
    for q in event.get("qualifier", []):
        if q["qualifierId"] == qid:
            return q.get("value")
    return None

def has_qualifier(event, qid):
    return any(q["qualifierId"] == qid for q in event.get("qualifier", []))

# ── data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data…")
def load_data():
    base  = os.path.dirname(__file__)
    files = sorted(glob.glob(os.path.join(base, "**/*.json"), recursive=True))

    passes      = []
    shots       = []
    touches     = []
    def_acts    = []
    net_edges   = []   # pass-network: {passer, recipient, season, x, y}
    sonar_passes = []  # all Arsenal passes with angle+dist for sonar
    net_pos   = []   # per-event positions for avg position node placement

    for fpath in files:
        fname  = os.path.basename(fpath)
        season = fpath.split(os.sep)[-2]

        with open(fpath) as fp:
            data = json.load(fp)
        if "event" not in data:
            continue

        events = data["event"]
        if not any(e.get("playerName") == "B. Mead" for e in events):
            continue

        # Arsenal's contestant ID for this file
        arsenal_id = next(
            e["contestantId"] for e in events if e.get("playerName") == "B. Mead"
        )

        # ── sonar: ALL Arsenal passes (for per-player direction wheels) ────
        import math as _math
        for e in events:
            if (e.get("contestantId") != arsenal_id
                    or e.get("typeId") != 1):
                continue
            px, py = e.get("x"), e.get("y")
            ex_str = get_qualifier(e, 140)
            ey_str = get_qualifier(e, 141)
            if None in (px, py, ex_str, ey_str):
                continue
            dx = float(ex_str) - px
            dy = float(ey_str) - py
            angle = _math.degrees(_math.atan2(dy, dx))
            dist  = _math.hypot(dx, dy)
            sonar_passes.append({
                "player":  e.get("playerName", ""),
                "angle":   angle,
                "dist":    dist,
                "outcome": e.get("outcome", 0),
                "season":  season,
            })

        # ── pass network: all successful Arsenal passes ─────────────────────
        event_idx = {e["id"]: i for i, e in enumerate(events)}
        for i, e in enumerate(events):
            if (e.get("contestantId") != arsenal_id
                    or e.get("typeId") != 1
                    or e.get("outcome") != 1):
                continue
            passer = e.get("playerName")
            px, py = e.get("x"), e.get("y")
            if not passer or px is None:
                continue
            # recipient = first subsequent event by same team
            for j in range(i + 1, min(i + 6, len(events))):
                ne = events[j]
                if ne.get("contestantId") == arsenal_id and ne.get("playerName"):
                    recipient = ne["playerName"]
                    if recipient != passer:
                        net_edges.append({
                            "passer": passer, "recipient": recipient,
                            "season": season,
                        })
                    break
            # store position for average-position node
            net_pos.append({"player": passer, "x": px, "y": py, "season": season})

        # ── Beth Mead events ────────────────────────────────────────────────
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

            base_rec = dict(x=x, y=y, season=season, outcome=outcome, period=period)

            if tid == 1:
                ex = get_qualifier(e, 140)
                ey = get_qualifier(e, 141)
                if ex is None or ey is None:
                    continue
                end_x, end_y = float(ex), float(ey)
                is_key  = has_qualifier(e, 210)
                is_prog = outcome == 1 and end_x - x >= 10 and end_x > 50
                passes.append({**base_rec,
                    "end_x": end_x, "end_y": end_y,
                    "key_pass": is_key, "progressive": is_prog})
            elif tid in (13, 14, 15, 16):
                shots.append({**base_rec, "type_id": tid})

            if tid in (7, 8, 12, 74):
                label_map = {7: "Tackle", 8: "Interception", 12: "Clearance", 74: "Block"}
                def_acts.append({**base_rec, "action": label_map[tid]})

            touches.append(base_rec)

    return passes, shots, touches, def_acts, net_edges, net_pos, sonar_passes


passes, shots, touches, def_acts, net_edges, net_pos, sonar_passes = load_data()
seasons_available = sorted(set(p["season"] for p in passes))

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px 0'>
      <div style='font-size:20px;font-weight:800;color:#e6edf3;letter-spacing:-0.3px'>Beth Mead</div>
      <div style='font-size:12px;color:#8b949e;margin-top:2px'>Arsenal WFC · Forward</div>
      <div style='height:1px;background:#30363d;margin:14px 0'></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🗓 Season")
    selected_seasons = st.multiselect(
        "season", options=seasons_available, default=seasons_available,
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:1px;background:#30363d;margin:12px 0'></div>",
                unsafe_allow_html=True)
    st.markdown("### 🎯 Pass Filters")
    show_succ   = st.checkbox("Successful",             value=True)
    show_unsucc = st.checkbox("Unsuccessful",           value=True)
    show_prog   = st.checkbox("Progressive",            value=True)
    show_key    = st.checkbox("Shot Assist / Key Pass", value=True)

    st.markdown("<div style='height:1px;background:#30363d;margin:12px 0'></div>",
                unsafe_allow_html=True)
    st.markdown("### 🥅 Shot Filters")
    show_goals   = st.checkbox("Goal (16)",                value=True)
    show_on_tgt  = st.checkbox("On Target / Saved (15)",   value=True)
    show_post    = st.checkbox("Post (14)",                 value=True)
    show_off_tgt = st.checkbox("Off Target (13)",          value=True)

    st.markdown("<div style='height:1px;background:#30363d;margin:12px 0'></div>",
                unsafe_allow_html=True)
    st.markdown("### 🛡 Defensive Filters")
    all_def = ["Tackle", "Interception", "Clearance", "Block"]
    sel_def = st.multiselect("def", all_def, default=all_def,
                             label_visibility="collapsed")

    st.markdown("<div style='height:1px;background:#30363d;margin:16px 0 8px 0'></div>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;color:#484f58;line-height:1.6'>Data: Opta / WSL<br>Viz: mplsoccer</div>",
                unsafe_allow_html=True)

# ── filter helpers ────────────────────────────────────────────────────────────

def pass_type(p):
    if p["key_pass"]:     return "key"
    if p["progressive"]:  return "progressive"
    if p["outcome"] == 1: return "successful"
    return "unsuccessful"

def f_passes(data):
    out = []
    for p in data:
        if p["season"] not in selected_seasons: continue
        pt = pass_type(p)
        if pt == "key"          and not show_key:    continue
        if pt == "progressive"  and not show_prog:   continue
        if pt == "successful"   and not show_succ:   continue
        if pt == "unsuccessful" and not show_unsucc: continue
        out.append(p)
    return out

def f_shots(data):
    out = []
    for s in data:
        if s["season"] not in selected_seasons: continue
        tid = s["type_id"]
        if tid == 16 and not show_goals:   continue
        if tid == 15 and not show_on_tgt:  continue
        if tid == 14 and not show_post:    continue
        if tid == 13 and not show_off_tgt: continue
        out.append(s)
    return out

def f_touches(data):
    return [t for t in data if t["season"] in selected_seasons]

def f_def(data):
    return [d for d in data if d["season"] in selected_seasons
            and d["action"] in sel_def]

# ── season label ──────────────────────────────────────────────────────────────

def season_label():
    if not selected_seasons:
        return "No season selected"
    if len(selected_seasons) == len(seasons_available):
        return "All Seasons"
    return " · ".join(s.replace("WSL ", "") for s in selected_seasons)

# ── metric row ────────────────────────────────────────────────────────────────

def metric_row(fp, fs, fd):
    total_p = len(fp)
    succ_p  = sum(1 for p in fp if p["outcome"] == 1)
    goals   = sum(1 for s in fs if s["type_id"] == 16)
    shots_t = len(fs)

    cols = st.columns(8)
    metrics = [
        ("Passes",        f"{total_p:,}"),
        ("Pass Acc",      f"{round(succ_p/total_p*100,1) if total_p else 0}%"),
        ("Progressive",   f"{sum(1 for p in fp if p['progressive']):,}"),
        ("Shot Assists",  f"{sum(1 for p in fp if p['key_pass']):,}"),
        ("Shots",         f"{shots_t:,}"),
        ("Goals",         f"{goals:,}"),
        ("Tackles",       f"{sum(1 for d in fd if d['action']=='Tackle'):,}"),
        ("Interceptions", f"{sum(1 for d in fd if d['action']=='Interception'):,}"),
    ]
    for col, (label, val) in zip(cols, metrics):
        col.metric(label, val)

# ── pitch factory ─────────────────────────────────────────────────────────────

def make_pitch(vertical=False, half=False, figsize=(16, 10), line_zorder=1):
    cls = VerticalPitch if vertical else Pitch
    kwargs = dict(
        pitch_type="opta",
        pitch_color=PITCH_BG,
        line_color=PITCH_LINE,
        linewidth=1.0,
        goal_type="box",
        line_zorder=line_zorder,
    )
    if vertical:
        kwargs["half"] = half
    pitch = cls(**kwargs)
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(PITCH_BG)
    return pitch, fig, ax

def add_title(fig, title, subtitle=""):
    fig.text(0.5, 0.98, title, ha="center", va="top",
             fontsize=14, fontweight="800", color="#e6edf3",
             fontfamily="DejaVu Sans")
    if subtitle:
        fig.text(0.5, 0.945, subtitle, ha="center", va="top",
                 fontsize=9, color="#8b949e")

def legend(ax, items, loc="lower left"):
    handles = [mpatches.Patch(color=c, label=l) for c, l in items]
    ax.legend(handles=handles, loc=loc, fontsize=8.5,
              framealpha=0.35, facecolor="#161b22",
              edgecolor="#30363d", labelcolor="#c9d1d9",
              handlelength=1.2, borderpad=0.7, labelspacing=0.5)

# ── header ────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:4px'>
  <span style='font-size:26px;font-weight:800;color:#e6edf3;letter-spacing:-0.5px'>Beth Mead</span>
  <span style='font-size:13px;color:#8b949e;font-weight:500'>WSL Analytics Dashboard</span>
</div>
<div style='font-size:12px;color:#484f58;margin-bottom:16px'>{season_label()}</div>
""", unsafe_allow_html=True)

# ── tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🎯  Pass Map",
    "🥅  Shot Map",
    "🔥  Heat Map",
    "🗺  Territory",
    "🔗  Pass Network",
    "🧭  Pass Sonars",
    "🛡  Defensive",
    "📊  Percentile",
])
tab_pass, tab_shot, tab_heat, tab_terr, tab_net, tab_sonar, tab_def, tab_perc = tabs

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – PASS MAP
# ═══════════════════════════════════════════════════════════════════════════════

with tab_pass:
    fp = f_passes(passes)
    fs = f_shots(shots)
    fd = f_def(def_acts)
    metric_row(fp, fs, fd)
    st.markdown("<hr>", unsafe_allow_html=True)

    pitch, fig, ax = make_pitch(figsize=(16, 10))
    SORT = {"unsuccessful": 0, "successful": 1, "progressive": 2, "key": 3}

    for p in sorted(fp, key=lambda p: SORT[pass_type(p)]):
        pt    = pass_type(p)
        color = PASS_COLOR[pt]
        alpha = 0.28 if pt == "unsuccessful" else (0.82 if pt in ("key","progressive") else 0.52)
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"],
                     ax=ax, color=color, alpha=alpha,
                     width=1.1, headwidth=4, headlength=4)

    items = []
    if show_succ:   items.append((PASS_COLOR["successful"],   "Successful"))
    if show_unsucc: items.append((PASS_COLOR["unsuccessful"], "Unsuccessful"))
    if show_prog:   items.append((PASS_COLOR["progressive"],  "Progressive"))
    if show_key:    items.append((PASS_COLOR["key"],          "Shot Assist"))
    if items: legend(ax, items)

    add_title(fig, "Pass Map  ·  B. Mead",
              f"{season_label()}  ·  {len(fp):,} passes  ·  "
              f"{sum(1 for p in fp if p['progressive']):,} progressive  ·  "
              f"{sum(1 for p in fp if p['key_pass']):,} shot assists")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – SHOT MAP
# ═══════════════════════════════════════════════════════════════════════════════

with tab_shot:
    fp = f_passes(passes)
    fs = f_shots(shots)
    fd = f_def(def_acts)
    metric_row(fp, fs, fd)
    st.markdown("<hr>", unsafe_allow_html=True)

    pitch = VerticalPitch(
        pitch_type="opta", pitch_color=PITCH_BG, line_color=PITCH_LINE,
        linewidth=1.0, goal_type="box", half=True, line_zorder=2,
    )
    fig, ax = pitch.draw(figsize=(10, 8))
    fig.patch.set_facecolor(FIG_BG)

    for tid in (13, 14, 15, 16):
        color, marker, size, alpha, zorder = SHOT_STYLE[tid]
        subset = [s for s in fs if s["type_id"] == tid]
        if subset:
            pitch.scatter(
                [s["x"] for s in subset], [s["y"] for s in subset],
                ax=ax, c=color, marker=marker, s=size,
                alpha=alpha, zorder=zorder,
                edgecolors="white", linewidths=0.3,
            )

    counts = {tid: sum(1 for s in fs if s["type_id"] == tid) for tid in (16,15,14,13)}
    legend(ax, [
        (SHOT_STYLE[16][0], f"Goal ({counts[16]})"),
        (SHOT_STYLE[15][0], f"On Target ({counts[15]})"),
        (SHOT_STYLE[14][0], f"Post ({counts[14]})"),
        (SHOT_STYLE[13][0], f"Off Target ({counts[13]})"),
    ], loc="lower center")

    add_title(fig, "Shot Map  ·  B. Mead",
              f"{season_label()}  ·  {len(fs)} shots  ·  {counts[16]} goals")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – HEAT MAP
# ═══════════════════════════════════════════════════════════════════════════════

with tab_heat:
    fp = f_passes(passes)
    fs = f_shots(shots)
    fd = f_def(def_acts)
    ft = f_touches(touches)
    metric_row(fp, fs, fd)
    st.markdown("<hr>", unsafe_allow_html=True)

    pitch = Pitch(
        pitch_type="opta", pitch_color=PITCH_BG, line_color=PITCH_LINE,
        linewidth=1.2, goal_type="box", line_zorder=2,
    )
    fig, ax = pitch.draw(figsize=(16, 10))
    fig.patch.set_facecolor(FIG_BG)

    xs = [t["x"] for t in ft]
    ys = [t["y"] for t in ft]

    if xs:
        cmap = LinearSegmentedColormap.from_list(
            "mead_heat", [PITCH_BG, "#1a3a2a", "#3fb950", "#e3b341", "#f78166", "#ffffff"], N=256
        )
        pitch.kdeplot(xs, ys, ax=ax, cmap=cmap, fill=True,
                      levels=100, alpha=0.88, bw_adjust=0.65, zorder=1)

    add_title(fig, "Heat Map  ·  B. Mead  (all actions)",
              f"{season_label()}  ·  {len(ft):,} actions")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – TERRITORY MAP
# ═══════════════════════════════════════════════════════════════════════════════

with tab_terr:
    fp = f_passes(passes)
    fs = f_shots(shots)
    fd = f_def(def_acts)
    ft = f_touches(touches)
    metric_row(fp, fs, fd)
    st.markdown("<hr>", unsafe_allow_html=True)

    pitch = Pitch(
        pitch_type="opta", pitch_color=PITCH_BG, line_color=PITCH_LINE,
        linewidth=1.2, goal_type="box", line_zorder=2,
    )
    fig, ax = pitch.draw(figsize=(16, 10))
    fig.patch.set_facecolor(FIG_BG)

    if ft:
        xs = np.array([t["x"] for t in ft])
        ys = np.array([t["y"] for t in ft])

        # Bin into 12 × 8 zones and count actions per zone
        bin_stat = pitch.bin_statistic(xs, ys, statistic="count", bins=(12, 8))

        cmap_terr = LinearSegmentedColormap.from_list(
            "terr", ["#0d1117", "#0d2137", "#1a4a7a", "#58a6ff", "#e3b341"], N=256
        )
        pitch.heatmap(bin_stat, ax=ax, cmap=cmap_terr, alpha=0.85, zorder=1)

        # Annotate each zone with the count
        pitch.label_heatmap(
            bin_stat, ax=ax,
            color="#e6edf3", fontsize=7, ha="center", va="center",
            str_format="{:.0f}", zorder=3,
        )

    add_title(fig, "Territory Map  ·  B. Mead",
              f"{season_label()}  ·  action count per zone  ·  {len(ft):,} total actions")

    # colorbar
    sm = plt.cm.ScalarMappable(
        cmap=LinearSegmentedColormap.from_list(
            "terr", ["#0d1117", "#0d2137", "#1a4a7a", "#58a6ff", "#e3b341"], N=256),
        norm=plt.Normalize(vmin=0, vmax=max(
            bin_stat["statistic"].max(), 1))
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.018, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#8b949e", labelsize=8)
    cbar.outline.set_edgecolor("#30363d")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#8b949e")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – PASS NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

with tab_net:
    fp = f_passes(passes)
    fs = f_shots(shots)
    fd = f_def(def_acts)
    metric_row(fp, fs, fd)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── filter by selected seasons ────────────────────────────────────────
    edges_f = [e for e in net_edges if e["season"] in selected_seasons]
    pos_f   = [p for p in net_pos   if p["season"] in selected_seasons]

    if not edges_f:
        st.info("No pass network data for the selected season(s).")
    else:
        from collections import defaultdict, Counter
        import numpy as np

        # Average position per player
        avg_pos = defaultdict(lambda: {"x": [], "y": []})
        for p in pos_f:
            avg_pos[p["player"]]["x"].append(p["x"])
            avg_pos[p["player"]]["y"].append(p["y"])
        avg_pos = {
            pl: (np.mean(v["x"]), np.mean(v["y"]))
            for pl, v in avg_pos.items()
        }

        # Pass counts: directional edge → combine both directions
        pair_counts = Counter()
        for e in edges_f:
            pair = tuple(sorted([e["passer"], e["recipient"]]))
            pair_counts[pair] += 1

        # Per-player total passes (for node size)
        player_pass_count = Counter()
        for e in edges_f:
            player_pass_count[e["passer"]] += 1

        # Keep only players who appear in avg_pos
        players = [p for p in avg_pos if player_pass_count[p] > 0]

        # Min edge count threshold (top connections only to avoid clutter)
        min_edge = max(2, int(np.percentile(list(pair_counts.values()), 30)))

        # ── draw ──────────────────────────────────────────────────────────
        pitch = Pitch(
            pitch_type="opta", pitch_color=PITCH_BG, line_color="#2d333b",
            linewidth=0.8, goal_type="box", line_zorder=1,
        )
        fig, ax = pitch.draw(figsize=(16, 10))
        fig.patch.set_facecolor(FIG_BG)

        max_edge_cnt = max(pair_counts.values()) if pair_counts else 1
        max_node_cnt = max(player_pass_count[p] for p in players) if players else 1

        # ── edges ─────────────────────────────────────────────────────────
        for (p1, p2), cnt in pair_counts.items():
            if cnt < min_edge:
                continue
            if p1 not in avg_pos or p2 not in avg_pos:
                continue
            x1, y1 = avg_pos[p1]
            x2, y2 = avg_pos[p2]
            width  = 0.8 + 5.0 * (cnt / max_edge_cnt)
            alpha  = 0.25 + 0.55 * (cnt / max_edge_cnt)
            is_mead_edge = "B. Mead" in (p1, p2)
            color  = ACCENT_BLUE if is_mead_edge else "#484f58"
            ax.plot([x1, x2], [y1, y2],
                    color=color, linewidth=width, alpha=alpha,
                    solid_capstyle="round", zorder=2)

        # ── nodes ─────────────────────────────────────────────────────────
        for pl in players:
            if pl not in avg_pos:
                continue
            x, y   = avg_pos[pl]
            cnt    = player_pass_count[pl]
            is_mead = pl == "B. Mead"
            size   = 100 + 700 * (cnt / max_node_cnt)
            color  = ACCENT_YLW if is_mead else "#c9d1d9"
            edge_c = "#0d1117"
            lw     = 2.5 if is_mead else 1.0

            ax.scatter(x, y, s=size, c=color, zorder=4,
                       edgecolors=edge_c, linewidths=lw)

            # label
            label = pl.split(". ")[-1] if ". " in pl else pl
            fontsize  = 9.5 if is_mead else 7.5
            fontweight = "bold" if is_mead else "normal"
            fcolor     = ACCENT_YLW if is_mead else "#e6edf3"

            # offset label so it doesn't sit on top of node
            ax.text(x, y + 3.5, label,
                    ha="center", va="bottom",
                    fontsize=fontsize, fontweight=fontweight,
                    color=fcolor, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="#0d1117", alpha=0.6,
                              edgecolor="none"))

        # ── legend ────────────────────────────────────────────────────────
        legend(ax, [
            (ACCENT_YLW,  "B. Mead"),
            ("#c9d1d9",   "Teammate"),
            (ACCENT_BLUE, "Mead connection"),
            ("#484f58",   "Other connection"),
        ])

        # subtitle stats
        top_partner = max(
            [(p, c) for (p1, p2), c in pair_counts.items()
             for p in ([p1] if p2 == "B. Mead" else ([p2] if p1 == "B. Mead" else []))],
            key=lambda x: x[1], default=("—", 0)
        )
        total_mead_passes = sum(
            c for (p1, p2), c in pair_counts.items() if "B. Mead" in (p1, p2)
        )

        add_title(
            fig, "Pass Network  ·  B. Mead",
            f"{season_label()}  ·  Node size = passes played  ·  "
            f"Edge width = pass volume  ·  Top partner: {top_partner[0]} ({top_partner[1]})"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 – PASS SONARS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_sonar(ax, player_passes, title, highlight=False, fontsize_title=10):
    """Draw a pass direction sonar on a given polar Axes."""
    import numpy as np

    N_BINS   = 16
    bin_size = 360 / N_BINS
    bins     = np.linspace(-np.pi, np.pi, N_BINS + 1)
    theta    = bins[:-1] + (bins[1] - bins[0]) / 2   # bin centres

    succ_counts = np.zeros(N_BINS)
    fail_counts = np.zeros(N_BINS)
    avg_dist    = np.zeros(N_BINS)
    dist_count  = np.zeros(N_BINS)

    for p in player_passes:
        a = np.radians(p["angle"])
        # find bin
        b = int((a + np.pi) / (2 * np.pi) * N_BINS) % N_BINS
        if p["outcome"] == 1:
            succ_counts[b] += 1
        else:
            fail_counts[b] += 1
        avg_dist[b]   += p["dist"]
        dist_count[b] += 1

    avg_dist = np.where(dist_count > 0, avg_dist / dist_count, 0)
    max_cnt  = max(succ_counts + fail_counts) or 1

    width = 2 * np.pi / N_BINS * 0.88

    # background rings
    for r in [0.25, 0.5, 0.75, 1.0]:
        ring = plt.Circle((0, 0), r * max_cnt, transform=ax.transData._b,
                           fill=False, color="#30363d", linewidth=0.4, zorder=0)

    # unsuccessful (bottom layer)
    ax.bar(theta, fail_counts,
           width=width, bottom=0,
           color=ACCENT_ORG, alpha=0.55, zorder=2, linewidth=0)

    # successful stacked on top
    ax.bar(theta, succ_counts,
           width=width, bottom=fail_counts,
           color=ACCENT_YLW, alpha=0.85, zorder=3, linewidth=0)

    # average distance dot ring
    valid = dist_count > 0
    ax.scatter(theta[valid], avg_dist[valid] * (max_cnt / (avg_dist[valid].max() or 1)) * 0.55,
               s=8, color=ACCENT_BLUE, zorder=5, alpha=0.8)

    # direction labels
    dir_labels = {
        0:  "→",   # forward
        4:  "↑",   # left / up
        8:  "←",   # back
        12: "↓",   # right / down
    }
    for bidx, lbl in dir_labels.items():
        ax.text(theta[bidx], max_cnt * 1.25, lbl,
                ha="center", va="center", fontsize=9,
                color="#8b949e", fontweight="bold")

    # grid / style
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, max_cnt * 1.4)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_facecolor("#161b22")
    ax.spines["polar"].set_visible(False)

    total = int(succ_counts.sum() + fail_counts.sum())
    acc   = round(succ_counts.sum() / total * 100) if total else 0
    title_color = ACCENT_YLW if highlight else "#e6edf3"
    ax.set_title(
        f"{title}\n"
        f"{'─'*len(title)}\n"
        f"{total} passes · {acc}% acc",
        color=title_color, fontsize=fontsize_title,
        fontweight="bold" if highlight else "normal",
        pad=8, loc="center",
    )


with tab_sonar:
    fp = f_passes(passes)
    fs = f_shots(shots)
    fd = f_def(def_acts)
    metric_row(fp, fs, fd)
    st.markdown("<hr>", unsafe_allow_html=True)

    import numpy as np
    from collections import Counter, defaultdict

    sp_f = [p for p in sonar_passes if p["season"] in selected_seasons]

    if not sp_f:
        st.info("No sonar data for the selected season(s).")
    else:
        mead_sp = [p for p in sp_f if p["player"] == "B. Mead"]

        # ── large Mead sonar ──────────────────────────────────────────────
        fig_m = plt.figure(figsize=(6, 6), facecolor=FIG_BG)
        ax_m  = fig_m.add_subplot(111, projection="polar")
        draw_sonar(ax_m, mead_sp, "B. Mead", highlight=True, fontsize_title=12)

        # legend inside the big sonar
        import matplotlib.patches as mp
        handles = [
            mp.Patch(color=ACCENT_YLW, label="Successful"),
            mp.Patch(color=ACCENT_ORG, label="Unsuccessful"),
            mp.Patch(color=ACCENT_BLUE, label="Avg distance (scaled)"),
        ]
        ax_m.legend(handles=handles, loc="lower center",
                    bbox_to_anchor=(0.5, -0.18), ncol=3,
                    fontsize=7.5, framealpha=0.0,
                    labelcolor="#c9d1d9", handlelength=1.2)

        fig_m.suptitle(
            f"Pass Sonar  ·  B. Mead  ·  {season_label()}\n"
            f"→ = Forward (attacking)   ← = Backward   ↑↓ = Wide",
            color="#8b949e", fontsize=8, y=0.02,
        )
        plt.tight_layout()

        # center the big sonar
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.pyplot(fig_m, use_container_width=True)
        plt.close(fig_m)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── mini sonars for top partners ──────────────────────────────────
        # Find top pass partners (based on net_edges filtered by season)
        edges_f = [e for e in net_edges if e["season"] in selected_seasons]
        pair_counts = Counter()
        for e in edges_f:
            pair = tuple(sorted([e["passer"], e["recipient"]]))
            pair_counts[pair] += 1

        top_partners = sorted(
            [p for (p1, p2), _ in pair_counts.most_common()
             for p in ([p1] if p2 == "B. Mead" else ([p2] if p1 == "B. Mead" else []))],
            key=lambda p: -pair_counts[tuple(sorted([p, "B. Mead"]))]
        )
        # deduplicate keeping order
        seen = set()
        top_partners_unique = []
        for p in top_partners:
            if p not in seen and p != "B. Mead":
                seen.add(p)
                top_partners_unique.append(p)
        top_partners_unique = top_partners_unique[:8]

        if top_partners_unique:
            st.markdown(
                "<div style='font-size:13px;font-weight:700;color:#8b949e;"
                "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px'>"
                "Top Pass Partners — Direction Wheels</div>",
                unsafe_allow_html=True,
            )

            cols_per_row = 4
            for row_start in range(0, len(top_partners_unique), cols_per_row):
                row_players = top_partners_unique[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, partner in zip(cols, row_players):
                    with col:
                        partner_passes = [p for p in sp_f if p["player"] == partner]
                        fig_p, ax_p = plt.subplots(
                            figsize=(3.2, 3.2), subplot_kw={"projection": "polar"},
                            facecolor=FIG_BG,
                        )
                        short_name = partner.split(". ")[-1] if ". " in partner else partner
                        edge_cnt = pair_counts.get(
                            tuple(sorted([partner, "B. Mead"])), 0
                        )
                        draw_sonar(ax_p, partner_passes,
                                   f"{short_name}\n({edge_cnt} passes w/ Mead)",
                                   fontsize_title=8)
                        plt.tight_layout()
                        st.pyplot(fig_p, use_container_width=True)
                        plt.close(fig_p)

# TAB 7 – DEFENSIVE ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_def:
    fp = f_passes(passes)
    fs = f_shots(shots)
    fd = f_def(def_acts)
    metric_row(fp, fs, fd)
    st.markdown("<hr>", unsafe_allow_html=True)

    pitch, fig, ax = make_pitch(figsize=(16, 10))
    MARKER = {"Tackle": "s", "Interception": "D", "Clearance": "o", "Block": "^"}

    for d in fd:
        ax.scatter(d["x"], d["y"],
                   c=DEF_COLOR[d["action"]], marker=MARKER[d["action"]],
                   s=90, alpha=0.80, zorder=3,
                   edgecolors="#0d1117", linewidths=0.5)

    if sel_def:
        legend(ax, [
            (DEF_COLOR[a], f"{a} ({sum(1 for d in fd if d['action']==a)})")
            for a in sel_def
        ])

    add_title(fig, "Defensive Actions  ·  B. Mead",
              f"{season_label()}  ·  {len(fd)} actions")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 – PERCENTILE CHART
# ═══════════════════════════════════════════════════════════════════════════════

with tab_perc:
    st.markdown(
        "<div style='font-size:12px;color:#8b949e;margin-bottom:12px'>"
        "Bars show each season relative to Mead's personal best (100 = best season for that metric)."
        "</div>", unsafe_allow_html=True
    )

    season_stats = {}
    for s in seasons_available:
        sp = [p for p in passes   if p["season"] == s]
        ss = [sh for sh in shots  if sh["season"] == s]
        sd = [d  for d  in def_acts if d["season"] == s]
        total_p = len(sp)
        season_stats[s] = {
            "Passes":        total_p,
            "Pass Acc %":    round(sum(1 for p in sp if p["outcome"]==1)/total_p*100,1) if total_p else 0,
            "Progressive":   sum(1 for p in sp if p["progressive"]),
            "Shot Assists":  sum(1 for p in sp if p["key_pass"]),
            "Shots":         len(ss),
            "Goals":         sum(1 for sh in ss if sh["type_id"]==16),
            "Shot Acc %":    round(sum(1 for sh in ss if sh["type_id"] in (15,16))/len(ss)*100,1) if ss else 0,
            "Tackles":       sum(1 for d in sd if d["action"]=="Tackle"),
            "Interceptions": sum(1 for d in sd if d["action"]=="Interception"),
        }

    metrics  = list(next(iter(season_stats.values())).keys())
    max_vals = {m: max(season_stats[s][m] for s in season_stats) or 1 for m in metrics}

    BAR_COLORS = [ACCENT_YLW, ACCENT_BLUE, ACCENT_GRN, "#58a6ff",
                  ACCENT_ORG, ACCENT_YLW, ACCENT_GRN, ACCENT_ORG, ACCENT_PRP]

    display_seasons = [s for s in seasons_available if s in selected_seasons]
    cols_per_row    = min(len(display_seasons), 3)
    rows_list       = [display_seasons[i:i+cols_per_row]
                       for i in range(0, len(display_seasons), cols_per_row)]

    for row_seasons in rows_list:
        cols = st.columns(len(row_seasons))
        for col, s in zip(cols, row_seasons):
            with col:
                fig, ax = plt.subplots(figsize=(5, 4.8))
                fig.patch.set_facecolor(FIG_BG)
                ax.set_facecolor("#161b22")

                vals  = [season_stats[s][m] for m in metrics]
                pcts  = [v / max_vals[m] * 100 for v, m in zip(vals, metrics)]
                y_pos = np.arange(len(metrics))

                # background bars
                ax.barh(y_pos, [100]*len(metrics), color="#21262d", height=0.6,
                        edgecolor="none", zorder=1)
                # value bars
                bars = ax.barh(y_pos, pcts, color=BAR_COLORS[:len(metrics)],
                               height=0.6, alpha=0.9, edgecolor="none", zorder=2)

                for bar, val in zip(bars, vals):
                    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                            f"{val:g}", va="center", ha="left",
                            color="#e6edf3", fontsize=7.5, fontweight="700")

                ax.set_yticks(y_pos)
                ax.set_yticklabels(metrics, color="#8b949e", fontsize=8)
                ax.set_xlim(0, 120)
                ax.set_xlabel("% of personal best", color="#484f58", fontsize=7)
                ax.tick_params(colors="#484f58", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.axvline(100, color="#30363d", linewidth=0.8, linestyle="--", zorder=3)
                ax.set_title(s.replace("WSL ", ""), color=ACCENT_BLUE,
                             fontsize=10, fontweight="800", pad=10)
                ax.set_facecolor("#161b22")

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
