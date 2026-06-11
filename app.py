import json
import glob
import os
import math
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import streamlit as st
from mplsoccer import Pitch, VerticalPitch

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Beth Mead | Data Portfolio",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded",
)

# ── global CSS ────────────────────────────────────────────────────────────────

# Fonts via st.html (not sanitised)
st.html('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">')
# CSS wrapped in display:none div so text is hidden but styles apply
st.markdown('''<div style="display:none"><style>'

/* ═══ RESET & BASE ═══════════════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    background: #07090f !important;
    color: #e8eaf0 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ═══ HIDE STREAMLIT CHROME ════════════════════════════════════════════════ */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.viewerBadge_container__r5tak,
.stDeployButton { display: none !important; visibility: hidden !important; }

/* ═══ REMOVE DEFAULT PADDING ═════════════════════════════════════════════ */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* ═══ SCROLLBAR ══════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #07090f; }
::-webkit-scrollbar-thumb { background: #1e2530; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #2a3346; }

/* ═══ SIDEBAR ════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #0c0f18 !important;
    border-right: 1px solid #151c28 !important;
}
[data-testid="stSidebar"] * { color: #c8cdd8 !important; }
[data-testid="stSidebar"] .stMarkdown p {
    font-size: 12px !important;
    color: #5a6478 !important;
}

/* sidebar section labels */
[data-testid="stSidebar"] h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: #3b82f6 !important;
    margin: 16px 0 6px !important;
}

/* sidebar checkbox */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    font-size: 13px !important;
    color: #8a93a8 !important;
}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label:hover {
    color: #e8eaf0 !important;
}

/* sidebar multiselect */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #111520 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 8px !important;
}

/* ═══ TABS ════════════════════════════════════════════════════════════════ */
[data-baseweb="tab-list"] {
    background: #0c0f18 !important;
    border-bottom: 1px solid #151c28 !important;
    border-radius: 0 !important;
    padding: 0 24px !important;
    gap: 0 !important;
    border-top: none !important;
    border-left: none !important;
    border-right: none !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: #5a6478 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 14px 16px !important;
    text-transform: uppercase !important;
    transition: all 0.15s ease !important;
}
[data-baseweb="tab"]:hover {
    color: #c8cdd8 !important;
    background: transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #f59e0b !important;
    border-bottom: 2px solid #f59e0b !important;
    background: transparent !important;
}
[data-testid="stTabContent"] {
    padding: 32px 32px 48px !important;
    background: #07090f !important;
}

/* ═══ METRICS (custom, override native) ════════════════════════════════ */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f1420 0%, #111827 100%) !important;
    border: 1px solid #1a2235 !important;
    border-top: 2px solid #f59e0b !important;
    border-radius: 12px !important;
    padding: 20px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stMetric"]:hover {
    border-color: #f59e0b !important;
    border-top-color: #f59e0b !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #5a6478 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    line-height: 1.1 !important;
}

/* ═══ DIVIDER ════════════════════════════════════════════════════════════ */
hr {
    border: none !important;
    border-top: 1px solid #151c28 !important;
    margin: 24px 0 !important;
}

/* ═══ SELECT / DROPDOWN ══════════════════════════════════════════════════ */
[data-baseweb="select"] > div {
    background: #0f1420 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 8px !important;
    color: #c8cdd8 !important;
}

/* ═══ DATAFRAME ══════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid #1a2235 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
    background: #0c0f18 !important;
    color: #5a6478 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stDataFrame"] td {
    font-size: 13px !important;
    color: #c8cdd8 !important;
}

/* ═══ EXPANDER ═══════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
    border: 1px solid #1a2235 !important;
    border-radius: 10px !important;
    background: #0c0f18 !important;
}

/* ═══ INFO / WARNING ═════════════════════════════════════════════════════ */
[data-testid="stAlert"] {
    background: #0c0f18 !important;
    border: 1px solid #1a2235 !important;
    border-radius: 10px !important;
    color: #8a93a8 !important;
}
</style></div>''', unsafe_allow_html=True)

# ── palette ───────────────────────────────────────────────────────────────────

PITCH_BG   = "#0a0d15"
PITCH_LINE = "#d1d5db"
FIG_BG     = "#07090f"
C_BLUE     = "#3b82f6"
C_ORANGE   = "#f97316"
C_GREEN    = "#10b981"
C_PURPLE   = "#a78bfa"
C_YELLOW   = "#f59e0b"
C_MUTED    = "#374151"

PASS_COLOR = {"key": C_BLUE, "progressive": C_GREEN, "successful": C_YELLOW, "unsuccessful": C_ORANGE}
SHOT_STYLE = {16: (C_YELLOW,"*",300,.95,4), 15: (C_BLUE,"o",130,.85,3),
              14: (C_PURPLE,"D",110,.85,3), 13: (C_ORANGE,"X",90,.50,2)}
DEF_COLOR  = {"Tackle": C_ORANGE, "Interception": C_YELLOW, "Clearance": C_GREEN, "Block": C_BLUE}

# ── helpers ───────────────────────────────────────────────────────────────────

def get_q(event, qid):
    for q in event.get("qualifier", []):
        if q["qualifierId"] == qid:
            return q.get("value")
    return None

def has_q(event, qid):
    return any(q["qualifierId"] == qid for q in event.get("qualifier", []))

def make_pitch(vertical=False, half=False, figsize=(16,10), lz=1):
    cls = VerticalPitch if vertical else Pitch
    kw  = dict(pitch_type="opta", pitch_color=PITCH_BG, line_color=PITCH_LINE,
               linewidth=1.0, goal_type="box", line_zorder=lz)
    if vertical: kw["half"] = half
    p = cls(**kw)
    fig, ax = p.draw(figsize=figsize)
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(PITCH_BG)
    return p, fig, ax

def add_title(fig, title, sub=""):
    fig.text(.5,.98,title,ha="center",va="top",fontsize=14,fontweight="800",color="#e6edf3")
    if sub:
        fig.text(.5,.945,sub,ha="center",va="top",fontsize=9,color="#8b949e")

def add_legend(ax, items, loc="lower left"):
    h = [mpatches.Patch(color=c, label=l) for c,l in items]
    ax.legend(handles=h, loc=loc, fontsize=8.5, framealpha=.35,
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#c9d1d9", handlelength=1.2, borderpad=.7, labelspacing=.5)

# ── zone constants ────────────────────────────────────────────────────────────

HS_L = (17, 37)   # left  half-space  y range on Opta 0-100 scale
HS_R = (63, 83)   # right half-space  y range

def in_halfspace(y):
    return HS_L[0] <= y <= HS_L[1] or HS_R[0] <= y <= HS_R[1]

# Zone 14: central pocket between the penalty area and the midfield third
# x = 66-83 (depth), y = 21-79 (central channel, excludes wide areas)
Z14_X = (66, 83)
Z14_Y = (21, 79)

def in_zone14(x, y):
    return Z14_X[0] <= x <= Z14_X[1] and Z14_Y[0] <= y <= Z14_Y[1]

# ── data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data…")
def load_data():
    base  = os.path.dirname(__file__)
    # ── pre-filter to Arsenal files only by filename (saves ~2 GB of parsing) ──
    all_files = sorted(glob.glob(os.path.join(base,"**/*.json"), recursive=True))
    files = [f for f in all_files if "Arsenal" in os.path.basename(f)]

    passes = []; shots = []; touches = []; def_acts = []
    dribbles = []; net_edges = []; net_pos = []; sonar_passes = []
    match_rows = []

    for fpath in files:
        fname  = os.path.basename(fpath)
        season = fpath.split(os.sep)[-2]

        with open(fpath) as fp:
            data = json.load(fp)
        if "event" not in data:
            continue
        events = data["event"]

        # parse metadata from filename: "YYYY-MM-DD_Home - Away.json"
        date_str  = fname.split("_")[0] if "_" in fname else "unknown"
        teams_str = fname.replace(".json","").split("_",1)[1] if "_" in fname else fname
        parts     = teams_str.split(" - ")
        home_t    = parts[0].strip() if len(parts)>=2 else ""
        away_t    = parts[1].strip() if len(parts)>=2 else ""
        opponent  = away_t if "Arsenal" in home_t else home_t

        # ── single pass through all events ────────────────────────────────
        arsenal_id   = None
        match_agg    = {"passes":0,"succ_passes":0,"shots":0,"goals":0,
                        "key_passes":0,"prog_passes":0,"tackles":0,
                        "interceptions":0,"dribbles":0,"succ_dribbles":0}

        for i, e in enumerate(events):
            pid  = e.get("playerName")
            cid  = e.get("contestantId")
            tid  = e.get("typeId")
            x,y  = e.get("x"), e.get("y")
            outcome = e.get("outcome", 0)

            # discover Arsenal contestant ID on first Mead event
            if arsenal_id is None and pid == "B. Mead":
                arsenal_id = cid

            # ── Arsenal team-level (sonar + pass network) ─────────────────
            if arsenal_id and cid == arsenal_id:
                if tid == 1 and x is not None:
                    ex_s, ey_s = get_q(e,140), get_q(e,141)
                    if ex_s and ey_s:
                        dx, dy = float(ex_s)-x, float(ey_s)-y
                        # sonar — store as compact tuple (player, angle, dist, outcome, season)
                        sonar_passes.append((pid or "", math.degrees(math.atan2(dy,dx)),
                                             math.hypot(dx,dy), outcome, season))
                        # pass network
                        if outcome == 1 and pid:
                            net_pos.append((pid, x, y, season))
                            for j in range(i+1, min(i+6, len(events))):
                                ne = events[j]
                                if ne.get("contestantId")==arsenal_id and ne.get("playerName"):
                                    rec = ne["playerName"]
                                    if rec != pid:
                                        net_edges.append((pid, rec, season))
                                    break

            # ── Mead-only events ──────────────────────────────────────────
            if pid != "B. Mead" or x is None or y is None:
                continue

            period = e.get("periodId", 1)
            base   = dict(x=x, y=y, season=season, outcome=outcome,
                          period=period, date=date_str, opponent=opponent)
            touches.append(base)

            if tid == 1:
                ex_s, ey_s = get_q(e,140), get_q(e,141)
                if ex_s and ey_s:
                    end_x, end_y = float(ex_s), float(ey_s)
                    is_key  = has_q(e, 210)
                    is_prog = outcome==1 and end_x-x>=10 and end_x>50
                    passes.append({**base, "end_x":end_x, "end_y":end_y,
                                   "key_pass":is_key, "progressive":is_prog,
                                   "dist":math.hypot(end_x-x, end_y-y)})
                    match_agg["passes"] += 1
                    if outcome == 1: match_agg["succ_passes"] += 1
                    if is_key:       match_agg["key_passes"]  += 1
                    if is_prog:      match_agg["prog_passes"] += 1
            elif tid in (13,14,15,16):
                shots.append({**base, "type_id":tid, "foot":get_q(e,56) or "Unknown"})
                match_agg["shots"] += 1
                if tid == 16: match_agg["goals"] += 1
            elif tid == 3:
                dribbles.append(base)
                match_agg["dribbles"] += 1
                if outcome == 1: match_agg["succ_dribbles"] += 1

            if tid in (7,8,12,74):
                lm = {7:"Tackle",8:"Interception",12:"Clearance",74:"Block"}
                def_acts.append({**base, "action":lm[tid]})
                if tid == 7: match_agg["tackles"]       += 1
                if tid == 8: match_agg["interceptions"] += 1

        match_rows.append({"match_id":fname.replace(".json",""),
                            "date":date_str,"opponent":opponent,
                            "season":season, **match_agg})

    # back-fill progressive pass count in match_rows
    prog_by_match = Counter()
    for p in passes:
        if p["progressive"]:
            prog_by_match[p["date"]+"_"+p["opponent"]] += 1
    for row in match_rows:
        key = row["date"]+"_"+row["opponent"]
        row["prog_passes"] = prog_by_match[key]

    return passes,shots,touches,def_acts,dribbles,net_edges,net_pos,sonar_passes,match_rows


passes,shots,touches,def_acts,dribbles,net_edges,net_pos,sonar_passes,match_rows = load_data()
seasons_available = sorted(set(p["season"] for p in passes))

# ── navigation state ──────────────────────────────────────────────────────────

VIZZES = [
    {"id":"pass_map",   "icon":"🎯","title":"Pass Map",          "cat":"Passing",    "desc":"All passes coloured by type — successful, progressive, key passes and shot assists"},
    {"id":"pass_net",   "icon":"🔗","title":"Pass Network",      "cat":"Passing",    "desc":"Player nodes connected by pass volume with B. Mead centralised"},
    {"id":"pass_sonar", "icon":"🧭","title":"Pass Sonars",       "cat":"Passing",    "desc":"Direction wheels showing where Mead plays the ball and her top partners"},
    {"id":"half_space", "icon":"◈", "title":"Half-Space Passes", "cat":"Passing",    "desc":"Passes through the half-space channels with zone overlay and breakdown"},
    {"id":"crossings",  "icon":"↗️","title":"Crossings",         "cat":"Passing",    "desc":"Passes from wide channels into the final third"},
    {"id":"shot_map",   "icon":"🥅","title":"Shot Map",          "cat":"Attacking",  "desc":"Half-pitch shot scatter — goals, saves, posts and misses"},
    {"id":"shot_zones", "icon":"🎯","title":"Shot Zones",        "cat":"Attacking",  "desc":"Binned shot heatmap of the attacking half with individual shots overlaid"},
    {"id":"dribbles",   "icon":"🏃","title":"Dribbles",          "cat":"Attacking",  "desc":"Take-on scatter map showing where Mead beats opponents"},
    {"id":"zone14",     "icon":"🟡","title":"Zone 14",            "cat":"Attacking",  "desc":"Actions in and around Zone 14 — the pocket between the box and the midfield third"},
    {"id":"heat_map",   "icon":"🔥","title":"Heat Map",          "cat":"Movement",   "desc":"KDE density of all actions showing zones of influence"},
    {"id":"territory",  "icon":"🗺","title":"Territory Map",     "cat":"Movement",   "desc":"12×8 binned action-count grid revealing dominant zones"},
    {"id":"career",     "icon":"📈","title":"Career Timeline",   "cat":"Analysis",   "desc":"Season-by-season line charts for goals, assists, shots, passes and more"},
    {"id":"match_stats","icon":"📅","title":"Match Stats",       "cat":"Analysis",   "desc":"Bar chart and data table of any metric broken down per game"},
    {"id":"radar",      "icon":"🍕","title":"Radar Chart",       "cat":"Analysis",   "desc":"Spider chart comparing up to three seasons across ten metrics"},
    {"id":"style",      "icon":"📏","title":"Style Analysis",    "cat":"Analysis",   "desc":"Pass length distribution, shooting foot split and direction breakdown"},
    {"id":"percentile", "icon":"📊","title":"Percentile Chart",  "cat":"Analysis",   "desc":"Horizontal bars showing each season relative to career best"},
    {"id":"compare",    "icon":"🔄","title":"Season Compare",    "cat":"Comparison", "desc":"Side-by-side pass maps for any two seasons"},
    {"id":"opposition", "icon":"🆚","title":"Opposition",        "cat":"Comparison", "desc":"Any metric ranked and broken down by opponent team"},
    {"id":"archetypes", "icon":"🧬","title":"Archetypes",        "cat":"Comparison", "desc":"Six forward archetypes scored per season with radar and matrix"},
    {"id":"defensive",  "icon":"🛡","title":"Defensive Actions", "cat":"Defensive",  "desc":"Tackle, interception, clearance and block scatter map"},
    {"id":"sub_impact", "icon":"🔀","title":"Substitution Impact","cat":"Analysis",  "desc":"Monte Carlo simulation of player impact on/off the pitch — xG, xT, VAEP, EPV, G+, Goal Difference"},
    {"id":"xt_vaep",    "icon":"⚡","title":"xT & VAEP",          "cat":"Analysis",  "desc":"Expected Threat and VAEP scores computed from all WSL actions — season timelines and pitch heatmaps"},
    {"id":"match_metrics",  "icon":"📋","title":"Match Metrics",    "cat":"Analysis",  "desc":"199 per-team metrics for any WSL match — passing, shooting, defending, sequences, set pieces and ratios"},
    {"id":"player_metrics","icon":"👤","title":"Player Metrics",   "cat":"Analysis",  "desc":"250 pure player metrics per match — passing, shooting, defending, dribbling, set pieces, goalkeeping and advanced ratios"},
]

CAT_ORDER  = ["Passing","Attacking","Movement","Analysis","Comparison","Defensive"]
# keep sub_impact in Analysis visually
CAT_COLOR  = {"Passing":"#3b82f6","Attacking":"#f59e0b","Movement":"#10b981",
               "Analysis":"#a78bfa","Comparison":"#f97316","Defensive":"#ef4444"}

if "page" not in st.session_state:
    st.session_state.page = "home"

def nav(page_id):
    st.session_state.page = page_id


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # brand
    st.html("""
    <div style="padding:24px 4px 4px">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
        <div style="width:38px;height:38px;border-radius:50%;
             background:linear-gradient(135deg,#f59e0b,#ef4444);
             display:flex;align-items:center;justify-content:center;
             font-size:17px;flex-shrink:0">⚽</div>
        <div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:16px;
               font-weight:700;color:#f1f5f9;letter-spacing:-.2px">Beth Mead</div>
          <div style="font-size:10px;color:#4b5563">Arsenal WFC · Forward</div>
        </div>
      </div>
      <div style="height:1px;background:linear-gradient(90deg,#f59e0b44,transparent);
           margin-bottom:16px"></div>
    </div>""")

    # home button
    if st.button("🏠  Home", use_container_width=True, key="home_btn",
                 type="secondary"):
        nav("home")

    # nav by category
    st.html('<div style="height:1px;background:#151c28;margin:14px 0 10px"></div>')
    st.markdown("### Visualisations")

    for cat in CAT_ORDER:
        cat_vizzes = [v for v in VIZZES if v["cat"] == cat]
        color = CAT_COLOR[cat]
        st.html(f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.12em;color:{color};margin:12px 0 4px;padding-left:2px">'
                f'{cat}</div>')
        for v in cat_vizzes:
            active = st.session_state.page == v["id"]
            label  = f"{'▶ ' if active else ''}{v['icon']}  {v['title']}"
            if st.button(label, key=f"nav_{v['id']}", use_container_width=True,
                         type="primary" if active else "secondary"):
                nav(v["id"])

    # filters (shown only on viz pages)
    if st.session_state.page != "home":
        st.html('<div style="height:1px;background:#151c28;margin:16px 0 10px"></div>')
        st.markdown("### Season")
        selected_seasons = st.multiselect("season", seasons_available, seasons_available,
                                          label_visibility="collapsed")
        st.markdown("### Pass Filters")
        show_succ   = st.checkbox("Successful",             value=True)
        show_unsucc = st.checkbox("Unsuccessful",           value=True)
        show_prog   = st.checkbox("Progressive",            value=True)
        show_key    = st.checkbox("Shot Assist / Key Pass", value=True)
        st.markdown("### Shot Filters")
        show_goals   = st.checkbox("Goal",              value=True)
        show_on_tgt  = st.checkbox("On Target / Saved", value=True)
        show_post    = st.checkbox("Post",              value=True)
        show_off_tgt = st.checkbox("Off Target",        value=True)
        st.markdown("### Defensive Filters")
        all_def = ["Tackle","Interception","Clearance","Block"]
        sel_def = st.multiselect("def", all_def, all_def, label_visibility="collapsed")
    else:
        # defaults so filter vars always exist
        selected_seasons = seasons_available
        show_succ = show_unsucc = show_prog = show_key = True
        show_goals = show_on_tgt = show_post = show_off_tgt = True
        all_def = ["Tackle","Interception","Clearance","Block"]
        sel_def = all_def[:]

    st.html("""
    <div style="margin-top:24px;padding:12px;background:#0c0f18;
         border:1px solid #151c28;border-radius:8px;
         font-size:10px;color:#374151;line-height:1.8">
      <div style="color:#4b5563;font-weight:600;margin-bottom:3px">DATA SOURCE</div>
      Opta / Women's Super League<br>Seasons 2015 – 2026<br>
      <span style="color:#f59e0b">mplsoccer</span> visualisations
    </div>""")

# ── filter functions ──────────────────────────────────────────────────────────

def pass_type(p):
    if p["key_pass"]:     return "key"
    if p["progressive"]:  return "progressive"
    if p["outcome"]==1:   return "successful"
    return "unsuccessful"

def f_passes(data):
    out=[]
    for p in data:
        if p["season"] not in selected_seasons: continue
        pt = pass_type(p)
        if pt=="key"          and not show_key:    continue
        if pt=="progressive"  and not show_prog:   continue
        if pt=="successful"   and not show_succ:   continue
        if pt=="unsuccessful" and not show_unsucc: continue
        out.append(p)
    return out

def f_shots(data):
    out=[]
    for s in data:
        if s["season"] not in selected_seasons: continue
        tid=s["type_id"]
        if tid==16 and not show_goals:   continue
        if tid==15 and not show_on_tgt:  continue
        if tid==14 and not show_post:    continue
        if tid==13 and not show_off_tgt: continue
        out.append(s)
    return out

def f_season(data): return [d for d in data if d["season"] in selected_seasons]

def f_def(data): return [d for d in data if d["season"] in selected_seasons and d["action"] in sel_def]

def season_label():
    if not selected_seasons:          return "No season selected"
    if len(selected_seasons)==len(seasons_available): return "All Seasons"
    return " · ".join(s.replace("WSL ","") for s in selected_seasons)

# ── metric row ────────────────────────────────────────────────────────────────

def metric_row(fp, fs, fd, extra=None):
    total_p = len(fp); succ_p = sum(1 for p in fp if p["outcome"]==1)
    goals   = sum(1 for s in fs if s["type_id"]==16)
    data_m = [
        ("Passes",        f"{total_p:,}",                                          "🎯"),
        ("Pass Acc",      f"{round(succ_p/total_p*100,1) if total_p else 0}%",    "📊"),
        ("Progressive",   f"{sum(1 for p in fp if p['progressive']):,}",           "⬆️"),
        ("Shot Assists",  f"{sum(1 for p in fp if p['key_pass']):,}",              "🔑"),
        ("Shots",         f"{len(fs):,}",                                          "🔫"),
        ("Goals",         f"{goals:,}",                                            "⚽"),
        ("Tackles",       f"{sum(1 for d in fd if d['action']=='Tackle'):,}",      "🛡"),
        ("Interceptions", f"{sum(1 for d in fd if d['action']=='Interception'):,}","✂️"),
    ]
    cards = "".join(f"""
    <div style="background:linear-gradient(160deg,#0f1420,#0c111d);
         border:1px solid #1a2235;border-top:2px solid #f59e0b;
         border-radius:12px;padding:18px 16px;text-align:center;
         transition:transform .15s;cursor:default">
      <div style="font-size:18px;margin-bottom:6px">{icon}</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:24px;
           font-weight:700;color:#f1f5f9;line-height:1">{val}</div>
      <div style="font-size:10px;font-weight:600;letter-spacing:.1em;
           text-transform:uppercase;color:#4b5563;margin-top:6px">{lbl}</div>
    </div>""" for lbl,val,icon in data_m)
    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat(8,1fr);gap:12px;margin-bottom:8px'>"
        f"{cards}</div>",
        unsafe_allow_html=True,
    )

# ── sonar helper ──────────────────────────────────────────────────────────────

def draw_sonar(ax, player_passes, title, highlight=False, fontsize_title=10):
    N = 16
    bins  = np.linspace(-np.pi, np.pi, N+1)
    theta = bins[:-1] + (bins[1]-bins[0])/2
    succ  = np.zeros(N); fail = np.zeros(N)
    avg_d = np.zeros(N); cnt  = np.zeros(N)
    for p in player_passes:
        a = np.radians(p["angle"])
        b = int((a+np.pi)/(2*np.pi)*N) % N
        if p["outcome"]==1: succ[b]+=1
        else:               fail[b]+=1
        avg_d[b]+=p["dist"]; cnt[b]+=1
    avg_d = np.where(cnt>0, avg_d/cnt, 0)
    mx = max(succ+fail) or 1
    w  = 2*np.pi/N*0.88
    ax.bar(theta, fail,        width=w, color=C_ORANGE, alpha=.55, zorder=2, linewidth=0)
    ax.bar(theta, succ, bottom=fail, width=w, color=C_YELLOW, alpha=.85, zorder=3, linewidth=0)
    valid = cnt>0
    if valid.any():
        scaled = avg_d[valid]*(mx/(avg_d[valid].max() or 1))*0.55
        ax.scatter(theta[valid], scaled, s=8, color=C_BLUE, zorder=5, alpha=.8)
    for bidx,lbl in {0:"→",4:"↑",8:"←",12:"↓"}.items():
        ax.text(theta[bidx], mx*1.28, lbl, ha="center", va="center",
                fontsize=9, color="#8b949e", fontweight="bold")
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    ax.set_ylim(0, mx*1.45); ax.set_yticks([]); ax.set_xticks([])
    ax.set_facecolor("#161b22"); ax.spines["polar"].set_visible(False)
    total = int(succ.sum()+fail.sum())
    acc   = round(succ.sum()/total*100) if total else 0
    ax.set_title(f"{title}\n{total} passes · {acc}% acc",
                 color=C_YELLOW if highlight else "#e6edf3",
                 fontsize=fontsize_title,
                 fontweight="bold" if highlight else "normal", pad=8)

# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════

total_g = sum(1 for s in shots if s["type_id"]==16)
total_p = len(passes)
total_s = len(shots)

st.html(f"""
<div style="
  background: linear-gradient(135deg, #0c0f18 0%, #0f1525 50%, #0a0d15 100%);
  border-bottom: 1px solid #151c28;
  padding: 40px 36px 32px;
  position: relative;
  overflow: hidden;
">
  <!-- decorative gradient blob -->
  <div style="position:absolute;top:-60px;right:-60px;width:300px;height:300px;
       background:radial-gradient(circle,#f59e0b18 0%,transparent 70%);
       pointer-events:none"></div>
  <div style="position:absolute;bottom:-80px;right:200px;width:200px;height:200px;
       background:radial-gradient(circle,#3b82f618 0%,transparent 70%);
       pointer-events:none"></div>

  <div style="display:flex;align-items:flex-end;justify-content:space-between;
       flex-wrap:wrap;gap:24px;position:relative">

    <!-- left: player info -->
    <div>
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
        <div style="width:6px;height:36px;background:linear-gradient(180deg,#f59e0b,#ef4444);
             border-radius:3px"></div>
        <div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:42px;
               font-weight:800;color:#f1f5f9;letter-spacing:-1.5px;line-height:1">
            Beth Mead
          </div>
          <div style="font-size:13px;color:#6b7280;margin-top:4px;letter-spacing:.04em">
            Arsenal WFC &nbsp;·&nbsp; Right Winger &nbsp;·&nbsp;
            <span style="color:#f59e0b">WSL {seasons_available[0].replace("WSL ","") if seasons_available else ""} – {seasons_available[-1].replace("WSL ","") if seasons_available else ""}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- right: career headline stats -->
    <div style="display:flex;gap:32px;flex-wrap:wrap">
      <div style="text-align:center">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:36px;
             font-weight:800;color:#f59e0b;line-height:1">{total_g}</div>
        <div style="font-size:10px;font-weight:600;letter-spacing:.1em;
             text-transform:uppercase;color:#4b5563;margin-top:4px">Career Goals</div>
      </div>
      <div style="width:1px;background:#1e2a36;align-self:stretch"></div>
      <div style="text-align:center">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:36px;
             font-weight:800;color:#3b82f6;line-height:1">{total_p:,}</div>
        <div style="font-size:10px;font-weight:600;letter-spacing:.1em;
             text-transform:uppercase;color:#4b5563;margin-top:4px">Passes</div>
      </div>
      <div style="width:1px;background:#1e2a36;align-self:stretch"></div>
      <div style="text-align:center">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:36px;
             font-weight:800;color:#10b981;line-height:1">{total_s}</div>
        <div style="font-size:10px;font-weight:600;letter-spacing:.1em;
             text-transform:uppercase;color:#4b5563;margin-top:4px">Shots</div>
      </div>
      <div style="width:1px;background:#1e2a36;align-self:stretch"></div>
      <div style="text-align:center">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:36px;
             font-weight:800;color:#a78bfa;line-height:1">{len(seasons_available)}</div>
        <div style="font-size:10px;font-weight:600;letter-spacing:.1em;
             text-transform:uppercase;color:#4b5563;margin-top:4px">Seasons</div>
      </div>
    </div>
  </div>

  <!-- active filter strip -->
  <div style="margin-top:20px;display:flex;align-items:center;gap:8px;flex-wrap:wrap">
    <span style="font-size:10px;font-weight:600;letter-spacing:.1em;
         text-transform:uppercase;color:#374151">Filtered:</span>
    <span style="background:#f59e0b1a;border:1px solid #f59e0b44;border-radius:99px;
         padding:3px 12px;font-size:11px;color:#f59e0b;font-weight:600">
      {season_label()}
    </span>
  </div>
</div>
""")

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER  –  landing page or individual visualisation
# ═══════════════════════════════════════════════════════════════════════════════

DIVIDER = '<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>'

def back_btn():
    """Breadcrumb + back button shown at the top of every viz page."""
    cur = next((v for v in VIZZES if v["id"] == st.session_state.page), None)
    if not cur:
        return
    cat_color = CAT_COLOR.get(cur["cat"], "#f59e0b")
    st.html(f"""
    <div style="display:flex;align-items:center;gap:12px;
         padding:16px 0 0;margin-bottom:4px">
      <span style="font-size:11px;color:#374151">
        <span style="color:#4b5563">Home</span>
        <span style="margin:0 6px;color:#252c3a">›</span>
        <span style="color:{cat_color}">{cur['cat']}</span>
        <span style="margin:0 6px;color:#252c3a">›</span>
        <span style="color:#f1f5f9;font-weight:600">{cur['title']}</span>
      </span>
    </div>""")
    if st.button("← Back to Home", key="back_top"):
        nav("home")
    st.markdown(DIVIDER, unsafe_allow_html=True)

# ── substitution impact model helpers ────────────────────────────────────────

@st.cache_data
def _load_xg_csvs():
    base = os.path.dirname(__file__)
    frames = []
    for fp in glob.glob(os.path.join(base, "**", "xgCSV", "*.csv"), recursive=True):
        try:
            df = pd.read_csv(fp)
            df["season"] = os.path.basename(os.path.dirname(os.path.dirname(fp)))
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    full.columns = [c.strip() for c in full.columns]
    return full


def _mc_net(on_mu, off_mu, sig_on, sig_off, n, mtype):
    rng = np.random.default_rng()
    if mtype == "poisson":
        sim_on  = rng.poisson(np.clip(rng.normal(on_mu,  sig_on,  n), 0, None))
        sim_off = rng.poisson(np.clip(rng.normal(off_mu, sig_off, n), 0, None))
    else:
        sim_on  = rng.normal(on_mu,  sig_on,  n)
        sim_off = rng.normal(off_mu, sig_off, n)
    return sim_on - sim_off


def _pct(arr):
    return {k: float(v) for k, v in zip(
        ["p10","p25","p50","p75","p90","mean","pos_pct"],
        [np.percentile(arr,10), np.percentile(arr,25), np.percentile(arr,50),
         np.percentile(arr,75), np.percentile(arr,90), np.mean(arr), np.mean(arr>0)*100])}


@st.cache_data
def _load_xt_vaep_summary():
    fp = os.path.join(os.path.dirname(__file__), "player_vaep_xt_summary.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()
    return pd.read_csv(fp)


@st.cache_data
def _load_match_metrics():
    fp = os.path.join(os.path.dirname(__file__), "all_match_metrics.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()
    return pd.read_csv(fp, low_memory=False)


@st.cache_data
def _load_onoff():
    fp = os.path.join(os.path.dirname(__file__), "player_onoff_xt_vaep.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()
    return pd.read_csv(fp)


@st.cache_data
def _load_xt_vaep_actions():
    fp = os.path.join(os.path.dirname(__file__), "vaep_xt_results.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()
    return pd.read_csv(fp, usecols=["season","player","team","action_type","timeMin","x","y","end_x","end_y","xt_value","vaep_value"])


@st.cache_data
def _load_player_metrics():
    fp = os.path.join(os.path.dirname(__file__), "all_player_metrics.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()
    return pd.read_csv(fp, low_memory=False)


# ── LANDING PAGE ──────────────────────────────────────────────────────────────

if st.session_state.page == "home":
    # landing page cards
    by_cat = {}
    for v in VIZZES:
        by_cat.setdefault(v["cat"], []).append(v)

    for cat in CAT_ORDER:
        cat_vizzes = by_cat.get(cat, [])
        if not cat_vizzes:
            continue
        color = CAT_COLOR[cat]
        st.html(f"""
        <div style="margin:32px 0 14px">
          <div style="display:flex;align-items:center;gap:10px">
            <div style="width:3px;height:18px;background:{color};border-radius:2px"></div>
            <span style="font-family:'Space Grotesk',sans-serif;font-size:11px;
                 font-weight:700;text-transform:uppercase;letter-spacing:.12em;
                 color:{color}">{cat}</span>
          </div>
        </div>""")

        cols = st.columns(min(len(cat_vizzes), 4))
        for col, v in zip(cols, cat_vizzes):
            with col:
                st.html(f"""
                <div style="background:linear-gradient(160deg,#0f1420,#0c111d);
                     border:1px solid #1a2235;border-top:2px solid {color};
                     border-radius:12px;padding:20px 18px 14px;height:100%;
                     min-height:130px">
                  <div style="font-size:26px;margin-bottom:10px">{v['icon']}</div>
                  <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;
                       font-weight:700;color:#f1f5f9;margin-bottom:6px">{v['title']}</div>
                  <div style="font-size:11px;color:#4b5563;line-height:1.5">
                    {v['desc']}</div>
                </div>""")
                if st.button(f"Open  →", key=f"card_{v['id']}", use_container_width=True):
                    nav(v["id"])

    st.html("""
    <div style="margin-top:48px;padding:20px;background:#0c0f18;
         border:1px solid #151c28;border-radius:12px;
         display:flex;align-items:center;gap:16px">
      <div style="font-size:28px">📊</div>
      <div>
        <div style="font-size:13px;font-weight:600;color:#e8eaf0;margin-bottom:3px">
          19 visualisations across 6 categories</div>
        <div style="font-size:11px;color:#4b5563">
          Opta event data · WSL 2015–2026 · Beth Mead · Arsenal WFC
        </div>
      </div>
    </div>""")

# ═══════════════════════════════════════════════════════════════════════════════
# 1 – PASS MAP
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "pass_map":
    back_btn()

    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    pitch,fig,ax = make_pitch(figsize=(16,10))
    ORDER={"unsuccessful":0,"successful":1,"progressive":2,"key":3}
    for p in sorted(fp, key=lambda p:ORDER[pass_type(p)]):
        pt=pass_type(p); color=PASS_COLOR[pt]
        alpha=0.28 if pt=="unsuccessful" else (0.82 if pt in ("key","progressive") else 0.52)
        pitch.arrows(p["x"],p["y"],p["end_x"],p["end_y"],ax=ax,color=color,
                     alpha=alpha,width=1.1,headwidth=4,headlength=4)
    items=[]
    if show_succ:   items.append((PASS_COLOR["successful"],  "Successful"))
    if show_unsucc: items.append((PASS_COLOR["unsuccessful"],"Unsuccessful"))
    if show_prog:   items.append((PASS_COLOR["progressive"], "Progressive"))
    if show_key:    items.append((PASS_COLOR["key"],         "Shot Assist"))
    if items: add_legend(ax,items)
    add_title(fig,"Pass Map  ·  B. Mead",
              f"{season_label()}  ·  {len(fp):,} passes  ·  "
              f"{sum(1 for p in fp if p['progressive']):,} progressive  ·  "
              f"{sum(1 for p in fp if p['key_pass']):,} shot assists")
    plt.tight_layout(rect=[0,0,1,.94]); st.pyplot(fig,width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 2 – SHOT MAP
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "shot_map":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    pitch=VerticalPitch(pitch_type="opta",pitch_color=PITCH_BG,line_color=PITCH_LINE,
                        linewidth=1.0,goal_type="box",half=True,line_zorder=2)
    fig,ax=pitch.draw(figsize=(10,8)); fig.patch.set_facecolor(FIG_BG)
    for tid in (13,14,15,16):
        color,marker,size,alpha,zorder=SHOT_STYLE[tid]
        sub=[s for s in fs if s["type_id"]==tid]
        if sub:
            pitch.scatter([s["x"] for s in sub],[s["y"] for s in sub],ax=ax,
                          c=color,marker=marker,s=size,alpha=alpha,zorder=zorder,
                          edgecolors="white",linewidths=.3)
    counts={tid:sum(1 for s in fs if s["type_id"]==tid) for tid in (16,15,14,13)}
    add_legend(ax,[(SHOT_STYLE[16][0],f"Goal ({counts[16]})"),
                   (SHOT_STYLE[15][0],f"On Target ({counts[15]})"),
                   (SHOT_STYLE[14][0],f"Post ({counts[14]})"),
                   (SHOT_STYLE[13][0],f"Off Target ({counts[13]})")], loc="lower center")
    add_title(fig,"Shot Map  ·  B. Mead",
              f"{season_label()}  ·  {len(fs)} shots  ·  {counts[16]} goals")
    plt.tight_layout(rect=[0,0,1,.94]); st.pyplot(fig,width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 3 – HEAT MAP
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "heat_map":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts); ft=f_season(touches)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    pitch=Pitch(pitch_type="opta",pitch_color=PITCH_BG,line_color=PITCH_LINE,
                linewidth=1.2,goal_type="box",line_zorder=2)
    fig,ax=pitch.draw(figsize=(16,10)); fig.patch.set_facecolor(FIG_BG)
    xs=[t["x"] for t in ft]; ys=[t["y"] for t in ft]
    if xs:
        cmap=LinearSegmentedColormap.from_list(
            "heat",[PITCH_BG,"#1a3a2a","#3fb950","#e3b341","#f78166","#ffffff"],N=256)
        pitch.kdeplot(xs,ys,ax=ax,cmap=cmap,fill=True,levels=100,alpha=.88,bw_adjust=.65,zorder=1)
    add_title(fig,"Heat Map  ·  B. Mead  (all actions)",
              f"{season_label()}  ·  {len(ft):,} actions")
    plt.tight_layout(rect=[0,0,1,.94]); st.pyplot(fig,width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 4 – TERRITORY MAP
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "territory":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts); ft=f_season(touches)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    pitch=Pitch(pitch_type="opta",pitch_color=PITCH_BG,line_color=PITCH_LINE,
                linewidth=1.2,goal_type="box",line_zorder=2)
    fig,ax=pitch.draw(figsize=(16,10)); fig.patch.set_facecolor(FIG_BG)
    if ft:
        xs=np.array([t["x"] for t in ft]); ys=np.array([t["y"] for t in ft])
        bs=pitch.bin_statistic(xs,ys,statistic="count",bins=(12,8))
        cm=LinearSegmentedColormap.from_list("terr",
           ["#0d1117","#0d2137","#1a4a7a",C_BLUE,C_YELLOW],N=256)
        pitch.heatmap(bs,ax=ax,cmap=cm,alpha=.85,zorder=1)
        pitch.label_heatmap(bs,ax=ax,color="#e6edf3",fontsize=7,
                            ha="center",va="center",str_format="{:.0f}",zorder=3)
        sm=plt.cm.ScalarMappable(cmap=cm,norm=plt.Normalize(0,max(bs["statistic"].max(),1)))
        sm.set_array([])
        cbar=fig.colorbar(sm,ax=ax,orientation="vertical",fraction=.018,pad=.02)
        cbar.ax.yaxis.set_tick_params(color="#8b949e",labelsize=8)
        cbar.outline.set_edgecolor("#30363d")
        plt.setp(plt.getp(cbar.ax.axes,"yticklabels"),color="#8b949e")
    add_title(fig,"Territory Map  ·  B. Mead",
              f"{season_label()}  ·  action count per zone  ·  {len(ft):,} total")
    plt.tight_layout(rect=[0,0,1,.94]); st.pyplot(fig,width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 5 – PASS NETWORK
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "pass_net":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    # net tuples: (passer, recipient, season) and (player, x, y, season)
    edges_f=[(p,r) for p,r,s in net_edges if s in selected_seasons]
    pos_f  =[(pl,x,y) for pl,x,y,s in net_pos if s in selected_seasons]
    if not edges_f:
        st.info("No pass network data for selected season(s).")
    else:
        avg_pos=defaultdict(lambda:{"x":[],"y":[]})
        for pl,x,y in pos_f: avg_pos[pl]["x"].append(x); avg_pos[pl]["y"].append(y)
        avg_pos={pl:(np.mean(v["x"]),np.mean(v["y"])) for pl,v in avg_pos.items()}
        pair_counts=Counter()
        for p,r in edges_f: pair_counts[tuple(sorted([p,r]))]+=1
        player_pass=Counter(p for p,r in edges_f)
        players=[p for p in avg_pos if player_pass[p]>0]
        min_edge=max(2,int(np.percentile(list(pair_counts.values()),30)))
        pitch=Pitch(pitch_type="opta",pitch_color=PITCH_BG,line_color="#2d333b",
                    linewidth=.8,goal_type="box",line_zorder=1)
        fig,ax=pitch.draw(figsize=(16,10)); fig.patch.set_facecolor(FIG_BG)
        mx_e=max(pair_counts.values()) or 1; mx_n=max(player_pass[p] for p in players) if players else 1
        for (p1,p2),cnt in pair_counts.items():
            if cnt<min_edge or p1 not in avg_pos or p2 not in avg_pos: continue
            x1,y1=avg_pos[p1]; x2,y2=avg_pos[p2]
            color=C_BLUE if "B. Mead" in (p1,p2) else "#484f58"
            ax.plot([x1,x2],[y1,y2],color=color,
                    linewidth=.8+5*(cnt/mx_e),alpha=.25+.55*(cnt/mx_e),
                    solid_capstyle="round",zorder=2)
        for pl in players:
            if pl not in avg_pos: continue
            x,y=avg_pos[pl]; is_mead=(pl=="B. Mead")
            ax.scatter(x,y,s=100+700*(player_pass[pl]/mx_n),
                       c=C_YELLOW if is_mead else "#c9d1d9",
                       zorder=4,edgecolors="#0d1117",linewidths=2.5 if is_mead else 1.0)
            label=pl.split(". ")[-1] if ". " in pl else pl
            ax.text(x,y+3.5,label,ha="center",va="bottom",
                    fontsize=9.5 if is_mead else 7.5,
                    fontweight="bold" if is_mead else "normal",
                    color=C_YELLOW if is_mead else "#e6edf3",zorder=5,
                    bbox=dict(boxstyle="round,pad=.2",facecolor="#0d1117",alpha=.6,edgecolor="none"))
        add_legend(ax,[(C_YELLOW,"B. Mead"),(("#c9d1d9"),"Teammate"),
                       (C_BLUE,"Mead connection"),("#484f58","Other")])
        top=max([(p,c) for (p1,p2),c in pair_counts.items()
                 for p in ([p1] if p2=="B. Mead" else ([p2] if p1=="B. Mead" else []))],
                key=lambda x:x[1],default=("—",0))
        add_title(fig,"Pass Network  ·  B. Mead",
                  f"{season_label()}  ·  node size = passes  ·  top partner: {top[0]} ({top[1]})")
        plt.tight_layout(rect=[0,0,1,.94]); st.pyplot(fig,width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 6 – PASS SONARS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "pass_sonar":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    # sonar tuples: (player, angle, dist, outcome, season)
    sp_f=[{"player":pl,"angle":a,"dist":d,"outcome":o,"season":s}
          for pl,a,d,o,s in sonar_passes if s in selected_seasons]
    if not sp_f:
        st.info("No sonar data for selected season(s).")
    else:
        mead_sp=[p for p in sp_f if p["player"]=="B. Mead"]
        fig_m=plt.figure(figsize=(6,6),facecolor=FIG_BG)
        ax_m=fig_m.add_subplot(111,projection="polar")
        draw_sonar(ax_m,mead_sp,"B. Mead",highlight=True,fontsize_title=12)
        ax_m.legend(handles=[mpatches.Patch(color=C_YELLOW,label="Successful"),
                              mpatches.Patch(color=C_ORANGE,label="Unsuccessful"),
                              mpatches.Patch(color=C_BLUE,  label="Avg distance")],
                    loc="lower center",bbox_to_anchor=(.5,-.18),ncol=3,
                    fontsize=7.5,framealpha=0,labelcolor="#c9d1d9",handlelength=1.2)
        fig_m.suptitle(f"Pass Sonar  ·  B. Mead  ·  {season_label()}\n"
                       "→ Forward  ← Backward  ↑↓ Wide",color="#8b949e",fontsize=8,y=.02)
        plt.tight_layout()
        _,mid,_=st.columns([1,2,1])
        with mid: st.pyplot(fig_m,width="stretch")
        plt.close(fig_m)
        st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
        edges_f=[(p,r) for p,r,s in net_edges if s in selected_seasons]
        pair_counts=Counter(tuple(sorted([p,r])) for p,r in edges_f)
        top_p=[]
        for (p1,p2),_ in pair_counts.most_common():
            for p in ([p1] if p2=="B. Mead" else ([p2] if p1=="B. Mead" else [])):
                if p not in top_p: top_p.append(p)
        top_p=[p for p in top_p if p!="B. Mead"][:8]
        if top_p:
            st.markdown("<div style='font-size:13px;font-weight:700;color:#8b949e;"
                        "text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px'>"
                        "Top Pass Partners — Direction Wheels</div>",unsafe_allow_html=True)
            for row_s in [top_p[i:i+4] for i in range(0,len(top_p),4)]:
                cols=st.columns(4)
                for col,partner in zip(cols,row_s):
                    with col:
                        pp=[p for p in sp_f if p["player"]==partner]
                        fig_p,ax_p=plt.subplots(figsize=(3.2,3.2),
                            subplot_kw={"projection":"polar"},facecolor=FIG_BG)
                        sn=partner.split(". ")[-1] if ". " in partner else partner
                        ec=pair_counts.get(tuple(sorted([partner,"B. Mead"])),0)
                        draw_sonar(ax_p,pp,f"{sn}\n({ec} w/ Mead)",fontsize_title=8)
                        plt.tight_layout(); st.pyplot(fig_p,width="stretch"); plt.close(fig_p)

# ═══════════════════════════════════════════════════════════════════════════════
# 7 – DRIBBLE MAP
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "dribbles":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    dr=f_season(dribbles)
    succ_d=sum(1 for d in dr if d["outcome"]==1)
    total_d=len(dr)
    c1,c2,c3=st.columns(3)
    c1.metric("Total Take-Ons", f"{total_d:,}")
    c2.metric("Successful",     f"{succ_d:,}")
    c3.metric("Success Rate",   f"{round(succ_d/total_d*100,1) if total_d else 0}%")
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    pitch,fig,ax=make_pitch(figsize=(16,10))
    for d in dr:
        color=C_GREEN if d["outcome"]==1 else C_ORANGE
        alpha=.80 if d["outcome"]==1 else .50
        ax.scatter(d["x"],d["y"],c=color,s=80,alpha=alpha,
                   zorder=3,edgecolors="#0d1117",linewidths=.5)
    add_legend(ax,[(C_GREEN,f"Successful ({succ_d})"),(C_ORANGE,f"Unsuccessful ({total_d-succ_d})")])
    add_title(fig,"Dribble / Take-On Map  ·  B. Mead",
              f"{season_label()}  ·  {total_d} take-ons  ·  {round(succ_d/total_d*100,1) if total_d else 0}% success rate")
    plt.tight_layout(rect=[0,0,1,.94]); st.pyplot(fig,width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 8 – CROSSING MAP
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "crossings":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    # crosses: passes from wide channels (y<25 or y>75) into final third (end_x>67)
    all_p=f_season(passes)
    crosses=[p for p in all_p if (p["y"]<25 or p["y"]>75) and p["end_x"]>67]
    succ_c=sum(1 for c in crosses if c["outcome"]==1)
    c1,c2,c3=st.columns(3)
    c1.metric("Crosses",    f"{len(crosses):,}")
    c2.metric("Successful", f"{succ_c:,}")
    c3.metric("Success Rate",f"{round(succ_c/len(crosses)*100,1) if crosses else 0}%")
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    pitch,fig,ax=make_pitch(figsize=(16,10))
    for c in crosses:
        color=C_BLUE if c["outcome"]==1 else C_ORANGE
        alpha=.75 if c["outcome"]==1 else .40
        pitch.arrows(c["x"],c["y"],c["end_x"],c["end_y"],ax=ax,
                     color=color,alpha=alpha,width=1.2,headwidth=4,headlength=4)
    add_legend(ax,[(C_BLUE,f"Successful ({succ_c})"),(C_ORANGE,f"Unsuccessful ({len(crosses)-succ_c})")])
    add_title(fig,"Crossing Map  ·  B. Mead",
              f"{season_label()}  ·  passes from wide channels into final third  ·  {len(crosses)} crosses")
    plt.tight_layout(rect=[0,0,1,.94]); st.pyplot(fig,width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# 9 – HALF-SPACE PASSES
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "half_space":
    back_btn()
    fp = f_passes(passes); fs = f_shots(shots); fd = f_def(def_acts)
    metric_row(fp, fs, fd)
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>', unsafe_allow_html=True)

    all_p = f_season(passes)

    # Classify passes
    hs_passes = [p for p in all_p if in_halfspace(p["y"]) or in_halfspace(p["end_y"])]
    hs_from   = [p for p in all_p if in_halfspace(p["y"])]
    hs_into   = [p for p in all_p if not in_halfspace(p["y"]) and in_halfspace(p["end_y"])]
    hs_thru   = [p for p in all_p if in_halfspace(p["y"]) and in_halfspace(p["end_y"])]

    succ_hs = sum(1 for p in hs_passes if p["outcome"] == 1)

    # sub-metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Half-Space Passes",  f"{len(hs_passes):,}")
    c2.metric("From Half-Space",    f"{len(hs_from):,}")
    c3.metric("Into Half-Space",    f"{len(hs_into):,}")
    c4.metric("HS Accuracy",        f"{round(succ_hs/len(hs_passes)*100,1) if hs_passes else 0}%")

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:16px 0 24px"></div>', unsafe_allow_html=True)

    # ── filter toggle ─────────────────────────────────────────────────────
    hs_view = st.radio(
        "Show passes",
        ["All half-space", "From half-space", "Into half-space", "Through half-space"],
        horizontal=True,
    )
    view_map = {
        "All half-space":       hs_passes,
        "From half-space":      hs_from,
        "Into half-space":      hs_into,
        "Through half-space":   hs_thru,
    }
    view_passes = view_map[hs_view]

    # ── pitch ─────────────────────────────────────────────────────────────
    pitch = Pitch(pitch_type="opta", pitch_color=PITCH_BG, line_color=PITCH_LINE,
                  linewidth=1.0, goal_type="box", line_zorder=3)
    fig, ax = pitch.draw(figsize=(16, 10))
    fig.patch.set_facecolor(FIG_BG)

    # shade half-space zones
    for y0, y1 in (HS_L, HS_R):
        ax.axhspan(y0, y1, color="#f59e0b", alpha=0.06, zorder=1)
        ax.axhspan(y0, y1, color="#f59e0b", alpha=0.0, zorder=1)
        # dashed boundary lines
        for y_line in (y0, y1):
            ax.axhline(y_line, color="#f59e0b", linewidth=0.7,
                       linestyle="--", alpha=0.35, zorder=2)

    # zone labels
    for y_mid, label in [((HS_L[0]+HS_L[1])/2, "LEFT\nHALF-SPACE"),
                          ((HS_R[0]+HS_R[1])/2, "RIGHT\nHALF-SPACE")]:
        ax.text(2, y_mid, label, ha="left", va="center",
                fontsize=6.5, color="#f59e0b", alpha=0.55,
                fontweight="700", fontfamily="monospace")

    # draw arrows
    ORDER = {"unsuccessful": 0, "successful": 1, "progressive": 2, "key": 3}
    for p in sorted(view_passes, key=lambda p: ORDER[pass_type(p)]):
        pt    = pass_type(p)
        color = PASS_COLOR[pt]
        alpha = 0.25 if pt == "unsuccessful" else (0.85 if pt in ("key","progressive") else 0.55)
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"],
                     ax=ax, color=color, alpha=alpha,
                     width=1.2, headwidth=4, headlength=4, zorder=4)

    items = [
        (PASS_COLOR["successful"],   f"Successful ({sum(1 for p in view_passes if p['outcome']==1 and not p['key_pass'] and not p['progressive'])})"),
        (PASS_COLOR["unsuccessful"], f"Unsuccessful ({sum(1 for p in view_passes if p['outcome']==0)})"),
        (PASS_COLOR["progressive"],  f"Progressive ({sum(1 for p in view_passes if p['progressive'])})"),
        (PASS_COLOR["key"],          f"Shot Assist ({sum(1 for p in view_passes if p['key_pass'])})"),
        ("#f59e0b",                  "Half-Space Zone"),
    ]
    add_legend(ax, items)

    succ_v = sum(1 for p in view_passes if p["outcome"]==1)
    total_v = len(view_passes)
    add_title(
        fig, "Half-Space Passes  ·  B. Mead",
        f"{season_label()}  ·  {hs_view}  ·  "
        f"{total_v:,} passes  ·  {round(succ_v/total_v*100,1) if total_v else 0}% accuracy"
    )
    plt.tight_layout(rect=[0, 0, 1, .94])
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # ── zone split breakdown ─────────────────────────────────────────────
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:24px 0 20px"></div>', unsafe_allow_html=True)
    st.markdown("**Zone breakdown**", unsafe_allow_html=False)

    bc1, bc2, bc3 = st.columns(3)
    for col, zone_passes, label in [
        (bc1, hs_from, "Passes FROM half-space"),
        (bc2, hs_into, "Passes INTO half-space"),
        (bc3, hs_thru, "Passes THROUGH half-space"),
    ]:
        total_z = len(zone_passes)
        succ_z  = sum(1 for p in zone_passes if p["outcome"] == 1)
        prog_z  = sum(1 for p in zone_passes if p["progressive"])
        key_z   = sum(1 for p in zone_passes if p["key_pass"])
        col.markdown(
            f"<div style='background:#0f1420;border:1px solid #1a2235;"
            f"border-left:3px solid #f59e0b;border-radius:10px;"
            f"padding:16px 18px;margin-bottom:8px'>"
            f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:.1em;color:#f59e0b;margin-bottom:10px'>{label}</div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px'>"
            f"<div><div style='font-size:22px;font-weight:800;color:#f1f5f9'>{total_z}</div>"
            f"<div style='font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:.06em'>Total</div></div>"
            f"<div><div style='font-size:22px;font-weight:800;color:#10b981'>"
            f"{round(succ_z/total_z*100,1) if total_z else 0}%</div>"
            f"<div style='font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:.06em'>Accuracy</div></div>"
            f"<div><div style='font-size:22px;font-weight:800;color:#3b82f6'>{prog_z}</div>"
            f"<div style='font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:.06em'>Progressive</div></div>"
            f"<div><div style='font-size:22px;font-weight:800;color:#a78bfa'>{key_z}</div>"
            f"<div style='font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:.06em'>Shot Assists</div></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

# 10 – SHOT ZONES
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "shot_zones":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    pitch=VerticalPitch(pitch_type="opta",pitch_color=PITCH_BG,line_color=PITCH_LINE,
                        linewidth=1.0,goal_type="box",half=True,line_zorder=2)
    fig,ax=pitch.draw(figsize=(10,9)); fig.patch.set_facecolor(FIG_BG)
    fs_all=f_season(shots)
    if fs_all:
        xs=np.array([s["x"] for s in fs_all]); ys=np.array([s["y"] for s in fs_all])
        bs=pitch.bin_statistic(xs,ys,statistic="count",bins=(6,5))
        cm=LinearSegmentedColormap.from_list("sz",
           ["#0d1117","#1a3a2a",C_GREEN,C_YELLOW,C_ORANGE],N=256)
        pitch.heatmap(bs,ax=ax,cmap=cm,alpha=.75,zorder=1)
        pitch.label_heatmap(bs,ax=ax,color="#e6edf3",fontsize=9,
                            ha="center",va="center",str_format="{:.0f}",zorder=3)
    # overlay individual shots
    for tid in (13,14,15,16):
        color,marker,size,alpha,zorder=SHOT_STYLE[tid]
        sub=[s for s in fs if s["type_id"]==tid]
        if sub:
            pitch.scatter([s["x"] for s in sub],[s["y"] for s in sub],
                          ax=ax,c=color,marker=marker,s=size*.7,
                          alpha=alpha,zorder=zorder+1,edgecolors="#0d1117",linewidths=.3)
    counts={tid:sum(1 for s in fs if s["type_id"]==tid) for tid in (16,15,14,13)}
    add_legend(ax,[(SHOT_STYLE[16][0],f"Goal ({counts[16]})"),
                   (SHOT_STYLE[15][0],f"On Target ({counts[15]})"),
                   (SHOT_STYLE[14][0],f"Post ({counts[14]})"),
                   (SHOT_STYLE[13][0],f"Off Target ({counts[13]})")],loc="lower center")
    add_title(fig,"Shot Zones  ·  B. Mead",
              f"{season_label()}  ·  zone count + individual shots overlaid")
    plt.tight_layout(rect=[0,0,1,.94]); st.pyplot(fig,width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 10 – CAREER TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "career":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    career = {}
    for s in seasons_available:
        sp=[p for p in passes   if p["season"]==s]
        ss=[sh for sh in shots  if sh["season"]==s]
        career[s]={
            "Goals":       sum(1 for sh in ss if sh["type_id"]==16),
            "Shot Assists":sum(1 for p  in sp if p["key_pass"]),
            "Shots":       len(ss),
            "Passes":      len(sp),
            "Progressive": sum(1 for p  in sp if p["progressive"]),
            "Dribbles":    sum(1 for d  in dribbles if d["season"]==s),
        }
    slabels=[s.replace("WSL ","") for s in seasons_available]
    fig,axes=plt.subplots(2,3,figsize=(16,8),facecolor=FIG_BG)
    fig.patch.set_facecolor(FIG_BG)
    metrics_career=["Goals","Shot Assists","Shots","Passes","Progressive","Dribbles"]
    colors_career=[C_YELLOW,C_BLUE,C_ORANGE,C_GREEN,C_GREEN,C_PURPLE]
    for ax,(metric,color) in zip(axes.flat,zip(metrics_career,colors_career)):
        vals=[career[s][metric] for s in seasons_available]
        ax.fill_between(range(len(slabels)),vals,color=color,alpha=.25)
        ax.plot(range(len(slabels)),vals,color=color,linewidth=2.5,marker="o",
                markersize=6,markerfacecolor=FIG_BG,markeredgewidth=2)
        for i,v in enumerate(vals):
            ax.text(i,v+max(vals)*.04,str(v),ha="center",color=color,
                    fontsize=8,fontweight="bold")
        ax.set_xticks(range(len(slabels))); ax.set_xticklabels(slabels,rotation=45,
                    ha="right",color="#8b949e",fontsize=7.5)
        ax.set_facecolor("#161b22"); ax.set_title(metric,color=color,fontsize=11,fontweight="700",pad=8)
        ax.tick_params(colors="#484f58"); ax.yaxis.set_tick_params(labelcolor="#8b949e",labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#30363d"); sp.set_linewidth(.5)
        ax.grid(axis="y",color="#21262d",linewidth=.5,zorder=0)
    fig.suptitle(f"Career Timeline  ·  B. Mead  ·  {season_label()}",
                 color="#e6edf3",fontsize=14,fontweight="800",y=.98)
    plt.tight_layout(rect=[0,0,1,.95]); st.pyplot(fig,width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 11 – MATCH BY MATCH
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "match_stats":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    mr_f=[r for r in match_rows if r["season"] in selected_seasons]
    mr_f.sort(key=lambda r:r["date"])
    if not mr_f:
        st.info("No match data for selected season(s).")
    else:
        import pandas as pd
        stat_choice=st.selectbox("Metric",
            ["passes","succ_passes","shots","goals","key_passes","prog_passes",
             "dribbles","succ_dribbles","tackles","interceptions"],
            format_func=lambda x: x.replace("_"," ").title())
        fig,ax=plt.subplots(figsize=(max(12,len(mr_f)*.45),5),facecolor=FIG_BG)
        ax.set_facecolor("#161b22")
        vals=[r[stat_choice] for r in mr_f]
        labels=[f"{r['date'][5:]}  {r['opponent'][:12]}" for r in mr_f]
        colors=[C_YELLOW if v==max(vals) else C_BLUE for v in vals]
        bars=ax.bar(range(len(vals)),vals,color=colors,alpha=.85,width=.7,edgecolor="none")
        avg=np.mean(vals)
        ax.axhline(avg,color=C_ORANGE,linewidth=1.5,linestyle="--",label=f"Avg {avg:.1f}",alpha=.8)
        ax.legend(fontsize=9,framealpha=0,labelcolor="#c9d1d9")
        for i,v in enumerate(vals):
            if v>0: ax.text(i,v+.05,str(v),ha="center",va="bottom",color="#e6edf3",fontsize=6.5)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels,rotation=55,
                ha="right",color="#8b949e",fontsize=6.5)
        ax.set_title(f"{stat_choice.replace('_',' ').title()} per Match  ·  B. Mead  ·  {season_label()}",
                     color="#e6edf3",fontsize=12,fontweight="800",pad=10)
        ax.tick_params(colors="#484f58"); ax.yaxis.set_tick_params(labelcolor="#8b949e",labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#30363d"); sp.set_linewidth(.5)
        ax.grid(axis="y",color="#21262d",linewidth=.5,zorder=0)
        plt.tight_layout(); st.pyplot(fig,width="stretch"); plt.close(fig)

        # data table
        df=pd.DataFrame(mr_f)[["date","season","opponent",
            "passes","succ_passes","shots","goals","key_passes","prog_passes",
            "dribbles","succ_dribbles","tackles","interceptions"]]
        df.columns=[c.replace("_"," ").title() for c in df.columns]
        st.dataframe(df,width="stretch",hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 12 – RADAR / PIZZA CHART
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "radar":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;color:#8b949e;margin-bottom:12px'>"
                "Each season normalised to Mead's career best (100 = personal best). "
                "Select up to 3 seasons to compare.</div>",unsafe_allow_html=True)

    compare_seasons=st.multiselect("Compare seasons",seasons_available,
                                   seasons_available[-3:] if len(seasons_available)>=3 else seasons_available,
                                   max_selections=3,key="radar_sel")
    if not compare_seasons:
        st.info("Select at least one season.")
    else:
        radar_metrics=["Passes","Pass Acc%","Progressive","Shot Assists","Shots",
                       "Goals","Shot Acc%","Tackles","Interceptions","Dribbles"]
        season_vals={}
        for s in seasons_available:
            sp=[p for p in passes   if p["season"]==s]
            ss=[sh for sh in shots  if sh["season"]==s]
            sd=[d  for d  in def_acts if d["season"]==s]
            sdr=[d for d in dribbles  if d["season"]==s]
            tp=len(sp); ts=len(ss)
            season_vals[s]=[
                tp,
                round(sum(1 for p in sp if p["outcome"]==1)/tp*100,1) if tp else 0,
                sum(1 for p in sp if p["progressive"]),
                sum(1 for p in sp if p["key_pass"]),
                ts,
                sum(1 for sh in ss if sh["type_id"]==16),
                round(sum(1 for sh in ss if sh["type_id"] in (15,16))/ts*100,1) if ts else 0,
                sum(1 for d in sd if d["action"]=="Tackle"),
                sum(1 for d in sd if d["action"]=="Interception"),
                len(sdr),
            ]
        max_v=[max(season_vals[s][i] for s in seasons_available) or 1 for i in range(len(radar_metrics))]
        N=len(radar_metrics)
        angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=[angles[0]]
        RADAR_COLORS=[C_YELLOW,C_BLUE,C_GREEN]
        fig=plt.figure(figsize=(8,8),facecolor=FIG_BG)
        ax=fig.add_subplot(111,projection="polar"); ax.set_facecolor("#161b22")
        # background rings
        for r in [.25,.5,.75,1.0]:
            ax.plot(angles,[r*100]*len(angles),color="#30363d",linewidth=.5,zorder=0)
            ax.fill(angles,[r*100]*len(angles),color="#161b22",zorder=0)
        ax.fill(angles,[100]*len(angles),color="#21262d",alpha=.5,zorder=0)
        for i,s in enumerate(compare_seasons):
            vals=[season_vals[s][j]/max_v[j]*100 for j in range(N)]; vals+=[vals[0]]
            color=RADAR_COLORS[i]
            ax.plot(angles,vals,color=color,linewidth=2.2,zorder=3)
            ax.fill(angles,vals,color=color,alpha=.15,zorder=2)
            ax.scatter(angles[:-1],vals[:-1],s=60,color=color,zorder=5,
                       edgecolors=FIG_BG,linewidths=1.5)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics,color="#c9d1d9",fontsize=9,fontweight="600")
        ax.set_yticks([]); ax.set_ylim(0,115); ax.spines["polar"].set_color("#30363d")
        ax.grid(False)
        # ring labels
        for r,lbl in [(25,"25%"),(50,"50%"),(75,"75%"),(100,"100%")]:
            ax.text(0,r+2,lbl,ha="center",va="bottom",color="#484f58",fontsize=7)
        ax.legend(handles=[mpatches.Patch(color=RADAR_COLORS[i],label=s.replace("WSL ",""))
                           for i,s in enumerate(compare_seasons)],
                  loc="lower center",bbox_to_anchor=(.5,-.12),ncol=len(compare_seasons),
                  fontsize=9,framealpha=0,labelcolor="#c9d1d9",handlelength=1.5)
        fig.suptitle(f"Performance Radar  ·  B. Mead\n100 = career best season",
                     color="#e6edf3",fontsize=13,fontweight="800",y=.98)
        plt.tight_layout(); st.pyplot(fig,width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 13 – STYLE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "style":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    all_p=f_season(passes); all_s=f_season(shots)
    fig=plt.figure(figsize=(16,6),facecolor=FIG_BG)
    gs=gridspec.GridSpec(1,3,figure=fig,wspace=.35)

    # ── pass length histogram ──────────────────────────────────────────────
    ax1=fig.add_subplot(gs[0]); ax1.set_facecolor("#161b22")
    dists=[p["dist"] for p in all_p]
    if dists:
        bins=np.linspace(0,max(dists)+1,25)
        succ_d=[p["dist"] for p in all_p if p["outcome"]==1]
        fail_d=[p["dist"] for p in all_p if p["outcome"]==0]
        ax1.hist(fail_d,bins=bins,color=C_ORANGE,alpha=.6,label="Unsuccessful",edgecolor="none")
        ax1.hist(succ_d,bins=bins,color=C_YELLOW,alpha=.7,label="Successful",edgecolor="none")
        ax1.axvline(np.mean(dists),color=C_BLUE,linewidth=1.5,linestyle="--",
                    label=f"Mean {np.mean(dists):.1f}y")
    ax1.set_title("Pass Length Distribution",color="#e6edf3",fontsize=11,fontweight="700",pad=8)
    ax1.set_xlabel("Distance (pitch %)",color="#8b949e",fontsize=8)
    ax1.set_ylabel("Count",color="#8b949e",fontsize=8)
    ax1.legend(fontsize=8,framealpha=0,labelcolor="#c9d1d9")
    ax1.tick_params(colors="#484f58"); ax1.yaxis.set_tick_params(labelcolor="#8b949e")
    ax1.xaxis.set_tick_params(labelcolor="#8b949e")
    for sp in ax1.spines.values(): sp.set_color("#30363d")
    ax1.set_facecolor("#161b22"); ax1.grid(axis="y",color="#21262d",linewidth=.5)

    # ── shooting foot donut ───────────────────────────────────────────────
    ax2=fig.add_subplot(gs[1]); ax2.set_facecolor(FIG_BG)
    foot_counts=Counter(s["foot"] for s in all_s)
    if foot_counts:
        labels_f=list(foot_counts.keys()); sizes=list(foot_counts.values())
        donut_colors=[C_YELLOW,C_BLUE,C_GREEN,C_ORANGE,C_PURPLE][:len(labels_f)]
        wedges,_=ax2.pie(sizes,labels=None,colors=donut_colors,
                         startangle=90,wedgeprops=dict(width=.5,edgecolor=FIG_BG,linewidth=2))
        for i,(lbl,sz) in enumerate(zip(labels_f,sizes)):
            pct=round(sz/sum(sizes)*100,1)
            ax2.text(0,-.05-i*.18,f"● {lbl}:  {sz} shots  ({pct}%)",
                     ha="center",va="top",color=donut_colors[i],fontsize=9,fontweight="600",
                     transform=ax2.transAxes)
    ax2.set_title("Shooting Foot",color="#e6edf3",fontsize=11,fontweight="700",pad=8)

    # ── pass direction breakdown ─────────────────────────────────────────
    ax3=fig.add_subplot(gs[2]); ax3.set_facecolor("#161b22")
    sp_mead=[{"player":pl,"angle":a,"dist":d,"outcome":o,"season":s}
             for pl,a,d,o,s in sonar_passes
             if pl=="B. Mead" and s in selected_seasons]
    def dir_label(angle):
        if -45<=angle<45:   return "Forward"
        if 45<=angle<135:   return "Left"
        if angle>=135 or angle<-135: return "Backward"
        return "Right"
    dir_counts=Counter(dir_label(p["angle"]) for p in sp_mead)
    dir_order=["Forward","Left","Backward","Right"]
    dir_colors=[C_GREEN,C_BLUE,C_ORANGE,C_PURPLE]
    dir_vals=[dir_counts.get(d,0) for d in dir_order]
    bars3=ax3.barh(dir_order,dir_vals,color=dir_colors,alpha=.85,edgecolor="none",height=.55)
    for bar,v in zip(bars3,dir_vals):
        ax3.text(bar.get_width()+max(dir_vals)*.01,bar.get_y()+bar.get_height()/2,
                 str(v),va="center",color="#e6edf3",fontsize=9,fontweight="600")
    ax3.set_title("Pass Direction",color="#e6edf3",fontsize=11,fontweight="700",pad=8)
    ax3.set_xlabel("Passes",color="#8b949e",fontsize=8)
    ax3.tick_params(colors="#484f58"); ax3.yaxis.set_tick_params(labelcolor="#c9d1d9")
    ax3.xaxis.set_tick_params(labelcolor="#8b949e")
    for sp in ax3.spines.values(): sp.set_color("#30363d")
    ax3.grid(axis="x",color="#21262d",linewidth=.5)

    fig.suptitle(f"Style Analysis  ·  B. Mead  ·  {season_label()}",
                 color="#e6edf3",fontsize=14,fontweight="800",y=1.02)
    plt.tight_layout(); st.pyplot(fig,width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 14 – SEASON COMPARE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "compare":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: s1=st.selectbox("Season A",seasons_available,
                              index=max(0,len(seasons_available)-2),key="cmp1")
    with c2: s2=st.selectbox("Season B",seasons_available,
                              index=len(seasons_available)-1,key="cmp2")
    fig,axes=plt.subplots(1,2,figsize=(16,9),facecolor=FIG_BG)
    for ax,season,label,color in zip(axes,[s1,s2],["A","B"],[C_YELLOW,C_BLUE]):
        sp=[p for p in passes if p["season"]==season]
        pitch=Pitch(pitch_type="opta",pitch_color=PITCH_BG,line_color=PITCH_LINE,
                    linewidth=.9,goal_type="box",line_zorder=1)
        pitch.draw(ax=ax)
        ax.set_facecolor(PITCH_BG)
        ORDER={"unsuccessful":0,"successful":1,"progressive":2,"key":3}
        for p in sorted(sp,key=lambda p:ORDER[pass_type(p)]):
            pt=pass_type(p); c=PASS_COLOR[pt]
            alpha=0.25 if pt=="unsuccessful" else (0.80 if pt in ("key","progressive") else 0.45)
            pitch.arrows(p["x"],p["y"],p["end_x"],p["end_y"],ax=ax,
                         color=c,alpha=alpha,width=1.0,headwidth=3.5,headlength=3.5)
        total=len(sp); succ=sum(1 for p in sp if p["outcome"]==1)
        ax.set_title(f"{season.replace('WSL ','')}  ·  {total} passes  ·  "
                     f"{round(succ/total*100,1) if total else 0}% acc",
                     color=color,fontsize=11,fontweight="700",pad=8)
    fig.suptitle(f"Season Comparison  ·  B. Mead",
                 color="#e6edf3",fontsize=14,fontweight="800",y=.98)
    plt.tight_layout(rect=[0,0,1,.95]); st.pyplot(fig,width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 15 – OPPOSITION BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "opposition":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    mr_f=[r for r in match_rows if r["season"] in selected_seasons]
    if not mr_f:
        st.info("No data for selected season(s).")
    else:
        oppo_stats=defaultdict(lambda:defaultdict(int))
        oppo_games=defaultdict(int)
        for r in mr_f:
            opp=r["opponent"]
            oppo_games[opp]+=1
            for k in ["passes","succ_passes","shots","goals","key_passes",
                      "prog_passes","dribbles","tackles","interceptions"]:
                oppo_stats[opp][k]+=r[k]
        opps=sorted(oppo_games.keys())
        stat_op=st.selectbox("Metric",
            ["passes","shots","goals","key_passes","prog_passes","dribbles","tackles","interceptions"],
            format_func=lambda x:x.replace("_"," ").title(),key="oppo_stat")
        vals_op=[oppo_stats[o][stat_op] for o in opps]
        games_op=[oppo_games[o] for o in opps]
        fig,ax=plt.subplots(figsize=(max(12,len(opps)*.7),5),facecolor=FIG_BG)
        ax.set_facecolor("#161b22")
        sort_idx=np.argsort(vals_op)[::-1]
        opps_s=[opps[i] for i in sort_idx]; vals_s=[vals_op[i] for i in sort_idx]
        games_s=[games_op[i] for i in sort_idx]
        colors_op=[C_YELLOW if v==max(vals_s) else C_BLUE for v in vals_s]
        bars_op=ax.bar(range(len(vals_s)),vals_s,color=colors_op,alpha=.85,width=.7,edgecolor="none")
        for i,(v,g) in enumerate(zip(vals_s,games_s)):
            if v>0: ax.text(i,v+.05,f"{v}\n({g}g)",ha="center",va="bottom",
                            color="#e6edf3",fontsize=7)
        ax.set_xticks(range(len(opps_s)))
        ax.set_xticklabels([o.replace(" WFC","").replace(" FCW","").replace(" FC","")
                            for o in opps_s],rotation=45,ha="right",color="#8b949e",fontsize=8)
        ax.set_title(f"{stat_op.replace('_',' ').title()} vs Each Opponent  ·  B. Mead  ·  {season_label()}",
                     color="#e6edf3",fontsize=12,fontweight="800",pad=10)
        ax.tick_params(colors="#484f58"); ax.yaxis.set_tick_params(labelcolor="#8b949e",labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.grid(axis="y",color="#21262d",linewidth=.5)
        plt.tight_layout(); st.pyplot(fig,width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 16 – ARCHETYPES
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "archetypes":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:12px;color:#8b949e;margin-bottom:16px'>"
        "Each archetype is scored from Mead's season stats (normalised to career best). "
        "The dominant archetype per season is highlighted. "
        "All seasons shown regardless of sidebar season filter.</div>",
        unsafe_allow_html=True,
    )

    # ── build per-season stat dict ────────────────────────────────────────
    arch_data = {}
    for s in seasons_available:
        sp  = [p for p in passes   if p["season"]==s]
        ss  = [sh for sh in shots  if sh["season"]==s]
        sd  = [d  for d  in def_acts if d["season"]==s]
        sdr = [d  for d  in dribbles if d["season"]==s]
        scr = [p  for p  in sp if (p["y"]<25 or p["y"]>75) and p["end_x"]>67]
        tp  = len(sp); ts = len(ss); tdr = len(sdr)
        arch_data[s] = {
            "goals":        sum(1 for sh in ss if sh["type_id"]==16),
            "shots":        ts,
            "shot_acc":     round(sum(1 for sh in ss if sh["type_id"] in (15,16))/ts*100,1) if ts else 0,
            "key_passes":   sum(1 for p in sp if p["key_pass"]),
            "progressive":  sum(1 for p in sp if p["progressive"]),
            "pass_acc":     round(sum(1 for p in sp if p["outcome"]==1)/tp*100,1) if tp else 0,
            "dribbles":     tdr,
            "drib_success": round(sum(1 for d in sdr if d["outcome"]==1)/tdr*100,1) if tdr else 0,
            "crosses":      len(scr),
            "tackles":      sum(1 for d in sd if d["action"]=="Tackle"),
            "interceptions":sum(1 for d in sd if d["action"]=="Interception"),
        }

    # normalise each raw metric to 0-100 across all seasons
    all_metric_keys = list(next(iter(arch_data.values())).keys())
    max_m = {m: max(arch_data[s][m] for s in seasons_available) or 1 for m in all_metric_keys}
    norm  = {s: {m: arch_data[s][m]/max_m[m]*100 for m in all_metric_keys} for s in seasons_available}

    # score each season vs each archetype
    def arch_score(season_norm, weights):
        total_w = sum(weights.values())
        return sum(season_norm[m]*w for m,w in weights.items()) / total_w

    scores = {}  # scores[season][archetype] = 0-100
    for s in seasons_available:
        scores[s] = {a: arch_score(norm[s], w) for a,w in ARCHETYPES.items()}

    dominant = {s: max(scores[s], key=scores[s].get) for s in seasons_available}

    # ── TOP SECTION: archetype summary cards ─────────────────────────────
    st.markdown("### Dominant Archetype per Season")
    cols = st.columns(min(len(seasons_available), 6))
    for i, s in enumerate(seasons_available):
        col = cols[i % len(cols)]
        dom = dominant[s]
        color = ARCH_COLORS[dom]
        col.markdown(
            f"<div style='background:#161b22;border:1px solid {color};"
            f"border-radius:8px;padding:12px 10px;text-align:center;margin-bottom:8px'>"
            f"<div style='font-size:10px;color:#8b949e;text-transform:uppercase;"
            f"letter-spacing:.06em'>{s.replace('WSL ','')}</div>"
            f"<div style='font-size:18px;margin:4px 0'>{dom.split()[0]}</div>"
            f"<div style='font-size:11px;font-weight:700;color:{color}'>"
            f"{' '.join(dom.split()[1:])}</div>"
            f"<div style='font-size:10px;color:#484f58;margin-top:4px'>"
            f"score {scores[s][dom]:.0f}/100</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── MAIN CHART: stacked radar comparison across all archetypes ────────
    st.markdown("### Archetype Score Breakdown — All Seasons")

    fig, axes = plt.subplots(
        2, 3, figsize=(16, 9),
        subplot_kw={"projection": "polar"},
        facecolor=FIG_BG,
    )
    fig.patch.set_facecolor(FIG_BG)

    for ax, (arch_name, weights) in zip(axes.flat, ARCHETYPES.items()):
        ax.set_facecolor("#161b22")
        color = ARCH_COLORS[arch_name]
        slabels = [s.replace("WSL ","") for s in seasons_available]
        N = len(seasons_available)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += [angles[0]]
        vals   = [scores[s][arch_name] for s in seasons_available]
        vals  += [vals[0]]

        # background rings
        for r in [25, 50, 75, 100]:
            ax.plot(angles, [r]*len(angles), color="#30363d", linewidth=.4)
        ax.fill(angles, [100]*len(angles), color="#21262d", alpha=.4)
        ax.plot(angles, vals, color=color, linewidth=2.2, zorder=3)
        ax.fill(angles, vals, color=color, alpha=.20, zorder=2)
        ax.scatter(angles[:-1], vals[:-1], s=50, color=color,
                   zorder=5, edgecolors=FIG_BG, linewidths=1.5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(slabels, color="#8b949e", fontsize=6.5)
        ax.set_yticks([]); ax.set_ylim(0, 115)
        ax.spines["polar"].set_color("#30363d"); ax.grid(False)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_title(arch_name, color=color, fontsize=10,
                     fontweight="700", pad=10)
        # mark the dominant seasons
        for i, s in enumerate(seasons_available):
            if dominant[s] == arch_name:
                ax.scatter([angles[i]], [vals[i]], s=120, color=color,
                           zorder=6, edgecolors="white", linewidths=1.5,
                           marker="*")

    fig.suptitle("Archetype Radars  ·  B. Mead  ·  ★ = dominant season",
                 color="#e6edf3", fontsize=13, fontweight="800", y=.99)
    plt.tight_layout(rect=[0,0,1,.97])
    st.pyplot(fig, width="stretch"); plt.close(fig)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── HEATMAP: season × archetype score grid ────────────────────────────
    st.markdown("### Season × Archetype Score Matrix")
    arch_names  = list(ARCHETYPES.keys())
    matrix      = np.array([[scores[s][a] for a in arch_names] for s in seasons_available])
    slabels_short = [s.replace("WSL ","") for s in seasons_available]

    fig2, ax2 = plt.subplots(figsize=(14, max(4, len(seasons_available)*0.55+1.5)),
                              facecolor=FIG_BG)
    ax2.set_facecolor(FIG_BG)
    cmap_m = LinearSegmentedColormap.from_list("arch",["#161b22","#1a3a2a",C_GREEN,C_YELLOW],N=256)
    im = ax2.imshow(matrix, aspect="auto", cmap=cmap_m, vmin=0, vmax=100)

    ax2.set_xticks(range(len(arch_names)))
    ax2.set_xticklabels(arch_names, color="#c9d1d9", fontsize=9, rotation=20, ha="right")
    ax2.set_yticks(range(len(slabels_short)))
    ax2.set_yticklabels(slabels_short, color="#8b949e", fontsize=9)

    for i in range(len(seasons_available)):
        for j in range(len(arch_names)):
            val = matrix[i,j]
            is_dom = (dominant[seasons_available[i]] == arch_names[j])
            ax2.text(j, i, f"{val:.0f}", ha="center", va="center",
                     fontsize=9 if is_dom else 8,
                     fontweight="bold" if is_dom else "normal",
                     color="white" if val > 50 else "#8b949e")
            if is_dom:
                ax2.add_patch(plt.Rectangle((j-.45, i-.45), .9, .9,
                              fill=False, edgecolor="white", linewidth=1.5))

    for spine in ax2.spines.values(): spine.set_visible(False)
    ax2.tick_params(colors="#484f58")
    cb = fig2.colorbar(im, ax=ax2, fraction=.015, pad=.02)
    cb.ax.yaxis.set_tick_params(color="#8b949e", labelsize=8)
    cb.outline.set_edgecolor("#30363d")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#8b949e")
    fig2.suptitle("Score Matrix  ·  white border = dominant archetype for that season",
                  color="#8b949e", fontsize=9, y=1.01)
    plt.tight_layout()
    st.pyplot(fig2, width="stretch"); plt.close(fig2)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── ARCHETYPE DEFINITIONS ─────────────────────────────────────────────
    st.markdown("### Archetype Definitions")
    def_cols = st.columns(3)
    for i, (name, weights) in enumerate(ARCHETYPES.items()):
        col = def_cols[i % 3]
        color = ARCH_COLORS[name]
        metric_lines = "".join(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:2px 0;border-bottom:1px solid #21262d'>"
            f"<span style='color:#8b949e;font-size:11px'>{m.replace('_',' ').title()}</span>"
            f"<span style='color:{color};font-size:11px;font-weight:700'>×{w}</span></div>"
            for m,w in weights.items()
        )
        col.markdown(
            f"<div style='background:#161b22;border:1px solid #30363d;"
            f"border-left:3px solid {color};border-radius:8px;"
            f"padding:12px 14px;margin-bottom:12px'>"
            f"<div style='font-size:13px;font-weight:700;color:{color};"
            f"margin-bottom:8px'>{name}</div>"
            f"{metric_lines}</div>",
            unsafe_allow_html=True,
        )

# 17 – DEFENSIVE ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "defensive":
    back_btn()
    fp=f_passes(passes); fs=f_shots(shots); fd=f_def(def_acts)
    metric_row(fp,fs,fd); st.markdown('<div style="height:1px;background:linear-gradient(90deg,#f59e0b33,#3b82f633,transparent);margin:20px 0 28px"></div>',unsafe_allow_html=True)
    pitch,fig,ax=make_pitch(figsize=(16,10))
    MARKER={"Tackle":"s","Interception":"D","Clearance":"o","Block":"^"}
    for d in fd:
        ax.scatter(d["x"],d["y"],c=DEF_COLOR[d["action"]],marker=MARKER[d["action"]],
                   s=90,alpha=.80,zorder=3,edgecolors="#0d1117",linewidths=.5)
    if sel_def:
        add_legend(ax,[(DEF_COLOR[a],f"{a} ({sum(1 for d in fd if d['action']==a)})")
                       for a in sel_def])
    add_title(fig,"Defensive Actions  ·  B. Mead",
              f"{season_label()}  ·  {len(fd)} actions")
    plt.tight_layout(rect=[0,0,1,.94]); st.pyplot(fig,width="stretch"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 17 – PERCENTILE CHART
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# ZONE 14
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "zone14":
    back_btn()
    fp = f_passes(passes); fs = f_shots(shots); fd = f_def(def_acts)
    metric_row(fp, fs, fd)
    st.markdown(DIVIDER, unsafe_allow_html=True)

    all_p  = f_season(passes)
    all_s  = f_season(shots)
    all_t  = f_season(touches)
    all_dr = f_season(dribbles)

    # ── classify ──────────────────────────────────────────────────────────
    # passes: from Z14, into Z14, key passes from Z14
    p_from  = [p for p in all_p if in_zone14(p["x"], p["y"])]
    p_into  = [p for p in all_p if not in_zone14(p["x"], p["y"]) and in_zone14(p["end_x"], p["end_y"])]
    p_z14   = [p for p in all_p if in_zone14(p["x"], p["y"]) or in_zone14(p["end_x"], p["end_y"])]
    shots_z = [s for s in all_s if in_zone14(s["x"], s["y"])]
    goals_z = [s for s in shots_z if s["type_id"] == 16]
    drbs_z  = [d for d in all_dr if in_zone14(d["x"], d["y"])]
    touch_z = [t for t in all_t if in_zone14(t["x"], t["y"])]
    key_z   = [p for p in p_from if p["key_pass"]]

    # ── sub-metrics ───────────────────────────────────────────────────────
    r1c1, r1c2, r1c3, r1c4, r1c5, r1c6 = st.columns(6)
    for col, label, val, color in [
        (r1c1, "Zone 14 Touches",   len(touch_z),                         C_YELLOW),
        (r1c2, "Passes From Z14",   len(p_from),                          C_BLUE),
        (r1c3, "Passes Into Z14",   len(p_into),                          C_BLUE),
        (r1c4, "Key Passes Z14",    len(key_z),                           C_GREEN),
        (r1c5, "Shots From Z14",    len(shots_z),                         C_ORANGE),
        (r1c6, "Goals From Z14",    len(goals_z),                         C_YELLOW),
    ]:
        col.markdown(
            f"<div style='background:linear-gradient(160deg,#0f1420,#0c111d);"
            f"border:1px solid #1a2235;border-top:2px solid {color};"
            f"border-radius:10px;padding:16px;text-align:center'>"
            f"<div style='font-family:Space Grotesk,sans-serif;font-size:26px;"
            f"font-weight:800;color:{color};line-height:1'>{val}</div>"
            f"<div style='font-size:9px;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:.1em;color:#4b5563;margin-top:5px'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(DIVIDER, unsafe_allow_html=True)

    # ── view toggle ───────────────────────────────────────────────────────
    z14_view = st.radio(
        "Show",
        ["All actions", "Passes from Zone 14", "Passes into Zone 14",
         "Shots", "Dribbles", "Heat density"],
        horizontal=True,
    )

    # ── build two-panel layout ────────────────────────────────────────────
    left_col, right_col = st.columns([3, 1])

    with left_col:
        pitch = Pitch(pitch_type="opta", pitch_color=PITCH_BG, line_color=PITCH_LINE,
                      linewidth=1.0, goal_type="box", line_zorder=3)
        fig, ax = pitch.draw(figsize=(13, 9))
        fig.patch.set_facecolor(FIG_BG)

        # ── Zone 14 overlay ───────────────────────────────────────────────
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (Z14_X[0], Z14_Y[0]),
            Z14_X[1] - Z14_X[0],
            Z14_Y[1] - Z14_Y[0],
            linewidth=1.5, edgecolor=C_YELLOW, facecolor=C_YELLOW,
            alpha=0.10, zorder=1, linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(
            (Z14_X[0]+Z14_X[1])/2, (Z14_Y[0]+Z14_Y[1])/2,
            "ZONE 14", ha="center", va="center",
            fontsize=11, fontweight="900", color=C_YELLOW, alpha=0.35,
            fontfamily="monospace", zorder=2,
        )

        # ── render chosen view ────────────────────────────────────────────
        if z14_view == "All actions":
            ORDER = {"unsuccessful":0,"successful":1,"progressive":2,"key":3}
            for p in sorted(p_z14, key=lambda p: ORDER[pass_type(p)]):
                color = PASS_COLOR[pass_type(p)]
                alpha = 0.25 if p["outcome"]==0 else (0.80 if pass_type(p) in ("key","progressive") else 0.50)
                pitch.arrows(p["x"],p["y"],p["end_x"],p["end_y"],
                             ax=ax,color=color,alpha=alpha,width=1.1,headwidth=4,headlength=4,zorder=4)
            for s in shots_z:
                c,mk,sz,al,zo = SHOT_STYLE[s["type_id"]]
                pitch.scatter([s["x"]],[s["y"]],ax=ax,c=c,marker=mk,s=sz,alpha=al,zorder=zo+3,
                              edgecolors="white",linewidths=.3)

        elif z14_view == "Passes from Zone 14":
            for p in sorted(p_from, key=lambda p: ORDER[pass_type(p)] if False else 0):
                color = PASS_COLOR[pass_type(p)]
                alpha = 0.28 if p["outcome"]==0 else (0.82 if pass_type(p) in ("key","progressive") else 0.55)
                pitch.arrows(p["x"],p["y"],p["end_x"],p["end_y"],
                             ax=ax,color=color,alpha=alpha,width=1.3,headwidth=4.5,headlength=4.5,zorder=4)

        elif z14_view == "Passes into Zone 14":
            for p in p_into:
                color = C_BLUE if p["outcome"]==1 else C_ORANGE
                alpha = 0.65 if p["outcome"]==1 else 0.35
                pitch.arrows(p["x"],p["y"],p["end_x"],p["end_y"],
                             ax=ax,color=color,alpha=alpha,width=1.2,headwidth=4,headlength=4,zorder=4)

        elif z14_view == "Shots":
            for tid in (13,14,15,16):
                sub = [s for s in shots_z if s["type_id"]==tid]
                if sub:
                    c,mk,sz,al,zo = SHOT_STYLE[tid]
                    pitch.scatter([s["x"] for s in sub],[s["y"] for s in sub],
                                  ax=ax,c=c,marker=mk,s=sz*1.2,alpha=al,zorder=zo+3,
                                  edgecolors="white",linewidths=.5)

        elif z14_view == "Dribbles":
            for d in drbs_z:
                c = C_GREEN if d["outcome"]==1 else C_ORANGE
                ax.scatter(d["x"],d["y"],c=c,s=90,alpha=0.80,zorder=4,
                           edgecolors="#0d1117",linewidths=.5)
            if drbs_z:
                add_legend(ax,[(C_GREEN,f"Successful ({sum(1 for d in drbs_z if d['outcome']==1)})"),
                               (C_ORANGE,f"Unsuccessful ({sum(1 for d in drbs_z if d['outcome']==0)})")])

        elif z14_view == "Heat density":
            if touch_z:
                cmap_z = LinearSegmentedColormap.from_list(
                    "z14", [PITCH_BG,"#3d2800",C_YELLOW,"#ffffff"], N=256)
                pitch.kdeplot([t["x"] for t in all_t],[t["y"] for t in all_t],
                              ax=ax,cmap=cmap_z,fill=True,levels=100,
                              alpha=.80,bw_adjust=.65,zorder=1)

        add_title(fig, "Zone 14  ·  B. Mead",
                  f"{season_label()}  ·  {z14_view}  ·  "
                  f"x={Z14_X[0]}–{Z14_X[1]}, y={Z14_Y[0]}–{Z14_Y[1]} (Opta 0-100)")
        plt.tight_layout(rect=[0,0,1,.94])
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with right_col:
        # ── Z14 season breakdown table ────────────────────────────────────
        st.markdown(
            "<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:.1em;color:#f59e0b;margin-bottom:12px'>Season Breakdown</div>",
            unsafe_allow_html=True,
        )
        for s in seasons_available:
            if s not in selected_seasons:
                continue
            sp = [p for p in passes   if p["season"]==s]
            ss = [sh for sh in shots  if sh["season"]==s]
            st_  = [t for t in touches if t["season"]==s]
            pf  = sum(1 for p in sp if in_zone14(p["x"],p["y"]))
            pi  = sum(1 for p in sp if not in_zone14(p["x"],p["y"]) and in_zone14(p["end_x"],p["end_y"]))
            sf  = sum(1 for sh in ss if in_zone14(sh["x"],sh["y"]))
            gf  = sum(1 for sh in ss if in_zone14(sh["x"],sh["y"]) and sh["type_id"]==16)
            tz  = sum(1 for t in st_ if in_zone14(t["x"],t["y"]))
            st.markdown(
                f"<div style='background:#0f1420;border:1px solid #1a2235;"
                f"border-left:2px solid #f59e0b;border-radius:8px;"
                f"padding:10px 12px;margin-bottom:8px'>"
                f"<div style='font-size:11px;font-weight:700;color:#f59e0b;"
                f"margin-bottom:6px'>{s.replace('WSL ','')}</div>"
                f"<div style='font-size:10px;color:#6b7280;line-height:1.9'>"
                f"Touches: <b style='color:#e8eaf0'>{tz}</b><br>"
                f"Passes from: <b style='color:#3b82f6'>{pf}</b><br>"
                f"Passes into: <b style='color:#3b82f6'>{pi}</b><br>"
                f"Shots: <b style='color:#f97316'>{sf}</b><br>"
                f"Goals: <b style='color:#f59e0b'>{gf}</b>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "percentile":
    back_btn()
    st.markdown("<div style='font-size:12px;color:#8b949e;margin-bottom:12px'>"
                "Bars show each season relative to Mead's personal best "
                "(100 = best season for that metric).</div>",unsafe_allow_html=True)
    season_stats={}
    for s in seasons_available:
        sp=[p for p in passes   if p["season"]==s]
        ss=[sh for sh in shots  if sh["season"]==s]
        sd=[d  for d  in def_acts if d["season"]==s]
        sdr=[d for d in dribbles if d["season"]==s]
        tp=len(sp); ts=len(ss)
        season_stats[s]={
            "Passes":        tp,
            "Pass Acc %":    round(sum(1 for p in sp if p["outcome"]==1)/tp*100,1) if tp else 0,
            "Progressive":   sum(1 for p in sp if p["progressive"]),
            "Shot Assists":  sum(1 for p in sp if p["key_pass"]),
            "Shots":         ts,
            "Goals":         sum(1 for sh in ss if sh["type_id"]==16),
            "Shot Acc %":    round(sum(1 for sh in ss if sh["type_id"] in (15,16))/ts*100,1) if ts else 0,
            "Tackles":       sum(1 for d in sd if d["action"]=="Tackle"),
            "Interceptions": sum(1 for d in sd if d["action"]=="Interception"),
            "Dribbles":      len(sdr),
        }
    metrics_p=list(next(iter(season_stats.values())).keys())
    max_v={m:max(season_stats[s][m] for s in season_stats) or 1 for m in metrics_p}
    BC=[C_YELLOW,C_BLUE,C_GREEN,C_ORANGE,C_ORANGE,C_YELLOW,C_GREEN,C_ORANGE,C_PURPLE,C_BLUE]
    disp=[s for s in seasons_available if s in selected_seasons]
    for row_s in [disp[i:i+3] for i in range(0,len(disp),3)]:
        cols=st.columns(len(row_s))
        for col,s in zip(cols,row_s):
            with col:
                fig,ax=plt.subplots(figsize=(5,4.8)); fig.patch.set_facecolor(FIG_BG)
                ax.set_facecolor("#161b22")
                vals=[season_stats[s][m] for m in metrics_p]
                pcts=[v/max_v[m]*100 for v,m in zip(vals,metrics_p)]
                yp=np.arange(len(metrics_p))
                ax.barh(yp,[100]*len(metrics_p),color="#21262d",height=.6,edgecolor="none",zorder=1)
                bars=ax.barh(yp,pcts,color=BC[:len(metrics_p)],height=.6,alpha=.9,edgecolor="none",zorder=2)
                for bar,v in zip(bars,vals):
                    ax.text(bar.get_width()+2,bar.get_y()+bar.get_height()/2,
                            f"{v:g}",va="center",ha="left",color="#e6edf3",fontsize=7.5,fontweight="700")
                ax.set_yticks(yp); ax.set_yticklabels(metrics_p,color="#8b949e",fontsize=8)
                ax.set_xlim(0,120); ax.set_xlabel("% of personal best",color="#484f58",fontsize=7)
                ax.tick_params(colors="#484f58",labelsize=7)
                for sp in ax.spines.values(): sp.set_visible(False)
                ax.axvline(100,color="#30363d",linewidth=.8,linestyle="--",zorder=3)
                ax.set_title(s.replace("WSL ",""),color=C_BLUE,fontsize=10,fontweight="800",pad=10)
                plt.tight_layout(); st.pyplot(fig,width="stretch"); plt.close(fig)

elif st.session_state.page == "sub_impact":
    back_btn()
    st.markdown("""
<div style='margin-bottom:24px'>
<span style='font-size:2rem;font-weight:900;color:#e8eaf0'>Substitution Impact Model</span><br>
<span style='color:#8b949e;font-size:.95rem'>
Monte Carlo simulation of player on/off-pitch contribution across six metrics.
Configure per-metric means from your data, then run the simulator.
</span>
</div>
""", unsafe_allow_html=True)

    xg_df = _load_xg_csvs()

    st.markdown("### Quick-fill from xG data")

    if xg_df.empty:
        st.info("No xG CSV files found — enter metric values manually below.")
        selected_player = None
        player_xg_on = 0.0
        player_goals_on = 0.0
    else:
        all_players = sorted(xg_df["PlayerId"].dropna().unique().tolist()) if "PlayerId" in xg_df.columns else []
        col_p1, col_p2 = st.columns([2, 1])
        with col_p1:
            selected_player = st.selectbox("Player", ["(manual entry)"] + all_players)
        with col_p2:
            role_filter = st.selectbox("Role context", ["All appearances", "Starter", "Substitute"])

        if selected_player and selected_player != "(manual entry)" and "PlayerId" in xg_df.columns:
            pdf = xg_df[xg_df["PlayerId"] == selected_player].copy()
            if "timeMin" in pdf.columns and role_filter != "All appearances":
                first_mins = pdf.groupby(["season", "Date"])["timeMin"].min().reset_index()
                if role_filter == "Starter":
                    starter_matches = first_mins[first_mins["timeMin"] <= 25][["season", "Date"]]
                    pdf = pdf.merge(starter_matches, on=["season", "Date"])
                else:
                    sub_matches = first_mins[first_mins["timeMin"] > 45][["season", "Date"]]
                    pdf = pdf.merge(sub_matches, on=["season", "Date"])

            player_xg_on    = float(pdf["xG"].sum())    if "xG"     in pdf.columns else 0.0
            player_goals_on = float(pdf["isGoal"].sum()) if "isGoal" in pdf.columns else 0.0
            n_apps = pdf["Date"].nunique() if "Date" in pdf.columns else 0
            gplus_on = player_goals_on - player_xg_on
            st.markdown(
                f"<span style='color:#8b949e;font-size:.85rem'>"
                f"Found <b style='color:#e8eaf0'>{n_apps}</b> appearances · "
                f"xG sum <b style='color:#3b82f6'>{player_xg_on:.2f}</b> · "
                f"Goals <b style='color:#f59e0b'>{player_goals_on:.0f}</b> · "
                f"G+ <b style='color:#10b981'>{gplus_on:+.2f}</b>"
                f"</span>", unsafe_allow_html=True)
        else:
            player_xg_on = 0.0
            player_goals_on = 0.0

    # ── on/off real data from computed CSV ──────────────────────────────────────
    _onoff_df = _load_onoff()
    _has_player = (selected_player if 'selected_player' in dir() else None) and selected_player != "(manual entry)"

    _xt_on_pm   = 0.020;  _xt_off_pm   = 0.010
    _vaep_on_pm = 0.001;  _vaep_off_pm = 0.001
    _n_onoff_matches = 0

    if _has_player and not _onoff_df.empty:
        _poo = _onoff_df[_onoff_df["player"] == selected_player].copy()
        if role_filter == "Starter":
            _poo = _poo[_poo["role"] == "starter"]
        elif role_filter == "Substitute":
            _poo = _poo[_poo["role"] == "substitute"]
        if not _poo.empty:
            _n_onoff_matches = len(_poo)
            _xt_on_pm   = float(_poo["on_xt_per_min"].mean())
            _xt_off_pm  = float(_poo["off_xt_per_min"].mean())
            _vaep_on_pm = float(_poo["on_vaep_per_min"].mean())
            _vaep_off_pm= float(_poo["off_vaep_per_min"].mean())
            _xt_diff    = _xt_on_pm - _xt_off_pm
            _vaep_diff  = _vaep_on_pm - _vaep_off_pm

            # On/off summary chart
            st.markdown("### On / Off pitch impact (from match data)")
            _oo_fig, _oo_axes = plt.subplots(1, 2, figsize=(10, 3))
            _oo_fig.patch.set_facecolor(FIG_BG)
            for _ax, (vals, lbls, title, colors) in zip(_oo_axes, [
                ([_xt_on_pm, _xt_off_pm], ["On", "Off"], "xT per minute", [C_GREEN, "#30363d"]),
                ([_vaep_on_pm, _vaep_off_pm], ["On", "Off"], "VAEP per minute", ["#a78bfa", "#30363d"]),
            ]):
                _ax.set_facecolor("#161b22")
                _bars = _ax.bar(lbls, vals, color=colors, edgecolor="none", width=0.5)
                for b, v in zip(_bars, vals):
                    _ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(vals)*0.03,
                             f"{v:.4f}", ha="center", va="bottom", color="#e8eaf0", fontsize=9, fontweight="700")
                _ax.set_title(title, color="#e8eaf0", fontsize=10, fontweight="800")
                _ax.tick_params(colors="#8b949e", labelsize=9)
                for sp in _ax.spines.values(): sp.set_visible(False)
                _ax.set_facecolor("#161b22")
            _oo_fig.suptitle(
                f"{selected_player} · {_n_onoff_matches} matches · {role_filter}",
                color="#8b949e", fontsize=9, y=1.01)
            plt.tight_layout()
            st.pyplot(_oo_fig, use_container_width=True)
            plt.close(_oo_fig)

            st.markdown(
                f"<span style='color:#8b949e;font-size:.8rem'>"
                f"xT diff (on−off): <b style='color:#10b981'>{_xt_diff:+.4f}</b>/min · "
                f"VAEP diff: <b style='color:#a78bfa'>{_vaep_diff:+.5f}</b>/min · "
                f"Matches: <b style='color:#e8eaf0'>{_n_onoff_matches}</b>"
                f"</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Metric inputs (per minute averages)")
    st.markdown("<span style='color:#8b949e;font-size:.85rem'>On/off values pre-filled from the computed xT &amp; VAEP model. "
                "Edit any field then run the simulation.</span>", unsafe_allow_html=True)

    _METRICS = [
        {"key": "xg",    "label": "xG",              "type": "poisson", "weight": 2.0, "color": "#3b82f6"},
        {"key": "xt",    "label": "xT",               "type": "normal",  "weight": 1.5, "color": "#10b981"},
        {"key": "vaep",  "label": "VAEP",             "type": "normal",  "weight": 1.5, "color": "#a78bfa"},
        {"key": "epv",   "label": "EPV",              "type": "normal",  "weight": 1.0, "color": "#f97316"},
        {"key": "gplus", "label": "G+ (Goals − xG)", "type": "normal",  "weight": 1.0, "color": "#f59e0b"},
        {"key": "gd",    "label": "Goal Diff Added",  "type": "normal",  "weight": 2.0, "color": "#ef4444"},
    ]

    _gplus_default = round((player_goals_on - player_xg_on), 3) if _has_player else 0.01
    _defaults = {
        "xg":    (round(player_xg_on, 3), 0.05),
        "xt":    (round(_xt_on_pm, 5),   round(abs(_xt_on_pm) * 0.3, 5)),
        "vaep":  (round(_vaep_on_pm, 5), round(abs(_vaep_on_pm) * 0.3, 5)),
        "epv":   (0.10, 0.03),
        "gplus": (_gplus_default, 0.05),
        "gd":    (0.15, 0.08),
    }

    metric_inputs = {}
    hcols = st.columns([2, 1.5, 1.5, 1.5, 1.5, 1])
    for hc, hl in zip(hcols, ["Metric","On-pitch mean","Off-pitch mean","σ (on)","σ (off)","Weight"]):
        hc.markdown(f"<span style='color:#484f58;font-size:.8rem'>{hl}</span>", unsafe_allow_html=True)

    for m in _METRICS:
        k = m["key"]
        on_def, sig_def = _defaults.get(k, (0.10, 0.05))
        mcols = st.columns([2, 1.5, 1.5, 1.5, 1.5, 1])
        mcols[0].markdown(f"<span style='color:{m['color']};font-weight:700'>{m['label']}</span>", unsafe_allow_html=True)
        # use real off-pitch value for xt/vaep, else 70% fallback
        off_real = {"xt": round(_xt_off_pm, 5), "vaep": round(_vaep_off_pm, 5)}.get(k, on_def * 0.7)
        on_mean  = mcols[1].number_input("", value=float(on_def),   step=0.001, key=f"on_{k}",  label_visibility="collapsed", format="%.4f")
        off_mean = mcols[2].number_input("", value=float(off_real), step=0.001, key=f"off_{k}", label_visibility="collapsed", format="%.4f")
        sig_on   = mcols[3].number_input("", value=float(sig_def),       step=0.005, key=f"son_{k}", label_visibility="collapsed", format="%.3f")
        sig_off  = mcols[4].number_input("", value=float(sig_def),       step=0.005, key=f"sof_{k}", label_visibility="collapsed", format="%.3f")
        weight   = mcols[5].number_input("", value=m["weight"], step=0.5, min_value=0.0, max_value=5.0, key=f"w_{k}", label_visibility="collapsed", format="%.1f")
        metric_inputs[k] = {"on": on_mean, "off": off_mean, "sig_on": sig_on, "sig_off": sig_off,
                            "weight": weight, "type": m["type"], "color": m["color"], "label": m["label"]}

    st.markdown("---")
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    with sc1:
        n_sims = st.selectbox("Simulations", [5_000, 10_000, 50_000], index=1, format_func=lambda x: f"{x:,}")
    with sc2:
        sub_var = st.checkbox("Sub variance boost (×1.3σ)", value=True,
                              help="Inflates σ for substitute appearances — reflects higher situational variance")
    with sc3:
        run_btn = st.button("▶  Run Monte Carlo", use_container_width=True, type="primary")

    if run_btn:
        results = {}
        composite_sims = None
        total_weight = sum(v["weight"] for v in metric_inputs.values())

        with st.spinner("Running simulations…"):
            for k, cfg in metric_inputs.items():
                s_on  = cfg["sig_on"]  * (1.3 if sub_var else 1.0)
                s_off = cfg["sig_off"] * (1.3 if sub_var else 1.0)
                sims = _mc_net(cfg["on"], cfg["off"], s_on, s_off, n_sims, cfg["type"])
                results[k] = _pct(sims)
                results[k]["raw"] = sims
                w = cfg["weight"] / total_weight if total_weight > 0 else 1
                composite_sims = sims * w if composite_sims is None else composite_sims + sims * w

        comp = _pct(composite_sims)

        st.markdown("### Composite Net Impact Score")
        cc1, cc2, cc3, cc4 = st.columns(4)
        def _scard(col, lbl, val, clr):
            col.markdown(
                f"<div style='background:#161b22;border-radius:10px;padding:16px 20px;text-align:center'>"
                f"<div style='color:#8b949e;font-size:.8rem;margin-bottom:4px'>{lbl}</div>"
                f"<div style='color:{clr};font-size:1.8rem;font-weight:900'>{val:+.3f}</div>"
                f"</div>", unsafe_allow_html=True)
        _scard(cc1, "Mean",       comp["mean"],    C_BLUE)
        _scard(cc2, "P10",        comp["p10"],     "#ef4444")
        _scard(cc3, "P90",        comp["p90"],     C_GREEN)
        _scard(cc4, "% Positive", comp["pos_pct"], "#f59e0b")

        st.markdown("<br>", unsafe_allow_html=True)

        fig_c, ax_c = plt.subplots(figsize=(12, 3.5))
        fig_c.patch.set_facecolor(FIG_BG); ax_c.set_facecolor("#161b22")
        ax_c.hist(composite_sims, bins=120, color=C_BLUE, alpha=0.75, edgecolor="none", density=True)
        ax_c.axvline(0, color="#484f58", linewidth=1.2, linestyle="--")
        ax_c.axvline(comp["mean"], color=C_BLUE,    linewidth=2,   label=f"Mean {comp['mean']:+.3f}")
        ax_c.axvline(comp["p10"],  color="#ef4444", linewidth=1.5, linestyle=":", label=f"P10 {comp['p10']:+.3f}")
        ax_c.axvline(comp["p90"],  color=C_GREEN,   linewidth=1.5, linestyle=":", label=f"P90 {comp['p90']:+.3f}")
        ylim_top = ax_c.get_ylim()[1] or 1
        ax_c.fill_betweenx([0, ylim_top], comp["p10"], comp["p90"], color=C_BLUE, alpha=0.08)
        ax_c.set_title("Composite Net Impact Distribution (weighted)", color="#e8eaf0", fontsize=11, fontweight="800", pad=10)
        ax_c.set_xlabel("Net impact (on − off)", color="#484f58", fontsize=9)
        ax_c.set_ylabel("Density", color="#484f58", fontsize=9)
        ax_c.tick_params(colors="#484f58", labelsize=8)
        for sp in ax_c.spines.values(): sp.set_visible(False)
        ax_c.legend(frameon=False, labelcolor="#e8eaf0", fontsize=8)
        plt.tight_layout(); st.pyplot(fig_c, use_container_width=True); plt.close(fig_c)

        st.markdown("---")
        st.markdown("### Per-metric breakdown")

        fig_m, axes = plt.subplots(1, len(_METRICS), figsize=(16, 4))
        fig_m.patch.set_facecolor(FIG_BG)
        for i, m in enumerate(_METRICS):
            k = m["key"]; ax = axes[i]; ax.set_facecolor("#161b22")
            r = results[k]
            pv  = [r["p10"], r["p25"], r["p50"], r["p75"], r["p90"]]
            lbls = ["P10", "P25", "P50", "P75", "P90"]
            clrs = ["#ef4444", "#f97316", m["color"], "#10b981", "#10b981"]
            y = np.arange(len(lbls))
            ax.barh(y, pv, color=clrs, alpha=0.8, edgecolor="none", height=0.6)
            ax.axvline(0, color="#484f58", linewidth=0.8, linestyle="--")
            for j, v in enumerate(pv):
                ax.text(v + (0.003 if v >= 0 else -0.003), j, f"{v:+.3f}",
                        va="center", ha="left" if v >= 0 else "right",
                        color="#e8eaf0", fontsize=7, fontweight="700")
            ax.set_yticks(y); ax.set_yticklabels(lbls, color="#8b949e", fontsize=8)
            ax.set_title(m["label"], color=m["color"], fontsize=10, fontweight="800", pad=8)
            ax.tick_params(colors="#484f58", labelsize=7)
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.set_xlabel(f"{r['pos_pct']:.0f}% positive", color="#484f58", fontsize=7)
        plt.tight_layout(); st.pyplot(fig_m, use_container_width=True); plt.close(fig_m)

        st.markdown("### Summary table")
        rows = []
        for m in _METRICS:
            k = m["key"]; r = results[k]
            on_v = metric_inputs[k]["on"]; off_v = metric_inputs[k]["off"]
            status = "Positive" if r["mean"] > 0.01 else ("Negative" if r["mean"] < -0.01 else "Neutral")
            rows.append({"Metric": m["label"], "On mean": f"{on_v:.3f}", "Off mean": f"{off_v:.3f}",
                         "Net (on−off)": f"{on_v - off_v:+.3f}", "Sim mean": f"{r['mean']:+.3f}",
                         "P10": f"{r['p10']:+.3f}", "P90": f"{r['p90']:+.3f}",
                         "% Positive": f"{r['pos_pct']:.1f}%", "Status": status})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

elif st.session_state.page == "xt_vaep":
    back_btn()
    st.markdown("""
<div style='margin-bottom:24px'>
<span style='font-size:2rem;font-weight:900;color:#e8eaf0'>xT &amp; VAEP</span><br>
<span style='color:#8b949e;font-size:.95rem'>
Expected Threat (xT) and VAEP computed from all 1,243 WSL match files.
xT measures how much each action increases the probability of scoring;
VAEP values actions by the change in scoring/conceding probability over a 3-action context window.
</span>
</div>
""", unsafe_allow_html=True)

    _summary = _load_xt_vaep_summary()
    _actions = _load_xt_vaep_actions()

    if _summary.empty:
        st.error("player_vaep_xt_summary.csv not found — run compute_vaep_xt.py first.")
        st.stop()

    # ── controls ─────────────────────────────────────────────────────────────
    tv1, tv2, tv3 = st.columns([2, 1, 1])
    with tv1:
        _all_players = sorted(_summary["player"].dropna().unique().tolist())
        _sel_player  = st.selectbox("Player", _all_players,
                                    index=_all_players.index("B. Mead") if "B. Mead" in _all_players else 0)
    with tv2:
        _metric_choice = st.selectbox("Primary metric", ["xT", "VAEP"])
    with tv3:
        _norm_choice = st.selectbox("Normalisation", ["Total", "Per action", "Per 90"])

    _pm = _summary[_summary["player"] == _sel_player].copy()
    _pm = _pm.sort_values("season")

    # ── season timeline ───────────────────────────────────────────────────────
    if not _pm.empty:
        _col_map = {
            ("xT",  "Total"):      "total_xt",
            ("xT",  "Per action"): "xt_per_action",
            ("xT",  "Per 90"):     "total_xt",   # will divide by estimated mins
            ("VAEP","Total"):      "total_vaep",
            ("VAEP","Per action"): "vaep_per_action",
            ("VAEP","Per 90"):     "vaep_per90",
        }
        _ycol = _col_map[(_metric_choice, _norm_choice)]
        _yvals = _pm[_ycol].values
        _seasons_lbl = [s.replace("WSL_","WSL ") for s in _pm["season"].values]
        _mcolor = "#3b82f6" if _metric_choice == "xT" else "#a78bfa"

        fig_t, ax_t = plt.subplots(figsize=(12, 3.8))
        fig_t.patch.set_facecolor(FIG_BG); ax_t.set_facecolor("#161b22")
        x_pos = np.arange(len(_seasons_lbl))
        _bars = ax_t.bar(x_pos, _yvals, color=_mcolor, alpha=0.8, edgecolor="none", width=0.6)
        for bar, v in zip(_bars, _yvals):
            ax_t.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(_yvals)*0.02 if max(_yvals)>0 else 0.01),
                      f"{v:.2f}", ha="center", va="bottom", color="#e8eaf0", fontsize=8, fontweight="700")
        ax_t.set_xticks(x_pos); ax_t.set_xticklabels(_seasons_lbl, rotation=35, ha="right", color="#8b949e", fontsize=8)
        ax_t.axhline(0, color="#484f58", linewidth=0.8, linestyle="--")
        ax_t.set_title(f"{_sel_player} — {_metric_choice} ({_norm_choice}) by season",
                       color="#e8eaf0", fontsize=11, fontweight="800", pad=10)
        ax_t.set_ylabel(f"{_metric_choice} ({_norm_choice})", color="#484f58", fontsize=9)
        ax_t.tick_params(colors="#484f58", labelsize=8)
        for sp in ax_t.spines.values(): sp.set_visible(False)
        plt.tight_layout(); st.pyplot(fig_t, use_container_width=True); plt.close(fig_t)

        # season summary cards
        best_s = _pm.loc[_pm[_ycol].idxmax(), "season"].replace("WSL_","WSL ")
        career_tot_xt   = _pm["total_xt"].sum()
        career_tot_vaep = _pm["total_vaep"].sum()
        career_acts     = _pm["n_actions"].sum()
        cc1, cc2, cc3, cc4 = st.columns(4)
        def _xcard(col, lbl, val, fmt, clr):
            col.markdown(
                f"<div style='background:#161b22;border-radius:10px;padding:14px 18px;text-align:center'>"
                f"<div style='color:#8b949e;font-size:.75rem;margin-bottom:4px'>{lbl}</div>"
                f"<div style='color:{clr};font-size:1.5rem;font-weight:900'>{fmt.format(val)}</div>"
                f"</div>", unsafe_allow_html=True)
        _xcard(cc1, "Career total xT",   career_tot_xt,   "{:.2f}",  "#3b82f6")
        _xcard(cc2, "Career total VAEP", career_tot_vaep, "{:+.2f}", "#a78bfa")
        _xcard(cc3, "Total actions",     career_acts,     "{:,}",    "#f59e0b")
        _xcard(cc4, f"Best season ({_metric_choice})", _pm[_ycol].max(), "{:.3f}", _mcolor)

    st.markdown("---")

    # ── pitch heatmap of xT-generating actions ────────────────────────────────
    st.markdown("### xT action heatmap")
    if not _actions.empty:
        _hm_seasons = sorted(_summary["season"].unique().tolist())
        _hm_sel = st.multiselect("Seasons", [s.replace("WSL_","WSL ") for s in _hm_seasons],
                                 default=[s.replace("WSL_","WSL ") for s in _hm_seasons[-2:]])
        _hm_sel_raw = [s.replace("WSL ","WSL_") for s in _hm_sel]

        _pa = _actions[
            (_actions["player"] == _sel_player) &
            (_actions["season"].isin(_hm_sel_raw)) &
            (_actions["xt_value"] > 0)
        ].copy()

        if len(_pa) > 0:
            _pitch = Pitch(pitch_type="opta", pitch_color="#161b22", line_color="#30363d")
            fig_h, ax_h = _pitch.draw(figsize=(12, 7))
            fig_h.patch.set_facecolor(FIG_BG)
            _pitch.kdeplot(_pa["x"], _pa["y"], ax=ax_h, cmap="Blues",
                           fill=True, levels=100, alpha=0.7, thresh=0.05)
            sc = ax_h.scatter(_pa["x"], _pa["y"], c=_pa["xt_value"],
                              cmap="YlOrRd", s=18, alpha=0.6, edgecolors="none",
                              vmin=0, vmax=_pa["xt_value"].quantile(0.95))
            plt.colorbar(sc, ax=ax_h, label="xT value", shrink=0.6)
            ax_h.set_title(f"{_sel_player} — positive xT actions ({', '.join(_hm_sel)})",
                           color="#e8eaf0", fontsize=11, fontweight="800", pad=10)
            plt.tight_layout(); st.pyplot(fig_h, use_container_width=True); plt.close(fig_h)
        else:
            st.info("No positive-xT actions found for this player/season selection.")

    st.markdown("---")

    # ── league-wide leaderboard ───────────────────────────────────────────────
    st.markdown("### League leaderboard")
    lb1, lb2, lb3 = st.columns([1, 1, 1])
    with lb1:
        _lb_metric = st.selectbox("Metric", ["total_xt","xt_per_action","total_vaep","vaep_per90","vaep_per_action"], key="lb_metric")
    with lb2:
        _lb_season = st.selectbox("Season", ["All seasons"] + [s.replace("WSL_","WSL ") for s in sorted(_summary["season"].unique())], key="lb_season")
    with lb3:
        _lb_min = st.number_input("Min actions", value=300, step=50, key="lb_min")

    _lb_df = _summary.copy()
    if _lb_season != "All seasons":
        _lb_df = _lb_df[_lb_df["season"] == _lb_season.replace("WSL ","WSL_")]
        _lb_agg = _lb_df[_lb_df["n_actions"] >= _lb_min].nlargest(20, _lb_metric)[
            ["player","team","season","n_actions","total_xt","xt_per_action","total_vaep","vaep_per90"]]
    else:
        _lb_agg = _lb_df.groupby(["player","team"]).agg(
            n_actions=("n_actions","sum"),
            total_xt=("total_xt","sum"),
            total_vaep=("total_vaep","sum"),
        ).reset_index()
        _lb_agg["xt_per_action"]  = _lb_agg["total_xt"]   / _lb_agg["n_actions"]
        _lb_agg["vaep_per_action"]= _lb_agg["total_vaep"]  / _lb_agg["n_actions"]
        _lb_agg["vaep_per90"]     = _lb_agg["vaep_per_action"] * 90
        _lb_agg = _lb_agg[_lb_agg["n_actions"] >= _lb_min].nlargest(20, _lb_metric)[
            ["player","team","n_actions","total_xt","xt_per_action","total_vaep","vaep_per90"]]

    st.dataframe(_lb_agg.round(4), use_container_width=True, hide_index=True)

elif st.session_state.page == "match_metrics":
    back_btn()
    st.markdown("""
<div style='margin-bottom:18px'>
<span style='font-size:2rem;font-weight:900;color:#e8eaf0'>Match Metrics</span><br>
<span style='color:#8b949e;font-size:.9rem'>199 per-team metrics for any WSL match across all seasons.</span>
</div>""", unsafe_allow_html=True)

    _mm = _load_match_metrics()
    if _mm.empty:
        st.error("all_match_metrics.csv not found — run: python extract_match_metrics.py --all")
        st.stop()

    # ── selectors ────────────────────────────────────────────────────────────
    mm_c1, mm_c2 = st.columns([1, 2])
    with mm_c1:
        _mm_seasons = sorted(_mm["match_id"].str.extract(r'^(\d{4}-\d{2}-\d{2})')[0].dropna().str[:4].unique().tolist(), reverse=True)
        _mm_season_sel = st.selectbox("Season (year)", _mm_seasons)
    with mm_c2:
        _mm_season_mask = _mm["match_id"].str.startswith(_mm_season_sel)
        _mm_matches = sorted(_mm[_mm_season_mask]["match_id"].unique().tolist())
        _mm_match_labels = [m.replace("_"," — ",1).replace("-"," ",2) for m in _mm_matches]
        _mm_idx = st.selectbox("Match", range(len(_mm_matches)), format_func=lambda i: _mm_matches[i].split("_",1)[1].replace(" - "," vs ") if "_" in _mm_matches[i] else _mm_matches[i])

    _sel_match = _mm_matches[_mm_idx]
    _sel_rows  = _mm[_mm["match_id"] == _sel_match].copy()

    if _sel_rows.empty:
        st.warning("No data for this match.")
        st.stop()

    # ── metric categories ─────────────────────────────────────────────────────
    _CATS = {
        "Passing":     [c for c in _sel_rows.columns if any(c.startswith(p) for p in
                        ("passes_","pass_","crosses","through_","free_kick_pass","throw_in_pass",
                         "counterattack_pass","assists","deep_","key_pass"))],
        "Shooting":    [c for c in _sel_rows.columns if any(c.startswith(p) for p in
                        ("shots_","goals","shot_","conversion_","big_chance","penalty","penalties",
                         "avg_shot","first_shot","first_goal"))],
        "Defending":   [c for c in _sel_rows.columns if any(c.startswith(p) for p in
                        ("tackles","interception","clearance","block","aerial","fouls","yellow_card",
                         "red_card","gk_save","def_action","errors_","offsides","opp_shot","opp_goal","opp_on_target"))],
        "Possession":  [c for c in _sel_rows.columns if any(c.startswith(p) for p in
                        ("dribble","ball_","total_actions","possession","actions_","opp_half","ppda"))],
        "Sequences":   [c for c in _sel_rows.columns if c.startswith("seq") or c.startswith("sequence") or c=="passes_per_sequence"],
        "Set Pieces":  [c for c in _sel_rows.columns if any(c.startswith(p) for p in
                        ("corners","free_kicks","throw_ins","goal_kick","direct_free","set_piece",
                         "goals_from_corner","goals_from_free","shots_from_set","gk_throw","gk_distrib","penalties_scored","penalties_conceded"))],
        "Top Players": [c for c in _sel_rows.columns if c.startswith("top_")],
        "Result":      [c for c in _sel_rows.columns if any(c.startswith(p) for p in
                        ("result","goals_scored","goals_conceded","ht_","goal_difference","clean_","came_","threw_",
                         "shots_ratio","on_target_ratio","unique_players","substitutions"))],
        "Ratios":      [c for c in _sel_rows.columns if any(c.startswith(p) for p in
                        ("shots_per","progressive_pass","passes_into_box_pct","pass_into_ft","def_vs",
                         "recovery_rate","attack_third","pressure_index","shot_quality","box_entry",
                         "shot_conversion","cross_to","long_pass_accuracy","dribble_to","aerial_dominance",
                         "actions_per_minute","shot_dominance"))],
        "Match Meta":  [c for c in _sel_rows.columns if c.startswith("match_") or c.startswith("period")],
    }
    # dedup
    _seen = set()
    for cat in _CATS:
        _CATS[cat] = [c for c in _CATS[cat] if c not in _seen and not _seen.add(c)]

    # ── header score card ─────────────────────────────────────────────────────
    _teams = _sel_rows["team"].tolist()
    _row_h = _sel_rows[_sel_rows["side"]=="home"].iloc[0] if "home" in _sel_rows["side"].values else _sel_rows.iloc[0]
    _row_a = _sel_rows[_sel_rows["side"]=="away"].iloc[0] if "away" in _sel_rows["side"].values else _sel_rows.iloc[1]

    st.markdown(f"""
<div style='background:#161b22;border-radius:14px;padding:24px;text-align:center;margin-bottom:20px'>
  <div style='display:flex;justify-content:space-between;align-items:center'>
    <div style='font-size:1.3rem;font-weight:800;color:#e8eaf0;flex:1;text-align:left'>{_row_h.get('team','Home')}</div>
    <div style='font-size:2.8rem;font-weight:900;color:#3b82f6;padding:0 32px'>
      {int(_row_h.get('goals_scored',0))} – {int(_row_a.get('goals_scored',0))}
    </div>
    <div style='font-size:1.3rem;font-weight:800;color:#e8eaf0;flex:1;text-align:right'>{_row_a.get('team','Away')}</div>
  </div>
  <div style='color:#484f58;font-size:.8rem;margin-top:8px'>{_sel_match.split("_")[0] if "_" in _sel_match else ""} · HT {int(_row_h.get('ht_goals_scored',0))}–{int(_row_a.get('ht_goals_scored',0))}</div>
</div>""", unsafe_allow_html=True)

    # ── key metrics comparison bar ─────────────────────────────────────────────
    _KEY_METRICS = [
        ("Possession %","possession_pct","#3b82f6"),
        ("Shots","shots_total","#f59e0b"),
        ("Shots on target","shots_on_target","#10b981"),
        ("Pass accuracy %","pass_accuracy_pct","#a78bfa"),
        ("Passes","passes_total","#f97316"),
        ("Tackles won","tackles_won","#ef4444"),
        ("Big chances","big_chances","#fcd34d"),
        ("PPDA","ppda","#6ee7b7"),
    ]
    _n_key = len(_KEY_METRICS)
    fig_key, axes_key = plt.subplots(1, _n_key, figsize=(18, 2.8))
    fig_key.patch.set_facecolor(FIG_BG)
    hn, an = _row_h.get('team','Home'), _row_a.get('team','Away')
    for i, (lbl, col, clr) in enumerate(_KEY_METRICS):
        ax = axes_key[i]; ax.set_facecolor("#161b22")
        hv = float(_row_h.get(col, 0) or 0)
        av = float(_row_a.get(col, 0) or 0)
        total = hv + av or 1
        ax.barh([1,0], [hv/total*100, av/total*100], color=[clr,"#30363d"], height=0.5, edgecolor="none")
        ax.set_xlim(0,100); ax.set_yticks([0,1])
        ax.set_yticklabels([an[:12], hn[:12]], color="#8b949e", fontsize=7)
        ax.set_title(lbl, color="#e8eaf0", fontsize=8, fontweight="700", pad=6)
        ax.text(hv/total*100+1, 1, f"{hv:g}", va='center', ha='left', color=clr,  fontsize=8, fontweight="800")
        ax.text(av/total*100+1, 0, f"{av:g}", va='center', ha='left', color="#8b949e", fontsize=8)
        ax.tick_params(left=False, bottom=False, labelbottom=False)
        for sp in ax.spines.values(): sp.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_key, use_container_width=True)
    plt.close(fig_key)

    st.markdown("---")

    # ── category tabs ─────────────────────────────────────────────────────────
    _tab_names = [k for k, v in _CATS.items() if v]
    _tabs = st.tabs(_tab_names)

    _CAT_COLORS = {"Passing":"#3b82f6","Shooting":"#f59e0b","Defending":"#ef4444",
                   "Possession":"#10b981","Sequences":"#a78bfa","Set Pieces":"#f97316",
                   "Top Players":"#fcd34d","Result":"#6ee7b7","Ratios":"#818cf8","Match Meta":"#6b7280"}

    for tab, cat_name in zip(_tabs, _tab_names):
        cols_in_cat = _CATS[cat_name]
        clr = _CAT_COLORS.get(cat_name, C_BLUE)
        with tab:
            # numeric columns → bar chart
            num_cols = [c for c in cols_in_cat if pd.api.types.is_numeric_dtype(_sel_rows[c])]
            str_cols = [c for c in cols_in_cat if c not in num_cols]

            if num_cols:
                # bar chart: side-by-side grouped
                _n = len(num_cols)
                _ncols = min(_n, 6)
                _nrows = math.ceil(_n / _ncols)
                fig_c, axes_c = plt.subplots(_nrows, _ncols, figsize=(min(18, _ncols*3), _nrows*2.8))
                fig_c.patch.set_facecolor(FIG_BG)
                axes_flat = np.array(axes_c).flatten() if _n > 1 else [axes_c]
                for j, col in enumerate(num_cols):
                    ax = axes_flat[j]; ax.set_facecolor("#161b22")
                    hv = float(_row_h.get(col, 0) or 0)
                    av = float(_row_a.get(col, 0) or 0)
                    bars = ax.bar([hn[:10], an[:10]], [hv, av], color=[clr,"#484f58"], edgecolor="none", width=0.5)
                    for b, v in zip(bars, [hv, av]):
                        ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(hv,av,0.01)*0.04,
                                f"{v:g}", ha='center', va='bottom', color="#e8eaf0", fontsize=8, fontweight="700")
                    ax.set_title(col.replace("_"," "), color="#8b949e", fontsize=7.5, pad=4)
                    ax.tick_params(colors="#484f58", labelsize=7)
                    for sp in ax.spines.values(): sp.set_visible(False)
                    ax.set_facecolor("#161b22")
                for j in range(len(num_cols), len(axes_flat)):
                    axes_flat[j].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_c, use_container_width=True)
                plt.close(fig_c)

            # string columns as table
            if str_cols:
                _str_data = {c: [_row_h.get(c,''), _row_a.get(c,'')] for c in str_cols}
                _str_df = pd.DataFrame(_str_data, index=[hn, an]).T
                st.dataframe(_str_df, use_container_width=True)

    st.markdown("---")

    # ── full flat table ───────────────────────────────────────────────────────
    with st.expander("Full metrics table (all 199)"):
        all_metric_cols = [c for c in _sel_rows.columns if c not in ("match_id","date","side")]
        _flat = _sel_rows.set_index("team")[all_metric_cols].T.reset_index()
        _flat.columns = ["Metric"] + list(_flat.columns[1:])
        st.dataframe(_flat, use_container_width=True, hide_index=True)

# ── PLAYER METRICS PAGE ───────────────────────────────────────────────────────
elif st.session_state.page == "player_metrics":
    st.markdown("## 👤 Player Metrics")

    _pm = _load_player_metrics()
    if _pm.empty:
        st.error("all_player_metrics.csv not found — run: python extract_player_metrics.py --all")
        st.stop()

    META_COLS = ["match_id", "date", "player", "team", "side"]
    METRIC_COLS = [c for c in _pm.columns if c not in META_COLS]

    CATEGORIES = {
        "Passing":          [c for c in METRIC_COLS if c.startswith("pass") or c in ("deep_completions","key_passes","key_pass_rate_pct","through_balls","through_balls_successful","through_ball_accuracy_pct","crosses_total","crosses_successful","cross_accuracy_pct","crosses_left_channel","crosses_right_channel","free_kick_passes","throw_in_passes","throw_in_success_pct","counterattack_passes","avg_pass_start_x","cross_accuracy_rate")],
        "Shooting":         [c for c in METRIC_COLS if c.startswith("shot") or c.startswith("goal") or c in ("xG_total","xG_per_shot","goals_minus_xG","xG_on_target","xG_open_play","xG_set_piece","shot_distance_avg_csv","avg_shot_angle","shots_assisted","shots_regular_play","big_chances","big_chances_scored","big_chances_missed","big_chance_conversion_pct","penalties_taken","penalties_scored","conversion_rate_pct")],
        "Dribbling":        [c for c in METRIC_COLS if c.startswith("dribble") or c.startswith("dribb") or c in ("ball_touches","touches_own_third","touches_mid_third","touches_final_third","touches_in_box","carries_forward","carries_opp_half","progressive_actions")],
        "Defending":        [c for c in METRIC_COLS if c.startswith("tackle") or c.startswith("interc") or c.startswith("clearance") or c.startswith("block") or c.startswith("aerial") or c.startswith("foul") or c in ("yellow_cards","red_cards","errors_leading_shot","errors_leading_goal","defensive_actions_total","def_actions_own_third","def_actions_mid_third","def_actions_final_third","pressures_opp_half","duels_total","duels_won","ball_recoveries","ball_recoveries_own_half","ball_recoveries_opp_half","tackle_interception_ratio")],
        "Positioning":      [c for c in METRIC_COLS if c in ("total_actions","actions_own_third","actions_mid_third","actions_final_third","actions_own_half","actions_opp_half","avg_action_x","avg_action_y","actions_period1","actions_period2","first_action_minute","last_action_minute","minutes_active_proxy","actions_per_minute_active","counterattack_involvements","set_piece_involvements","counterattack_action_pct","set_piece_involvement_pct","opp_half_action_pct","territorial_dominance_index")],
        "Set Pieces":       [c for c in METRIC_COLS if c.startswith("corner") or c.startswith("set_piece") or c.startswith("direct_free") or c.startswith("free_kick") or c.startswith("throw_in") or c.startswith("goal_kick") or c.startswith("penalty") or c in ("goals_from_corners","goals_from_free_kicks")],
        "Goalkeeping":      [c for c in METRIC_COLS if c.startswith("gk_")],
        "Advanced Ratios":  [c for c in METRIC_COLS if c in ("shot_on_target_per_pass","dribble_to_key_pass_ratio","aerial_contribution_index","defensive_load_index","attacking_contribution_index","shot_quality_index","pass_progression_index","pressing_contribution_pct","goal_involvement","xG_involvement","shots_on_target_per_shot","passes_per_action","duel_dominance_index","cross_accuracy_rate","tackle_interception_ratio","shots_per_touch","key_pass_per_touch","counterattack_action_pct","set_piece_involvement_pct","opp_half_action_pct")],
    }

    # ── filters ───────────────────────────────────────────────────────────────
    _col1, _col2, _col3 = st.columns([2, 2, 2])
    with _col1:
        _seasons = sorted(_pm["match_id"].str[:4].unique(), reverse=True)
        _sel_season = st.selectbox("Season start year", ["All"] + list(_seasons))
    with _col2:
        _teams = sorted(_pm["team"].dropna().unique())
        _sel_team = st.selectbox("Team", ["All"] + _teams)
    with _col3:
        _players = sorted(_pm["player"].dropna().unique())
        _sel_player = st.selectbox("Player", ["All"] + _players)

    _filt = _pm.copy()
    if _sel_season != "All":
        _filt = _filt[_filt["match_id"].str.startswith(_sel_season)]
    if _sel_team != "All":
        _filt = _filt[_filt["team"] == _sel_team]
    if _sel_player != "All":
        _filt = _filt[_filt["player"] == _sel_player]

    if _filt.empty:
        st.warning("No data for the selected filters.")
        st.stop()

    # aggregate (sum for counts, mean for ratios/pct)
    _num = _filt[META_COLS[:3] + METRIC_COLS].groupby("player")
    _agg = _num[METRIC_COLS].sum().reset_index()
    _agg_mean = _filt.groupby("player")[METRIC_COLS].mean().reset_index()
    PCT_COLS = [c for c in METRIC_COLS if c.endswith("_pct") or c.endswith("_ratio") or c.endswith("_index") or "per_" in c or c.startswith("avg_") or c.endswith("_proxy")]
    for col in PCT_COLS:
        _agg[col] = _agg_mean[col]

    st.markdown(f"**{len(_filt):,} player-match rows** · **{len(_agg)} unique players** · **250 metrics**")

    # ── category tabs ─────────────────────────────────────────────────────────
    _tabs = st.tabs(list(CATEGORIES.keys()))
    for _tab, (_cat_name, _cat_cols) in zip(_tabs, CATEGORIES.items()):
        with _tab:
            _cat_cols = [c for c in _cat_cols if c in _agg.columns]
            if not _cat_cols:
                st.info("No metrics in this category.")
                continue

            _metric = st.selectbox(f"Metric — {_cat_name}", _cat_cols, key=f"pm_metric_{_cat_name}")
            _top_n = st.slider("Top N players", 5, 30, 15, key=f"pm_topn_{_cat_name}")

            _plot_df = _agg[["player", _metric]].dropna().sort_values(_metric, ascending=False).head(_top_n)

            fig_pm, ax_pm = plt.subplots(figsize=(10, max(4, _top_n * 0.35)))
            fig_pm.patch.set_facecolor("#07090f")
            ax_pm.set_facecolor("#0d1117")
            bars = ax_pm.barh(_plot_df["player"], _plot_df[_metric], color="#a78bfa")
            ax_pm.invert_yaxis()
            ax_pm.set_xlabel(_metric.replace("_", " ").title(), color="#9ca3af")
            ax_pm.tick_params(colors="#9ca3af")
            ax_pm.spines[:].set_color("#1f2937")
            for bar in bars:
                ax_pm.text(bar.get_width() + bar.get_width() * 0.01, bar.get_y() + bar.get_height() / 2,
                           f"{bar.get_width():.2f}", va="center", ha="left", fontsize=8, color="#e8eaf0")
            plt.tight_layout()
            st.pyplot(fig_pm)
            plt.close(fig_pm)

            with st.expander(f"Full {_cat_name} table ({len(_cat_cols)} metrics)"):
                _show = _agg[["player"] + _cat_cols].sort_values(_cat_cols[0], ascending=False)
                st.dataframe(_show, use_container_width=True, hide_index=True)

    # ── full table ────────────────────────────────────────────────────────────
    with st.expander("Full table — all 250 metrics"):
        st.dataframe(_agg.sort_values(METRIC_COLS[0], ascending=False), use_container_width=True, hide_index=True)
