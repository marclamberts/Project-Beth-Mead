import json
import glob
import os
import math
from collections import defaultdict, Counter

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

# ── half-space zone constants ────────────────────────────────────────────────

HS_L = (17, 37)   # left  half-space  y range on Opta 0-100 scale
HS_R = (63, 83)   # right half-space  y range

def in_halfspace(y):
    return HS_L[0] <= y <= HS_L[1] or HS_R[0] <= y <= HS_R[1]

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
]

CAT_ORDER  = ["Passing","Attacking","Movement","Analysis","Comparison","Defensive"]
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
