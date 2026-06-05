"""
Build a rich Excel dashboard from all_shots_full.csv (Eredivisie edition)
Sheets:
  1. 📈 Dashboard         — KPI cards + 6 charts
  2. 📋 README            — column glossary
  3. 📊 Raw Data          — full dataset with conditional formatting
  4. 👤 Player Summary    — career aggregated KPIs (all seasons combined)
  5. 📅 Season Summary    — per-season trends
  6. 🧬 Archetypes        — k-means archetype breakdown
  7. 💪 Pressure Analysis
  8. ⚽ Shot Types
  9. 🎯 Goal Zone Analysis
  10. 🔝 Top Shots        — top 100 shots by psxG
  11. 📊 Player Percentiles

Usage:
    python build_eredivisie_excel.py [csv_path]
Output:
    eredivisie_shot_dashboard.xlsx
"""

import sys
import os
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule, CellIsRule, FormulaRule
from openpyxl.chart import BarChart, LineChart, PieChart, Reference
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_csv(arg=None):
    """Return path to the shots CSV, checking several candidate locations."""
    candidates = []
    if arg:
        candidates.append(arg)
    candidates += [
        os.path.join(SCRIPT_DIR, "all_shots_full.csv"),
        os.path.join(os.getcwd(), "all_shots_full.csv"),
        os.path.expanduser("~/all_shots_full.csv"),
        "/data/all_shots_full.csv",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Could not find all_shots_full.csv. "
        "Pass the path as a command-line argument: "
        "python build_eredivisie_excel.py /path/to/all_shots_full.csv"
    )

CSV_IN  = resolve_csv(sys.argv[1] if len(sys.argv) > 1 else None)
XLSX_OUT = os.path.join(SCRIPT_DIR, "eredivisie_shot_dashboard.xlsx")

# ── Colour palette ──────────────────────────────────────────────────────────
C_HEADER_BG   = "1A2B4A"   # dark navy
C_HEADER_FG   = "FFFFFF"
C_SUBHDR_BG   = "2E5FA3"   # mid blue
C_SUBHDR_FG   = "FFFFFF"
C_ALT1        = "EEF3FA"   # light blue-grey row
C_ALT2        = "FFFFFF"
C_ACCENT      = "E8A020"   # amber
C_GOOD        = "1A7A4A"   # dark green
C_BAD         = "C0392B"   # red
C_GOOD_LIGHT  = "D5F0E0"
C_BAD_LIGHT   = "FAD7D3"
C_NEUTRAL     = "F5F5F5"
C_BORDER      = "B0BEC5"

# ── Style helpers ───────────────────────────────────────────────────────────
def hdr_font(bold=True, size=11, color=C_HEADER_FG):
    return Font(bold=bold, size=size, color=color, name="Calibri")

def body_font(bold=False, size=10, color="000000"):
    return Font(bold=bold, size=size, color=color, name="Calibri")

def fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)

def border(color=C_BORDER):
    s = Side(style='thin', color=color)
    return Border(left=s, right=s, top=s, bottom=s)

def center():
    return Alignment(horizontal="center", vertical="center", wrap_text=True)

def left():
    return Alignment(horizontal="left", vertical="center", wrap_text=True)

def style_header_row(ws, row_num, n_cols, bg=C_HEADER_BG, fg=C_HEADER_FG, height=28):
    ws.row_dimensions[row_num].height = height
    for col in range(1, n_cols + 1):
        cell = ws.cell(row=row_num, column=col)
        cell.font      = Font(bold=True, size=10, color=fg, name="Calibri")
        cell.fill      = fill(bg)
        cell.alignment = center()
        cell.border    = border(C_BORDER)

def shade_rows(ws, start_row, end_row, n_cols):
    for r in range(start_row, end_row + 1):
        bg = C_ALT1 if r % 2 == 0 else C_ALT2
        for c in range(1, n_cols + 1):
            cell = ws.cell(row=r, column=c)
            if cell.fill.fgColor.rgb in ("00000000", "FFFFFFFF", "00FFFFFF"):
                cell.fill = fill(bg)
            cell.border    = border()
            cell.alignment = left()
            cell.font      = body_font()

def autofit(ws, min_w=8, max_w=30):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            try:
                val = str(cell.value) if cell.value is not None else ""
                max_len = max(max_len, len(val))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max(max_len + 2, min_w), max_w)

def write_df(ws, df, start_row=1, start_col=1, hdr_bg=C_HEADER_BG, shade=True):
    cols = list(df.columns)
    n_cols = len(cols)
    for ci, col in enumerate(cols, start_col):
        c = ws.cell(row=start_row, column=ci, value=col)
        c.font      = Font(bold=True, size=10, color=C_HEADER_FG, name="Calibri")
        c.fill      = fill(hdr_bg)
        c.alignment = center()
        c.border    = border()
    ws.row_dimensions[start_row].height = 28
    for ri, row in enumerate(df.itertuples(index=False), start_row + 1):
        for ci, val in enumerate(row, start_col):
            c = ws.cell(row=ri, column=ci, value=val)
            c.border    = border()
            c.alignment = left()
            c.font      = body_font()
            if shade:
                c.fill = fill(C_ALT1 if ri % 2 == 0 else C_ALT2)
    return start_row + len(df)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print(f"Loading data from {CSV_IN} ...")
df = pd.read_csv(CSV_IN)
print(f"  {len(df):,} shots, {len(df.columns)} columns")

# Clean up
df['season']      = df['season'].fillna('Unknown')
df['player_name'] = df['player_name'].fillna('Unknown')
df['archetype']   = df['archetype'].fillna('Unknown')
df['goal_zone']   = df['goal_zone'].fillna('No Frame Data')

# Derived helpers
df['outcome'] = np.select(
    [df['is_goal']==1, df['is_post']==1, df['is_blocked']==1],
    ['Goal', 'Post', 'Saved/Blocked'], default='Miss'
)
df['body_part'] = np.select(
    [df['is_header']==1, df['is_right_foot']==1, df['is_left_foot']==1],
    ['Header', 'Right Foot', 'Left Foot'], default='Other'
)

seasons_list = sorted(df['season'].unique().tolist())
n_seasons    = df['season'].nunique()

# ══════════════════════════════════════════════════════════════════════════════
# BUILD WORKBOOK
# ══════════════════════════════════════════════════════════════════════════════
wb = Workbook()
wb.remove(wb.active)

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 1 (inserted at position 0): Dashboard
# ══════════════════════════════════════════════════════════════════════════════
print("Sheet: Dashboard...")
ws_dash = wb.create_sheet("📈 Dashboard")
ws_dash.sheet_view.showGridLines = False
ws_dash.sheet_view.showRowColHeaders = False

# ── Title ──
ws_dash.merge_cells("A1:R1")
c = ws_dash["A1"]
c.value     = "Eredivisie Shot Analytics Dashboard"
c.font      = Font(bold=True, size=22, color=C_HEADER_FG, name="Calibri")
c.fill      = fill(C_HEADER_BG)
c.alignment = center()
ws_dash.row_dimensions[1].height = 46

ws_dash.merge_cells("A2:R2")
c = ws_dash["A2"]
c.value = (
    f"Dataset: {len(df):,} shots  |  "
    f"{df['player_name'].nunique()} players  |  "
    f"{n_seasons} seasons  |  "
    f"{int(df['is_goal'].sum())} goals  |  "
    f"Seasons: {', '.join(str(s) for s in seasons_list)}"
)
c.font      = Font(bold=False, size=12, color=C_HEADER_FG, name="Calibri")
c.fill      = fill(C_SUBHDR_BG)
c.alignment = center()
ws_dash.row_dimensions[2].height = 24

# ── KPI cards ──
def kpi_card(ws, row, col, label, value, sub=None, bg=C_HEADER_BG, fg=C_HEADER_FG):
    ws.merge_cells(start_row=row,   start_column=col, end_row=row,   end_column=col+2)
    ws.merge_cells(start_row=row+1, start_column=col, end_row=row+1, end_column=col+2)
    if sub:
        ws.merge_cells(start_row=row+2, start_column=col, end_row=row+2, end_column=col+2)
    for r in range(row, row + (3 if sub else 2)):
        ws.row_dimensions[r].height = 22

    c_lbl = ws.cell(row=row,   column=col, value=label)
    c_val = ws.cell(row=row+1, column=col, value=value)
    c_lbl.font = Font(bold=False, size=9,  color="AAAAAA", name="Calibri")
    c_val.font = Font(bold=True,  size=18, color=fg,       name="Calibri")
    c_lbl.fill = fill(bg); c_val.fill = fill(bg)
    c_lbl.alignment = center(); c_val.alignment = center()
    if sub:
        c_sub = ws.cell(row=row+2, column=col, value=sub)
        c_sub.font = Font(bold=False, size=9, color="CCCCCC", name="Calibri")
        c_sub.fill = fill(bg); c_sub.alignment = center()

# Compute KPI values
total_shots      = len(df)
total_goals      = int(df['is_goal'].sum())
total_xg         = df['xg'].sum()
goals_above_xg   = total_goals - total_xg
avg_xg_per_shot  = df['xg'].mean()
conversion_rate  = df['is_goal'].mean() * 100
top_scorer_row   = df.groupby('player_name')['is_goal'].sum().idxmax()
top_scorer_goals = int(df.groupby('player_name')['is_goal'].sum().max())
best_xg_player   = df.groupby('player_name')['xg'].sum().idxmax()
best_xg_total    = df.groupby('player_name')['xg'].sum().max()

kpis = [
    ("Total Shots",          f"{total_shots:,}",                    None),
    ("Total Goals",          f"{total_goals:,}",                    None),
    ("Leagues Seasons",      f"{n_seasons}",                        f"{', '.join(str(s) for s in seasons_list)}"),
    ("Avg xG / Shot",        f"{avg_xg_per_shot:.4f}",              None),
    ("Conversion Rate",      f"{conversion_rate:.1f}%",             None),
    ("Goals Above xG",       f"{goals_above_xg:+.1f}",             None),
    ("Top Scorer (Goals)",   f"{top_scorer_row}",                   f"{top_scorer_goals} goals"),
    ("Best xG Performer",    f"{best_xg_player}",                   f"{best_xg_total:.1f} xG"),
]

kpi_row = 4
for i, (lbl, val, sub) in enumerate(kpis):
    col_start = 1 + i * 3
    if col_start + 2 > 30:
        break
    kpi_card(ws_dash, kpi_row, col_start, lbl, val, sub)

row_after_kpis = kpi_row + (3 if any(k[2] for k in kpis) else 2) + 1
ws_dash.row_dimensions[row_after_kpis - 1].height = 12  # gap

chart_start_row = row_after_kpis + 1
data_col = 22  # hidden chart data starts here

# ── Chart 1: Shot Outcomes (Pie) ──
outcome_data = df['outcome'].value_counts().reset_index()
outcome_data.columns = ['Outcome', 'Count']

ws_dash.cell(row=chart_start_row, column=data_col,   value="Outcome")
ws_dash.cell(row=chart_start_row, column=data_col+1, value="Count")
for ri, row in enumerate(outcome_data.itertuples(index=False), chart_start_row+1):
    ws_dash.cell(row=ri, column=data_col,   value=row[0])
    ws_dash.cell(row=ri, column=data_col+1, value=row[1])

pie = PieChart()
pie.title  = "Shot Outcomes"
pie.style  = 10
pie_labels  = Reference(ws_dash, min_col=data_col,   min_row=chart_start_row+1, max_row=chart_start_row+len(outcome_data))
pie_data    = Reference(ws_dash, min_col=data_col+1, min_row=chart_start_row,   max_row=chart_start_row+len(outcome_data))
pie.add_data(pie_data, titles_from_data=True)
pie.set_categories(pie_labels)
pie.width = 12; pie.height = 10
ws_dash.add_chart(pie, "A9")

# ── Chart 2: xG & Goals by Archetype (Bar) ──
arch_chart_data = (
    df.groupby('archetype')[['xg','is_goal']]
      .agg({'xg':'sum','is_goal':'sum'})
      .reset_index()
      .rename(columns={'xg':'Total xG','is_goal':'Goals'})
      .sort_values('Total xG', ascending=False)
)
dc2 = data_col + 4
ws_dash.cell(row=chart_start_row, column=dc2,   value="Archetype")
ws_dash.cell(row=chart_start_row, column=dc2+1, value="Total xG")
ws_dash.cell(row=chart_start_row, column=dc2+2, value="Goals")
for ri, row in enumerate(arch_chart_data.itertuples(index=False), chart_start_row+1):
    ws_dash.cell(row=ri, column=dc2,   value=row[0])
    ws_dash.cell(row=ri, column=dc2+1, value=round(float(row[1]), 2))
    ws_dash.cell(row=ri, column=dc2+2, value=int(row[2]))

bar = BarChart()
bar.type  = "bar"
bar.title = "Total xG & Goals by Shot Archetype"
bar.y_axis.title = "Value"
bar.style = 10
bar_data = Reference(ws_dash, min_col=dc2+1, max_col=dc2+2,
                     min_row=chart_start_row, max_row=chart_start_row+len(arch_chart_data))
bar_cats = Reference(ws_dash, min_col=dc2, min_row=chart_start_row+1, max_row=chart_start_row+len(arch_chart_data))
bar.add_data(bar_data, titles_from_data=True)
bar.set_categories(bar_cats)
bar.width = 18; bar.height = 12
ws_dash.add_chart(bar, "G9")

# ── Chart 3: Avg xG/Shot & Goal Rate by Season (Line) ──
season_xg = (
    df.groupby('season')[['xg','is_goal']]
      .mean()
      .reset_index()
      .sort_values('season')
)
season_xg.columns = ['Season', 'Avg xG/Shot', 'Avg Goal Rate']
dc3 = data_col + 8
ws_dash.cell(row=chart_start_row, column=dc3,   value="Season")
ws_dash.cell(row=chart_start_row, column=dc3+1, value="Avg xG/Shot")
ws_dash.cell(row=chart_start_row, column=dc3+2, value="Goal Rate")
for ri, row in enumerate(season_xg.itertuples(index=False), chart_start_row+1):
    ws_dash.cell(row=ri, column=dc3,   value=str(row[0]))
    ws_dash.cell(row=ri, column=dc3+1, value=round(float(row[1]), 4))
    ws_dash.cell(row=ri, column=dc3+2, value=round(float(row[2]), 4))

line = LineChart()
line.title   = "Avg xG/Shot & Goal Rate by Eredivisie Season"
line.y_axis.title = "Rate"
line.style   = 10
line_data = Reference(ws_dash, min_col=dc3+1, max_col=dc3+2,
                      min_row=chart_start_row, max_row=chart_start_row+len(season_xg))
line_cats = Reference(ws_dash, min_col=dc3, min_row=chart_start_row+1, max_row=chart_start_row+len(season_xg))
line.add_data(line_data, titles_from_data=True)
line.set_categories(line_cats)
line.width = 18; line.height = 12
ws_dash.add_chart(line, "Q9")

# ── Chart 4: Top 15 Players by xG (Bar) ──
top15 = (
    df.groupby('player_name')['xg']
      .sum()
      .nlargest(15)
      .reset_index()
      .sort_values('xg')
)
top15.columns = ['Player', 'Total xG']
dc4 = data_col + 13
ws_dash.cell(row=chart_start_row, column=dc4,   value="Player")
ws_dash.cell(row=chart_start_row, column=dc4+1, value="Total xG")
for ri, row in enumerate(top15.itertuples(index=False), chart_start_row+1):
    ws_dash.cell(row=ri, column=dc4,   value=row[0])
    ws_dash.cell(row=ri, column=dc4+1, value=round(float(row[1]), 2))

bar2 = BarChart()
bar2.type    = "bar"
bar2.title   = "Top 15 Players — Total xG (Career)"
bar2.y_axis.title = "Total xG"
bar2.style   = 10
bar2_data = Reference(ws_dash, min_col=dc4+1, min_row=chart_start_row, max_row=chart_start_row+15)
bar2_cats = Reference(ws_dash, min_col=dc4,   min_row=chart_start_row+1, max_row=chart_start_row+15)
bar2.add_data(bar2_data, titles_from_data=True)
bar2.set_categories(bar2_cats)
bar2.width = 18; bar2.height = 14
ws_dash.add_chart(bar2, "A25")

# ── Chart 5: Avg xG by Body Part (Bar) ──
bp = df.groupby('body_part')['xg'].mean().reset_index()
bp.columns = ['Body Part', 'Avg xG']
dc5 = data_col + 16
ws_dash.cell(row=chart_start_row, column=dc5,   value="Body Part")
ws_dash.cell(row=chart_start_row, column=dc5+1, value="Avg xG")
for ri, row in enumerate(bp.itertuples(index=False), chart_start_row+1):
    ws_dash.cell(row=ri, column=dc5,   value=row[0])
    ws_dash.cell(row=ri, column=dc5+1, value=round(float(row[1]), 4))

bar3 = BarChart()
bar3.title   = "Avg xG by Body Part"
bar3.y_axis.title = "Avg xG"
bar3.style   = 10
bar3_data = Reference(ws_dash, min_col=dc5+1, min_row=chart_start_row, max_row=chart_start_row+len(bp))
bar3_cats = Reference(ws_dash, min_col=dc5,   min_row=chart_start_row+1, max_row=chart_start_row+len(bp))
bar3.add_data(bar3_data, titles_from_data=True)
bar3.set_categories(bar3_cats)
bar3.width = 12; bar3.height = 10
ws_dash.add_chart(bar3, "Q25")

# ── Chart 6: Goals vs xG — Top 20 Players (Bar) ──
top20 = (
    df.groupby('player_name')
      .agg(xG=('xg','sum'), Goals=('is_goal','sum'))
      .reset_index()
      .nlargest(20, 'xG')
      .sort_values('xG')
)
dc6 = data_col + 19
ws_dash.cell(row=chart_start_row, column=dc6,   value="Player")
ws_dash.cell(row=chart_start_row, column=dc6+1, value="xG")
ws_dash.cell(row=chart_start_row, column=dc6+2, value="Goals")
for ri, row in enumerate(top20.itertuples(index=False), chart_start_row+1):
    ws_dash.cell(row=ri, column=dc6,   value=row[0])
    ws_dash.cell(row=ri, column=dc6+1, value=round(float(row[1]), 2))
    ws_dash.cell(row=ri, column=dc6+2, value=int(row[2]))

bar4 = BarChart()
bar4.type    = "bar"
bar4.title   = "Goals vs xG — Top 20 Players (Career)"
bar4.y_axis.title = "Value"
bar4.style   = 10
bar4_data = Reference(ws_dash, min_col=dc6+1, max_col=dc6+2,
                      min_row=chart_start_row, max_row=chart_start_row+20)
bar4_cats = Reference(ws_dash, min_col=dc6, min_row=chart_start_row+1, max_row=chart_start_row+20)
bar4.add_data(bar4_data, titles_from_data=True)
bar4.set_categories(bar4_cats)
bar4.width = 20; bar4.height = 14
ws_dash.add_chart(bar4, "G25")

# ── Column widths on Dashboard ──
for col in range(1, 32):
    ws_dash.column_dimensions[get_column_letter(col)].width = 5

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 2: README / Glossary
# ══════════════════════════════════════════════════════════════════════════════
print("Sheet: README...")
ws = wb.create_sheet("📋 README")
ws.sheet_view.showGridLines = False
ws.column_dimensions['A'].width = 26
ws.column_dimensions['B'].width = 18
ws.column_dimensions['C'].width = 55

ws.merge_cells("A1:C1")
c = ws["A1"]
c.value     = "Eredivisie Shot Analytics Dashboard — Column Glossary"
c.font      = Font(bold=True, size=16, color=C_HEADER_FG, name="Calibri")
c.fill      = fill(C_HEADER_BG)
c.alignment = center()
ws.row_dimensions[1].height = 38

ws.merge_cells("A2:C2")
c = ws["A2"]
c.value = (
    f"Dataset: {len(df):,} shots  |  "
    f"Players: {df['player_name'].nunique()}  |  "
    f"Seasons: {n_seasons} ({', '.join(str(s) for s in seasons_list)})  |  "
    f"Goals: {int(df['is_goal'].sum())}"
)
c.font      = Font(bold=False, size=11, color=C_HEADER_FG, name="Calibri")
c.fill      = fill(C_SUBHDR_BG)
c.alignment = center()
ws.row_dimensions[2].height = 22

glossary = [
    ("COLUMN", "TYPE", "DESCRIPTION"),
    ("─── IDENTIFIERS ───", "", ""),
    ("match_file",       "text",    "Source JSON file name"),
    ("season",           "text",    "Eredivisie season label"),
    ("player_id",        "text",    "Opta player UUID"),
    ("player_name",      "text",    "Player display name"),
    ("contestant_id",    "text",    "Opta team UUID"),
    ("─── SPATIAL ───", "", ""),
    ("x / y",            "0–100",   "Shot origin on Opta pitch grid (100×100)"),
    ("y_sym",            "0–50",    "Symmetric lateral distance from pitch centre"),
    ("distance",         "metres",  "Euclidean distance to goal centre"),
    ("log_distance",     "float",   "log(distance+1) — compressed for modelling"),
    ("angle",            "degrees", "Open angle subtended by the goal mouth"),
    ("angle_sin",        "0–1",     "sin(angle) — used as model feature"),
    ("─── ZONE FLAGS ───", "", ""),
    ("in_six_yard",      "0/1",     "Shot from within the 6-yard box"),
    ("in_penalty_box",   "0/1",     "Shot from within the penalty area"),
    ("central_y",        "0/1",     "Shot from central corridor (y_sym < 15)"),
    ("─── GOAL FRAME ───", "", ""),
    ("goal_y_raw",       "0–100",   "Opta Q102 — lateral position in goal mouth"),
    ("goal_y_norm",      "-1–1",    "Normalised lateral (clipped); -1=far post, 1=near"),
    ("goal_h_raw",       "0–100",   "Opta Q231 — height in goal mouth (0=ground)"),
    ("goal_h_norm",      "0–1",     "Normalised height"),
    ("goal_frame_dist",  "metres",  "Distance from ball landing to nearest corner"),
    ("corner_zone",      "0/1",     "1 if directed to top corner of goal"),
    ("placement_score",  "0–100",   "Geometry index: how hard the placement was to save"),
    ("goal_zone",        "text",    "6-zone label: Top/Bottom × Left/Centre/Right"),
    ("─── SHOT TYPE ───", "", ""),
    ("is_header",        "0/1",     "Headed shot"),
    ("is_right_foot",    "0/1",     "Right-foot shot"),
    ("is_left_foot",     "0/1",     "Left-foot shot"),
    ("weak_foot",        "0/1",     "Shot taken with weaker foot"),
    ("is_volley",        "0/1",     "Volley"),
    ("is_deflected",     "0/1",     "Deflected before reaching frame"),
    ("is_first_time",    "0/1",     "First-time finish"),
    ("is_big_chance",    "0/1",     "Opta-tagged big chance"),
    ("is_fast_break",    "0/1",     "Counter-attack situation"),
    ("is_from_corner",   "0/1",     "Originated from corner"),
    ("is_free_kick",     "0/1",     "Direct/indirect free kick"),
    ("is_penalty",       "0/1",     "Penalty"),
    ("is_set_piece",     "0/1",     "Any set piece"),
    ("is_open_play",     "0/1",     "Open play"),
    ("is_pull_back",     "0/1",     "Shot from cut-back / pull-back"),
    ("under_pressure",   "0/1",     "Defender within close range"),
    ("is_intentional",   "0/1",     "Intentional shot (not deflection)"),
    ("─── OUTCOMES ───", "", ""),
    ("is_goal",          "0/1",     "Shot resulted in a goal"),
    ("is_on_target",     "0/1",     "Shot on target (saved or goal)"),
    ("is_post",          "0/1",     "Hit the post/crossbar"),
    ("is_blocked",       "0/1",     "Blocked by defender/keeper"),
    ("─── xG MODELS ───", "", ""),
    ("xg",               "0–1",     "Pre-shot Expected Goals (XGBoost)"),
    ("psxg",             "0–1",     "Post-shot xG — conditions on ball placement"),
    ("situation_danger", "0–1",     "Danger level from context alone (no geometry)"),
    ("p_ontarget",       "0–1",     "Predicted probability of being on target"),
    ("xgot",             "0–1",     "xG on Target — given shot reaches frame"),
    ("─── DIFFERENTIALS ───", "", ""),
    ("placement_quality","float",   "psxG − xG: shooter placement above opportunity"),
    ("execution_quality","float",   "xG − situation_danger: positional intelligence"),
    ("ot_over_exp",      "float",   "is_on_target − p_ontarget: accuracy overperformance"),
    ("save_difficulty",  "float",   "psxG for blocked shots: how hard to save"),
    ("finishing_luck",   "float",   "is_goal − psxG: luck above shot quality"),
    ("─── COMPOSITE INDICES ───", "", ""),
    ("technique_index",  "0–60",    "Bonus composite: volley/first-time/big-chance etc."),
    ("shot_power",       "0–100",   "Estimated power index"),
    ("contact_quality",  "0–100",   "Weighted: 50% placement + 30% psxG + 20% technique"),
    ("xg_index",         "0–100",   "xG scaled to 0–100 for display"),
    ("psxg_index",       "0–100",   "psxG scaled to 0–100"),
    ("─── MULTI-OUTCOME MODEL ───", "", ""),
    ("p_miss",           "0–1",     "Predicted P(miss)"),
    ("p_post",           "0–1",     "Predicted P(post)"),
    ("p_save",           "0–1",     "Predicted P(saved)"),
    ("p_goal_mo",        "0–1",     "Predicted P(goal) from 4-class model"),
    ("─── ARCHETYPES & ROLLING ───", "", ""),
    ("archetype_id",     "0–5",     "K-means cluster ID (k=6)"),
    ("archetype",        "text",    "Named shot archetype"),
    ("rolling_xg10",     "0–1",     "Rolling mean xG over last 10 shots (per player)"),
    ("rolling_psxg10",   "0–1",     "Rolling mean psxG over last 10 shots"),
    ("rolling_goals10",  "0–1",     "Rolling goal rate over last 10 shots"),
]

for ri, (col, typ, desc) in enumerate(glossary, 4):
    is_section = col.startswith("───")
    is_hdr     = col == "COLUMN"
    ws.row_dimensions[ri].height = 18 if not is_section else 16
    for ci, val in enumerate([col, typ, desc], 1):
        display_val = val if not is_section else (col if ci == 1 else "")
        c = ws.cell(row=ri, column=ci, value=display_val)
        if is_hdr:
            c.font  = Font(bold=True, size=10, color=C_HEADER_FG, name="Calibri")
            c.fill  = fill(C_SUBHDR_BG)
        elif is_section:
            c.font  = Font(bold=True, size=9, color=C_ACCENT, name="Calibri")
            c.fill  = fill("F0F4FA")
        else:
            c.font  = body_font(size=9)
            c.fill  = fill(C_ALT1 if ri % 2 == 0 else C_ALT2)
        c.alignment = left()
        c.border    = border()

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 3: Raw Data
# ══════════════════════════════════════════════════════════════════════════════
print("Sheet: Raw Data...")
ws2 = wb.create_sheet("📊 Raw Data")
ws2.sheet_view.showGridLines = False
ws2.freeze_panes = "A2"

display_cols = [
    'season','player_name','match_file','time_min','period_id',
    'x','y','distance','angle','outcome','body_part',
    'xg','psxg','situation_danger','p_ontarget','xgot',
    'placement_quality','execution_quality','finishing_luck',
    'placement_score','contact_quality','shot_power','technique_index',
    'is_header','is_penalty','is_big_chance','is_first_time','is_volley',
    'under_pressure','is_fast_break','is_deflected','weak_foot',
    'p_miss','p_post','p_save','p_goal_mo',
    'archetype','rolling_xg10','rolling_psxg10',
    'goal_zone','corner_zone','goal_y_norm','goal_h_norm'
]
display_cols = [c for c in display_cols if c in df.columns]
raw = df[display_cols].copy()

float_cols = raw.select_dtypes(include='float64').columns
for c in float_cols:
    raw[c] = raw[c].round(4)

n_cols = len(display_cols)
style_header_row(ws2, 1, n_cols)
for ci, col in enumerate(display_cols, 1):
    ws2.cell(row=1, column=ci, value=col)

for ri, row in enumerate(raw.itertuples(index=False), 2):
    for ci, val in enumerate(row, 1):
        c = ws2.cell(
            row=ri, column=ci,
            value=val if not (isinstance(val, float) and np.isnan(val)) else None
        )
        c.fill      = fill(C_ALT1 if ri % 2 == 0 else C_ALT2)
        c.border    = border()
        c.alignment = left()
        c.font      = body_font(size=9)

xg_col = display_cols.index('xg') + 1
ws2.conditional_formatting.add(
    f"{get_column_letter(xg_col)}2:{get_column_letter(xg_col)}{len(raw)+1}",
    ColorScaleRule(start_type='min', start_color='FFFFFF',
                   end_type='max',   end_color='1A7A4A')
)
out_col = display_cols.index('outcome') + 1
ws2.conditional_formatting.add(
    f"{get_column_letter(out_col)}2:{get_column_letter(out_col)}{len(raw)+1}",
    CellIsRule(operator='equal', formula=['"Goal"'],
               fill=fill(C_GOOD_LIGHT), font=Font(bold=True, color=C_GOOD, name="Calibri"))
)
autofit(ws2, min_w=8, max_w=22)
ws2.column_dimensions['A'].width = 12
ws2.column_dimensions['B'].width = 22

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 4: Player Summary (career — all seasons combined)
# ══════════════════════════════════════════════════════════════════════════════
print("Sheet: Player Summary (career)...")
ws3 = wb.create_sheet("👤 Player Summary")
ws3.sheet_view.showGridLines = False
ws3.freeze_panes = "A2"

# Title banner
ws3.merge_cells("A1:V1")
c = ws3["A1"]
c.value     = "Eredivisie Player Career Summary (all seasons combined)"
c.font      = Font(bold=True, size=13, color=C_HEADER_FG, name="Calibri")
c.fill      = fill(C_HEADER_BG)
c.alignment = center()
ws3.row_dimensions[1].height = 30

player_agg = df.groupby('player_name').agg(
    Shots          = ('is_goal', 'count'),
    Goals          = ('is_goal', 'sum'),
    On_Target      = ('is_on_target', 'sum'),
    Posts          = ('is_post', 'sum'),
    Blocked        = ('is_blocked', 'sum'),
    xG_Total       = ('xg', 'sum'),
    xG_per_Shot    = ('xg', 'mean'),
    psxG_Total     = ('psxg', 'sum'),
    psxG_per_Shot  = ('psxg', 'mean'),
    Goals_minus_xG = ('finishing_luck', 'sum'),
    Placement_Qual = ('placement_quality', 'mean'),
    Contact_Quality= ('contact_quality', 'mean'),
    Shot_Power     = ('shot_power', 'mean'),
    Avg_Distance   = ('distance', 'mean'),
    Avg_xGOT       = ('xgot', 'mean'),
    Big_Chances    = ('is_big_chance', 'sum'),
    Penalties      = ('is_penalty', 'sum'),
    Headers        = ('is_header', 'sum'),
    Under_Pressure = ('under_pressure', 'sum'),
    Seasons_Active = ('season', 'nunique'),
).reset_index()

player_agg['Conversion_pct']   = (player_agg['Goals'] / player_agg['Shots'] * 100).round(1)
player_agg['OT_pct']           = (player_agg['On_Target'] / player_agg['Shots'] * 100).round(1)
for col in ['xG_Total','psxG_Total']:
    player_agg[col] = player_agg[col].round(2)
for col in ['xG_per_Shot','psxG_per_Shot','Placement_Qual','Avg_xGOT']:
    player_agg[col] = player_agg[col].round(4)
player_agg['Goals_minus_xG']   = player_agg['Goals_minus_xG'].round(2)
player_agg['Contact_Quality']  = player_agg['Contact_Quality'].round(1)
player_agg['Shot_Power']       = player_agg['Shot_Power'].round(1)
player_agg['Avg_Distance']     = player_agg['Avg_Distance'].round(1)
player_agg = player_agg.sort_values('xG_Total', ascending=False)

player_agg.columns = [
    'Player','Shots','Goals','On Target','Posts','Blocked',
    'Total xG','xG/Shot','Total psxG','psxG/Shot','Goals−xG',
    'Placement Quality','Contact Quality','Shot Power',
    'Avg Distance (m)','Avg xGOT','Big Chances','Penalties','Headers',
    'Under Pressure','Seasons Active','Conv%','OT%'
]

n_cols = len(player_agg.columns)
style_header_row(ws3, 2, n_cols)
for ci, col in enumerate(player_agg.columns, 1):
    ws3.cell(row=2, column=ci, value=col)
for ri, row in enumerate(player_agg.itertuples(index=False), 3):
    for ci, val in enumerate(row, 1):
        c = ws3.cell(row=ri, column=ci, value=val if not (isinstance(val, float) and np.isnan(val)) else None)
        c.fill      = fill(C_ALT1 if ri % 2 == 0 else C_ALT2)
        c.border    = border()
        c.alignment = left()
        c.font      = body_font(size=9)

for col_name, col_idx in [('Goals−xG', 11), ('Total xG', 7), ('Contact Quality', 13)]:
    col_letter = get_column_letter(col_idx)
    ws3.conditional_formatting.add(
        f"{col_letter}3:{col_letter}{len(player_agg)+2}",
        ColorScaleRule(start_type='min', start_color=C_BAD_LIGHT,
                       mid_type='percentile', mid_value=50, mid_color='FFFFFF',
                       end_type='max', end_color=C_GOOD_LIGHT)
    )
autofit(ws3, min_w=9, max_w=20)

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 5: Season Summary
# ══════════════════════════════════════════════════════════════════════════════
print("Sheet: Season Summary...")
ws4 = wb.create_sheet("📅 Season Summary")
ws4.sheet_view.showGridLines = False

ws4.merge_cells("A1:P1")
c = ws4["A1"]
c.value     = "Eredivisie — Season-by-Season Trends"
c.font      = Font(bold=True, size=13, color=C_HEADER_FG, name="Calibri")
c.fill      = fill(C_HEADER_BG)
c.alignment = center()
ws4.row_dimensions[1].height = 30

season_agg = df.groupby('season').agg(
    Matches        = ('match_file', 'nunique'),
    Shots          = ('is_goal', 'count'),
    Goals          = ('is_goal', 'sum'),
    On_Target      = ('is_on_target', 'sum'),
    xG_Total       = ('xg', 'sum'),
    xG_per_Shot    = ('xg', 'mean'),
    psxG_Total     = ('psxg', 'sum'),
    Avg_Distance   = ('distance', 'mean'),
    Headers_pct    = ('is_header', 'mean'),
    Penalty_pct    = ('is_penalty', 'mean'),
    Pressure_pct   = ('under_pressure', 'mean'),
    BigChance_pct  = ('is_big_chance', 'mean'),
    Contact_Quality= ('contact_quality', 'mean'),
    Players        = ('player_name', 'nunique'),
).reset_index()

season_agg['Conv_pct']    = (season_agg['Goals'] / season_agg['Shots'] * 100).round(1)
season_agg['OT_pct']      = (season_agg['On_Target'] / season_agg['Shots'] * 100).round(1)
season_agg['xG_Total']    = season_agg['xG_Total'].round(1)
season_agg['xG_per_Shot'] = season_agg['xG_per_Shot'].round(4)
season_agg['psxG_Total']  = season_agg['psxG_Total'].round(1)
season_agg['Avg_Distance']= season_agg['Avg_Distance'].round(1)
season_agg['Headers_pct'] = (season_agg['Headers_pct']*100).round(1)
season_agg['Penalty_pct'] = (season_agg['Penalty_pct']*100).round(1)
season_agg['Pressure_pct']= (season_agg['Pressure_pct']*100).round(1)
season_agg['BigChance_pct']=(season_agg['BigChance_pct']*100).round(1)
season_agg['Contact_Quality']= season_agg['Contact_Quality'].round(1)
season_agg = season_agg.sort_values('season')

season_agg.columns = [
    'Season','Matches','Shots','Goals','On Target','Total xG','xG/Shot',
    'Total psxG','Avg Distance (m)','Header %','Penalty %',
    'Under Pressure %','Big Chance %','Avg Contact Quality','Players','Conv%','OT%'
]
n_cols = len(season_agg.columns)
style_header_row(ws4, 2, n_cols)
for ci, col in enumerate(season_agg.columns, 1):
    ws4.cell(row=2, column=ci, value=col)
for ri, row in enumerate(season_agg.itertuples(index=False), 3):
    for ci, val in enumerate(row, 1):
        c = ws4.cell(row=ri, column=ci, value=val)
        c.fill = fill(C_ALT1 if ri % 2 == 0 else C_ALT2)
        c.border = border(); c.alignment = left(); c.font = body_font(size=9)
autofit(ws4)

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 6: Archetypes
# ══════════════════════════════════════════════════════════════════════════════
print("Sheet: Archetypes...")
ws5 = wb.create_sheet("🧬 Archetypes")
ws5.sheet_view.showGridLines = False

arch_agg = df.groupby('archetype').agg(
    Shots           = ('is_goal','count'),
    Goals           = ('is_goal','sum'),
    xG_Total        = ('xg','sum'),
    xG_per_Shot     = ('xg','mean'),
    psxG_per_Shot   = ('psxg','mean'),
    Placement_Score = ('placement_score','mean'),
    Contact_Quality = ('contact_quality','mean'),
    Avg_Distance    = ('distance','mean'),
    On_Target_pct   = ('is_on_target','mean'),
    Pressure_pct    = ('under_pressure','mean'),
    Header_pct      = ('is_header','mean'),
    Goals_minus_xG  = ('finishing_luck','sum'),
).reset_index()

arch_agg['Conv_pct']      = (arch_agg['Goals'] / arch_agg['Shots'] * 100).round(1)
arch_agg['On_Target_pct'] = (arch_agg['On_Target_pct']*100).round(1)
arch_agg['Pressure_pct']  = (arch_agg['Pressure_pct']*100).round(1)
arch_agg['Header_pct']    = (arch_agg['Header_pct']*100).round(1)
for col in ['xG_Total','xG_per_Shot','psxG_per_Shot','Placement_Score',
            'Contact_Quality','Avg_Distance','Goals_minus_xG']:
    arch_agg[col] = arch_agg[col].round(2)
arch_agg = arch_agg.sort_values('Shots', ascending=False)

arch_agg.columns = [
    'Archetype','Shots','Goals','Total xG','xG/Shot','psxG/Shot',
    'Avg Placement Score','Avg Contact Quality','Avg Distance (m)',
    'On Target %','Under Pressure %','Header %','Goals−xG','Conv%'
]
n_cols = len(arch_agg.columns)
style_header_row(ws5, 1, n_cols)
for ci, col in enumerate(arch_agg.columns, 1):
    ws5.cell(row=1, column=ci, value=col)
for ri, row in enumerate(arch_agg.itertuples(index=False), 2):
    for ci, val in enumerate(row, 1):
        c = ws5.cell(row=ri, column=ci, value=val)
        c.fill = fill(C_ALT1 if ri % 2 == 0 else C_ALT2)
        c.border = border(); c.alignment = left(); c.font = body_font(size=9)
autofit(ws5)

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 7: Pressure Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("Sheet: Pressure Analysis...")
ws6 = wb.create_sheet("💪 Pressure Analysis")
ws6.sheet_view.showGridLines = False

press_agg = df.groupby(['player_name','under_pressure']).agg(
    Shots        = ('is_goal','count'),
    Goals        = ('is_goal','sum'),
    xG_per_Shot  = ('xg','mean'),
    psxG_per_Shot= ('psxg','mean'),
    Avg_Distance = ('distance','mean'),
    Contact_Qual = ('contact_quality','mean'),
).reset_index()
press_agg['under_pressure'] = press_agg['under_pressure'].map(
    {0:'Not Under Pressure', 1:'Under Pressure'}
)
press_agg['Conv_pct'] = (press_agg['Goals'] / press_agg['Shots'] * 100).round(1)
press_agg = press_agg[press_agg['Shots'] >= 10].sort_values(['player_name','under_pressure'])
for col in ['xG_per_Shot','psxG_per_Shot','Avg_Distance','Contact_Qual']:
    press_agg[col] = press_agg[col].round(3)
press_agg.columns = [
    'Player','Situation','Shots','Goals','xG/Shot','psxG/Shot',
    'Avg Distance (m)','Contact Quality','Conv%'
]
n_cols = len(press_agg.columns)
style_header_row(ws6, 1, n_cols)
for ci, col in enumerate(press_agg.columns, 1):
    ws6.cell(row=1, column=ci, value=col)
for ri, row in enumerate(press_agg.itertuples(index=False), 2):
    for ci, val in enumerate(row, 1):
        c = ws6.cell(row=ri, column=ci, value=val)
        c.fill = fill(C_ALT1 if ri % 2 == 0 else C_ALT2)
        c.border = border(); c.alignment = left(); c.font = body_font(size=9)
ws6.conditional_formatting.add(
    f"B2:B{len(press_agg)+1}",
    CellIsRule(operator='equal', formula=['"Under Pressure"'],
               fill=fill("FFF3CD"), font=Font(color="856404", name="Calibri", bold=True))
)
autofit(ws6)

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 8: Shot Types
# ══════════════════════════════════════════════════════════════════════════════
print("Sheet: Shot Types...")
ws7 = wb.create_sheet("⚽ Shot Types")
ws7.sheet_view.showGridLines = False

row_offset = 1
for dim in ['body_part','outcome','archetype']:
    grp = df.groupby(dim).agg(
        Shots          = ('is_goal','count'),
        Goals          = ('is_goal','sum'),
        On_Target      = ('is_on_target','sum'),
        xG_per_Shot    = ('xg','mean'),
        psxG_per_Shot  = ('psxg','mean'),
        Avg_Distance   = ('distance','mean'),
        Contact_Quality= ('contact_quality','mean'),
        Placement_Score= ('placement_score','mean'),
    ).reset_index()
    grp['Conv_pct'] = (grp['Goals'] / grp['Shots'] * 100).round(1)
    grp['OT_pct']   = (grp['On_Target'] / grp['Shots'] * 100).round(1)
    for col in ['xG_per_Shot','psxG_per_Shot','Avg_Distance','Contact_Quality','Placement_Score']:
        grp[col] = grp[col].round(3)
    grp = grp.sort_values('Shots', ascending=False)
    grp.columns = [dim.replace('_',' ').title(),'Shots','Goals','On Target',
                   'xG/Shot','psxG/Shot','Avg Distance','Contact Quality',
                   'Placement Score','Conv%','OT%']

    c = ws7.cell(row=row_offset, column=1, value=f"  By {dim.replace('_',' ').title()}")
    c.font = Font(bold=True, size=12, color=C_HEADER_FG, name="Calibri")
    c.fill = fill(C_SUBHDR_BG)
    ws7.merge_cells(start_row=row_offset, start_column=1, end_row=row_offset, end_column=len(grp.columns))
    ws7.row_dimensions[row_offset].height = 24
    row_offset += 1

    n_cols = len(grp.columns)
    style_header_row(ws7, row_offset, n_cols, bg=C_HEADER_BG)
    for ci, col in enumerate(grp.columns, 1):
        ws7.cell(row=row_offset, column=ci, value=col)
    row_offset += 1

    for ri, row in enumerate(grp.itertuples(index=False)):
        for ci, val in enumerate(row, 1):
            c = ws7.cell(row=row_offset, column=ci, value=val)
            c.fill = fill(C_ALT1 if ri % 2 == 0 else C_ALT2)
            c.border = border(); c.alignment = left(); c.font = body_font(size=9)
        row_offset += 1
    row_offset += 2
autofit(ws7)

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 9: Goal Zone Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("Sheet: Goal Zone Analysis...")
ws8 = wb.create_sheet("🎯 Goal Zone Analysis")
ws8.sheet_view.showGridLines = False

zone_agg = df[df['is_on_target']==1].groupby('goal_zone').agg(
    Shots          = ('is_goal','count'),
    Goals          = ('is_goal','sum'),
    Avg_psxG       = ('psxg','mean'),
    Avg_xGOT       = ('xgot','mean'),
    Avg_Placement  = ('placement_score','mean'),
    Contact_Quality= ('contact_quality','mean'),
).reset_index()
zone_agg['Conv_pct'] = (zone_agg['Goals'] / zone_agg['Shots'] * 100).round(1)
for col in ['Avg_psxG','Avg_xGOT','Avg_Placement','Contact_Quality']:
    zone_agg[col] = zone_agg[col].round(3)
zone_agg = zone_agg.sort_values('Goals', ascending=False)
zone_agg.columns = ['Goal Zone','Shots','Goals','Avg psxG','Avg xGOT',
                    'Avg Placement Score','Avg Contact Quality','Conv%']
n_cols = len(zone_agg.columns)
style_header_row(ws8, 1, n_cols)
for ci, col in enumerate(zone_agg.columns, 1):
    ws8.cell(row=1, column=ci, value=col)
for ri, row in enumerate(zone_agg.itertuples(index=False), 2):
    for ci, val in enumerate(row, 1):
        c = ws8.cell(row=ri, column=ci, value=val)
        c.fill = fill(C_ALT1 if ri % 2 == 0 else C_ALT2)
        c.border = border(); c.alignment = left(); c.font = body_font(size=9)
conv_col = get_column_letter(8)
ws8.conditional_formatting.add(
    f"{conv_col}2:{conv_col}{len(zone_agg)+1}",
    ColorScaleRule(start_type='min', start_color='FFFFFF',
                   end_type='max', end_color=C_GOOD_LIGHT)
)
autofit(ws8)

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 10: Top Shots (by psxG)
# ══════════════════════════════════════════════════════════════════════════════
print("Sheet: Top Shots...")
ws9 = wb.create_sheet("🔝 Top Shots")
ws9.sheet_view.showGridLines = False

top_cols = ['season','player_name','time_min','outcome','body_part',
            'xg','psxg','xgot','placement_score','contact_quality',
            'shot_power','placement_quality','distance','angle',
            'archetype','goal_zone','is_big_chance','under_pressure',
            'is_first_time','is_volley','weak_foot']
top_cols = [c for c in top_cols if c in df.columns]
top = df[top_cols].dropna(subset=['psxg']).sort_values('psxg', ascending=False).head(100)
for col in top.select_dtypes(include='float64').columns:
    top[col] = top[col].round(4)

n_cols = len(top.columns)
style_header_row(ws9, 2, n_cols)

ws9.insert_rows(1)
ws9.merge_cells(f"A1:{get_column_letter(n_cols)}1")
c = ws9["A1"]
c.value     = "Eredivisie — Top 100 Shots by Post-Shot xG"
c.font      = Font(bold=True, size=14, color=C_HEADER_FG, name="Calibri")
c.fill      = fill(C_HEADER_BG)
c.alignment = center()
ws9.row_dimensions[1].height = 32

style_header_row(ws9, 2, n_cols)
for ci, col in enumerate(top.columns, 1):
    ws9.cell(row=2, column=ci, value=col)
for ri, row in enumerate(top.itertuples(index=False), 3):
    for ci, val in enumerate(row, 1):
        c = ws9.cell(row=ri, column=ci, value=val if not (isinstance(val, float) and np.isnan(val)) else None)
        c.fill = fill(C_ALT1 if ri % 2 == 0 else C_ALT2)
        c.border = border(); c.alignment = left(); c.font = body_font(size=9)

out_col_idx = list(top.columns).index('outcome') + 1
ws9.conditional_formatting.add(
    f"{get_column_letter(out_col_idx)}3:{get_column_letter(out_col_idx)}{len(top)+2}",
    CellIsRule(operator='equal', formula=['"Goal"'],
               fill=fill("FFF0B3"), font=Font(bold=True, color="7D5A00", name="Calibri"))
)
autofit(ws9, max_w=22)

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 11: Player Percentiles
# ══════════════════════════════════════════════════════════════════════════════
print("Sheet: Player Percentiles...")
ws_pct = wb.create_sheet("📊 Player Percentiles")
ws_pct.sheet_view.showGridLines = False

metrics = ['xg','psxg','situation_danger','p_ontarget','xgot',
           'placement_quality','contact_quality','shot_power',
           'placement_score','distance']
metric_labels = [
    'xG/Shot','psxG/Shot','Situation Danger','P(On Target)','xGOT',
    'Placement Quality','Contact Quality','Shot Power',
    'Placement Score','Avg Distance (m)'
]

# Filter to metrics that exist
valid = [(m, l) for m, l in zip(metrics, metric_labels) if m in df.columns]
metrics       = [m for m, _ in valid]
metric_labels = [l for _, l in valid]

player_means = df.groupby('player_name')[metrics].mean()
player_shots = df.groupby('player_name')['is_goal'].count()
eligible     = player_shots[player_shots >= 10].index
player_means = player_means.loc[player_means.index.isin(eligible)]

pct_df = player_means.rank(pct=True) * 100
pct_df = pct_df.round(1)
pct_df.columns = metric_labels
pct_df = pct_df.reset_index()
pct_df.columns = ['Player'] + metric_labels
pct_df = pct_df.sort_values('xG/Shot', ascending=False)

means_df = player_means.copy()
means_df.columns = [f"{l} (raw)" for l in metric_labels]
means_df = means_df.round(4).reset_index()
means_df.columns = ['Player'] + [f"{l} (raw)" for l in metric_labels]

combined = pct_df.merge(means_df, on='Player')
ordered_cols = ['Player']
for lbl in metric_labels:
    ordered_cols.append(lbl)
    ordered_cols.append(f"{lbl} (raw)")
combined = combined[ordered_cols]

# Title banner
ws_pct.merge_cells(f"A1:{get_column_letter(len(combined.columns))}1")
c = ws_pct["A1"]
c.value     = "Eredivisie Player Percentile Rankings (min. 10 shots)"
c.font      = Font(bold=True, size=13, color=C_HEADER_FG, name="Calibri")
c.fill      = fill(C_HEADER_BG)
c.alignment = center()
ws_pct.row_dimensions[1].height = 30

n_cols = len(combined.columns)
style_header_row(ws_pct, 2, n_cols)
for ci, col in enumerate(combined.columns, 1):
    ws_pct.cell(row=2, column=ci, value=col)
for ri, row in enumerate(combined.itertuples(index=False), 3):
    for ci, val in enumerate(row, 1):
        c = ws_pct.cell(row=ri, column=ci, value=val)
        c.fill = fill(C_ALT1 if ri % 2 == 0 else C_ALT2)
        c.border = border(); c.alignment = left(); c.font = body_font(size=9)

for col_i, col_name in enumerate(combined.columns[1:], 2):
    if '(raw)' not in col_name:
        col_letter = get_column_letter(col_i)
        ws_pct.conditional_formatting.add(
            f"{col_letter}3:{col_letter}{len(combined)+2}",
            ColorScaleRule(start_type='min', start_color=C_BAD_LIGHT,
                           mid_type='percentile', mid_value=50, mid_color='FFFFFF',
                           end_type='max', end_color=C_GOOD_LIGHT)
        )
autofit(ws_pct, max_w=18)

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
print(f"Saving to {XLSX_OUT} ...")
wb.save(XLSX_OUT)
print("Done.")
print(f"Sheets: {[s.title for s in wb.worksheets]}")
print(f"Output: {XLSX_OUT}")
