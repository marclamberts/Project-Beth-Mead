"""
Eredivisie Bayesian Finishing — Scouting Workbook
Reads bayesian_finishing.csv and produces eredivisie_scouting.xlsx
"""
import sys
import math
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                              GradientFill)
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule, FormulaRule
from openpyxl.chart import BarChart, Reference, ScatterChart, Series
from openpyxl.chart.series import SeriesLabel
from openpyxl.chart.label import DataLabelList

# ── Colours ───────────────────────────────────────────────────────────────────
NAVY      = "0D1B2A"
TEAL      = "1B7A78"
GOLD      = "E8C33A"
SILVER    = "C0C0C0"
GREEN_HI  = "1DB954"
RED_HI    = "E05252"
ORANGE    = "F09B2A"
WHITE     = "FFFFFF"
LIGHT_BG  = "F4F7FA"
MID_BG    = "E0E8F0"
DARK_TEXT = "0D1B2A"
GREY_LINE = "C5CDD6"
RANK_GOOD = "C6EFCE"   # Excel-style green
RANK_MID  = "FFEB9C"   # Excel-style yellow
RANK_BAD  = "FFC7CE"   # Excel-style red

def px(hex_str):
    return PatternFill("solid", fgColor=hex_str)

def font(bold=False, size=11, color=DARK_TEXT, italic=False):
    return Font(bold=bold, size=size, color=color, italic=italic,
                name="Calibri")

def border_thin():
    s = Side(style="thin", color=GREY_LINE)
    return Border(left=s, right=s, top=s, bottom=s)

def border_medium():
    s = Side(style="medium", color=NAVY)
    return Border(left=s, right=s, top=s, bottom=s)

def center(wrap=False):
    return Alignment(horizontal="center", vertical="center", wrap_text=wrap)

def left(wrap=False):
    return Alignment(horizontal="left", vertical="center", wrap_text=wrap)

def pct(v, decimals=1):
    if pd.isna(v): return ""
    return f"{v*100:.{decimals}f}%"

def fmt_num(v, decimals=3):
    if pd.isna(v): return ""
    return round(float(v), decimals)

# ── Data prep ─────────────────────────────────────────────────────────────────
CSV = sys.argv[1] if len(sys.argv) > 1 else \
      "/root/.claude/uploads/38f603cc-253e-405e-be34-75fe371fb5e5/70a1abf2-bayesian_finishing.csv"

df = pd.read_csv(CSV)

# Derived columns
df["goals_above_xg"]  = df["goals"] - df["xg_sum"]
df["goals_above_exp_pct"] = df["goals_above_xg"] / df["goals"].clip(1)
df["ci_width"]        = df["eb_ci_hi"] - df["eb_ci_lo"]
df["reliability"]     = 1 - df["shrinkage"]           # 1 = fully reliable
df["xg_per_shot"]     = df["xg_sum"] / df["n"]
df["finishing_edge"]  = df["eb_vs_xg"]                # alias

# Percentile ranks (higher = better)
for col, ascending in [
    ("eb_mean",     False),
    ("eb_vs_xg",    False),
    ("reliability", False),
    ("raw_rate",    False),
    ("n",           False),
]:
    df[f"pct_{col}"] = df[col].rank(pct=True, ascending=ascending) * 100

# Tier labels based on eb_mean + reliability
def tier(row):
    if row["n"] < 10:         return "Insufficient data"
    if row["eb_mean"] >= 0.155 and row["reliability"] >= 0.55: return "Elite"
    if row["eb_mean"] >= 0.130 and row["reliability"] >= 0.40: return "Strong"
    if row["eb_mean"] >= 0.110:                                 return "Average"
    if row["eb_mean"] >= 0.085:                                 return "Below Avg"
    return "Weak"

TIER_ORDER = {"Elite": 0, "Strong": 1, "Average": 2,
              "Below Avg": 3, "Weak": 4, "Insufficient data": 5}
TIER_COLOR = {
    "Elite":             "1DB954",
    "Strong":            "5BA85A",
    "Average":           "A8A83A",
    "Below Avg":         "E07A3A",
    "Weak":              "E05252",
    "Insufficient data": "AAAAAA",
}
df["tier"] = df.apply(tier, axis=1)
df["tier_rank"] = df["tier"].map(TIER_ORDER)

league_mean = (df["eb_alpha_post"] - df["goals"]).mean() / \
              ((df["eb_alpha_post"] - df["goals"]).mean() +
               (df["eb_beta_post"] - (df["n"] - df["goals"])).mean())
# Simpler: use the prior mean (alpha_prior / (alpha_prior + beta_prior))
# alpha_prior = eb_alpha_post - goals, beta_prior = eb_beta_post - (n - goals)
df["alpha_prior"] = df["eb_alpha_post"] - df["goals"]
df["beta_prior"]  = df["eb_beta_post"]  - (df["n"] - df["goals"])
league_mean = (df["alpha_prior"] / (df["alpha_prior"] + df["beta_prior"])).mean()

# Subsets
df_q30  = df[df["n"] >= 30].copy()
df_q10  = df[df["n"] >= 10].copy()
df_all  = df.copy()

# Sort orders
elite_sort  = df_q10.sort_values(["tier_rank", "eb_mean"], ascending=[True, False])
finedge_sort = df_q10.sort_values("eb_vs_xg", ascending=False)
volume_sort  = df_q30.sort_values("eb_mean", ascending=False)
hidden_sort  = df[(df["n"] >= 5) & (df["n"] < 30)].sort_values("eb_mean", ascending=False)

# ── Workbook ──────────────────────────────────────────────────────────────────
wb = Workbook()
wb.remove(wb.active)

# ═════════════════════════════════════════════════════════════════════════════
# Sheet 1: 🏆 Scout Overview
# ═════════════════════════════════════════════════════════════════════════════
def write_banner(ws, row, text, subtitle="", bg=NAVY, fg=WHITE, height=30):
    ws.row_dimensions[row].height = height
    c = ws.cell(row=row, column=1, value=text)
    c.font = Font(bold=True, size=14, color=fg, name="Calibri")
    c.fill = px(bg)
    c.alignment = center()
    if subtitle:
        row += 1
        ws.row_dimensions[row].height = 18
        c2 = ws.cell(row=row, column=1, value=subtitle)
        c2.font = Font(bold=False, size=9, color=fg, name="Calibri", italic=True)
        c2.fill = px(bg)
        c2.alignment = center()
        return row + 1
    return row + 1

def merge_banner(ws, row, col1, col2, text, bg=NAVY, fg=WHITE, size=13, height=28):
    ws.merge_cells(start_row=row, start_column=col1,
                   end_row=row,   end_column=col2)
    c = ws.cell(row=row, column=col1, value=text)
    c.font = Font(bold=True, size=size, color=fg, name="Calibri")
    c.fill = px(bg)
    c.alignment = center()
    ws.row_dimensions[row].height = height
    return row + 1

def kpi_card(ws, row, col, label, value, sub="", bg=TEAL, fg=WHITE):
    ws.merge_cells(start_row=row, start_column=col,
                   end_row=row,   end_column=col+1)
    ws.merge_cells(start_row=row+1, start_column=col,
                   end_row=row+1,   end_column=col+1)
    ws.merge_cells(start_row=row+2, start_column=col,
                   end_row=row+2,   end_column=col+1)
    for r in [row, row+1, row+2]:
        ws.row_dimensions[r].height = 22
    lc = ws.cell(row=row,   column=col, value=label)
    vc = ws.cell(row=row+1, column=col, value=value)
    sc = ws.cell(row=row+2, column=col, value=sub)
    lc.font = Font(bold=False, size=8,  color=fg, name="Calibri")
    vc.font = Font(bold=True,  size=18, color=GOLD, name="Calibri")
    sc.font = Font(bold=False, size=8,  color=fg, name="Calibri", italic=True)
    for c in [lc, vc, sc]:
        c.fill = px(bg); c.alignment = center()

ws1 = wb.create_sheet("🏆 Scout Overview")
ws1.sheet_view.showGridLines = False
ws1.column_dimensions["A"].width = 2

# Title
ws1.merge_cells("B1:P1")
c = ws1.cell(row=1, column=2,
             value="⚽  EREDIVISIE SCOUTING REPORT  —  BAYESIAN FINISHING ANALYSIS")
c.font = Font(bold=True, size=16, color=WHITE, name="Calibri")
c.fill = px(NAVY); c.alignment = center()
ws1.row_dimensions[1].height = 36

ws1.merge_cells("B2:P2")
c = ws1.cell(row=2, column=2,
             value=f"Empirical Bayes Beta-Binomial  |  635 Players  |  Seasons 2022-23, 2023-24, 2024-25  |  League Mean: {league_mean*100:.2f}%")
c.font = Font(bold=False, size=9, color=WHITE, name="Calibri", italic=True)
c.fill = px(TEAL); c.alignment = center()
ws1.row_dimensions[2].height = 18

# KPI row
kpi_data = [
    ("Total Players",        f"{len(df):,}",       "in dataset",              NAVY),
    ("Qualified (≥10 shots)", f"{len(df_q10):,}",  "reliable estimates",      TEAL),
    ("Elite Finishers",       str((df_q10["tier"]=="Elite").sum()), "eb_mean ≥ 15.5%", "1B7A78"),
    ("League Mean Conv.",     f"{league_mean*100:.2f}%", "EB prior mean",      NAVY),
    ("Top EB Mean",           f"{df_q10['eb_mean'].max()*100:.2f}%", df_q10.loc[df_q10['eb_mean'].idxmax(),'player_name'], TEAL),
    ("Best Finishing Edge",   f"+{df_q10['eb_vs_xg'].max()*100:.2f}%", df_q10.loc[df_q10['eb_vs_xg'].idxmax(),'player_name'], "1B7A78"),
    ("Highest Volume",        f"{int(df['n'].max())} shots", df.loc[df['n'].idxmax(),'player_name'], NAVY),
    ("Hidden Gems (5-29)",    str(len(hidden_sort)), "small-sample leaders",   TEAL),
]
col_start = 2
for i, (lbl, val, sub, bg) in enumerate(kpi_data):
    col = col_start + i * 2
    kpi_card(ws1, 4, col, lbl, val, sub, bg=bg)
    ws1.column_dimensions[get_column_letter(col)].width   = 12
    ws1.column_dimensions[get_column_letter(col+1)].width = 2

# ── Tier Summary table ────────────────────────────────────────────────────────
merge_banner(ws1, 8, 2, 10, "TIER DISTRIBUTION  (min 10 shots)", bg=NAVY)
tier_headers = ["Tier", "Players", "Avg EB Mean", "Avg n", "Avg Goals", "Avg xG Sum",
                "Avg Edge", "Avg CI Width", "Description"]
for ci, h in enumerate(tier_headers, 2):
    c = ws1.cell(row=9, column=ci, value=h)
    c.font = font(bold=True, color=WHITE); c.fill = px(TEAL)
    c.alignment = center(); c.border = border_thin()

tier_desc = {
    "Elite":             "Proven Eredivisie finishers — primary targets",
    "Strong":            "Reliable above-average finishers — strong options",
    "Average":           "League-average finishing — positional value key",
    "Below Avg":         "Underperforming expectations — context needed",
    "Weak":              "Poor finishing — limited scouting priority",
    "Insufficient data": "< 10 shots — cannot assess reliably",
}
tier_groups = df_q10.groupby("tier").agg(
    players=("player_name","count"),
    eb_mean=("eb_mean","mean"),
    n=("n","mean"),
    goals=("goals","mean"),
    xg_sum=("xg_sum","mean"),
    edge=("eb_vs_xg","mean"),
    ci_w=("ci_width","mean"),
).reset_index()
tier_groups["rank"] = tier_groups["tier"].map(TIER_ORDER)
tier_groups = tier_groups.sort_values("rank")
# Add insufficient data row
insuf = df[df["tier"] == "Insufficient data"]
extra = pd.DataFrame([{
    "tier": "Insufficient data", "players": len(insuf),
    "eb_mean": insuf["eb_mean"].mean() if len(insuf) else 0,
    "n": insuf["n"].mean() if len(insuf) else 0,
    "goals": insuf["goals"].mean() if len(insuf) else 0,
    "xg_sum": insuf["xg_sum"].mean() if len(insuf) else 0,
    "edge": insuf["eb_vs_xg"].mean() if len(insuf) else 0,
    "ci_w": insuf["ci_width"].mean() if len(insuf) else 0,
    "rank": 5,
}])
tier_groups = pd.concat([tier_groups, extra], ignore_index=True)

for ri, row_data in enumerate(tier_groups.itertuples(), 10):
    tc = TIER_COLOR.get(row_data.tier, "AAAAAA")
    row_vals = [
        row_data.tier,
        int(row_data.players),
        f"{row_data.eb_mean*100:.2f}%",
        f"{row_data.n:.0f}",
        f"{row_data.goals:.1f}",
        f"{row_data.xg_sum:.1f}",
        f"{row_data.edge*100:+.2f}%",
        f"{row_data.ci_w*100:.2f}pp",
        tier_desc.get(row_data.tier, ""),
    ]
    for ci, val in enumerate(row_vals, 2):
        c = ws1.cell(row=ri, column=ci, value=val)
        c.border = border_thin()
        c.alignment = center() if ci != 10 else left()
        if ci == 2:
            c.font = Font(bold=True, size=10, color=WHITE, name="Calibri")
            c.fill = px(tc)
        else:
            c.font = font(size=10)
            c.fill = px(LIGHT_BG if ri % 2 == 0 else WHITE)
    ws1.row_dimensions[ri].height = 18

# ── Method note ───────────────────────────────────────────────────────────────
note_row = 10 + len(tier_groups) + 1
ws1.merge_cells(start_row=note_row, start_column=2, end_row=note_row, end_column=16)
c = ws1.cell(row=note_row, column=2,
             value="ℹ  Empirical Bayes shrinks noisy small-sample rates toward the league prior.  "
                   "eb_mean = Bayesian conversion estimate.  "
                   "eb_vs_xg = finishing skill above xG expectation.  "
                   "Reliability = 1 − shrinkage (1.0 = fully data-driven).")
c.font = Font(size=8, color="555555", italic=True, name="Calibri")
c.alignment = left(wrap=True)
ws1.row_dimensions[note_row].height = 28

# ═════════════════════════════════════════════════════════════════════════════
# Sheet 2: 🎯 Elite Targets
# ═════════════════════════════════════════════════════════════════════════════
ws2 = wb.create_sheet("🎯 Elite Targets")
ws2.sheet_view.showGridLines = False

ws2.merge_cells("A1:N1")
c = ws2.cell(row=1, column=1, value="🎯  ELITE FINISHING TARGETS  —  Sorted by Tier, then EB Mean")
c.font = Font(bold=True, size=14, color=WHITE, name="Calibri")
c.fill = px(NAVY); c.alignment = center(); ws2.row_dimensions[1].height = 32

ws2.merge_cells("A2:N2")
c = ws2.cell(row=2, column=1,
             value="Players with ≥ 10 shots  |  Tiers: Elite (≥15.5% EB) → Strong → Average → Below Avg → Weak")
c.font = Font(size=9, color=WHITE, italic=True, name="Calibri")
c.fill = px(TEAL); c.alignment = center(); ws2.row_dimensions[2].height = 16

headers2 = ["#", "Player", "Tier", "Shots", "Goals", "Raw Conv%",
            "Avg xG/Shot", "EB Mean", "EB 90% CI Low", "EB 90% CI High",
            "EB vs xG", "Shrinkage", "Reliability", "Scouting Note"]
col_w2 = [5, 22, 14, 7, 7, 10, 11, 10, 13, 13, 11, 11, 11, 40]
for ci, (h, w) in enumerate(zip(headers2, col_w2), 1):
    ws2.column_dimensions[get_column_letter(ci)].width = w
    c = ws2.cell(row=3, column=ci, value=h)
    c.font = font(bold=True, color=WHITE, size=9)
    c.fill = px(NAVY); c.alignment = center(); c.border = border_thin()
ws2.row_dimensions[3].height = 20

def scouting_note(row):
    notes = []
    if row["tier"] == "Elite":
        notes.append("Primary target")
    if row["eb_vs_xg"] > 0.03:
        notes.append(f"outperforms xG by {row['eb_vs_xg']*100:.1f}pp")
    elif row["eb_vs_xg"] < -0.03:
        notes.append(f"underperforms xG by {abs(row['eb_vs_xg'])*100:.1f}pp")
    if row["reliability"] > 0.70:
        notes.append("high reliability")
    elif row["reliability"] < 0.40:
        notes.append("small sample — verify")
    if row["ci_width"] < 0.06:
        notes.append("tight CI")
    return " | ".join(notes) if notes else "Meets league standard"

for ri, (_, row) in enumerate(elite_sort.iterrows(), 4):
    tc = TIER_COLOR.get(row["tier"], "AAAAAA")
    row_bg = LIGHT_BG if ri % 2 == 0 else WHITE
    vals = [
        ri - 3,
        row["player_name"],
        row["tier"],
        int(row["n"]),
        int(row["goals"]),
        pct(row["raw_rate"]),
        pct(row["xg_per_shot"]),
        pct(row["eb_mean"]),
        pct(row["eb_ci_lo"]),
        pct(row["eb_ci_hi"]),
        f"{row['eb_vs_xg']*100:+.2f}%",
        pct(row["shrinkage"]),
        pct(row["reliability"]),
        scouting_note(row),
    ]
    for ci, val in enumerate(vals, 1):
        c = ws2.cell(row=ri, column=ci, value=val)
        c.border = border_thin(); c.font = font(size=9)
        if ci == 3:
            c.font = Font(bold=True, size=9, color=WHITE, name="Calibri")
            c.fill = px(tc); c.alignment = center()
        elif ci in (8, 11):
            # EB mean and edge — colour code
            c.fill = px(row_bg)
            c.alignment = center()
        elif ci == 14:
            c.fill = px(row_bg); c.alignment = left(wrap=True)
        else:
            c.fill = px(row_bg); c.alignment = center()
    ws2.row_dimensions[ri].height = 16

# Freeze header
ws2.freeze_panes = "A4"

# Conditional format: EB Mean column (H = col 8)
max_r = 3 + len(elite_sort)
col_h = get_column_letter(8)
col_k = get_column_letter(11)
ws2.conditional_formatting.add(
    f"{col_h}4:{col_h}{max_r}",
    ColorScaleRule(start_type="min", start_color="FFC7CE",
                   mid_type="percentile", mid_value=50, mid_color="FFEB9C",
                   end_type="max", end_color="C6EFCE")
)
ws2.conditional_formatting.add(
    f"{col_k}4:{col_k}{max_r}",
    ColorScaleRule(start_type="min", start_color="FFC7CE",
                   mid_type="num", mid_value=0, mid_color="FFFFFF",
                   end_type="max", end_color="C6EFCE")
)

# ═════════════════════════════════════════════════════════════════════════════
# Sheet 3: 📈 Finishing Edge (EB vs xG)
# ═════════════════════════════════════════════════════════════════════════════
ws3 = wb.create_sheet("📈 Finishing Edge")
ws3.sheet_view.showGridLines = False

ws3.merge_cells("A1:L1")
c = ws3.cell(row=1, column=1, value="📈  FINISHING EDGE  —  EB Posterior Mean  vs  Expected Goals Rate")
c.font = Font(bold=True, size=14, color=WHITE, name="Calibri")
c.fill = px(NAVY); c.alignment = center(); ws3.row_dimensions[1].height = 32

ws3.merge_cells("A2:L2")
c = ws3.cell(row=2, column=1,
             value="Positive edge = finishes better than xG predicts.  Sorted by eb_vs_xg DESC.  Min 10 shots.")
c.font = Font(size=9, color=WHITE, italic=True, name="Calibri")
c.fill = px(TEAL); c.alignment = center(); ws3.row_dimensions[2].height = 16

headers3 = ["#", "Player", "Tier", "Shots", "Goals", "xG Sum",
            "Avg xG/Shot", "EB Mean", "EB vs xG", "CI Low", "CI High", "CI Width"]
col_w3   = [5, 22, 14, 7, 7, 10, 11, 10, 11, 10, 10, 10]
for ci, (h, w) in enumerate(zip(headers3, col_w3), 1):
    ws3.column_dimensions[get_column_letter(ci)].width = w
    c = ws3.cell(row=3, column=ci, value=h)
    c.font = font(bold=True, color=WHITE, size=9)
    c.fill = px(NAVY); c.alignment = center(); c.border = border_thin()
ws3.row_dimensions[3].height = 20

for ri, (_, row) in enumerate(finedge_sort.iterrows(), 4):
    tc = TIER_COLOR.get(row["tier"], "AAAAAA")
    row_bg = LIGHT_BG if ri % 2 == 0 else WHITE
    edge_pos = row["eb_vs_xg"] >= 0
    vals = [
        ri - 3,
        row["player_name"],
        row["tier"],
        int(row["n"]),
        int(row["goals"]),
        f"{row['xg_sum']:.2f}",
        pct(row["xg_per_shot"]),
        pct(row["eb_mean"]),
        f"{row['eb_vs_xg']*100:+.2f}%",
        pct(row["eb_ci_lo"]),
        pct(row["eb_ci_hi"]),
        f"{row['ci_width']*100:.2f}pp",
    ]
    for ci, val in enumerate(vals, 1):
        c = ws3.cell(row=ri, column=ci, value=val)
        c.border = border_thin(); c.font = font(size=9); c.alignment = center()
        if ci == 3:
            c.font = Font(bold=True, size=9, color=WHITE, name="Calibri")
            c.fill = px(tc)
        elif ci == 9:
            c.fill = px("C6EFCE" if edge_pos else "FFC7CE")
            c.font = Font(bold=True, size=9,
                         color="006100" if edge_pos else "9C0006",
                         name="Calibri")
        else:
            c.fill = px(row_bg)

    ws3.row_dimensions[ri].height = 16

ws3.freeze_panes = "A4"

# ═════════════════════════════════════════════════════════════════════════════
# Sheet 4: 🔍 Hidden Gems (5-29 shots)
# ═════════════════════════════════════════════════════════════════════════════
ws4 = wb.create_sheet("🔍 Hidden Gems")
ws4.sheet_view.showGridLines = False

ws4.merge_cells("A1:K1")
c = ws4.cell(row=1, column=1, value="🔍  HIDDEN GEMS  —  Small-Sample High EB Mean (5–29 shots)")
c.font = Font(bold=True, size=14, color=WHITE, name="Calibri")
c.fill = px(NAVY); c.alignment = center(); ws4.row_dimensions[1].height = 32

ws4.merge_cells("A2:K2")
c = ws4.cell(row=2, column=1,
             value="High shrinkage — treat as leads to investigate further, not confirmed finishers.  "
                   "Sorted by EB mean DESC.")
c.font = Font(size=9, color=WHITE, italic=True, name="Calibri")
c.fill = px(ORANGE); c.alignment = center(); ws4.row_dimensions[2].height = 16

headers4 = ["#", "Player", "Shots", "Goals", "Raw Conv%",
            "Avg xG/Shot", "EB Mean", "EB vs xG", "Shrinkage",
            "Reliability", "Verdict"]
col_w4 = [5, 22, 7, 7, 10, 11, 10, 11, 10, 10, 35]
for ci, (h, w) in enumerate(zip(headers4, col_w4), 1):
    ws4.column_dimensions[get_column_letter(ci)].width = w
    c = ws4.cell(row=3, column=ci, value=h)
    c.font = font(bold=True, color=WHITE, size=9)
    c.fill = px(ORANGE); c.alignment = center(); c.border = border_thin()
ws4.row_dimensions[3].height = 20

def hidden_verdict(row):
    if row["eb_mean"] >= 0.15:
        return "🔥 Strong signal — scout immediately"
    if row["eb_mean"] >= 0.13 and row["eb_vs_xg"] > 0.01:
        return "✅ Promising — outperforms xG, needs more data"
    if row["eb_mean"] >= 0.12:
        return "👀 Interesting — follow for next 10+ shots"
    return "⏳ Inconclusive — insufficient evidence"

for ri, (_, row) in enumerate(hidden_sort.iterrows(), 4):
    row_bg = LIGHT_BG if ri % 2 == 0 else WHITE
    vals = [
        ri - 3,
        row["player_name"],
        int(row["n"]),
        int(row["goals"]),
        pct(row["raw_rate"]),
        pct(row["xg_per_shot"]),
        pct(row["eb_mean"]),
        f"{row['eb_vs_xg']*100:+.2f}%",
        pct(row["shrinkage"]),
        pct(row["reliability"]),
        hidden_verdict(row),
    ]
    for ci, val in enumerate(vals, 1):
        c = ws4.cell(row=ri, column=ci, value=val)
        c.border = border_thin(); c.font = font(size=9); c.alignment = center()
        if ci == 11:
            c.alignment = left()
        c.fill = px(row_bg)
    ws4.row_dimensions[ri].height = 16

ws4.freeze_panes = "A4"

# ═════════════════════════════════════════════════════════════════════════════
# Sheet 5: 📊 Full Rankings
# ═════════════════════════════════════════════════════════════════════════════
ws5 = wb.create_sheet("📊 Full Rankings")
ws5.sheet_view.showGridLines = False

ws5.merge_cells("A1:O1")
c = ws5.cell(row=1, column=1, value="📊  FULL PLAYER RANKINGS  —  All 635 Players, Sorted by EB Mean")
c.font = Font(bold=True, size=14, color=WHITE, name="Calibri")
c.fill = px(NAVY); c.alignment = center(); ws5.row_dimensions[1].height = 32

headers5 = ["Rank", "Player", "Tier", "Shots", "Goals", "xG Sum",
            "Raw Conv%", "xG/Shot", "EB Mean", "CI Low", "CI High",
            "EB vs xG", "Shrinkage", "Reliability", "Goals vs xG"]
col_w5 = [6, 22, 14, 7, 7, 10, 10, 9, 9, 9, 9, 10, 10, 10, 11]
for ci, (h, w) in enumerate(zip(headers5, col_w5), 1):
    ws5.column_dimensions[get_column_letter(ci)].width = w
    c = ws5.cell(row=2, column=ci, value=h)
    c.font = font(bold=True, color=WHITE, size=9)
    c.fill = px(TEAL); c.alignment = center(); c.border = border_thin()
ws5.row_dimensions[2].height = 20

df_ranked = df.sort_values("eb_mean", ascending=False).reset_index(drop=True)
for ri, (_, row) in enumerate(df_ranked.iterrows(), 3):
    tc = TIER_COLOR.get(row["tier"], "AAAAAA")
    row_bg = LIGHT_BG if ri % 2 == 1 else WHITE
    vals = [
        ri - 2,
        row["player_name"],
        row["tier"],
        int(row["n"]),
        int(row["goals"]),
        f"{row['xg_sum']:.2f}",
        pct(row["raw_rate"]),
        pct(row["xg_per_shot"]),
        pct(row["eb_mean"]),
        pct(row["eb_ci_lo"]),
        pct(row["eb_ci_hi"]),
        f"{row['eb_vs_xg']*100:+.2f}%",
        pct(row["shrinkage"]),
        pct(row["reliability"]),
        f"{row['goals_above_xg']:+.2f}",
    ]
    for ci, val in enumerate(vals, 1):
        c = ws5.cell(row=ri, column=ci, value=val)
        c.border = border_thin(); c.font = font(size=9); c.alignment = center()
        if ci == 3:
            c.font = Font(bold=True, size=8, color=WHITE, name="Calibri")
            c.fill = px(tc)
        else:
            c.fill = px(row_bg)
    ws5.row_dimensions[ri].height = 15

# Conditional formats
max_r5 = 2 + len(df_ranked)
ws5.conditional_formatting.add(
    f"I3:I{max_r5}",
    ColorScaleRule(start_type="min", start_color="FFC7CE",
                   mid_type="num",   mid_value=league_mean, mid_color="FFEB9C",
                   end_type="max",   end_color="C6EFCE")
)
ws5.conditional_formatting.add(
    f"L3:L{max_r5}",
    ColorScaleRule(start_type="min", start_color="FFC7CE",
                   mid_type="num",   mid_value=0, mid_color="FFFFFF",
                   end_type="max",   end_color="C6EFCE")
)
ws5.freeze_panes = "A3"

# ═════════════════════════════════════════════════════════════════════════════
# Sheet 6: 📉 Underperformers
# ═════════════════════════════════════════════════════════════════════════════
ws6 = wb.create_sheet("📉 Underperformers")
ws6.sheet_view.showGridLines = False

ws6.merge_cells("A1:L1")
c = ws6.cell(row=1, column=1,
             value="📉  UNDERPERFORMERS  —  EB vs xG < −2%  (Finishing below expectation)")
c.font = Font(bold=True, size=14, color=WHITE, name="Calibri")
c.fill = px("8B0000"); c.alignment = center(); ws6.row_dimensions[1].height = 32

ws6.merge_cells("A2:L2")
c = ws6.cell(row=2, column=1,
             value="Players whose EB finishing rate is materially below their xG rate.  "
                   "Could indicate poor technique, bad luck, or positional finishing role.  Min 10 shots.")
c.font = Font(size=9, color=WHITE, italic=True, name="Calibri")
c.fill = px(RED_HI); c.alignment = center(wrap=True); ws6.row_dimensions[2].height = 20

under = df_q10[df_q10["eb_vs_xg"] < -0.02].sort_values("eb_vs_xg")
headers6 = ["#", "Player", "Tier", "Shots", "Goals", "xG Sum",
            "Raw Conv%", "xG/Shot", "EB Mean", "EB vs xG", "Shrinkage", "Analysis"]
col_w6 = [5, 22, 14, 7, 7, 10, 10, 10, 10, 11, 10, 40]
for ci, (h, w) in enumerate(zip(headers6, col_w6), 1):
    ws6.column_dimensions[get_column_letter(ci)].width = w
    c = ws6.cell(row=3, column=ci, value=h)
    c.font = font(bold=True, color=WHITE, size=9)
    c.fill = px("8B0000"); c.alignment = center(); c.border = border_thin()

def under_analysis(row):
    gap = abs(row["eb_vs_xg"])
    if gap > 0.05:
        base = "Significant underperformance"
    elif gap > 0.03:
        base = "Moderate underperformance"
    else:
        base = "Slight underperformance"
    if row["n"] < 30:
        base += " — small sample, could regress positively"
    elif row["reliability"] > 0.6:
        base += " — high reliability, pattern likely real"
    return base

for ri, (_, row) in enumerate(under.iterrows(), 4):
    tc = TIER_COLOR.get(row["tier"], "AAAAAA")
    row_bg = "FFF0F0" if ri % 2 == 0 else WHITE
    vals = [
        ri - 3, row["player_name"], row["tier"],
        int(row["n"]), int(row["goals"]),
        f"{row['xg_sum']:.2f}", pct(row["raw_rate"]),
        pct(row["xg_per_shot"]), pct(row["eb_mean"]),
        f"{row['eb_vs_xg']*100:+.2f}%",
        pct(row["shrinkage"]), under_analysis(row),
    ]
    for ci, val in enumerate(vals, 1):
        c = ws6.cell(row=ri, column=ci, value=val)
        c.border = border_thin(); c.font = font(size=9); c.alignment = center()
        if ci == 3:
            c.font = Font(bold=True, size=9, color=WHITE, name="Calibri"); c.fill = px(tc)
        elif ci == 10:
            c.fill = px("FFC7CE")
            c.font = Font(bold=True, size=9, color="9C0006", name="Calibri")
        elif ci == 12:
            c.alignment = left(wrap=True); c.fill = px(row_bg)
        else:
            c.fill = px(row_bg)
    ws6.row_dimensions[ri].height = 16

ws6.freeze_panes = "A4"

# ═════════════════════════════════════════════════════════════════════════════
# Sheet 7: 📋 Methodology
# ═════════════════════════════════════════════════════════════════════════════
ws7 = wb.create_sheet("📋 Methodology")
ws7.sheet_view.showGridLines = False
ws7.column_dimensions["A"].width = 3
ws7.column_dimensions["B"].width = 26
ws7.column_dimensions["C"].width = 70

ws7.merge_cells("B1:C1")
c = ws7.cell(row=1, column=2, value="📋  METHODOLOGY & COLUMN GLOSSARY")
c.font = Font(bold=True, size=14, color=WHITE, name="Calibri")
c.fill = px(NAVY); c.alignment = center(); ws7.row_dimensions[1].height = 30

method_rows = [
    ("MODEL", "Empirical Bayes Beta-Binomial", ""),
    ("Prior", "Beta(α, β) fitted via marginal log-likelihood maximisation across all players",
     "α and β are the league-wide shape parameters"),
    ("Posterior", "Beta(α + goals_i, β + n_i − goals_i)",
     "Per-player posterior after observing their shots and goals"),
    ("Shrinkage", "eb_mean = (α + goals_i) / (α + β + n_i)",
     "Pulls sparse players toward league mean; high-volume players mostly unaffected"),
    ("90% CI", "scipy.stats.beta.ppf(0.05 / 0.95, α_post, β_post)",
     "Lower and upper bound of 90% credible interval"),
    ("", "", ""),
    ("COLUMNS", "", ""),
    ("player_name", "Player name", ""),
    ("n", "Total shots taken (non-penalty)", "Used as sample size"),
    ("goals", "Total goals scored", ""),
    ("xg_sum", "Sum of pre-shot xG values", "From XGBoost xG model"),
    ("raw_rate", "goals / n", "Naïve conversion rate — noisy for small n"),
    ("xg_rate", "xg_sum / n", "Average xG per shot"),
    ("eb_alpha_post", "α + goals", "Posterior Beta shape parameter"),
    ("eb_beta_post", "β + (n − goals)", "Posterior Beta shape parameter"),
    ("eb_mean", "Bayesian conversion estimate", "Primary finishing quality metric"),
    ("eb_ci_lo", "5th percentile of posterior", "Lower bound of 90% credible interval"),
    ("eb_ci_hi", "95th percentile of posterior", "Upper bound of 90% credible interval"),
    ("eb_vs_xg", "eb_mean − xg_rate", "Finishing skill above/below xG expectation"),
    ("shrinkage", "How much pulled toward prior", "1.0 = all prior, 0.0 = all data"),
    ("reliability", "1 − shrinkage", "Higher = more data-driven"),
    ("ci_width", "eb_ci_hi − eb_ci_lo", "Uncertainty band width"),
    ("tier", "Scouting tier", "Elite / Strong / Average / Below Avg / Weak / Insufficient data"),
    ("", "", ""),
    ("TIERS", "", ""),
    ("Elite", "EB mean ≥ 15.5% AND reliability ≥ 55%", "Primary scouting targets"),
    ("Strong", "EB mean ≥ 13.0% AND reliability ≥ 40%", "Reliable above-average finishers"),
    ("Average", "EB mean ≥ 11.0%", "League standard"),
    ("Below Avg", "EB mean ≥ 8.5%", "Underperforming — context needed"),
    ("Weak", "EB mean < 8.5%", "Poor finishing — not a primary target"),
    ("Insufficient data", "< 10 shots", "Cannot assess reliably"),
]

for ri, (field, value, note) in enumerate(method_rows, 2):
    bg = MID_BG if field and field == field.upper() else (LIGHT_BG if ri % 2 == 0 else WHITE)
    for ci, (val, col) in enumerate([(field, "B"), (value if not note else f"{value}", "C")], 2):
        c = ws7.cell(row=ri, column=ci)
        c.fill = px(bg if field != field.upper() or not field else TEAL if ci == 2 else MID_BG)
        if field and field == field.upper() and field not in ("", "COLUMNS", "TIERS", "MODEL"):
            c.value = val if ci == 2 else ""
        elif field and field == field.upper():
            c.value = val if ci == 2 else ""
            c.font = Font(bold=True, size=10, color=WHITE if ci == 2 else DARK_TEXT, name="Calibri")
            c.fill = px(NAVY if ci == 2 else MID_BG)
        else:
            c.value = val
            c.font = font(bold=(ci==2 and field != ""), size=9)
            c.fill = px(bg)
        c.alignment = left(wrap=True)
        c.border = border_thin()
    if note:
        c3 = ws7.cell(row=ri, column=4, value=f"  ℹ {note}")
        c3.font = Font(size=8, italic=True, color="555555", name="Calibri")
        c3.fill = px(bg); c3.alignment = left()
        ws7.column_dimensions["D"].width = 45
    ws7.row_dimensions[ri].height = 18

# ═════════════════════════════════════════════════════════════════════════════
# Sheet 8: 📊 Bar Chart — Top 30 EB Mean
# ═════════════════════════════════════════════════════════════════════════════
ws8 = wb.create_sheet("📊 Charts")
ws8.sheet_view.showGridLines = False

ws8.merge_cells("A1:Z1")
c = ws8.cell(row=1, column=1, value="📊  CHARTS  —  Top 30 Finishers by EB Mean  |  Finishing Edge Leaders")
c.font = Font(bold=True, size=13, color=WHITE, name="Calibri")
c.fill = px(NAVY); c.alignment = center(); ws8.row_dimensions[1].height = 28

# Data table for chart (top 30 by eb_mean, min 10 shots)
top30 = df_q10.nlargest(30, "eb_mean").sort_values("eb_mean")
data_row = 3
ws8.cell(row=data_row, column=1, value="Player").font = font(bold=True, size=9)
ws8.cell(row=data_row, column=2, value="EB Mean (%)").font = font(bold=True, size=9)
ws8.cell(row=data_row, column=3, value="CI Low (%)").font = font(bold=True, size=9)
ws8.cell(row=data_row, column=4, value="CI High (%)").font = font(bold=True, size=9)
ws8.cell(row=data_row, column=5, value="Edge (%)").font = font(bold=True, size=9)

for ci in range(1, 6):
    ws8.cell(row=data_row, column=ci).fill = px(TEAL)
    ws8.cell(row=data_row, column=ci).font = Font(bold=True, size=9, color=WHITE, name="Calibri")
    ws8.cell(row=data_row, column=ci).alignment = center()

for ri, (_, row) in enumerate(top30.iterrows(), data_row + 1):
    ws8.cell(row=ri, column=1, value=row["player_name"]).font = font(size=9)
    ws8.cell(row=ri, column=2, value=round(row["eb_mean"] * 100, 2))
    ws8.cell(row=ri, column=3, value=round(row["eb_ci_lo"] * 100, 2))
    ws8.cell(row=ri, column=4, value=round(row["eb_ci_hi"] * 100, 2))
    ws8.cell(row=ri, column=5, value=round(row["eb_vs_xg"] * 100, 2))
    ws8.row_dimensions[ri].height = 14
ws8.column_dimensions["A"].width = 22
for col in ["B","C","D","E"]:
    ws8.column_dimensions[col].width = 12

chart_end = data_row + len(top30)

# Bar chart: EB Mean
bar = BarChart()
bar.type = "bar"; bar.grouping = "clustered"
bar.title = "Top 30 Finishers — EB Posterior Mean Conversion %"
bar.y_axis.title = "Player"
bar.x_axis.title = "EB Mean (%)"
bar.width = 22; bar.height = 18
bar.style = 10

data_ref = Reference(ws8, min_col=2, max_col=2,
                     min_row=data_row, max_row=chart_end)
cats_ref = Reference(ws8, min_col=1,
                     min_row=data_row+1, max_row=chart_end)
bar.add_data(data_ref, titles_from_data=True)
bar.set_categories(cats_ref)
bar.series[0].graphicalProperties.solidFill = "1B7A78"
ws8.add_chart(bar, "G3")

# Edge chart (top 25 positive edge)
top_edge = df_q10[df_q10["eb_vs_xg"] > 0].nlargest(25, "eb_vs_xg").sort_values("eb_vs_xg")
edge_start = chart_end + 2
ws8.cell(row=edge_start, column=1, value="Player").font = Font(bold=True, size=9, color=WHITE, name="Calibri")
ws8.cell(row=edge_start, column=2, value="EB vs xG (pp)").font = Font(bold=True, size=9, color=WHITE, name="Calibri")
for ci in [1, 2]:
    ws8.cell(row=edge_start, column=ci).fill = px(NAVY)
    ws8.cell(row=edge_start, column=ci).alignment = center()

for ri, (_, row) in enumerate(top_edge.iterrows(), edge_start + 1):
    ws8.cell(row=ri, column=1, value=row["player_name"])
    ws8.cell(row=ri, column=2, value=round(row["eb_vs_xg"] * 100, 2))
    ws8.row_dimensions[ri].height = 14

edge_end = edge_start + len(top_edge)
bar2 = BarChart()
bar2.type = "bar"; bar2.grouping = "clustered"
bar2.title = "Top 25 Finishing Edge — EB Mean above xG Rate"
bar2.x_axis.title = "Finishing Edge (percentage points above xG)"
bar2.width = 22; bar2.height = 15
bar2.style = 10
data_ref2 = Reference(ws8, min_col=2, max_col=2, min_row=edge_start, max_row=edge_end)
cats_ref2 = Reference(ws8, min_col=1, min_row=edge_start+1, max_row=edge_end)
bar2.add_data(data_ref2, titles_from_data=True)
bar2.set_categories(cats_ref2)
bar2.series[0].graphicalProperties.solidFill = "E8C33A"
anchor_row = edge_start
ws8.add_chart(bar2, f"G{anchor_row}")

# ═════════════════════════════════════════════════════════════════════════════
# Sheet 9: 🔢 Percentile Matrix
# ═════════════════════════════════════════════════════════════════════════════
ws9 = wb.create_sheet("🔢 Percentile Matrix")
ws9.sheet_view.showGridLines = False

ws9.merge_cells("A1:I1")
c = ws9.cell(row=1, column=1, value="🔢  PERCENTILE MATRIX  —  Rank among all 635 players (100 = best)")
c.font = Font(bold=True, size=14, color=WHITE, name="Calibri")
c.fill = px(NAVY); c.alignment = center(); ws9.row_dimensions[1].height = 30

pct_headers = ["Rank", "Player", "Tier", "EB Mean Pct",
               "Edge Pct", "Reliability Pct", "Raw Conv Pct", "Volume Pct", "n"]
pct_widths  = [6, 22, 14, 13, 10, 15, 13, 11, 7]
for ci, (h, w) in enumerate(zip(pct_headers, pct_widths), 1):
    ws9.column_dimensions[get_column_letter(ci)].width = w
    c = ws9.cell(row=2, column=ci, value=h)
    c.font = font(bold=True, color=WHITE, size=9)
    c.fill = px(TEAL); c.alignment = center(); c.border = border_thin()
ws9.row_dimensions[2].height = 20

pct_sorted = df_q10.sort_values("pct_eb_mean", ascending=True).reset_index(drop=True)

def pct_fill(val):
    if val >= 80: return "C6EFCE", "006100"
    if val >= 60: return "E2EFDA", "375623"
    if val >= 40: return "FFEB9C", "9C6500"
    if val >= 20: return "FCE4D6", "843C0C"
    return "FFC7CE", "9C0006"

for ri, (_, row) in enumerate(pct_sorted.iterrows(), 3):
    tc = TIER_COLOR.get(row["tier"], "AAAAAA")
    vals_raw = [
        ri - 2, row["player_name"], row["tier"],
        row["pct_eb_mean"], row["pct_eb_vs_xg"],
        row["pct_reliability"], row["pct_raw_rate"],
        row["pct_n"], int(row["n"]),
    ]
    for ci, val in enumerate(vals_raw, 1):
        c = ws9.cell(row=ri, column=ci)
        if ci in (4, 5, 6, 7, 8):
            bg, fg = pct_fill(val)
            c.value = f"{val:.0f}"
            c.fill = px(bg); c.font = Font(bold=True, size=9, color=fg, name="Calibri")
        elif ci == 3:
            c.value = val
            c.font = Font(bold=True, size=9, color=WHITE, name="Calibri"); c.fill = px(tc)
        else:
            c.value = val
            row_bg = LIGHT_BG if ri % 2 == 0 else WHITE
            c.fill = px(row_bg); c.font = font(size=9)
        c.alignment = center(); c.border = border_thin()
    ws9.row_dimensions[ri].height = 15

ws9.freeze_panes = "A3"

# ═════════════════════════════════════════════════════════════════════════════
# Output
# ═════════════════════════════════════════════════════════════════════════════
OUT = "eredivisie_scouting.xlsx"
wb.save(OUT)
print(f"✅  Saved {OUT}")
print(f"   Sheets: {[s.title for s in wb.worksheets]}")
print(f"   Players: {len(df)}  |  Qualified (≥10): {len(df_q10)}  |  "
      f"Elite: {(df_q10['tier']=='Elite').sum()}")
