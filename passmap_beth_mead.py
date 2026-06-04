import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from mplsoccer import Pitch

# ── collect all passes ─────────────────────────────────────────────────────────

def get_qualifier(event, qid):
    for q in event.get("qualifier", []):
        if q["qualifierId"] == qid:
            return q.get("value")
    return None

def has_qualifier(event, qid):
    return any(q["qualifierId"] == qid for q in event.get("qualifier", []))


def normalize(x, y, flip):
    if flip:
        return 100 - x, 100 - y
    return x, y


all_passes = []

files = sorted(glob.glob(
    "/Users/user/Documents/GitHub/Project-Beth-Mead/**/*.json",
    recursive=True
))

for fpath in files:
    fname = os.path.basename(fpath)

    with open(fpath) as fp:
        data = json.load(fp)

    if "event" not in data:
        continue
    events = data["event"]

    # Identify Arsenal's contestant ID from Beth Mead events
    arsenal_id = None
    for e in events:
        if e.get("playerName") == "B. Mead":
            arsenal_id = e["contestantId"]
            break
    if arsenal_id is None:
        continue

    # Determine if Arsenal is home (first team in filename)
    teams = fname.replace(".json", "").split("_", 1)[1] if "_" in fname else fname
    teams_parts = teams.split(" - ")
    home_team_name = teams_parts[0].strip() if len(teams_parts) >= 2 else ""
    arsenal_is_home = "Arsenal" in home_team_name

    for e in events:
        if e.get("playerName") != "B. Mead":
            continue
        if e.get("typeId") != 1:  # passes only
            continue

        x = e.get("x")
        y = e.get("y")
        end_x_str = get_qualifier(e, 140)
        end_y_str = get_qualifier(e, 141)

        if x is None or y is None or end_x_str is None or end_y_str is None:
            continue

        end_x = float(end_x_str)
        end_y = float(end_y_str)
        period = e.get("periodId", 1)
        outcome = e.get("outcome", 0)  # 1=success, 0=fail

        # Normalise so Arsenal always attacks left→right
        # Home: period1 no flip, period2 flip
        # Away: period1 flip, period2 no flip
        if arsenal_is_home:
            flip = period == 2
        else:
            flip = period == 1

        x, y = normalize(x, y, flip)
        end_x, end_y = normalize(end_x, end_y, flip)

        is_key_pass = has_qualifier(e, 210)   # shot assist
        # Progressive: successful pass that gains ≥10 yards toward goal
        is_progressive = (outcome == 1
                          and end_x - x >= 10
                          and end_x > 50)

        all_passes.append({
            "x": x, "y": y,
            "end_x": end_x, "end_y": end_y,
            "outcome": outcome,
            "key_pass": is_key_pass,
            "progressive": is_progressive,
            "season": fpath.split("/")[-2],
        })

print(f"Total passes collected: {len(all_passes)}")


# ── draw ───────────────────────────────────────────────────────────────────────

BG = "#1a1a1a"
LINE_COLOR = "white"

# Colour priority: key pass > progressive > successful > unsuccessful
# key pass / shot assist → light blue
# progressive → green
# successful → gold/orange
# unsuccessful → red

def pass_color(p):
    if p["key_pass"]:
        return "#5bc8f5"   # light blue
    if p["progressive"]:
        return "#4ecb71"   # green
    if p["outcome"] == 1:
        return "#f5a623"   # orange/gold
    return "#e63946"       # red

pitch = Pitch(
    pitch_type="opta",
    pitch_color=BG,
    line_color=LINE_COLOR,
    linewidth=1.2,
    goal_type="box",
)

fig, ax = pitch.draw(figsize=(16, 11))
fig.patch.set_facecolor(BG)

# Sort so unsuccessful drawn first (bottom layer), then successful, progressive, key
def sort_key(p):
    if p["key_pass"]:
        return 3
    if p["progressive"]:
        return 2
    if p["outcome"] == 1:
        return 1
    return 0

all_passes_sorted = sorted(all_passes, key=sort_key)

for p in all_passes_sorted:
    color = pass_color(p)
    alpha = 0.35 if p["outcome"] == 0 else 0.55
    if p["key_pass"] or p["progressive"]:
        alpha = 0.80

    pitch.arrows(
        p["x"], p["y"],
        p["end_x"], p["end_y"],
        ax=ax,
        color=color,
        alpha=alpha,
        width=1.2,
        headwidth=4,
        headlength=4,
    )

# ── legend ────────────────────────────────────────────────────────────────────

legend_items = [
    mpatches.Patch(color="#f5a623", label="Successful pass"),
    mpatches.Patch(color="#e63946", label="Unsuccessful pass"),
    mpatches.Patch(color="#4ecb71", label="Progressive pass"),
    mpatches.Patch(color="#5bc8f5", label="Shot assist / Key pass"),
]

leg = ax.legend(
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

# ── titles ────────────────────────────────────────────────────────────────────

seasons = sorted(set(p["season"] for p in all_passes))
season_range = f"{seasons[0].replace('WSL ', '')} – {seasons[-1].replace('WSL ', '')}"

total = len(all_passes)
succ = sum(1 for p in all_passes if p["outcome"] == 1)
key = sum(1 for p in all_passes if p["key_pass"])
prog = sum(1 for p in all_passes if p["progressive"])

fig.text(
    0.5, 0.97,
    "B. Mead — Pass Map  |  All WSL Seasons",
    ha="center", va="top",
    fontsize=18, fontweight="bold",
    color="white",
)
fig.text(
    0.5, 0.935,
    f"WSL {season_range}  ·  {total} passes  ·  {succ} successful  ·  {prog} progressive  ·  {key} shot assists",
    ha="center", va="top",
    fontsize=10,
    color="#cccccc",
)

plt.tight_layout(rect=[0, 0, 1, 0.935])
out = "/Users/user/Documents/GitHub/Project-Beth-Mead/passmap_beth_mead_all_seasons.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
