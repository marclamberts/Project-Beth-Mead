"""
Expected Cross Model (xC) — Two-Stage Logistic Regression
==========================================================
Stage 1  P(completed)         Origin features → will the cross reach a teammate?
Stage 2  P(dangerous|completed) Destination features → will it generate a shot?

xC = Stage1 × Stage2

Run:
    python expected_cross_model.py
Outputs:
    xc_model_results.csv     — original data enriched with xC predictions
    xc_model_plots.png       — 6-panel diagnostic & pitch visualisation
"""

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, roc_curve,
    average_precision_score
)
from sklearn.pipeline import Pipeline

from mplsoccer import Pitch

# ---------------------------------------------------------------------------
# Constants & colour palette
# ---------------------------------------------------------------------------
PITCH_LEN_M = 105.0
PITCH_WID_M = 68.0

NAVY  = "#0D1B2A"
TEAL  = "#1B7A78"
GOLD  = "#E8C33A"
GREEN = "#1DB954"
RED   = "#E05252"
WHITE = "#FFFFFF"

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": NAVY,
    "axes.facecolor":   "#111E2B",
    "axes.edgecolor":   "#2A3E52",
    "text.color":       "#EEEEEE",
    "axes.labelcolor":  "#CCCCCC",
    "xtick.color":      "#AAAAAA",
    "ytick.color":      "#AAAAAA",
    "grid.color":       "#1E2E3E",
    "grid.alpha":        0.6,
    "font.family":      "DejaVu Sans",
})

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _dist_to_goal_m(x_opta, y_opta):
    """Metres from an Opta coordinate to the centre of the attacking goal."""
    dx = (1.0 - x_opta / 100) * PITCH_LEN_M
    dy = (0.5 - y_opta / 100) * PITCH_WID_M
    return math.hypot(dx, dy)


def build_stage1_features(df):
    """
    Features available at the moment of the cross (origin only).
    No knowledge of where the ball lands.
    """
    f = pd.DataFrame(index=df.index)
    f["x_norm"]          = df["x"] / 100.0
    f["y_norm"]          = df["y"] / 100.0
    f["distance_m"]      = df["distance_m"]
    f["flight_metric"]   = df["flight_metric"].fillna(1.0)
    f["curve_score"]     = df["curve_score"]
    f["goal_diff"]       = df["goal_diff"].clip(-3, 3)
    f["minute"]          = df["minute"] / 90.0
    f["dist_to_goal_m"]  = df.apply(
        lambda r: _dist_to_goal_m(r["x"], r["y"]), axis=1)
    f["is_deep"]         = (df["x"] >= 85).astype(float)
    f["from_right"]      = (df["y"] < 50).astype(float)
    f["centrality_orig"] = 1.0 - 2.0 * (df["y"] / 100.0 - 0.5).abs()
    return f


def build_stage2_features(df):
    """
    Features describing where the cross landed (only used on completed crosses).
    """
    f = pd.DataFrame(index=df.index)
    f["end_x_norm"]      = df["end_x"] / 100.0
    f["end_y_norm"]      = df["end_y"] / 100.0
    f["centrality_dest"] = 1.0 - 2.0 * (df["end_y"] / 100.0 - 0.5).abs()
    f["dist_to_goal_dest"] = df.apply(
        lambda r: _dist_to_goal_m(r["end_x"], r["end_y"]), axis=1)
    # Near/far post distances (posts at y=31.3 and y=68.7 in Opta ≈ 21.1m from centre)
    near_post_y = 31.3
    far_post_y  = 68.7
    f["near_post_dist"] = df.apply(
        lambda r: math.hypot((1.0 - r["end_x"]/100) * PITCH_LEN_M,
                             (near_post_y/100 - r["end_y"]/100) * PITCH_WID_M), axis=1)
    f["far_post_dist"]  = df.apply(
        lambda r: math.hypot((1.0 - r["end_x"]/100) * PITCH_LEN_M,
                             (far_post_y/100  - r["end_y"]/100) * PITCH_WID_M), axis=1)
    f["into_box"]       = (
        (df["end_x"] >= 83) & (df["end_y"] >= 21) & (df["end_y"] <= 79)
    ).astype(float)
    f["distance_m"]     = df["distance_m"]
    f["angle_deg"]      = df["angle_deg"]
    f["flight_metric"]  = df["flight_metric"].fillna(1.0)
    f["dest_central"]   = (df["dest_zone"] == "Center").astype(float)
    return f


# ---------------------------------------------------------------------------
# Model building helpers
# ---------------------------------------------------------------------------

def make_pipeline(C=1.0, class_weight=None):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=C, class_weight=class_weight,
            max_iter=1000, solver="lbfgs", random_state=42
        )),
    ])


def cv_metrics(pipeline, X, y, cv=5):
    """Return mean AUC and Brier score from stratified k-fold CV."""
    skf  = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    aucs = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc")
    brier = cross_val_score(pipeline, X, y, cv=skf,
                            scoring="neg_brier_score")
    return aucs.mean(), aucs.std(), (-brier).mean()


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(df):
    # ── Stage 1 ──────────────────────────────────────────────────────────────
    X1  = build_stage1_features(df).values
    y1  = df["outcome"].values                  # 1 = completed

    pipe1 = make_pipeline(C=0.5, class_weight="balanced")
    pipe1.fit(X1, y1)
    cv_auc1, cv_std1, brier1 = cv_metrics(make_pipeline(C=0.5, class_weight="balanced"),
                                           X1, y1)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    comp   = df[df["outcome"] == 1].copy()
    X2     = build_stage2_features(comp).values
    y2     = comp["result"].isin(["goal", "shot_ot", "shot"]).astype(int).values

    pipe2 = make_pipeline(C=1.0)
    pipe2.fit(X2, y2)
    cv_auc2, cv_std2, brier2 = cv_metrics(make_pipeline(C=1.0), X2, y2)

    print("=" * 52)
    print("  EXPECTED CROSS MODEL — TRAINING RESULTS")
    print("=" * 52)
    print(f"  Stage 1 — P(completed)              "
          f"n={len(df):>4} | pos={y1.sum():>3}")
    print(f"    CV AUC  : {cv_auc1:.3f} ± {cv_std1:.3f}")
    print(f"    Brier   : {brier1:.4f}")
    print()
    print(f"  Stage 2 — P(dangerous | completed)  "
          f"n={len(comp):>4} | pos={y2.sum():>3}")
    print(f"    CV AUC  : {cv_auc2:.3f} ± {cv_std2:.3f}")
    print(f"    Brier   : {brier2:.4f}")
    print("=" * 52)

    return pipe1, pipe2, cv_auc1, cv_auc2


# ---------------------------------------------------------------------------
# Predict xC for every cross
# ---------------------------------------------------------------------------

def predict_xc(df, pipe1, pipe2):
    df = df.copy()

    X1 = build_stage1_features(df).values
    df["xc_stage1"] = pipe1.predict_proba(X1)[:, 1]

    # Stage 2: score everything (using end_x/end_y where available, else median)
    df2 = df.copy()
    df2["end_x"] = df2["end_x"].fillna(df2["end_x"].median())
    df2["end_y"] = df2["end_y"].fillna(df2["end_y"].median())
    df2["angle_deg"] = df2["angle_deg"].fillna(df2["angle_deg"].median())

    X2 = build_stage2_features(df2).values
    df["xc_stage2"] = pipe2.predict_proba(X2)[:, 1]

    # Combined xC
    df["xc"] = df["xc_stage1"] * df["xc_stage2"]
    return df


# ---------------------------------------------------------------------------
# Calibration & ROC helpers
# ---------------------------------------------------------------------------

def _ax_style(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#2A3E52")
    ax.grid(alpha=0.25, color="#2A3E52")


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_all(df, pipe1, pipe2):
    """Six-panel diagnostic + pitch figure."""
    DANGEROUS = ["goal", "shot_ot", "shot"]

    X1  = build_stage1_features(df).values
    y1  = df["outcome"].values

    comp   = df[df["outcome"] == 1].copy()
    X2     = build_stage2_features(comp).values
    y2     = comp["result"].isin(DANGEROUS).astype(int).values

    p1_prob = pipe1.predict_proba(X1)[:, 1]
    p2_prob = pipe2.predict_proba(X2)[:, 1]

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(NAVY)
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── Panel 1: Calibration curves ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#111E2B")
    for prob, y, label, col in [
        (p1_prob, y1, "Stage 1  P(completed)", TEAL),
        (p2_prob, y2, "Stage 2  P(dangerous)", GREEN),
    ]:
        frac, mean_pred = calibration_curve(y, prob, n_bins=8)
        ax1.plot(mean_pred, frac, "o-", color=col, linewidth=1.8,
                 markersize=4, label=label, zorder=3)
    ax1.plot([0, 1], [0, 1], "--", color="#AAAAAA", linewidth=1.0,
             label="Perfect", alpha=0.6)
    ax1.set_xlabel("Mean predicted probability", fontsize=8)
    ax1.set_ylabel("Fraction of positives", fontsize=8)
    ax1.set_title("Calibration Curves", color=GOLD, fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7, facecolor="#1A2B3C", edgecolor="#2A3E52")
    _ax_style(ax1)

    # ── Panel 2: ROC curves ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#111E2B")
    for prob, y, label, col in [
        (p1_prob, y1, "Stage 1", TEAL),
        (p2_prob, y2, "Stage 2", GREEN),
    ]:
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        ax2.plot(fpr, tpr, color=col, linewidth=1.8,
                 label=f"{label}  AUC={auc:.3f}", zorder=3)
    ax2.plot([0, 1], [0, 1], "--", color="#AAAAAA", linewidth=1.0, alpha=0.5)
    ax2.set_xlabel("False Positive Rate", fontsize=8)
    ax2.set_ylabel("True Positive Rate", fontsize=8)
    ax2.set_title("ROC Curves", color=GOLD, fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7, facecolor="#1A2B3C", edgecolor="#2A3E52")
    _ax_style(ax2)

    # ── Panel 3: Feature coefficients (Stage 2, more meaningful) ─────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#111E2B")
    feat_names2 = [
        "end_x", "end_y", "centrality", "dist_to_goal",
        "near_post", "far_post", "into_box",
        "distance", "angle", "flight", "dest_central",
    ]
    coefs = pipe2.named_steps["clf"].coef_[0]
    order = np.argsort(np.abs(coefs))
    ax3.barh(range(len(coefs)), coefs[order],
             color=[GREEN if c > 0 else RED for c in coefs[order]],
             alpha=0.85, edgecolor="#2A3E52", linewidth=0.5)
    ax3.set_yticks(range(len(coefs)))
    ax3.set_yticklabels([feat_names2[i] for i in order], fontsize=7)
    ax3.axvline(0, color="#AAAAAA", linewidth=0.8)
    ax3.set_title("Stage 2 Coefficients\n(dangerous cross predictors)",
                  color=GOLD, fontsize=9, fontweight="bold")
    _ax_style(ax3)

    # ── Panel 4: xC distribution by result ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor("#111E2B")
    result_order = ["goal", "shot_ot", "shot", "completed", "cleared", "incomplete"]
    result_cols  = [GREEN, "#5BA85A", TEAL, GOLD, "#E07A3A", RED]
    for res, col in zip(result_order, result_cols):
        sub = df[df["result"] == res]["xc"]
        if len(sub) > 1:
            ax4.hist(sub, bins=15, alpha=0.55, color=col, label=res, density=True)
    ax4.set_xlabel("xC value", fontsize=8)
    ax4.set_ylabel("Density", fontsize=8)
    ax4.set_title("xC Distribution by Result", color=GOLD, fontsize=10,
                  fontweight="bold")
    ax4.legend(fontsize=6, facecolor="#1A2B3C", edgecolor="#2A3E52")
    _ax_style(ax4)

    # ── Panel 5: xC mean by result (bar) ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor("#111E2B")
    means  = [df[df["result"] == r]["xc"].mean() for r in result_order]
    counts = [len(df[df["result"] == r]) for r in result_order]
    bars   = ax5.bar(result_order, means, color=result_cols,
                     edgecolor="#2A3E52", linewidth=0.6, width=0.65, alpha=0.85)
    for bar, cnt in zip(bars, counts):
        ax5.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.001, f"n={cnt}",
                 ha="center", va="bottom", fontsize=7, color="#AAAAAA")
    ax5.set_ylabel("Mean xC", fontsize=8)
    ax5.set_title("Mean xC by Actual Result", color=GOLD, fontsize=10,
                  fontweight="bold")
    ax5.tick_params(axis="x", labelsize=7, rotation=20)
    _ax_style(ax5)

    # ── Panel 6: Season trend — actual danger rate vs mean xC ────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#111E2B")
    season_stats = (
        df.groupby("season")
        .agg(
            actual_rate  = ("result", lambda x: x.isin(DANGEROUS).mean()),
            mean_xc      = ("xc", "mean"),
            n            = ("xc", "count"),
        )
        .reset_index()
        .sort_values("season")
    )
    x_idx = range(len(season_stats))
    ax6.plot(x_idx, season_stats["actual_rate"] * 100, "o-",
             color=GREEN, linewidth=1.8, markersize=5, label="Actual danger%")
    ax6.plot(x_idx, season_stats["mean_xc"] * 100, "s--",
             color=GOLD, linewidth=1.5, markersize=4, label="Mean xC ×100")
    ax6.set_xticks(list(x_idx))
    ax6.set_xticklabels(
        [s.replace("WSL ", "") for s in season_stats["season"]],
        rotation=40, fontsize=6
    )
    ax6.set_ylabel("%", fontsize=8)
    ax6.set_title("Actual Danger Rate vs Mean xC\nby Season",
                  color=GOLD, fontsize=9, fontweight="bold")
    ax6.legend(fontsize=7, facecolor="#1A2B3C", edgecolor="#2A3E52")
    _ax_style(ax6)

    # ── Panel 7: Origin xC heatmap on pitch (Stage 1 probability surface) ────
    ax7 = fig.add_subplot(gs[2, 0:2])
    pitch = Pitch(pitch_type="opta", pitch_color=NAVY,
                  line_color="#2A3E52", line_zorder=2)
    pitch.draw(ax=ax7)

    # Build a fine grid and predict Stage 1 over it
    gx = np.linspace(40, 100, 60)
    gy = np.linspace(0,  100, 40)
    GX, GY = np.meshgrid(gx, gy)
    grid_flat = np.c_[GX.ravel(), GY.ravel()]

    med = df[["distance_m", "flight_metric", "curve_score",
               "goal_diff", "minute"]].median()

    grid_df = pd.DataFrame({
        "x": grid_flat[:, 0],
        "y": grid_flat[:, 1],
        "distance_m":    med["distance_m"],
        "flight_metric": med["flight_metric"],
        "curve_score":   med["curve_score"],
        "goal_diff":     0.0,
        "minute":        45.0,
    })
    grid_xc1 = pipe1.predict_proba(
        build_stage1_features(grid_df).values
    )[:, 1].reshape(GX.shape)

    cmap_xc = LinearSegmentedColormap.from_list(
        "xc", [NAVY, "#1B7A78", GOLD, RED])
    hm = ax7.pcolormesh(GX, GY, grid_xc1,
                        cmap=cmap_xc, alpha=0.75, shading="gouraud",
                        vmin=0, vmax=grid_xc1.max())
    plt.colorbar(hm, ax=ax7, shrink=0.7, pad=0.01,
                 label="P(cross completed) — Stage 1")

    # Overlay actual crosses coloured by xC
    pitch.scatter(df["x"], df["y"], ax=ax7,
                  c=df["xc"], cmap=cmap_xc, s=12, alpha=0.6, zorder=4)

    ax7.set_title("xC Stage 1 Surface — P(Completed) by Origin\n"
                  "(dots = actual crosses, coloured by full xC)",
                  color=GOLD, fontsize=10, fontweight="bold", pad=8)

    # ── Panel 8: Destination danger heatmap (Stage 2) ────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    pitch2 = Pitch(pitch_type="opta", pitch_color=NAVY,
                   line_color="#2A3E52", line_zorder=2)
    pitch2.draw(ax=ax8)

    gx2 = np.linspace(50, 100, 50)
    gy2 = np.linspace(0,  100, 50)
    GX2, GY2 = np.meshgrid(gx2, gy2)
    grid_flat2 = np.c_[GX2.ravel(), GY2.ravel()]

    med2 = comp[["distance_m", "angle_deg", "flight_metric"]].median()
    grid_df2 = pd.DataFrame({
        "end_x": grid_flat2[:, 0],
        "end_y": grid_flat2[:, 1],
        "distance_m":   med2["distance_m"],
        "angle_deg":    med2["angle_deg"],
        "flight_metric": med2["flight_metric"],
        "dest_zone":    "Center",
    })
    grid_xc2 = pipe2.predict_proba(
        build_stage2_features(grid_df2).values
    )[:, 1].reshape(GX2.shape)

    hm2 = ax8.pcolormesh(GX2, GY2, grid_xc2,
                          cmap=cmap_xc, alpha=0.75, shading="gouraud",
                          vmin=0, vmax=grid_xc2.max())
    plt.colorbar(hm2, ax=ax8, shrink=0.7, pad=0.01,
                 label="P(dangerous) — Stage 2")

    # Overlay completed cross destinations
    comp_plot = comp.dropna(subset=["end_x", "end_y"])
    pitch2.scatter(comp_plot["end_x"], comp_plot["end_y"], ax=ax8,
                   c=pipe2.predict_proba(
                       build_stage2_features(comp_plot).values
                   )[:, 1],
                   cmap=cmap_xc, s=14, alpha=0.7, zorder=4)

    ax8.set_title("Stage 2 Surface — P(Dangerous) by Destination\n"
                  "(dots = completed crosses)",
                  color=GOLD, fontsize=9, fontweight="bold", pad=8)

    fig.suptitle("Expected Cross Model (xC) — Beth Mead WSL 2015–2026",
                 fontsize=15, fontweight="bold", color=WHITE, y=0.99)

    out = "/home/user/Project-Beth-Mead/xc_model_plots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=NAVY)
    print(f"Plots saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Top / worst crosses by xC
# ---------------------------------------------------------------------------

def print_top_crosses(df, n=10):
    cols = ["date", "opponent", "minute", "x", "y", "end_x", "end_y",
            "distance_m", "angle_deg", "result",
            "xc_stage1", "xc_stage2", "xc"]
    print("\n  TOP crosses by xC (highest threat):")
    print(df.nlargest(n, "xc")[cols].to_string(index=False))
    print("\n  LOWEST xC (poor delivery / low-threat attempts):")
    print(df.nsmallest(n, "xc")[cols].to_string(index=False))


def season_summary(df):
    DANGEROUS = ["goal", "shot_ot", "shot"]
    g = df.groupby("season").agg(
        crosses      = ("xc", "count"),
        actual_rate  = ("result", lambda x: x.isin(DANGEROUS).mean()),
        mean_xc      = ("xc", "mean"),
        sum_xc       = ("xc", "sum"),
        mean_stage1  = ("xc_stage1", "mean"),
        mean_stage2  = ("xc_stage2", "mean"),
    ).round(4)
    print("\n  Season summary:")
    print(g.to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA = "/home/user/Project-Beth-Mead/cross_analysis_results.csv"
    print(f"Loading {DATA}...")
    df = pd.read_csv(DATA)

    pipe1, pipe2, auc1, auc2 = train(df)
    df = predict_xc(df, pipe1, pipe2)

    season_summary(df)
    print_top_crosses(df)

    plot_all(df, pipe1, pipe2)

    out = "/home/user/Project-Beth-Mead/xc_model_results.csv"
    df.to_csv(out, index=False)
    print(f"\nEnriched data saved → {out}")
    print(f"New columns: xc_stage1, xc_stage2, xc")
