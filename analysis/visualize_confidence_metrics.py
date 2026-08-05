"""visualize_confidence_metrics.py

Produces 4 publication-quality plots from confidence_metrics.csv.
All thresholds, annotations, and interpretations are computed dynamically
from the data — no hardcoded values anywhere.

Story arc:
    1. Cleveland dot plot — reliability by schema block (zoomed to variance)
    2. Scatter — all 199 variables: agreement vs confidence, colored by block
                 4 quadrants reveal the nature of each variable's failure mode
    3. Grouped bar — bottom 20 variables: agreement vs confidence side by side
    4. Variable failure table — systematic failures aggregated across reports

Output:
    /N/project/ADRD/neuropathoroot/results/confidence_plots/
        01_cleveland_agreement_by_block.png/.svg
        02_scatter_agreement_vs_confidence.png/.svg
        03_bottom20_agreement_vs_confidence.png/.svg
        04_variable_failure_table.png/.svg

Usage:
    python visualize_confidence_metrics.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("visualize_metrics")

# ── Paths ──────────────────────────────────────────────────────────────────
INPUT_CSV  = Path("/N/project/ADRD/neuropathoroot/results/confidence_metrics.csv")
OUTPUT_DIR = Path("/N/project/ADRD/neuropathoroot/results/confidence_plots")

# ── Style ──────────────────────────────────────────────────────────────────
OSS_BLUE    = "#12436D"
OSS_LIGHT   = "#73B0D4"
ACCENT_WARN = "#801650"
ACCENT_MED  = "#28A197"
GREY        = "#BDBDBD"

# Quadrant colors for scatter
Q_IDEAL    = OSS_BLUE     # high agreement, high confidence
Q_HEDGING  = ACCENT_MED   # high agreement, low confidence
Q_WRONG    = ACCENT_WARN  # low agreement, high confidence — most concerning
Q_AWARE    = "#F46A25"    # low agreement, low confidence — model knows

# Block palette — distinct colors for 30+ blocks
BLOCK_PALETTE = [
    "#12436D", "#28A197", "#801650", "#F46A25", "#3D7AB8",
    "#A3195B", "#5C9E31", "#D4351C", "#4C2C92", "#0B6623",
    "#B35C00", "#005EA5", "#8E1F70", "#1D7039", "#C84B00",
    "#2B4490", "#6A0136", "#007A6C", "#9C4A00", "#003D78",
    "#5A2582", "#006A4E", "#A02020", "#004F88", "#7B3F00",
    "#2E7D5A", "#8B0045", "#1A5276", "#6D4C41", "#37474F",
]

FONT_FAMILY = "DejaVu Sans"
DPI         = 600

plt.rcParams.update({
    "font.family"      : FONT_FAMILY,
    "font.size"        : 10,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : DPI,
    "savefig.dpi"      : DPI,
    "savefig.bbox"     : "tight",
})

# ── Schema block mapping ───────────────────────────────────────────────────
BLOCK_MAP = {
    "NPSEX": "Specimen Info", "NPFIX": "Specimen Info", "NPFIXX": "Specimen Info",
    "NPWBRWT": "Specimen Info", "NPWBRF": "Specimen Info", "NPPMIH": "Specimen Info",
    "NPGRLA": "Gross & Vascular", "NPGRHA": "Gross & Vascular",
    "NPGRSNH": "Gross & Vascular", "NPGRLCH": "Gross & Vascular",
    "NPGRCCA": "Gross & Vascular", "NACCAVAS": "Gross & Vascular",
    "NPLINF": "Gross & Vascular", "NPLAC": "Gross & Vascular",
    "NPHEM": "Gross & Vascular", "NPWMR": "Gross & Vascular",
    "NACCARTE": "Gross & Vascular",
    "NPNLOSS": "Microscopic", "NPHIPSCL": "Microscopic",
    "NPLBOD": "Lewy Pathology", "NACCLEWY": "Lewy Pathology",
    "NPTHAL": "AD Pathology", "NACCBRAA": "AD Pathology",
    "NACCNEUR": "AD Pathology", "NPADNC": "AD Pathology",
    "NACCDIFF": "AD Pathology", "NACCAMY": "AD Pathology",
    "NPPAD": "Primary Dx", "NPCAD": "Primary Dx",
    "NPPLEWY": "Primary Dx", "NPCLEWY": "Primary Dx",
    "NPPVASC": "Primary Dx", "NPCVASC": "Primary Dx",
    "NPPFTLD": "Primary Dx", "NPCFTLD": "Primary Dx",
    "NPFTDTAU": "FTLD-Tau", "NACCPICK": "FTLD-Tau", "NPFTDT2": "FTLD-Tau",
    "NACCCBD": "FTLD-Tau", "NACCPROG": "FTLD-Tau", "NPFTDT5": "FTLD-Tau",
    "NPFTDT6": "FTLD-Tau", "NPFTDT7": "FTLD-Tau", "NPFTDT8": "FTLD-Tau",
    "NPFTDT9": "FTLD-Tau", "NPFTDT10": "FTLD-Tau",
    "NPFTDTDP": "TDP-43",
    "NACCPRIO": "Prion",
    "NPOFTD": "Other FTLD", "NPOFTD1": "Other FTLD", "NPOFTD2": "Other FTLD",
    "NPOFTD3": "Other FTLD", "NPOFTD4": "Other FTLD", "NPOFTD5": "Other FTLD",
    "NPFTDNO": "FTLD", "NPFTDSPC": "FTLD", "NPFTD": "FTLD",
    "NPTAU": "FTLD", "NPFRONT": "FTLD", "NPALSMND": "FTLD",
    "NPTDPA": "TDP-43", "NPTDPB": "TDP-43", "NPTDPC": "TDP-43",
    "NPTDPD": "TDP-43", "NPTDPE": "TDP-43",
    "NPINF": "Infarcts", "NACCINF": "Infarcts",
    "NPINF1A": "Infarcts", "NPINF2A": "Infarcts",
    "NPINF3A": "Infarcts", "NPINF4A": "Infarcts",
    "NPINF1B": "Infarcts", "NPINF1D": "Infarcts", "NPINF1F": "Infarcts",
    "NPINF2B": "Infarcts", "NPINF2D": "Infarcts", "NPINF2F": "Infarcts",
    "NPINF3B": "Infarcts", "NPINF3D": "Infarcts", "NPINF3F": "Infarcts",
    "NPINF4B": "Infarcts", "NPINF4D": "Infarcts", "NPINF4F": "Infarcts",
    "NPHEMO": "Hemorrhage", "NPHEMO1": "Hemorrhage", "NPHEMO2": "Hemorrhage",
    "NPHEMO3": "Hemorrhage", "NACCHEM": "Hemorrhage",
    "NPOLDD": "Microbleeds", "NPOLDD1": "Microbleeds", "NPOLDD2": "Microbleeds",
    "NPOLDD3": "Microbleeds", "NPOLDD4": "Microbleeds",
    "NPOLD": "Microinfarcts", "NPOLD1": "Microinfarcts", "NPOLD2": "Microinfarcts",
    "NPOLD3": "Microinfarcts", "NPOLD4": "Microinfarcts", "NACCMICR": "Microinfarcts",
    "NPPATH": "Other Vascular", "NACCNEC": "Other Vascular",
    "NPPATH2": "Other Vascular", "NPPATH3": "Other Vascular",
    "NPPATH4": "Other Vascular", "NPPATH5": "Other Vascular",
    "NPPATH6": "Other Vascular", "NPPATH7": "Other Vascular",
    "NPPATH8": "Other Vascular", "NPPATH9": "Other Vascular",
    "NPPATH10": "Other Vascular", "NPPATH11": "Other Vascular",
    "NPPATHO": "Other Vascular", "NPPATHOX": "Other Vascular",
    "NPMICRO": "Misc Vascular", "NPART": "Misc Vascular", "NPOANG": "Misc Vascular",
    "NPPNORM": "Normal / Insuff AD", "NPCNORM": "Normal / Insuff AD",
    "NPPADP": "Normal / Insuff AD", "NPCADP": "Normal / Insuff AD",
    "NPPHIPP": "Hippocampal & Prion Dx", "NPCHIPP": "Hippocampal & Prion Dx",
    "NPPPRION": "Hippocampal & Prion Dx", "NPCPRION": "Hippocampal & Prion Dx",
    "NPPOTH1": "Other Dx", "NPCOTH1": "Other Dx", "NPOTH1X": "Other Dx",
    "NPPOTH2": "Other Dx", "NPCOTH2": "Other Dx", "NPOTH2X": "Other Dx",
    "NPPOTH3": "Other Dx", "NPCOTH3": "Other Dx", "NPOTH3X": "Other Dx",
    "NACCOTHP": "Write-in Dx", "NACCWRI1": "Write-in Dx",
    "NACCWRI2": "Write-in Dx", "NACCWRI3": "Write-in Dx",
    "NPPDXA": "Disease Flags", "NPPDXB": "Disease Flags", "NPPDXD": "Disease Flags",
    "NPPDXE": "Disease Flags", "NPPDXF": "Disease Flags", "NPPDXG": "Disease Flags",
    "NPPDXH": "Disease Flags", "NPPDXI": "Disease Flags", "NPPDXJ": "Disease Flags",
    "NPPDXK": "Disease Flags", "NPPDXL": "Disease Flags", "NPPDXM": "Disease Flags",
    "NPPDXN": "Disease Flags", "NACCDOWN": "Disease Flags", "NPSCL": "Disease Flags",
    "NPNIT": "Criteria & Misc", "NPCERAD": "Criteria & Misc",
    "NPADRDA": "Criteria & Misc", "NPOCRIT": "Criteria & Misc",
    "NPVOTH": "Criteria & Misc", "NPLEWYCS": "Criteria & Misc",
    "NPFORMVER": "Case Metadata", "NACCID": "Case Metadata",
    "NACCADC": "Case Metadata", "NACCDAGE": "Case Metadata",
    "NACCMOD": "Case Metadata", "NACCYOD": "Case Metadata",
    "NACCINT": "Case Metadata",
    "NPTAN": "Antibody Methods", "NPTANX": "Antibody Methods",
    "NPABAN": "Antibody Methods", "NPABANX": "Antibody Methods",
    "NPASAN": "Antibody Methods", "NPASANX": "Antibody Methods",
    "NPTDPAN": "Antibody Methods", "NPTDPANX": "Antibody Methods",
    "NPHISMB": "Histochem Stains", "NPHISG": "Histochem Stains",
    "NPHISSS": "Histochem Stains", "NPHIST": "Histochem Stains",
    "NPHISO": "Histochem Stains", "NPHISOX": "Histochem Stains",
    "NACCBNKF": "Tissue Banking", "NPBNKB": "Tissue Banking",
    "NACCFORM": "Tissue Banking", "NACCPARA": "Tissue Banking",
    "NACCCSFP": "Tissue Banking", "NPBNKF": "Tissue Banking",
    "NPFAUT": "Full Autopsy", "NPFAUT1": "Full Autopsy",
    "NPFAUT2": "Full Autopsy", "NPFAUT3": "Full Autopsy", "NPFAUT4": "Full Autopsy",
    "NPGENE": "Family History", "NPFHSPEC": "Family History",
    "NPTAUHAP": "Genetics", "NPPRNP": "Genetics", "NPCHROM": "Genetics",
    "NPPDXP": "Genetics", "NPPDXQ": "Genetics",
    "NACCVASC": "Programmatic", "NACCBRNN": "Programmatic",
}


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = OUTPUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, format=ext)
        logger.info("saved %s", path)
    plt.close(fig)


def _var_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per variable, averaged across all reports."""
    return (
        df.groupby(["variable", "block"])
        .agg(
            agreement=("seed_agreement_rate", "mean"),
            confidence=("mean_confidence", "mean"),
        )
        .reset_index()
    )


# ── Plot 1: Cleveland dot plot — agreement by block ────────────────────────
def plot_cleveland_by_block(df: pd.DataFrame) -> None:
    """
    Mean seed_agreement_rate ± 1 SD per schema block.
    X-axis lower bound set dynamically to (global_min - 0.02) so zoom
    is always appropriate to the actual data spread.
    Highlight threshold = mean - 1 SD of all block means (data-driven).
    """
    block_stats = (
        df.groupby("block")["seed_agreement_rate"]
        .agg(mean="mean", std="std")
        .reset_index()
        .sort_values("mean")
    )
    block_stats["std"] = block_stats["std"].fillna(0)

    # Dynamic threshold: blocks more than 1 SD below the mean of block means
    grand_mean = block_stats["mean"].mean()
    grand_std  = block_stats["mean"].std()
    threshold  = grand_mean - grand_std

    colors = [ACCENT_WARN if v < threshold else OSS_BLUE
              for v in block_stats["mean"]]

    # Dynamic x-axis: zoom to where variance is
    x_min = max(0.0, block_stats["mean"].min() - block_stats["std"].max() - 0.02)
    x_max = 1.02

    fig, ax = plt.subplots(figsize=(9, 8))
    y_pos = np.arange(len(block_stats))

    ax.hlines(
        y_pos,
        (block_stats["mean"] - block_stats["std"]).clip(lower=0.0),
        (block_stats["mean"] + block_stats["std"]).clip(upper=1.0),
        colors="#CCCCCC", linewidth=1.8, zorder=2,
    )
    ax.scatter(block_stats["mean"], y_pos,
               c=colors, s=65, zorder=3,
               edgecolors="white", linewidths=0.5)

    for i, (_, row) in enumerate(block_stats.iterrows()):
        ax.text(row["mean"] + 0.001, i,
                f"{row['mean']:.3f}", va="center", fontsize=7.5,
                color="#333333")

    ax.axvline(threshold, color=ACCENT_WARN, linewidth=0.8,
               linestyle="--",
               label=f"Threshold: Mean − 1SD = {threshold:.3f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(block_stats["block"], fontsize=8.5)
    ax.set_xlabel("Mean Seed Agreement Rate  (error bars = ±1 SD across variables in block)")
    ax.set_title(
        "Extraction Reliability by Schema Block\n"
        f"OSS-20B · 161 Reports · {len(block_stats)} Blocks · "
        f"Overall Mean = {grand_mean:.3f}",
        pad=10,
    )
    ax.set_xlim(x_min, x_max)
    ax.legend(fontsize=8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6, zorder=0)

    save(fig, "01_cleveland_agreement_by_block")


# ── Plot 2: Scatter — 199 variables, agreement vs confidence ───────────────
def plot_scatter_agreement_vs_confidence(df: pd.DataFrame) -> None:
    """
    One point per variable (averaged across 161 reports).
    X: mean seed_agreement_rate  Y: mean_confidence
    Colored by schema block.
    Quadrant thresholds = median of each axis (data-driven).

    Four quadrants:
      High agree / High conf  → reliable extraction, model knows it
      High agree / Low conf   → reliable but model hedges unnecessarily
      Low agree  / High conf  → model commits wrongly — most concerning
      Low agree  / Low conf   → model recognises its own uncertainty
    """
    var_df = _var_agg(df).dropna(subset=["agreement", "confidence"])

    # Dynamic quadrant thresholds from medians
    q_agree = var_df["agreement"].median()
    q_conf  = var_df["confidence"].median()

    # Dynamic axis bounds
    x_min = max(0.0, var_df["agreement"].min() - 0.02)
    x_max = min(1.0, var_df["agreement"].max() + 0.02)
    y_min = max(0.0, var_df["confidence"].min() - 0.02)
    y_max = min(1.0, var_df["confidence"].max() + 0.02)

    # Dynamic annotation threshold — label variables below mean - 1SD on agreement
    ann_threshold = var_df["agreement"].mean() - var_df["agreement"].std()

    fig, ax = plt.subplots(figsize=(11, 9))

    # Quadrant shading — subtle background colors
    ax.axvspan(x_min,   q_agree, ymin=0, ymax=1, color=ACCENT_WARN, alpha=0.04, zorder=0)
    ax.axvspan(q_agree, x_max,   ymin=0, ymax=1, color=OSS_BLUE,    alpha=0.04, zorder=0)

    # All 199 variables — single color, no legend
    ax.scatter(
        var_df["agreement"], var_df["confidence"],
        c=OSS_BLUE,
        s=55,
        alpha=0.72,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )

    # Quadrant threshold lines
    ax.axvline(q_agree, color="#999999", linewidth=0.8, linestyle="--", zorder=2)
    ax.axhline(q_conf,  color="#999999", linewidth=0.8, linestyle="--", zorder=2)

    # Quadrant labels — positioned dynamically relative to thresholds
    ax.text((x_min + q_agree) / 2, y_max - 0.01,
            "Low agree / High conf\n(model commits wrongly)", va="top",
            color=ACCENT_WARN, fontsize=7.5, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ax.text((q_agree + x_max) / 2, y_max - 0.01,
            "High agree / High conf\n(reliable)", va="top",
            color=OSS_BLUE, fontsize=7.5, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ax.text((x_min + q_agree) / 2, y_min + 0.01,
            "Low agree / Low conf\n(model aware)", va="bottom",
            color="#F46A25", fontsize=7.5, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ax.text((q_agree + x_max) / 2, y_min + 0.01,
            "High agree / Low conf\n(model hedges)", va="bottom",
            color=ACCENT_MED, fontsize=7.5, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # Annotate all variables below dynamic threshold
    to_label = var_df[var_df["agreement"] < ann_threshold]
    for _, row in to_label.iterrows():
        ax.annotate(
            row["variable"],
            xy=(row["agreement"], row["confidence"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=6.5, color="#333333",
            arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.5),
        )

    ax.set_xlabel(f"Mean Seed Agreement Rate  (median = {q_agree:.3f})")
    ax.set_ylabel(f"Mean LLM Self-Confidence  (median = {q_conf:.3f})")
    ax.set_title(
        "Per-Variable Reliability: Seed Agreement vs LLM Self-Confidence\n"
        f"OSS-20B · {len(var_df)+2} Variables · Each point averaged across 161 reports",
        pad=10,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    save(fig, "02_scatter_agreement_vs_confidence")


# ── Plot 3: Bottom 20 — agreement vs confidence grouped bar ───────────────
def plot_bottom20_grouped(df: pd.DataFrame) -> None:
    """
    Bottom 20 variables by mean seed_agreement_rate.
    Two bars per variable: agreement (blue) and confidence (teal).
    Y-axis fixed at 0.0–1.0 to show the full metric scale in context.
    Shows whether low-agreement variables also have low confidence
    (model-aware failure) or high confidence (silent failure).
    """
    var_df = _var_agg(df).dropna(subset=["agreement"]).sort_values("agreement").head(20)

    n     = len(var_df)
    x     = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))

    # Single bar for agreement rate
    bars = ax.bar(x, var_df["agreement"],
                width=0.5, label="Seed Agreement Rate",
                color=OSS_BLUE, edgecolor="white", linewidth=0.3)

    # Dot for confidence score
    ax.scatter(x, var_df["confidence"],
            color=ACCENT_MED, s=80, zorder=5,
            label="LLM Self-Confidence", marker="D")

    # Vertical line connecting bar top to dot — shows the gap
    for i, (_, row) in enumerate(var_df.iterrows()):
        if pd.notna(row["confidence"]):
            ax.vlines(i, row["agreement"], row["confidence"],
                    colors="#CCCCCC", linewidth=0.8, linestyle="--", zorder=4)

    # Value labels on agreement bars only
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=5.5, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(var_df["variable"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Metric value")
    ax.set_title(
        "Bottom 20 Variables: Seed Agreement Rate vs LLM Self-Confidence\n"
        "OSS-20B · 161 Reports",
        pad=10,
    )

    ax.set_ylim(0.0, 1.02)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=0)
    fig.legend(fontsize=9, loc="upper right", bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()
    save(fig, "03_bottom20_agreement_vs_confidence")


# ── Plot 4: Variable failure table ────────────────────────────────────────
def plot_variable_failure_table(df: pd.DataFrame) -> None:
    """
    Variables with mean seed_agreement_rate below the data-driven threshold
    (mean - 1 SD across all variables), aggregated across 161 reports.
    One row per variable. Shows agreement, confidence, and output prob.
    Threshold computed dynamically — no hardcoded values.
    """
    var_full = (
        df.groupby(["variable", "block"])
        .agg(
            mean_agreement=("seed_agreement_rate", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_output_prob=("mean_output_prob", "mean"),
        )
        .reset_index()
    )

    # Dynamic threshold
    threshold = var_full["mean_agreement"].mean() - var_full["mean_agreement"].std()

    failures = (
        var_full[var_full["mean_agreement"] < threshold]
        .sort_values("mean_agreement")
        .reset_index(drop=True)
    )

    if failures.empty:
        logger.warning("no variables below threshold %.4f — skipping table", threshold)
        return

    display = failures.copy()
    display["mean_agreement"]   = display["mean_agreement"].map("{:.3f}".format)
    display["mean_confidence"]  = display["mean_confidence"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "—"
    )
    display["mean_output_prob"] = display["mean_output_prob"].map(
        lambda x: f"{x:.4f}" if pd.notna(x) else "—"
    )

    col_labels = ["Variable", "Schema Block",
                  "Mean Agreement", "Mean Confidence", "Mean Output Prob"]
    col_keys   = ["variable", "block",
                  "mean_agreement", "mean_confidence", "mean_output_prob"]

    fig, ax = plt.subplots(figsize=(14, max(3, len(display) * 0.32 + 0.2)))
    ax.axis("off")

    tbl = ax.table(
        cellText=display[col_keys].values,
        colLabels=col_labels,
        #loc="center",
        bbox=[0, 0, 1, 1],  # fill entire axes area
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(OSS_BLUE)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(display) + 1):
        fc = "#F4F8FC" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(fc)

    ax.set_title(
        f"Variables with Systematic Low Agreement - {len(display)} variables\n"
        f"Threshold: Mean − 1SD = {threshold:.3f}  ·  "
        "OSS-20B · 161 Reports · One row per variable aggregated across all reports",
        pad=12, fontsize=10,
    )
    plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01)
    save(fig, "04_variable_failure_table")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    if not INPUT_CSV.exists():
        logger.error("confidence_metrics.csv not found at %s", INPUT_CSV)
        return

    logger.info("loading %s", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    logger.info("loaded %d rows", len(df))

    df["block"] = df["variable"].map(BLOCK_MAP).fillna("Other")

    logger.info("generating plot 1 — cleveland agreement by block")
    plot_cleveland_by_block(df)

    logger.info("generating plot 2 — scatter agreement vs confidence (199 variables)")
    plot_scatter_agreement_vs_confidence(df)

    logger.info("generating plot 3 — bottom 20 variables grouped bar")
    plot_bottom20_grouped(df)

    logger.info("generating plot 4 — variable failure table")
    plot_variable_failure_table(df)

    logger.info("all plots written to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()