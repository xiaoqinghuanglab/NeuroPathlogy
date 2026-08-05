"""analyze_residual.py - Residual variable cohort analysis.

Reads per-report per-seed .residual.json files from the production output
directory, applies majority voting across 5 seeds, then produces:

FIGURES (600 DPI PNG + SVG):
    results/residual_analysis/
        asymmetry_distributions.png/.svg   - box plots, 4 weight delta variables
        cow_diameter_profile.png/.svg      - CoW vessel diameters, measured vessels only
        cc_thickness_profile.png/.svg      - corpus callosum segment thickness line plot
        pathology_severity_freq.png/.svg   - stacked bar, ordinal severity frequencies

── CONFIGURE ──────────────────────────────────────────────────────────────────
"""

import json
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# ── Configuration ───────────────────────────────────────────────────────────
MODEL        = "oss-20b"
PROJECT      = Path("/N/project/ADRD/neuropathoroot")
INPUT_DIR    = PROJECT / "output" / MODEL / "residual_extraction"
OUTPUT_DIR   = PROJECT / "results" / "residual_analysis"
N_SEEDS      = 5
CONT_UNKNOWN = 9999.0
ORD_UNKNOWN  = 9

# ── UK Government Analysis Function palette ─────────────────────────────────
DARK_BLUE  = "#12436D"
DARK_PINK  = "#801650"
TURQUOISE  = "#28A197"
ORANGE     = "#F46A25"
DARK_GREY  = "#3D3D3D"
MID_GREY   = "#A8A8A8"

plt.rcParams.update({
    "font.family":           "DejaVu Sans",
    "font.size":             9,
    "axes.titlesize":        11,
    "axes.titleweight":      "bold",
    "axes.labelsize":        10,
    "axes.labelweight":      "normal",
    "xtick.labelsize":       8.5,
    "ytick.labelsize":       8.5,
    "legend.fontsize":       8,
    "legend.title_fontsize": 9,
    "figure.dpi":            150,
    "savefig.dpi":           600,
    "savefig.bbox":          "tight",
    "axes.spines.top":       False,
    "axes.spines.right":     False,
})

# ── Variable definitions ─────────────────────────────────────────────────────
DELTA_VARS = [
    "hemibrain_weight_delta_g",
    "cerebral_weight_delta_g",
    "cerebellar_weight_delta_g",
    "brainstem_weight_delta_g",
]
DELTA_LABELS = ["Hemibrain\n(g)", "Cerebral\n(g)", "Cerebellar\n(g)", "Brainstem\n(g)"]
DELTA_COLORS = [DARK_BLUE, DARK_PINK, TURQUOISE, ORANGE]

COW_VARS = [
    "cow_basilar_mm",
    "cow_vertebral_right_mm", "cow_vertebral_left_mm",
    "cow_ica_right_mm",       "cow_ica_left_mm",
    "cow_mca_right_mm",       "cow_mca_left_mm",
    "cow_aca_right_mm",       "cow_aca_left_mm",
    "cow_pca_right_mm",       "cow_pca_left_mm",
    "cow_pcom_right_mm",      "cow_pcom_left_mm",
    "cow_acom_mm",
]
COW_LABELS = [
    "Basilar",
    "Vertebral R", "Vertebral L",
    "ICA R",       "ICA L",
    "MCA R",       "MCA L",
    "ACA R",       "ACA L",
    "PCA R",       "PCA L",
    "PCom R",      "PCom L",
    "ACom",
]

CC_VARS = [
    "corpus_callosum_genu_mm",
    "corpus_callosum_body_anterior_mm",
    "corpus_callosum_body_mid_mm",
    "corpus_callosum_body_posterior_mm",
    "corpus_callosum_splenium_mm",
]
CC_LABELS = ["Genu", "Anterior\nBody", "Mid\nBody", "Posterior\nBody", "Splenium"]

ORDINAL_VARS = [
    "lateral_ventricle_enlargement_severity",
    "amygdala_atrophy_severity",
    "dilated_perivascular_spaces",
    "cerebellar_atrophy_severity",
    "globus_pallidus_neuronal_loss_severity",
    "basal_ganglia_atrophy",
    "thalamic_degeneration_severity",
    "brainstem_atrophy_severity",
    "purkinje_cell_loss_severity",
    "dentate_nucleus_atrophy",
    "artag_severity",
    "gvd_hippocampus_severity",
    "hirano_bodies_hippocampus_severity",
]
ORDINAL_LABELS = [
    "Lat. Ventricle\nEnlargement",
    "Amygdala\nAtrophy",
    "Dilated Periv.\nSpaces",
    "Cerebellar\nAtrophy",
    "Globus Pallidus\nNeuronal Loss",
    "Basal Ganglia\nAtrophy",
    "Thalamic\nDegeneration",
    "Brainstem\nAtrophy",
    "Purkinje Cell\nLoss",
    "Dentate Nucleus\nAtrophy",
    "ARTAG",
    "GVD\nHippocampus",
    "Hirano Bodies\nHippocampus",
]

# Severity color scheme: none/absent, mild, moderate, severe
SEV_COLORS = {
    0: "#D0E8F2",  # none/absent - light blue
    1: "#FEF0D9",  # mild - light yellow
    2: "#F46A25",  # moderate - orange
    3: "#CC0000",  # severe - red
}
SEV_LABELS = {0: "None (0)", 1: "Mild (1)", 2: "Moderate (2)", 3: "Severe (3)"}


# ==============================================================================
#  Data loading and majority vote
# ==============================================================================

def flatten_residual(d: dict) -> dict:
    SKIP = {"field_annotations", "extraction_confidence", "extraction_notes",
            "report_id", "seed", "model"}
    out = {}
    for k, v in d.items():
        if k in SKIP:
            continue
        if isinstance(v, dict):
            for sk, sv in v.items():
                if not isinstance(sv, dict):
                    out[sk] = sv
        else:
            out[k] = v
    return out


def normalize(v):
    if v is None:
        return None
    if isinstance(v, float):
        return round(v, 2)
    try:
        f = round(float(v), 2)
        return int(f) if f == int(f) else f
    except (TypeError, ValueError):
        return v


def majority_vote(values):
    normed = [normalize(v) for v in values]
    counts = Counter(k for k in normed if k is not None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def load_data(input_dir: Path) -> dict[str, dict]:
    """Returns {report_id: {var: majority_voted_value}}"""
    pdf_seeds: dict[str, dict[int, dict]] = {}
    for f in sorted(input_dir.glob("*.residual.json")):
        for seed in range(N_SEEDS):
            suffix = f"_seed{seed}.residual.json"
            if f.name.endswith(suffix):
                report_id = f.name[: -len(suffix)]
                try:
                    data = flatten_residual(json.loads(f.read_text()))
                    pdf_seeds.setdefault(report_id, {})[seed] = data
                except Exception:
                    pass
                break

    results = {}
    for report_id, seeds in pdf_seeds.items():
        all_vars = set(v for d in seeds.values() for v in d)
        results[report_id] = {
            var: majority_vote([seeds.get(s, {}).get(var) for s in range(N_SEEDS)])
            for var in all_vars
        }
    return results


# ==============================================================================
#  Figure 1 - Weight asymmetry box plots
# ==============================================================================

def plot_asymmetry(case_data: dict, out_dir: Path):
    delta_vals = {v: [] for v in DELTA_VARS}
    for row in case_data.values():
        for var in DELTA_VARS:
            val = row.get(var)
            if val is not None and val != CONT_UNKNOWN:
                delta_vals[var].append(float(val))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    positions = np.arange(len(DELTA_VARS))
    bp = ax.boxplot(
        [delta_vals[v] for v in DELTA_VARS],
        positions=positions,
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.2, color=DARK_GREY),
        capprops=dict(linewidth=1.2, color=DARK_GREY),
        flierprops=dict(marker="o", markerfacecolor=MID_GREY,
                        markeredgecolor=MID_GREY, markersize=4, alpha=0.6),
        boxprops=dict(linewidth=1.2),
    )
    for patch, color in zip(bp["boxes"], DELTA_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.axhline(0, color=DARK_GREY, linewidth=1.0, linestyle="--",
               alpha=0.6, label="No asymmetry (delta = 0)")

    # Place n= label just above the upper whisker cap of each box
    for i, (var, whisker) in enumerate(zip(DELTA_VARS, bp["caps"][1::2])):
        cap_y = whisker.get_ydata()[1]
        ax.text(i, cap_y + 0.5, f"n={len(delta_vals[var])}",
                ha="center", va="bottom", fontsize=7.5, color=DARK_GREY)

    ax.set_xticks(positions)
    ax.set_xticklabels(DELTA_LABELS, fontsize=9)
    ax.set_ylabel("Weight asymmetry - right minus left (g)", fontsize=10)
    ax.set_title(
        "Bilateral Brain Weight Asymmetry Across the Cohort\n"
        "Negative values indicate left hemisphere heavier than right",
        fontsize=10, pad=12
    )
    ax.grid(axis="y", alpha=0.3, linewidth=0.6, zorder=0)
    ax.legend(loc="upper right", frameon=True, edgecolor="#CCCCCC", framealpha=0.95)

    plt.tight_layout()
    _save(fig, out_dir, "asymmetry_distributions")


# ==============================================================================
#  Figure 2 - CoW vessel diameter profile (measured vessels only)
# ==============================================================================

def plot_cow_diameter(case_data: dict, out_dir: Path):
    means, sds, ns, labels_used = [], [], [], []

    for var, label in zip(COW_VARS, COW_LABELS):
        vals = []
        for row in case_data.values():
            v = row.get(var)
            if v is not None and v != CONT_UNKNOWN and float(v) > 0.0:
                vals.append(float(v))
        if vals:
            means.append(float(np.mean(vals)))
            sds.append(float(np.std(vals)))
            ns.append(len(vals))
            labels_used.append(label)

    y = np.arange(len(labels_used))

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels_used) * 0.55)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    ax.barh(y, means, height=0.55, color=DARK_BLUE, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.errorbar(means, y, xerr=sds, fmt="none",
                ecolor=DARK_GREY, elinewidth=1.0, capsize=3.5,
                capthick=1.0, alpha=0.7)

    for i, (m, s, n) in enumerate(zip(means, sds, ns)):
        ax.text(m + s + 0.05, i,
                f"{m:.1f} +/- {s:.1f}  (n={n})",
                va="center", ha="left", fontsize=7.5, color=DARK_GREY)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_used, fontsize=9)
    ax.set_xlabel("Diameter (mm)", fontsize=10)
    ax.set_xlim(0, 6)
    ax.set_title(
        "Circle of Willis Vessel Diameters\n"
        "Measured vessels only (mean +/- SD)",
        fontsize=11, pad=10
    )
    ax.grid(axis="x", alpha=0.3, linewidth=0.6, zorder=0)

    plt.tight_layout()
    _save(fig, out_dir, "cow_diameter_profile")


# ==============================================================================
#  Figure 3 - Corpus callosum thickness profile
# ==============================================================================

def plot_cc_profile(case_data: dict, out_dir: Path):
    cc_vals = {v: [] for v in CC_VARS}
    for row in case_data.values():
        for var in CC_VARS:
            val = row.get(var)
            if val is not None and val != CONT_UNKNOWN:
                cc_vals[var].append(float(val))

    means   = [np.mean(cc_vals[v]) if cc_vals[v] else np.nan for v in CC_VARS]
    sds     = [np.std(cc_vals[v])  if cc_vals[v] else np.nan for v in CC_VARS]
    ns      = [len(cc_vals[v]) for v in CC_VARS]
    x       = np.arange(len(CC_VARS))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    # Shaded SD band
    ax.fill_between(x,
                    [m - s for m, s in zip(means, sds)],
                    [m + s for m, s in zip(means, sds)],
                    color=DARK_BLUE, alpha=0.15, label="Mean +/- SD")

    # Mean line
    ax.plot(x, means, color=DARK_BLUE, linewidth=2.2,
            marker="o", markersize=7, markerfacecolor=DARK_BLUE,
            markeredgecolor="white", markeredgewidth=1.2, label="Mean thickness")

    # Annotate each point with mean and n
    for i, (m, s, n) in enumerate(zip(means, sds, ns)):
        if not np.isnan(m):
            ax.text(i, m + s + 0.4,
                    f"{m:.1f} mm\n(n={n})",
                    ha="center", va="bottom", fontsize=7.5, color=DARK_GREY)

    ax.set_xticks(x)
    ax.set_xticklabels(CC_LABELS, fontsize=9.5)
    ax.set_ylabel("Thickness (mm)", fontsize=10)
    ax.set_xlabel("Corpus callosum segment", fontsize=10)
    ax.set_title(
        "Corpus Callosum Segment Thickness Across the Cohort\n"
        "Mean +/- SD; anterior to posterior",
        fontsize=11, pad=10
    )
    ax.grid(axis="y", alpha=0.3, linewidth=0.6, zorder=0)
    ax.legend(loc="lower right", frameon=True, edgecolor="#CCCCCC", framealpha=0.95)

    plt.tight_layout()
    _save(fig, out_dir, "cc_thickness_profile")


# ==============================================================================
#  Figure 4 - Pathology severity frequency stacked bar
# ==============================================================================

def plot_severity_freq(case_data: dict, out_dir: Path):
    n_total = len(case_data)

    # Count severity codes per variable (only 0/1/2/3 - exclude 9/missing)
    freq = {var: {0: 0, 1: 0, 2: 0, 3: 0} for var in ORDINAL_VARS}
    n_reported = {var: 0 for var in ORDINAL_VARS}

    for row in case_data.values():
        for var in ORDINAL_VARS:
            val = row.get(var, ORD_UNKNOWN)
            if val in (0, 1, 2, 3):
                freq[var][val] += 1
                n_reported[var] += 1

    # Convert to percentage of all cases (denominator = n_total throughout)
    pct = {var: {code: 0.0 for code in (0, 1, 2, 3)} for var in ORDINAL_VARS}
    for var in ORDINAL_VARS:
        if n_reported[var] > 0:
            for code in (0, 1, 2, 3):
                pct[var][code] = 100.0 * freq[var][code] / n_total

    x = np.arange(len(ORDINAL_VARS))
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    bottoms = np.zeros(len(ORDINAL_VARS))
    for code in (0, 1, 2, 3):
        heights = np.array([pct[var][code] for var in ORDINAL_VARS])
        ax.bar(x, heights, bottom=bottoms,
               color=SEV_COLORS[code], label=SEV_LABELS[code],
               edgecolor="white", linewidth=0.5, width=0.65)
        # Label segments that are wide enough to read
        for i, (h, b) in enumerate(zip(heights, bottoms)):
            if h >= 5:
                ax.text(i, b + h / 2, f"{h:.0f}%",
                        ha="center", va="center",
                        fontsize=7, color=DARK_GREY, fontweight="bold")
        bottoms += heights

    # Add "not mentioned %" as a text annotation above each bar
    for i, var in enumerate(ORDINAL_VARS):
        not_mentioned_pct = 100.0 * (n_total - n_reported[var]) / n_total
        ax.text(i, bottoms[i] + 0.8,
                f"{not_mentioned_pct:.0f}%\nnot\nmentioned",
                ha="center", va="bottom", fontsize=6, color=MID_GREY)

    ax.set_xticks(x)
    ax.set_xticklabels(ORDINAL_LABELS, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Percentage of cases (%)", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_title(
        "Regional Neuropathological Severity - Frequency Across the Cohort\n"
        "Denominator: all cases; unlabelled remainder = not mentioned",
        fontsize=10, pad=10
    )
    ax.grid(axis="y", alpha=0.3, linewidth=0.6, zorder=0)
    ax.legend(loc="upper right", frameon=True, edgecolor="#CCCCCC",
              framealpha=0.95, title="Severity grade")

    plt.tight_layout()
    _save(fig, out_dir, "pathology_severity_freq")


# ==============================================================================
#  Save helper
# ==============================================================================

def _save(fig, out_dir: Path, stem: str):
    for ext in ("png", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, format=ext, bbox_inches="tight",
                    facecolor="white", dpi=600 if ext == "png" else None)
        print(f"  Saved {path.name}")
    plt.close(fig)


# ==============================================================================
#  Main
# ==============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading residual JSONs from {INPUT_DIR} ...")
    case_data = load_data(INPUT_DIR)
    n = len(case_data)
    print(f"  Loaded {n} reports")

    if n == 0:
        print("No data found. Check INPUT_DIR.")
        return

    print("\nGenerating figures ...")
    plot_asymmetry(case_data, OUTPUT_DIR)
    plot_cow_diameter(case_data, OUTPUT_DIR)
    plot_cc_profile(case_data, OUTPUT_DIR)
    plot_severity_freq(case_data, OUTPUT_DIR)

    print(f"""
Done.
  {OUTPUT_DIR}/
    asymmetry_distributions.png / .svg   - weight asymmetry box plots
    cow_diameter_profile.png / .svg      - CoW diameters, measured vessels only
    cc_thickness_profile.png / .svg      - corpus callosum segment thickness
    pathology_severity_freq.png / .svg   - ordinal severity frequency stacked bar
""")


if __name__ == "__main__":
    main()