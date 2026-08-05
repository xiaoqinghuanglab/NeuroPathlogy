"""compare_models_gt.py — Per-variable model comparison with ground truth.

OUTPUT
------
results_gt/
    plots_png/
        plot_FIELDNAME.png          600 DPI
    plots_svg/
        plot_FIELDNAME.svg          vector, for figure editing
    model_comparison_gt.xlsx
        Sheet "Summary"       — model scorecard, % correct, consistently wrong variables,
                                perfect variables, best model per report
        Sheet "Variable Detail" — all variables × all reports × models, GT match at a glance
        Sheet "Details"       — one row per (variable × report × model), all 5 seed
                                values, mean, majority, GT value, match flag.
                                flat table of ~516 rows (4 reports × 43 vars × 3 models)
        Sheet "Legend"        — colour key, metric definitions, plot guide

COLOURS  UK Government Analysis Function palette (distinct, colourblind-safe)
    OSS-20B      → Dark Blue  #12436D
    Qwen2.5-14B  → Dark Pink  #801650
    Llama3.1-8B  → Turquoise  #28A197

Font: DejaVu Sans (Arial equivalent), medical journal figure sizes
"""

import csv
import json
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.cell.cell import TYPE_STRING
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend import Legend
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="No artists with labels")

_ROOT = Path("/N/project/ADRD/neuropathoroot/")
OSS_DIR     = _ROOT / "output_gt" / "oss-20b"
QWEN_DIR    = _ROOT / "output_gt" / "qwen2.5-14b"
LLAMA_DIR   = _ROOT / "output_gt" / "llama3.1-8b"
TRUTH_FILE  = _ROOT / "csv_input" / "ground_truth.csv"
RESULTS_DIR = _ROOT / "results_gt"

SEEDS = [0, 1, 2, 3, 4]
SKIP  = {"extraction_notes", "extraction_confidence", "field_annotations"}

MODELS = [
    ("OSS-20B",     "oss"),
    ("Qwen2.5-14B", "qwen"),
    ("Llama3.1-8B", "llama"),
]

MODEL_COLORS = {
    "oss":   "#12436D",
    "qwen":  "#801650",
    "llama": "#28A197",
}

GT_MATCH_COLOR    = "#117733"
GT_MISMATCH_COLOR = "#CC3311"
TIE_COLOR         = "#B7770D"  # amber text
TIE_FILL          = "FEF3E2"   # amber cell fill
TIE_BORDER        = "EF9F27"   # amber border

plt.rcParams.update({
    "font.family":           "DejaVu Sans",
    "font.size":             9,
    "axes.titlesize":        11,
    "axes.titleweight":      "bold",
    "axes.labelsize":        10,
    "axes.labelweight":      "normal",
    "xtick.labelsize":       8,
    "ytick.labelsize":       9,
    "legend.fontsize":       8,
    "legend.title_fontsize": 9,
    "figure.dpi":            150,
    "savefig.dpi":           600,
    "savefig.bbox":          "tight",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def flatten(d):
    out = {}
    for k, v in d.items():
        if k in SKIP:          # skip field_annotations, extraction_notes, etc.
            continue
        if isinstance(v, dict):
            for sk, sv in v.items():
                if not isinstance(sv, dict):   # only take scalar values
                    out[sk] = sv
        else:
            out[k] = v
    return out


def normalize(v):
    if v is None:
        return None
    if isinstance(v, dict) or isinstance(v, list):
        return str(v)   # convert nested objects to string so they're hashable
    if isinstance(v, float):
        return round(v, 1)
    try:
        f = round(float(v), 1)
        return int(f) if f == int(f) else f
    except (TypeError, ValueError):
        return str(v) if not isinstance(v, str) else v


def majority_vote(values):
    normed = [normalize(v) for v in values]
    counts = {}
    for v in normed:
        key = "__null__" if v is None else v
        counts[key] = counts.get(key, 0) + 1
    majority_key = max(counts, key=counts.get)
    return None if majority_key == "__null__" else majority_key


def compute_stats(values):
    normed = [normalize(v) for v in values]
    counts = {}
    for v in normed:
        key = "__null__" if v is None else v
        counts[key] = counts.get(key, 0) + 1
    max_count = max(counts.values())
    top_keys  = [k for k, c in counts.items() if c == max_count]
    is_tie    = len(top_keys) > 1

    maj      = majority_vote(values)
    per_seed = [100.0 if normalize(v) == maj else 0.0 for v in values]
    mean_acc = float(np.mean(per_seed))
    std_acc  = float(np.std(per_seed))

    # comma-separated string of all tied values, e.g. "0,2"
    if is_tie:
        tied_str = ",".join(
            str(k) if k != "__null__" else "—" for k in top_keys
        )
    else:
        tied_str = None

    return per_seed, mean_acc, std_acc, maj, is_tie, tied_str


def gt_match(maj, gt_val):
    if gt_val is None and maj is None:
        return "match"
    if gt_val is None and maj is not None:
        return "mismatch"
    if maj is None:
        return "mismatch"
    return "match" if normalize(maj) == normalize(gt_val) else "mismatch"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_model_data(output_dir: Path) -> dict:
    data = defaultdict(dict)
    for f in sorted(output_dir.glob("*.extracted.json")):
        for seed in SEEDS:
            if f"_seed{seed}.extracted" in f.name:
                pdf_name = f.name.replace(f"_seed{seed}.extracted.json", "")
                with open(f) as fh:
                    data[pdf_name][seed] = flatten(json.load(fh))
                break
    return dict(data)


def load_ground_truth(truth_path: Path) -> dict:
    truth = {}
    with open(truth_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fields = list(row.keys())
            report = row[fields[0]].replace(".pdf", "").replace(" copy", "").strip()
            truth[report] = {}
            for field in fields[1:]:
                val = row[field].strip()
                if val == "" or val.lower() == "none":
                    truth[report][field] = None
                else:
                    try:
                        f = round(float(val), 1)
                        truth[report][field] = int(f) if f == int(f) else f
                    except ValueError:
                        truth[report][field] = val
    return truth


def get_truth_row(truth, pdf):
    return (truth.get(pdf)
            or truth.get(pdf.replace(" copy", ""))
            or truth.get(pdf + " copy")
            or {})


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_plot(field, all_pdfs, stats_for_field, truth, png_dir, svg_dir):
    """
    stats_for_field: dict[mk] -> dict[pdf] -> {vals, per_seed, mean, std, maj}
    Saves one PNG (600 DPI) to png_dir and one SVG to svg_dir.
    """
    n_models  = len(MODELS)
    n_reports = len(all_pdfs)
    bar_width  = 0.22
    group_gap  = 0.32
    group_size = n_models * bar_width + group_gap

    # fig_w = max(12, n_reports * 3.0)
    max_val_len = max(
        len(str(stats_for_field[mk][pdf]["maj"] or "—"))
        for _, mk in MODELS for pdf in all_pdfs
    )
    fig_w = max(14, n_reports * 3.0 + max_val_len * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    group_centers = np.arange(n_reports) * group_size

    for mi, (model_label, mk) in enumerate(MODELS):
        color  = MODEL_COLORS[mk]
        offset = (mi - (n_models - 1) / 2) * bar_width

        for pi, pdf in enumerate(all_pdfs):
            s     = stats_for_field[mk][pdf]
            x     = group_centers[pi] + offset
            mean  = s["mean"]
            std   = s["std"]
            maj   = s["maj"]
            is_tie   = s["is_tie"]
            tied_str = s["tied_str"]

            truth_row = get_truth_row(truth, pdf)
            gt_val    = truth_row.get(field)
            match     = gt_match(maj, gt_val)

            # Reduce opacity for mismatches — pre-attentive visual cue
            bar_alpha = 0.55 if is_tie else (0.90 if match == "match" else 0.55)

            ax.bar(x, mean, width=bar_width * 0.88,
                   color=color, alpha=bar_alpha, zorder=2,
                   edgecolor="white", linewidth=0.6,
                   #label=model_label if pi == 0 else "_nolegend_")
                   label="_nolegend_")

            if is_tie:
                ax.plot(x, 104, marker=r"$\sim$",
                        color=TIE_COLOR, markersize=11,
                        markerfacecolor=TIE_COLOR,
                        markeredgecolor=TIE_COLOR, markeredgewidth=0.3, zorder=6,
                        linestyle="none")
            else:
                # GT marker floats just above bar top — no fixed dead-space y
                
                if match == "match":
                    ax.plot(x, 104, marker="$✓$",
                            color=GT_MATCH_COLOR, markersize=10,
                            markerfacecolor=GT_MATCH_COLOR,
                            markeredgecolor=GT_MATCH_COLOR, markeredgewidth=0.3, zorder=6,
                            linestyle="none")
                else:
                    ax.plot(x, 104, marker="$✕$",
                            color=GT_MISMATCH_COLOR, markersize=9,
                            markerfacecolor=GT_MISMATCH_COLOR,
                            markeredgecolor=GT_MISMATCH_COLOR, markeredgewidth=0.3, zorder=6,
                            linestyle="none")

            # Mean % inside bar if tall enough
            if mean >= 18:
                ax.text(x, mean / 2, f"{mean:.0f}%",
                        ha="center", va="center",
                        fontsize=6.5, fontweight="bold",
                        color="white", zorder=7)

            # Majority value + GT below x-axis
            maj_str  = tied_str if is_tie else (str(maj) if maj is not None else "—")
            gt_str  = str(gt_val) if gt_val is not None else "—"
            ax.text(x, -6,    f"v={maj_str}",
                    ha="center", va="top", fontsize=5.5,
                    color=TIE_COLOR if is_tie else color,
                    fontweight="bold", zorder=7)
            ax.text(x, -10.5, f"GT={gt_str}",
                    ha="center", va="top", fontsize=5.5,
                    color="#555555", zorder=7)

    # Group dividers
    for i in range(n_reports - 1):
        mid = (group_centers[i] + group_centers[i + 1]) / 2
        ax.axvline(mid, color="#cccccc", linestyle="--", linewidth=0.7, zorder=1)

    ax.axhline(100, color="#666666", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhspan(80, 100, color="#e8f5e9", alpha=0.35, zorder=0)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(all_pdfs, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(-16, 115)
    ax.set_xlim(group_centers[0] - group_size * 0.6,
                group_centers[-1] + group_size * 0.6)
    ax.set_ylabel("Mean agreement across seeds (%)", labelpad=6)
    ax.set_xlabel("Report", labelpad=6)
    ax.set_title(
        f"Extraction Agreement vs Ground Truth  —  {field}\n"
        "Bar = mean agreement (5 seeds)  ·  "
        "Marker = Majority vs Ground Truth",
        pad=10, fontsize=10
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", length=3, width=0.8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6, zorder=0)

    model_patches = [
        mpatches.Patch(color=MODEL_COLORS[mk], label=label, alpha=0.9)
        for label, mk in MODELS
    ]
    gt_handles = [
        mlines.Line2D([], [], marker="$✓$", color=GT_MATCH_COLOR,
                      markerfacecolor=GT_MATCH_COLOR,
                      markeredgecolor=GT_MATCH_COLOR,
                      markersize=8, linestyle="none",
                      label="Majority = GT"),
        mlines.Line2D([], [], marker="$✕$", color=GT_MISMATCH_COLOR,
                      markerfacecolor=GT_MISMATCH_COLOR,
                      markeredgecolor=GT_MISMATCH_COLOR,
                      markersize=7, linestyle="none",
                      label="Majority ≠ GT"),
        mlines.Line2D([], [], marker=r"$\sim$", color=TIE_COLOR,
                      markerfacecolor=TIE_COLOR,
                      markeredgecolor=TIE_COLOR,
                      markersize=9, linestyle="none",
                      label="Seeds tied (v= shows all values)"),
    ]
    
    # legends
    leg1 = Legend(ax, model_patches, [h.get_label() for h in model_patches],
                bbox_to_anchor=(1.01, 1.0), loc="upper left",
                frameon=True, edgecolor="#cccccc", title="Model", framealpha=0.95)
    ax.add_artist(leg1)

    leg2 = Legend(ax, gt_handles, ["Majority = GT", "Majority ≠ GT", "Seeds tied"],
                bbox_to_anchor=(1.01, 0.52), loc="upper left",
                frameon=True, edgecolor="#cccccc", title="Ground Truth", framealpha=0.95)
    ax.add_artist(leg2)

    # opacity note as figure text below the legends
    ax.text(1.02, 0.10,
        "Bar Opacity:\nFull = GT Match\nFaded = Mismatch or Tie",
        transform=ax.transAxes,
        fontsize=8, color="#000000",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", linewidth=0.8, alpha=0.95))

    plt.tight_layout(rect=[0, 0, 0.80, 1])
    fig.savefig(png_dir / f"plot_{field}.png", format="png", bbox_inches="tight")
    fig.savefig(svg_dir / f"plot_{field}.svg", format="svg", bbox_inches="tight")
    plt.close()


# ── XLSX style helpers ────────────────────────────────────────────────────────

HDR_DARK   = "1C2833"
WHITE      = "FFFFFF"
THIN       = Side(style="thin", color="CCCCCC")
TIE_SIDE   = Side(style="thin", color=TIE_BORDER)


def _thin():
    return Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

def _bot_med():
    return Border(left=THIN, right=THIN, top=THIN,
                  bottom=Side(style="medium", color="AAAAAA"))

def _fill(h):
    return PatternFill("solid", fgColor=h)

def _font(bold=False, size=9, color="000000"):
    return Font(name="Arial", bold=bold, size=size, color=color)

def _hdr(size=9):
    return Font(name="Arial", bold=True, size=size, color=WHITE)

def _ctr():
    return Alignment(horizontal="center", vertical="center", wrap_text=True)

def _lft():
    return Alignment(horizontal="left", vertical="center", wrap_text=True)

def _tie_border():
    return Border(left=TIE_SIDE, right=TIE_SIDE, top=TIE_SIDE, bottom=TIE_SIDE)


MODEL_FILL_HEX = {"oss": "D6E4F0", "qwen": "F5D5E5", "llama": "D1F2EB"}
MODEL_HDR_HEX  = {mk: MODEL_COLORS[mk].lstrip("#") for mk in MODEL_COLORS}


# ── Sheet 1: Summary ──────────────────────────────────────────────────────────

def build_summary_sheet(ws, all_fields, all_pdfs, all_stats, truth):
    ws.sheet_view.showGridLines = False
    n_reports = len(all_pdfs)

    def _sec_hdr(ws, row, title, ncol):
        ws.row_dimensions[row].height = 18
        c = ws.cell(row, 1, title)
        c.font = Font(name="Arial", bold=True, size=9, color=WHITE)
        c.fill = _fill("2C3E50")
        c.alignment = _lft()
        c.border = _thin()
        for ci in range(2, ncol + 1):
            cc = ws.cell(row, ci, "")
            cc.fill = _fill("2C3E50")
            cc.border = _thin()
        ws.merge_cells(start_row=row, start_column=1,
                       end_row=row, end_column=ncol)
        return row + 1

    def _col_hdr_row(ws, row, headers, widths):
        ws.row_dimensions[row].height = 32
        for ci, (h, w) in enumerate(zip(headers, widths), 1):
            c = ws.cell(row, ci, h)
            c.font = _hdr(); c.fill = _fill(HDR_DARK)
            c.alignment = _ctr(); c.border = _thin()
            ws.column_dimensions[get_column_letter(ci)].width = w
        return row + 1

    # ── Pre-compute per (model, field, pdf) outcomes ──────────────────────────
    # outcome: "match" | "mismatch" | "tie"
    def outcome(mk, field, pdf):
        s = all_stats[mk][field][pdf]
        if s["is_tie"]:
            return "tie"
        gt_val = get_truth_row(truth, pdf).get(field)
        return gt_match(s["maj"], gt_val)

    row = 1

    # ── Title ─────────────────────────────────────────────────────────────────
    ncol_max = 2 + n_reports  # model | reports
    ws.row_dimensions[row].height = 22
    t = ws.cell(row, 1, "Extraction Summary — Model Performance vs Ground Truth")
    t.font = _hdr(12); t.fill = _fill(HDR_DARK); t.alignment = _ctr()
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ncol_max)
    row += 1

    # ── Section 1: Model Scorecard ────────────────────────────────────────────
    row = _sec_hdr(ws, row, "1. Model scorecard  (✓ match  ✕ mismatch  ? tied)", ncol_max)
    hdrs = ["Model"] + list(all_pdfs) + ["Overall"]
    wids = [16] + [max(12, len(p) + 2) for p in all_pdfs] + [18]
    row = _col_hdr_row(ws, row, hdrs, wids)

    for model_label, mk in MODELS:
        ws.row_dimensions[row].height = 20
        c = ws.cell(row, 1, model_label)
        c.font = Font(name="Arial", bold=True, size=8, color=WHITE)
        c.fill = _fill(MODEL_HDR_HEX[mk]); c.alignment = _ctr(); c.border = _thin()

        total = {"match": 0, "mismatch": 0, "tie": 0}
        for pi, pdf in enumerate(all_pdfs):
            m = sum(outcome(mk, f, pdf) == "match"    for f in all_fields)
            x = sum(outcome(mk, f, pdf) == "mismatch" for f in all_fields)
            t_ = sum(outcome(mk, f, pdf) == "tie"     for f in all_fields)
            total["match"] += m; total["mismatch"] += x; total["tie"] += t_
            c = ws.cell(row, 2 + pi, f"✓ {m}  ✕ {x}  ? {t_}")
            c.font = _font(size=8); c.alignment = _ctr(); c.border = _thin()
            c.fill = _fill("D5F5E3") if m > x else (_fill(TIE_FILL) if m == x else _fill("FADBD8"))

        c = ws.cell(row, 2 + n_reports,
                    f"✓ {total['match']}  ✕ {total['mismatch']}  ? {total['tie']}")
        c.font = _font(size=8, bold=True); c.alignment = _ctr(); c.border = _thin()
        c.fill = _fill("EBF5FB")
        row += 1

    row += 1  # spacer

    # ── Section 2: % Correct ─────────────────────────────────────────────────
    row = _sec_hdr(ws, row, "2. % correct per model", ncol_max)
    row = _col_hdr_row(ws, row, hdrs, wids)

    for model_label, mk in MODELS:
        ws.row_dimensions[row].height = 18
        c = ws.cell(row, 1, model_label)
        c.font = Font(name="Arial", bold=True, size=8, color=WHITE)
        c.fill = _fill(MODEL_HDR_HEX[mk]); c.alignment = _ctr(); c.border = _thin()

        total_m, total_d = 0, 0
        for pi, pdf in enumerate(all_pdfs):
            m  = sum(outcome(mk, f, pdf) == "match"    for f in all_fields)
            x  = sum(outcome(mk, f, pdf) == "mismatch" for f in all_fields)
            t_ = sum(outcome(mk, f, pdf) == "tie"      for f in all_fields)
            denom = m + x + t_
            pct = round(100 * m / denom) if denom > 0 else 0
            total_m += m; total_d += denom
            c = ws.cell(row, 2 + pi, f"{pct}%")
            c.font = _font(size=8, bold=True); c.alignment = _ctr(); c.border = _thin()
            c.fill = _fill("D5F5E3") if pct >= 80 else (_fill(TIE_FILL) if pct >= 60 else _fill("FADBD8"))

        overall_pct = round(100 * total_m / total_d) if total_d > 0 else 0
        c = ws.cell(row, 2 + n_reports, f"{overall_pct}%")
        c.font = _font(size=8, bold=True); c.alignment = _ctr(); c.border = _thin()
        c.fill = _fill("D5F5E3") if overall_pct >= 80 else (_fill(TIE_FILL) if overall_pct >= 60 else _fill("FADBD8"))
        row += 1

    row += 1  # spacer

    # ── Section 3: Consistently wrong variables ───────────────────────────────
    row = _sec_hdr(ws, row, "3. Consistently wrong variables (wrong or tied in 2+ reports, any model)", ncol_max)
    sec3_hdrs  = ["Variable", "OSS-20B", "Qwen2.5-14B", "Llama3.1-8B", "Wrong in"]
    sec3_wids  = [22, 14, 14, 14, 10]
    row = _col_hdr_row(ws, row, sec3_hdrs, sec3_wids)

    for field in all_fields:
        # per-model outcome strings e.g. "✓✕✕✓"
        model_strings = {}
        model_wrong_counts = {}
        for model_label, mk in MODELS:
            outcomes = [outcome(mk, field, pdf) for pdf in all_pdfs]
            s = ""
            for o in outcomes:
                s += "✓" if o == "match" else ("?" if o == "tie" else "✕")
            model_strings[mk] = s
            model_wrong_counts[mk] = sum(1 for o in outcomes if o in ("mismatch", "tie"))

        max_wrong = max(model_wrong_counts.values())
        if max_wrong < 2:
            continue  # skip — not consistently wrong

        ws.row_dimensions[row].height = 18
        # Variable
        c = ws.cell(row, 1, field)
        c.font = _font(size=8, bold=True); c.alignment = _lft(); c.border = _thin()
        c.fill = _fill("EBF5FB")

        # Per-model outcome string
        for ci, (_, mk) in enumerate(MODELS, 2):
            s_str = model_strings[mk]
            c = ws.cell(row, ci, s_str)
            c.font = _font(size=8); c.alignment = _ctr(); c.border = _thin()
            has_wrong = "✕" in s_str or "?" in s_str
            c.fill = _fill("FADBD8") if model_wrong_counts[mk] >= 2 else _fill("D5F5E3")

        # Wrong in N reports
        c = ws.cell(row, 5, f"{max_wrong} / {n_reports}")
        c.font = _font(size=8, bold=True); c.alignment = _ctr(); c.border = _thin()
        c.fill = _fill("FADBD8") if max_wrong == n_reports else _fill(TIE_FILL)

        row += 1

    row += 1  # spacer

    # ── Section 4: Cross-report consistency ──────────────────────────────────
    row = _sec_hdr(ws, row, "4. Cross-report consistency per model", ncol_max)
    row = _col_hdr_row(ws, row,
        ["Model", "Always correct", "Always wrong", "Inconsistent"],
        [16, 16, 16, 16])

    for model_label, mk in MODELS:
        always_right  = 0
        always_wrong  = 0
        inconsistent  = 0

        for field in all_fields:
            outcomes = [outcome(mk, field, pdf) for pdf in all_pdfs]
            if all(o == "match" for o in outcomes):
                always_right += 1
            elif all(o != "match" for o in outcomes):
                always_wrong += 1
            else:
                inconsistent += 1

        ws.row_dimensions[row].height = 40
        c = ws.cell(row, 1, model_label)
        c.font = Font(name="Arial", bold=True, size=8, color=WHITE)
        c.fill = _fill(MODEL_HDR_HEX[mk]); c.alignment = _ctr(); c.border = _thin()

        c = ws.cell(row, 2, always_right)
        c.font = _font(size=8, bold=True); c.alignment = _ctr(); c.border = _thin()
        c.fill = _fill("D5F5E3")

        c = ws.cell(row, 3, always_wrong)
        c.font = _font(size=8, bold=True); c.alignment = _ctr(); c.border = _thin()
        c.fill = _fill("FADBD8")

        c = ws.cell(row, 4, inconsistent)
        c.font = _font(size=8, bold=True); c.alignment = _ctr(); c.border = _thin()
        c.fill = _fill(TIE_FILL)

        row += 1

    row += 1  # spacer

    # ── Section 5: Best model per report ─────────────────────────────────────
    row = _sec_hdr(ws, row, "5. Best model per report", ncol_max)
    row = _col_hdr_row(ws, row, ["Report", "Best model", "% correct", "Runner-up", "% correct"], [14, 16, 12, 16, 12])

    for pdf in all_pdfs:
        pcts = {}
        for model_label, mk in MODELS:
            m = sum(outcome(mk, f, pdf) == "match"    for f in all_fields)
            x = sum(outcome(mk, f, pdf) == "mismatch" for f in all_fields)
            t_ = sum(outcome(mk, f, pdf) == "tie"     for f in all_fields)
            denom = m + x + t_
            pcts[mk] = (model_label, round(100 * m / denom) if denom > 0 else 0)

        ranked = sorted(pcts.items(), key=lambda kv: kv[1][1], reverse=True)
        best_mk,  (best_label,  best_pct)   = ranked[0]
        runup_mk, (runup_label, runup_pct)  = ranked[1]

        ws.row_dimensions[row].height = 18
        c = ws.cell(row, 1, pdf)
        c.font = _font(size=8); c.alignment = _ctr(); c.border = _thin(); c.fill = _fill("EBF5FB")

        c = ws.cell(row, 2, best_label)
        c.font = Font(name="Arial", bold=True, size=8, color=WHITE)
        c.fill = _fill(MODEL_HDR_HEX[best_mk]); c.alignment = _ctr(); c.border = _thin()

        c = ws.cell(row, 3, f"{best_pct}%")
        c.font = _font(size=8, bold=True); c.alignment = _ctr(); c.border = _thin()
        c.fill = _fill("D5F5E3")

        c = ws.cell(row, 4, runup_label)
        c.font = Font(name="Arial", bold=True, size=8, color=WHITE)
        c.fill = _fill(MODEL_HDR_HEX[runup_mk]); c.alignment = _ctr(); c.border = _thin()

        c = ws.cell(row, 5, f"{runup_pct}%")
        c.font = _font(size=8); c.alignment = _ctr(); c.border = _thin()
        c.fill = _fill("EBF5FB")
        row += 1

    ws.column_dimensions["C"].width = 16
    ws.column_dimensions["E"].width = 16
    ws.freeze_panes = "B3"


# ── Sheet 2: Details ──────────────────────────────────────────────────────────

def build_details_sheet(ws, all_fields, all_pdfs, all_stats, truth):
    """
    Flat table — one row per (variable × report × model).
    Strict superset of every CSV the old script produced:
      old: N_vars × N_reports files, each with 3 model rows + seed cols
      new: one table, auto-filter enabled, sortable/filterable in Excel.
    At 199 vars × N_reports × 3 models this stays fully navigable;
    199 separate sheets would not.
    """
    ws.sheet_view.showGridLines = False

    headers = ["Variable", "Report", "Model",
               "Seed 0", "Seed 1", "Seed 2", "Seed 3", "Seed 4",
               "Mean (%)", "Majority", "GT Value", "GT Match"]
    col_widths = [24, 22, 14, 8, 8, 8, 8, 8, 9, 14, 12, 10]

    ws.row_dimensions[1].height = 22
    t = ws.cell(1, 1, "Seed-Level Detail — All Variables × Reports × Models")
    t.font = _hdr(12); t.fill = _fill(HDR_DARK); t.alignment = _ctr()
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))

    ws.row_dimensions[2].height = 38
    for ci, (hdr, w) in enumerate(zip(headers, col_widths), 1):
        c = ws.cell(2, ci, hdr)
        c.font = _hdr(); c.fill = _fill(HDR_DARK); c.alignment = _ctr(); c.border = _thin()
        ws.column_dimensions[get_column_letter(ci)].width = w

    row = 3
    for field in all_fields:
        for pdf in all_pdfs:
            truth_row = get_truth_row(truth, pdf)
            gt_val    = truth_row.get(field)
            gt_str   = gt_val if gt_val is not None else "—"

            for model_label, mk in MODELS:
                s         = all_stats[mk][field][pdf]
                is_tie   = s["is_tie"]
                tied_str = s["tied_str"]
                match    = gt_match(s["maj"], gt_val)
                maj_disp = tied_str if is_tie else (s["maj"] if s["maj"] is not None else "—")
                match_str = "?" if is_tie else ("✓" if match == "match" else "✕") #✗
                seed_vals = [v if v is not None else "—" for v in s["vals"]]

                row_data = ([field, pdf, model_label]
                            + seed_vals
                            + [round(s["mean"], 1),
                               maj_disp, gt_str, match_str])

                mfill = MODEL_FILL_HEX.get(mk, "FFFFFF")
                for ci, val in enumerate(row_data, 1):
                    c = ws.cell(row, ci, val)
                    c.alignment = _ctr(); c.border = _thin()
                    c.font = _font(size=8.5)
                    if ci == 2:  # Report column
                        c.data_type = TYPE_STRING
                        c.number_format = "@"
                    if ci == len(headers):      # GT Match
                        if is_tie:
                            c.fill = _fill(TIE_FILL)
                            c.font = _font(size=8.5, bold=True, color=TIE_COLOR.lstrip("#"))
                        elif match == "match":
                            c.fill = _fill("D5F5E3")
                            c.font = _font(size=8.5, bold=True, color="145A32")
                        else:
                            c.fill = _fill("FADBD8")
                            c.font = _font(size=8.5, bold=True, color="922B21")
                    elif ci in (1, 2):
                        c.fill = _fill("EBF5FB")
                    elif ci == 10:              # Majority
                        if is_tie:
                            c.fill = _fill(TIE_FILL)
                            c.font = _font(size=8.5, color=TIE_COLOR.lstrip("#"))
                        else:
                            c.fill = _fill(mfill)
                    else:
                        c.fill = _fill(mfill)
                   

                ws.row_dimensions[row].height = 16
                row += 1

    ws.freeze_panes = "D3"
    ws.auto_filter.ref = f"A2:{get_column_letter(len(headers))}{row - 1}"


# ── Sheet 3: Legend ───────────────────────────────────────────────────────────

def build_legend_sheet(ws):
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 60

    ws.row_dimensions[1].height = 22
    t = ws.cell(1, 1, "Legend & Notes")
    t.font = _hdr(12); t.fill = _fill(HDR_DARK); t.alignment = _ctr()
    ws.merge_cells("A1:B1")

    sections = [
        ("SUMMARY SHEET", None),
        ("✓ N  ✕ N  ? N",          "Match / mismatch / tie counts per model per report (Section 1)"),
        ("% correct",              "Matches ÷ Total Variables per report (Section 2)"),
        ("Green ≥ 80%",            "High performance — model reliable on this report"),
        ("Amber 60–79%",           "Moderate performance — some failures"),
        ("Red < 60%",              "Poor performance — systematic issues likely"),
        ("CELL COLOURS", None),
        ("Green cell (✓ Match)",  "Model's majority value matches the ground truth"),
        ("Red cell (✕ Mismatch)", "Model's majority value differs from the ground truth"),
        ("Amber cell (? Tied)",      "GT match indeterminate — majority was a tie"),
        ("MODEL COLOURS", None),
        ("OSS-20B — Dark Blue #12436D",    "GPT-OSS 20B model"),
        ("Qwen2.5-14B — Dark Pink #801650","Qwen 2.5 14B Instruct model"),
        ("Llama3.1-8B — Turquoise #28A197","LLaMA 3.1 8B Instruct model"),
        ("METRICS", None),
        ("Mean (%)",  "Average per-seed agreement with the majority vote (5 seeds)"),
        ("Majority",  "Most common extracted value across the 5 seeds"),
        ("GT Value",  "Ground truth value"),
        ("GT Match",  "Whether majority value equals GT (✓ / ✕ / ?)"),
        ("PLOTS", None),
        ("Bar height",        "Mean agreement (%) — same as Mean (%) in Details sheet"),
        ("✓ above bar",       "Majority value matches ground truth"),
        ("✕ above bar",       "Majority value does not match ground truth"),
        ("~ above bar",       "Seeds tied — no single majority; v = shows all tied values"),
        ("Green band 80–100", "High-agreement zone for visual reference"),
        ("Bar opacity",       "Full opacity = GT match; reduced opacity = GT mismatch or tied majority"),
    ]

    MODEL_LEGEND_FILLS = {
        "OSS-20B":     MODEL_HDR_HEX["oss"],
        "Qwen2.5-14B": MODEL_HDR_HEX["qwen"],
        "Llama3.1-8B": MODEL_HDR_HEX["llama"],
    }

    for ri, (key, val) in enumerate(sections, 2):
        ws.row_dimensions[ri].height = 18
        is_section = val is None
        kc = ws.cell(ri, 1, key)
        kc.alignment = _lft(); kc.border = _thin()
        if is_section:
            kc.font = Font(name="Arial", bold=True, size=9, color=WHITE)
            kc.fill = _fill("2C3E50")
            vc = ws.cell(ri, 2, "")
            vc.fill = _fill("2C3E50"); vc.border = _thin()
        elif key in MODEL_LEGEND_FILLS:
            kc.font = Font(name="Arial", bold=True, size=9, color=WHITE)
            kc.fill = _fill(MODEL_LEGEND_FILLS[key])
            vc = ws.cell(ri, 2, val)
            vc.font = _font(size=9); vc.alignment = _lft()
            vc.fill = _fill("FDFEFE"); vc.border = _thin()
        else:
            kc.font = _font(size=9)
            kc.fill = _fill("EBF5FB")
            vc = ws.cell(ri, 2, val)
            vc.font = _font(size=9); vc.alignment = _lft()
            vc.fill = _fill("FDFEFE"); vc.border = _thin()


def build_xlsx(all_fields, all_pdfs, all_stats, truth, out_path):
    wb = openpyxl.Workbook()

    ws_sum = wb.active
    ws_sum.title = "Summary"
    build_summary_sheet(ws_sum, all_fields, all_pdfs, all_stats, truth)

    ws_det = wb.create_sheet("Details")
    build_details_sheet(ws_det, all_fields, all_pdfs, all_stats, truth)

    ws_leg = wb.create_sheet("Legend")
    build_legend_sheet(ws_leg)

    wb.save(out_path)
    print(f"  Saved XLSX → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_data = {
        "oss":   load_model_data(OSS_DIR),
        "qwen":  load_model_data(QWEN_DIR),
        "llama": load_model_data(LLAMA_DIR),
    }
    truth = load_ground_truth(TRUTH_FILE)

    png_dir = RESULTS_DIR / "plots_png"
    svg_dir = RESULTS_DIR / "plots_svg"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    all_pdfs = sorted(set(pdf for md in all_data.values() for pdf in md))
    all_fields = sorted(set(
        field
        for md in all_data.values()
        for pd_ in md.values()
        for sd in pd_.values()
        for field in sd
        if field not in SKIP
    ))

    print(f"PDFs: {len(all_pdfs)}  |  Fields: {len(all_fields)}")

    # Pre-compute stats once — reused by both plots and XLSX
    all_stats = {mk: {field: {} for field in all_fields} for _, mk in MODELS}

    for _, mk in MODELS:
        for field in all_fields:
            for pdf in all_pdfs:
                pdf_data = all_data[mk].get(pdf, {})
                vals = [pdf_data.get(seed, {}).get(field) for seed in SEEDS]
                per_seed, mean_acc, std_acc, maj, is_tie, tied_str = compute_stats(vals)
                all_stats[mk][field][pdf] = {
                    "vals": vals, "per_seed": per_seed,
                    "mean": mean_acc, "std": std_acc, "maj": maj,
                    "is_tie": is_tie, "tied_str": tied_str,
                }

    # Plots
    for fi, field in enumerate(all_fields):
        print(f"  [{fi+1:>3}/{len(all_fields)}] {field}")
        make_plot(
            field, all_pdfs,
            {mk: all_stats[mk][field] for _, mk in MODELS},
            truth, png_dir, svg_dir
        )

    # Unified XLSX
    xlsx_path = RESULTS_DIR / "model_comparison_gt.xlsx"
    print("\nBuilding XLSX ...")
    build_xlsx(all_fields, all_pdfs, all_stats, truth, xlsx_path)

    n_detail_rows = len(all_fields) * len(all_pdfs) * len(MODELS)
    print(f"""
Done.
  {png_dir}/   — {len(all_fields)} PNGs (600 DPI)
  {svg_dir}/   — {len(all_fields)} SVGs (vector)
  {xlsx_path}
    Summary         : scorecard + % correct + problem variables + best model per report
    Variable Detail : {len(all_fields)} variables × {len(all_pdfs)} reports × 3 models
    Details         : {n_detail_rows} rows  (all 5 seed values per row)
    Legend          : colour key + metric definitions
""")


if __name__ == "__main__":
    main()