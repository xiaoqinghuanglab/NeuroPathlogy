"""compare_models_gt.py — per-variable model comparison against ground truth.

Output (results_gt_161/):
  overall_accuracy_by_model.png/svg  - 3-bar accuracy summary
  model_comparison_gt.xlsx
    Summary  - scorecard, % correct, worst variables, cross-report consistency, best model/report
    Details  - one row per (variable, report, model): seeds, mean, majority, GT, match
    Legend   - colour key + definitions

Scoring is restricted to variables that are (a) in the 199-var NACC NP list
(rdd-np.csv), (b) actually extracted, (c) have a column in the GT file, and
(d) not in UNSCORABLE_VARIABLES (NACCINT, NPFORMVER - verified absent from
the source report text, so scoring them would penalize the model for not
knowing something the report never states). Fields failing (c) or (d) still
show in Details (real seed data) but GT Match is a greyed "-" - nothing to
score against.

NPWBRWT and NPPMIH use a small numeric tolerance (±1 gram / ±0.3 hours)
instead of exact match - reports state brain weight to 1 decimal but the
NACC field is integer-only, so small gaps there are rounding, not
extraction error (see NUMERIC_TOLERANCES).

-4/-4.4 = "this report's NP Form version never collected this variable" -
NOT proof the report itself has no info (models extract real, consistent
values here, e.g. NPPMIH hours). Excluded from scoring entirely for all
199 vars rather than guessed at (see KNOWN_SENTINEL_VALUES / gt_match).

For cells with real GT: variable-specific "No"/absent and
missing/unknown/not-assessed codes are treated as equivalent where verified
in ABSENCE_EQUIVALENTS. Nullable primary/contributing diagnosis fields are
handled separately: null means the role is indeterminate from the report and
is therefore NOT equivalent to explicit code 2 (No).
Verified per variable, not assumed (e.g. NACCBRNN's
absent code is 1, not 0 - its 0 means "pathology present").

Colours: UK Gov Analysis Function palette.
  OSS-20B #12436D | Qwen2.5-14B #801650 | Llama3.1-8B #28A197
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
from collections import defaultdict
from pathlib import Path

_ROOT = Path("/N/project/ADRD/neuropathoroot/")
OSS_DIR        = _ROOT / "output" / "oss-20b"
QWEN_DIR       = _ROOT / "output" / "qwen2.5-14b"
LLAMA_DIR      = _ROOT / "output" / "llama3.1-8b"
TRUTH_FILE     = _ROOT / "csv_input" / "nacc_np_ground_truth.csv"
VALID_VARS_FILE = _ROOT / "csv_input" / "rdd-np.csv"   # canonical 199-variable NACC NP form list
RESULTS_DIR    = _ROOT / "results_gt_161"

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

TIE_COLOR         = "#B7770D"  # amber text
TIE_FILL          = "FEF3E2"   # amber cell fill

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


KNOWN_SENTINEL_VALUES = {-4, -4.4}
# "Form version didn't collect this variable" - not proof the report text has
# no info (models extract real, consistent values on -4.4 cells). Excluded
# from scoring entirely rather than guessed at. See gt_match.

UNSCORABLE_VARIABLES = {
    # Not present in the source autopsy report text at all - verified against
    # the actual sample reports in the project (4/4 reports checked, zero
    # mentions of either concept). Scoring these would penalize the model for
    # not knowing something the report never states, not for a real
    # extraction failure. Still shown in Details (real seed data), but
    # treated like a missing GT column - excluded from Summary/plots/accuracy.
    "NACCINT",    # time since last clinical visit - requires UDS visit dates, not autopsy content
    "NPFORMVER",  # NP Form version number - database metadata, not report content
}

NUMERIC_TOLERANCES = {
    # Small numeric differences here are expected measurement/rounding noise,
    # not extraction errors - verified against actual report text.
    "NPWBRWT": 1,    # brain weight (grams): reports state 1-decimal precision
                     # (e.g. "1,005.8 grams") but the NACC field is integer-only,
                     # so a correct read will differ from GT by ordinary rounding.
    "NPPMIH":  0.3,  # postmortem interval (hours): small tolerance for rounding
                     # when converting minutes to decimal hours.
}


def gt_match(maj, gt_val, field=None, absence_equiv=None):
    """absence_equiv: {variable: codes meaning "no answer" for that var}.
    Sentinel (-4/-4.4) is always excluded. For nullable primary/contributing
    diagnosis fields, null is semantically distinct from explicit code 2:
    null means the role cannot be determined from the report, whereas 2 means
    the report establishes that the diagnosis is not in that role.
    For other variables, variable-specific absence-equivalence rules remain
    in effect.
    """
    gt_norm  = normalize(gt_val)
    maj_norm = normalize(maj)

    if gt_norm in KNOWN_SENTINEL_VALUES:
        return "excluded"

    equiv_set = absence_equiv.get(field, set()) if absence_equiv else set()

    # For nullable primary/contributing diagnosis fields, null has a
    # distinct meaning: the report did not provide enough information
    # to determine the role. Do not treat null as equivalent to explicit
    # NACC code 2 (No).
    NULLABLE_ROLE_FIELDS = {
        "NPPAD", "NPCAD",
        "NPPLEWY", "NPCLEWY",
        "NPPVASC", "NPCVASC",
        "NPPFTLD", "NPCFTLD",
        "NPPNORM", "NPCNORM",
        "NPPADP", "NPCADP",
        "NPPHIPP", "NPCHIPP",
        "NPPPRION", "NPCPRION",
        "NPPOTH1", "NPCOTH1",
        "NPPOTH2", "NPCOTH2",
        "NPPOTH3", "NPCOTH3",
    }

    if field in NULLABLE_ROLE_FIELDS:
        # Exact semantic comparison for nullable role fields:
        # 1 = explicitly Yes, 2 = explicitly No, null = indeterminate.
        if gt_norm is None or maj_norm is None:
            return "match" if gt_norm is None and maj_norm is None else "mismatch"

        if field in NUMERIC_TOLERANCES and isinstance(gt_norm, (int, float)) and isinstance(maj_norm, (int, float)):
            if abs(maj_norm - gt_norm) <= NUMERIC_TOLERANCES[field]:
                return "match"

        return "match" if maj_norm == gt_norm else "mismatch"

    def is_absence(v):
        return v is None or v in equiv_set

    if is_absence(gt_norm) and is_absence(maj_norm):
        return "match"

    if gt_norm is None or maj_norm is None:
        return "mismatch"

    if field in NUMERIC_TOLERANCES and isinstance(gt_norm, (int, float)) and isinstance(maj_norm, (int, float)):
        if abs(maj_norm - gt_norm) <= NUMERIC_TOLERANCES[field]:
            return "match"

    return "match" if maj_norm == gt_norm else "mismatch"


ABSENCE_EQUIVALENTS = {
    # variable: {codes treated as non-informative / absence-equivalent}.
    # Verified per variable, not assumed to be 0.
    #
    # For nullable primary/contributing diagnosis fields, code 2 is NOT listed
    # here because null and explicit No (2) now have distinct meanings.
    # gt_match() handles those nullable role fields separately.
    # Verified against rdd-np.csv and the source PDF. Two false positives
    # excluded (code text containing "no"/"not assessed" as a qualifier on a
    # positive finding, not the actual answer): NPHIPSCL code 3 ("Present but
    # laterality not assessed") and NPALSMND code 5 ("Yes, with no specific
    # inclusions"). Two rdd-np.csv transcription errors also corrected here
    # using the PDF value: NACCINT (999->9999), NACCBRNN (8->9).
    "NACCAMY": {0, 8, 9},
    "NACCARTE": {0, 8, 9},
    "NACCAVAS": {0, 8, 9},
    "NACCBNKF": {0, 9},
    "NACCBRAA": {0, 8, 9},
    "NACCBRNN": {1, 9},
    "NACCCBD": {0, 8, 9},
    "NACCCSFP": {0, 9},
    "NACCDAGE": {999},
    "NACCDIFF": {0, 8, 9},
    "NACCDOWN": {7},
    "NACCFORM": {0, 9},
    "NACCHEM": {0, 8, 9},
    "NACCINF": {0, 8, 9},
    "NACCINT": {9999},
    "NACCLEWY": {0, 8, 9},
    "NACCMICR": {0, 8, 9},
    "NACCMOD": {99},
    "NACCNEC": {0, 8, 9},
    "NACCNEUR": {0, 8, 9},
    "NACCOTHP": {0, 8, 9},
    "NACCPARA": {0, 9},
    "NACCPICK": {0, 8, 9},
    "NACCPRIO": {0, 8, 9},
    "NACCPROG": {0, 8, 9},
    "NACCVASC": {0, 9},
    "NACCYOD": {9999},
    "NPABAN": {8},
    "NPADNC": {8, 9},
    "NPADRDA": {9},
    "NPALSMND": {0, 8, 9},
    "NPART": {2, 3, 9},
    "NPASAN": {8},
    "NPBNKB": {0, 9},
    "NPBNKF": {0, 9},
    "NPCERAD": {9},
    "NPCHROM": {13, 50, 99},
    "NPFAUT": {0, 9},
    "NPFRONT": {2, 3, 9},
    "NPFTD": {3, 4, 9},
    "NPFTDNO": {2, 3, 9},
    "NPFTDSPC": {2, 3, 9},
    "NPFTDT10": {0, 8, 9},
    "NPFTDT2": {0, 8, 9},
    "NPFTDT5": {0, 8, 9},
    "NPFTDT6": {0, 8, 9},
    "NPFTDT7": {0, 8, 9},
    "NPFTDT8": {0, 8, 9},
    "NPFTDT9": {0, 8, 9},
    "NPFTDTAU": {0, 8, 9},
    "NPFTDTDP": {0, 8, 9},
    "NPGENE": {3, 9},
    "NPGRCCA": {0, 8, 9},
    "NPGRHA": {0, 8, 9},
    "NPGRLA": {0, 8, 9},
    "NPGRLCH": {0, 8, 9},
    "NPGRSNH": {0, 8, 9},
    "NPHEM": {2, 3, 9},
    "NPHEMO": {0, 8, 9},
    "NPHEMO1": {0, 8, 9},
    "NPHEMO2": {0, 8, 9},
    "NPHEMO3": {0, 8, 9},
    "NPHIPSCL": {0, 8, 9},
    "NPHISG": {0},
    "NPHISMB": {0},
    "NPHISO": {0},
    "NPHISSS": {0},
    "NPHIST": {0},
    "NPINF": {0, 8, 9},
    "NPINF1A": {88, 99},
    "NPINF1B": {88.8, 99.9},
    "NPINF1D": {88.8, 99.9},
    "NPINF1F": {88.8, 99.9},
    "NPINF2A": {88, 99},
    "NPINF2B": {88.8, 99.9},
    "NPINF2D": {88.8, 99.9},
    "NPINF2F": {88.8, 99.9},
    "NPINF3A": {88, 99},
    "NPINF3B": {88.8, 99.9},
    "NPINF3D": {88.8, 99.9},
    "NPINF3F": {88.8, 99.9},
    "NPINF4A": {88, 99},
    "NPINF4B": {88.8, 99.9},
    "NPINF4D": {88.8, 99.9},
    "NPINF4F": {88.8, 99.9},
    "NPLAC": {2, 3, 9},
    "NPLBOD": {0, 8, 9},
    "NPLEWYCS": {9},
    "NPLINF": {2, 3, 9},
    "NPMICRO": {2, 3, 9},
    "NPNIT": {9},
    "NPNLOSS": {0, 8, 9},
    "NPOANG": {2, 3, 9},
    "NPOCRIT": {9},
    "NPOFTD": {0, 8, 9},
    "NPOFTD1": {0, 8, 9},
    "NPOFTD2": {0, 8, 9},
    "NPOFTD3": {0, 8, 9},
    "NPOFTD4": {0, 8, 9},
    "NPOFTD5": {0, 8, 9},
    "NPOLD": {0, 8, 9},
    "NPOLD1": {8, 9},
    "NPOLD2": {8, 9},
    "NPOLD3": {8, 9},
    "NPOLD4": {8, 9},
    "NPOLDD": {0, 8, 9},
    "NPOLDD1": {8, 9},
    "NPOLDD2": {8, 9},
    "NPOLDD3": {8, 9},
    "NPOLDD4": {8, 9},
    "NPPATH": {0, 8, 9},
    "NPPATH10": {0, 8, 9},
    "NPPATH11": {0, 8, 9},
    "NPPATH2": {0, 8, 9},
    "NPPATH3": {0, 8, 9},
    "NPPATH4": {0, 8, 9},
    "NPPATH5": {0, 8, 9},
    "NPPATH6": {0, 8, 9},
    "NPPATH7": {0, 8, 9},
    "NPPATH8": {0, 8, 9},
    "NPPATH9": {0, 8, 9},
    "NPPATHO": {0},
    "NPPDXA": {0, 8, 9},
    "NPPDXB": {0, 8, 9},
    "NPPDXD": {0, 8, 9},
    "NPPDXE": {0, 8, 9},
    "NPPDXF": {0, 8, 9},
    "NPPDXG": {0, 8, 9},
    "NPPDXH": {0, 8, 9},
    "NPPDXI": {0, 8, 9},
    "NPPDXJ": {0, 8, 9},
    "NPPDXK": {0, 8, 9},
    "NPPDXL": {0, 8, 9},
    "NPPDXM": {0, 8, 9},
    "NPPDXN": {0, 8, 9},
    "NPPDXP": {0, 8, 9},
    "NPPDXQ": {0, 8, 9},
    "NPPMIH": {99.9},
    "NPPRNP": {9},
    "NPSCL": {2, 3, 9},
    "NPTAN": {8},
    "NPTAU": {2, 3, 9},
    "NPTAUHAP": {9},
    "NPTDPA": {0, 8, 9},
    "NPTDPAN": {8},
    "NPTDPB": {0, 8, 9},
    "NPTDPC": {0, 8, 9},
    "NPTDPD": {0, 8, 9},
    "NPTDPE": {0, 8, 9},
    "NPTHAL": {8, 9},
    "NPVOTH": {2, 3, 9},
    "NPWBRWT": {9999},
    "NPWMR": {0, 8, 9},
}


def classify_outcome(s, gt_val, field=None, absence_equiv=None):
    """Single source of truth for classifying a (model, field, report) cell,
    used everywhere instead of each call site re-deriving it separately.
    Sentinel GT is checked FIRST, before tie status: if ground truth is
    unusable, it doesn't matter whether the model's 5 seeds agreed with each
    other - the cell is unscorable either way. Checking tie first (as an
    earlier version of this script did in 3 separate places) meant a tied
    cell landing on a sentinel-GT report was wrongly counted as a scored
    "tie" (penalized) instead of excluded.
    Returns "excluded" | "tie" | "match" | "mismatch".
    """
    if normalize(gt_val) in KNOWN_SENTINEL_VALUES:
        return "excluded"
    if s["is_tie"]:
        return "tie"
    return gt_match(s["maj"], gt_val, field, absence_equiv)


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
    dupes = []
    with open(truth_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fields = list(row.keys())
            report = row[fields[0]].replace(".pdf", "").replace(" copy", "").strip()
            if report in truth:
                dupes.append(report)
            truth[report] = {}
            for field in fields[1:]:
                val = row[field].strip()
                if val == "" or val.lower() == "none":
                    truth[report][field] = None
                else:
                    try:
                        num = round(float(val), 1)
                        truth[report][field] = int(num) if num == int(num) else num
                    except ValueError:
                        truth[report][field] = val
    if dupes:
        print(f"WARNING: {len(dupes)} duplicate report IDs in {truth_path} (last row wins, earlier ones silently dropped): {dupes}")
    return truth


def load_valid_vars(path: Path) -> set:
    """Canonical 199 NACC NP form variables."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["VariableName"].strip() for row in reader if row.get("VariableName")}


def load_gt_columns(truth_path: Path) -> set:
    """Column headers present in the GT CSV (minus the report-id column).
    A variable can be in the 199-list and extracted but have no column
    here - that's "no GT at all" for it, not a blank value."""
    with open(truth_path, newline="") as f:
        header = next(csv.reader(f))
    return set(header[1:])


def get_truth_row(truth, pdf):
    for key in (pdf, pdf.replace(" copy", ""), pdf + " copy"):
        if key in truth:
            return truth[key]
    return {}


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_overall_accuracy_plot(acc_stats, out_dir):
    """3 bars, one per model - % correct on cells with real (non-sentinel) GT."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    fig.patch.set_facecolor("white")

    labels  = [label for label, _ in MODELS]
    pcts    = [acc_stats[mk]["pct"] for _, mk in MODELS]
    colors  = [MODEL_COLORS[mk] for _, mk in MODELS]

    bars = ax.bar(labels, pcts, color=colors, width=0.55, edgecolor="white")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 2, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 112)
    ax.set_ylabel("% correct (all scored variables × reports)")
    ax.set_title("Overall Accuracy by Model", fontsize=11, fontweight="bold")
    ax.axhline(100, color="#666666", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "overall_accuracy_by_model.png", format="png", bbox_inches="tight")
    fig.savefig(out_dir / "overall_accuracy_by_model.svg", format="svg", bbox_inches="tight")
    plt.close()


# ── XLSX style helpers ────────────────────────────────────────────────────────

HDR_DARK   = "1C2833"
WHITE      = "FFFFFF"
THIN       = Side(style="thin", color="CCCCCC")


def _thin():
    return Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

def _fill(h):
    return PatternFill("solid", fgColor=h)

def _font(bold=False, size=9, color="000000", italic=False):
    return Font(name="Arial", bold=bold, size=size, color=color, italic=italic)

def _hdr(size=9):
    return Font(name="Arial", bold=True, size=size, color=WHITE)

def _ctr():
    return Alignment(horizontal="center", vertical="center", wrap_text=True)

def _lft():
    return Alignment(horizontal="left", vertical="center", wrap_text=True)


MODEL_FILL_HEX = {"oss": "D6E4F0", "qwen": "F5D5E5", "llama": "D1F2EB"}
MODEL_HDR_HEX  = {mk: MODEL_COLORS[mk].lstrip("#") for mk in MODEL_COLORS}


# ── Sheet 1: Summary ──────────────────────────────────────────────────────────

def build_summary_sheet(ws, all_fields, all_pdfs, all_stats, truth, absence_equiv):
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
    # outcome: "match" | "mismatch" | "tie" | "excluded" | "not_run"
    def outcome(mk, field, pdf):
        s = all_stats[mk][field].get(pdf)
        if s is None:
            return "not_run"
        gt_val = get_truth_row(truth, pdf).get(field)
        return classify_outcome(s, gt_val, field, absence_equiv)

    row = 1

    # ── Title ─────────────────────────────────────────────────────────────────
    ncol_max = 2 + n_reports  # model | reports
    ws.row_dimensions[row].height = 22
    t = ws.cell(row, 1, "Extraction Summary — Model Performance vs Ground Truth")
    t.font = _hdr(12); t.fill = _fill(HDR_DARK); t.alignment = _ctr()
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ncol_max)
    row += 1

    # ── Section 1: Model Scorecard ────────────────────────────────────────────
    row = _sec_hdr(ws, row, "1. Model scorecard  (✓ match  ✕ mismatch  ? tied  ⊘ GT excluded)", ncol_max)
    hdrs = ["Model"] + list(all_pdfs) + ["Overall"]
    wids = [16] + [max(12, len(p) + 2) for p in all_pdfs] + [18]
    row = _col_hdr_row(ws, row, hdrs, wids)

    for model_label, mk in MODELS:
        ws.row_dimensions[row].height = 20
        c = ws.cell(row, 1, model_label)
        c.font = Font(name="Arial", bold=True, size=8, color=WHITE)
        c.fill = _fill(MODEL_HDR_HEX[mk]); c.alignment = _ctr(); c.border = _thin()

        total = {"match": 0, "mismatch": 0, "tie": 0, "excluded": 0}
        for pi, pdf in enumerate(all_pdfs):
            m = sum(outcome(mk, f, pdf) == "match"    for f in all_fields)
            x = sum(outcome(mk, f, pdf) == "mismatch" for f in all_fields)
            t_ = sum(outcome(mk, f, pdf) == "tie"     for f in all_fields)
            e_ = sum(outcome(mk, f, pdf) == "excluded" for f in all_fields)
            total["match"] += m; total["mismatch"] += x; total["tie"] += t_; total["excluded"] += e_
            c = ws.cell(row, 2 + pi, f"✓ {m}  ✕ {x}  ? {t_}  ⊘{e_}")
            c.font = _font(size=8); c.alignment = _ctr(); c.border = _thin()
            c.fill = _fill("D5F5E3") if m > x else (_fill(TIE_FILL) if m == x else _fill("FADBD8"))

        c = ws.cell(row, 2 + n_reports,
                    f"✓ {total['match']}  ✕ {total['mismatch']}  ? {total['tie']}  ⊘{total['excluded']}")
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
    # Counts + % instead of a per-report tick string, which is unreadable
    # once n_reports gets past a handful. Threshold is percentage-based per
    # variable (not a fixed count off the total report count), since a
    # variable's scorable population varies a lot depending on how many of
    # its cells are sentinel-excluded - a fixed count would need the same
    # absolute number of wrong answers regardless of whether that's 10% or
    # 80% of that variable's actual scorable reports.
    MIN_WRONG_PCT = 10
    MIN_WRONG_COUNT = 2  # guard against flagging noise on tiny scorable samples

    row = _sec_hdr(ws, row, f"3. Consistently wrong variables (wrong/tied in ≥{MIN_WRONG_PCT}% of scorable reports, any model, min {MIN_WRONG_COUNT})", ncol_max)
    sec3_hdrs = ["Variable", "OSS-20B", "Qwen2.5-14B", "Llama3.1-8B", "Worst model"]
    sec3_wids = [22, 16, 16, 16, 14]
    row = _col_hdr_row(ws, row, sec3_hdrs, sec3_wids)

    label_by_mk = {mk: lbl for lbl, mk in MODELS}

    field_wrong = []
    for field in all_fields:
        # counts[mk] = (wrong, scorable) - scorable excludes "excluded"/"not_run"
        # cells, so the % below reflects the actual scored population per
        # model, not the total report count (which can overstate the
        # denominator when reports are excluded or never extracted).
        counts = {}
        for _, mk in MODELS:
            outcomes = [outcome(mk, field, pdf) for pdf in all_pdfs]
            scorable = [o for o in outcomes if o not in ("excluded", "not_run")]
            wrong = sum(1 for o in scorable if o in ("mismatch", "tie"))
            counts[mk] = (wrong, len(scorable))
        max_pct = max((100 * w / s if s else 0) for w, s in counts.values())
        max_wrong = max(w for w, _ in counts.values())
        field_wrong.append((field, counts, max_pct, max_wrong))

    field_wrong = [fw for fw in field_wrong if fw[2] >= MIN_WRONG_PCT and fw[3] >= MIN_WRONG_COUNT]
    field_wrong.sort(key=lambda fw: fw[2], reverse=True)  # worst first, by rate not raw count

    for field, counts, max_pct, max_wrong in field_wrong:
        ws.row_dimensions[row].height = 18
        c = ws.cell(row, 1, field)
        c.font = _font(size=8, bold=True); c.alignment = _lft(); c.border = _thin()
        c.fill = _fill("EBF5FB")

        for ci, (_, mk) in enumerate(MODELS, 2):
            wrong_n, scorable_n = counts[mk]
            pct = round(100 * wrong_n / scorable_n) if scorable_n else 0
            c = ws.cell(row, ci, f"{wrong_n}/{scorable_n} ({pct}%)")
            c.font = _font(size=8); c.alignment = _ctr(); c.border = _thin()
            c.fill = _fill("FADBD8") if pct >= 30 else (_fill(TIE_FILL) if pct >= 10 else _fill("D5F5E3"))

        worst_mk = max(counts, key=lambda mk: counts[mk][0])
        c = ws.cell(row, 5, label_by_mk[worst_mk])
        c.font = Font(name="Arial", bold=True, size=8, color=WHITE)
        c.fill = _fill(MODEL_HDR_HEX[worst_mk]); c.alignment = _ctr(); c.border = _thin()

        row += 1

    row += 1  # spacer

    # ── Section 4: Cross-report consistency ──────────────────────────────────
    # Percentage bands, not a strict 100%/0% requirement: with 161 reports,
    # "always correct" would mean literally zero misses ever, which lumps a
    # variable the model gets right 160/161 times into the same bucket as one
    # it gets right half the time. RELIABLE_PCT matches Section 2's own
    # ≥80% "high performance" cutoff, for one consistent standard across the
    # whole sheet rather than two different numbers both meaning "reliable".
    RELIABLE_PCT   = 80
    UNRELIABLE_PCT = 20

    row = _sec_hdr(ws, row, "4. Cross-report consistency per model", ncol_max)
    row = _col_hdr_row(ws, row,
        ["Model", f"Reliable (≥{RELIABLE_PCT}%)", f"Unreliable (≤{UNRELIABLE_PCT}%)", "Inconsistent"],
        [16, 18, 18, 16])

    for model_label, mk in MODELS:
        reliable     = 0
        unreliable   = 0
        inconsistent = 0

        for field in all_fields:
            outcomes = [o for o in (outcome(mk, field, pdf) for pdf in all_pdfs) if o not in ("excluded", "not_run")]
            if not outcomes:
                continue  # every report's GT was a sentinel for this field — nothing to score
            pct = 100 * sum(1 for o in outcomes if o == "match") / len(outcomes)
            if pct >= RELIABLE_PCT:
                reliable += 1
            elif pct <= UNRELIABLE_PCT:
                unreliable += 1
            else:
                inconsistent += 1

        ws.row_dimensions[row].height = 40
        c = ws.cell(row, 1, model_label)
        c.font = Font(name="Arial", bold=True, size=8, color=WHITE)
        c.fill = _fill(MODEL_HDR_HEX[mk]); c.alignment = _ctr(); c.border = _thin()

        c = ws.cell(row, 2, reliable)
        c.font = _font(size=8, bold=True); c.alignment = _ctr(); c.border = _thin()
        c.fill = _fill("D5F5E3")

        c = ws.cell(row, 3, unreliable)
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

def build_details_sheet(ws, all_fields, all_pdfs, all_stats, truth, gt_columns, absence_equiv):
    """One row per (variable, report, model). all_fields is the full
    199 ∩ extracted set, not the narrower scored_fields - vars with no GT
    column still show seed data but GT Value/Match are a greyed "-".
    Sentinel GT (-4/-4.4) shows the real value but also greys out Match,
    since those cells are excluded from scoring."""
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
        has_gt = field in gt_columns and field not in UNSCORABLE_VARIABLES
        for pdf in all_pdfs:
            gt_val = get_truth_row(truth, pdf).get(field) if has_gt else None
            gt_str = gt_val if gt_val is not None else "—"

            for model_label, mk in MODELS:
                s = all_stats[mk][field].get(pdf)
                if s is None:
                    row_data = [field, pdf, model_label] + ["n/a"]*5 + ["n/a", "n/a", gt_str, "n/a"]
                    match = "not_run"
                else:
                    is_tie   = s["is_tie"]
                    tied_str = s["tied_str"]
                    maj_disp = tied_str if is_tie else (s["maj"] if s["maj"] is not None else "—")
                    seed_vals = [v if v is not None else "—" for v in s["vals"]]

                    if not has_gt:
                        match     = "no_gt"
                        match_str = "—"
                    else:
                        match     = classify_outcome(s, gt_val, field, absence_equiv)
                        match_str = "?" if match == "tie" else ("—" if match == "excluded" else ("✓" if match == "match" else "✕"))

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
                    if match == "not_run":
                        c.fill = _fill("D6DBDF")
                        c.font = _font(size=8.5, italic=True, color="707B7C") if ci != 2 else _font(size=8.5, color="707B7C")
                        continue
                    if ci == len(headers):      # GT Match
                        if match in ("no_gt", "excluded"):
                            c.fill = _fill("EAEDED")
                            c.font = _font(size=8.5, color="909497")
                        elif match == "tie":
                            c.fill = _fill(TIE_FILL)
                            c.font = _font(size=8.5, bold=True, color=TIE_COLOR.lstrip("#"))
                        elif match == "match":
                            c.fill = _fill("D5F5E3")
                            c.font = _font(size=8.5, bold=True, color="145A32")
                        else:
                            c.fill = _fill("FADBD8")
                            c.font = _font(size=8.5, bold=True, color="922B21")
                    elif ci == 11:              # GT Value
                        if not has_gt:
                            c.fill = _fill("EAEDED")
                            c.font = _font(size=8.5, color="909497")
                        else:
                            c.fill = _fill(mfill)
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
        ("✓ / ✕ / ? / ⊘",   "Match / mismatch / tie / excluded from scoring"),
        ("% correct",        "Matches ÷ scored cells (excludes ⊘)"),
        ("Green / Amber / Red", "≥80% / 60–79% / <60%"),
        ("Section 3 threshold", "Lists variables wrong/tied in ≥10% of scorable reports, any model (min 2)"),
        ("Section 4 bands",     "Reliable ≥80% correct, Unreliable ≤20% correct, Inconsistent in between (per variable, per model)"),
        ("MODEL COLOURS", None),
        ("OSS-20B — #12436D",     "GPT-OSS 20B"),
        ("Qwen2.5-14B — #801650", "Qwen 2.5 14B Instruct"),
        ("Llama3.1-8B — #28A197", "LLaMA 3.1 8B Instruct"),
        ("DETAILS SHEET METRICS", None),
        ("Mean (%)",  "Seed agreement with the majority (5 seeds)"),
        ("Majority",  "Most common value across the 5 seeds"),
        ("GT Match",  "✓ match / ✕ mismatch / ? tie / — not scored"),
        ("SCORING RULES", None),
        ("199-variable scope",    "Only rdd-np.csv's 199 NACC NP variables, extracted, with a GT column"),
        ("Grey '—'",              "No GT column, or NACCINT/NPFORMVER (not present in report text) — shown but not scored"),
        ("Grey italic 'n/a' row", "Report never extracted for that model — excluded, not counted as null"),
        ("-4 / -4.4 sentinel",    "\"Form version didn't collect this\" — excluded from scoring, not a match or mismatch"),
        ("Absence equivalence",   "Per variable: null, its 'No'/absent code, and its missing/unknown code all count as the same answer (verified per variable — e.g. NACCBRNN's absent code is 1, not 0)"),
        ("NPWBRWT / NPPMIH tolerance", "±1 gram / ±0.3 hours counted as a match (rounding, not error)"),
        ("PLOTS", None),
        ("overall_accuracy_by_model", "3-bar % correct per model, real GT only — matches the console print"),
    ]

    MODEL_LEGEND_FILLS = {
        "OSS-20B — #12436D":     MODEL_HDR_HEX["oss"],
        "Qwen2.5-14B — #801650": MODEL_HDR_HEX["qwen"],
        "Llama3.1-8B — #28A197": MODEL_HDR_HEX["llama"],
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


def compute_model_accuracy_stats(all_fields, all_pdfs, all_stats, truth, absence_equiv):
    """% correct over cells with real (non-sentinel) GT. Ties count as
    scored-but-wrong, same as on the Summary sheet.
    Returns {model_key: {"pct", "n_scored", "n_excluded"}}."""
    out = {}
    for _, mk in MODELS:
        match = scored = excluded = 0
        for field in all_fields:
            for pdf in all_pdfs:
                s = all_stats[mk][field].get(pdf)
                if s is None:
                    continue  # never extracted, not a null answer
                gt_val = get_truth_row(truth, pdf).get(field)
                outcome = classify_outcome(s, gt_val, field, absence_equiv)
                if outcome == "excluded":
                    excluded += 1
                    continue
                scored += 1
                if outcome == "match":
                    match += 1

        out[mk] = {
            "pct": round(100 * match / scored, 1) if scored else 0.0,
            "n_scored": scored, "n_excluded": excluded,
        }
    return out


def compute_null_bias_stats(all_fields, all_pdfs, all_stats, truth, absence_equiv):
    """Splits scored-cell accuracy into null_rate (how often the model says
    nothing when a real value was expected - null-bias, not necessarily
    wrong) vs acc_when_answered (accuracy on the cells it did answer -
    isolates real extraction accuracy from the null-bias effect)."""
    out = {}
    for _, mk in MODELS:
        scored = null_n = answered_n = answered_correct = 0
        for field in all_fields:
            for pdf in all_pdfs:
                s = all_stats[mk][field].get(pdf)
                if s is None:
                    continue  # never extracted, not a null answer
                gt_val = get_truth_row(truth, pdf).get(field)
                outcome = classify_outcome(s, gt_val, field, absence_equiv)
                if outcome in ("tie", "excluded"):
                    continue
                scored += 1
                if s["maj"] is None:
                    null_n += 1
                else:
                    answered_n += 1
                    if outcome == "match":
                        answered_correct += 1

        out[mk] = {
            "scored": scored,
            "null_rate": round(100 * null_n / scored, 1) if scored else 0.0,
            "null_n": null_n,
            "acc_when_answered": round(100 * answered_correct / answered_n, 1) if answered_n else 0.0,
            "answered_n": answered_n,
        }
    return out


def build_xlsx(all_fields, scored_fields, all_pdfs, all_stats, truth, gt_columns, absence_equiv, out_path):
    wb = openpyxl.Workbook()

    ws_sum = wb.active
    ws_sum.title = "Summary"
    build_summary_sheet(ws_sum, scored_fields, all_pdfs, all_stats, truth, absence_equiv)

    ws_det = wb.create_sheet("Details")
    build_details_sheet(ws_det, all_fields, all_pdfs, all_stats, truth, gt_columns, absence_equiv)

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
    valid_vars = load_valid_vars(VALID_VARS_FILE)
    gt_columns = load_gt_columns(TRUTH_FILE)

    # Fail loud on the inputs that can go wrong without raising an exception:
    # a wrong directory path just returns {} (no error), a wrong CSV header
    # just returns an empty set (no error). Either silently produces a
    # "successful" run with a hollow report instead of a crash, so check
    # explicitly rather than only finding out from an empty-looking XLSX.
    for label, mk, out_dir in [("OSS-20B", "oss", OSS_DIR), ("Qwen2.5-14B", "qwen", QWEN_DIR), ("Llama3.1-8B", "llama", LLAMA_DIR)]:
        if not all_data[mk]:
            print(f"WARNING: {label} has ZERO reports loaded from {out_dir} - check the path, or the job hasn't run yet.")
    if len(valid_vars) != 199:
        print(f"WARNING: expected 199 variables from {VALID_VARS_FILE}, got {len(valid_vars)} - check the file/header.")
    if not gt_columns:
        raise RuntimeError(f"No columns found in ground truth file {TRUTH_FILE} - check the path/header.")
    if not any(all_data.values()):
        raise RuntimeError("No reports loaded for any model - check OSS_DIR/QWEN_DIR/LLAMA_DIR.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_pdfs = sorted(set(pdf for md in all_data.values() for pdf in md))

    reports_without_gt = [pdf for pdf in all_pdfs if not get_truth_row(truth, pdf)]
    if reports_without_gt:
        print(f"WARNING: {len(reports_without_gt)} reports have model output but NO matching row in "
              f"ground truth at all (name mismatch, or genuinely absent from GT) - every field for these "
              f"scores as if GT were blank, which can look like a spurious match: {reports_without_gt}")

    extracted_fields = set(
        field
        for md in all_data.values()
        for pd_ in md.values()
        for sd in pd_.values()
        for field in sd
        if field not in SKIP
    )
    all_fields = sorted(extracted_fields & valid_vars)
    scored_fields = sorted((set(all_fields) & gt_columns) - UNSCORABLE_VARIABLES)   # narrower set used for Summary, plots, accuracy stats

    dropped_not_199 = extracted_fields - valid_vars
    if dropped_not_199:
        print(f"Ignoring {len(dropped_not_199)} extracted fields not in the 199-variable NACC form list: {sorted(dropped_not_199)}")

    no_gt_fields = set(all_fields) - gt_columns
    if no_gt_fields:
        print(f"{len(no_gt_fields)} fields extracted and in the 199-var list but not in the GT file "
              f"(shown in Details with seed data, greyed GT, excluded from scoring): {sorted(no_gt_fields)}")

    unscorable_present = set(all_fields) & UNSCORABLE_VARIABLES
    if unscorable_present:
        print(f"{len(unscorable_present)} fields excluded from scoring - not present in the source report "
              f"text (verified against sample reports), not an extraction failure: {sorted(unscorable_present)}")

    print(f"PDFs: {len(all_pdfs)}  |  Fields in Details sheet: {len(all_fields)}  |  Fields scored: {len(scored_fields)}")
    print("GT of -4/-4.4 ('form version did not collect this') is excluded from scoring for all variables "
          "- see KNOWN_SENTINEL_VALUES.")

    # A report can be missing entirely from one model's output (SLURM chunk
    # never ran, job failed) while present for the others. Skip that
    # (model, report) rather than treat an empty pdf_data as maj=None -
    # that would look like a confident null and skew null_rate/accuracy.
    all_stats = {mk: {field: {} for field in all_fields} for _, mk in MODELS}
    missing_reports = {mk: sorted(set(all_pdfs) - set(all_data[mk])) for _, mk in MODELS}
    for _, mk in MODELS:
        for field in all_fields:
            for pdf in all_pdfs:
                if pdf not in all_data[mk]:
                    continue
                pdf_data = all_data[mk][pdf]
                vals = [pdf_data.get(seed, {}).get(field) for seed in SEEDS]
                per_seed, mean_acc, std_acc, maj, is_tie, tied_str = compute_stats(vals)
                all_stats[mk][field][pdf] = {
                    "vals": vals, "per_seed": per_seed,
                    "mean": mean_acc, "std": std_acc, "maj": maj,
                    "is_tie": is_tie, "tied_str": tied_str,
                }

    for label, mk in MODELS:
        if missing_reports[mk]:
            print(f"{label}: {len(missing_reports[mk])} reports never extracted, excluded from stats "
                  f"(not counted as null): {missing_reports[mk]}")

    # Computed twice: strict (exact match only) vs with absence-equivalence,
    # so the print shows the size of that leniency's effect. Sentinel
    # exclusion is identical in both - gt_match doesn't use absence_equiv for it.
    acc_strict = compute_model_accuracy_stats(scored_fields, all_pdfs, all_stats, truth, {})
    acc_equiv  = compute_model_accuracy_stats(scored_fields, all_pdfs, all_stats, truth, ABSENCE_EQUIVALENTS)
    acc_stats = acc_equiv  # used by the bar plot - the reported headline metric

    print("\n% correct per model — strict vs with variable-specific absence-equivalence, "
      "real non-sentinel GT only:")
    for label, mk in MODELS:
        s, e = acc_strict[mk], acc_equiv[mk]
        print(f"  {label:12s}  strict: {s['pct']:5.1f}%   with equivalence: {e['pct']:5.1f}%   "
              f"(n={e['n_scored']} scored, {e['n_excluded']} excluded)")

    bias_stats = compute_null_bias_stats(scored_fields, all_pdfs, all_stats, truth, ABSENCE_EQUIVALENTS)
    print("\nNull-bias breakdown (how often each model answers at all, and accuracy when it does):")
    for label, mk in MODELS:
        b = bias_stats[mk]
        print(f"  {label:12s}  null on {b['null_rate']:5.1f}% of cells ({b['null_n']} of {b['scored']})   "
              f"acc when it DOES answer: {b['acc_when_answered']:5.1f}%  (n={b['answered_n']})")

    print("\nBuilding plot ...")
    make_overall_accuracy_plot(acc_stats, RESULTS_DIR)

    xlsx_path = RESULTS_DIR / "model_comparison_gt.xlsx"
    print("\nBuilding XLSX ...")
    build_xlsx(all_fields, scored_fields, all_pdfs, all_stats, truth, gt_columns, ABSENCE_EQUIVALENTS, xlsx_path)

    n_detail_rows = len(all_fields) * len(all_pdfs) * len(MODELS)
    print(f"""
Done.
  {RESULTS_DIR}/
    overall_accuracy_by_model.png/svg
    {xlsx_path.name}
      Summary : scorecard, % correct, worst variables, best model/report ({len(scored_fields)} scored fields)
      Details : {n_detail_rows} rows ({len(all_fields)} fields incl. no-GT ones)
      Legend  : colour key + definitions
""")


if __name__ == "__main__":
    main()