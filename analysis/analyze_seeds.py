"""analyze_seeds.py — Show all seed outputs per PDF and compute consistency.

Set PIPELINE at the top to switch between extraction pipelines:
    PIPELINE = "primary"   # *.extracted.json  — 199 NACC variables
    PIPELINE = "residual"  # *.residual.json   — 49 residual variables + programmatic
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# ── SWITCH HERE ─────────────────────────────────────────────────────────────
PIPELINE = "residual"   # "primary" | "residual"
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, str] = {
    "oss-120b":     "openai/gpt-oss-120b",
    "oss-20b":      "openai/gpt-oss-20b",
    "llama3.1-8b":  "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-14b":  "Qwen/Qwen2.5-14B-Instruct",
}

if PIPELINE == "primary":
    MODELS_TO_RUN  = list(MODEL_ALIASES.keys())
    FILE_SUFFIX    = ".extracted.json"
    SEED_MARKER    = "_seed{seed}.extracted"   # used to strip pdf name
    OUTPUT_SUBDIR  = ""                        # directly under output_gt/{MODEL}
    LOG_SUBDIR     = "seeds_analysis_gt"
    LOG_PREFIX     = "seeds_analysis"
    SKIP_KEYS      = {"extraction_notes", "extraction_confidence", "field_annotations"}
elif PIPELINE == "residual":
    MODELS_TO_RUN  = list(MODEL_ALIASES.keys())
    FILE_SUFFIX    = ".residual.json"
    SEED_MARKER    = "_seed{seed}.residual"
    OUTPUT_SUBDIR  = "residual_extraction"     # under output_gt/{MODEL}/residual_extraction
    LOG_SUBDIR     = "seeds_analysis_gt_residual"
    LOG_PREFIX     = "seeds_analysis_residual"
    SKIP_KEYS      = {"extraction_notes", "extraction_confidence", "field_annotations"}
else:
    raise ValueError(f"Unknown PIPELINE={PIPELINE!r}. Must be 'primary' or 'residual'.")


def flatten(d):
    items = {}
    for k, v in d.items():
        if k in SKIP_KEYS:
            continue
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                # Only include leaf values (skip nested dicts like field_annotations entries)
                if not isinstance(sub_v, dict):
                    items[sub_k] = sub_v
        else:
            items[k] = v
    return items


def normalize(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip().lower().rstrip(".")
    return v


def consistency(values):
    normalized = [normalize(v) for v in values]

    top_count = max(normalized.count(v) for v in set(normalized))
    winners = [v for v in set(normalized) if normalized.count(v) == top_count]
    pct = top_count / len(normalized) * 100

    if len(winners) > 1:
        return f"{int(pct)}% -> TIE {sorted(str(w) for w in winners)}", None, True, int(pct), winners
    return f"{int(pct)}% -> {winners[0]}", winners[0], False, int(pct), []


def main():

    for MODEL in MODELS_TO_RUN:
        # Build output dir based on pipeline
        base_dir   = Path(f"/N/project/ADRD/neuropathoroot/output_gt/{MODEL}")
        output_dir = base_dir / OUTPUT_SUBDIR if OUTPUT_SUBDIR else base_dir

        log_file = base_dir / LOG_SUBDIR / f"{LOG_PREFIX}_{MODEL}.log"

        if not output_dir.exists():
            print(f"[SKIP] {MODEL}: output dir not found — {output_dir}")
            continue

        log_file.parent.mkdir(parents=True, exist_ok=True)

        seeds = [0, 1, 2, 3, 4]

        lines = []

        def emit(line=""):
            print(line)
            lines.append(line)

        csv_rows = []

        # ── Group files by PDF name ────────────────────────────────────────
        pdf_groups = defaultdict(dict)
        for f in sorted(output_dir.glob(f"*{FILE_SUFFIX}")):
            for seed in seeds:
                marker = SEED_MARKER.format(seed=seed)
                if f"_seed{seed}" in f.name and f.name.endswith(FILE_SUFFIX):
                    pdf_name = f.name.replace(
                        f"_seed{seed}{FILE_SUFFIX}", ""
                    )
                    pdf_groups[pdf_name][seed] = f
                    break

        for pdf_name in sorted(pdf_groups.keys()):
            emit("\n" + "=" * 75)
            emit(f"PDF: {pdf_name}")
            emit("=" * 75)

            seed_data = {}
            for seed in seeds:
                fpath = pdf_groups[pdf_name].get(seed)
                if fpath and fpath.exists():
                    with open(fpath) as f:
                        try:
                            seed_data[seed] = flatten(json.load(f))
                        except json.JSONDecodeError as e:
                            emit(f"  [WARNING] Could not parse seed{seed}: {e}")
                            seed_data[seed] = {}
                else:
                    seed_data[seed] = {}

            all_fields = sorted(set(
                k for d in seed_data.values()
                for k in d.keys()
            ))

            emit(f"\n{'VARIABLE':<30} {'seed0':>8} {'seed1':>8} {'seed2':>8} {'seed3':>8} {'seed4':>9}")
            emit("-" * 75)
            for field in all_fields:
                vals = [str(seed_data[s].get(field, "-")) for s in seeds]
                emit(f"{field:<30} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>9}")

            emit(f"\n{'CONSISTENCY TABLE':^75}")
            emit("-" * 75)
            emit(f"{'VARIABLE':<30} {'CONSISTENCY'}")
            emit("-" * 75)
            for field in all_fields:
                vals = [seed_data[s].get(field) for s in seeds]
                label, majority, is_tie, pct, winners = consistency(vals)
                emit(f"{field:<30} {label}")
                csv_rows.append({
                    "pdf":           pdf_name,
                    "variable":      field,
                    "seed0":         seed_data[0].get(field),
                    "seed1":         seed_data[1].get(field),
                    "seed2":         seed_data[2].get(field),
                    "seed3":         seed_data[3].get(field),
                    "seed4":         seed_data[4].get(field),
                    "majority":      majority,
                    "is_tie":        is_tie,
                    "tied_values":   str(sorted(str(w) for w in winners)) if is_tie else None,
                    "agreement_pct": pct,
                })

        emit("\n" + "=" * 75)
        emit("DONE")
        emit("=" * 75)

        with open(log_file, "w") as f:
            f.write("\n".join(lines))
        print(f"\nLog saved to: {log_file.resolve()}")

        csv_path = log_file.parent / f"{LOG_PREFIX}_{MODEL}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "pdf", "variable", "seed0", "seed1", "seed2", "seed3", "seed4",
                "majority", "is_tie", "tied_values", "agreement_pct"
            ])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()