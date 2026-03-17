"""analyze_seeds.py — Show all seed outputs per PDF and compute consistency."""

import csv
import json
from collections import defaultdict
from pathlib import Path

MODEL = "qwen2.5-14b"
OUTPUT_DIR = f"/N/project/ADRD/neuropathoroot/output/{MODEL}"
LOG_FILE = f"/N/project/ADRD/neuropathoroot/NeuroPathlogy/seeds_analysis/seeds_analysis_{MODEL}.log"

SKIP_KEYS = {"extraction_notes", "extraction_confidence", "field_annotations"}


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


def consistency(values):
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "N/A (all null)", None, False, 0, []
    
    top_count = max(non_null.count(v) for v in set(non_null))
    winners = [v for v in set(non_null) if non_null.count(v) == top_count]
    pct = top_count / len(values) * 100
    
    if len(winners) > 1:
        return f"{int(pct)}% -> TIE {sorted(winners)}", None, True, int(pct), winners
    return f"{int(pct)}% -> {winners[0]}", winners[0], False, int(pct), []


def main():
    output_dir = Path(OUTPUT_DIR)
    seeds = [0, 1, 2, 3, 4]

    lines = []

    def emit(line=""):
        print(line)
        lines.append(line)

    csv_rows = []

    pdf_groups = defaultdict(dict)
    for f in sorted(output_dir.glob("*.extracted.json")):
        for seed in seeds:
            if f"_seed{seed}.extracted" in f.name:
                pdf_name = f.name.replace(f"_seed{seed}.extracted.json", "")
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
            # emit(f"{field:<30} {consistency(vals)}")
            label, majority, is_tie, pct, winners = consistency(vals)
            emit(f"{field:<30} {label}")
            csv_rows.append({
                "pdf": pdf_name,
                "variable": field,
                "seed0": seed_data[0].get(field),
                "seed1": seed_data[1].get(field),
                "seed2": seed_data[2].get(field),
                "seed3": seed_data[3].get(field),
                "seed4": seed_data[4].get(field),
                "majority": majority,
                "is_tie": is_tie,
                "tied_values": str(sorted(winners)) if is_tie else None,
                "agreement_pct": pct,
            })

    emit("\n" + "=" * 75)
    emit("DONE")
    emit("=" * 75)

    log_path = Path(LOG_FILE)
    with open(log_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nLog saved to: {log_path.resolve()}")

    csv_path = Path(LOG_FILE).parent / f"seeds_analysis_{MODEL}.csv"
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