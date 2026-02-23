"""analyze_seeds.py — Show all seed outputs per PDF and compute consistency."""

import json
import argparse
from collections import defaultdict
from pathlib import Path


def flatten(d):
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                items[sub_k] = sub_v
        else:
            items[k] = v
    return items


def consistency(values):
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "N/A (all null)"
    majority = max(set(non_null), key=non_null.count)
    pct = non_null.count(majority) / len(values) * 100
    return f"{int(pct)}% -> {majority}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    seeds = [42, 123, 456, 789, 1011]
    skip = {"extraction_notes", "extraction_confidence"}

    pdf_groups = defaultdict(dict)
    for f in sorted(output_dir.glob("*.extracted.json")):
        for seed in seeds:
            if f"_seed{seed}.extracted" in f.name:
                pdf_name = f.name.replace(f"_seed{seed}.extracted.json", "")
                pdf_groups[pdf_name][seed] = f
                break

    for pdf_name in sorted(pdf_groups.keys()):
        print("\n" + "=" * 75)
        print(f"PDF: {pdf_name}")
        print("=" * 75)

        seed_data = {}
        for seed in seeds:
            fpath = pdf_groups[pdf_name].get(seed)
            if fpath and fpath.exists():
                with open(fpath) as f:
                    seed_data[seed] = flatten(json.load(f))
            else:
                seed_data[seed] = {}

        all_fields = sorted(set(
            k for d in seed_data.values()
            for k in d.keys()
            if k not in skip
        ))

        print(f"\n{'VARIABLE':<30} {'seed42':>8} {'seed123':>8} {'seed456':>8} {'seed789':>8} {'seed1011':>9}")
        print("-" * 75)
        for field in all_fields:
            vals = [str(seed_data[s].get(field, "-")) for s in seeds]
            print(f"{field:<30} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>9}")

        print(f"\n{'CONSISTENCY TABLE':^75}")
        print("-" * 75)
        print(f"{'VARIABLE':<30} {'CONSISTENCY'}")
        print("-" * 75)
        for field in all_fields:
            vals = [seed_data[s].get(field) for s in seeds]
            print(f"{field:<30} {consistency(vals)}")

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()
