"""compute_confidence_metrics.py

Computes per-variable confidence metrics across 5 seeds for all reports.

For each report x variable combination:
    - consensus_value       : modal value across seeds
    - seed_values           : raw list of values across seeds
    - seed_agreement_rate   : fraction of seeds matching consensus value
    - mean_confidence       : mean LLM self-reported confidence across seeds
    - std_confidence        : std dev of self-reported confidence across seeds
    - confidence_tier       : high / medium / low / missing_annotation
    - mean_logprob          : mean output-token logprob across seeds (post-reasoning)
    - mean_output_prob      : mean exp(logprob) across seeds — human-readable probability
    - mean_entropy          : mean entropy over top-5 alternatives across seeds (bits)

Output:
    /N/project/ADRD/neuropathoroot/results/confidence_metrics.csv

Notes:
    - Value keys (block fields) are master — any variable missing from
      field_annotations is still included with NaN confidence columns.
    - programmatic block (NACCVASC, NACCBRNN) is included.
    - Free-text / string-valued variables are included as-is.
    - Seeds 0-4 expected; missing seed files are skipped with a warning.
    - Logprob columns are None when logprob files are absent or when vLLM
      degraded to logprobs=False due to NaN (cascade fallback in main.py).
    - entropy is None when top_logprobs=1 fallback was used (only 1 alternative).
"""

from __future__ import annotations

import json
import logging
import warnings
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("confidence_metrics")

# ── Hardcoded paths ────────────────────────────────────────────────────────
INPUT_DIR  = Path("/N/project/ADRD/neuropathoroot/output/oss-20b")
OUTPUT_DIR = Path("/N/project/ADRD/neuropathoroot/results")
OUTPUT_CSV = OUTPUT_DIR / "confidence_metrics.csv"
LOGPROBS_DIR = INPUT_DIR / "logprobs"

SEEDS      = [0, 1, 2, 3, 4]

# Blocks to skip entirely (not extracted variables)
SKIP_KEYS = {
    "field_annotations",
    "extraction_confidence",
    "extraction_notes",
}

# Confidence tier thresholds
TIER_HIGH   = 0.8
TIER_MEDIUM = 0.6


# ── Helpers ────────────────────────────────────────────────────────────────

def get_confidence_tier(mean_conf: float | None) -> str:
    if mean_conf is None:
        return "missing_annotation"
    if mean_conf >= TIER_HIGH:
        return "high"
    if mean_conf >= TIER_MEDIUM:
        return "medium"
    return "low"


def flatten_values(data: dict) -> dict[str, object]:
    """
    Flatten all block-level variable values into a single dict.
    Master source of truth — every variable that exists here gets a row.
    Skips SKIP_KEYS. Handles nested block dicts and the programmatic block.
    """
    flat = {}
    for key, val in data.items():
        if key in SKIP_KEYS:
            continue
        if isinstance(val, dict):
            # nested block — flatten one level
            for var, var_val in val.items():
                flat[var] = var_val
        else:
            # shouldn't happen at top level but handle gracefully
            flat[key] = val
    return flat


def get_annotation_confidence(data: dict, var: str) -> float | None:
    """Pull confidence from field_annotations for a variable. None if missing."""
    annotations = data.get("field_annotations", {})
    ann = annotations.get(var)
    if ann is None:
        return None
    if isinstance(ann, dict):
        conf = ann.get("confidence")
        return float(conf) if conf is not None else None
    return None


def compute_consensus(values: list) -> object:
    """
    Modal value across seeds.
    For ties, returns the value that appears first among tied winners.
    None is treated as a legitimate value, not missing.
    """
    if not values:
        return None
    # Convert to hashable for Counter — wrap lists/dicts as strings
    def hashable(v):
        if isinstance(v, (list, dict)):
            return json.dumps(v, sort_keys=True)
        return v

    counts = Counter(hashable(v) for v in values)
    modal_hashable = counts.most_common(1)[0][0]

    # Return original (unhashed) value
    for v in values:
        if hashable(v) == modal_hashable:
            return v
    return None


def seed_agreement_rate(values: list, consensus) -> float:
    """Fraction of seeds whose value matches the consensus."""
    if not values:
        return 0.0
    def hashable(v):
        if isinstance(v, (list, dict)):
            return json.dumps(v, sort_keys=True)
        return v
    consensus_h = hashable(consensus)
    matches = sum(1 for v in values if hashable(v) == consensus_h)
    return round(matches / len(values), 4)


def load_seed_data(report_id: str) -> dict[int, dict]:
    """
    Load all available seed JSONs for a report.
    Returns {seed: parsed_json}. Missing seeds are skipped with a warning.
    """
    seed_data = {}
    for seed in SEEDS:
        path = INPUT_DIR / f"{report_id}_seed{seed}.extracted.json"
        if not path.exists():
            logger.warning("missing seed file: %s", path)
            continue
        try:
            with open(path, encoding="utf-8") as f:
                seed_data[seed] = json.load(f)
        except Exception as e:
            logger.warning("failed to load %s: %s", path, e)
    return seed_data


def get_all_report_ids() -> list[str]:
    """
    Discover all unique report IDs from seed0 files in INPUT_DIR.
    Report ID = filename stem with _seed0 stripped.
    """
    report_ids = []
    for f in sorted(INPUT_DIR.glob("*_seed0.extracted.json")):
        report_id = f.stem.replace("_seed0.extracted", "")
        report_ids.append(report_id)
    return report_ids


def load_seed_logprobs(report_id: str) -> dict[int, dict]:
    """
    Load all available logprob JSONs for a report.
    Returns {seed: {var: {logprob, prob, entropy}}}. Missing files skipped silently.
    """
    seed_lp = {}
    for seed in SEEDS:
        path = LOGPROBS_DIR / f"{report_id}_seed{seed}.logprobs.json"
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                seed_lp[seed] = json.load(f)
        except Exception as e:
            logger.warning("failed to load logprobs %s: %s", path, e)
    return seed_lp


# ── Core processing ────────────────────────────────────────────────────────

def process_report(report_id: str) -> list[dict]:
    """
    Process all seeds for one report.
    Returns list of row dicts, one per variable.
    """
    seed_data    = load_seed_data(report_id)
    logprob_data = load_seed_logprobs(report_id)

    if not seed_data:
        logger.error("no seed data found for report %s — skipping", report_id)
        return []

    # Use seed0 (or first available) to establish the master variable list
    master_seed = seed_data[min(seed_data.keys())]
    master_vars = flatten_values(master_seed)

    # Also union variables across all seeds in case a seed has extras
    all_vars: set[str] = set(master_vars.keys())
    for sd in seed_data.values():
        all_vars.update(flatten_values(sd).keys())

    rows = []
    for var in sorted(all_vars):
        # Collect value and confidence from each seed
        seed_values      = []
        seed_confidences = []
        seed_logprobs    = []
        seed_probs       = []
        seed_entropies   = []

        for seed in SEEDS:
            if seed not in seed_data:
                continue

            sd   = seed_data[seed]
            flat = flatten_values(sd)
            val  = flat.get(var)
            conf = get_annotation_confidence(sd, var)

            seed_values.append(val)
            if conf is not None:
                seed_confidences.append(conf)

            # Logprob metrics
            if seed in logprob_data:
                lp_entry = logprob_data[seed].get(var)
                if lp_entry is not None:
                    if lp_entry.get("logprob") is not None:
                        seed_logprobs.append(lp_entry["logprob"])
                    if lp_entry.get("prob") is not None:
                        seed_probs.append(lp_entry["prob"])
                    if lp_entry.get("entropy") is not None:
                        seed_entropies.append(lp_entry["entropy"])

        consensus  = compute_consensus(seed_values)
        agreement  = seed_agreement_rate(seed_values, consensus)

        mean_conf  = round(mean(seed_confidences), 4) if seed_confidences else None
        std_conf   = round(stdev(seed_confidences), 4) if len(seed_confidences) > 1 else 0.0
        tier       = get_confidence_tier(mean_conf)

        mean_logprob     = round(mean(seed_logprobs),   6) if seed_logprobs   else None
        mean_output_prob = round(mean(seed_probs),      6) if seed_probs      else None
        mean_entropy     = round(mean(seed_entropies),  6) if seed_entropies  else None

        rows.append({
            "report_id"           : report_id,
            "variable"            : var,
            "consensus_value"     : consensus,
            "seed_values"         : seed_values,
            "seed_agreement_rate" : agreement,
            "mean_confidence"     : mean_conf,
            "std_confidence"      : std_conf,
            "confidence_tier"     : tier,
            "mean_logprob"        : mean_logprob,
            "mean_output_prob"    : mean_output_prob,
            "mean_entropy"        : mean_entropy,
        })

    return rows


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_ids = get_all_report_ids()
    if not report_ids:
        logger.error("no extracted JSON files found in %s", INPUT_DIR)
        return

    logger.info("found %d reports in %s", len(report_ids), INPUT_DIR)

    all_rows = []
    for i, report_id in enumerate(report_ids, 1):
        logger.info("[%d/%d] processing %s", i, len(report_ids), report_id)
        rows = process_report(report_id)
        all_rows.extend(rows)

    if not all_rows:
        logger.error("no rows produced — check input files")
        return

    df = pd.DataFrame(all_rows)

    # seed_values is a list — store as string for CSV compatibility
    df["seed_values"] = df["seed_values"].apply(
        lambda v: json.dumps(v) if isinstance(v, list) else str(v)
    )

    df.to_csv(OUTPUT_CSV, index=False)
    logger.info("wrote %d rows to %s", len(df), OUTPUT_CSV)

    # ── Summary stats ──────────────────────────────────────────────────────
    logger.info("─── Summary ───────────────────────────────────────")
    logger.info("total rows          : %d", len(df))
    logger.info("unique reports      : %d", df["report_id"].nunique())
    logger.info("unique variables    : %d", df["variable"].nunique())
    logger.info("mean agreement rate : %.4f", df["seed_agreement_rate"].mean())

    tier_counts = df["confidence_tier"].value_counts()
    for tier, count in tier_counts.items():
        pct = 100 * count / len(df)
        logger.info("  tier %-20s: %d rows (%.1f%%)", tier, count, pct)

    low_agreement = df[df["seed_agreement_rate"] < 0.6]
    logger.info(
        "variables with seed agreement < 0.6 : %d (%.1f%%)",
        len(low_agreement),
        100 * len(low_agreement) / len(df),
    )

    logprob_coverage = df["mean_logprob"].notna().sum()
    logger.info(
        "rows with logprob data              : %d (%.1f%%)",
        logprob_coverage,
        100 * logprob_coverage / len(df),
    )
    entropy_coverage = df["mean_entropy"].notna().sum()
    logger.info(
        "rows with entropy data              : %d (%.1f%%)",
        entropy_coverage,
        100 * entropy_coverage / len(df),
    )


if __name__ == "__main__":
    main()