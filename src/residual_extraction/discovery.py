"""discovery.py — Stage 1A: Agentic neuropathologist reasoning pass.

Reads each autopsy report (PDF or text), prompts OSS-20B as an expert
neuropathologist to identify all clinically significant findings NOT captured
by the 199 NACC NP Form variables, and saves structured observations per
report as {report_id}.discovery.json.

Single-seed discovery pass — breadth across 161 reports matters here,
not seed-level consistency (that comes in Stage 2 structured extraction).

Output per report:
    {
        "report_id": "6368",
        "model": "openai/gpt-oss-20b",
        "clinical_observations": [
            {
                "category": "brainstem asymmetry",
                "finding": "Marked depigmentation right SN, normal left SN",
                "clinical_significance": "Asymmetric dopaminergic neuron loss..."
            },
            ...
        ],
        "overall_impression": "Primary finding is asymmetric brainstem pathology..."
    }

Usage (single report — used by SLURM loop):
    python discovery.py --input report.pdf --vllm-url http://localhost:8000

Usage (full directory — used for local testing):
    python discovery.py --reports-dir /path/to/reports --vllm-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from json_repair import repair_json
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from primary_extraction.main import load_report, strip_json_fences, _sanitize_for_json

logger = logging.getLogger("discovery")

# ---------------------------------------------------------------------------
# Build NACC boundary string from rddnp.csv
# ---------------------------------------------------------------------------

def build_nacc_boundary(rddnp_path: Path) -> str:
    """Read rddnp.csv and build the 199-variable boundary string for the prompt.

    Uses VariableName + ShortDescriptor only.

    Format per line:
        NPTHAL: Thal phase of amyloid deposition
        NACCBRAA: Braak neurofibrillary tangle stage
        ...
    """
    if not rddnp_path.exists():
        raise FileNotFoundError(
            f"rddnp.csv not found at {rddnp_path}. "
            "Set --rddnp-path to the correct location."
        )

    lines = []
    with open(rddnp_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            var  = row["VariableName"].strip()
            desc = row["ShortDescriptor"].strip()
            if var:
                lines.append(f"  {var}: {desc}")

    if len(lines) != 199:
        logger.warning(
            "Expected 199 variables in rddnp.csv, found %d. "
            "Check the file for extra or missing rows.",
            len(lines),
        )

    boundary = (
        "The following 199 variables are ALREADY captured by the NACC NP Form.\n"
        "Do NOT report findings covered by any of these variables.\n"
        "Your task is to identify clinically significant information the report\n"
        "contains that is NOT captured by any variable below.\n\n"
        + "\n".join(lines)
    )
    logger.info("Built NACC boundary from %s (%d variables)", rddnp_path, len(lines))
    return boundary


# ---------------------------------------------------------------------------
# Pydantic output model
# ---------------------------------------------------------------------------

class ClinicalObservation(BaseModel):
    """Single finding beyond the NACC 199 variables."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(
        description=(
            "Short label for the type of finding. Examples: "
            "'weight asymmetry', 'vascular morphology', 'brainstem pigmentation', "
            "'protein pathology spatial', 'structural anomaly', 'incidental finding', "
            "'regional atrophy pattern'. Use your own judgment — do not force into these."
        )
    )
    finding: str = Field(
        description=(
            "The specific finding as observed in the report. Be precise and include "
            "any measurements, laterality, severity, or anatomical detail mentioned. "
            "Use standard clinical terminology even if the report uses informal language."
        )
    )
    clinical_significance: str = Field(
        description=(
            "Why this finding matters clinically. What does it suggest about the "
            "pathological process, disease asymmetry, or diagnosis? Write at least "
            "one full clinical sentence explaining the pathological significance."
        )
    )


class DiscoveryOutput(BaseModel):
    """Full discovery output for one report."""

    model_config = ConfigDict(extra="ignore")

    report_id: str = Field(
        description="The report identifier extracted from the filename."
    )
    clinical_observations: List[ClinicalObservation] = Field(
        description=(
            "List of all clinically significant findings in this report that are "
            "NOT captured by the 199 NACC NP Form variables listed in the prompt. "
            "Be thorough — include measurements, asymmetries, spatial details, "
            "structural anomalies, and incidental findings. "
            "Do not include findings already covered by the NACC variables above. "
            "Minimum 3 observations per report if the report contains any detail "
            "beyond the NACC variables."
        )
    )
    overall_impression: str = Field(
        description=(
            "A 2-3 sentence neuropathologist-style summary of the most clinically "
            "notable findings in this report that go beyond what NACC captures. "
            "What would you highlight to a colleague reviewing this case?"
        )
    )

    @model_validator(mode="after")
    def at_least_one_observation(self) -> "DiscoveryOutput":
        if len(self.clinical_observations) == 0:
            raise ValueError(
                "clinical_observations must contain at least one finding. "
                "If the report truly contains nothing beyond NACC variables, "
                "record that explicitly as an observation."
            )
        return self


# ---------------------------------------------------------------------------
# Prompt construction — boundary built at runtime from rddnp.csv
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert neuropathologist with 30 years of experience reading brain \
autopsy reports. You have been asked to review the following report and identify \
all clinically significant findings that are NOT captured by the standard NACC \
Neuropathology Form.

{nacc_boundary}

YOUR TASK:
Read this report carefully as you would in a clinical setting. Identify every \
finding that a neuropathologist would consider significant but that is absent \
from the NACC variable list above.

SCOPE: Focus exclusively on findings relevant to the brain, brainstem, cerebellum, \
cranial nerves, spinal cord, and intracranial vasculature. Do NOT report findings \
from other organ systems (kidneys, liver, heart, lungs, skin, scalp, systemic \
malignancies, or any non-neurological tissue) even if they are mentioned in the \
report. If a finding has no plausible connection to brain pathology, neurodegeneration, \
cognitive decline, or neuropathological diagnosis, exclude it entirely.

CATEGORIES: Assign each observation to one of the following categories. If a finding \
clearly belongs to none of these, you may use a brief descriptive label, but prefer \
the listed categories:
  - hemisphere_weights
  - cerebellar_weights
  - brainstem_weights
  - corpus_callosum_morphometry
  - hippocampal_morphometry
  - amygdala_morphometry
  - circle_of_willis_diameters
  - cow_anatomical_variant
  - substantia_nigra_pigmentation
  - locus_coeruleus_pigmentation
  - brainstem_morphometry
  - regional_cortical_atrophy
  - subcortical_atrophy
  - ventricular_dilation
  - white_matter_changes
  - vascular_pathology
  - protein_pathology_spatial
  - neuronal_loss_regional
  - incidental_structural_finding
  - incidental_pathology

OBSERVATION GRANULARITY RULES — follow these exactly:

1. BILATERAL MEASUREMENTS: For any bilateral brain structure where the report gives \
separate left and right values, create one SEPARATE observation for the right side \
and one SEPARATE observation for the left side. Never combine both sides into a \
single asymmetry or combined observation. This applies to any bilateral structure \
including but not limited to hemibrain weights, cerebral hemisphere weights, \
cerebellar hemisphere weights, brainstem half weights, hippocampal measurements, \
amygdala measurements, and optic nerve measurements.
   CORRECT: "Right hemibrain weight: 487.0 grams (fresh)"
            "Left hemibrain weight: 465.0 grams (fresh)"  (two separate observations)
   WRONG:   "Hemispheric weight asymmetry: right 487g vs left 465g"

2. CIRCLE OF WILLIS VESSEL DIAMETERS: Create one SEPARATE observation per named \
vessel. Do not combine all vessel measurements into a single observation.
   CORRECT: "Basilar artery diameter: 3.0 mm"
            "Right internal carotid artery diameter: 4.0 mm"
            "Left internal carotid artery diameter: 3.0 mm"  (each a separate observation)
   WRONG:   "Circle of Willis: basilar 3mm, right ICA 4mm, left ICA 3mm"

3. PRESERVE NUMERIC VALUES: When a measurement is given numerically in the report \
(grams, mm, cm), always include the exact numeric value in the finding field. \
Never convert a numeric measurement into a qualitative severity description.
   CORRECT: "Right cerebellar hemisphere weight: 58.5 grams"
   WRONG:   "Cerebellar weight asymmetry: moderate"

CLINICAL SIGNIFICANCE RULES:
- Write at least one full clinical sentence explaining the pathological significance.
- Frame significance in terms of relevance to neurodegeneration, Alzheimer disease \
neuropathologic change, cognitive decline, motor dysfunction, or neuropathological \
diagnosis. Do not report general autopsy observations with no connection to brain disease.
- If a finding is clinically ambiguous, explain which neuropathological process it \
may reflect and why it warrants structured capture across a cohort.

Be thorough for brain-relevant findings. A finding present in the report but absent \
from NACC is valuable even if it seems minor — it may recur across the cohort.

Use standard clinical terminology. If the report uses informal language, \
translate it to proper neuropathological terms in your output.

OUTPUT FORMAT: valid JSON only — no markdown fences, no commentary before or after.
The JSON must have EXACTLY these top-level keys:
  - report_id: string
  - clinical_observations: list of objects, each with keys: category, finding, clinical_significance
  - overall_impression: string

Do not add any other top-level keys. Do not add fields like 'additional_findings', \
'measurements', 'summary', or anything else outside the three keys above.
"""

USER_PROMPT = "NEUROPATHOLOGY REPORT:\n\n{report_text}"


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def run_discovery(
    report_text: str,
    report_id: str,
    client: OpenAI,
    system_prompt: str,
    model: str = "openai/gpt-oss-20b",
    max_new_tokens: int = 32768,
    temperature: float = 0.01,
    top_p: float = 0.9,
    reasoning_effort: str = "high",
    max_retries: int = 3,
) -> DiscoveryOutput:
    """Run the agentic discovery pass for one report."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": USER_PROMPT.format(report_text=report_text)},
    ]

    extra_body = {
        "repetition_penalty": 1.1,
        "reasoning_effort": reasoning_effort,
    }

    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        logger.info("report %s — attempt %d/%d", report_id, attempt, max_retries)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=0,
                extra_body=extra_body,
            )
            raw = response.choices[0].message.content or ""
            logger.debug("raw output: %r", raw[:300])

        except Exception as e:
            logger.error("API call failed: %s", e)
            raise

        # Parse JSON
        cleaned = strip_json_fences(raw)
        try:
            content = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("JSON decode failed, attempting repair: %s", e)
            try:
                content = repair_json(cleaned, return_objects=True)
                if not isinstance(content, dict) or not content:
                    raise ValueError("repair produced empty result")
                logger.info("JSON repaired successfully")
            except Exception as repair_err:
                last_exc = repair_err
                messages = messages[:2] + [
                    {"role": "assistant", "content": cleaned},
                    {"role": "user",
                     "content": (
                         f"Your response was not valid JSON. Error: {e}. "
                         "Output valid JSON only, no markdown."
                     )},
                ]
                continue

        content["report_id"] = report_id

        # Pydantic validation
        try:
            return DiscoveryOutput.model_validate(content)
        except Exception as val_err:
            last_exc = val_err
            logger.warning("Validation failed: %s", val_err)
            messages = messages[:2] + [
                {"role": "assistant",
                 "content": json.dumps(_sanitize_for_json(content), indent=2)},
                {"role": "user",
                 "content": (
                     f"Your JSON had validation errors:\n{val_err}\n"
                     "Fix these errors and output corrected JSON only."
                 )},
            ]
            continue

    raise last_exc


# ---------------------------------------------------------------------------
# Shared processing logic
# ---------------------------------------------------------------------------

def process_report(
    report_path: Path,
    output_dir: Path,
    client: OpenAI,
    system_prompt: str,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    reasoning_effort: str,
    max_retries: int,
) -> bool:
    """Process one report. Returns True on success, False on failure.

    Resume logic:
      - If .discovery.json exists: skip (already done).
      - If .discovery.failed exists: delete it and retry — failed markers
        are cleared automatically on re-run so the job always attempts
        every report without a successful JSON.
      - If neither exists: run fresh.
    """
    report_id = report_path.stem
    out_path    = output_dir / f"{report_id}.discovery.json"
    failed_path = output_dir / f"{report_id}.discovery.failed"

    # Already successfully completed — skip
    if out_path.exists():
        logger.info("skipping %s — already exists", report_id)
        return True

    # Previous failure marker — clear it and retry
    if failed_path.exists():
        logger.info("clearing previous failure for %s — retrying", report_id)
        failed_path.unlink()

    logger.info("processing %s", report_id)
    try:
        report_text = load_report(report_path)
        result = run_discovery(
            report_text=report_text,
            report_id=report_id,
            client=client,
            system_prompt=system_prompt,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            max_retries=max_retries,
        )

        out_data = result.model_dump(mode="json")

        out_path.write_text(
            json.dumps(_sanitize_for_json(out_data), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "wrote %s (%d observations)",
            out_path.name, len(result.clinical_observations),
        )
        return True

    except Exception as e:
        logger.error("FAILED %s: %s", report_id, e)
        failed_path.write_text(str(e), encoding="utf-8")
        return False  # caller continues to next report — job does not exit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Stage 1A: Agentic neuropathologist discovery pass.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input: either a single report (SLURM loop) or a directory (local testing)
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--input", type=Path,
        help="Single report file (.pdf or .txt) — used by SLURM loop",
    )
    input_group.add_argument(
        "--reports-dir", type=Path,
        help="Directory of reports — used for local testing",
    )

    ap.add_argument(
        "--output-dir", type=Path,
        required=True,
        help=(
            "Directory to write .discovery.json files. "
            "Set dynamically in SLURM script based on model alias."
        ),
    )
    ap.add_argument(
        "--rddnp-path", type=Path,
        default=Path("/N/project/ADRD/neuropathoroot/csv_input/rdd-np.csv"),
        help="Path to rddnp.csv — source of 199 NACC variable names and descriptors",
    )
    ap.add_argument(
        "--vllm-url", type=str, required=True,
        help="vLLM server URL e.g. http://localhost:8000",
    )
    ap.add_argument(
        "--model", type=str, default="openai/gpt-oss-20b",
    )
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.01)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument(
        "--reasoning-effort", type=str, default="high",
        choices=["low", "medium", "high"],
    )
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build NACC boundary from rddnp.csv — automated, no hardcoding
    nacc_boundary = build_nacc_boundary(args.rddnp_path)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(nacc_boundary=nacc_boundary)
    logger.info("System prompt length: %d chars", len(system_prompt))

    # vLLM client — same pattern as main.py
    client = OpenAI(
        base_url=f"{args.vllm_url}/v1",
        api_key="dummy",
    )

    shared_kwargs = dict(
        output_dir=args.output_dir,
        client=client,
        system_prompt=system_prompt,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        reasoning_effort=args.reasoning_effort,
        max_retries=args.max_retries,
    )

    # ── Single report mode (SLURM loop) ───────────────────────────────────
    if args.input is not None:
        if not args.input.exists():
            logger.error("file not found: %s", args.input)
            sys.exit(1)
        # Always returns True/False — never exits with code 1 so SLURM
        # loop continues to the next report in the chunk regardless of outcome
        process_report(report_path=args.input, **shared_kwargs)
        sys.exit(0)

    # ── Directory mode (local testing) ────────────────────────────────────
    reports = sorted([
        f for f in args.reports_dir.iterdir()
        if f.suffix.lower() in {".pdf", ".txt"}
    ])
    if not reports:
        logger.error("No reports found in %s", args.reports_dir)
        sys.exit(1)

    logger.info("Found %d reports", len(reports))

    completed = sum(
        process_report(report_path=r, **shared_kwargs) for r in reports
    )
    failed = len(reports) - completed
    logger.info(
        "Done. completed=%d  failed=%d  total=%d",
        completed, failed, len(reports),
    )


if __name__ == "__main__":
    main()