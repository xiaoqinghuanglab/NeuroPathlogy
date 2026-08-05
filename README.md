# Neuropathology Report Extractor

Structured data extraction from NACC neuropathology autopsy reports,
using open-weight LLMs served on local GPU infrastructure, with
schema-driven prompting, self-correcting validation, and per-field
extraction audit trails. All inference runs on local compute, no report
text is sent to any external API.

This repository accompanies the manuscript's methodology and contains the
full extraction pipeline: primary NACC-variable extraction, residual
(non-NACC) variable discovery, and the downstream ADNC classification
analysis.

## What It Does

Given a neuropathology autopsy report (PDF or plain text), the pipeline:

1. Parses the report (auto-detects PDF vs text)
2. Builds the extraction prompt directly from a Pydantic schema class.
   Each field, its type, and its description are defined once in the
   schema, and the pipeline turns that definition into the corresponding
   instruction for the LLM automatically. Adding or changing a field in
   the schema updates the prompt with it, so nobody edits prompt text by
   hand.
3. Sends the prompt to an LLM running on a local vLLM server
4. Validates the response against the schema with Pydantic
5. If validation fails, feeds the errors back to the LLM and re-asks
   (up to N retries)
6. Writes a clean JSON file with extracted variables and per-field
   annotations (evidence, confidence, reasoning notes)

There are two extraction pipelines in this repo:

- **Primary extraction**, the 199 official NACC neuropathology variables,
  extracted across 7 schema-driven passes to keep prompts within context
  limits while preserving cross-field dependency integrity.
- **Residual extraction**, 53 additional variables not covered by the
  NACC dictionary, surfaced through an LLM-driven discovery pass,
  embedding/clustering, and human-curated schema review, then extracted
  per report.

Production runs use **`openai/gpt-oss-20b`** exclusively. `Llama-3.1-8B`
and `Qwen2.5-14B` were used only during an earlier 4-report ground-truth
pilot to compare candidate models before selecting OSS-20B for the full
161-report run.

## Project Structure

```
NeuroPathlogy/
├── config/                        # vLLM serving configs
│   ├── GPT-OSS_Hopper.yaml
│   └── Llama_Qwen_Hopper.yaml
├── data/
│   └── rdd-np.csv                 # NACC data dictionary (variable reference)
├── src/
│   ├── primary_extraction/
│   │   ├── main.py                # CLI entry point: 7-pass NACC extraction
│   │   └── schema1a.py ... schema6.py   # Pydantic schema, one per pass
│   └── residual_extraction/
│       ├── discovery.py           # Stage 1A, free-text finding discovery
│       ├── cluster.py             # Stage 1B, embed, cluster, LLM-label
│       └── extract.py             # Stage 2, per-report residual extraction
├── slurm/
│   ├── run_neuro_vllm.slurm       # primary extraction batch job
│   ├── run_neuro_discovery.slurm  # residual Stage 1A batch job
│   ├── run_neuro_cluster.slurm    # residual Stage 1B batch job
│   └── run_neuro_residual.slurm   # residual Stage 2 batch job
├── analysis/
│   ├── analyze_seeds.py                # multi-seed consistency analysis
│   ├── analyze_residual.py             # figures from residual JSON outputs
│   ├── cohort_summary.py               # cohort stats summary
│   ├── cohort_by_adnc.py               # cohort stats, stratified by ADNC
│   ├── compute_confidence_metrics.py   # builds confidence_metrics.csv
│   ├── visualize_confidence_metrics.py # plots from confidence_metrics.csv
│   ├── extract_variable_matrix.py      # builds master variable matrix (xlsx)
│   ├── build_table_s1.py               # builds manuscript Table S1
│   └── compare_models_gt.py            # GT seed-consistency comparison across models
├── notebooks/
│   ├── eda_preprocessing.ipynb
│   └── model_development.ipynb         # ADNC classification (LR / RF / XGBoost)
├── requirements.txt
└── README.md
```

> **Note on paths:** input reports, per-report model outputs, and other
> generated artifacts (`reports/`, `output/`, `results/`, caches, logs,
> containers) live outside this repository on the compute cluster and are
> not tracked in git, see `.gitignore`. `data/rdd-np.csv` (the NACC
> variable dictionary) is the one reference dataset tracked here.

## Setup

### Requirements

- Python 3.12
- CUDA-capable GPU(s). Pipeline was run on H100 GPUs
- vLLM, served via Apptainer container (see `slurm/` scripts for the
  serving setup)
- Access to `openai/gpt-oss-20b` (and, for the pilot comparison,
  `meta-llama/Llama-3.1-8B-Instruct` and `Qwen/Qwen2.5-14B-Instruct`)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Pipeline

All scripts connect to a vLLM server running locally on the same GPU
node, rather than loading a model directly in-process or calling an
external API. The `slurm/` scripts each start that local vLLM server,
wait for it to be healthy, run the corresponding stage, then shut the
server down. See each `.slurm` file for the exact sequence and Apptainer
bind paths.

### 1. Primary extraction (199 NACC variables)

```bash
python src/primary_extraction/main.py \
    --input report.pdf \
    --output-dir output/ \
    --model oss-20b \
    --seeds 0 1 2 3 4 \
    --vllm-url http://localhost:PORT \
    --temperature 0.01 \
    --top-p 0.9 \
    --reasoning-effort medium \
    --max-new-tokens 16384 \
    --max-retries 3 \
    --logprobs
```

Batch version: `sbatch slurm/run_neuro_vllm.slurm`

### 2. Residual extraction (53 additional variables)

Three stages, run in order:

**Stage 1A, Discovery.** Reads every report once (single seed since
breadth matters here, not consistency) and flags clinically significant
findings not covered by the 199 NACC variables.

```bash
python src/residual_extraction/discovery.py \
    --input report.pdf \
    --output-dir output/residual_discovery \
    --rddnp-path data/rdd-np.csv \
    --model openai/gpt-oss-20b \
    --vllm-url http://localhost:PORT \
    --max-new-tokens 32768
```

Batch version: `sbatch slurm/run_neuro_discovery.slurm`

**Stage 1B, Embed, cluster, label.** Pools every discovered observation
across all reports, embeds with PubMedBERT, reduces dimensionality with
UMAP, clusters with HDBSCAN, then has the LLM label each cluster into a
candidate variable definition. Produces a draft schema plus a human-review
spreadsheet (KEEP / DROP / EDIT decisions).

```bash
python src/residual_extraction/cluster.py \
    --discovery-dir output/residual_discovery \
    --output-dir output/residual_schema \
    --rddnp-path data/rdd-np.csv \
    --vllm-url http://localhost:PORT \
    --model openai/gpt-oss-20b \
    --embed-device cuda
```

Batch version: `sbatch slurm/run_neuro_cluster.slurm`

**Stage 1C, Human review** (not a script). The curated KEEP/DROP/EDIT
spreadsheet from Stage 1B becomes the final residual variable schema.

**Stage 2, Per-report residual extraction**, using the human-curated
schema:

```bash
python src/residual_extraction/extract.py \
    --input report.pdf \
    --output-dir output/residual_extraction \
    --model openai/gpt-oss-20b \
    --seeds 0 1 2 3 4 \
    --vllm-url http://localhost:PORT \
    --max-new-tokens 16384
```

Batch version: `sbatch slurm/run_neuro_residual.slurm`

## Output Format

Each stage writes JSON with the extracted variables plus a
`field_annotations` block recording the evidence and confidence behind
every extracted value:

```json
{
  "specimen_info": {
    "NPSEX": null,
    "NPFIX": 1,
    "NPWBRWT": 991.7
  },
  "ad_pathology": {
    "NPTHAL": 4,
    "NACCBRAA": 1,
    "NACCNEUR": 0,
    "NPADNC": 0
  },
  "field_annotations": {
    "NPWBRWT": {
      "confidence": 1.0,
      "evidence": "The fresh brain weighs 991.7 grams.",
      "note": null
    },
    "NPADNC": {
      "confidence": 0.8,
      "evidence": "low likelihood",
      "note": "Composite ADNC interpreted as minimal due to CERAD 0."
    }
  },
  "extraction_confidence": "high",
  "extraction_notes": null
}
```

### Understanding `field_annotations`

| Key | Type | Meaning |
|---|---|---|
| `confidence` | float 0–1 | Certainty the value is correct. ≥0.8 = unambiguous; 0.6–0.8 = inferred; <0.6 = review recommended |
| `evidence` | string | Verbatim phrase from the report that drove the extraction decision |
| `note` | string or null | Reasoning note (only populated when extraction required judgment) |

Use `confidence < 0.7` as a filter to identify fields needing human review.

## Multi-Seed Consistency Analysis

Each report is run with 5 seeds per stage to test extraction stability.

```bash
python analysis/analyze_seeds.py
```

(Edit the `PIPELINE` switch at the top of the script to choose
`"primary"` or `"residual"`.)

High consistency (≥80% majority agreement) across seeds indicates reliable
extraction. Low consistency on a variable, combined with low `confidence`
in `field_annotations`, pinpoints where the report language is genuinely
ambiguous.

## Customizing the Extraction Schema

Primary-extraction fields live in `src/primary_extraction/schema1a.py`
through `schema6.py`, one file per pass. Each pass's schema is a Pydantic
model; the pipeline reads it at runtime and auto-generates the
corresponding LLM prompt, so schema edits take effect immediately without
touching prompt text.

### Adding a new field

Add it to the relevant sub-model in the corresponding pass file, with a
`Field(description=...)`:

```python
class SpecimenInfo(BaseModel):
    # ... existing fields ...
    NPBRNWT_FRESH: Optional[float] = Field(
        None,
        description="Fresh (pre-fixation) brain weight in grams. Range 100–2500."
    )
```

The `description` string is what the LLM sees. Make it specific: include
the full code table, example report phrases, and any disambiguation
notes.

### Supported field types

| Type | Example | Validation |
|---|---|---|
| **Integer** | `Field(None, description="...")` | Range via `ge`/`le`; enum via `@field_validator` |
| **Float** | `Field(None, ge=0.0, le=2500.0, ...)` | Range via `ge`/`le` |
| **Boolean** | `Field(None, description="...")` | True/False |
| **Enum** | `Optional[SeverityCode]` | Restricted to enum values; allowed list auto-rendered in prompt |
| **String** | `Field(None, description="...")` | Optional min/max length, regex pattern |
| **Cross-field** | `@model_validator(mode="after")` | Compare multiple fields; used for derived-field consistency |

## How the Re-Ask Loop Works

```
report → [build prompt from schema] → LLM → [parse JSON] → [Pydantic validate]
                                        ↑                           |
                                        |     validation error      |
                                        +←——— [feed errors back] ←——+
```

If the LLM output fails JSON parsing or Pydantic validation, the errors
are appended as a follow-up message so the model can self-correct, up to
`--max-retries` times (default: 3). Derived-field consistency is also
enforced by Pydantic, e.g. `NACCHEM=0` while `NPHEM=1` triggers a re-ask
with the specific contradiction.

## Downstream Analysis

- **`analysis/`**: consistency checks, cohort summaries, confidence
  metrics, and the variable matrix / Table S1 generation used in the
  manuscript.
- **`notebooks/model_development.ipynb`**: ADNC classification
  (Logistic Regression, Random Forest, XGBoost) using primary-only and
  primary+residual feature sets, target `NPADNC_bin`.

## Troubleshooting

**Validation keeps failing**: check the model's response with
`--verbose`/debug logging. If a field consistently fails, its
`description` in the relevant `schema*.py` file may need more
disambiguation. The `evidence` field in passing outputs shows what
language the model is trying to map.

**`reasoning_effort` ignored**: this parameter is only consumed by
`gpt-oss` models via the chat template; for other models it is silently
ignored.

**PDF extraction is empty**: some scanned PDFs may need OCR preprocessing
This tool handles text-based PDFs only.

## License

MIT