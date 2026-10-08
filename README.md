# Neuropathology Report Extractor

Structured data extraction from NACC neuropathology autopsy reports,
using open-weight LLMs served on local GPU infrastructure, with
schema-driven prompting, self-correcting validation, and per-field
extraction audit trails. All inference runs on local compute, no report
text is sent to any external API.

This repository accompanies the manuscript's methodology and contains the
full analysis workflow: primary NACC-variable extraction, novel
(non-NACC) variable discovery and extraction, and the downstream ADNC
classification analysis.

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
5. If validation fails, feeds the errors back to the LLM and
   re-asks within the configured attempt limit.
6. Writes a clean JSON file with extracted variables and per-field
   annotations (evidence, confidence, reasoning notes)

There are two extraction pipelines in this repo:

- **Primary extraction**, the 199 official NACC neuropathology variables,
  extracted across 7 schema-driven passes to reduce prompt and output complexity,
  group related neuropathology variables together, and improve schema adherence.
- **Novel extraction**, 53 additional variables not covered by the
  NACC dictionary. These are identified through LLM-based discovery,
  embedding/clustering, human schema review, and subsequent per-report
  extraction.

Production runs use **`openai/gpt-oss-20b`** for the manuscript analyses. `Llama-3.1-8B` and `Qwen2.5-14B` were also evaluated as candidate models by comparing their extracted NACC variables with NACC-provided reference values across the 161-report cohort. OSS-20B achieved the highest overall extraction accuracy and was selected for subsequent novel-variable extraction and downstream analyses.

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
│   └── residual_extraction/       # Novel variables extraction
│       ├── discovery.py           # Stage 1A, free-text finding discovery
│       ├── cluster.py             # Stage 1B, embed, cluster, LLM-label
│       └── extract.py             # Stage 2, per-report novel extraction
├── slurm/
│   ├── run_neuro_vllm.slurm       # primary extraction batch job
│   ├── run_neuro_discovery.slurm  # novel Stage 1A batch job
│   ├── run_neuro_cluster.slurm    # novel Stage 1B batch job
│   └── run_neuro_residual.slurm   # novel Stage 2 batch job
├── analysis/
│   ├── analyze_seeds.py                # multi-seed consistency analysis
│   ├── analyze_residual.py             # figures from novel JSON outputs
│   ├── cohort_summary.py               # cohort stats summary
│   ├── cohort_by_adnc.py               # cohort stats, stratified by ADNC
│   ├── compute_confidence_metrics.py   # builds confidence_metrics.csv
│   ├── visualize_confidence_metrics.py # plots from confidence_metrics.csv
│   ├── extract_variable_matrix.py      # builds master variable matrix (xlsx)
│   └── compare_models_gt.py            # compares model extractions with NACC reference values
├── notebooks/
│   ├── eda_preprocessing.ipynb
│   └── model_development.ipynb         # ADNC classification (LR / RF / XGBoost)
├── requirements.txt
└── README.md
```

> **Note on paths:** input reports, per-report model outputs, and other
> generated artifacts (`reports/`, `output/`, `results/`, caches, logs,
> containers) live outside this repository on the compute cluster and are
> not tracked in Git, see `.gitignore`. `data/rdd-np.csv` (the NACC
> variable dictionary) is the one reference dataset tracked here.

## Setup

### Requirements

- Python 3.12
- CUDA-capable GPU(s). Pipeline was run on H100 GPUs
- vLLM, served via Apptainer container (see `slurm/` scripts for the
  serving setup)
- Access to `openai/gpt-oss-20b`. Reproducing the candidate-model comparison
  additionally requires `meta-llama/Llama-3.1-8B-Instruct` and
  `Qwen/Qwen2.5-14B-Instruct`.

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Pipeline

The extraction and discovery scripts connect to a vLLM server running locally
on the same GPU node, rather than loading a model directly in-process or
calling an external API. The `slurm/` scripts start that local vLLM server,
wait for it to be healthy, run the corresponding stage, then shut the server
down. See each `.slurm` file for the exact sequence and Apptainer bind paths.

The direct Python commands assume that a local vLLM server is already running.
For normal cluster execution, use the corresponding SLURM script, which starts
the server, runs the stage, and shuts it down.

### Run Modes

The primary extraction SLURM script supports two execution modes controlled
by `RUN_MODE`:

- **`RUN_MODE="gt"`**: 4-report validation run using `reports_gt/` and
  `output_gt/<model>/`. This mode is used to test model, schema, and pipeline
  changes before running the full cohort.
- **`RUN_MODE="prod"`**: full production extraction using `reports/` and
  `output/<model>/`. Production reports are distributed across SLURM array
  tasks, with each array task processing a fixed-size chunk of reports.

Set the desired mode in `slurm/run_neuro_vllm.slurm`:

```bash
RUN_MODE="gt"
```

or

```bash
RUN_MODE="prod"
```

For GT validation, submit a standard SLURM job:

```bash
sbatch slurm/run_neuro_vllm.slurm
```

For production, submit the script as a SLURM array:

```bash
sbatch --array=0-8 slurm/run_neuro_vllm.slurm
```

The production array range depends on the number of reports and the
`CHUNK_SIZE` configured in the SLURM script. The current configuration uses
`CHUNK_SIZE=4` for GT mode and `CHUNK_SIZE=18` for production.

Each SLURM task starts a local vLLM server on its allocated GPU node, processes
its assigned reports across 5 seeds, writes the extracted JSON outputs, and
then shuts down the server.

Recommended workflow:

`change → GT validation → inspect outputs/logs → production array run`

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

#### Resume behavior

Primary extraction is resumable at both the seed and pass level.

- `report_seed0.extracted.json` means that seed is complete and will be skipped.
- `report_seed0_pass3.json` means that pass completed successfully and can be
  loaded when the run resumes.
- `report_seed0_pass3.failed` means that pass exhausted the configured
  extraction attempts. On rerun, intermediate files for that seed are cleared
  and the seed is rerun cleanly.

This allows interrupted or partially failed jobs to be rerun without
reprocessing already completed seeds.

The seven passes are `schema1a`, `schema1b`, `schema2`, `schema3`,
`schema4`, `schema5`, and `schema6`; `schema1a` and `schema1b` are
separate passes even though their filenames share the `schema1` prefix.

### 2. Novel extraction (53 additional variables)

The novel-variable workflow has four stages, run in order:

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
across all reports. PubMedBERT converts each free-text finding into a
biomedical semantic embedding so that differently worded findings
describing similar neuropathology can be compared numerically. UMAP
reduces the embedding dimensionality, HDBSCAN groups semantically similar
findings, and the LLM converts each resulting cluster into a candidate
variable definition.

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

**Stage 1C, Review** (not a script). The curated KEEP/DROP/EDIT
spreadsheet from Stage 1B becomes the final novel-variable schema.

**Stage 2, Per-report novel extraction**, using the human-curated
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

The extraction stages write JSON with the extracted variables plus a
`field_annotations` block recording the evidence and confidence behind
each extracted value. Discovery and clustering stages additionally
produce candidate findings, schema definitions, and human-review outputs.
The example below shows an extraction-stage output.

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

Model-reported confidence can be considered together with extraction evidence and
multi-seed agreement when identifying fields that may warrant additional review.

## Multi-Seed Consistency Analysis

Primary NACC extraction and final novel-variable extraction run each
report with 5 seeds to assess extraction stability. Novel discovery
Stage 1A uses a single run because its purpose is broad candidate
discovery rather than final structured extraction.

```bash
python analysis/analyze_seeds.py
```

(Edit the `PIPELINE` switch at the top of the script to choose
`"primary"` or `"residual"`.)

Multi-seed agreement provides an empirical measure of extraction stability.
Lower agreement, particularly when accompanied by lower model-reported confidence,
can be used to identify variables or reports that warrant closer inspection.

## Preparing Analysis-Ready Variable Matrices

Before running the downstream notebooks, convert the per-report extraction
outputs into report-by-variable matrices using
`analysis/extract_variable_matrix.py`.

The script combines the 5 seed-level outputs for each report using majority
vote and can be run in two modes:

- **`MODE="primary"`**: processes the 199 NACC-variable extraction outputs
  (`*.extracted.json`).
- **`MODE="residual"`**: processes the novel-variable extraction outputs
  (`*.residual.json`), including the 49 LLM-extracted variables and 4
  programmatically derived weight-difference variables.

Set the desired mode near the top of
`analysis/extract_variable_matrix.py`:

```python
MODE = "primary"
MODEL = "oss-20b"
```

Run:

```bash
python analysis/extract_variable_matrix.py
```

This produces:

```text
results/variable_matrix_oss-20b_primary.xlsx
```

Then switch to:

```python
MODE = "residual"
MODEL = "oss-20b"
```

and run the same command again:

```bash
python analysis/extract_variable_matrix.py
```

This produces:

```text
results/variable_matrix_oss-20b_residual.xlsx
```

Each workbook contains three sheets:

- **Variable Summary**: per-variable availability across reports.
- **Binary Matrix**: report × variable matrix indicating whether each
  variable contains an informative value.
- **Raw Values**: majority-voted raw value for each report and variable.

The primary and novel variable-matrix workbooks are then used by
`notebooks/eda_preprocessing.ipynb` to construct the merged analysis dataset
used for downstream ADNC modeling.

The preprocessing sequence is therefore:

```text
5-seed extraction outputs
        ↓
extract_variable_matrix.py
        ↓
primary + novel variable matrices
        ↓
eda_preprocessing.ipynb
        ↓
analysis-ready merged dataset
        ↓
model_development.ipynb
```

Run the notebooks in this order:

```text
1. notebooks/eda_preprocessing.ipynb
2. notebooks/model_development.ipynb
```


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
| **Cross-field** | `@model_validator(mode="after")` | Compare multiple fields; enforce consistency and conditional rules |

## How the Re-Ask Loop Works

```
report → [build prompt from schema] → LLM → [parse JSON] → [Pydantic validate]
                                        ↑                           |
                                        |     validation error      |
                                        +←——— [feed errors back] ←——+
```

If the LLM output fails JSON parsing or Pydantic validation, the errors
are appended as a follow-up message so the model can self-correct, up to
`--max-retries` total generation attempts (default: 3).
Cross-field consistency is also enforced by Pydantic. For example,
`NACCHEM` must be consistent with the extracted `NPHEMO` and `NPOLDD`
parent values. An inconsistent combination triggers validation failure
and a corrected re-ask.

## Downstream Analysis

- **`analysis/`**: consistency checks, cohort summaries, confidence
  metrics, variable-matrix generation, and Table S1 generation used in
  the manuscript.
- **`notebooks/eda_preprocessing.ipynb`**: prepares the merged
  analysis-ready dataset from the primary and novel variable matrices.
- **`notebooks/model_development.ipynb`**: performs downstream ADNC
  classification using the dataset produced by the EDA/preprocessing notebook.

## Reproducing the Manuscript Workflow

At a high level, the manuscript analysis can be reproduced in the following order:

1. Run primary extraction for the 199 NACC variables.
2. To reproduce the model-selection analysis, compare `OSS-20B`,
   `Qwen2.5-14B`, and `Llama3.1-8B` extractions against the NACC-provided
   reference values; `OSS-20B` was selected for subsequent analyses.
3. Run novel Stage 1A to discover findings beyond the NACC schema.
4. Run novel Stage 1B to embed, cluster, and generate candidate variable
   definitions.
5. Manually review the candidate definitions and finalize the novel schema.
6. Run novel Stage 2 to extract the finalized novel variables across reports.
7. Run `analysis/extract_variable_matrix.py` in both `primary` and `residual`
   modes to generate the majority-voted variable matrices.
8. Run `notebooks/eda_preprocessing.ipynb` to construct the merged
   analysis-ready dataset.
9. Run the remaining analysis scripts as needed to generate consistency
   metrics, cohort summaries, confidence metrics, and manuscript outputs.
10. Run `notebooks/model_development.ipynb` to compare Existing NACC versus
    Combined NACC + novel predictors for binary ADNC classification.

NACC-provided reference values used for model evaluation are stored separately
on the protected cluster and are not tracked in Git. See
`analysis/compare_models_gt.py` for the configured input path.

## Troubleshooting

**Validation keeps failing**: check the model's response with
`--verbose`/debug logging. If a field consistently fails, its
`description` in the relevant `schema*.py` file may need more
disambiguation. The `evidence` field in passing outputs shows what
language the model is trying to map.

**`reasoning_effort` ignored**: this parameter is only consumed by
`gpt-oss` models via the chat template; for other models it is silently
ignored.

**PDF extraction is empty**: some scanned PDFs may require OCR preprocessing.
The current pipeline handles text-based PDFs and does not perform OCR.

## License

MIT