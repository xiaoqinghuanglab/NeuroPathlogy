# Neuropathology Report Extractor

Structured data extraction from NACC neuropathology autopsy reports using local HuggingFace models with schema-driven prompting, self-correcting validation, and per-field extraction audit trails.

## What It Does

Given a neuropathology report (PDF or plain text), this tool:

1. Parses the report (auto-detects PDF vs text)
2. Generates a prompt from the Pydantic schema (so you never write prompts by hand)
3. Sends it to a locally-loaded LLM on your GPUs
4. Validates the output against the schema with Pydantic
5. If validation fails, feeds the errors back to the LLM and re-asks (up to N retries)
6. Writes a clean JSON file with extracted variables and per-field annotations

The key idea: **`schema.py` is the single source of truth** for the extraction schema. You define what to extract there — field names, types, constraints, enums, descriptions — and the pipeline auto-generates the LLM prompt, validates the output, and handles retries. No prompt editing needed.

## Project Structure

```
├── seeds_analysis/
│   └── analyze_seeds.py # Consistency analysis across multi-seed runs
├── main.py          # CLI entry point: load model, run extraction, write JSON
├── schema.py        # Pydantic schema: defines all 43 NACC fields, validators, descriptions
├── run_neuro.slurm  # SLURM job submission script
├── requirements.txt
└── README.md
```

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU(s) with sufficient VRAM
- HuggingFace model access (may require `hf auth login` for gated models)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Supported Models

| Alias | HuggingFace Model ID | Notes |
|---|---|---|
| `oss-20b` | `openai/gpt-oss-20b` | MoE, good balance of speed and quality; supports `--reasoning-effort` |
| `qwen2.5-14b` | `Qwen/Qwen2.5-14B-Instruct` | MoE, good balance of speed and quality |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | Dense, fastest inference |

You can also pass any full HuggingFace model ID directly (e.g., `-m mistralai/Mistral-7B-Instruct-v0.3`).

## Quick Start

### Basic usage (plain text report)

```bash
python main.py -i report.txt -m qwen2.5-14b --num-gpus 1
```

### PDF report

```bash
python main.py -i report.pdf -m oss-20b --num-gpus 2
```

### All options

```bash
python main.py \
  -i report.pdf \                   # input file (required)
  -o output \                       # path to write output JSON files
  -m oss-20b \                      # model alias or full HF ID (required)
  --seeds 0 1 2 3 4 \               # run extraction with multiple random seeds
  --num-gpus 1 \                    # number of GPUs (default: 1)
  --dtype bfloat16 \                # model dtype: auto, bfloat16, float16 (default: auto)
  --input-format pdf \              # force input format: text or pdf (default: auto-detect)
  --temperature 0.01 \              # sampling temperature (default: 0.01)
  --top-p 0.9 \                     # nucleus sampling top-p (default: 0.9)
  --reasoning-effort medium \       # reasoning effort for gpt-oss models: low/medium/high (default: medium)
  --max-new-tokens 16384 \          # max tokens to generate (default: 32768)
  --max-retries 3 \                 # validation retry attempts (default: 3)
  --verbose \                       # debug logging
```

## Running on HPC (SLURM)

The project includes a SLURM job script for batch processing reports on GPU clusters.

```bash
sbatch run_neuro.slurm
```

The script will:

1. Create/activate the project virtual environment
2. Install dependencies from requirements.txt
3. Load the Qwen2.5-14B model
4. Run extraction across all reports
5. Execute multiple seeds (default: 0–4)
6. Save outputs in:

```bash
output/qwen2.5-14b/
```

## Output Format

The output is a JSON file with two layers: the extracted NACC variables (nested by domain) and a `field_annotations` block that records the evidence and confidence for every extracted value.

```json
{
  "specimen_info": {
    "NPSEX": null,
    "NPFIX": 1,
    "NPWBRWT": 991.7,
    "NPWBRF": 1,
    "NPPMIH": 99.9,
    "NPFIXX": null
  },
  "gross_findings": {
    "NPGRLA": 1,
    "NPGRHA": 0,
    "NPGRSNH": 2,
    "NPGRLCH": 0,
    "NPGRCCA": 2,
    "NACCBRNN": 0
  },
  "vascular_pathology": {
    "NACCAVAS": 3,
    "NPLINF": 2,
    "NPLAC": 2,
    "NPHEM": 2,
    "NPWMR": 3,
    "NACCARTE": 3,
    "NACCVASC": 1,
    "NACCINF": 0,
    "NACCHEM": 0
  },
  "microscopic_findings": {
    "NPNLOSS": 3,
    "NPHIPSCL": 0,
    "NACCLEWY": 0,
    "NPLBOD": 0
  },
  "ad_pathology": {
    "NPTHAL": 4,
    "NACCBRAA": 1,
    "NACCNEUR": 0,
    "NPADNC": 0,
    "NACCDIFF": 3,
    "NACCAMY": 3
  },
  "diagnostic_codes": {
    "NACCCBD": 0,
    "NPPVASC": 2,
    "NPPAD": 1,
    "NPCAD": 2,
    "NPPLEWY": 2,
    "NPCLEWY": 2,
    "NPCVASC": 1,
    "NPPFTLD": 2,
    "NACCPROG": 0,
    "NACCPICK": 0,
    "NPFTDTDP": 1,
    "NACCPRIO": 0
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
    },
    "NACCDIFF": {
      "confidence": 1.0,
      "evidence": "Diffuse plaques. Severe:",
      "note": null
    }
  },
  "extraction_confidence": "high",
  "extraction_notes": null
}
```

Most variables must always contain a numeric code (0, 8, or 9). Only NPSEX, NPFIX, and NPFIXX may be `null` when not stated in the report. `field_annotations` contains an entry for every non-null field, and for ambiguous null fields where the absence was itself uncertain.

### Understanding `field_annotations`

Each entry has three keys:

| Key | Type | Meaning |
|---|---|---|
| `confidence` | float 0–1 | Certainty the value is correct. ≥0.8 = unambiguous; 0.6–0.8 = inferred; <0.6 = review recommended |
| `evidence` | string | Verbatim phrase from the report that drove the extraction decision |
| `note` | string or null | Reasoning note — only populated when the extraction required judgment (ambiguous language, severity mapping, conflicting statements). Null for straightforward extractions |

Use `confidence < 0.7` as a filter to identify fields needing human review.

## Multi-Seed Consistency Analysis

Running the same report with multiple random seeds tests extraction stability. Use `analyze_seeds.py` to compare outputs:

```bash
# Run 5 seeds
python main.py \
  -i report.pdf \
  -m qwen2.5-14b \
  --seeds 0 1 2 3 4 \
  -o output

# Compare
python seeds_analysis/analyze_seeds.py --output-dir output
```

High consistency (≥80% majority agreement) across seeds indicates reliable extraction. Low consistency on a variable, combined with low `confidence` in `field_annotations`, pinpoints where the report language is genuinely ambiguous.

## Customizing the Schema

All extraction fields live in `schema.py`. The pipeline reads the schema at runtime, so changes take effect immediately.

### Adding a new field

Add it to the relevant sub-model with a `Field(description=...)`:

```python
class SpecimenInfo(BaseModel):
    # ... existing fields ...
    NPBRNWT_FRESH: Optional[float] = Field(
        None,
        description="Fresh (pre-fixation) brain weight in grams. Range 100–2500."
    )
```

The `description` string is what the LLM sees. Make it specific: include the full code table, example report phrases, and any disambiguation notes.

### Supported field types

| Type | Example | Validation |
|---|---|---|
| **Integer** | `Field(None, description="...")` | Range via `ge`/`le`; enum via `@field_validator` |
| **Float** | `Field(None, ge=0.0, le=2500.0, ...)` | Range via `ge`/`le` |
| **Boolean** | `Field(None, description="...")` | True/False |
| **Enum** | `Optional[SeverityCode]` | Restricted to enum values; allowed list auto-rendered in prompt |
| **String** | `Field(None, description="...")` | Optional min/max length, regex pattern |
| **Dict** | `Optional[Dict[str, str]]` | Key-value pairs |
| **List** | `Optional[List[str]]` | List of items |
| **Cross-field** | `@model_validator(mode="after")` | Compare multiple fields; used for derived field consistency |

### Adding a new enum

```python
class FixationType(int, Enum):
    formalin = 1
    paraformaldehyde = 2
    other = 7

class SpecimenInfo(BaseModel):
    NPFIX: Optional[FixationType] = Field(None,
        description="Fixation method: 1=Formalin, 2=Paraformaldehyde, 7=Other")
```

The format instructions will automatically list the allowed integer values.

## How the Re-Ask Loop Works

```
report → [build prompt from schema] → LLM → [parse JSON] → [Pydantic validate]
                                        ↑                           |
                                        |     validation error      |
                                        +←——— [feed errors back] ←——+
```

If the LLM output fails JSON parsing or Pydantic validation, the errors are appended as a follow-up user message so the model can self-correct. This runs up to `--max-retries` times (default: 3). Derived field consistency is also enforced by Pydantic (e.g., `NACCHEM=0` while `NPHEM=1` will trigger a re-ask with the specific contradiction).

## Troubleshooting

**Out of memory**: Use a smaller model. The `oss-*` models are MoE architectures and are more memory-efficient than their parameter count suggests.

**Model download fails**: Run `hf auth login` and ensure you have access to the model. Some models (e.g., Llama) require accepting a license on the HF model page.

**PDF extraction is empty**: Try installing `pdfplumber` as a fallback (`pip install pdfplumber`). Some scanned PDFs may need OCR preprocessing — this tool handles text-based PDFs only.

**Validation keeps failing**: Check `--verbose` output. If the model consistently fails on a field, the description in `schema.py` may need more disambiguation. The `evidence` field in `field_annotations` in passing outputs will show what language the model is trying to map.

**`reasoning_effort` ignored**: This parameter is only consumed by `gpt-oss` models via the chat template. For other models (e.g., Llama) it is silently ignored.

## License

MIT