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

The key idea: **`schema_full.py` is the single source of truth.** You define what to extract there — field names, types, constraints, enums, descriptions — and the pipeline auto-generates the LLM prompt, validates the output, and handles retries. No prompt editing needed.

## Project Structure

```
├── main.py          # CLI entry point: load model, run extraction, write JSON
├── schema_full.py   # Pydantic schema: defines all 40 NACC fields, validators, descriptions
├── analyze_seeds.py # Consistency analysis across multi-seed runs
└── README.md
```

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU(s) with sufficient VRAM
- HuggingFace model access (may require `huggingface-cli login` for gated models)

### Install Dependencies

```bash
pip install torch transformers accelerate pydantic

# for PDF support (install at least one):
pip install pymupdf       # recommended, fast C backend
# or
pip install pdfplumber    # pure-python alternative
```

## Supported Models

| Alias | HuggingFace Model ID | Notes |
|---|---|---|
| `oss-120b` | `openai/gpt-oss-120b` | MoE, strongest extraction quality; supports `--reasoning-effort` |
| `oss-20b` | `openai/gpt-oss-20b` | MoE, good balance of speed and quality; supports `--reasoning-effort` |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | Dense, fastest inference |

You can also pass any full HuggingFace model ID directly (e.g., `-m mistralai/Mistral-7B-Instruct-v0.3`).

## Quick Start

### Basic usage (plain text report)

```bash
python main.py -i report.txt -m oss-120b --num-gpus 4
```

### PDF report

```bash
python main.py -i report.pdf -m oss-20b --num-gpus 2
```

### Specify output path

```bash
python main.py -i report.pdf -m llama3.1-8b -o results/patient_001.json
```

### All options

```bash
python main.py \
  -i report.pdf \                   # input file (required)
  -m oss-120b \                     # model alias or full HF ID (required)
  --num-gpus 4 \                    # number of GPUs (default: 1)
  --dtype bfloat16 \                # model dtype: auto, bfloat16, float16 (default: auto)
  --input-format pdf \              # force input format: text or pdf (default: auto-detect)
  --temperature 0.01 \              # sampling temperature (default: 0.01)
  --top-p 0.9 \                     # nucleus sampling top-p (default: 0.9)
  --reasoning-effort medium \       # reasoning effort for gpt-oss models: low/medium/high (default: medium)
  --max-new-tokens 32768 \          # max tokens to generate (default: 32768)
  --max-retries 3 \                 # validation retry attempts (default: 3)
  --seed 42 \                       # random seed for reproducibility
  --verbose \                       # debug logging
  -o output.json                    # output path (default: <input>.extracted.json)
```

## Output Format

The output is a JSON file with two layers: the extracted NACC variables (nested by domain) and a `field_annotations` block that records the evidence and confidence for every extracted value.

```json
{
  "specimen_info": {
    "NPSEX": 1,
    "NPFIX": 1,
    "NPWBRWT": 1219.2,
    "NPWBRF": 1,
    "NPPMIH": null,
    "NPFIXX": null
  },
  "gross_findings": {
    "NPGRLA": 1,
    "NPGRHA": 2,
    "NPGRSNH": 1,
    "NPGRLCH": 1,
    "NPGRCCA": null,
    "NACCBRNN": 0
  },
  "vascular_pathology": {
    "NACCAVAS": 2,
    "NPLINF": 2,
    "NPLAC": 2,
    "NPHEM": 2,
    "NPWMR": 1,
    "NACCARTE": 3,
    "NACCVASC": 1,
    "NACCINF": 0,
    "NACCHEM": 0
  },
  "microscopic_findings": {
    "NPNLOSS": 1,
    "NPHIPSCL": null,
    "NACCLEWY": 0,
    "NPLBOD": 0
  },
  "ad_pathology": {
    "NPTHAL": null,
    "NACCBRAA": null,
    "NACCNEUR": null,
    "NPADNC": null,
    "NACCDIFF": null,
    "NACCAMY": null
  },
  "diagnostic_codes": {
    "NACCCBD": 0,
    "NPPVASC": 1,
    "NPPAD": 2,
    "NPCAD": 2,
    "NPPLEWY": 2,
    "NPCLEWY": 2,
    "NPCVASC": 1,
    "NPPFTLD": 2,
    "NACCPROG": 0,
    "NACCPICK": 0,
    "NPFTDTDP": 0,
    "NACCPRIO": 0
  },
  "field_annotations": {
    "NPSEX": {
      "confidence": 1.0,
      "evidence": "71-year-old male",
      "note": null
    },
    "NPWBRWT": {
      "confidence": 1.0,
      "evidence": "Brain weight: 1219.2 g (fixed)",
      "note": null
    },
    "NACCAVAS": {
      "confidence": 0.8,
      "evidence": "moderate atherosclerosis of the circle of Willis",
      "note": null
    },
    "NPGRSNH": {
      "confidence": 0.6,
      "evidence": "mild pallor of the substantia nigra",
      "note": "Report says 'mild pallor' — coded as 1 (Mild). Some residual pigmentation visible."
    }
  },
  "extraction_confidence": "moderate",
  "extraction_notes": null
}
```

Fields the LLM cannot determine from the report are `null`. `field_annotations` contains an entry for every non-null field, and for ambiguous null fields where the absence was itself uncertain.

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
for seed in 42 123 456 789 1011; do
    python main.py -i report.pdf -m oss-20b \
        --seed $seed -o output/report_seed${seed}.extracted.json
done

# Compare
python analyze_seeds.py --output-dir output
```

High consistency (≥80% majority agreement) across seeds indicates reliable extraction. Low consistency on a variable, combined with low `confidence` in `field_annotations`, pinpoints where the report language is genuinely ambiguous.

## Customizing the Schema

All extraction fields live in `schema_full.py`. The pipeline reads the schema at runtime, so changes take effect immediately.

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

**Out of memory**: Lower `--num-gpus` or use a smaller model. The `oss-*` models are MoE architectures and are more memory-efficient than their parameter count suggests.

**Model download fails**: Run `huggingface-cli login` and ensure you have access to the model. Some models (e.g., Llama) require accepting a license on the HF model page.

**PDF extraction is empty**: Try installing `pdfplumber` as a fallback (`pip install pdfplumber`). Some scanned PDFs may need OCR preprocessing — this tool handles text-based PDFs only.

**Validation keeps failing**: Check `--verbose` output. If the model consistently fails on a field, the description in `schema_full.py` may need more disambiguation. The `evidence` field in `field_annotations` in passing outputs will show what language the model is trying to map.

**`reasoning_effort` ignored**: This parameter is only consumed by `gpt-oss` models via the chat template. For other models (e.g., Llama) it is silently ignored.

## License

MIT
