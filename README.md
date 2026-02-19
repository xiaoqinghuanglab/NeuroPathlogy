# Neuropathology Report Extractor

Structured data extraction from neuropathology reports using local HuggingFace models with schema-driven prompting and self-correcting validation.

## What It Does

Given a neuropathology report (PDF or plain text), this tool:

1. Parses the report (auto-detects PDF vs text)
2. Generates a prompt from the Pydantic schema (so you never write prompts by hand)
3. Sends it to a locally-loaded LLM on your GPUs
4. Validates the output against the schema with Pydantic
5. If validation fails, feeds the errors back to the LLM and re-asks (up to N retries)
6. Writes a clean JSON file

The key idea: **`schema.py` is the single source of truth.** You define what to extract there — field names, types, constraints, enums — and the pipeline auto-generates the LLM prompt, validates the output, and handles retries. No prompt editing needed.

## Project Structure

```
├── main.py       # CLI entry point: load model, run extraction, write JSON
├── schema.py     # Pydantic schema: defines all fields, types, validators
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
| `oss-120b` | `openai/gpt-oss-120b` | MoE, strongest extraction quality |
| `oss-20b` | `openai/gpt-oss-20b` | MoE, good balance of speed and quality |
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
  -i report.pdf \              # input file (required)
  -m oss-120b \                # model alias or full HF ID (required)
  --num-gpus 4 \               # number of GPUs (default: 1)
  --dtype bfloat16 \           # model dtype: auto, bfloat16, float16 (default: auto)
  --input-format pdf \         # force input format: text or pdf (default: auto-detect)
  --temperature 0.3 \          # sampling temperature (default: 0.3)
  --top-p 0.9 \                # nucleus sampling top-p (default: 0.9)
  --max-new-tokens 32768 \     # max tokens to generate (default: 32768)
  --max-retries 3 \            # validation retry attempts (default: 3)
  --verbose \                  # debug logging
  -o output.json               # output path (default: <input>.extracted.json)
```

## Output Format

The output is a JSON file matching the schema defined in `schema.py`. Example:

```json
{
  "patient": {
    "patient_name_last": "Smith",
    "patient_name_first": "John",
    "date_of_birth": "1958-03-12",
    "age": 67,
    "sex": "M",
    "clinical_history_summary": "Progressive headaches and left-sided weakness for 3 weeks."
  },
  "report": {
    "report_date": "2025-01-15",
    "report_time": "14:30",
    "accession_number": "S25-1234"
  },
  "specimen": {
    "specimen_type": "resection",
    "specimen_site": "right frontal lobe",
    "laterality": "right"
  },
  "histopathology": {
    "morphological_description": "Highly cellular glial neoplasm with nuclear atypia...",
    "mitotic_count": "12 per 10 HPF",
    "necrosis_present": true,
    "microvascular_proliferation": true,
    "invasion_pattern": "diffuse infiltration of adjacent cortex"
  },
  "molecular_markers": {
    "idh_status": "IDH-wildtype",
    "mgmt_promoter": "unmethylated",
    "one_p_19q": "intact",
    "egfr_amplification": "amplified",
    "tert_promoter": "mutant",
    "cdkn2a": "homozygously_deleted",
    "h3_status": "wildtype",
    "ki67_index": 30.0,
    "p53_expression": "strong diffuse nuclear",
    "atrx_expression": "retained",
    "other_markers": null
  },
  "diagnosis": {
    "integrated_diagnosis": "Glioblastoma, IDH-wildtype, WHO grade 4",
    "tumor_type": "Glioblastoma",
    "who_grade": "4",
    "histological_subtype": null,
    "additional_diagnoses": null
  },
  "extraction_confidence": "high",
  "extraction_notes": null
}
```

Fields the LLM cannot determine from the report will be `null`.

## Customizing the Schema

All extraction fields live in `schema.py`. The pipeline reads the schema at runtime, so changes take effect immediately — no prompt editing required.

### Adding a new field

Add it to the relevant sub-model with a `Field(description=...)`:

```python
class PatientDemographics(BaseModel):
    # ... existing fields ...
    mrn: Optional[str] = Field(None, min_length=4, max_length=20,
        description="Medical record number")
```

The `description` string is what the LLM sees in its instructions. Make it specific.

### Supported field types

`schema.py` includes commented-out examples for each type. Here's a summary:

| Type | Example | Validation |
|---|---|---|
| **String** | `Field(None, description="...")` | Optional min/max length, regex pattern |
| **Integer** | `Field(None, ge=0, le=120, ...)` | Range via `ge`/`le` |
| **Float** | `Field(None, ge=0.0, le=100.0, ...)` | Range via `ge`/`le` |
| **Boolean** | `Field(None, description="...")` | True/False |
| **Enum** | `Field(None, description="...")` | Restricted to enum values |
| **Date string** | `Field(None, description="...YYYY-MM-DD...")` | `@field_validator` |
| **Time string** | `Field(None, description="...HH:MM...")` | `@field_validator` |
| **Dict** | `Optional[Dict[str, str]]` | Key-value pairs |
| **List** | `Optional[List[str]]` | List of items |
| **Cross-field** | `@model_validator(mode="after")` | Compare multiple fields |

### Adding a new enum

Define it, then use it as a field type:

```python
class FixationType(str, Enum):
    formalin = "formalin"
    frozen = "frozen"
    other = "other"

class SpecimenInfo(BaseModel):
    fixation: Optional[FixationType] = Field(None,
        description="Fixation method used")
```

The format instructions will automatically list the allowed values.

### Adding a new sub-model

Define a new `BaseModel` subclass, then add it as a field on `NeuropathologyExtraction`:

```python
class SurgicalMargins(BaseModel):
    model_config = ConfigDict(extra="forbid")
    margin_status: Optional[str] = Field(None, description="Margin status (positive/negative)")
    closest_margin_mm: Optional[float] = Field(None, ge=0.0, description="Closest margin in mm")

class NeuropathologyExtraction(BaseModel):
    # ... existing fields ...
    margins: SurgicalMargins = Field(
        default_factory=SurgicalMargins, description="Surgical margin assessment")
```

## How the Re-Ask Loop Works

```
report → [build prompt from schema] → LLM → [parse JSON] → [Pydantic validate]
                                        ↑                           |
                                        |     validation error      |
                                        +←——— [feed errors back] ←——+
```

If the LLM output fails JSON parsing or Pydantic validation, the errors are appended as a follow-up user message so the model can self-correct. This runs up to `--max-retries` times (default: 3). The Pydantic error messages are specific enough (e.g., "age must be >= 0 and <= 120, got 200") that the model usually fixes them in one retry.

## Troubleshooting

**Out of memory**: Lower `--num-gpus` or use a smaller model. The `oss-*` models are MoE architectures and are more memory-efficient than their parameter count suggests.

**Model download fails**: Run `huggingface-cli login` and ensure you have access to the model. Some models (e.g., Llama) require accepting a license on the HF model page.

**PDF extraction is empty**: Try installing `pdfplumber` as a fallback (`pip install pdfplumber`). Some scanned PDFs may need OCR preprocessing — this tool handles text-based PDFs only.

**Validation keeps failing**: Check `--verbose` output. If the model consistently fails on a field, consider relaxing the constraint in `schema.py` or providing a more specific `description`.

## License

MIT
