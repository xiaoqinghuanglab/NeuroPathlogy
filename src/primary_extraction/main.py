"""main: single-pass structured extraction for neuropathology reports.

Loads a HuggingFace model locally onto GPUs, reads a report (PDF or text),
extracts structured data via schema-driven prompting, validates with Pydantic,
and re-asks on validation failure.

Supported models:
    oss-20b     -> openai/gpt-oss-20b
    llama3.1-8b -> meta-llama/Llama-3.1-8B-Instruct
    qwen2.5-14b -> Qwen/Qwen2.5-14B-Instruct

Usage:
    python main.py -i report.pdf --vllm-url http://localhost:8000
    python main.py -i report.pdf --vllm-url http://localhost:8000 -m oss-20b
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import re
import math
from pathlib import Path
from typing import Optional, Type, get_args
from json_repair import repair_json
import inspect

from openai import OpenAI
from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema1a import build_pass1a_format_instructions, get_pass1a_model
from schema1b import build_pass1b_format_instructions, get_pass1b_model
from schema2 import build_pass2_format_instructions, get_pass2_model
from schema3 import build_pass3_format_instructions, get_pass3_model
from schema4 import build_pass4_format_instructions, get_pass4_model
from schema5 import build_pass5_format_instructions, get_pass5_model
from schema6 import build_pass6_format_instructions, get_pass6_model

logger = logging.getLogger("medace")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, str] = {
    "oss-120b": "openai/gpt-oss-120b",
    "oss-20b": "openai/gpt-oss-20b",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
}

SUPPORTED_ALIASES = list(MODEL_ALIASES.keys())

# Pass registry — maps pass number to (model getter, format instructions builder)
PASS_REGISTRY = {
    1: (get_pass1a_model, build_pass1a_format_instructions),
    2: (get_pass1b_model, build_pass1b_format_instructions),
    3: (get_pass2_model, build_pass2_format_instructions),
    4: (get_pass3_model, build_pass3_format_instructions),
    5: (get_pass4_model, build_pass4_format_instructions),
    6: (get_pass5_model, build_pass5_format_instructions),
    7: (get_pass6_model, build_pass6_format_instructions),
}

DATA_BLOCKS = {
    "specimen_info", "gross_and_vascular", "microscopic_findings",
    "lewy_pathology", "ad_pathology", "primary_contrib_dx",
    "ftld_tau", "tdp_flag", "prion_flag", "other_ftld", "ftld",
    "tdp43_distribution", "infarct_parent", "infarct_counts",
    "infarct_sizes", "hemorrhage", "microbleeds", "microinfarcts",
    "other_vascular", "misc_vascular", "normal_and_insuff_ad",
    "hippocampal_and_prion_dx", "other_dx_pairs", "write_in_dx",
    "other_disease_flags", "criteria_and_misc", "case_metadata",
    "antibody_methods", "histochem_stains", "tissue_banking",
    "full_autopsy", "family_history", "genetics"
}

ANNOTATION_ONLY_KEYS = {
    "value",
    "field_annotations",
    "annotation", 
    "annotations",       
    "note",
    "evidence",          
    "confidence",        
    "extraction_confidence",
    "extraction_notes"
}

def resolve_model(alias: str) -> str:
    """Map short alias -> full HF model ID.  Pass-through if already full."""
    return MODEL_ALIASES.get(alias, alias)


# ---------------------------------------------------------------------------
# Input loading: PDF or plaintext
# ---------------------------------------------------------------------------


def load_report(path: Path, fmt: Optional[str] = None) -> str:
    """Load report text.  Auto-detects PDF by extension unless overridden."""
    is_pdf = (fmt == "pdf") if fmt else (path.suffix.lower() == ".pdf")
    if is_pdf:
        return read_pdf(path)
    return path.read_text(encoding="utf-8")


def read_pdf(path: Path) -> str:
    """Extract text from PDF.  pymupdf (fast) -> pdfplumber (fallback)."""
    for mod_name in ("pymupdf", "fitz"):
        try:
            import importlib

            mod = importlib.import_module(mod_name)
            doc = mod.open(str(path))
            pages = [page.get_text() for page in doc]
            doc.close()
            text = "\n\n".join(pages)
            if text.strip():
                return text
        except ImportError:
            continue

    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n\n".join(pages)
        if text.strip():
            return text
    except ImportError:
        pass

    raise ImportError(
        "No PDF library found.  Install one of:\n"
        "  pip install pymupdf        # recommended\n"
        "  pip install pdfplumber      # pure-python fallback"
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(vllm_url: str) -> OpenAI:
    """Create OpenAI client pointing at local vLLM server."""
    logger.info("connecting to vLLM server at %s", vllm_url)
    client = OpenAI(
        base_url=f"{vllm_url}/v1",
        api_key="dummy",
    )
    return client


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_text(
    client: OpenAI,
    messages: list[dict],
    *,
    model: str = "openai/gpt-oss-20b",
    max_new_tokens: int = 32768,
    temperature: float = 0.01,
    top_p: float = 0.9,
    seed: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    capture_logprobs: bool = False,
) -> tuple[str, object]:
    """Call vLLM server via OpenAI-compatible API.

    Logprob cascade — tries in order until success:
        (logprobs=True,  top_logprobs=5)  -> entropy + prob + logprob
        (logprobs=True,  top_logprobs=1)  -> prob + logprob, no entropy
        (logprobs=False, top_logprobs=None) -> no logprobs, extraction only
    Only NaN 400 errors trigger fallback. All other errors raise immediately.
    The final (False, None) attempt is physically incapable of producing NaN.
    """
    if seed is not None:
        logger.info("seed set to %d", seed)
    if reasoning_effort is not None:
        logger.info("reasoning_effort=%s", reasoning_effort)

    extra_body = {
        "repetition_penalty": 1.1,
    }
    if reasoning_effort is not None:
        extra_body["reasoning_effort"] = reasoning_effort

    attempts = (
        [(True, 5), (True, 1), (False, None)]
        if capture_logprobs
        else [(False, None)]
    )

    last_exc = None
    for logprobs_flag, top_lp in attempts:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                logprobs=logprobs_flag,
                top_logprobs=top_lp,
                extra_body=extra_body,
            )
            if logprobs_flag != capture_logprobs:
                logger.warning(
                    "logprobs degraded to logprobs=%s top_logprobs=%s due to vLLM NaN",
                    logprobs_flag, top_lp,
                )
            result = response.choices[0].message.content or ""
            logger.debug("RAW MODEL OUTPUT: %r", result)
            return result, response

        except Exception as e:
            if "nan" in str(e).lower() and logprobs_flag:
                logger.warning(
                    "vLLM NaN in logprobs (top_logprobs=%s), trying next fallback",
                    top_lp,
                )
                last_exc = e
                continue
            raise

    raise last_exc


# ---------------------------------------------------------------------------
# Prompt construction (schema-driven)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a clinical NLP system that extracts structured NACC neuropathology variables from autopsy reports.

{format_instructions}

EXTRACTION RULES:
- Extract only what is explicitly stated or clearly implied by the report text.
- Do not hallucinate values.
- All numeric codes must be exact integers from the allowed set in each field's description.
- For NPWBRWT: extract the numeric value in grams as a float (e.g. 991.7, 1200.0).
- For severity scales: map report language carefully:
    "mild" → 1, "moderate" → 2, "severe" → 3, "no/none/absent" → 0.
    "mild to moderate" → 2 (Moderate) or "moderate to severe" or "moderately severe" → 3 (Severe).
- For PresentAbsent fields (NPLINF, NPLAC, NPHEM): 1=Present/Yes, 2=Absent/No.
- OCR quality may be imperfect: if a word appears misspelled or garbled, infer the most likely intended term from context (e.g. "diqquse" → "diffuse"). Do not skip or null a field solely due to apparent OCR errors.

ANNOTATION RULES (field_annotations):
- For every non-null field you extract, add an entry in field_annotations keyed by the variable name.
- evidence: copy the exact phrase or sentence from the report that drove your decision.
- note: write a brief reasoning note ONLY when the extraction required judgment (indirect language,
  ambiguous severity, conflicting statements). Leave null for clear-cut extractions.
- confidence: assign a float 0.0-1.0 reflecting how certain you are:
    1.0 = exact match, unambiguous language
    0.8 = clearly implied, minor paraphrase
    0.6 = indirect or requires inference
    0.4 = ambiguous, best guess among plausible codes
    below 0.4 = consider leaving field null instead

OUTPUT: valid JSON only — no markdown fences, no commentary before or after the JSON object.
- field_annotations.evidence must always be a single string, never a list or array.\
"""

USER_PROMPT = "NEUROPATHOLOGY REPORT:\n\n{report_text}"


def build_messages(report_text: str, model_cls: Type[BaseModel], format_builder=None) -> list[dict]:
    fmt = format_builder() if format_builder is not None else build_pass1a_format_instructions()
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(format_instructions=fmt)},
        {"role": "user", "content": USER_PROMPT.format(report_text=report_text)},
    ]

# ---------------------------------------------------------------------------
# Guardrailed extraction (validation + re-ask)
# ---------------------------------------------------------------------------

def strip_json_fences(text: str) -> str:
    """Extract JSON from model output.
    Handles gpt-oss channel format, ```json fences, and extra trailing data.
    """
    text = text.strip()

    text = re.sub(r"<report_analysis>.*?</report_analysis>", "", text, flags=re.DOTALL).strip()
    
    # gpt-oss channel format: extract content from final channel
    if "<|channel|>final<|message|>" in text:
        start = text.find("<|channel|>final<|message|>") + len("<|channel|>final<|message|>")
        end = text.find("<|end|>", start)
        text = text[start:end].strip() if end != -1 else text[start:].strip()
        logger.debug("Extracted from final channel: %r", text)

    # handle <|return|> end token
    if "<|return|>" in text:
        text = text[:text.find("<|return|>")].strip()

    # remove ```json fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    # handle extra data after JSON object — find the true end of the JSON
    if text.startswith("{"):
        depth = 0
        for i, ch in enumerate(text):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    text = text[: i + 1]
                    break

    return text.strip()


FINAL_CHANNEL_MARKER = "<|channel|>final<|message|>"
LOGPROB_END_MARKER   = "<|end|>"


def _build_char_to_token_map(
    token_items: list, token_start: int, token_end: int
) -> tuple[str, list[int]]:
    """Concatenate final-channel tokens into one string with char->token index map."""
    full_str    = ""
    char_to_tok = []
    for i in range(token_start, token_end):
        tok = token_items[i].token
        full_str    += tok
        char_to_tok += [i] * len(tok)
    return full_str, char_to_tok


def _find_final_channel_bounds(token_items: list) -> tuple[int, int]:
    """Return (start_idx, end_idx) of final JSON tokens in the token stream."""
    tokens = [t.token for t in token_items]

    # Find start
    start = 0
    found = False
    for i, tok in enumerate(tokens):
        if FINAL_CHANNEL_MARKER in tok:
            start = i + 1
            found = True
            break

    if not found:
        for i, tok in enumerate(tokens):
            if "<|channel|>" in tok:
                window = "".join(tokens[i:i+5])
                if "final" in window and "<|message|>" in window:
                    for j in range(i, min(i+5, len(tokens))):
                        if "<|message|>" in tokens[j]:
                            start = j + 1
                            break
                    break

    # Find end
    end = len(token_items)
    for i in range(start, len(tokens)):
        if LOGPROB_END_MARKER in tokens[i]:
            end = i
            break

    return start, end


def extract_logprobs_from_response(response, var_names: list[str]) -> dict[str, dict]:
    """
    Extract per-variable output-token logprobs from a vLLM response.
    Only operates on final-channel tokens (post-reasoning).

    Returns dict: {var_name: {"logprob": float, "prob": float, "entropy": float}}
    for successfully aligned variables. Missing variables are omitted.
    """
    import math

    logprob_data = response.choices[0].logprobs
    if logprob_data is None or logprob_data.content is None:
        return {}

    token_items         = logprob_data.content
    final_start, final_end = _find_final_channel_bounds(token_items)

    if final_start >= final_end:
        return {}

    full_str, char_to_tok = _build_char_to_token_map(
        token_items, final_start, final_end
    )

    results = {}
    for var in var_names:
        pattern = rf'"{re.escape(var)}"\s*:\s*'
        m = re.search(pattern, full_str)
        if m is None:
            continue

        val_char_start = m.end()
        if val_char_start >= len(char_to_tok):
            continue

        tok_idx = char_to_tok[val_char_start]
        t       = token_items[tok_idx]

        # Compute entropy over top_logprobs distribution
        entropy = None
        if t.top_logprobs and len(t.top_logprobs) >= 2:
            probs = [math.exp(alt.logprob) for alt in t.top_logprobs]
            # Normalize in case top-k probs don't sum to 1
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            entropy = round(
                -sum(p * math.log2(p) for p in probs if p > 0),
                6
            )

        results[var] = {
            "logprob" : round(t.logprob, 6),
            "prob"    : round(math.exp(t.logprob), 6),
            "entropy" : entropy,
        }

    return results


def truncate_at_repetition(text: str, threshold: int = 5) -> str:
    """Truncate JSON text at the point where a phrase starts repeating
    inside a string value. Uses a sliding window over comma-separated
    segments within string values to detect repetition."""
    # find the repetition start by looking for the same segment
    # appearing consecutively more than threshold times
    segments = text.split(", ")
    seen_count = {}
    seen_first_pos = {}
    
    char_pos = 0
    for i, seg in enumerate(segments):
        key = seg.strip()
        if len(key) < 10:  # skip short segments like nulls, numbers
            char_pos += len(seg) + 2
            continue
        if key in seen_count:
            seen_count[key] += 1
            if seen_count[key] >= threshold:
                # return text up to the first occurrence of this segment
                return text[:seen_first_pos[key]].rstrip(", ").strip()
        else:
            seen_count[key] = 1
            seen_first_pos[key] = char_pos
        char_pos += len(seg) + 2
    
    return text


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None for JSON compliance."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def normalize_llm_output(data: dict) -> dict:
    """
    Fixes two LLM output format issues before Pydantic validation:

    1. Wrapped scalars: {"value": 1, "field_annotations": {...}} → 1
       with annotation rescued to top-level field_annotations.

    2. Standalone field_annotations nested inside a data block
       → stripped out and merged up to top-level field_annotations.
    """
    def _unwrap_value(val):
        """Recursively unwrap {"value": X, ...} dicts to bare scalar."""
        if (isinstance(val, dict)
                and "value" in val
                and all(k in ANNOTATION_ONLY_KEYS for k in val)):
            return _unwrap_value(val["value"])
        return val

    top_annotations = data.get("field_annotations") or {}

    # Fix Pattern C: block output as single-element list containing a dict
    for block_name in list(data.keys()):
        if block_name not in DATA_BLOCKS:
            continue
        block = data[block_name]
        if (isinstance(block, list) 
                and len(block) == 1 
                and isinstance(block[0], dict)):
            logger.debug("unwrapping list-wrapped block: %s", block_name)
            data[block_name] = block[0]
            
    for block_name, block in data.items():
        if not isinstance(block, dict) or block_name not in DATA_BLOCKS:
            continue

        # Fix Pattern A: standalone field_annotations inside a data block
        if "field_annotations" in block:
            nested = block.pop("field_annotations")
            if isinstance(nested, dict):
                for k, v in nested.items():
                    top_annotations.setdefault(k, v)

        # Fix Pattern B: wrapped scalars (with recursive unwrap)
        for field_name, field_val in list(block.items()):
            if (isinstance(field_val, dict)
                    and "value" in field_val
                    and all(k in ANNOTATION_ONLY_KEYS for k in field_val)):
                # Rescue annotation before unwrapping
                if "field_annotations" in field_val:
                    ann = field_val["field_annotations"]
                    if isinstance(ann, dict):
                        top_annotations.setdefault(field_name, ann)
                elif "annotation" in field_val:
                    ann = field_val["annotation"]
                    if isinstance(ann, dict):
                        top_annotations.setdefault(field_name, ann)
                elif "annotations" in field_val:
                    ann = field_val["annotations"]
                    if isinstance(ann, dict):
                        top_annotations.setdefault(field_name, ann)
                # Recursively unwrap to bare scalar
                block[field_name] = _unwrap_value(field_val["value"])

    data["field_annotations"] = top_annotations

    # Fix Pattern D: strip stray top-level keys
    for key in list(data.keys()):
        if key not in DATA_BLOCKS and key not in ANNOTATION_ONLY_KEYS:
            logger.debug("stripping stray top-level key: %s", key[:100])
            data.pop(key)

    for block_name, block in data.items():
        if not isinstance(block, dict) or block_name not in DATA_BLOCKS:
            continue
        for key in list(block.keys()):
            if not key.isupper() and key not in DATA_BLOCKS and key not in ANNOTATION_ONLY_KEYS:
                logger.debug("stripping stray key from %s: %s", block_name, key[:100])
                block.pop(key)

    return data


def build_field_router(model_cls: Type[BaseModel]) -> dict[str, str]:
    """
    Derives variable -> block_name mapping from a Pydantic pass model.
    
    Inspects each field of the pass model. If the field's type is itself
    a BaseModel subclass (i.e. a sub-block), maps every variable in that
    sub-model to the block name.
    """
    router = {}
    for block_name, field_info in model_cls.model_fields.items():
        # Get the actual type, unwrapping Optional if needed
        annotation = field_info.annotation
        args = get_args(annotation)
        block_type = args[0] if args else annotation
        
        if inspect.isclass(block_type) and issubclass(block_type, BaseModel):
            for var_name in block_type.model_fields:
                router[var_name] = block_name
    return router


def reroute_fields(data: dict, model_cls: Type[BaseModel]) -> dict:
    """
    Moves any variable found in the wrong block to its correct block,
    as defined by the Pydantic schema for this pass.
    
    Only operates on known variables — unknown keys are left untouched.
    """
    router = build_field_router(model_cls)

    for key, value in list(data.items()):
        if key in router and not isinstance(value, dict):
            correct_block = router[key]
            logger.debug("rerouting root variable %s -> %s", key, correct_block)
            target = data.setdefault(correct_block, {})
            if key not in target:
                target[key] = data.pop(key)
            else:
                data.pop(key)
    
    for block_name, block in list(data.items()):
        if not isinstance(block, dict) or block_name not in DATA_BLOCKS:
            continue
        for var_name in list(block.keys()):
            correct_block = router.get(var_name)
            if correct_block is None:
                continue  # not a known schema variable, leave it
            if correct_block != block_name:
                logger.debug(
                    "rerouting %s from %s -> %s",
                    var_name, block_name, correct_block
                )
                target = data.setdefault(correct_block, {})
                if var_name not in target:
                    target[var_name] = block.pop(var_name)
                else:
                    block.pop(var_name)
    return data


def extract(
    report_text: str,
    client: OpenAI,
    *,
    model: str = "openai/gpt-oss-20b",
    model_cls: Optional[Type[BaseModel]] = None,
    format_builder=None,
    temperature: float = 0.01,
    top_p: float = 0.9,
    max_new_tokens: int = 32768,
    max_retries: int = 3,
    seed: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    capture_logprobs: bool = False,
) -> tuple[BaseModel, dict]:
    """Single-pass extraction with Pydantic validation and hinted re-ask."""
    if model_cls is None:
        model_cls = get_pass1a_model()

    base_messages = build_messages(report_text, model_cls, format_builder=format_builder)
    current_messages = list(base_messages)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        logger.info("attempt %d/%d", attempt, max_retries)

        raw, response = generate_text(
            client,
            current_messages,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            reasoning_effort=reasoning_effort,
            capture_logprobs=capture_logprobs,
        )

        # parse JSON
        cleaned = strip_json_fences(raw)
        logger.debug("cleaned text for JSON parsing: %r", cleaned)
        try:
            content = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("JSON decode failed, attempting repair: %s", e)
            try:
                repaired = repair_json(cleaned, return_objects=True)
                if isinstance(repaired, dict) and repaired:
                    logger.info("JSON repaired successfully")
                    content = repaired
                else:
                    raise ValueError("repair produced empty or non-dict result")
            except Exception as repair_err:
                last_error = repair_err
                logger.error("JSON repair failed. Original error: %s | Repair error: %s", e, repair_err)
                logger.error("INVALID JSON (PARSE FAILURE):\n%s", cleaned)
                assistant_content = truncate_at_repetition(cleaned) if cleaned.startswith("{") else ""
                current_messages = list(base_messages) + [
                    {"role": "assistant", "content": assistant_content},
                    {
                        "role": "user",
                        "content": f"Your response was not valid JSON. Error: {e}. Output valid JSON only.",
                    },
                ]
                continue
        
        content = normalize_llm_output(content)
        content = reroute_fields(content, model_cls)

        # Pydantic validation
        try:
            validated = model_cls.model_validate(content)
            if capture_logprobs:
                all_vars = []
                for fname, fval in validated:
                    if isinstance(fval, BaseModel):
                        all_vars.extend(
                            k for k in fval.__class__.model_fields.keys()
                            if k.isupper()
                        )
                    else:
                        if fname.isupper():
                            all_vars.append(fname)
                return validated, extract_logprobs_from_response(response, all_vars)
            return validated, {}
        except ValidationError as e:
            last_error = e
            logger.warning(
                "Validation failed (%d errors): %s", e.error_count(), e.errors()
            )
            logger.error("INVALID JSON (SCHEMA VALIDATION FAILED):\n%s", json.dumps(content, indent=2))
            current_messages = list(base_messages) + [
                {"role": "assistant", "content": json.dumps(_sanitize_for_json(content), indent=2)},
                {
                    "role": "user",
                    "content": (
                        f"Your JSON had validation errors:\n{e}\n"
                        "Fix these errors and output corrected JSON only."
                    ),
                },
            ]
            continue

    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------------------
# Pass result flattening and merging
# ---------------------------------------------------------------------------


def flatten_pass_result(result: BaseModel) -> dict:
    """Return nested dict preserving sub-model structure.

    Top-level keys are sub-model names (e.g. 'specimen_info', 'infarct_parent').
    Skips field_annotations, extraction_confidence, extraction_notes.
    """
    SKIP = {"field_annotations", "extraction_confidence", "extraction_notes"}
    nested = {}
    for field_name, field_value in result:
        if field_name in SKIP:
            continue
        if isinstance(field_value, BaseModel):
            section = {}
            for sub_name, sub_value in field_value:
                section[sub_name] = sub_value.value if hasattr(sub_value, "value") else sub_value
            nested[field_name] = section
        else:
            nested[field_name] = field_value.value if hasattr(field_value, "value") else field_value
    return nested


def compute_programmatic(flat: dict) -> dict:
    """Compute NACCVASC and NACCBRNN from the merged flat variable dict.

    NACCVASC — v10-11 sources per RDD-NP PDF:
        npwmr, nphemo, npold, npoldd, nparter(=NACCARTE), npavas(=NACCAVAS),
        npinf, npamy(=NACCAMY), nppath
        1 = at least one vascular pathology present
        0 = all absent
        9 = some 0, remainder not assessed or missing

    NACCBRNN — v10-11 criteria per RDD-NP PDF:
        0 = some pathologic change present - takes priority
        9 = at least one required variable is not assessed/missing/unknown
        1 = no major pathologic change present (all criteria met)

    Note: 5 PDF criteria variables (NPPDXC, NPPDXR, NPPDXS, NPPDXT, NPARTAG)
    are absent from the 199-variable CSV schema and are skipped.
    """
    def _get(key, default=0):
        v = flat.get(key)
        return v if v is not None else default

    # ── NACCVASC ──────────────────────────────────────────────────────────
    # v10-11 source variables (raw parent vars per PDF):
    # npwmr, nphemo, npold, npoldd, nparter, npavas, npinf, npamy, nppath
    vasc_severity = ["NPWMR", "NACCARTE", "NACCAVAS", "NACCAMY"]       # >= 1 means present
    vasc_binary   = ["NPHEMO", "NPOLD", "NPOLDD", "NPINF", "NPPATH"]   # == 1 means present
    vasc_all      = vasc_severity + vasc_binary

    vasc_positive = (
        any(1 <= _get(k) <= 3 for k in vasc_severity) or
        any(_get(k) == 1 for k in vasc_binary)
    )
    vasc_unknown = any(_get(k) in {8, 9} for k in vasc_all)

    if vasc_positive:      # any >= 1 -> code 1, regardless of 9s elsewhere
        naccvasc = 1
    elif vasc_unknown:     # no positives, but at least one 9 -> code 9
        naccvasc = 9
    else:                  # all 0 -> code 0
        naccvasc = 0

    # ── NACCBRNN ──────────────────────────────────────────────────────────
    # NACCBRNN=1 requires ALL of the following (v10-11 criteria, RDD-NP PDF):

    # Braak stage 0, I, or II
    braak_ok   = _get("NACCBRAA") in {0, 1, 2}
    # No amyloid pathology
    amyloid_ok = (_get("NACCNEUR") == 0 and
                  _get("NACCDIFF") == 0 and
                  _get("NPTHAL")   == 0)
    # No or mild vascular pathology
    vasc_ok    = (_get("NACCAMY")   in {0, 1} and
                  _get("NPINF")    == 0 and
                  _get("NPHEMO")   == 0 and
                  _get("NPOLD")    == 0 and
                  _get("NPOLDD")   == 0 and
                  _get("NACCAVAS") in {0, 1} and
                  _get("NACCARTE") in {0, 1} and
                  _get("NPWMR")    in {0, 1} and
                  _get("NPPATH")   == 0)
    # No Lewy bodies
    lewy_ok    = _get("NPLBOD") == 0
    # No other pathology (e.g., TDP, FTLD, tangleonly dementia, prion disease, etc.)
    other_ok   = (_get("NPNLOSS")  in {0, 1} and
                  _get("NPHIPSCL") == 0 and
                  _get("NPTDPA")   == 0 and
                  _get("NPTDPB")   == 0 and
                  _get("NPTDPC")   == 0 and
                  _get("NPTDPD")   == 0 and
                  _get("NPTDPE")   == 0 and
                  _get("NPFTDTAU") == 0 and
                  _get("NPFTDTDP") == 0 and
                  _get("NPALSMND") == 0 and
                  _get("NPOFTD")   == 0 and
                  _get("NPPDXA")   == 0 and
                  _get("NPPDXB")   == 0 and
                  _get("NPPDXD")   == 0 and
                  _get("NPPDXE")   == 0 and
                  _get("NPPDXF")   == 0 and
                  _get("NPPDXG")   == 0 and
                  _get("NPPDXH")   == 0 and
                  _get("NPPDXI")   == 0 and
                  _get("NPPDXJ")   == 0 and
                  _get("NPPDXK")   == 0 and
                  _get("NPPDXL")   == 0 and
                  _get("NPPDXM")   == 0 and
                  _get("NPPDXN")   == 0)

    # All required source variables for the 9-check
    any_missing = any(_get(k) in {8, 9} for k in [
        "NACCBRAA",
        "NACCNEUR", "NACCDIFF", "NPTHAL",
        "NACCAMY", "NPINF", "NPHEMO", "NPOLD", "NPOLDD",
        "NACCAVAS", "NACCARTE", "NPWMR", "NPPATH",
        "NPLBOD",
        "NPNLOSS", "NPHIPSCL",
        "NPTDPA", "NPTDPB", "NPTDPC", "NPTDPD", "NPTDPE",
        "NPFTDTAU", "NPFTDTDP", "NPALSMND", "NPOFTD",
        "NPPDXA", "NPPDXB", "NPPDXD", "NPPDXE", "NPPDXF",
        "NPPDXG", "NPPDXH", "NPPDXI", "NPPDXJ", "NPPDXK",
        "NPPDXL", "NPPDXM", "NPPDXN",
    ])

    # Priority: 0 > 9 > 1
    if not (braak_ok and amyloid_ok and vasc_ok and lewy_ok and other_ok):
        naccbrnn = 0
    elif any_missing:
        naccbrnn = 9
    else:
        naccbrnn = 1

    return {"NACCVASC": naccvasc, "NACCBRNN": naccbrnn}


def merge_pass_results(pass_results: dict[int, dict]) -> dict:
    """Merge nested dicts from all 7 passes into one final output dict."""
    merged = {}
    merged_annotations = {}
    merged_confidence = {}
    merged_notes = {}

    for pass_num in sorted(pass_results.keys()):
        flat = pass_results[pass_num]
        for k, v in flat.items():
            if k == "field_annotations":
                if isinstance(v, dict):
                    merged_annotations.update(v)
            elif k == "extraction_confidence":
                if v is not None:
                    merged_confidence[f"pass{pass_num}"] = v
            elif k == "extraction_notes":
                if v is not None:
                    merged_notes[f"pass{pass_num}"] = v
            else:
                merged[k] = v

    # Build flat lookup for compute_programmatic
    flat_lookup = {}
    for k, v in merged.items():
        if isinstance(v, dict):
            flat_lookup.update(v)

    programmatic = compute_programmatic(flat_lookup)
    merged["programmatic"] = programmatic

    if merged_annotations:
        merged["field_annotations"] = merged_annotations
    if merged_confidence:
        merged["extraction_confidence"] = merged_confidence
    if merged_notes:
        merged["extraction_notes"] = merged_notes

    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract structured data from a neuropathology report (local HF model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Model aliases: "
        + ", ".join(f"{k} -> {v}" for k, v in MODEL_ALIASES.items()),
    )
    ap.add_argument(
        "-i", "--input", required=True, type=Path, help="Path to report (.txt or .pdf)"
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write output JSON files",
    )
    ap.add_argument(
        "-m",
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Model name as served by vLLM (default: openai/gpt-oss-20b)",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="One or more seeds to run (e.g. --seeds 0 1 2 3 4)",
    )
    ap.add_argument(
        "--vllm-url",
        type=str,
        required=True,
        help="URL of the vLLM server (e.g. http://localhost:8000)",
    )
    ap.add_argument(
        "--input-format",
        type=str,
        default=None,
        choices=["text", "pdf"],
        help="Override auto-detected input format",
    )
    ap.add_argument("--temperature", type=float, default=0.01)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort for gpt-oss models (default: medium; ignored by other models)",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=32768,
        help="Max new tokens to generate (default: 32768)",
    )
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--logprobs",
        action="store_true",
        default=False,
        help="Capture and store output-token logprobs per variable",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # --- load input ---
    if not args.input.exists():
        logger.error("file not found: %s", args.input)
        sys.exit(1)

    seeds = args.seeds if args.seeds is not None else [0]
    out_dir = args.output_dir or args.input.parent

    # Resume: check which seeds still need work
    # A seed is complete if its final .extracted.json exists.
    # A seed is partially done if some _pass{N}.json intermediate files exist.
    pending_seeds = []
    for seed in seeds:
        out_path = out_dir / f"{args.input.stem}_seed{seed}.extracted.json"
        if out_path.exists():
            logger.info("skipping seed %d — final output already exists: %s", seed, out_path)
        else:
            pending_seeds.append(seed)

    if not pending_seeds:
        logger.info("all seeds already processed, nothing to do")
        return
    
    report_text = load_report(args.input, args.input_format)
    logger.info("loaded %d chars from %s", len(report_text), args.input)

    # --- resolve & load model ---
    model_id = resolve_model(args.model)
    logger.info("vLLM URL: %s | model: %s", args.vllm_url, model_id)
    client = load_model(args.vllm_url)

    for seed in pending_seeds:
        out_path = out_dir / f"{args.input.stem}_seed{seed}.extracted.json"
        logger.info("running seed %d -> %s", seed, out_path)

        logprobs_dir  = out_dir / "logprobs"
        logprobs_path = logprobs_dir / f"{args.input.stem}_seed{seed}.logprobs.json"
        seed_logprobs: dict[str, dict] = {}

        if args.logprobs:
            logprobs_dir.mkdir(parents=True, exist_ok=True)

        # ── Seed-level failure check ───────────────────────────────────────
        # If any pass has a .failed marker, delete ALL intermediate files
        # for this seed and re-run from scratch to ensure complete logprob
        # capture across all passes.
        any_pass_failed = any(
            (out_dir / f"{args.input.stem}_seed{seed}_pass{pass_num}.failed").exists()
            for pass_num in PASS_REGISTRY
        )
        if any_pass_failed:
            logger.warning(
                "seed %d — found failed pass(es), clearing all intermediate "
                "files for full re-run with complete logprob capture", seed
            )
            for pass_num in PASS_REGISTRY:
                for ext in (".json", ".failed"):
                    p = out_dir / f"{args.input.stem}_seed{seed}_pass{pass_num}{ext}"
                    if p.exists():
                        p.unlink()
                        logger.debug("cleared: %s", p)

        pass_results: dict[int, dict] = {}

        for pass_num, (get_model, build_fmt) in PASS_REGISTRY.items():
            pass_path = out_dir / f"{args.input.stem}_seed{seed}_pass{pass_num}.json"

            # Resume: skip pass if intermediate file already exists
            if pass_path.exists():
                logger.info("seed %d pass %d — intermediate exists, loading: %s", seed, pass_num, pass_path)
                with open(pass_path, encoding="utf-8") as f:
                    pass_results[pass_num] = json.load(f)
                continue

            logger.info("seed %d pass %d/%d", seed, pass_num, len(PASS_REGISTRY))
            pass_model_cls = get_model()

            try:
                result, pass_logprobs = extract(
                    report_text,
                    client,
                    model=model_id,
                    model_cls=pass_model_cls,
                    format_builder=build_fmt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    max_retries=args.max_retries,
                    seed=seed,
                    reasoning_effort=args.reasoning_effort,
                    capture_logprobs=args.logprobs,
                )

                if args.logprobs and pass_logprobs:
                    seed_logprobs.update(pass_logprobs)
                    logger.debug(
                        "seed %d pass %d — captured logprobs for %d variables",
                        seed, pass_num, len(pass_logprobs)
                    )
            except Exception as e:
                logger.error("seed %d pass %d FAILED after all retries: %s", seed, pass_num, e)
                # Write a partial marker so we know this pass failed
                pass_path.with_suffix(".failed").write_text(str(e), encoding="utf-8")
                # Continue to next pass — partial results still saved
                continue

            # Flatten nested Pydantic model to variable-level dict
            flat = flatten_pass_result(result)

            # Preserve annotations and metadata separately
            raw = _sanitize_for_json(result.model_dump(mode="json"))
            flat["field_annotations"] = raw.get("field_annotations") or {}
            flat["extraction_confidence"] = raw.get("extraction_confidence")
            flat["extraction_notes"] = raw.get("extraction_notes")

            # Write intermediate pass file
            pass_path.write_text(
                json.dumps(_sanitize_for_json(flat), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("seed %d pass %d written: %s", seed, pass_num, pass_path)
            pass_results[pass_num] = flat

        if not pass_results:
            logger.error("seed %d — all passes failed, no output written", seed)
            continue


        # Clean up intermediate pass files only if all passes succeeded
        any_failed = any(
            (out_dir / f"{args.input.stem}_seed{seed}_pass{pass_num}.failed").exists()
            for pass_num in PASS_REGISTRY
        )
        
        if any_failed:
            logger.warning(
                "seed %d — some passes failed, skipping merge. "
                "Fix failures and re-run to complete.", seed
            )
            continue

        # Merge all pass results and compute programmatic variables
        out_data = merge_pass_results(pass_results)

        # Write final merged output
        out_path.write_text(
            json.dumps(_sanitize_for_json(out_data), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("wrote final output: %s", out_path)

        if args.logprobs and seed_logprobs:
            logprobs_path.write_text(
                json.dumps(seed_logprobs, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("wrote logprobs: %s", logprobs_path)

        for pass_num in PASS_REGISTRY:
            pass_path = out_dir / f"{args.input.stem}_seed{seed}_pass{pass_num}.json"
            if pass_path.exists():
                pass_path.unlink()
                logger.debug("cleaned up intermediate: %s", pass_path)

        print(json.dumps(_sanitize_for_json(out_data), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()