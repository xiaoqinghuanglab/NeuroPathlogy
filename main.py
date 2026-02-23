"""main: single-pass structured extraction for neuropathology reports.

Loads a HuggingFace model locally onto GPUs, reads a report (PDF or text),
extracts structured data via schema-driven prompting, validates with Pydantic,
and re-asks on validation failure.

Supported models:
    oss-120b    → openai/gpt-oss-120b
    oss-20b     → openai/gpt-oss-20b
    llama3.1-8b → meta-llama/Llama-3.1-8B-Instruct

Usage:
    python main.py -i report.txt -m oss-120b --num-gpus 4
    python main.py -i report.pdf -m oss-20b --num-gpus 2
    python main.py -i report.txt -m llama3.1-8b
    python main.py -i report.pdf -m oss-120b --num-gpus 4 -o result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Type

import torch
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema import build_format_instructions, get_extraction_model

logger = logging.getLogger("medace")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, str] = {
    "oss-120b": "openai/gpt-oss-120b",
    "oss-20b": "openai/gpt-oss-20b",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

SUPPORTED_ALIASES = list(MODEL_ALIASES.keys())


def resolve_model(alias: str) -> str:
    """Map short alias → full HF model ID.  Pass-through if already full."""
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
    """Extract text from PDF.  pymupdf (fast) → pdfplumber (fallback)."""
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


def load_model(
    model_id: str,
    num_gpus: int = 1,
    dtype: str = "auto",
) -> tuple:
    """Load model + tokenizer with automatic multi-GPU sharding."""
    available = torch.cuda.device_count()
    effective = min(num_gpus, available) if available > 0 else 0
    if effective > 0 and effective < available:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(effective))
        logger.info(
            "CUDA_VISIBLE_DEVICES set to %s (requested %d, available %d)",
            os.environ["CUDA_VISIBLE_DEVICES"],
            num_gpus,
            available,
        )

    torch_dtype = torch.bfloat16 if dtype == "auto" else getattr(torch, dtype)

    logger.info("loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    logger.info(
        "loading model: %s on %d GPU(s), dtype=%s",
        model_id,
        effective or 1,
        torch_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("model loaded: %.1fB params", model.num_parameters() / 1e9)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_text(
    model,
    tokenizer,
    messages: list[dict],
    *,
    max_new_tokens: int = 32768,
    temperature: float = 0.3,
    top_p: float = 0.9,
    do_sample: bool = True,
    seed: Optional[int] = None,
) -> str:
    """Apply chat template, generate, decode."""
    if seed is not None:
        torch.manual_seed(seed)
        logger.info("random seed set to %d", seed)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    # strip prompt tokens, keep special tokens for channel parsing
    generated = outputs[0, inputs["input_ids"].shape[1] :]
    result = tokenizer.decode(generated, skip_special_tokens=False)
    logger.debug("RAW MODEL OUTPUT: %r", result)
    return result


# ---------------------------------------------------------------------------
# Prompt construction (schema-driven)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a clinical NLP system specializing in neuropathology report extraction.
Read the neuropathology report and extract structured data into JSON.

{format_instructions}

RULES:
- Extract only what is explicitly stated or clearly implied.
- Use null for any field that cannot be determined.
- Do not hallucinate.  If a molecular test is not mentioned, use "not_tested".
- For Ki-67, extract the numeric percentage (e.g., 30 not 0.30).
- For dates, use YYYY-MM-DD format.  For times, use HH:MM 24-hour format.
- For patient name, extract Last and First separately.
- Output valid JSON only — no markdown fences, no commentary."""

USER_PROMPT = "NEUROPATHOLOGY REPORT:\n\n{report_text}"


def build_messages(report_text: str, model_cls: Type[BaseModel]) -> list[dict]:
    fmt = build_format_instructions(model_cls)
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


def extract(
    report_text: str,
    model,
    tokenizer,
    *,
    model_cls: Optional[Type[BaseModel]] = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_new_tokens: int = 32768,
    max_retries: int = 3,
    seed: Optional[int] = None,
) -> BaseModel:
    """Single-pass extraction with Pydantic validation and hinted re-ask."""
    if model_cls is None:
        model_cls = get_extraction_model()

    base_messages = build_messages(report_text, model_cls)
    current_messages = list(base_messages)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        logger.info("attempt %d/%d", attempt, max_retries)

        raw = generate_text(
            model,
            tokenizer,
            current_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )

        # parse JSON
        cleaned = strip_json_fences(raw)
        logger.debug("cleaned text for JSON parsing: %r", cleaned)
        try:
            content = json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning("JSON decode failed: %s", e)
            current_messages = list(base_messages) + [
                {"role": "assistant", "content": cleaned},
                {
                    "role": "user",
                    "content": f"Your response was not valid JSON. Error: {e}. Output valid JSON only.",
                },
            ]
            continue

        # Pydantic validation
        try:
            return model_cls.model_validate(content)
        except ValidationError as e:
            last_error = e
            logger.warning(
                "Validation failed (%d errors): %s", e.error_count(), e.errors()
            )
            current_messages = list(base_messages) + [
                {"role": "assistant", "content": cleaned},
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
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract structured data from a neuropathology report (local HF model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Model aliases: "
        + ", ".join(f"{k} → {v}" for k, v in MODEL_ALIASES.items()),
    )
    ap.add_argument(
        "-i", "--input", required=True, type=Path, help="Path to report (.txt or .pdf)"
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <input>.extracted.json)",
    )
    ap.add_argument(
        "-m",
        "--model",
        required=True,
        help=f"Model alias ({', '.join(SUPPORTED_ALIASES)}) or full HF model ID",
    )
    ap.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs to use (default: 1)"
    )
    ap.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16"],
        help="Model dtype (default: auto → bfloat16)",
    )
    ap.add_argument(
        "--input-format",
        type=str,
        default=None,
        choices=["text", "pdf"],
        help="Override auto-detected input format",
    )
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=32768,
        help="Max new tokens to generate (default: 32768)",
    )
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    ap.add_argument("--verbose", action="store_true")
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

    report_text = load_report(args.input, args.input_format)
    logger.info("loaded %d chars from %s", len(report_text), args.input)

    # --- resolve & load model ---
    model_id = resolve_model(args.model)
    logger.info("model: %s → %s", args.model, model_id)

    model, tokenizer = load_model(model_id, num_gpus=args.num_gpus, dtype=args.dtype)

    # --- extract ---
    model_cls = get_extraction_model()
    result = extract(
        report_text,
        model,
        tokenizer,
        model_cls=model_cls,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
        seed=args.seed,
    )

    # --- output ---
    out_data = result.model_dump(mode="json")
    out_path = args.output or args.input.with_suffix(".extracted.json")
    out_path.write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("wrote %s", out_path)

    print(json.dumps(out_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
