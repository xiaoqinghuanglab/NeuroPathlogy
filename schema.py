"""Pydantic schema for structured neuropathology report extraction.

SINGLE SOURCE OF TRUTH — edit this file to change what gets extracted.
main.py imports `get_extraction_model()` and `build_format_instructions()`
and never hardcodes field names.

Fields are drawn from the NACC Neuropathology (NP) data dictionary.
Work with your PI before adding/removing variables.

HOW TO ADD FIELDS:
  1. Add a field to the appropriate sub-model (or create a new one).
  2. Give it a `description=` — this auto-populates the LLM prompt.
  3. Add validators (range, enum, regex) — these auto-trigger on re-ask.
  4. That's it. The prompt and validation pipeline adapt automatically.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-models (grouped by NACC domain)
# ---------------------------------------------------------------------------

class GrossFindings(BaseModel):
    """NACC gross neuropathological findings."""
    model_config = ConfigDict(extra="forbid")

    NPGRLA: Optional[int] = Field(
        None,
        description=(
            "Lobar atrophy presence. "
            "0=None, 1=Yes; also 8=not assessed, 9=unknown, -4=N/A."
        ),
    )
    NPGRHA: Optional[int] = Field(
        None,
        description=(
            "Hippocampus atrophy severity. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe; also 8=not assessed, 9=unknown, -4=N/A."
        ),
    )
    NPGRSNH: Optional[int] = Field(
        None,
        description=(
            "Substantia nigra hypopigmentation severity. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe; also 8=not assessed, 9=unknown, -4=N/A."
        ),
    )
    NPGRLCH: Optional[int] = Field(
        None,
        description=(
            "Locus coeruleus hypopigmentation severity. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe; also 8=not assessed, 9=unknown, -4=N/A."
        ),
    )

    @model_validator(mode="after")
    def validate_gross_codes(self) -> "GrossFindings":
        la_valid  = {0, 1, 8, 9, -4}
        sev_valid = {0, 1, 2, 3, 8, 9, -4}
        checks = [
            ("NPGRLA",  self.NPGRLA,  la_valid),
            ("NPGRHA",  self.NPGRHA,  sev_valid),
            ("NPGRSNH", self.NPGRSNH, sev_valid),
            ("NPGRLCH", self.NPGRLCH, sev_valid),
        ]
        for fname, val, allowed in checks:
            if val is not None and val not in allowed:
                raise ValueError(f"{fname}={val} not in allowed codes {sorted(allowed)}")
        return self


class VascularPathology(BaseModel):
    """NACC vascular neuropathological findings."""
    model_config = ConfigDict(extra="forbid")

    NACCAVAS: Optional[int] = Field(
        None,
        description=(
            "Atherosclerosis severity. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe; also 8=not assessed, 9=unknown, -4=N/A."
        ),
    )
    NPLINF: Optional[int] = Field(
        None,
        description=(
            "Lacunar infarcts presence. "
            "0=Absent, 1=Present; also 8=not assessed, 9=unknown, -4=N/A."
        ),
    )
    NPLAC: Optional[int] = Field(
        None,
        description=(
            "Large vessel cortical infarcts presence. "
            "0=Absent, 1=Present; also 8=not assessed, 9=unknown, -4=N/A."
        ),
    )
    NPHEM: Optional[int] = Field(
        None,
        description=(
            "Hemorrhage (micro or macro) presence. "
            "0=Absent, 1=Present; also 8=not assessed, 9=unknown, -4=N/A."
        ),
    )

    @model_validator(mode="after")
    def validate_vascular_codes(self) -> "VascularPathology":
        sev_valid  = {0, 1, 2, 3, 8, 9, -4}
        pres_valid = {0, 1, 8, 9, -4}
        checks = [
            ("NACCAVAS", self.NACCAVAS, sev_valid),
            ("NPLINF",   self.NPLINF,   pres_valid),
            ("NPLAC",    self.NPLAC,    pres_valid),
            ("NPHEM",    self.NPHEM,    pres_valid),
        ]
        for fname, val, allowed in checks:
            if val is not None and val not in allowed:
                raise ValueError(f"{fname}={val} not in allowed codes {sorted(allowed)}")
        return self


class MicroscopicFindings(BaseModel):
    """NACC microscopic neuropathological findings."""
    model_config = ConfigDict(extra="forbid")

    NPWMR: Optional[int] = Field(
        None,
        description=(
            "White matter rarefaction severity. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe; also 8=not assessed, 9=unknown, -4=N/A."
        ),
    )
    NPNLOSS: Optional[int] = Field(
        None,
        description=(
            "Neuronal loss severity (cortical). "
            "0=None, 1=Mild, 2=Moderate, 3=Severe; also 8=not assessed, 9=unknown, -4=N/A."
        ),
    )

    @model_validator(mode="after")
    def validate_micro_codes(self) -> "MicroscopicFindings":
        sev_valid = {0, 1, 2, 3, 8, 9, -4}
        for fname, val in [("NPWMR", self.NPWMR), ("NPNLOSS", self.NPNLOSS)]:
            if val is not None and val not in sev_valid:
                raise ValueError(f"{fname}={val} not in allowed codes {sorted(sev_valid)}")
        return self


class DiagnosticCodes(BaseModel):
    """NACC diagnostic/etiological classification fields."""
    model_config = ConfigDict(extra="forbid")

    NACCCBD: Optional[int] = Field(
        None,
        description=(
            "Corticobasal degeneration (CBD) subtype. "
            "0=Absent, 1=CBD-NK (not otherwise specified), 2=CBD-AD (with AD), "
            "3=CBD-PSP (with PSP), 4=CBD-FTLD-TDP, 5=CBD-Other; "
            "also 8=not assessed, 9=unknown, -4=N/A."
        ),
    )
    NPPVASC: Optional[int] = Field(
        None,
        description=(
            "Primary vascular etiology. "
            "0=No, 1=Yes; also 8=not assessed, 9=unknown, -4=N/A."
        ),
    )

    @model_validator(mode="after")
    def validate_diag_codes(self) -> "DiagnosticCodes":
        cbd_valid  = {0, 1, 2, 3, 4, 5, 8, 9, -4}
        pvas_valid = {0, 1, 8, 9, -4}
        if self.NACCCBD is not None and self.NACCCBD not in cbd_valid:
            raise ValueError(f"NACCCBD={self.NACCCBD} not in {sorted(cbd_valid)}")
        if self.NPPVASC is not None and self.NPPVASC not in pvas_valid:
            raise ValueError(f"NPPVASC={self.NPPVASC} not in {sorted(pvas_valid)}")
        return self


# ---------------------------------------------------------------------------
# Top-level extraction model
# ---------------------------------------------------------------------------

class NeuropathologyExtraction(BaseModel):
    """Structured extraction from a single NACC neuropathology report.

    All numeric codes must match the allowed values listed in each field's
    description. Use null if a field cannot be determined from the report.
    """
    model_config = ConfigDict(extra="forbid")

    # ── Specimen metadata ──────────────────────────────────────────────────
    NPSEX: Optional[int] = Field(
        None,
        description="Subject sex. 1=Male, 2=Female.",
    )
    NPFIX: Optional[int] = Field(
        None,
        description="Fixative used. 1=Formalin, 2=Paraformaldehyde, 7=Other; -4=N/A.",
    )
    NPWBRWT: Optional[int] = Field(
        None,
        description=(
            "Whole brain weight in grams (valid range 100–2500). "
            "Use 9999 for unknown, -4 for N/A."
        ),
    )

    # ── Domain sub-models ──────────────────────────────────────────────────
    gross_findings:       GrossFindings      = Field(default_factory=GrossFindings)
    vascular_pathology:   VascularPathology  = Field(default_factory=VascularPathology)
    microscopic_findings: MicroscopicFindings = Field(default_factory=MicroscopicFindings)
    diagnostic_codes:     DiagnosticCodes    = Field(default_factory=DiagnosticCodes)

    # ── Extraction meta ────────────────────────────────────────────────────
    extraction_confidence: Optional[str] = Field(
        None,
        description="Overall confidence in this extraction: high, moderate, or low.",
    )
    extraction_notes: Optional[str] = Field(
        None,
        description="Caveats, ambiguities, or fields that could not be resolved.",
    )

    @field_validator("NPSEX")
    @classmethod
    def validate_sex(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v not in {1, 2}:
            raise ValueError(f"NPSEX must be 1 (Male) or 2 (Female), got {v}")
        return v

    @field_validator("NPFIX")
    @classmethod
    def validate_fixative(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v not in {1, 2, 7, -4}:
            raise ValueError(f"NPFIX must be 1, 2, 7, or -4, got {v}")
        return v

    @field_validator("NPWBRWT")
    @classmethod
    def validate_brain_weight(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v in {9999, -4}:          # special codes bypass range check
            return v
        if not (100 <= v <= 2500):
            raise ValueError(f"NPWBRWT must be 100–2500 g (or 9999/-4), got {v}")
        return v

    @model_validator(mode="after")
    def confidence_is_valid(self) -> "NeuropathologyExtraction":
        if self.extraction_confidence and self.extraction_confidence not in (
            "high", "moderate", "low"
        ):
            raise ValueError("extraction_confidence must be high, moderate, or low")
        return self


# ---------------------------------------------------------------------------
# Public API — main.py imports only these two functions
# ---------------------------------------------------------------------------

def get_extraction_model() -> Type[BaseModel]:
    """Return the top-level extraction model class.

    main.py calls this so it never hardcodes the schema class name.
    To swap the entire schema, change this return value.
    """
    return NeuropathologyExtraction


def unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Unwrap Optional[X] → (X, True); non-optional → (annotation, False)."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union and type(None) in args:
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1:
            return inner[0], True
    return annotation, False


def describe_field(name: str, info, indent: int = 0) -> List[str]:
    """Recursively describe a Pydantic field for prompt instructions."""
    prefix = "  " * indent
    lines: List[str] = []
    desc = info.description or ""

    raw_ann = info.annotation
    inner, is_optional = unwrap_optional(raw_ann)
    opt_tag = " (optional)" if is_optional else ""

    # enum
    if isinstance(inner, type) and issubclass(inner, Enum):
        allowed = [e.value for e in inner]
        lines.append(f"{prefix}- {name}{opt_tag}: {desc}. Allowed values: {allowed}")
        return lines

    # nested BaseModel — recurse
    if isinstance(inner, type) and issubclass(inner, BaseModel):
        lines.append(f"{prefix}- {name}: {desc}")
        for sub_name, sub_info in inner.model_fields.items():
            lines.extend(describe_field(sub_name, sub_info, indent + 1))
        return lines

    # numeric constraints from metadata
    constraints = []
    if info.metadata:
        for m in info.metadata:
            if hasattr(m, "ge"):
                constraints.append(f">= {m.ge}")
            if hasattr(m, "le"):
                constraints.append(f"<= {m.le}")
    c_str = f" [{', '.join(constraints)}]" if constraints else ""

    lines.append(f"{prefix}- {name}{opt_tag}: {desc}{c_str}")
    return lines


def build_format_instructions(model: Optional[Type[BaseModel]] = None) -> str:
    """Walk the Pydantic schema and emit LLM-friendly format instructions.

    If no model is passed, uses get_extraction_model().
    """
    if model is None:
        model = get_extraction_model()

    lines = [
        "Respond with a single JSON object conforming to this schema:",
        "",
        f"Root model: {model.__name__}",
        "",
    ]
    for name, info in model.model_fields.items():
        lines.extend(describe_field(name, info, indent=0))
    lines += [
        "",
        "Use null for any field whose value cannot be determined from the report.",
        "Do not invent information. Extract only what is explicitly stated or clearly implied.",
        "All numeric codes must be exact integers from the allowed set in each field description.",
    ]
    return "\n".join(lines)
