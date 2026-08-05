"""schema1b.py — Pass 1b: FTLD-tau, TDP-43, and Prion  (13 vars).

Pass 1b variables:
  FTLD-tau:      NPFTDTAU, NACCPICK, NACCCBD, NACCPROG, NPFTDT2, NPFTDT5,
                 NPFTDT6, NPFTDT7, NPFTDT8, NPFTDT9, NPFTDT10
  TDP-43 flag:   NPFTDTDP
  Prion:         NACCPRIO
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# FieldAnnotation
# ---------------------------------------------------------------------------

class FieldAnnotation(BaseModel):
    """Extraction audit trail for a single variable."""

    model_config = ConfigDict(extra="forbid")

    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Probability that this extracted value is correct. "
            "1.0=exact unambiguous match; 0.8=clearly implied; "
            "0.6=indirect inference; 0.4=ambiguous best guess; "
            "below 0.4: prefer null over a guess."
        ),
    )
    evidence: Optional[str] = Field(
        None,
        description=(
            "A SINGLE string - never an array. "
            "Verbatim phrase or sentence from the report that supports this value. "
            "Required for every non-null extraction."
        ),
    )
    note: Optional[str] = Field(
        None,
        description=(
            "Reasoning note. Write ONLY when extraction was non-trivial: "
            "indirect language, conflicting statements, mapping from narrative to a code, "
            "or when you had to choose between two plausible codes. "
            "Leave null for clear-cut extractions."
        ),
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class YesNoCode(int, Enum):
    no = 0
    yes = 1
    not_assessed = 8
    missing = 9


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class FTLDTauPathology(BaseModel):
    """FTLD-tau pathology - parent + 10 subtypes (11 variables)."""
    model_config = ConfigDict(extra="forbid")

    NPFTDTAU: YesNoCode = Field(
        description=(
            "FTLD with tau pathology (FTLD-tau) or other tauopathy present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Code 1 if ANY FTLD-tau subtype or named tauopathy is present. "
            "Code 1 only when tau pathology is the primary or a "
            "major independent disease process (e.g., Pick's, PSP, CBD, CTE, MAPT mutation tauopathy). "
            "Do NOT code 1 for tau copathology that occurs incidentally in an AD or vascular case "
            "tau deposition secondary to AD is not FTLD-tau. "
            "If NPFTDTAU=0, all 10 children below must also be 0."
        ),
    )
    NACCPICK: YesNoCode = Field(
        description=(
            "Pick's disease (PiD) present as a neuropathological diagnosis. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Pick's disease = 3-repeat tauopathy with Pick bodies (round, tau-positive inclusions) "
            "and ballooned neurons, with frontotemporal predilection. "
            "Look for: 'Pick disease', 'Pick bodies', 'PiD', '3-repeat tauopathy consistent with PiD'. "
            "If NPFTDTAU=0 → code 0. "
            "Code 1 regardless of whether it is primary or contributing."            
        ),
    )
    NPFTDT2: YesNoCode = Field(
        description=(
            "Other 3R tauopathy (includes MAPT mutation tauopathy) present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPFTDTAU=0 → code 0."
        ),
    )
    NACCCBD: YesNoCode = Field(
        description=(
            "Corticobasal degeneration (CBD) present as a neuropathological diagnosis. "
            "Codes: 0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "CBD is a 4-repeat tauopathy with astrocytic plaques, ballooned neurons, "
            "and tau inclusions in cortex and basal ganglia. "
            "Look for: 'corticobasal degeneration', 'CBD', 'corticobasal syndrome with pathology', "
            "'4-repeat tauopathy consistent with CBD', 'astrocytic plaques'. "
            "Note: CBS (clinical syndrome) ≠ CBD (pathological diagnosis); "
            "code 1 only when the PATHOLOGICAL diagnosis is CBD. "
            "CBD often co-occurs with AD pathology - code 1 regardless of co-pathology. "
            "If NPFTDTAU=0 → code 0."
        ),
    )
    NACCPROG: YesNoCode = Field(
        description=(
            "Progressive supranuclear palsy (PSP) present as a neuropathological diagnosis. "
            "Codes: 0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "PSP = 4-repeat tauopathy with tufted astrocytes, coiled bodies, globose NFTs "
            "predominantly in basal ganglia, subthalamic nucleus, SN, brainstem. "
            "Look for: 'progressive supranuclear palsy', 'PSP', 'PSP-Richardson', "
            "'tufted astrocytes', 'globose tangles in STN'. "
            "If NPFTDTAU=0 → code 0."
            "Code 1 regardless of whether PSP is primary or contributing."
        ),
    )
    NPFTDT5: YesNoCode = Field(
        description=(
            "Argyrophilic grains present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for 'argyrophilic grain disease', 'AGD'. "
            "If NPFTDTAU=0 → code 0."
        ),
    )
    NPFTDT6: YesNoCode = Field(
        description=(
            "Other 4R tauopathy present (sporadic multiple systems tauopathy, "
            "globular glial tauopathy, MAPT mutation tauopathy). "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPFTDTAU=0 → code 0."
        ),
    )
    NPFTDT7: YesNoCode = Field(
        description=(
            "Chronic traumatic encephalopathy (CTE) present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for 'CTE', 'chronic traumatic encephalopathy'. "
            "If NPFTDTAU=0 → code 0."
        ),
    )
    NPFTDT8: YesNoCode = Field(
        description=(
            "ALS/parkinsonism-dementia complex tauopathy present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPFTDTAU=0 → code 0."
        ),
    )
    NPFTDT9: YesNoCode = Field(
        description=(
            "Tangle-dominant disease present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for 'tangle-dominant dementia', 'tangle only dementia'. "
            "If NPFTDTAU=0 → code 0."
        ),
    )
    NPFTDT10: YesNoCode = Field(
        description=(
            "Other 3R+4R mixed tauopathy present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPFTDTAU=0 → code 0."
        ),
    )

    @model_validator(mode="after")
    def validate_ftdtau_children(self) -> "FTLDTauPathology":
        """If NPFTDTAU=0, all children must be 0."""
        if self.NPFTDTAU is not None and self.NPFTDTAU.value == 0:
            children = [
                "NACCPICK", "NPFTDT2", "NACCCBD", "NACCPROG",
                "NPFTDT5", "NPFTDT6", "NPFTDT7", "NPFTDT8", "NPFTDT9", "NPFTDT10"
            ]
            for child in children:
                val = getattr(self, child)
                if val is not None and val.value == 1:
                    raise ValueError(
                        f"{child}=1 (Yes) but NPFTDTAU=0 (No FTLD-tau) - inconsistent."
                    )
        return self


class TDPFlag(BaseModel):
    """TDP-43 pathology flag - 1 variable."""
    model_config = ConfigDict(extra="forbid")

    NPFTDTDP: YesNoCode = Field(
        description=(
            "TDP-43 proteinopathy present as a neuropathological finding. "
            "Codes: 0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "Code 1 if TDP-43 pathology is present anywhere in the report, regardless of "
            "whether it is the primary diagnosis or an incidental sub-finding under another diagnosis. "
            "TDP-43 pathology includes cytoplasmic inclusions, neurites, and intranuclear inclusions "
            "staining with TDP-43 IHC; subtypes A-E (Mackenzie classification). "
            "Look for: 'FTLD-TDP', 'TDP-43 pathology', 'TDP-43 inclusions', 'TDP-43 immunoreactive', "
            "'TDP-43 protein deposits', 'ALS-TDP', 'pTDP-43'. "
            "If TDP-43 is not mentioned → code 0. "
            "Use 8 only if TDP-43 staining was explicitly not performed. "
            "Note: hippocampal sclerosis in the elderly often has associated TDP-43 pathology."
        ),
    )


class PrionFlag(BaseModel):
    """Prion disease flag - 1 variable."""
    model_config = ConfigDict(extra="forbid")

    NACCPRIO: YesNoCode = Field(
        description=(
            "Prion disease present as a neuropathological diagnosis. "
            "Codes: 0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "Prion diseases include CJD (sporadic, familial, iatrogenic, variant), "
            "GSS, fatal insomnia, and kuru. "
            "Neuropathological hallmarks: spongiform change (vacuolation), neuronal loss, "
            "gliosis, PrP immunoreactivity, amyloid plaques (kuru-type or florid). "
            "Look for: 'Creutzfeldt-Jakob disease', 'CJD', 'prion disease', "
            "'spongiform encephalopathy', 'PrP-positive', 'spongiform change consistent with prion'. "
            "If not mentioned in the report → code 0. "
            "Code 1 if prion disease is diagnosed (confirmed or suspected on pathology)."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level Pass 1 model
# ---------------------------------------------------------------------------

class Pass1bExtraction(BaseModel):
    """Pass 1b extraction - 13 variables.

    Covers: FTLD-tau subtypes, TDP-43 flag, and prion flag
    """
    model_config = ConfigDict(extra="forbid")

    ftld_tau: FTLDTauPathology = Field(
        description=(
            "FTLD-tau pathology - NPFTDTAU and 10 subtypes: "
            "NACCPICK, NPFTDT2, NACCCBD, NACCPROG, NPFTDT5, NPFTDT6, "
            "NPFTDT7, NPFTDT8, NPFTDT9, NPFTDT10"
        )
    )
    tdp_flag: TDPFlag = Field(
        description="TDP-43 pathology flag - NPFTDTDP"
    )
    prion_flag: PrionFlag = Field(
        description="Prion disease flag - NACCPRIO"
    )
    field_annotations: Optional[Dict[str, FieldAnnotation]] = Field(
        None,
        description=(
            "Per-variable extraction audit keyed by variable name. "
            "Provide an entry for every non-null extracted variable. "
            "For null variables, add an entry ONLY when the absence itself was ambiguous "
            "(e.g. a finding is mentioned but cannot be coded). "
            "Omit entries for null variables that are simply absent from the report. "
            "Each entry is an object with three keys: "
            "(1) 'confidence': float 0.0-1.0 - certainty this value is correct "
            "(2) 'evidence': string - verbatim phrase or sentence from the report that drove the decision "
            "(required for every non-null extraction; null only when a finding is unambiguously absent); "
            "(3) 'note': string or null - brief reasoning note ONLY when extraction was non-trivial "
            "(ambiguous language, had to choose between codes, conflicting statements); "
            "leave null for straightforward extractions. "
            'Example: {"NPFTDTAU": {"confidence": 0.95, "evidence": "no tau pathology identified", "note": null}}'
        ),
    )
    extraction_confidence: Optional[str] = Field(
        None,
        description="Overall case-level confidence: high, moderate, or low.",
    )
    extraction_notes: Optional[str] = Field(
        None,
        description="Case-level caveats - illegible sections, ambiguities affecting many variables.",
    )

    @model_validator(mode="after")
    def validate_confidence_value(self) -> "Pass1bExtraction":
        if self.extraction_confidence and self.extraction_confidence not in {
            "high", "moderate", "low"
        }:
            raise ValueError("extraction_confidence must be 'high', 'moderate', or 'low'")
        return self

    @model_validator(mode="after")
    def enforce_null_policy(self) -> "Pass1bExtraction":
        """Non-nullable fields must not be null - use 0, 8, or 9 instead."""
        sections = [
            self.ftld_tau, self.tdp_flag,
            self.prion_flag,
        ]
        for section in sections:
            for field_name in section.model_fields:
                if getattr(section, field_name) is None:
                    raise ValueError(
                        f"{field_name} must not be null - use 0, 8, or 9, etc. instead (depending on variable)."
                    )
        return self


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_pass1b_model() -> Type[BaseModel]:
    """Return the top-level extraction model class."""
    return Pass1bExtraction


def _unwrap_optional(annotation: Any):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union and type(None) in args:
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1:
            return inner[0], True
    return annotation, False


def _describe_field(name: str, info, indent: int = 0) -> List[str]:
    """Recursively describe a Pydantic field for the LLM prompt."""
    prefix = "  " * indent
    lines: List[str] = []
    desc = info.description or ""
    raw_ann = info.annotation
    inner, is_optional = _unwrap_optional(raw_ann)
    opt_tag = " (optional)" if is_optional else ""

    if isinstance(inner, type) and issubclass(inner, Enum):
        allowed = [e.value for e in inner]
        lines.append(f"{prefix}- {name}{opt_tag}: {desc}. Allowed values: {allowed}")
        return lines

    if isinstance(inner, type) and issubclass(inner, BaseModel):
        lines.append(f"{prefix}- {name}: {desc}")
        for sub_name, sub_info in inner.model_fields.items():
            lines.extend(_describe_field(sub_name, sub_info, indent + 1))
        return lines

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


def build_pass1b_format_instructions() -> str:
    """Walk the Pydantic schema and emit LLM-facing format instructions."""
    model = Pass1bExtraction
    lines = [
        "PASS 1b - Extract ONLY the variables listed below.",
        "Respond with a single JSON object with EXACTLY these top-level keys:",
        "",
        "  ftld_tau, tdp_flag, prion_flag,",
        "  field_annotations, extraction_confidence, extraction_notes",
        "",
        "Each top-level key maps to a nested object containing its variables.",
        "NEVER output variables at the root level - they must always be nested.",
        "",
        "PRIORITY fields - extract these with the highest care:",
        "  NPFTDTAU, NACCPICK, NACCCBD, NACCPROG,",
        "",
        "NULL POLICY:",
        "  For every other variable: use an explicit numeric code - never null.",
        "  If a finding was assessed and absent → 0",
        "  If a structure was explicitly not examined or stain not performed → 8",
        "  If examined or mentioned but severity/result cannot be determined → 9",
        "",
        "FTLD-TAU CHILDREN: if NPFTDTAU=0, all 10 children must also be 0.",
        "",
    ]
    for name, info in model.model_fields.items():
        raw_ann = info.annotation
        inner, _ = _unwrap_optional(raw_ann)
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            lines.append(f"[{name}]")
            for sub_name, sub_info in inner.model_fields.items():
                lines.extend(_describe_field(sub_name, sub_info, indent=1))
            lines.append("")
        else:
            lines.extend(_describe_field(name, info, indent=0))
    lines += [
        "",
        "Do not invent information. Extract only what is explicitly stated or clearly implied.",
        "All integer codes must be exact values from the allowed set in each description.",
    ]
    return "\n".join(lines)