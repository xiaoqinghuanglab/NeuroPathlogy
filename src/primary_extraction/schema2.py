"""schema2.py - Pass 2: Other FTLD subtypes, TDP-43 distribution, legacy FTLD flags (17 vars).

Pass 2 variables:
  Other FTLD:     NPOFTD, NPOFTD1, NPOFTD2, NPOFTD3, NPOFTD4, NPOFTD5
  FTLD misc:      NPFTDNO, NPFTDSPC, NPFTD, NPTAU, NPFRONT, NPALSMND
  TDP-43 dist:    NPTDPA, NPTDPB, NPTDPC, NPTDPD, NPTDPE
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

class OtherFTLD(BaseModel):
    """Other FTLD subtypes - parent + 5 children (6 variables)."""
    model_config = ConfigDict(extra="forbid")

    NPOFTD: YesNoCode = Field(
        description=(
            "Other FTLD (not tau, not TDP-43) present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Code 1 if any NPOFTD subtype is present. "
            "If NPOFTD=0, all 5 children below must also be 0."
        ),
    )
    NPOFTD1: YesNoCode = Field(
        description=(
            "Atypical FTLD-U present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Ubiquitin-positive but tau/TDP-43/FUS negative inclusions - atypical pattern. "
            "If NPOFTD=0 → code 0."
        ),
    )
    NPOFTD2: YesNoCode = Field(
        description=(
            "NIFID (neuronal intermediate filament inclusion disease) present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "FUS proteinopathy with neuronal intermediate filament inclusions. "
            "Note: NIFID is a FUS proteinopathy - tau IHC reactivity as a secondary finding "
            "without a named tauopathy diagnosis does NOT justify code 1 here. "
            "If NPOFTD=0 → code 0."
        ),
    )
    NPOFTD3: YesNoCode = Field(
        description=(
            "BIBD (basophilic inclusion body disease) present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "FUS proteinopathy with basophilic inclusions. "
            "If NPOFTD=0 → code 0."
        ),
    )
    NPOFTD4: YesNoCode = Field(
        description=(
            "FTLD-UPS present (ubiquitin-proteasome system [ubiquitin or p62positive, tau/TDP-43/FUS negative inclusions]). "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPOFTD=0 → code 0."
        ),
    )
    NPOFTD5: YesNoCode = Field(
        description=(
            "FTLD-NOS present (includes dementia lacking distinctive histology [DLDH] or "
            "FTLD with no inclusions [FTLD-NI] detected by tau, TDP-43, or ubiquitin/p62 IHC) "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPOFTD=0 → code 0."
        ),
    )

    @model_validator(mode="after")
    def validate_oftd_children(self) -> "OtherFTLD":
        if self.NPOFTD is not None and self.NPOFTD.value == 0:
            children = ["NPOFTD1", "NPOFTD2", "NPOFTD3", "NPOFTD4", "NPOFTD5"]
            for child in children:
                val = getattr(self, child)
                if val is not None and val.value == 1:
                    raise ValueError(
                        f"{child}=1 (Yes) but NPOFTD=0 (No other FTLD) - inconsistent."
                    )
        return self


class FTLDFlags(BaseModel):
    """Legacy FTLD and related flags - 6 variables (v1-9 form)."""
    model_config = ConfigDict(extra="forbid")

    NPFTDNO: int = Field(
        description=(
            "FTD with no distinctive histopathology present. "
            "Codes: 1=Yes; 2=No; 3=Not assessed; 9=Missing/unknown. "
            "Use when FTD clinical syndrome present but no specific inclusion type identified. "
            "If not mentioned → code 2."
        ),
    )
    NPFTDSPC: int = Field(
        description=(
            "FTD 'not otherwise specified' present. "
            "Codes: 1=Yes; 2=No; 3=Not assessed; 9=Missing/unknown. "
            "If not mentioned → code 2."
        ),
    )
    NPFTD: int = Field(
        description=(
            "FTD with ubiquitin-positive, tau-negative inclusions present "
            "Codes: 1=FTD with motor neuron disease; 2=FTD without motor neuron disease; "
            "3=None present; 4=Not assessed; 9=Missing/unknown. "
            "Code 3 when FTLD pathology (e.g., FTLD-TDP) is present "
            "but no motor neuron disease is reported. "
            "Code 1 only when both FTD pathology and MND coexist. "
            "Code 2 only when FTD without MND is explicitly the primary diagnosis. "
            "If not mentioned → code 3."
        ),
    )
    NPTAU: int = Field(
        description=(
            "Other tauopathy present - specifically tangle-only dementia (neurofibrillary tangle dementia) "
            "or argyrophilic grain disease (AGD) as a standalone or primary diagnosis. "
            "Codes: 1=Yes; 2=No; 3=Not assessed; 9=Missing/unknown. "
            "Code 1 ONLY for: "
            "(1) Tangle-only dementia / neurofibrillary tangle dementia (NFT dementia) - "
            "high NFT burden without amyloid plaques, not meeting AD criteria. "
            "(2) Argyrophilic grain disease (AGD) as a named diagnosis. "
            "Do NOT code 1 for: "
            "- PART (Primary age-related tauopathy) - this is incidental aging-related tau, not a disease. "
            "- Tau copathology secondary to AD, CBD, PSP, or Pick's disease. "
            "If not mentioned → code 2."
        ),
    )
    NPFRONT: int = Field(
        description=(
            "Frontotemporal dementia and parkinsonism with tau-positive or "
            "argyrophilic inclusions present. "
            "Codes: 1=Yes; 2=No; 3=Not assessed; 9=Missing/unknown. "
            "If not mentioned → code 2."
        ),
    )
    NPALSMND: int = Field(
        description=(
            "ALS/motor neuron disease (MND) present as a neuropathological finding. "
            "Codes: 0=No; 1=Yes with TDP-43 inclusions in motor neurons; "
            "2=Yes with FUS inclusions; 3=Yes with SOD1 inclusions; "
            "4=Yes with other inclusions; 5=Yes with no specific inclusions; "
            "8=Not assessed; 9=Missing/unknown. "
            "Look for 'ALS', 'amyotrophic lateral sclerosis', 'motor neuron disease', 'MND'. "
            "If not mentioned → code 0."
        ),
    )

    @field_validator("NPFTDNO", "NPFTDSPC", "NPTAU", "NPFRONT")
    @classmethod
    def val_ftld(cls, v):
        if v not in {1, 2, 3, 9}:
            raise ValueError(f"FTLD flag must be 1, 2, 3, or 9, got {v}")
        return v

    @field_validator("NPFTD")
    @classmethod
    def val_npftd(cls, v):
        if v not in {1, 2, 3, 4, 9}:
            raise ValueError(f"NPFTD must be 1, 2, 3, 4, or 9, got {v}")
        return v

    @field_validator("NPALSMND")
    @classmethod
    def val_npalsmnd(cls, v):
        if v not in {0, 1, 2, 3, 4, 5, 8, 9}:
            raise ValueError(f"NPALSMND must be 0-5, 8, or 9, got {v}")
        return v


class TDP43Distribution(BaseModel):
    """TDP-43 immunoreactive inclusion distribution - 5 variables."""
    model_config = ConfigDict(extra="forbid")

    NPTDPA: YesNoCode = Field(
        description=(
            "TDP-43 immunoreactive inclusions present in spinal cord. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for TDP-43 inclusions specifically in spinal cord anterior horn cells "
            "or spinal cord motor neurons. "
            "If TDP-43 staining not performed → code 8. "
            "If TDP-43 staining performed but spinal cord not mentioned → code 0."
        ),
    )
    NPTDPB: YesNoCode = Field(
        description=(
            "TDP-43 immunoreactive inclusions present in amygdala. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If TDP-43 staining not performed → code 8. "
            "If performed but amygdala not mentioned → code 0."
        ),
    )
    NPTDPC: YesNoCode = Field(
        description=(
            "TDP-43 immunoreactive inclusions present in hippocampus. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Includes dentate gyrus, CA regions, subiculum. "
            "If TDP-43 staining not performed → code 8. "
            "If performed but hippocampus not mentioned → code 0."
        ),
    )
    NPTDPD: YesNoCode = Field(
        description=(
            "TDP-43 immunoreactive inclusions present in entorhinal or inferior temporal cortex. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If TDP-43 staining not performed → code 8. "
            "If performed but these regions not mentioned → code 0."
        ),
    )
    NPTDPE: YesNoCode = Field(
        description=(
            "TDP-43 immunoreactive inclusions present in neocortex. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Neocortex includes frontal, temporal, parietal, occipital cortices. "
            "If TDP-43 staining not performed → code 8. "
            "If performed but neocortex not mentioned → code 0."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level Pass 2 model
# ---------------------------------------------------------------------------

class Pass2Extraction(BaseModel):
    """Pass 2 extraction - 17 variables.

    Covers: other FTLD subtypes (NPOFTD + children), legacy FTLD flags,
    ALS/MND flag, and TDP-43 inclusion distribution by region.
    """
    model_config = ConfigDict(extra="forbid")

    other_ftld: OtherFTLD = Field(
        description=(
            "Other FTLD subtypes - "
            "NPOFTD, NPOFTD1, NPOFTD2, NPOFTD3, NPOFTD4, NPOFTD5"
        )
    )
    ftld: FTLDFlags = Field(
        description=(
            "FTLD and ALS/MND flags - "
            "NPFTDNO, NPFTDSPC, NPFTD, NPTAU, NPFRONT, NPALSMND"
        )
    )
    tdp43_distribution: TDP43Distribution = Field(
        description=(
            "TDP-43 inclusion distribution by region - "
            "NPTDPA (spinal cord), NPTDPB (amygdala), NPTDPC (hippocampus), "
            "NPTDPD (entorhinal/inferior temporal), NPTDPE (neocortex)"
        )
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
    def validate_confidence_value(self) -> "Pass2Extraction":
        if self.extraction_confidence and self.extraction_confidence not in {
            "high", "moderate", "low"
        }:
            raise ValueError("extraction_confidence must be 'high', 'moderate', or 'low'")
        return self

    @model_validator(mode="after")
    def enforce_null_policy(self) -> "Pass2Extraction":
        """Non-nullable fields must not be null - use 0, 8, or 9 instead."""
        sections = [self.other_ftld, self.ftld, self.tdp43_distribution]
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

def get_pass2_model() -> Type[BaseModel]:
    """Return the top-level extraction model class."""
    return Pass2Extraction


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


def build_pass2_format_instructions() -> str:
    """Walk the Pydantic schema and emit LLM-facing format instructions."""
    model = Pass2Extraction
    lines = [
        "PASS 2 - Extract ONLY the variables listed below.",
        "Respond with a single JSON object with EXACTLY these top-level keys:",
        "",
        "  other_ftld, ftld, tdp43_distribution,",
        "  field_annotations, extraction_confidence, extraction_notes",
        "",
        "Each top-level key maps to a nested object containing its variables.",
        "NEVER output variables at the root level - they must always be nested.",
        "",
        "NULL POLICY: No variables in this pass are nullable.",
        "  Use the appropriate not-assessed or absent code. Never null.",
        "",
        "NPOFTD CHILDREN: if NPOFTD=0, all 5 children must also be 0.",
        "",
        "FTLD FLAGS (NPFTDNO, NPFTDSPC, NPTAU, NPFRONT):",
        "  Use 1=Yes, 2=No, 3=Not assessed, 9=Missing. NOT 0/8.",
        "  If not mentioned → code 2 (No).",
        "",
        "NPFTD: 1=FTD with MND, 2=FTD without MND, 3=None present, 4=Not assessed, 9=Missing.",
        "  If not mentioned → code 3 (None present).",
        "",
        "NPALSMND: 0=No, 1=Yes TDP-43, 2=Yes FUS, 3=Yes SOD1, 4=Yes other, 5=Yes no specific, 8=Not assessed, 9=Missing.",
        "  If not mentioned → code 0.",
        "",
        "TDP-43 DISTRIBUTION: if TDP-43 staining was not performed → code 8 for all 5 regions.",
        "  If staining performed but a region is not mentioned → code 0 for that region.",
        "",
        "NIFID (NPOFTD2): FUS proteinopathy only. Tau secondary finding alone does NOT qualify.",
        "",
        "FIELD NAMING RULES:",
        "  Write each key exactly once.",
        "  Do not duplicate keys.",
        "  Do not omit any key from the requested block.",
        "",
        "TDP-43 DISTRIBUTION KEYS ARE DISTINCT:",
        "  NPTDPA = spinal cord - TDP-43 type A",
        "  NPTDPB = amygdala TDP-43 type B",
        "  NPTDPC = hippocampus TDP-43 type C",
        "  NPTDPD = entorhinal or inferior temporal cortex TDP-43 type D",
        "  NPTDPE = neocortex TDP-43 type E",
        "  Every one of NPTDPA, NPTDPB, NPTDPC, NPTDPD, and NPTDPE must appear once only.",
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