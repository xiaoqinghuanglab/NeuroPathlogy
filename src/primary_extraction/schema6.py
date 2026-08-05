"""schema6.py - Pass 6: Genetics, family history, full autopsy (12 vars).

Pass 6 variables:
  Autopsy:     NPFAUT, NPFAUT1, NPFAUT2, NPFAUT3, NPFAUT4
  Family hist: NPGENE, NPFHSPEC
  Genetics:    NPTAUHAP, NPPRNP, NPCHROM, NPPDXP, NPPDXQ

Notes:
  NPFHSPEC: blank (null) if NPGENE = 1, 3, or 9 (per CSV rule - absent from PDF).
  NPFAUT1-4: null if NPFAUT != 1.
  NPCHROM codes 8 and 9 are real clinical findings (Huntingtin mutation and Notch 3 mutation (CADASIL))
    - NOT null-equivalent. Special not-assessed code is 50, unknown is 99.
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
# Sub-models
# ---------------------------------------------------------------------------

class FullAutopsy(BaseModel):
    """Full autopsy findings - 5 variables."""
    model_config = ConfigDict(extra="forbid")

    NPFAUT: int = Field(
        description=(
            "Full autopsy performed (i.e., complete body autopsy including non-brain organs). "
            "Codes: 0=No; 1=Yes; 9=Missing/unknown. "
            "Code 1 only when the report explicitly describes examination of organs beyond the brain. "
            "Brain-only autopsy = code 0. NPFAUT1-4 are blank when NPFAUT=0. "
            "If not mentioned → code 0."
        ),
    )
    NPFAUT1: Optional[str] = Field(
        None,
        description=(
            "First major finding from full autopsy - free-text. "
            "Populate ONLY when NPFAUT=1. Otherwise null."
        ),
    )
    NPFAUT2: Optional[str] = Field(
        None,
        description=(
            "Second major finding from full autopsy - free-text. "
            "Populate ONLY when NPFAUT=1 and at least two findings. Otherwise null."
        ),
    )
    NPFAUT3: Optional[str] = Field(
        None,
        description=(
            "Third major finding from full autopsy - free-text. "
            "Populate ONLY when NPFAUT=1 and at least three findings. Otherwise null."
        ),
    )
    NPFAUT4: Optional[str] = Field(
        None,
        description=(
            "Fourth major finding from full autopsy - free-text. "
            "Populate ONLY when NPFAUT=1 and at least four findings. Otherwise null."
        ),
    )

    @field_validator("NPFAUT")
    @classmethod
    def val_npfaut(cls, v):
        if v not in {0, 1, 9}:
            raise ValueError(f"NPFAUT must be 0, 1, or 9, got {v}")
        return v

    @model_validator(mode="after")
    def validate_faut_children(self) -> "FullAutopsy":
        if self.NPFAUT != 1:
            for child in ["NPFAUT1", "NPFAUT2", "NPFAUT3", "NPFAUT4"]:
                if getattr(self, child) is not None:
                    raise ValueError(
                        f"{child} must be null when NPFAUT != 1, got '{getattr(self, child)}'."
                    )
        return self


class FamilyHistory(BaseModel):
    """Family history - 2 variables."""
    model_config = ConfigDict(extra="forbid")

    NPGENE: int = Field(
        description=(
            "Family history of neurodegenerative disorder. "
            "Codes: 1=Family history of similar neurodegenerative disorder; "
            "2=Family history of other (dissimilar) neurodegenerative disorder; "
            "3=No family history of similar or dissimilar disorder; "
            "4=Family history of BOTH similar and dissimilar disorders; "
            "9=Family history unknown/not available/missing. "
            "Look for family history section in the report."
        ),
    )
    NPFHSPEC: Optional[str] = Field(
        None,
        description=(
            "Specify family history - free-text. "
            "Code NPFHSPEC - null if NPGENE = 1, 3, or 9. "
            "Populate ONLY when NPGENE = 2 or 4 (dissimilar or both types of FH). "
            "Describe the specific family history when populated."
        ),
    )

    @field_validator("NPGENE")
    @classmethod
    def val_npgene(cls, v):
        if v not in {1, 2, 3, 4, 9}:
            raise ValueError(f"NPGENE must be 1, 2, 3, 4, or 9, got {v}")
        return v

    @model_validator(mode="after")
    def validate_npfhspec(self) -> "FamilyHistory":
        """NPFHSPEC must be null when NPGENE = 1, 3, or 9 (CSV rule)."""
        if self.NPGENE in {1, 3, 9} and self.NPFHSPEC is not None:
            raise ValueError(
                f"NPFHSPEC must be null when NPGENE={self.NPGENE} "
                f"(conditional blank rule: blank if NPGENE = 1, 3, or 9)."
            )
        return self


class GeneticsFindings(BaseModel):
    """Genetic and chromosomal findings - 5 variables."""
    model_config = ConfigDict(extra="forbid")

    NPTAUHAP: int = Field(
        description=(
            "Tau haplotype. "
            "Codes: 1=H1/H1; 2=H1/H2; 3=H2/H2; 4=Other polymorphism (e.g. A0); "
            "9=Missing/unknown/not assessed. "
            "Look for 'tau haplotype', 'MAPT haplotype', 'H1', 'H2'. "
            "If not mentioned → code 9."
        ),
    )
    NPPRNP: int = Field(
        description=(
            "PRNP codon 129 genotype. "
            "Codes: 1=MM (Met/Met); 2=MV (Met/Val); 3=VV (Val/Val); "
            "9=Missing/unknown/not assessed. "
            "Look for 'PRNP', 'codon 129', 'Met/Val'. "
            "If not mentioned → code 9."
        ),
    )
    NPCHROM: int = Field(
        description=(
            "Genetic or chromosomal abnormalities. "
            "Codes: 1=APP mutation; 2=PS1 (PSEN1) mutation; 3=PS2 (PSEN2) mutation; "
            "4=Tau (MAPT) mutation; 5=Alpha-synuclein mutation; 6=Parkin mutation; "
            "7=PRNP mutation; 8=Huntingtin mutation; 9=Notch3 mutation (CADASIL); "
            "10=Other known genetic mutation (e.g., ABri, neuroserpin); 11=Down syndrome; "
            "12=Other chromosomal abnormality; 13=No known genetic or chromosomal abnormality; "
            "50=Not assessed; 99=Missing/unknown. "
            "IMPORTANT: codes 8 and 9 here are REAL clinical findings (Huntingtin mutation "
            "and Notch3 mutation (CADASIL)) - they are NOT null-equivalent in this variable. "
            "If genetic testing not mentioned → code 50 (not assessed)."
        ),
    )
    NPPDXP: int = Field(
        description=(
            "AD-related genes identified (APOE, APP, PSEN1, PSEN2, etc.). "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for APOE genotype, APP mutation, presenilin mutations. "
            "If not mentioned → code 0."
        ),
    )
    NPPDXQ: int = Field(
        description=(
            "FTLD-related genes identified (GRN, C9orf72, MAPT, etc.). "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for GRN mutation, C9orf72 expansion, MAPT mutation. "
            "If not mentioned → code 0."
        ),
    )

    @field_validator("NPTAUHAP")
    @classmethod
    def val_tauhap(cls, v):
        if v not in {1, 2, 3, 4, 9}:
            raise ValueError(f"NPTAUHAP must be 1, 2, 3, 4, or 9, got {v}")
        return v

    @field_validator("NPPRNP")
    @classmethod
    def val_npprnp(cls, v):
        if v not in {1, 2, 3, 9}:
            raise ValueError(f"NPPRNP must be 1, 2, 3, or 9, got {v}")
        return v

    @field_validator("NPCHROM")
    @classmethod
    def val_npchrom(cls, v):
        if v not in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 99}:
            raise ValueError(
                f"NPCHROM must be 1-13, 50, or 99, got {v}"
            )
        return v

    @field_validator("NPPDXP", "NPPDXQ")
    @classmethod
    def val_gene_flags(cls, v):
        if v not in {0, 1, 8, 9}:
            raise ValueError(f"Gene flag must be 0, 1, 8, or 9, got {v}")
        return v


# ---------------------------------------------------------------------------
# Top-level Pass 6 model
# ---------------------------------------------------------------------------

class Pass6Extraction(BaseModel):
    """Pass 6 extraction - 12 variables.

    Covers: full autopsy findings, family history, and genetic/chromosomal findings.
    This is the final LLM pass. NACCVASC and NACCBRNN are computed programmatically
    in main.py after all 6 passes complete.
    """
    model_config = ConfigDict(extra="forbid")

    full_autopsy: FullAutopsy = Field(
        description="NPFAUT, NPFAUT1, NPFAUT2, NPFAUT3, NPFAUT4"
    )
    family_history: FamilyHistory = Field(
        description="NPGENE, NPFHSPEC"
    )
    genetics: GeneticsFindings = Field(
        description="NPTAUHAP, NPPRNP, NPCHROM, NPPDXP, NPPDXQ"
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
    def validate_confidence_value(self) -> "Pass6Extraction":
        if self.extraction_confidence and self.extraction_confidence not in {
            "high", "moderate", "low"
        }:
            raise ValueError("extraction_confidence must be 'high', 'moderate', or 'low'")
        return self

    @model_validator(mode="after")
    def enforce_null_policy(self) -> "Pass6Extraction":
        """Non-nullable fields must not be null."""
        nullable = {"NPFAUT1", "NPFAUT2", "NPFAUT3", "NPFAUT4", "NPFHSPEC"}
        sections = [self.full_autopsy, self.family_history, self.genetics]
        for section in sections:
            for field_name in section.model_fields:
                if field_name not in nullable and getattr(section, field_name) is None:
                    raise ValueError(
                        f"{field_name} must not be null - use appropriate code (0, 9, 50, 99, etc.) (depending on variable)."
                    )
        return self


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_pass6_model() -> Type[BaseModel]:
    """Return the top-level extraction model class."""
    return Pass6Extraction


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


def build_pass6_format_instructions() -> str:
    """Walk the Pydantic schema and emit LLM-facing format instructions."""
    model = Pass6Extraction
    lines = [
        "PASS 6 - Extract ONLY the genetics, family history, and autopsy variables below.",
        "This is the final extraction pass.",
        "Respond with a single JSON object with EXACTLY these top-level keys:",
        "",
        "  full_autopsy, family_history, genetics,",
        "  field_annotations, extraction_confidence, extraction_notes",
        "",
        "Each top-level key maps to a nested object containing its variables.",
        "NEVER output variables at the root level - they must always be nested.",
        "",
        "NULL POLICY:",
        "  Nullable fields: NPFAUT1-4 (null unless NPFAUT=1), NPFHSPEC (see rule below).",
        "  All other fields must have explicit numeric codes.",
        "",
        "NPFHSPEC CONDITIONAL RULE:",
        "  null if NPGENE = 1, 3, or 9.",
        "  Populate only when NPGENE = 2 or 4.",
        "",
        "NPFAUT: codes 0=No, 1=Yes, 9=Missing/unknown. No code 8.",
        "NPFAUT1-4: null if NPFAUT != 1.",
        "",
        "NPTAUHAP: 1=H1/H1, 2=H1/H2, 3=H2/H2, 4=Other polymorphism (e.g., A0), 9=Missing/Unknown/not assessed.",
        "  If not mentioned → code 9.",
        "",
        "NPCHROM has 15 distinct codes - read description carefully.",
        "  Codes 8 and 9 are REAL clinical findings (Huntingtin mutation and Notch 3 mutation (CADASIL)).",
        "  Not-assessed = 50. Missing/unknown = 99.",
        "  If genetic testing not mentioned → code 50.",
        "",
        "NPPDXP, NPPDXQ: if gene testing not mentioned → code 0.",
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