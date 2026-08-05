"""schema5.py - Pass 5: Metadata, staining methods, tissue banking (27 vars).

Pass 5 variables:
  Case metadata:    NPFORMVER, NACCID, NACCADC, NACCDAGE, NACCMOD, NACCYOD, NACCINT
  Antibodies:       NPTAN, NPTANX, NPABAN, NPABANX, NPASAN, NPASANX, NPTDPAN, NPTDPANX
  Histochem stains: NPHISMB, NPHISG, NPHISSS, NPHIST, NPHISO, NPHISOX
  Banking/tissue:   NACCBNKF, NPBNKB, NACCFORM, NACCPARA, NACCCSFP, NPBNKF
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

class CaseMetadata(BaseModel):
    """Case-level administrative metadata - 7 variables."""
    model_config = ConfigDict(extra="forbid")

    NPFORMVER: Optional[int] = Field(
        None,
        description=(
            "NP form version number. "
            "Codes: 1=v1, 7=v7, 8=v8, 9=v9, 10=v10, 11=v11. "
            "Look for form version stated in the report header. "
            "Null if not stated."
        ),
    )
    NACCID: Optional[str] = Field(
        None,
        description=(
            "NACC subject ID. "
            "Format: prefix 'NACC' followed by exactly six digits (e.g. NACC123456). "
            "Look for a NACC ID in the report header. "
            "Null if not stated."
        ),
    )
    NACCADC: Optional[int] = Field(
        None,
        description=(
            "ADC (Alzheimer's Disease Center) code at which subject was seen. "
            "Integer code 100-9999. Null if not stated in the report."
        ),
    )
    NACCDAGE: Optional[int] = Field(
        None,
        description=(
            "Age at death in years. Valid range: 15-120. "
            "Special codes: 888=Not applicable, 999=Unknown. "
            "Look for 'age at death', 'died at age X', or age in demographics. "
            "Null if not stated."
        ),
    )
    NACCMOD: Optional[int] = Field(
        None,
        description=(
            "Month of death. Codes: 0-12 (0=unknown month, 1=January through 12=December). "
            "Special codes: 88=Not applicable, 99=Unknown. "
            "Look for date of death in report header. "
            "Null if not stated."
        ),
    )
    NACCYOD: Optional[int] = Field(
        None,
        description=(
            "Year of death. Four-digit integer (e.g. 2018). "
            "Must not precede 1970. "
            "Special codes: 8888=Not applicable, 9999=Unknown. "
            "Look for date of death in report header. "
            "Null if not stated."
        ),
    )
    NACCINT: Optional[float] = Field(
        None,
        description=(
            "Time interval in months between last visit and death. "
            "Valid range: 0 or greater (no upper limit). "
            "Special codes: 8888=Not applicable, 9999=Unknown. "
            "Look for 'interval since last visit', 'time since last exam'. "
            "Null if not stated."
        ),
    )

    @field_validator("NPFORMVER")
    @classmethod
    def val_npformver(cls, v):
        if v is not None and v not in {1, 7, 8, 9, 10, 11}:
            raise ValueError(f"NPFORMVER must be 1, 7, 8, 9, 10, or 11, got {v}")
        return v

    @field_validator("NACCMOD")
    @classmethod
    def val_month(cls, v):
        if v is not None and v not in set(range(0, 13)) | {88, 99}:
            raise ValueError(f"NACCMOD must be 0-12, 88, or 99, got {v}")
        return v

    @field_validator("NACCDAGE")
    @classmethod
    def val_dage(cls, v):
        if v is not None and not ((15 <= v <= 120) or v in {888, 999}):
            raise ValueError(f"NACCDAGE must be 15-120, 888, or 999, got {v}")
        return v

    @field_validator("NACCYOD")
    @classmethod
    def val_year(cls, v):
        if v is not None and not ((1970 <= v <= 2100) or v in {8888, 9999}):
            raise ValueError(f"NACCYOD must be 1970-2100, 8888, or 9999, got {v}")
        return v

    @field_validator("NACCINT")
    @classmethod
    def val_naccint(cls, v):
        if v is not None and not (v >= 0 or v in {8888, 9999}):
            raise ValueError(f"NACCINT must be >= 0, 8888, or 9999, got {v}")
        return v


class AntibodyMethods(BaseModel):
    """IHC antibody methods used - 8 variables."""
    model_config = ConfigDict(extra="forbid")

    NPTAN: Optional[int] = Field(
        None,
        description=(
            "Tau antibody used for scoring. "
            "Codes: 1=Non-phospho specific; 2=PHF1; 3=CP13; 4=AT8; "
            "7=Other (specify NPTANX); 8=Not assessed. "
            "Look for tau antibody name in staining methods section. "
            "If a tau antibody is mentioned but does not match any of the named codes, code 7 and record the name in NPTANX. "
            "Null if not mentioned."
        ),
    )
    NPTANX: Optional[str] = Field(
        None,
        description=(
            "Tau antibody other - free-text specify. "
            "Populate ONLY when NPTAN=7. Record the exact antibody name as written in the report. "
            "Null in all other cases."
        ),
    )
    NPABAN: Optional[int] = Field(
        None,
        description=(
            "Amyloid-beta antibody used for scoring. "
            "Codes: 1=4G8; 2=10D5; 7=Other (specify NPABANX); 8=Not assessed. "
            "Look in the staining/methods section for antibodies against amyloid-beta, Aβ, β-amyloid, or β-protein. "
            "If the named clone is not 4G8 or 10D5, code 7 and record the exact clone/name in NPABANX. "
            "Common 'Other' examples: NAB-228, 21F12, 6F/3D, 4D7S. "
            "Code 8 only if the report explicitly states amyloid-beta IHC was not performed. "
            "Null if amyloid-beta IHC is not mentioned at all."
        ),
    )
    NPABANX: Optional[str] = Field(
        None,
        description=(
            "Amyloid-beta antibody other - free-text specify. "
            "Populate ONLY when NPABAN=7. "
            "Record the exact antibody clone or name as written in the report (e.g. 'NAB-228', '21F12'). "
            "Null in all other cases."
        ),
    )
    NPASAN: Optional[int] = Field(
        None,
        description=(
            "Alpha-synuclein antibody used for scoring. "
            "Codes: 1=Non-phospho specific (e.g. LB509); 2=Phospho-specific (e.g. pSYN#64); "
            "7=Other (specify NPASANX); 8=Not assessed. "
            "Look for 'alpha-synuclein', 'a-synuclein', or 'synuclein' antibody in the staining/methods section. "
            "If a synuclein antibody is mentioned but no specific clone is named, default to 1 (non-phospho specific). "
            "Only code 7 if an antibody name is explicitly given that does not match LB509 or pSYN#64 type antibodies. "
            "Null if no synuclein antibody is mentioned at all."
        ),
    )
    NPASANX: Optional[str] = Field(
        None,
        description=(
            "Alpha-synuclein antibody other - free-text specify. "
            "Populate ONLY when NPASAN=7. Record the exact antibody name as written in the report. "
            "Null in all other cases."
        ),
    )
    NPTDPAN: Optional[int] = Field(
        None,
        description=(
            "TDP-43 antibody used. "
            "Codes: 1=Non-phospho specific; 2=Phospho-specific (e.g. pS409/410); "
            "7=Other (specify NPTDPANX); 8=Not assessed. "
            "Look for 'TDP-43', 'pTDP-43', 'pS409/410'. Null if not mentioned."
        ),
    )
    NPTDPANX: Optional[str] = Field(
        None,
        description=(
            "TDP-43 antibody other - free-text specify. "
            "Populate ONLY when NPTDPAN=7. Record the exact antibody name as written in the report. "
            "Null in all other cases."
        ),
    )

    @field_validator("NPTAN")
    @classmethod
    def val_nptan(cls, v):
        if v is not None and v not in {1, 2, 3, 4, 7, 8}:
            raise ValueError(f"NPTAN must be 1, 2, 3, 4, 7, or 8, got {v}")
        return v

    @field_validator("NPABAN")
    @classmethod
    def val_npaban(cls, v):
        if v is not None and v not in {1, 2, 7, 8}:
            raise ValueError(f"NPABAN must be 1, 2, 7, or 8, got {v}")
        return v

    @field_validator("NPASAN", "NPTDPAN")
    @classmethod
    def val_synuclein_tdp_ab(cls, v):
        if v is not None and v not in {1, 2, 7, 8}:
            raise ValueError(f"Antibody field must be 1, 2, 7, or 8, got {v}")
        return v


class HistochemStains(BaseModel):
    """Histochemical stains used - 6 variables."""
    model_config = ConfigDict(extra="forbid")

    NPHISMB: int = Field(
        description=(
            "Modified Bielschowsky silver stain used. "
            "Codes: 0=No; 1=Yes. "
            "Look for 'Bielschowsky', 'modified Bielschowsky'. "
            "If not mentioned → code 0."
        ),
    )
    NPHISG: int = Field(
        description=(
            "Gallyas silver stain used. "
            "Codes: 0=No; 1=Yes. "
            "Look for 'Gallyas'. If not mentioned → code 0."
        ),
    )
    NPHISSS: int = Field(
        description=(
            "Other silver stain used. "
            "Codes: 0=No; 1=Yes. "
            "Look for other silver stain methods not listed above. "
            "If not mentioned → code 0."
        ),
    )
    NPHIST: int = Field(
        description=(
            "Thioflavin stain used (thioflavin S or T). "
            "Codes: 0=No; 1=Yes. "
            "Look for 'thioflavin S', 'thioflavin T', 'ThioS', 'ThioT'. "
            "If not mentioned → code 0."
        ),
    )
    NPHISO: int = Field(
        description=(
            "Other histochemical stain used beyond the standard panel. "
            "Codes: 0=No; 1=Yes. "
            "The standard panel already captured by dedicated variables: "
            "Gallyas silver stain (NPHISG), modified Bielschowsky (NPHISMB), "
            "other silver stain (NPHISSS), thioflavin S or T (NPHIST). "
            "Do NOT code 1 for: H&E (hematoxylin and eosin), LFB (Luxol fast blue), "
            "H&E-LFB combined — these are routine stains used in every case, not special histochemical stains. "
            "Code 1 ONLY for additional special stains explicitly named in the report "
            "that are not in the standard panel above — examples: iron stain (Perl's method / Prussian blue), "
            "Woelcke-Heidenhain (myelin stain), PAS, Congo red, Masson trichrome, Bodian. "
            "If NPHISO=1 → populate NPHISOX with the stain name. "
            "If not mentioned or only routine stains used → code 0."
        ),
    )
    NPHISOX: Optional[str] = Field(
        None,
        description=(
            "Other histochemical stain - free-text specify. "
            "Populate ONLY when NPHISO=1. "
            "Record the exact stain name as written in the report. "
            "If multiple qualifying stains are present, record all names separated by a semicolon. "
            "Null in all other cases."
        ),
    )

    @field_validator("NPHISMB", "NPHISG", "NPHISSS", "NPHIST", "NPHISO")
    @classmethod
    def val_stain(cls, v):
        if v not in {0, 1}:
            raise ValueError(f"Stain flag must be 0 or 1, got {v}")
        return v


class TissueBanking(BaseModel):
    """Tissue banking and specimen availability - 6 variables."""
    model_config = ConfigDict(extra="forbid")

    NACCBNKF: int = Field(
        description=(
            "Banked frozen brain tissue available. "
            "Codes: 0=No; 1=Yes; 9=Missing/unknown. "
            "Look for mentions of frozen tissue storage, frozen sections, "
            "'stored at -70C', '-80C', 'frozen hemisphere'. "
            "If explicitly mentioned → code 1. If not mentioned → code 0."
        ),
    )
    NPBNKB: int = Field(
        description=(
            "Banked frozen wedge of cerebellum or other sample stored for future DNA preparation. "
            "Codes: 0=No; 1=Yes; 9=Missing/unknown. "
            "Look in the FROZEN TISSUE SAMPLES section of the report. "
            "Code 1 if any cerebellar tissue is listed as stored frozen — "
            "this includes: 'cerebellum (bags X)', 'right cerebellum: unsliced', "
            "'cerebellar cortex specimens', 'wedge of cerebellum', or any equivalent phrasing. "
            "The tissue does not need to be explicitly labeled as 'for DNA prep' — "
            "frozen cerebellar storage is sufficient to code 1. "
            "Code 0 if the frozen tissue section is present but contains no cerebellar sample, "
            "or if frozen tissue storage is not mentioned at all."
        ),
    )
    NACCFORM: int = Field(
        description=(
            "Formalin- or paraformaldehyde-fixed brain available. "
            "Codes: 0=No; 1=Yes; 9=Missing/unknown. "
            "Look for 'fixed in formalin', 'formalin-fixed', 'fixed hemisphere'. "
            "If not mentioned → code 0."
        ),
    )
    NACCPARA: int = Field(
        description=(
            "Paraffin-embedded blocks of brain regions available. "
            "Codes: 0=No; 1=Yes; 9=Missing/unknown. "
            "Look for 'paraffin blocks', 'FFPE', 'paraffin-embedded'. "
            "If not mentioned → code 0."
        ),
    )
    NACCCSFP: int = Field(
        description=(
            "Banked postmortem CSF available. "
            "Codes: 0=No; 1=Yes; 9=Missing/unknown. "
            "Look for 'CSF', 'cerebrospinal fluid', 'postmortem CSF'. "
            "If not mentioned → code 0."
        ),
    )
    NPBNKF: int = Field(
        description=(
            "Banked postmortem blood or serum available. "
            "Codes: 0=No; 1=Yes; 9=Missing/unknown. "
            "Look for 'blood', 'serum', 'plasma' stored at autopsy. "
            "If not mentioned → code 0."
        ),
    )

    @field_validator("NACCBNKF", "NPBNKB", "NACCFORM", "NACCPARA", "NACCCSFP", "NPBNKF")
    @classmethod
    def val_banking(cls, v):
        if v not in {0, 1, 9}:
            raise ValueError(f"Banking field must be 0, 1, or 9, got {v}")
        return v


# ---------------------------------------------------------------------------
# Top-level Pass 5 model
# ---------------------------------------------------------------------------

class Pass5Extraction(BaseModel):
    """Pass 5 extraction - 27 variables.

    Covers: case metadata, IHC antibody methods, histochemical stains,
    and tissue banking/specimen availability.
    """
    model_config = ConfigDict(extra="forbid")

    case_metadata: CaseMetadata = Field(
        description=(
            "Case metadata - NPFORMVER, NACCID, NACCADC, NACCDAGE, "
            "NACCMOD, NACCYOD, NACCINT (all nullable if not stated)"
        )
    )
    antibody_methods: AntibodyMethods = Field(
        description=(
            "IHC antibody methods - NPTAN, NPTANX, NPABAN, NPABANX, "
            "NPASAN, NPASANX, NPTDPAN, NPTDPANX (all nullable if not mentioned)"
        )
    )
    histochem_stains: HistochemStains = Field(
        description=(
            "Histochemical stains - NPHISMB, NPHISG, NPHISSS, NPHIST, NPHISO, NPHISOX"
        )
    )
    tissue_banking: TissueBanking = Field(
        description=(
            "Tissue banking - NACCBNKF, NPBNKB, NACCFORM, NACCPARA, NACCCSFP, NPBNKF"
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
    def validate_confidence_value(self) -> "Pass5Extraction":
        if self.extraction_confidence and self.extraction_confidence not in {
            "high", "moderate", "low"
        }:
            raise ValueError("extraction_confidence must be 'high', 'moderate', or 'low'")
        return self


    @model_validator(mode="after")
    def enforce_null_policy(self) -> "Pass5Extraction":
        """Non-nullable fields must not be null - use 0, 1, or 9 instead."""
        # All case_metadata and antibody_methods fields are nullable
        # Only histochem stains and banking fields are non-nullable
        non_nullable_sections = [self.histochem_stains, self.tissue_banking]
        nullable_fields = {"NPHISOX"}

        for section in non_nullable_sections:
            for field_name in section.model_fields:
                if field_name not in nullable_fields and getattr(section, field_name) is None:
                    raise ValueError(
                        f"{field_name} must not be null - use 0, 1, or 9 (depending on variable)."
                    )
        return self


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_pass5_model() -> Type[BaseModel]:
    """Return the top-level extraction model class."""
    return Pass5Extraction


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


def build_pass5_format_instructions() -> str:
    """Walk the Pydantic schema and emit LLM-facing format instructions."""
    model = Pass5Extraction
    lines = [
        "PASS 5 - Extract ONLY the metadata, staining, and banking variables below.",
        "Respond with a single JSON object with EXACTLY these top-level keys:",
        "",
        "  case_metadata, antibody_methods, histochem_stains, tissue_banking,",
        "  field_annotations, extraction_confidence, extraction_notes",
        "",
        "Each top-level key maps to a nested object containing its variables.",
        "NEVER output variables at the root level - they must always be nested.",
        "",
        "NULL POLICY:",
        "  All case_metadata fields are nullable - null if not stated in the report.",
        "  All antibody_methods fields are nullable - null if not mentioned.",
        "  Specify fields (NPTANX, NPABANX, NPASANX, NPTDPANX, NPHISOX):",
        "    null unless their parent flag is set to 'Other'.",
        "  Stain flags (NPHISMB etc.) and banking flags: never null - use 0 if absent.",
        "",
        "SPECIFY FIELDS (populate ONLY when parent has the stated code, otherwise null):",
        "  NPTANX: null unless NPTAN=7",
        "  NPABANX: null unless NPABAN=7",
        "  NPASANX: null unless NPASAN=7",
        "  NPTDPANX: null unless NPTDPAN=7",
        "  NPHISOX: null unless NPHISO=1",
        "",
        "STAIN FLAGS: code 1 if the stain is explicitly mentioned in the methods section.",
        "  If not mentioned → code 0.",
        "",
        "BANKING FLAGS: code 1 if the report explicitly mentions the specimen type is banked.",
        "  If not mentioned → code 0.",
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