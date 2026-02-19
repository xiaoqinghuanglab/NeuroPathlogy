"""Pydantic schema for structured neuropathology report extraction.

SINGLE SOURCE OF TRUTH — edit this file to change what gets extracted.
main.py imports `get_extraction_model()` and `build_format_instructions()`
and never hardcodes field names.

HOW TO ADD FIELDS:
  1. Add a field to the appropriate sub-model (or create a new one).
  2. Give it a `description=` — this auto-populates the LLM prompt.
  3. Add validators (range, enum, regex) — these auto-trigger on re-ask.
  4. That's it.  The prompt and validation pipeline adapt automatically.

See the commented-out examples below each sub-model for every supported type.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enums — define constrained vocabularies here
# ---------------------------------------------------------------------------


class SpecimenType(str, Enum):
    biopsy = "biopsy"
    resection = "resection"
    autopsy = "autopsy"
    stereotactic_biopsy = "stereotactic_biopsy"
    other = "other"


class TumorGrade(str, Enum):
    """WHO CNS5 grading."""

    grade_1 = "1"
    grade_2 = "2"
    grade_3 = "3"
    grade_4 = "4"
    not_applicable = "N/A"


class IDHStatus(str, Enum):
    mutant = "IDH-mutant"
    wildtype = "IDH-wildtype"
    not_tested = "not_tested"
    indeterminate = "indeterminate"


class MGMTStatus(str, Enum):
    methylated = "methylated"
    unmethylated = "unmethylated"
    not_tested = "not_tested"
    indeterminate = "indeterminate"


class OneP19QStatus(str, Enum):
    codeleted = "codeleted"
    intact = "intact"
    not_tested = "not_tested"
    indeterminate = "indeterminate"


class EGFRStatus(str, Enum):
    amplified = "amplified"
    not_amplified = "not_amplified"
    not_tested = "not_tested"
    indeterminate = "indeterminate"


class TERTStatus(str, Enum):
    mutant = "mutant"
    wildtype = "wildtype"
    not_tested = "not_tested"
    indeterminate = "indeterminate"


class CDKN2AStatus(str, Enum):
    deleted = "homozygously_deleted"
    not_deleted = "not_deleted"
    not_tested = "not_tested"
    indeterminate = "indeterminate"


class H3Status(str, Enum):
    K27M_mutant = "H3K27M-mutant"
    G34_mutant = "H3G34-mutant"
    wildtype = "wildtype"
    not_tested = "not_tested"
    indeterminate = "indeterminate"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class PatientDemographics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_name_last: Optional[str] = Field(
        None, description="Patient last (family) name"
    )
    patient_name_first: Optional[str] = Field(
        None, description="Patient first (given) name"
    )
    date_of_birth: Optional[str] = Field(
        None, description="Date of birth in YYYY-MM-DD format"
    )
    age: Optional[int] = Field(None, ge=0, le=120, description="Patient age in years")
    sex: Optional[str] = Field(None, description="Patient sex (M/F/Other)")
    clinical_history_summary: Optional[str] = Field(
        None, description="Brief clinical history as stated in the report"
    )

    @field_validator("date_of_birth")
    @classmethod
    def validate_dob_format(cls, v: Optional[str]) -> Optional[str]:
        """Ensure DOB is a valid YYYY-MM-DD date string."""
        if v is None:
            return v
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"date_of_birth must be YYYY-MM-DD, got '{v}'")
        return v

    # ----------------------------------------------------------------
    # ILLUSTRATION: other field types you can add here
    # ----------------------------------------------------------------
    # # string with min/max length
    # mrn: Optional[str] = Field(None, min_length=4, max_length=20,
    #     description="Medical record number")
    #
    # # integer with range check
    # weight_kg: Optional[int] = Field(None, ge=1, le=300,
    #     description="Patient weight in kilograms")
    #
    # # float with range check
    # height_cm: Optional[float] = Field(None, ge=30.0, le=250.0,
    #     description="Patient height in centimeters")
    #
    # # boolean
    # is_pregnant: Optional[bool] = Field(None,
    #     description="Whether the patient is currently pregnant")
    #
    # # regex-validated string
    # phone: Optional[str] = Field(None, pattern=r"^\d{3}-\d{3}-\d{4}$",
    #     description="Phone number in XXX-XXX-XXXX format")


class ReportMetadata(BaseModel):
    """Timestamps and identifiers on the report itself."""

    model_config = ConfigDict(extra="forbid")

    report_date: Optional[str] = Field(
        None, description="Date the report was issued, in YYYY-MM-DD format"
    )
    report_time: Optional[str] = Field(
        None, description="Time the report was issued, in HH:MM (24h) format"
    )
    accession_number: Optional[str] = Field(
        None, description="Pathology accession/case number"
    )

    @field_validator("report_date")
    @classmethod
    def validate_report_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"report_date must be YYYY-MM-DD, got '{v}'")
        return v

    @field_validator("report_time")
    @classmethod
    def validate_report_time(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            parts = v.split(":")
            assert len(parts) == 2
            h, m = int(parts[0]), int(parts[1])
            assert 0 <= h <= 23 and 0 <= m <= 59
        except (ValueError, AssertionError):
            raise ValueError(f"report_time must be HH:MM (24h), got '{v}'")
        return v

    # ----------------------------------------------------------------
    # ILLUSTRATION: date/time types
    # ----------------------------------------------------------------
    # # if you prefer native date objects (LLM still outputs string,
    # # Pydantic coerces automatically):
    # collection_date: Optional[date] = Field(None,
    #     description="Specimen collection date")
    #
    # # full datetime
    # received_datetime: Optional[datetime] = Field(None,
    #     description="Specimen receipt datetime in ISO format")


class SpecimenInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    specimen_type: Optional[SpecimenType] = Field(
        None,
        description="Type of specimen (biopsy, resection, autopsy, stereotactic_biopsy, other)",
    )
    specimen_site: Optional[str] = Field(
        None, description="Anatomical site (e.g., 'right frontal lobe')"
    )
    laterality: Optional[str] = Field(
        None, description="Side: left, right, midline, bilateral, or not specified"
    )

    # ----------------------------------------------------------------
    # ILLUSTRATION: enum field
    # ----------------------------------------------------------------
    # # just define a new Enum class above and use it:
    # fixation: Optional[FixationType] = Field(None,
    #     description="Fixation method used")


class MolecularMarkers(BaseModel):
    """Key molecular/genetic markers per WHO CNS5."""

    model_config = ConfigDict(extra="forbid")

    idh_status: Optional[IDHStatus] = Field(None, description="IDH1/2 mutation status")
    mgmt_promoter: Optional[MGMTStatus] = Field(
        None, description="MGMT promoter methylation status"
    )
    one_p_19q: Optional[OneP19QStatus] = Field(
        None, description="1p/19q codeletion status"
    )
    egfr_amplification: Optional[EGFRStatus] = Field(
        None, description="EGFR amplification status"
    )
    tert_promoter: Optional[TERTStatus] = Field(
        None, description="TERT promoter mutation status"
    )
    cdkn2a: Optional[CDKN2AStatus] = Field(
        None, description="CDKN2A/B homozygous deletion status"
    )
    h3_status: Optional[H3Status] = Field(
        None, description="Histone H3 alteration status"
    )
    ki67_index: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Ki-67 proliferation index as percentage (0-100)",
    )
    p53_expression: Optional[str] = Field(
        None,
        description="p53 IHC pattern (e.g., 'strong diffuse nuclear', 'wildtype pattern', 'null')",
    )
    atrx_expression: Optional[str] = Field(
        None, description="ATRX expression status (retained / lost)"
    )
    other_markers: Optional[Dict[str, str]] = Field(
        None, description="Additional molecular/IHC markers as {marker_name: result}"
    )

    # ----------------------------------------------------------------
    # ILLUSTRATION: dict and list fields
    # ----------------------------------------------------------------
    # # free-form key-value pairs
    # fish_results: Optional[Dict[str, str]] = Field(None,
    #     description="FISH results as {probe: result}")
    #
    # # list of strings
    # mutations_detected: Optional[List[str]] = Field(None,
    #     description="List of mutations detected by NGS panel")


class HistopathologyFindings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    morphological_description: Optional[str] = Field(
        None, description="Key histological features described by the pathologist"
    )
    mitotic_count: Optional[str] = Field(
        None, description="Mitotic count as reported (e.g., '5 per 10 HPF')"
    )
    necrosis_present: Optional[bool] = Field(
        None, description="Whether necrosis is present"
    )
    microvascular_proliferation: Optional[bool] = Field(
        None, description="Whether microvascular proliferation is present"
    )
    invasion_pattern: Optional[str] = Field(
        None,
        description="Invasion pattern if described (e.g., 'diffuse infiltration of cortex')",
    )

    # ----------------------------------------------------------------
    # ILLUSTRATION: boolean + cross-field validator
    # ----------------------------------------------------------------
    # calcification_present: Optional[bool] = Field(None,
    #     description="Whether calcification is present")
    #
    # @model_validator(mode="after")
    # def necrosis_implies_high_grade(self) -> "HistopathologyFindings":
    #     """Example: warn if necrosis is present without microvascular proliferation."""
    #     if self.necrosis_present and not self.microvascular_proliferation:
    #         # could raise ValueError to enforce, or just log a warning
    #         pass
    #     return self


class Diagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    integrated_diagnosis: Optional[str] = Field(
        None,
        description="Final integrated WHO CNS5 diagnosis (e.g., 'Glioblastoma, IDH-wildtype, WHO grade 4')",
    )
    tumor_type: Optional[str] = Field(
        None,
        description="Tumor entity name (e.g., 'Glioblastoma', 'Oligodendroglioma')",
    )
    who_grade: Optional[TumorGrade] = Field(
        None, description="WHO CNS tumor grade (1-4 or N/A)"
    )
    histological_subtype: Optional[str] = Field(
        None, description="Histological subtype if specified"
    )
    additional_diagnoses: Optional[List[str]] = Field(
        None, description="Any secondary or incidental diagnoses"
    )


# ---------------------------------------------------------------------------
# Top-level extraction model
# ---------------------------------------------------------------------------


class NeuropathologyExtraction(BaseModel):
    """Structured extraction from a single neuropathology report."""

    model_config = ConfigDict(extra="forbid")

    patient: PatientDemographics = Field(
        default_factory=PatientDemographics, description="Patient demographics"
    )
    report: ReportMetadata = Field(
        default_factory=ReportMetadata, description="Report metadata (dates, IDs)"
    )
    specimen: SpecimenInfo = Field(
        default_factory=SpecimenInfo, description="Specimen information"
    )
    histopathology: HistopathologyFindings = Field(
        default_factory=HistopathologyFindings, description="Histopathological findings"
    )
    molecular_markers: MolecularMarkers = Field(
        default_factory=MolecularMarkers, description="Molecular and IHC markers"
    )
    diagnosis: Diagnosis = Field(
        default_factory=Diagnosis, description="Diagnostic conclusions"
    )
    extraction_confidence: Optional[str] = Field(
        None, description="Overall confidence: high, moderate, or low"
    )
    extraction_notes: Optional[str] = Field(
        None, description="Caveats, ambiguities, or unresolvable information"
    )

    @model_validator(mode="after")
    def confidence_is_valid(self) -> "NeuropathologyExtraction":
        if self.extraction_confidence and self.extraction_confidence not in (
            "high",
            "moderate",
            "low",
        ):
            raise ValueError("extraction_confidence must be high, moderate, or low")
        return self


# ---------------------------------------------------------------------------
# Public API — main.py imports only these
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

    # nested BaseModel
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
    lines.append("")
    lines.append(
        "Use null for any field whose value cannot be determined from the report."
    )
    lines.append(
        "Do not invent information. Extract only what is explicitly stated or clearly implied."
    )
    return "\n".join(lines)
