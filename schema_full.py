"""schema_full.py — Extended NACC Neuropathology extraction schema (40 variables).

Priority variables (from original schema.py) have full detailed descriptions.
Additional variables have short descriptions to keep the prompt compact.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SeverityCode(int, Enum):
    none = 0
    mild = 1
    moderate = 2
    severe = 3
    not_assessed = 8
    missing = 9

class PresentAbsentCode(int, Enum):
    yes = 1
    no = 2
    not_assessed = 3
    missing = 9

class YesNoCode(int, Enum):
    no = 0
    yes = 1
    not_assessed = 8
    missing = 9

class BraakStage(int, Enum):
    stage_0 = 0
    stage_I = 1
    stage_II = 2
    stage_III = 3
    stage_IV = 4
    stage_V = 5
    stage_VI = 6
    other_tauopathy = 7
    not_assessed = 8
    missing = 9

class ThalPhase(int, Enum):
    phase_0 = 0
    phase_1 = 1
    phase_2 = 2
    phase_3 = 3
    phase_4 = 4
    phase_5 = 5
    not_assessed = 8
    missing = 9

class CERADScore(int, Enum):
    no_plaques = 0
    sparse = 1
    moderate = 2
    frequent = 3
    not_assessed = 8
    missing = 9

class ADNCScore(int, Enum):
    not_AD = 0
    low = 1
    intermediate = 2
    high = 3
    not_assessed = 8
    missing = 9

class LewyBodyPattern(int, Enum):
    no_lewy = 0
    brainstem_predominant = 1
    limbic_transitional = 2
    neocortical_diffuse = 3
    amygdala_predominant = 4
    olfactory_bulb = 5
    not_assessed = 8
    missing = 9


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class SpecimenInfo(BaseModel):
    """Basic specimen details."""
    model_config = ConfigDict(extra="forbid")

    # ── PRIORITY VARIABLES (full descriptions) ──────────────────────────────
    NPSEX: Optional[int] = Field(
        None,
        description=(
            "Subject sex. Allowable codes: 1=Male, 2=Female. "
            "Extract from demographics section of report."
        )
    )
    NPFIX: Optional[int] = Field(
        None,
        description=(
            "Fixative used for brain preservation. "
            "1=Formalin, 2=Paraformaldehyde, 7=Other (specify in NPFIXX). "
            "Look for phrases like 'fixed in formalin' or 'paraformaldehyde fixation'."
        )
    )
    NPWBRWT: Optional[int] = Field(
        None,
        description=(
            "Whole brain weight in grams. Allowable range: 100-2500g. "
            "9999=unknown. Extract the numeric value only — do not include units. "
            "Look for 'brain weight', 'cerebral weight', or 'weighs Xg'."
        )
    )

    # ── ADDITIONAL VARIABLES (short descriptions) ────────────────────────────
    NPWBRF: Optional[int] = Field(
        None, description="Brain weight fresh or fixed: 1=Fresh, 2=Fixed, 8=N/A"
    )
    NPPMIH: Optional[float] = Field(
        None, description="Postmortem interval hours (0.0-98.9; 99.9=unknown)"
    )
    NPFIXX: Optional[str] = Field(
        None, description="Fixative other specify (only if NPFIX=7)"
    )

    @field_validator("NPSEX")
    @classmethod
    def val_sex(cls, v):
        if v is not None and v not in {1, 2}:
            raise ValueError(f"NPSEX must be 1 or 2, got {v}")
        return v

    @field_validator("NPFIX")
    @classmethod
    def val_fix(cls, v):
        if v is not None and v not in {1, 2, 7, -4}:
            raise ValueError(f"NPFIX must be 1, 2, 7, or -4, got {v}")
        return v

    @field_validator("NPWBRWT")
    @classmethod
    def val_weight(cls, v):
        if v is not None and v != 9999 and not (100 <= v <= 2500):
            raise ValueError(f"NPWBRWT must be 100-2500 or 9999, got {v}")
        return v


class GrossFindings(BaseModel):
    """Gross macroscopic examination findings."""
    model_config = ConfigDict(extra="forbid")

    # ── PRIORITY VARIABLES (full descriptions) ──────────────────────────────
    NPGRLA: Optional[int] = Field(
        None,
        description=(
            "Lobar atrophy present on gross examination. "
            "0=None, 1=Yes, 8=Not assessed, 9=Missing/unknown. "
            "Look for mentions of focal lobar atrophy in any lobe."
        )
    )
    NPGRHA: Optional[SeverityCode] = Field(
        None,
        description=(
            "Severity of hippocampal atrophy on gross examination. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Look for 'hippocampal atrophy', 'medial temporal atrophy'."
        )
    )
    NPGRSNH: Optional[SeverityCode] = Field(
        None,
        description=(
            "Severity of substantia nigra hypopigmentation. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Look for 'pallor of substantia nigra', 'depigmentation', 'hypopigmentation'."
        )
    )
    NPGRLCH: Optional[SeverityCode] = Field(
        None,
        description=(
            "Severity of locus ceruleus hypopigmentation. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Look for 'locus coeruleus pallor', 'depigmentation of locus ceruleus'."
        )
    )

    # ── ADDITIONAL VARIABLES (short descriptions) ────────────────────────────
    NPGRCCA: Optional[SeverityCode] = Field(
        None, description="Cerebral cortex atrophy: 0=None 1=Mild 2=Moderate 3=Severe 8 9"
    )
    NACCBRNN: Optional[int] = Field(
        None, description="No major neuropath change: 0=Some present, 1=None, 8=Missing"
    )

    @field_validator("NPGRLA")
    @classmethod
    def val_grla(cls, v):
        if v is not None and v not in {0, 1, 8, 9, -4}:
            raise ValueError(f"NPGRLA must be 0, 1, 8, 9, or -4, got {v}")
        return v


class VascularPathology(BaseModel):
    """Vascular and ischemic pathology."""
    model_config = ConfigDict(extra="forbid")

    # ── PRIORITY VARIABLES (full descriptions) ──────────────────────────────
    NACCAVAS: Optional[SeverityCode] = Field(
        None,
        description=(
            "Severity of atherosclerosis of the circle of Willis. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Look for 'atherosclerosis', 'arteriosclerosis' of circle of Willis or major vessels. "
            "If report says 'moderate to severe' code as 3 (Severe)."
        )
    )
    NPLINF: Optional[PresentAbsentCode] = Field(
        None,
        description=(
            "Large arterial (cortical) infarcts present. "
            "1=Yes, 2=No, 3=Not assessed, 9=Unknown. "
            "Look for 'cortical infarct', 'territorial infarct', 'large vessel infarct'."
        )
    )
    NPLAC: Optional[PresentAbsentCode] = Field(
        None,
        description=(
            "One or more lacunes (small artery infarcts and/or hemorrhages) present. "
            "1=Yes, 2=No, 3=Not assessed, 9=Unknown. "
            "Look for 'lacunar infarct', 'lacune', 'small vessel infarct'."
        )
    )
    NPHEM: Optional[PresentAbsentCode] = Field(
        None,
        description=(
            "Single or multiple hemorrhages present. "
            "1=Yes, 2=No, 3=Not assessed, 9=Unknown. "
            "Look for 'hemorrhage', 'bleeding', 'hematoma' in any region."
        )
    )
    NPWMR: Optional[SeverityCode] = Field(
        None,
        description=(
            "White matter rarefaction severity. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Look for 'white matter rarefaction', 'leukoaraiosis', 'periventricular changes'."
        )
    )

    # ── ADDITIONAL VARIABLES (short descriptions) ────────────────────────────
    NACCARTE: Optional[SeverityCode] = Field(
        None, description="Arteriolosclerosis: 0=None 1=Mild 2=Moderate 3=Severe 8 9"
    )
    NACCVASC: Optional[int] = Field(
        None, description="Vascular pathology present (derived): 0=No 1=Yes 9=Unknown"
    )
    NACCINF: Optional[int] = Field(
        None, description="Infarct and lacunes present (derived): 0=No 1=Yes 8 9"
    )
    NACCHEM: Optional[int] = Field(
        None, description="Hemorrhages and microbleeds present (derived): 0=No 1=Yes 8 9"
    )

    @field_validator("NACCVASC", "NACCINF", "NACCHEM")
    @classmethod
    def val_derived(cls, v):
        if v is not None and v not in {0, 1, 8, 9}:
            raise ValueError(f"Derived field must be 0, 1, 8, or 9, got {v}")
        return v


class MicroscopicFindings(BaseModel):
    """Microscopic and cellular findings."""
    model_config = ConfigDict(extra="forbid")

    # ── PRIORITY VARIABLES (full descriptions) ──────────────────────────────
    NPNLOSS: Optional[SeverityCode] = Field(
        None,
        description=(
            "Severity of neuron loss in the substantia nigra. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Look for 'neuronal loss in substantia nigra', 'depopulation of SN neurons'."
        )
    )

    # ── ADDITIONAL VARIABLES (short descriptions) ────────────────────────────
    NPHIPSCL: Optional[int] = Field(
        None, description="Hippocampal sclerosis CA1/subiculum: 0=None 1=Unilateral 2=Bilateral 3=Present/NOS 8 9"
    )
    NACCLEWY: Optional[int] = Field(
        None, description="Lewy body pattern (derived): 0=None 1=Brainstem 2=Limbic 3=Neocortical 4=Unspecified 8 9"
    )
    NPLBOD: Optional[LewyBodyPattern] = Field(
        None, description="Lewy body pathology: 0=No 1=Brainstem 2=Limbic 3=Neocortical 4=Amygdala 5=Olfactory 8 9"
    )


class ADPathology(BaseModel):
    """Alzheimer's disease ABC scores."""
    model_config = ConfigDict(extra="forbid")

    # ── ADDITIONAL VARIABLES (short descriptions) ────────────────────────────
    NPTHAL: Optional[ThalPhase] = Field(
        None, description="Thal phase amyloid A score: 0-5, 8=Not assessed, 9=Unknown"
    )
    NACCBRAA: Optional[BraakStage] = Field(
        None, description="Braak stage neurofibrillary B score: 0-6, 7=Other tauopathy, 8, 9"
    )
    NACCNEUR: Optional[CERADScore] = Field(
        None, description="CERAD neuritic plaques C score: 0=None 1=Sparse 2=Moderate 3=Frequent 8 9"
    )
    NPADNC: Optional[ADNCScore] = Field(
        None, description="NIA-AA ADNC overall ABC: 0=Not AD 1=Low 2=Intermediate 3=High 8 9"
    )
    NACCDIFF: Optional[CERADScore] = Field(
        None, description="Diffuse plaques CERAD: 0=None 1=Sparse 2=Moderate 3=Frequent 8 9"
    )
    NACCAMY: Optional[SeverityCode] = Field(
        None, description="Cerebral amyloid angiopathy: 0=None 1=Mild 2=Moderate 3=Severe 8 9"
    )


class DiagnosticCodes(BaseModel):
    """Primary and contributing final diagnoses."""
    model_config = ConfigDict(extra="forbid")

    # ── PRIORITY VARIABLES (full descriptions) ──────────────────────────────
    NACCCBD: Optional[YesNoCode] = Field(
        None,
        description=(
            "Corticobasal degeneration (CBD) present. "
            "0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "Look for 'corticobasal degeneration', 'CBD', 'corticobasal syndrome'."
        )
    )
    NPPVASC: Optional[int] = Field(
        None,
        description=(
            "Vascular disease as PRIMARY pathologic diagnosis. "
            "1=Yes, 2=No. "
            "Code 1 only if vascular disease is listed as the primary/principal diagnosis."
        )
    )

    # ── ADDITIONAL VARIABLES (short descriptions) ────────────────────────────
    NPPAD: Optional[int] = Field(
        None, description="Alzheimer's disease primary diagnosis: 1=Yes 2=No"
    )
    NPCAD: Optional[int] = Field(
        None, description="Alzheimer's disease contributing diagnosis: 1=Yes 2=No"
    )
    NPPLEWY: Optional[int] = Field(
        None, description="Lewy body disease primary diagnosis: 1=Yes 2=No"
    )
    NPCLEWY: Optional[int] = Field(
        None, description="Lewy body disease contributing diagnosis: 1=Yes 2=No"
    )
    NPCVASC: Optional[int] = Field(
        None, description="Vascular disease contributing diagnosis: 1=Yes 2=No"
    )
    NPPFTLD: Optional[int] = Field(
        None, description="FTLD primary diagnosis: 1=Yes 2=No"
    )
    NACCPROG: Optional[YesNoCode] = Field(
        None, description="PSP progressive supranuclear palsy: 0=No 1=Yes 8 9"
    )
    NACCPICK: Optional[YesNoCode] = Field(
        None, description="Pick's disease PiD: 0=No 1=Yes 8 9"
    )
    NPFTDTDP: Optional[YesNoCode] = Field(
        None, description="FTLD-TDP TDP-43 pathology: 0=No 1=Yes 8 9"
    )
    NACCPRIO: Optional[YesNoCode] = Field(
        None, description="Prion disease: 0=No 1=Yes 8 9"
    )

    @field_validator("NPPVASC", "NPPAD", "NPCAD", "NPPLEWY", "NPCLEWY", "NPCVASC", "NPPFTLD")
    @classmethod
    def val_binary(cls, v):
        if v is not None and v not in {1, 2}:
            raise ValueError(f"Primary/contributing diagnosis must be 1 or 2, got {v}")
        return v


# ---------------------------------------------------------------------------
# Top-level extraction model
# ---------------------------------------------------------------------------

class NeuropathologyExtraction(BaseModel):
    """Structured NACC neuropathology extraction — 40 variables.

    Priority fields (from original schema): NPSEX, NPFIX, NPWBRWT, NPGRLA,
    NPGRHA, NPGRSNH, NPGRLCH, NACCAVAS, NPLINF, NPLAC, NPHEM, NPWMR,
    NPNLOSS, NACCCBD, NPPVASC — extract these with highest care.
    """
    model_config = ConfigDict(extra="forbid")

    specimen_info: SpecimenInfo = Field(
        default_factory=SpecimenInfo,
        description="Specimen details — PRIORITY: NPSEX, NPFIX, NPWBRWT"
    )
    gross_findings: GrossFindings = Field(
        default_factory=GrossFindings,
        description="Gross findings — PRIORITY: NPGRLA, NPGRHA, NPGRSNH, NPGRLCH"
    )
    vascular_pathology: VascularPathology = Field(
        default_factory=VascularPathology,
        description="Vascular pathology — PRIORITY: NACCAVAS, NPLINF, NPLAC, NPHEM, NPWMR"
    )
    microscopic_findings: MicroscopicFindings = Field(
        default_factory=MicroscopicFindings,
        description="Microscopic findings — PRIORITY: NPNLOSS"
    )
    ad_pathology: ADPathology = Field(
        default_factory=ADPathology,
        description="AD ABC scores (Thal/Braak/CERAD/ADNC)"
    )
    diagnostic_codes: DiagnosticCodes = Field(
        default_factory=DiagnosticCodes,
        description="Final diagnoses — PRIORITY: NACCCBD, NPPVASC"
    )

    extraction_confidence: Optional[str] = Field(
        None, description="Overall confidence: high, moderate, or low"
    )
    extraction_notes: Optional[str] = Field(
        None, description="Caveats, ambiguities, or unresolvable fields"
    )

    @model_validator(mode="after")
    def confidence_is_valid(self) -> "NeuropathologyExtraction":
        if self.extraction_confidence and self.extraction_confidence not in {
            "high", "moderate", "low"
        }:
            raise ValueError("extraction_confidence must be high, moderate, or low")
        return self


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_extraction_model() -> Type[BaseModel]:
    return NeuropathologyExtraction


def unwrap_optional(annotation: Any):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union and type(None) in args:
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1:
            return inner[0], True
    return annotation, False


def describe_field(name: str, info, indent: int = 0) -> List[str]:
    prefix = "  " * indent
    lines: List[str] = []
    desc = info.description or ""

    raw_ann = info.annotation
    inner, is_optional = unwrap_optional(raw_ann)
    opt_tag = " (optional)" if is_optional else ""

    if isinstance(inner, type) and issubclass(inner, Enum):
        allowed = [e.value for e in inner]
        lines.append(f"{prefix}- {name}{opt_tag}: {desc}. Allowed values: {allowed}")
        return lines

    if isinstance(inner, type) and issubclass(inner, BaseModel):
        lines.append(f"{prefix}- {name}: {desc}")
        for sub_name, sub_info in inner.model_fields.items():
            lines.extend(describe_field(sub_name, sub_info, indent + 1))
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


def build_format_instructions(model: Optional[Type[BaseModel]] = None) -> str:
    if model is None:
        model = get_extraction_model()

    lines = [
        "Respond with a single JSON object conforming to this schema:",
        "",
        f"Root model: {model.__name__}",
        "",
        "PRIORITY fields — extract these with highest care:",
        "NPSEX, NPFIX, NPWBRWT, NPGRLA, NPGRHA, NPGRSNH, NPGRLCH,",
        "NACCAVAS, NPLINF, NPLAC, NPHEM, NPWMR, NPNLOSS, NACCCBD, NPPVASC",
        "",
    ]
    for name, info in model.model_fields.items():
        lines.extend(describe_field(name, info, indent=0))
    lines.append("")
    lines.append("Use null for any field whose value cannot be determined from the report.")
    lines.append("Do not invent information. Extract only what is explicitly stated or clearly implied.")
    return "\n".join(lines)
