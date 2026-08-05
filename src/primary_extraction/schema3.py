"""schema3.py - Pass 3: Vascular detail (51 vars).

Pass 3 variables:
  Infarcts:      NPINF, NACCINF, NPINF1A/B/D/F, NPINF2A/B/D/F,
                 NPINF3A/B/D/F, NPINF4A/B/D/F
  Hemorrhage:    NPHEMO, NPHEMO1, NPHEMO2, NPHEMO3
  Microbleeds:   NPOLDD, NPOLDD1, NPOLDD2, NPOLDD3, NPOLDD4, NACCHEM
  Microinfarcts: NPOLD, NPOLD1, NPOLD2, NPOLD3, NPOLD4, NACCMICR
  Other vasc:    NPPATH, NACCNEC, NPPATH2-11, NPPATHO, NPPATHOX
  Misc vasc:     NPMICRO, NPART, NPOANG

Parent-child mirroring rules (confirmed NACC + PDF):
  NPINF count fields (NPINF1A/2A/3A/4A):
    parent=0 → 0, parent=8 → 88, parent=9 → 99
  NPINF size fields (NPINF1B/D/F × 4 regions):
    parent=0 → null (blank),
    parent=8 → 88.8, parent=9 → 99.9
  NPHEMO1/2/3 (under NPHEMO):
    parent=0 → 0, parent=8 → 8, parent=9 → 9
  NPOLD1/2/3/4 (under NPOLD):
    parent=0 → 0, parent=8 → 8, parent=9 → 9
  NPOLDD1/2/3/4 (under NPOLDD):
    parent=0 → 0, parent=8 → 8, parent=9 → 9
  NACCNEC, NPPATH2-11 (under NPPATH):
    parent=0 → 0, parent=8 → 8, parent=9 → 9
  NPPATHO (under NPPATH, only 0/1 codes):
    parent=0 → 0, parent=8 → 0, parent=9 → 0
  NPPATHOX: blank/null if NPPATHO != 1
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

class InfarctParent(BaseModel):
    """NPINF parent + derived NACCINF - 2 variables."""
    model_config = ConfigDict(extra="forbid")

    NPINF: int = Field(
        description=(
            "Old infarcts observed grossly (including lacunes). "
            "Codes: 0=None observed; 1=Yes, observed; 8=Not assessed; 9=Unknown. "
            "This is the parent variable - its value controls all NPINF child fields."
        ),
    )
    NACCINF: int = Field(
        description=(
            "Derived: any infarct or lacune present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Copy the value of NPINF as it is. "
            "This is a derived field."
        ),
    )

    @field_validator("NPINF")
    @classmethod
    def val_npinf(cls, v):
        if v not in {0, 1, 8, 9}:
            raise ValueError(f"NPINF must be 0, 1, 8, or 9, got {v}")
        return v

    @field_validator("NACCINF")
    @classmethod
    def val_naccinf(cls, v):
        if v not in {0, 1, 8, 9}:
            raise ValueError(f"NACCINF must be 0, 1, 8, or 9, got {v}")
        return v

    @model_validator(mode="after")
    def validate_naccinf(self) -> "InfarctParent":
        if self.NACCINF != self.NPINF:
            raise ValueError(
                f"NACCINF={self.NACCINF} must equal NPINF={self.NPINF}."
            )
        return self


class InfarctCounts(BaseModel):
    """NPINF count children (NPINF1A/2A/3A/4A) - 4 variables.

    Allowable codes: 0-87 (count), 88=Not assessed, 99=Missing/unknown.
    Parent mirroring: NPINF=0 → 0, NPINF=8 → 88, NPINF=9 → 99.
    """
    model_config = ConfigDict(extra="forbid")

    NPINF1A: int = Field(
        description=(
            "Number of old infarcts in cerebral cortex. "
            "Codes: 0-87=count; 88=Not assessed; 99=Missing/unknown. "
            "If NPINF=0 → code 0. If NPINF=8 → code 88. If NPINF=9 → code 99."
        ),
    )
    NPINF2A: int = Field(
        description=(
            "Number of old infarcts in subcortical/periventricular white matter. "
            "Codes: 0-87=count; 88=Not assessed; 99=Missing/unknown. "
            "If NPINF=0 → code 0. If NPINF=8 → code 88. If NPINF=9 → code 99."
        ),
    )
    NPINF3A: int = Field(
        description=(
            "Number of old infarcts in deep cerebral gray matter or internal capsule. "
            "Codes: 0-87=count; 88=Not assessed; 99=Missing/unknown. "
            "If NPINF=0 → code 0. If NPINF=8 → code 88. If NPINF=9 → code 99."
        ),
    )
    NPINF4A: int = Field(
        description=(
            "Number of old infarcts in brainstem or cerebellum. "
            "Codes: 0-87=count; 88=Not assessed; 99=Missing/unknown. "
            "If NPINF=0 → code 0. If NPINF=8 → code 88. If NPINF=9 → code 99."
        ),
    )

    @field_validator("NPINF1A", "NPINF2A", "NPINF3A", "NPINF4A")
    @classmethod
    def val_infarct_count(cls, v):
        if not ((0 <= v <= 87) or v in {88, 99}):
            raise ValueError(f"Infarct count must be 0-87, 88, or 99, got {v}")
        return v


class InfarctSizes(BaseModel):
    """NPINF size children (12 variables - 3 sizes × 4 regions).

    Allowable codes: 0.0-20.0 (cm), 88.8=Not assessed or not applicable, 99.9=Missing/unknown.
    Parent mirroring (PDF + both sources agree):
      NPINF=0 → null (blank)
      NPINF=8 → 88.8 (not assessed)
      NPINF=9 → 99.9 (unknown)
    """
    model_config = ConfigDict(extra="forbid")

    NPINF1B: Optional[float] = Field(
        description=(
            "Size of largest old infarct in cerebral cortex (cm). "
            "Codes: 0.0-20.0=size in cm; 88.8=Not assessed or not applicable; 99.9=Missing/unknown. "
            "If NPINF=0 → null (blank). If NPINF=8 → 88.8. If NPINF=9 → 99.9."
        ),
    )
    NPINF1D: Optional[float] = Field(
        description=(
            "Size of second-largest old infarct in cerebral cortex (cm). "
            "Codes: 0.0-20.0; 88.8=Not applicable/not assessed; 99.9=Unknown. "
            "If only one infarct in this region → 88.8. "
            "If NPINF=0 → null (blank). If NPINF=8 → 88.8. If NPINF=9 → 99.9."
        ),
    )
    NPINF1F: Optional[float] = Field(
        description=(
            "Size of third-largest old infarct in cerebral cortex (cm). "
            "Codes: 0.0-20.0; 88.8=Not applicable/not assessed; 99.9=Unknown. "
            "If fewer than 3 infarcts in this region → 88.8. "
            "If NPINF=0 → null (blank). If NPINF=8 → 88.8. If NPINF=9 → 99.9."
        ),
    )
    NPINF2B: Optional[float] = Field(
        description=(
            "Size of largest old infarct in subcortical/periventricular white matter (cm). "
            "Codes: 0.0-20.0; 88.8=Not applicable/not assessed; 99.9=Unknown. "
            "If NPINF=0 → null (blank). If NPINF=8 → 88.8. If NPINF=9 → 99.9."
        ),
    )
    NPINF2D: Optional[float] = Field(
        description=(
            "Size of second-largest old infarct in subcortical/periventricular white matter (cm). "
            "Codes: 0.0-20.0; 88.8=Not applicable/not assessed; 99.9=Unknown. "
             "If NPINF=0 → null (blank). If NPINF=8 → 88.8. If NPINF=9 → 99.9."
        ),
    )
    NPINF2F: Optional[float] = Field(
        description=(
            "Size of third-largest old infarct in subcortical/periventricular white matter (cm). "
            "Codes: 0.0-20.0; 88.8=Not applicable/not assessed; 99.9=Unknown. "
             "If NPINF=0 → null (blank). If NPINF=8 → 88.8. If NPINF=9 → 99.9."
        ),
    )
    NPINF3B: Optional[float] = Field(
        description=(
            "Size of largest old infarct in deep cerebral gray matter or internal capsule (cm). "
            "Codes: 0.0-20.0; 88.8=Not applicable/not assessed; 99.9=Unknown. "
            "If NPINF=0 → null (blank). If NPINF=8 → 88.8. If NPINF=9 → 99.9."
        ),
    )
    NPINF3D: Optional[float] = Field(
        description=(
            "Size of second-largest old infarct in deep cerebral gray matter or internal capsule (cm). "
            "Codes: 0.0-20.0; 88.8=Not applicable/not assessed; 99.9=Unknown. "
            "If NPINF=0 → null (blank). If NPINF=8 → 88.8. If NPINF=9 → 99.9."
        ),
    )
    NPINF3F: Optional[float] = Field(
        description=(
            "Size of third-largest old infarct in deep cerebral gray matter or internal capsule (cm). "
            "Codes: 0.0-20.0; 88.8=Not applicable/not assessed; 99.9=Unknown. "
            "If NPINF=0 → null (blank). If NPINF=8 → 88.8. If NPINF=9 → 99.9."
        ),
    )
    NPINF4B: Optional[float] = Field(
        description=(
            "Size of largest old infarct in brainstem or cerebellum (cm). "
            "Codes: 0.0-20.0; 88.8=Not applicable/not assessed; 99.9=Unknown. "
            "If NPINF=0 → null (blank). If NPINF=8 → 88.8. If NPINF=9 → 99.9."
        ),
    )
    NPINF4D: Optional[float] = Field(
        description=(
            "Size of second-largest old infarct in brainstem or cerebellum (cm). "
            "Codes: 0.0-20.0; 88.8=Not applicable/not assessed; 99.9=Unknown. "
            "If NPINF=0 → null (blank). If NPINF=8 → 88.8. If NPINF=9 → 99.9."
        ),
    )
    NPINF4F: Optional[float] = Field(
        description=(
            "Size of third-largest old infarct in brainstem or cerebellum (cm). "
            "Codes: 0.0-20.0; 88.8=Not applicable/not assessed; 99.9=Unknown. "
            "If NPINF=0 → null (blank). If NPINF=8 → 88.8. If NPINF=9 → 99.9."
        ),
    )

    @field_validator(
        "NPINF1B", "NPINF1D", "NPINF1F",
        "NPINF2B", "NPINF2D", "NPINF2F",
        "NPINF3B", "NPINF3D", "NPINF3F",
        "NPINF4B", "NPINF4D", "NPINF4F",
    )
    @classmethod
    def val_infarct_size(cls, v):
        if v is None:
            return v
        if not ((0.0 <= v <= 20.0) or v in {88.8, 99.9}):
            raise ValueError(f"Infarct size must be 0.0-20.0, 88.8, 99.9, or null, got {v}")
        return v


class HemorrhageFindings(BaseModel):
    """NPHEMO parent + 3 children + NACCHEM - 5 variables.

    Parent mirroring: NPHEMO=0 → children=0, NPHEMO=8 → children=8, NPHEMO=9 → children=9.
    """
    model_config = ConfigDict(extra="forbid")

    NPHEMO: int = Field(
        description=(
            "Single or multiple old hemorrhages observed grossly. "
            "Codes: 0=None; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Parent variable for NPHEMO1/2/3."
        ),
    )
    NPHEMO1: int = Field(
        description=(
            "Single or multiple old hemorrhages observed grossly - "
            "Subdural or epidural hemorrhage present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPHEMO=0 → code NPHEMO1 as 0. If NPHEMO=8 → 8. If NPHEMO=9 → 9."
        ),
    )
    NPHEMO2: int = Field(
        description=(
            "Single or multiple old hemorrhages observed grossly - "
            "Primary parenchymal hemorrhage present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPHEMO=0 → code NPHEMO2 as 0. If NPHEMO=8 → 8. If NPHEMO=9 → 9."
        ),
    )
    NPHEMO3: int = Field(
        description=(
            "Single or multiple old hemorrhages observed grossly - "
            "Secondary parenchymal hemorrhage present (e.g., hemorrhagic transformation). "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPHEMO=0 → code NPHEMO3 as 0. If NPHEMO=8 → 8. If NPHEMO=9 → 9."
        ),
    )
    NACCHEM: int = Field(
        description=(
            "Derived: any hemorrhage or microbleed present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Code 1 if Parent NPHEMO=1 or Parent NPOLDD=1. "
            "Code 0 if Parent NPHEMO=0 or Parent NPOLDD=0. "
            "Code 8 if Parent NPHEMO=8 or Parent NPOLDD=8. "
            "Code 9 if Parent NPHEMO=9 or Parent NPOLDD=9. "
            "If NPHEMO=0 and NPOLDD=8 or 9 → code 9. "
            "If NPOLDD=0 and NPHEMO=8 or 9 → code 9. "
            "This is a Derived field."
        ),
    )

    @field_validator("NPHEMO", "NPHEMO1", "NPHEMO2", "NPHEMO3", "NACCHEM")
    @classmethod
    def val_hemo(cls, v):
        if v not in {0, 1, 8, 9}:
            raise ValueError(f"Hemorrhage field must be 0, 1, 8, or 9, got {v}")
        return v

    @model_validator(mode="after")
    def validate_hemo_children(self) -> "HemorrhageFindings":
        parent = self.NPHEMO
        child_map = {
            0: 0, 8: 8, 9: 9
        }
        if parent in child_map:
            expected = child_map[parent]
            for child in ["NPHEMO1", "NPHEMO2", "NPHEMO3"]:
                val = getattr(self, child)
                if val != expected:
                    raise ValueError(
                        f"{child}={val} inconsistent with NPHEMO={parent} "
                        f"(expected {expected})."
                    )
        return self


class MicrobleedFindings(BaseModel):
    """NPOLDD parent + 4 children - 5 variables.

    Parent mirroring: NPOLDD=0 → children=0, NPOLDD=8 → children=8, NPOLDD=9 → children=9.
    """
    model_config = ConfigDict(extra="forbid")

    NPOLDD: int = Field(
        description=(
            "Old cerebral microbleeds present. "
            "Codes: 0=None; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Parent variable for NPOLDD1-4."
        ),
    )
    NPOLDD1: int = Field(
        description=(
            "Number of cerebral microbleeds in cerebral cortex. "
            "Codes: 0=0; 1=1; 2=2; 3=3 or more; 8=Not assessed; 9=Unknown. "
            "If NPOLDD=0 → code NPOLDD1 as 0. If NPOLDD=8 → 8. If NPOLDD=9 → 9."
        ),
    )
    NPOLDD2: int = Field(
        description=(
            "Number of cerebral microbleeds in subcortical/periventricular white matter. "
            "Codes: 0=0; 1=1; 2=2; 3=3 or more; 8=Not assessed; 9=Unknown. "
            "If NPOLDD=0 → code NPOLDD2 as 0. If NPOLDD=8 → 8. If NPOLDD=9 → 9."
        ),
    )
    NPOLDD3: int = Field(
        description=(
            "Number of cerebral microbleeds in subcortical gray matter. "
            "Codes: 0=0; 1=1; 2=2; 3=3 or more; 8=Not assessed; 9=Unknown. "
            "If NPOLDD=0 → code NPOLDD3 as 0. If NPOLDD=8 → 8. If NPOLDD=9 → 9."
        ),
    )
    NPOLDD4: int = Field(
        description=(
            "Number of cerebral microbleeds in brainstem and cerebellum. "
            "Codes: 0=0; 1=1; 2=2; 3=3 or more; 8=Not assessed; 9=Unknown. "
            "If NPOLDD=0 → code NPOLDD4 as 0. If NPOLDD=8 → 8. If NPOLDD=9 → 9."
        ),
    )

    @field_validator("NPOLDD")
    @classmethod
    def val_npoldd(cls, v):
        if v not in {0, 1, 8, 9}:
            raise ValueError(f"NPOLDD must be 0, 1, 8, or 9, got {v}")
        return v

    @field_validator("NPOLDD1", "NPOLDD2", "NPOLDD3", "NPOLDD4")
    @classmethod
    def val_npoldd_child(cls, v):
        if v not in {0, 1, 2, 3, 8, 9}:
            raise ValueError(f"Microbleed count must be 0, 1, 2, 3, 8, or 9, got {v}")
        return v

    @model_validator(mode="after")
    def validate_npoldd_children(self) -> "MicrobleedFindings":
        parent = self.NPOLDD
        if parent in {0, 8, 9}:
            expected = {0: 0, 8: 8, 9: 9}[parent]
            for child in ["NPOLDD1", "NPOLDD2", "NPOLDD3", "NPOLDD4"]:
                val = getattr(self, child)
                if val != expected:
                    raise ValueError(
                        f"{child}={val} inconsistent with NPOLDD={parent} "
                        f"(expected {expected})."
                    )
        return self


class MicroinfarctFindings(BaseModel):
    """NPOLD parent + 4 children + NACCMICR - 6 variables.

    Parent mirroring: NPOLD=0 → children=0, NPOLD=8 → children=8, NPOLD=9 → children=9.
    """
    model_config = ConfigDict(extra="forbid")

    NPOLD: int = Field(
        description=(
            "Old microinfarcts not observed grossly (microscopic only). "
            "Codes: 0=None; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Parent variable for NPOLD1-4."
        ),
    )
    NPOLD1: int = Field(
        description=(
            "Number of old microinfarcts, not observed grossly in cerebral cortex. "
            "Codes: 0=0; 1=1; 2=2; 3=3 or more; 8=Not assessed; 9=Unknown. "
            "If NPOLD=0 → code NPOLD1 as 0. If NPOLD=8 → 8. If NPOLD=9 → 9."
        ),
    )
    NPOLD2: int = Field(
        description=(
            "Number of old microinfarcts, not observed grossly in subcortical/periventricular white matter. "
            "Codes: 0=0; 1=1; 2=2; 3=3 or more; 8=Not assessed; 9=Unknown. "
            "If NPOLD=0 → code NPOLD2 as 0. If NPOLD=8 → 8. If NPOLD=9 → 9."
        ),
    )
    NPOLD3: int = Field(
        description=(
            "Number of old microinfarcts, not observed grossly in subcortical gray matter. "
            "Codes: 0=0; 1=1; 2=2; 3=3 or more; 8=Not assessed; 9=Unknown. "
            "If NPOLD=0 → code NPOLD3 as 0. If NPOLD=8 → 8. If NPOLD=9 → 9."
        ),
    )
    NPOLD4: int = Field(
        description=(
            "Number of old microinfarcts, not observed grossly in brainstem and cerebellum. "
            "Codes: 0=0; 1=1; 2=2; 3=3 or more; 8=Not assessed; 9=Unknown. "
            "If NPOLD=0 → code NPOLD4 as 0. If NPOLD=8 → 8. If NPOLD=9 → 9."
        ),
    )
    NACCMICR: int = Field(
        description=(
            "Derived: microinfarcts present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Copy the value of NPOLD as it is. "
            "This is a derived field."
        ),
    )

    @field_validator("NPOLD")
    @classmethod
    def val_npold(cls, v):
        if v not in {0, 1, 8, 9}:
            raise ValueError(f"NPOLD must be 0, 1, 8, or 9, got {v}")
        return v

    @field_validator("NPOLD1", "NPOLD2", "NPOLD3", "NPOLD4")
    @classmethod
    def val_npold_child(cls, v):
        if v not in {0, 1, 2, 3, 8, 9}:
            raise ValueError(f"Microinfarct count must be 0, 1, 2, 3, 8, or 9, got {v}")
        return v

    @field_validator("NACCMICR")
    @classmethod
    def val_naccmicr(cls, v):
        if v not in {0, 1, 8, 9}:
            raise ValueError(f"NACCMICR must be 0, 1, 8, or 9, got {v}")
        return v

    @model_validator(mode="after")
    def validate_npold_children(self) -> "MicroinfarctFindings":
        parent = self.NPOLD
        if parent in {0, 8, 9}:
            expected = {0: 0, 8: 8, 9: 9}[parent]
            for child in ["NPOLD1", "NPOLD2", "NPOLD3", "NPOLD4"]:
                val = getattr(self, child)
                if val != expected:
                    raise ValueError(
                        f"{child}={val} inconsistent with NPOLD={parent} "
                        f"(expected {expected})."
                    )
        return self
    
    @model_validator(mode="after")
    def validate_naccmicr(self) -> "MicroinfarctFindings":
        if self.NACCMICR != self.NPOLD:
            raise ValueError(
                f"NACCMICR={self.NACCMICR} must equal NPOLD={self.NPOLD}."
            )
        return self


class OtherVascularPath(BaseModel):
    """NPPATH parent + 12 children + NPPATHOX - 14 variables.

    Parent mirroring: NPPATH=0 → children=0, NPPATH=8 → children=8, NPPATH=9 → children=9.
    Special: NPPATHO has only 0/1 codes - NPPATH=8 → NPPATHO=0, NPPATH=9 → NPPATHO=0.
    NPPATHOX: null unless NPPATHO=1.
    """
    model_config = ConfigDict(extra="forbid")

    NPPATH: int = Field(
        description=(
            "Other pathologic changes related to ischemic or vascular disease "
            "not previously specified by the dedicated vascular fields. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPPATH=0 → NACCNEC, NPPATH2-11, and NPPATHO are all forced to 0. "
            "Code 1 ONLY if the report explicitly mentions at least one of: "
            "laminar necrosis, acute neuronal necrosis, acute/subacute gross infarct, "
            "acute/subacute microinfarct, acute/subacute hemorrhage, "
            "vascular malformation, aneurysm, vasculitis, CADASIL, "
            "mineralization of blood vessels, or other acute ischemic change. "
            "Arteriolosclerosis and atherosclerosis alone do NOT trigger NPPATH=1 — "
            "they are captured by dedicated fields outside this pass. "
            "If none of the above findings are present → code 0."
        ),
    )
    NACCNEC: int = Field(
        description=(
            "Laminar necrosis present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPPATH=0 → code NACCNEC as 0. If NPPATH=8 → 8. If NPPATH=9 → 9."
        ),
    )
    NPPATH2: int = Field(
        description=(
            "Acute neuronal necrosis present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPPATH=0 → code NPPATH2 as 0. If NPPATH=8 → 8. If NPPATH=9 → 9."
        ),
    )
    NPPATH3: int = Field(
        description=(
            "Acute/subacute gross infarcts present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPPATH=0 → code NPPATH3 as 0. If NPPATH=8 → 8. If NPPATH=9 → 9."
        ),
    )
    NPPATH4: int = Field(
        description=(
            "Acute/subacute microinfarcts present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPPATH=0 → code NPPATH4 as 0. If NPPATH=8 → 8. If NPPATH=9 → 9."
        ),
    )
    NPPATH5: int = Field(
        description=(
            "Acute/subacute gross hemorrhage present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPPATH=0 → code NPPATH5 as 0. If NPPATH=8 → 8. If NPPATH=9 → 9."
        ),
    )
    NPPATH6: int = Field(
        description=(
            "Acute/subacute microhemorrhage present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPPATH=0 → code NPPATH6 as 0. If NPPATH=8 → 8. If NPPATH=9 → 9."
        ),
    )
    NPPATH7: int = Field(
        description=(
            "Vascular malformation of any type present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPPATH=0 → code NPPATH7 as 0. If NPPATH=8 → 8. If NPPATH=9 → 9."
        ),
    )
    NPPATH8: int = Field(
        description=(
            "Aneurysm of any type present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPPATH=0 → code NPPATH8 as 0. If NPPATH=8 → 8. If NPPATH=9 → 9."
        ),
    )
    NPPATH9: int = Field(
        description=(
            "Vasculitis of any type present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPPATH=0 → code NPPATH9 as 0. If NPPATH=8 → 8. If NPPATH=9 → 9."
        ),
    )
    NPPATH10: int = Field(
        description=(
            "CADASIL present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPPATH=0 → code NPPATH10 as 0. If NPPATH=8 → 8. If NPPATH=9 → 9."
        ),
    )
    NPPATH11: int = Field(
        description=(
            "Mineralization of blood vessels present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "If NPPATH=0 → code NPPATH11 as 0. If NPPATH=8 → 8. If NPPATH=9 → 9."
        ),
    )
    NPPATHO: int = Field(
        description=(
            "Other ischemic or vascular pathology present, beyond what is captured by "
            "NACCNEC and NPPATH2-11. "
            "Codes: 0=No; 1=Yes. (No 8 or 9 codes for this variable.) "
            "If NPPATH=0 → 0. If NPPATH=8 → 0. If NPPATH=9 → 0. "
            "If NPPATH=1 → code 1 ONLY if the report explicitly mentions vascular pathology "
            "not covered by: laminar necrosis (NACCNEC), acute neuronal necrosis (NPPATH2), "
            "acute/subacute gross infarcts (NPPATH3), acute/subacute microinfarcts (NPPATH4), "
            "acute/subacute gross hemorrhage (NPPATH5), acute/subacute microhemorrhage (NPPATH6), "
            "vascular malformation (NPPATH7), aneurysm (NPPATH8), vasculitis (NPPATH9), "
            "CADASIL (NPPATH10), mineralization of blood vessels (NPPATH11). "
            "IMPORTANT: Arteriolosclerosis is captured by NACCARTE — do NOT code NPPATHO=1 for it. "
            "Atherosclerosis is captured by NACCAVAS — do NOT code NPPATHO=1 for it. "
            "These two findings account for the vast majority of vascular pathology in these reports "
            "and must never trigger NPPATHO=1 regardless of severity or location described. "
            "If all vascular findings in the report are covered by the above → code 0. "
            "If not mentioned → code 0."
        ),
    )
    NPPATHOX: Optional[str] = Field(
        None,
        description=(
            "Free-text specify other ischemic or vascular pathology. "
            "Populate ONLY when NPPATHO=1. "
            "Record the exact finding as written in the report. "
            "Null in all other cases."
        ),
    )

    @field_validator("NPPATH")
    @classmethod
    def val_nppath(cls, v):
        if v not in {0, 1, 8, 9}:
            raise ValueError(f"NPPATH must be 0, 1, 8, or 9, got {v}")
        return v

    @field_validator(
        "NACCNEC", "NPPATH2", "NPPATH3", "NPPATH4", "NPPATH5",
        "NPPATH6", "NPPATH7", "NPPATH8", "NPPATH9", "NPPATH10", "NPPATH11"
    )
    @classmethod
    def val_nppath_child(cls, v):
        if v not in {0, 1, 8, 9}:
            raise ValueError(f"NPPATH child must be 0, 1, 8, or 9, got {v}")
        return v

    @field_validator("NPPATHO")
    @classmethod
    def val_nppatho(cls, v):
        if v not in {0, 1}:
            raise ValueError(f"NPPATHO must be 0 or 1, got {v}")
        return v

    @model_validator(mode="after")
    def validate_nppath_children(self) -> "OtherVascularPath":
        parent = self.NPPATH
        if parent in {0, 8, 9}:
            expected = {0: 0, 8: 8, 9: 9}[parent]
            standard_children = [
                "NACCNEC", "NPPATH2", "NPPATH3", "NPPATH4", "NPPATH5",
                "NPPATH6", "NPPATH7", "NPPATH8", "NPPATH9", "NPPATH10", "NPPATH11"
            ]
            for child in standard_children:
                val = getattr(self, child)
                if val != expected:
                    raise ValueError(
                        f"{child}={val} inconsistent with NPPATH={parent} "
                        f"(expected {expected})."
                    )
            # NPPATHO has no 8/9 codes - always 0 when parent is not 1
            if self.NPPATHO != 0:
                raise ValueError(
                    f"NPPATHO={self.NPPATHO} inconsistent with NPPATH={parent} "
                    f"(NPPATHO must be 0 when NPPATH != 1)."
                )
        return self


class MiscVascular(BaseModel):
    """Miscellaneous vascular variables - 3 variables."""
    model_config = ConfigDict(extra="forbid")

    NPMICRO: int = Field(
        description=(
            "Multiple microinfarcts present. "
            "Codes: 1=Present; 2=Absent; 3=Not assessed; 9=Unknown. "
            "If not mentioned → code 2."
        ),
    )
    NPART: int = Field(
        description=(
            "Subcortical arteriosclerotic leukoencephalopathy (Binswanger disease) present. "
            "Codes: 1=Present; 2=Absent; 3=Not assessed; 9=Unknown. "
            "Look for 'Binswanger', 'subcortical arteriosclerotic leukoencephalopathy', "
            "'SAE'. If not mentioned → code 2."
        ),
    )
    NPOANG: int = Field(
        description=(
            "Angiopathy other than amyloid angiopathy present. "
            "Codes: 1=Present; 2=Absent; 3=Not assessed; 9=Unknown. "
            "Look for non-CAA angiopathy: hypertensive angiopathy, diabetic angiopathy, etc. "
            "If not mentioned → code 2."
        ),
    )

    @field_validator("NPMICRO", "NPART", "NPOANG")
    @classmethod
    def val_misc(cls, v):
        if v not in {1, 2, 3, 9}:
            raise ValueError(f"Misc vascular field must be 1, 2, 3, or 9, got {v}")
        return v


# ---------------------------------------------------------------------------
# Top-level Pass 3 model
# ---------------------------------------------------------------------------

class Pass3Extraction(BaseModel):
    """Pass 3 extraction - 51 variables.

    Covers: infarct parent/derived, infarct counts (×4 regions), infarct sizes (×12),
    hemorrhage parent/children/derived, microbleed parent/children, microinfarct parent/children/derived,
    other vascular pathology parent/children, and miscellaneous vascular flags.
    """
    model_config = ConfigDict(extra="forbid")

    infarct_parent: InfarctParent = Field(
        description="NPINF (parent), NACCINF (derived)"
    )
    infarct_counts: InfarctCounts = Field(
        description="NPINF1A, NPINF2A, NPINF3A, NPINF4A - infarct counts by region"
    )
    infarct_sizes: InfarctSizes = Field(
        description=(
            "NPINF1B/D/F, NPINF2B/D/F, NPINF3B/D/F, NPINF4B/D/F - "
            "infarct sizes by region (cm)"
        )
    )
    hemorrhage: HemorrhageFindings = Field(
        description="NPHEMO, NPHEMO1, NPHEMO2, NPHEMO3, NACCHEM"
    )
    microbleeds: MicrobleedFindings = Field(
        description="NPOLDD, NPOLDD1, NPOLDD2, NPOLDD3, NPOLDD4"
    )
    microinfarcts: MicroinfarctFindings = Field(
        description="NPOLD, NPOLD1, NPOLD2, NPOLD3, NPOLD4, NACCMICR"
    )
    other_vascular: OtherVascularPath = Field(
        description=(
            "NPPATH, NACCNEC, NPPATH2-11, NPPATHO, NPPATHOX"
        )
    )
    misc_vascular: MiscVascular = Field(
        description="NPMICRO, NPART, NPOANG"
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
    def validate_confidence_value(self) -> "Pass3Extraction":
        if self.extraction_confidence and self.extraction_confidence not in {
            "high", "moderate", "low"
        }:
            raise ValueError("extraction_confidence must be 'high', 'moderate', or 'low'")
        return self

    @model_validator(mode="after")
    def enforce_null_policy(self) -> "Pass3Extraction":
        """Non-nullable fields must not be null."""
        nullable = {
            "NPPATHOX",
            "NPINF1B", "NPINF1D", "NPINF1F",
            "NPINF2B", "NPINF2D", "NPINF2F",
            "NPINF3B", "NPINF3D", "NPINF3F",
            "NPINF4B", "NPINF4D", "NPINF4F",
        }
        sections = [
            self.infarct_parent, self.infarct_counts, self.infarct_sizes,
            self.hemorrhage, self.microbleeds, self.microinfarcts,
            self.other_vascular, self.misc_vascular,
        ]
        for section in sections:
            for field_name in section.model_fields:
                if field_name not in nullable and getattr(section, field_name) is None:
                    raise ValueError(
                        f"{field_name} must not be null — use 0, 8, 9, 88, 99, 88.8, or 99.9, etc. (depending on variable)."
                    )
        return self

    @model_validator(mode="after")
    def validate_nacchem(self) -> "Pass3Extraction":
        nphemo = self.hemorrhage.NPHEMO
        npoldd = self.microbleeds.NPOLDD
        nacchem = self.hemorrhage.NACCHEM

        if nphemo == 1 or npoldd == 1:
            expected = 1
        elif nphemo == 0 and npoldd == 0:
            expected = 0
        elif nphemo == 8 and npoldd == 8:
            expected = 8
        elif nphemo == 9 and npoldd == 9:
            expected = 9
        else:
            # one is 0 and the other is 8 or 9
            expected = 9

        if nacchem != expected:
            raise ValueError(
                f"NACCHEM={nacchem} is inconsistent with NPHEMO={nphemo} and NPOLDD={npoldd}. "
                f"Expected {expected}."
            )
        return self


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_pass3_model() -> Type[BaseModel]:
    """Return the top-level extraction model class."""
    return Pass3Extraction


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


def build_pass3_format_instructions() -> str:
    """Walk the Pydantic schema and emit LLM-facing format instructions."""
    model = Pass3Extraction
    lines = [
        "PASS 3 - Extract ONLY the vascular variables listed below.",
        "Respond with a single JSON object with EXACTLY these top-level keys:",
        "",
        "  infarct_parent, infarct_counts, infarct_sizes, hemorrhage,",
        "  microbleeds, microinfarcts, other_vascular, misc_vascular,",
        "  field_annotations, extraction_confidence, extraction_notes",
        "",
        "Each top-level key maps to a nested object containing its variables.",
        "NEVER output variables at the root level - they must always be nested.",
        "",
        "NULL POLICY: NPPATHOX is nullable (null unless NPPATHO=1).",
        "  NPINF size fields (NPINF1B/D/F × 4 regions) are nullable (null when NPINF=0).",
        "  All other variables must have explicit numeric codes.",
        "",
        "PARENT-CHILD MIRRORING RULES - apply these exactly:",
        "  NPINF count fields (NPINF1A/2A/3A/4A):",
        "    NPINF=0 → 0  |  NPINF=8 → 88  |  NPINF=9 → 99",
        "  NPINF size fields (NPINF1B/D/F × 4 regions):",
        "    NPINF=0 → null (blank)",
        "    NPINF=8 → 88.8  |  NPINF=9 → 99.9",
        "  NPHEMO1/2/3:",
        "    NPHEMO=0 → 0  |  NPHEMO=8 → 8  |  NPHEMO=9 → 9",
        "  NPOLD1/2/3/4:",
        "    NPOLD=0 → 0  |  NPOLD=8 → 8  |  NPOLD=9 → 9",
        "  NPOLDD1/2/3/4:",
        "    NPOLDD=0 → 0  |  NPOLDD=8 → 8  |  NPOLDD=9 → 9",
        "  NACCNEC, NPPATH2-11:",
        "    NPPATH=0 → 0  |  NPPATH=8 → 8  |  NPPATH=9 → 9",
        "  NPPATHO (only codes 0/1 - no 8 or 9):",
        "    NPPATH=0 → 0  |  NPPATH=8 → 0  |  NPPATH=9 → 0",
        "",
        "DERIVED FIELDS:",
        "  NACCINF = NPINF (copy directly).",
        "  NACCMICR = NPOLD (copy directly).",
        "  NACCHEM: 1 if NPHEMO=1 or NPOLDD=1; 0 if both=0; 8 if both=8; 9 if both=9;",
        "    9 if one=0 and the other is 8 or 9.",
        "",
        "NPMICRO, NPART, NPOANG: use 1=Present, 2=Absent, 3=Not assessed, 9=Unknown.",
        "  If not mentioned in the report → code 2.",
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