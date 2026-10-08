"""schema4.py - Pass 4: Clinical dx summary, other disease flags, criteria (42 vars).

Pass 4 variables:
  Normal brain dx:   NPPNORM, NPCNORM
  AD insufficient:   NPPADP, NPCADP
  Hippocampal scl:   NPPHIPP, NPCHIPP
  Prion dx:          NPPPRION, NPCPRION
  Other dx pairs:    NPPOTH1, NPCOTH1, NPOTH1X,
                     NPPOTH2, NPCOTH2, NPOTH2X,
                     NPPOTH3, NPCOTH3, NPOTH3X
  Write-in dx:       NACCOTHP, NACCWRI1, NACCWRI2, NACCWRI3
  Other flags:       NPPDXA-N (13 disease flags), NACCDOWN, NPSCL
  Clinical:          NPNIT, NPCERAD, NPADRDA, NPOCRIT, NPVOTH, NPLEWYCS
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

class NormalAndInsuffAD(BaseModel):
    """Normal brain dx + insufficient AD dx pairs - 4 variables."""
    model_config = ConfigDict(extra="forbid")

    NPPNORM: Optional[int] = Field(
        None,
        description=(
            "Normal brain listed as PRIMARY diagnosis. "
            "Codes: 1=Yes; 2=No; null=Not stated or not determinable from the report. "
            "Code 1 only if the report explicitly concludes that the brain is normal "
            "or within normal limits for age with no significant pathology. "
            "Code 2 only when the report explicitly establishes that a normal-brain "
            "diagnosis is not present, such as when another specific pathological "
            "diagnosis is identified as the primary diagnosis. "
            "Do NOT use 2 merely because 'normal brain' is not mentioned. "
            "If the report does not provide enough information to determine whether "
            "normal brain is the primary diagnosis, use null."
        ),
    )

    NPCNORM: Optional[int] = Field(
        None,
        description=(
            "Normal brain listed as CONTRIBUTING diagnosis. "
            "Codes: 1=Yes; 2=No; null=Not stated or not determinable from the report. "
            "Code 1 only if the report explicitly identifies normal brain as a "
            "contributing diagnosis. "
            "Code 2 only when the report explicitly establishes that normal brain "
            "is not a contributing diagnosis. "
            "Do NOT use 2 merely because 'normal brain' is not mentioned. "
            "If the report does not provide enough information to determine whether "
            "normal brain is a contributing diagnosis, use null."
        ),
    )

    NPPADP: Optional[int] = Field(
        None,
        description=(
            "AD pathology present but insufficient for AD diagnosis - PRIMARY. "
            "Codes: 1=Yes; 2=No; null=Not stated or not determinable from the report. "
            "Code 1 when the report explicitly identifies subthreshold AD pathology "
            "as the primary neuropathological diagnosis, such as low or intermediate "
            "ADNC being the principal finding. "
            "Code 2 only when the report explicitly establishes that subthreshold AD "
            "pathology is not the primary diagnosis, such as when another diagnosis is "
            "explicitly identified as primary and the AD pathology is explicitly assigned "
            "a contributing/secondary role or explicitly absent. "
            "Do NOT use 2 merely because insufficient AD pathology is not mentioned. "
            "If the report does not provide enough information to determine whether "
            "insufficient AD pathology is the primary diagnosis, use null."
        ),
    )

    NPCADP: Optional[int] = Field(
        None,
        description=(
            "AD pathology present but insufficient for AD diagnosis - CONTRIBUTING. "
            "Codes: 1=Yes; 2=No; null=Not stated or not determinable from the report. "
            "Code 1 when subthreshold AD pathology is explicitly identified as a "
            "contributing or secondary neuropathological diagnosis alongside another "
            "primary diagnosis. "
            "Code 2 only when the report explicitly establishes that subthreshold AD "
            "pathology is not a contributing diagnosis. "
            "Do NOT use 2 merely because insufficient AD pathology is not mentioned. "
            "If the report does not provide enough information to determine whether "
            "insufficient AD pathology is a contributing diagnosis, use null."
        ),
    )

    @field_validator("NPPNORM", "NPCNORM", "NPPADP", "NPCADP")
    @classmethod
    def val_binary_nullable(cls, v):
        if v is not None and v not in {1, 2}:
            raise ValueError(f"Field must be 1, 2, or null, got {v}")
        return v

    @model_validator(mode="after")
    def val_primary_contributing_mutex(self) -> "NormalAndInsuffAD":
        if self.NPPNORM == 1 and self.NPCNORM == 1:
            raise ValueError("NPPNORM (primary) and NPCNORM (contributing) cannot both be 1")
        if self.NPPADP == 1 and self.NPCADP == 1:
            raise ValueError("NPPADP (primary) and NPCADP (contributing) cannot both be 1")
        return self


class HippocampalAndPrionDx(BaseModel):
    """Hippocampal sclerosis dx + prion dx pairs - 4 variables."""
    model_config = ConfigDict(extra="forbid")

    NPPHIPP: Optional[int] = Field(
        None,
        description=(
            "Hippocampal sclerosis listed as PRIMARY diagnosis. "
            "Codes: 1=Yes; 2=No; null=Not stated or not determinable from the report. "
            "Code 1 only if the report explicitly identifies hippocampal sclerosis "
            "as the principal neuropathological diagnosis. "
            "Code 2 only when the report explicitly establishes that hippocampal "
            "sclerosis is not the primary diagnosis. "
            "Do NOT use 2 merely because hippocampal sclerosis is not mentioned. "
            "If the report does not provide enough information to determine whether "
            "hippocampal sclerosis is the primary diagnosis, use null."
        ),
    )

    NPCHIPP: Optional[int] = Field(
        None,
        description=(
            "Hippocampal sclerosis listed as CONTRIBUTING diagnosis. "
            "Codes: 1=Yes; 2=No; null=Not stated or not determinable from the report. "
            "Code 1 only if hippocampal sclerosis is explicitly identified as a "
            "contributing or secondary neuropathological diagnosis. "
            "Code 2 only when the report explicitly establishes that hippocampal "
            "sclerosis is not a contributing diagnosis. "
            "Do NOT use 2 merely because hippocampal sclerosis is not mentioned. "
            "If the report does not provide enough information to determine whether "
            "hippocampal sclerosis is a contributing diagnosis, use null."
        ),
    )

    NPPPRION: Optional[int] = Field(
        None,
        description=(
            "Prion associated disease listed as PRIMARY diagnosis. "
            "Codes: 1=Yes; 2=No; null=Not stated or not determinable from the report. "
            "Code 1 only if prion disease is explicitly identified as the principal "
            "neuropathological diagnosis. "
            "Code 2 only when the report explicitly establishes that prion disease "
            "is not the primary diagnosis. "
            "Do NOT use 2 merely because prion disease is not mentioned. "
            "If the report does not provide enough information to determine whether "
            "prion disease is the primary diagnosis, use null."
        ),
    )

    NPCPRION: Optional[int] = Field(
        None,
        description=(
            "Prion associated disease listed as CONTRIBUTING diagnosis. "
            "Codes: 1=Yes; 2=No; null=Not stated or not determinable from the report. "
            "Code 1 only if prion disease is explicitly identified as a contributing "
            "or secondary neuropathological diagnosis. "
            "Code 2 only when the report explicitly establishes that prion disease "
            "is not a contributing diagnosis. "
            "Do NOT use 2 merely because prion disease is not mentioned. "
            "If the report does not provide enough information to determine whether "
            "prion disease is a contributing diagnosis, use null."
        ),
    )

    @field_validator("NPPHIPP", "NPCHIPP", "NPPPRION", "NPCPRION")
    @classmethod
    def val_binary_nullable(cls, v):
        if v is not None and v not in {1, 2}:
            raise ValueError(f"Field must be 1, 2, or null, got {v}")
        return v

    @model_validator(mode="after")
    def val_primary_contributing_mutex(self) -> "HippocampalAndPrionDx":
        if self.NPPHIPP == 1 and self.NPCHIPP == 1:
            raise ValueError("NPPHIPP (primary) and NPCHIPP (contributing) cannot both be 1")
        if self.NPPPRION == 1 and self.NPCPRION == 1:
            raise ValueError("NPPPRION (primary) and NPCPRION (contributing) cannot both be 1")
        return self


class OtherDxPairs(BaseModel):
    """Other primary/contributing dx pairs with free-text specify - 9 variables."""
    model_config = ConfigDict(extra="forbid")

    NPPOTH1: Optional[int] = Field(
        None,
        description=(
            "Other primary pathologic diagnosis 1. "
            "Codes: 1=Yes; 2=No; null=Not stated or not determinable from the report. "
            "Code 1 ONLY if there is a primary diagnosis in the report that has NO dedicated field "
            "anywhere else on the NP form. "
            "The following diagnoses ALL have dedicated fields in other passes and must NEVER appear here: "
            "Alzheimer's disease / ADNC, "
            "Lewy body disease, "
            "Cerebrovascular disease, arteriolosclerosis, atherosclerosis, "
            "FTLD-tau (including CBD, PSP, Pick's disease, CTE, argyrophilic grains), "
            "FTLD-TDP, "
            "FTLD-other (including NIFID, atypical FTLD-U), "
            "ALS/MND, "
            "Hippocampal sclerosis, "
            "Prion disease / CJD, "
            "Down syndrome. "
            "Code 1 for a qualifying diagnosis with NO dedicated field anywhere else. "
            "Populate NPOTH1X with the diagnosis name when NPPOTH1=1 or NPCOTH1=1. "
            "Code 2 only when the report establishes that no qualifying other primary "
            "diagnosis occupies this position. "
            "Do NOT use 2 merely because an other primary diagnosis is not mentioned. "
            "If the report does not provide enough information to determine whether "
            "a qualifying other primary diagnosis exists in this position, use null."
        ),
    )

    NPCOTH1: Optional[int] = Field(
        None,
        description=(
            "Other contributing pathologic diagnosis 1. "
            "Codes: 1=Yes; 2=No; null=Not stated or not determinable from the report. "
            "Applies the same exclusion logic as NPPOTH1 - diagnoses with dedicated fields "
            "must never appear here. "
            "Code 1 if the diagnosis in NPOTH1X is explicitly identified as contributing "
            "or secondary rather than primary. "
            "Code 2 only when the report establishes that the diagnosis occupying this "
            "position is not contributing. "
            "Do NOT use 2 merely because a contributing other diagnosis is not mentioned. "
            "If the report does not provide enough information to determine whether "
            "a qualifying other diagnosis is contributing, use null."
        ),
    )

    NPOTH1X: Optional[str] = Field(
        None,
        description=(
            "Other pathologic diagnosis 1 - free-text specify. "
            "Populate when NPPOTH1=1 OR NPCOTH1=1. Otherwise null. "
            "Record the diagnosis name exactly as stated in the report's final diagnosis section. "
            "Valid characters: no single quotes, double quotes, %, or &."
        ),
    )

    NPPOTH2: Optional[int] = Field(
        None,
        description=(
            "Other primary pathologic diagnosis 2. "
            "Codes: 1=Yes; 2=No; null=Not stated or not determinable from the report. "
            "Code 1 only if a second distinct qualifying other diagnosis exists beyond "
            "the one in NPOTH1X, applying the same exclusion logic as NPPOTH1. "
            "Populate NPOTH2X with the diagnosis name when NPPOTH2=1 or NPCOTH2=1. "
            "Code 2 only when the report establishes that no second qualifying other "
            "primary diagnosis exists in this position. "
            "Do NOT use 2 merely because a second other primary diagnosis is not mentioned. "
            "If the report does not provide enough information to determine whether "
            "a second qualifying other primary diagnosis exists, use null."
        ),
    )

    NPCOTH2: Optional[int] = Field(
        None,
        description=(
            "Other contributing pathologic diagnosis 2. "
            "Codes: 1=Yes; 2=No; null=Not stated or not determinable from the report. "
            "Code 1 if the second diagnosis in NPOTH2X is explicitly identified as "
            "contributing or secondary rather than primary. "
            "Code 2 only when the report establishes that the diagnosis occupying this "
            "position is not contributing. "
            "Do NOT use 2 merely because a second contributing diagnosis is not mentioned. "
            "If the report does not provide enough information to determine whether "
            "a second qualifying other diagnosis is contributing, use null."
        ),
    )
    
    NPOTH2X: Optional[str] = Field(
        None,
        description=(
            "Other pathologic diagnosis 2 - free-text specify. "
            "Populate when NPPOTH2=1 OR NPCOTH2=1. Otherwise null. "
            "Record the diagnosis name exactly as stated in the report's final diagnosis section. "
            "Valid characters: no single quotes, double quotes, %, or &."
        ),
    )

    NPPOTH3: Optional[int] = Field(
        None,
        description=(
            "Other primary pathologic diagnosis 3. "
            "Codes: 1=Yes; 2=No; null=Not stated or not determinable from the report. "
            "Code 1 only if a third distinct qualifying other diagnosis exists beyond "
            "NPOTH1X and NPOTH2X, applying the same exclusion logic as NPPOTH1. "
            "Populate NPOTH3X with the diagnosis name when NPPOTH3=1 or NPCOTH3=1. "
            "Code 2 only when the report establishes that no third qualifying other "
            "primary diagnosis exists in this position. "
            "Do NOT use 2 merely because a third other primary diagnosis is not mentioned. "
            "If the report does not provide enough information to determine whether "
            "a third qualifying other primary diagnosis exists, use null."
        ),
    )

    NPCOTH3: Optional[int] = Field(
        None,
        description=(
            "Other contributing pathologic diagnosis 3. "
            "Codes: 1=Yes; 2=No; null=Not stated or not determinable from the report. "
            "Code 1 if the third diagnosis in NPOTH3X is explicitly identified as "
            "contributing or secondary rather than primary. "
            "Code 2 only when the report establishes that the diagnosis occupying this "
            "position is not contributing. "
            "Do NOT use 2 merely because a third contributing diagnosis is not mentioned. "
            "If the report does not provide enough information to determine whether "
            "a third qualifying other diagnosis is contributing, use null."
        ),
    )

    NPOTH3X: Optional[str] = Field(
        None,
        description=(
            "Other pathologic diagnosis 3 - free-text specify. "
            "Populate when NPPOTH3=1 OR NPCOTH3=1. Otherwise null. "
            "Record the diagnosis name exactly as stated in the report's final diagnosis section. "
            "Valid characters: no single quotes, double quotes, %, or &."
        ),
    )

    @field_validator("NPPOTH1", "NPCOTH1", "NPPOTH2", "NPCOTH2", "NPPOTH3", "NPCOTH3")

    @classmethod
    def val_binary_nullable(cls, v):
        if v is not None and v not in {1, 2}:
            raise ValueError(f"Field must be 1, 2, or null, got {v}")
        return v

    @model_validator(mode="after")
    def val_primary_contributing_mutex(self) -> "OtherDxPairs":
        if self.NPPOTH1 == 1 and self.NPCOTH1 == 1:
            raise ValueError("NPPOTH1 (primary) and NPCOTH1 (contributing) cannot both be 1")
        if self.NPPOTH2 == 1 and self.NPCOTH2 == 1:
            raise ValueError("NPPOTH2 (primary) and NPCOTH2 (contributing) cannot both be 1")
        if self.NPPOTH3 == 1 and self.NPCOTH3 == 1:
            raise ValueError("NPPOTH3 (primary) and NPCOTH3 (contributing) cannot both be 1")
        return self

    @model_validator(mode="after")
    def val_specify_blank_if_no_dx(self) -> "OtherDxPairs":
        if self.NPPOTH1 == 2 and self.NPCOTH1 == 2 and self.NPOTH1X is not None:
            raise ValueError("NPOTH1X must be null when both NPPOTH1 and NPCOTH1 are 2")
        if self.NPPOTH1 == 1 or self.NPCOTH1 == 1:
            if self.NPOTH1X is None:
                raise ValueError("NPOTH1X must be populated when NPPOTH1=1 or NPCOTH1=1")

        if self.NPPOTH2 == 2 and self.NPCOTH2 == 2 and self.NPOTH2X is not None:
            raise ValueError("NPOTH2X must be null when both NPPOTH2 and NPCOTH2 are 2")
        if self.NPPOTH2 == 1 or self.NPCOTH2 == 1:
            if self.NPOTH2X is None:
                raise ValueError("NPOTH2X must be populated when NPPOTH2=1 or NPCOTH2=1")

        if self.NPPOTH3 == 2 and self.NPCOTH3 == 2 and self.NPOTH3X is not None:
            raise ValueError("NPOTH3X must be null when both NPPOTH3 and NPCOTH3 are 2")
        if self.NPPOTH3 == 1 or self.NPCOTH3 == 1:
            if self.NPOTH3X is None:
                raise ValueError("NPOTH3X must be populated when NPPOTH3=1 or NPCOTH3=1")

        return self


class WriteInDx(BaseModel):
    """NACCOTHP write-in diagnosis fields - 4 variables."""
    model_config = ConfigDict(extra="forbid")

    NACCOTHP: int = Field(
        description=(
            "Flag indicating whether one or more 'other' pathologic diagnoses exist "
            "that have no dedicated field anywhere else. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. Do not use 2. "
            "Code 1 ONLY if the report contains a diagnosis that cannot be captured by "
            "any dedicated field — not AD/ADNC, not Lewy body disease, not cerebrovascular "
            "disease, not any FTLD subtype, not ALS/MND, not hippocampal sclerosis, "
            "not prion disease, not Down syndrome. "
            "Examples that DO trigger NACCOTHP=1: Biondi bodies, axon and myelin loss of a "
            "specific named tract (e.g. fasciculus gracilis), NBIA, PART when explicitly "
            "named as a standalone diagnosis, any finding listed in the report's final "
            "diagnosis section that has no home in any other NP form variable. "
            "NACCOTHP=1 is independent of NPOTH1X/2X/3X — a diagnosis can appear in both. "
            "When in doubt → code 0. "
            "NACCWRI1/2/3 must be null if NACCOTHP != 1."
        ),
    )
    NACCWRI1: Optional[str] = Field(
        None,
        description=(
            "First other pathologic diagnosis write-in. "
            "Populate only when NACCOTHP=1. Otherwise null. "
            "Record the diagnosis name exactly as it appears in the report's final diagnosis section. "
            "Apply the same exclusion logic as NACCOTHP — do not write in AD, Lewy body, "
            "cerebrovascular disease, FTLD, ALS/MND, hippocampal sclerosis, prion disease, "
            "or Down syndrome. Only diagnoses with no dedicated field belong here."
        ),
    )
    NACCWRI2: Optional[str] = Field(
        None,
        description=(
            "Second other pathologic diagnosis write-in. "
            "Populate only when NACCOTHP=1 and a second qualifying diagnosis exists. Otherwise null. "
            "Same exclusion logic as NACCWRI1."
        ),
    )
    NACCWRI3: Optional[str] = Field(
        None,
        description=(
            "Third other pathologic diagnosis write-in. "
            "Populate only when NACCOTHP=1 and a third qualifying diagnosis exists. Otherwise null. "
            "Same exclusion logic as NACCWRI1."
        ),
    )

    @field_validator("NACCOTHP")
    @classmethod
    def val_naccothp(cls, v):
        if v not in {0, 1, 8, 9}:
            raise ValueError(f"NACCOTHP must be 0, 1, 8, or 9, got {v}")
        return v

    @model_validator(mode="after")
    def val_writein_blank_if_no_dx(self) -> "WriteInDx":
        if self.NACCOTHP != 1:
            if self.NACCWRI1 is not None:
                raise ValueError("NACCWRI1 must be null when NACCOTHP != 1")
            if self.NACCWRI2 is not None:
                raise ValueError("NACCWRI2 must be null when NACCOTHP != 1")
            if self.NACCWRI3 is not None:
                raise ValueError("NACCWRI3 must be null when NACCOTHP != 1")
        return self


class OtherDiseaseFlags(BaseModel):
    """Other disease presence flags (NPPDXA-N + NACCDOWN) - 15 variables."""
    model_config = ConfigDict(extra="forbid")

    NPPDXA: YesNoCode = Field(
        description=(
            "Pigment-spheroid degeneration / NBIA present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for 'NBIA', 'neurodegeneration with brain iron accumulation', "
            "'Hallervorden-Spatz', 'PKAN', 'pigment-spheroid degeneration'. "
            "If not mentioned → code 0. "
            "Do not use 2 for this variable."
        ),
    )
    NPPDXB: YesNoCode = Field(
        description=(
            "Multiple system atrophy (MSA) present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for 'MSA', 'multiple system atrophy', 'striatonigral degeneration', "
            "'olivopontocerebellar atrophy', 'glial cytoplasmic inclusions'. "
            "If not mentioned → code 0. "
            "Do not use 2 for this variable."
        ),
    )
    NPPDXD: YesNoCode = Field(
        description=(
            "Trinucleotide repeat disease present (Huntington disease, SCA, other). "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for 'Huntington', 'spinocerebellar ataxia', 'SCA', 'trinucleotide repeat'. "
            "If not mentioned → code 0. "
            "Do not use 2 for this variable."
        ),
    )
    NPPDXE: YesNoCode = Field(
        description=(
            "Malformation of cortical development present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for 'cortical dysplasia', 'heterotopia', 'lissencephaly', "
            "'polymicrogyria', 'focal cortical malformation'. "
            "If not mentioned → code 0. "
            "Do not use 2 for this variable."
        ),
    )
    NPPDXF: YesNoCode = Field(
        description=(
            "Metabolic/storage disorder of any type present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for lysosomal storage disorders, mitochondrial disease, "
            "metabolic encephalopathy. If not mentioned → code 0. "
            "Do not use 2 for this variable."
        ),
    )
    NPPDXG: YesNoCode = Field(
        description=(
            "White matter disease / leukodystrophy present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for 'leukodystrophy', 'adrenoleukodystrophy', 'metachromatic leukodystrophy'. "
            "If not mentioned → code 0. "
            "Do not use 2 for this variable."
        ),
    )
    NPPDXH: YesNoCode = Field(
        description=(
            "White matter disease, multiple sclerosis or other demyelinating disease present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for 'multiple sclerosis', 'MS', 'demyelinating disease', 'plaques'. "
            "If not mentioned → code 0. "
            "Do not use 2 for this variable."
        ),
    )
    NPPDXI: YesNoCode = Field(
        description=(
            "Acute contusion/traumatic brain injury of any type present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for 'acute TBI', 'acute contusion', 'acute traumatic injury'. "
            "If not mentioned → code 0. "
            "Do not use 2 for this variable."
        ),
    )
    NPPDXJ: YesNoCode = Field(
        description=(
            "Chronic contusion/traumatic brain injury of any type present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for 'chronic TBI', 'old contusion', 'remote traumatic injury'. "
            "If not mentioned → code 0. "
            "Do not use 2 for this variable."
        ),
    )
    NPPDXK: YesNoCode = Field(
        description=(
            "Primary neoplasm present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for primary brain tumors: glioma, meningioma, lymphoma, etc. "
            "If not mentioned → code 0. "
            "Do not use 2 for this variable."
        ),
    )
    NPPDXL: YesNoCode = Field(
        description=(
            "Metastatic neoplasm present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for 'metastasis', 'metastatic tumor', 'secondary neoplasm'. "
            "If not mentioned → code 0. "
            "Do not use 2 for this variable."
        ),
    )
    NPPDXM: YesNoCode = Field(
        description=(
            "Infectious process of any type present (encephalitis, abscess, etc.). "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for 'encephalitis', 'abscess', 'meningitis', 'viral', 'bacterial', "
            "'fungal infection'. If not mentioned → code 0. "
            "Do not use 2 for this variable."
        ),
    )
    NPPDXN: YesNoCode = Field(
        description=(
            "Herniation (any site) present. "
            "Codes: 0=No; 1=Yes; 8=Not assessed; 9=Unknown. "
            "Look for 'transtentorial herniation', 'uncal herniation', 'tonsillar herniation'. "
            "If not mentioned → code 0. "
            "Do not use 2 for this variable."
        ),
    )
    NACCDOWN: int = Field(
        description=(
            "Down syndrome derived flag. "
            "Codes: 1=Flag for known Down syndrome mutation; "
            "7=No flag for known mutation (not present, not assessed, missing, or unknown). "
            "Look for 'Down syndrome', 'trisomy 21'. "
            "If not mentioned → code 7."
        ),
    )
    NPSCL: int = Field(
        description=(
            "Medial temporal lobe sclerosis present (including hippocampal sclerosis). "
            "Codes: 1=Present; 2=Absent; 3=Not assessed; 9=Unknown. "
            "If not mentioned → code 2."
        ),
    )

    @field_validator("NPSCL")
    @classmethod
    def val_npscl(cls, v):
        if v not in {1, 2, 3, 9}:
            raise ValueError(f"NPSCL must be 1, 2, 3, or 9, got {v}")
        return v

    @field_validator("NACCDOWN")
    @classmethod
    def val_naccdown(cls, v):
        if v not in {1, 7}:
            raise ValueError(f"NACCDOWN must be 1 or 7, got {v}")
        return v


class CriteriaAndMiscDx(BaseModel):
    """Diagnostic criteria and miscellaneous dx flags - 6 variables."""
    model_config = ConfigDict(extra="forbid")

    NPNIT: int = Field(
        description=(
            "NIA/Reagan Institute criteria for AD met. "
            "Codes: "
            "1 = High likelihood of dementia being due to Alzheimer's disease. "
            "2 = Intermediate likelihood of dementia being due to Alzheimer's disease. "
            "3 = Low likelihood of dementia being due to Alzheimer's disease. "
            "4=Criteria not met; 5=Not done; 9=Missing/unknown. "
            "If not mentioned → code 9. "
            "Code 9 if NIA/Reagan Institute criteria are not explicitly referenced in the report. "
            "NIA-AA likelihood assessment (low/intermediate/high ADNC) is NOT the same as NIA/Reagan criteria "
            "and must not be used to code this variable."
        ),
    )
    NPCERAD: int = Field(
        description=(
            "CERAD criteria for AD met. "
            "Codes: 1=Definite AD; 2=Probable AD; 3=Possible AD; 4=Criteria not met; "
            "5=Not done; 9=Missing/unknown. "
            "If not mentioned → code 9."
        ),
    )
    NPADRDA: int = Field(
        description=(
            "ADRDA/Khachaturian criteria for AD met. "
            "Codes: 1=Alzheimer's disease; 2=Criteria not met; 3=Not done; 9=Missing/unknown. "
            "If not mentioned → code 9."
        ),
    )
    NPOCRIT: int = Field(
        description=(
            "Other criteria used for diagnosis. "
            "Codes: 1=Alzheimer's disease unspecified; 2=Criteria not met; 3=Not done; "
            "9=Missing/unknown. "
            "If not mentioned → code 2."
        ),
    )
    NPVOTH: int = Field(
        description=(
            "Other vascular diagnosis present. "
            "Codes: 1=Yes; 2=No; 3=Not assessed; 9=Missing/unknown. "
            "If not mentioned → code 2."
        ),
    )
    NPLEWYCS: int = Field(
        description=(
            "DLB clinical syndrome due to DLB pathology. "
            "Codes: 1=Low; 2=Intermediate; 3=High; 6=N/A (not applicable); 9=Missing/unknown. "
            "Code 1-3 if the report explicitly links DLB pathology to a clinical DLB syndrome "
            "with the stated likelihood. Code 6 if DLB pathology present but no clinical syndrome. "
            "If not mentioned → code 6."
        ),
    )

    @field_validator("NPNIT")
    @classmethod
    def val_npnit(cls, v):
        if v not in {1, 2, 3, 4, 5, 9}:
            raise ValueError(f"NPNIT must be 1, 2, 3, 4, 5, or 9, got {v}")
        return v

    @field_validator("NPCERAD")
    @classmethod
    def val_npcerad(cls, v):
        if v not in {1, 2, 3, 4, 5, 9}:
            raise ValueError(f"NPCERAD must be 1, 2, 3, 4, 5, or 9, got {v}")
        return v

    @field_validator("NPADRDA")
    @classmethod
    def val_npadrda(cls, v):
        if v not in {1, 2, 3, 9}:
            raise ValueError(f"NPADRDA must be 1, 2, 3, or 9, got {v}")
        return v

    @field_validator("NPOCRIT", "NPVOTH")
    @classmethod
    def val_present_absent(cls, v):
        if v not in {1, 2, 3, 9}:
            raise ValueError(f"Field must be 1, 2, 3, or 9, got {v}")
        return v

    @field_validator("NPLEWYCS")
    @classmethod
    def val_nplewycs(cls, v):
        if v not in {1, 2, 3, 6, 9}:
            raise ValueError(f"NPLEWYCS must be 1, 2, 3, 6, or 9, got {v}")
        return v


# ---------------------------------------------------------------------------
# Top-level Pass 4 model
# ---------------------------------------------------------------------------

class Pass4Extraction(BaseModel):
    """Pass 4 extraction - 42 variables.

    Covers: normal brain/insufficient AD dx pairs, hippocampal/prion dx pairs,
    other dx pairs with free-text, write-in diagnoses, other disease flags
    (NPPDXA-N, NACCDOWN, NPSCL), and diagnostic criteria.
    """
    model_config = ConfigDict(extra="forbid")

    normal_and_insuff_ad: NormalAndInsuffAD = Field(
        description="NPPNORM, NPCNORM, NPPADP, NPCADP"
    )
    hippocampal_and_prion_dx: HippocampalAndPrionDx = Field(
        description="NPPHIPP, NPCHIPP, NPPPRION, NPCPRION"
    )
    other_dx_pairs: OtherDxPairs = Field(
        description=(
            "NPPOTH1, NPCOTH1, NPOTH1X, "
            "NPPOTH2, NPCOTH2, NPOTH2X, "
            "NPPOTH3, NPCOTH3, NPOTH3X"
        )
    )
    write_in_dx: WriteInDx = Field(
        description="NACCOTHP, NACCWRI1, NACCWRI2, NACCWRI3"
    )
    other_disease_flags: OtherDiseaseFlags = Field(
        description=(
            "NPPDXA, NPPDXB, NPPDXD, NPPDXE, NPPDXF, NPPDXG, NPPDXH, "
            "NPPDXI, NPPDXJ, NPPDXK, NPPDXL, NPPDXM, NPPDXN, NACCDOWN, NPSCL"
        )
    )
    criteria_and_misc: CriteriaAndMiscDx = Field(
        description="NPNIT, NPCERAD, NPADRDA, NPOCRIT, NPVOTH, NPLEWYCS"
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
    def validate_confidence_value(self) -> "Pass4Extraction":
        if self.extraction_confidence and self.extraction_confidence not in {
            "high", "moderate", "low"
        }:
            raise ValueError("extraction_confidence must be 'high', 'moderate', or 'low'")
        return self

    @model_validator(mode="after")
    def enforce_null_policy(self) -> "Pass4Extraction":
        """Enforce extraction-layer null policy.
        Nullable fields:
          NPPNORM, NPCNORM
          NPPADP, NPCADP
          NPPHIPP, NPCHIPP
          NPPPRION, NPCPRION
          NPPOTH1, NPCOTH1
          NPPOTH2, NPCOTH2
          NPPOTH3, NPCOTH3
          NPOTH1X, NPOTH2X, NPOTH3X
          NACCWRI1, NACCWRI2, NACCWRI3
        Other variables must use their explicit NACC-style code.
        """
        nullable = {
            "NPPNORM",
            "NPCNORM",
            "NPPADP",
            "NPCADP",
            "NPPHIPP",
            "NPCHIPP",
            "NPPPRION",
            "NPCPRION",
            "NPPOTH1",
            "NPCOTH1",
            "NPOTH1X",
            "NPPOTH2",
            "NPCOTH2",
            "NPOTH2X",
            "NPPOTH3",
            "NPCOTH3",
            "NPOTH3X",
            "NACCWRI1",
            "NACCWRI2",
            "NACCWRI3",
        }
        sections = [
            self.normal_and_insuff_ad, self.hippocampal_and_prion_dx,
            self.other_dx_pairs, self.write_in_dx,
            self.other_disease_flags, self.criteria_and_misc,
        ]
        for section in sections:
            for field_name in section.model_fields:
                if field_name not in nullable and getattr(section, field_name) is None:
                    raise ValueError(
                        f"{field_name} must not be null - use its explicit "
                        "absence, not-assessed, or unknown code as defined for "
                        "that variable."
                    )
        return self


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_pass4_model() -> Type[BaseModel]:
    """Return the top-level extraction model class."""
    return Pass4Extraction


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


def build_pass4_format_instructions() -> str:
    """Walk the Pydantic schema and emit LLM-facing format instructions."""
    model = Pass4Extraction
    lines = [
        "PASS 4 - Extract ONLY the variables listed below.",
        "Respond with a single JSON object with EXACTLY these top-level keys:",
        "",
        "  normal_and_insuff_ad, hippocampal_and_prion_dx, other_dx_pairs,",
        "  write_in_dx, other_disease_flags, criteria_and_misc,",
        "  field_annotations, extraction_confidence, extraction_notes",
        "",
        "Each top-level key maps to a nested object containing its variables.",
        "NEVER output variables at the root level - they must always be nested.",
        "",
        "NULL POLICY:",
        "  Nullable primary/contributing diagnosis fields:",
        "    NPPNORM, NPCNORM, NPPADP, NPCADP,",
        "    NPPHIPP, NPCHIPP, NPPPRION, NPCPRION,",
        "    NPPOTH1, NPCOTH1, NPPOTH2, NPCOTH2, NPPOTH3, NPCOTH3.",
        "  These fields may be 1, 2, or null.",
        "  null = insufficient information in the report to determine the role.",
        "  Do NOT use 2 merely because the diagnosis is not mentioned.",
        "",
        "  Nullable specify/write-in fields:",
        "    NPOTH1X, NPOTH2X, NPOTH3X,",
        "    NACCWRI1, NACCWRI2, NACCWRI3.",
        "  These remain null when their corresponding parent condition is not met.",
        "",
        "  All other variables must use their explicit numeric/code value; never null.",
        "  IMPORTANT: 'not mentioned' is not automatically equivalent to No.",
        "",
        "PRIMARY/CONTRIBUTING PAIRS:",
        "  Use 1=Yes, 2=No, or null.",
        "  null means the report does not provide enough information to determine",
        "  that specific role.",
        "  Do NOT use 2 merely because the role is not mentioned.",
        "  For each pair, both members may be null when the report provides no",
        "  sufficient information about that diagnosis.",
        "  The only prohibited state is both members being 1.",
        "",
        "PRIMARY/CONTRIBUTING MUTEX FOR OTHER DX POSITIONS:",
        "  Each NPPOTHx/NPCOTHx position represents ONE diagnosis.",
        "  Two distinct diagnoses must use two distinct positions.",
        "  NPPOTH1 and NPCOTH1 cannot both be 1 because they describe",
        "  the primary/contributing role of the SAME diagnosis in NPOTH1X.",
        "  The same rule applies to NPPOTH2/NPCOTH2 and NPPOTH3/NPCOTH3.",
        "  Valid values for each pair field are 1=Yes, 2=No, or null=not determinable.",
        "  Do NOT use 2 merely because a role or diagnosis is not mentioned.",
        "  A pair may therefore be (1,2), (1,null), (2,1), (null,1),",
        "  (2,2), (2,null), (null,2), or (null,null).",
        "  The only prohibited combination is (1,1).",
        "  Code 0 is invalid for all pair fields.",
        "",
        "ORDERING RULE: Fill positions in order of clinical significance.",
        " Position 1 (NPPOTH1/NPCOTH1/NPOTH1X) = most clinically significant other diagnosis.",
        " Position 2 (NPPOTH2/NPCOTH2/NPOTH2X) = second most significant.",
        " Position 3 (NPPOTH3/NPCOTH3/NPOTH3X) = least significant or incidental finding.",
        " Do not reorder diagnoses solely based on whether they are primary or contributing.",
        " Fill positions in order of clinical significance, using the same ordering consistently.",        
        "",
        " NPOTH1X/2X/3X are for diagnoses NOT already captured by the primary/contributing pair fields",
        " Do not use NPOTH slots for: AD/ADNC, Lewy body disease, cerebrovascular disease,",
        " FTLD subtypes, ALS/MND, hippocampal sclerosis, prion disease, or Down syndrome.",
        " Those all have dedicated fields in this or other passes.",
        "",
        "DISEASE FLAGS (NPPDXA through NPPDXN):",
        "  If not mentioned in the report → code 0.",
        "  Use 8 only if the finding was explicitly not assessed.",
        "",
        "WRITE-IN DX (NACCWRI1/2/3): null unless NACCOTHP=1.",
        "",
        "CRITERIA FIELDS (NPNIT, NPCERAD, NPADRDA):",
        "  Use 9 if not mentioned - these are rarely documented explicitly.",
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