"""schema_full.py — Extended NACC Neuropathology extraction schema (40 variables).

SINGLE SOURCE OF TRUTH. main.py imports only get_extraction_model() and
build_format_instructions(). Edit descriptions here to change what the LLM sees.

Variable descriptions follow the NACC Neuropathology Data Dictionary v10+.
Each description embeds:
  - The clinical concept being coded
  - Full code-to-meaning mapping
  - Example report phrases that map to each code
  - Disambiguation notes for common ambiguities
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# FieldAnnotation — per-variable audit trail
# ---------------------------------------------------------------------------


class FieldAnnotation(BaseModel):
    """Extraction audit trail for a single variable."""

    model_config = ConfigDict(extra="forbid")

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Probability that this extracted value is correct. "
            "1.0=exact unambiguous match in report; "
            "0.8=clearly implied with minor paraphrase; "
            "0.6=indirect language or requires inference; "
            "0.4=ambiguous, best guess among plausible codes; "
            "below 0.4: prefer null over a guess."
        ),
    )
    evidence: Optional[str] = Field(
        None,
        description=(
            "Verbatim phrase or sentence from the report that supports this value. "
            "Required for every non-null extraction. "
            "Null only when the field is null because the finding is simply absent "
            "and unambiguously not mentioned."
        ),
    )
    note: Optional[str] = Field(
        None,
        description=(
            "Reasoning note. Write ONLY when extraction was non-trivial: "
            "indirect language, conflicting statements, mapping from narrative to a code, "
            "or when you had to choose between two plausible codes. "
            "Leave null for clear-cut extractions to avoid drowning signal."
        ),
    )


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
    present = 1
    absent = 2
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
    """Basic specimen and case-level details."""

    model_config = ConfigDict(extra="forbid")

    NPSEX: Optional[int] = Field(
        None,
        description=(
            "Subject biological sex recorded at autopsy. "
            "Codes: 1=Male, 2=Female. "
            "Look in the demographics or header section. "
            "Do not infer from pronouns alone — use null if not stated."
        ),
    )
    NPFIX: Optional[int] = Field(
        None,
        description=(
            "Fixative used for brain preservation before sectioning. "
            "Codes: 1=Formalin (most common; 'fixed in 10% formalin', 'formalin-fixed'), "
            "2=Paraformaldehyde ('PFA', '4% paraformaldehyde'), "
            "7=Other (must also populate NPFIXX with the fixative name). "
            "Look for phrases like 'brain was fixed in...', 'fixed overnight in...'. "
            "-4=N/A if fixation is not applicable."
        ),
    )
    NPWBRWT: Optional[float] = Field(
        None,
        description=(
            "Whole brain weight in grams, measured before or after fixation. "
            "Valid range: 100–2500 g. Special code: 9999=weight unknown. "
            "Extract the numeric value only — strip units (e.g., '1250 g' → 1250). "
            "If weight is reported as a range, use the midpoint. "
            "Look for 'brain weight', 'cerebral weight', 'combined weight', 'weighs X grams'. "
            "If only cerebellum or brainstem weight is given separately, do not add them "
            "unless the report explicitly states 'total brain weight'. "
            "Normal adult range: 1100–1400 g (male), 1000–1300 g (female)."
        ),
    )
    NPWBRF: Optional[int] = Field(
        None,
        description=(
            "Whether the recorded brain weight was measured fresh (before fixation) "
            "or after fixation. Fixation typically adds 5–10% weight. "
            "Codes: 1=Fresh (weighed at autopsy before fixation), "
            "2=Fixed (weighed after fixation period), "
            "8=Not assessed / not specified in report. "
            "Look for 'fresh weight', 'fixed weight', or context clues like 'weighed at autopsy'."
        ),
    )
    NPPMIH: Optional[float] = Field(
        None,
        description=(
            "Postmortem interval (PMI): time from death to brain fixation or freezing, in hours. "
            "Valid range: 0.0–98.9 hours; 99.9=unknown. "
            "Look for 'postmortem interval', 'PMI', 'time of death to autopsy'. "
            "Convert days to hours if needed (e.g., '2 days' → 48.0). "
            "If a range is given, use the midpoint."
        ),
    )
    NPFIXX: Optional[str] = Field(
        None,
        description=(
            "Free-text name of the fixative when NPFIX=7 (Other). "
            "Only populate if NPFIX was coded 7. "
            "Examples: 'glutaraldehyde', 'Bouin's solution', 'methanol'."
        ),
    )

    @field_validator("NPSEX")
    @classmethod
    def val_sex(cls, v):
        if v is not None and v not in {1, 2}:
            raise ValueError(f"NPSEX must be 1 (Male) or 2 (Female), got {v}")
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
            raise ValueError(f"NPWBRWT must be 100–2500 or 9999, got {v}")
        return v


class GrossFindings(BaseModel):
    """Macroscopic (gross) examination findings observed at autopsy."""

    model_config = ConfigDict(extra="forbid")

    NPGRLA: Optional[int] = Field(
        None,
        description=(
            "Presence of focal lobar atrophy on gross examination (any lobe). "
            "Codes: 0=None (no focal atrophy identified), 1=Yes (focal lobar atrophy present), "
            "8=Not assessed, 9=Unknown. "
            "Focal lobar atrophy means one or more lobes are selectively and disproportionately "
            "shrunken compared to the rest of the cortex. "
            "Look for: 'focal frontal atrophy', 'frontotemporal atrophy', 'parietal atrophy', "
            "'knife-edge gyri', 'asymmetric cortical atrophy'. "
            "DISTINGUISH from diffuse cortical atrophy (see NPGRCCA): NPGRLA=1 requires "
            "a specific lobe to be called out as preferentially affected. "
            "Global cortical thinning without focal emphasis → NPGRLA=0."
        ),
    )
    NPGRHA: Optional[SeverityCode] = Field(
        None,
        description=(
            "Severity of hippocampal atrophy on gross examination. "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "The hippocampus is assessed bilaterally on the medial temporal surface. "
            "Look for: 'hippocampal atrophy', 'hippocampal shrinkage', 'medial temporal atrophy', "
            "'small hippocampi', 'parahippocampal atrophy'. "
            "Grading guidance: Mild=subtle volume reduction compared to expected; "
            "Moderate=clearly shrunken, firm; Severe=markedly shrunken, leather-like. "
            "If only one side is affected but severity stated, code the severity as described. "
            "If the report says 'mild to moderate', code 2 (Moderate)."
        ),
    )
    NPGRSNH: Optional[SeverityCode] = Field(
        None,
        description=(
            "Severity of substantia nigra (SN) hypopigmentation on gross examination. "
            "Codes: 0=None (normal dark pigmentation), 1=Mild, 2=Moderate, 3=Severe (near-complete pallor), "
            "8=Not assessed, 9=Unknown. "
            "The SN is normally a darkly pigmented band in the midbrain tegmentum. "
            "Hypopigmentation (pallor, depigmentation) is the gross hallmark of Parkinson-spectrum disease. "
            "Look for: 'pallor of substantia nigra', 'depigmentation of SN', 'hypopigmented SN', "
            "'loss of pigmentation in the substantia nigra', 'pale substantia nigra'. "
            "Grading guidance: Mild=slightly pale compared to expected; "
            "Moderate=clearly paler than normal; Severe=almost completely depigmented/white. "
            "If the report only notes SN pigmentation is 'normal' or 'intact' → code 0."
        ),
    )
    NPGRLCH: Optional[SeverityCode] = Field(
        None,
        description=(
            "Severity of locus coeruleus (LC) hypopigmentation on gross examination. "
            "Codes: 0=None (normal blue-gray pigmentation), 1=Mild, 2=Moderate, 3=Severe, "
            "8=Not assessed, 9=Unknown. "
            "The LC is a pigmented nucleus in the dorsal pons; normally blue-gray ('coeruleus'=blue). "
            "Look for: 'locus coeruleus pallor', 'depigmentation of locus coeruleus', "
            "'hypopigmented locus ceruleus', 'pale LC'. "
            "Often assessed together with SN; both can be affected in Parkinson-spectrum disease. "
            "If the LC is not mentioned at all in the report → null (do not assume normal). "
            "If report says LC is 'normally pigmented' or 'intact' → code 0."
        ),
    )
    NPGRCCA: Optional[SeverityCode] = Field(
        None,
        description=(
            "Severity of DIFFUSE cerebral cortical atrophy on gross examination. "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "This captures global/bilateral cortical volume loss, not focal lobar atrophy (see NPGRLA). "
            "Look for: 'cortical atrophy', 'cerebral atrophy', 'widened sulci', "
            "'thinned gyri', 'reduced brain volume', 'generalized atrophy'. "
            "Grading: Mild=slight sulcal widening; Moderate=clearly widened sulci with thinned gyri; "
            "Severe=prominent sulcal widening, markedly thinned gyri, significantly reduced volume. "
            "If report describes 'mild diffuse atrophy' → code 1."
        ),
    )
    NACCBRNN: Optional[int] = Field(
        None,
        description=(
            "Derived flag: NO significant neuropathological changes identified. "
            "Codes: 0=Neuropathological changes ARE present (normal = no), "
            "1=No significant changes found (essentially normal brain), "
            "8=Not assessed or missing. "
            "Code 1 only when the report concludes the examination is essentially unremarkable "
            "or 'within normal limits for age'. "
            "If ANY pathological finding is present elsewhere in this extraction, code 0."
        ),
    )

    @field_validator("NPGRLA")
    @classmethod
    def val_grla(cls, v):
        if v is not None and v not in {0, 1, 8, 9, -4}:
            raise ValueError(f"NPGRLA must be 0, 1, 8, 9, or -4, got {v}")
        return v

    @field_validator("NACCBRNN")
    @classmethod
    def val_brnn(cls, v):
        if v is not None and v not in {0, 1, 8}:
            raise ValueError(f"NACCBRNN must be 0, 1, or 8, got {v}")
        return v


class VascularPathology(BaseModel):
    """Vascular and ischemic neuropathological findings."""

    model_config = ConfigDict(extra="forbid")

    NACCAVAS: Optional[SeverityCode] = Field(
        None,
        description=(
            "Severity of atherosclerosis of the circle of Willis and major cerebral arteries. "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Atherosclerosis = intimal plaque formation in large vessels (circle of Willis, "
            "basilar artery, MCA, ACA, PCA). Distinct from arteriolosclerosis (NACCARTE). "
            "Look for: 'atherosclerosis', 'arteriosclerosis', 'calcified plaques', "
            "'atheromatous plaques', 'stenosis' of named cerebral arteries. "
            "Grading: Mild=few scattered plaques, no luminal narrowing; "
            "Moderate=plaques with up to ~50% stenosis; "
            "Severe=extensive plaques, >50% stenosis or near-occlusion. "
            "If report says 'mild to moderate' → code 2. 'Moderate to severe' → code 3. "
            "If arteries are described as 'patent' or 'without significant atherosclerosis' → code 0."
        ),
    )
    NPLINF: Optional[PresentAbsentCode] = Field(
        None,
        description=(
            "Large arterial cortical infarcts (territorial infarcts) present. "
            "Codes: 1=Present, 2=Absent, 3=Not assessed, 9=Unknown. "
            "Large vessel (territorial) infarcts involve a cortical territory supplied by "
            "a named artery (MCA, ACA, PCA territory). Typically wedge-shaped, cortical+subcortical. "
            "Look for: 'cortical infarct', 'territorial infarct', 'large vessel infarct', "
            "'MCA territory infarct', 'wedge-shaped infarct', 'old infarct' in a cortical distribution. "
            "DISTINGUISH from lacunar infarcts (NPLAC): lacunes are small (<1.5 cm), deep, "
            "in basal ganglia/thalamus/pons/white matter. "
            "If no infarcts are mentioned and the report is otherwise complete → code 2 (Absent). "
            "Do not code 2 if infarcts were not assessed."
        ),
    )
    NPLAC: Optional[PresentAbsentCode] = Field(
        None,
        description=(
            "Lacunar infarcts (small vessel infarcts, <1.5 cm) present. "
            "Codes: 1=Present, 2=Absent, 3=Not assessed, 9=Unknown. "
            "Lacunes are small cavitated lesions from occlusion of penetrating arteries, "
            "typically in basal ganglia, thalamus, internal capsule, pons, or deep white matter. "
            "Look for: 'lacunar infarct', 'lacune', 'lacunae', 'small vessel infarct', "
            "'état lacunaire', 'cribriform change' (when referring to actual infarcts, not perivascular spaces). "
            "DISTINGUISH from dilated perivascular spaces (Virchow-Robin spaces): "
            "perivascular spaces are NOT infarcts — do not code those as lacunes. "
            "If no lacunes mentioned and report is complete → code 2 (Absent)."
        ),
    )
    NPHEM: Optional[PresentAbsentCode] = Field(
        None,
        description=(
            "Hemorrhage(s) present — includes microbleeds, petechiae, macrohemorrhage. "
            "Codes: 1=Present, 2=Absent, 3=Not assessed, 9=Unknown. "
            "Encompasses any hemorrhagic lesion: lobar hemorrhage, basal ganglia hemorrhage, "
            "cerebellar hemorrhage, pontine hemorrhage, cortical microbleeds, "
            "subarachnoid hemorrhage, subdural hematoma (if noted as neuropathological finding). "
            "Look for: 'hemorrhage', 'hematoma', 'hemosiderin deposits', 'microhemorrhage', "
            "'microbleed', 'petechiae', 'blood breakdown products', 'hemosiderin-laden macrophages'. "
            "If no hemorrhage mentioned and report is otherwise thorough → code 2 (Absent). "
            "Code 1 if ANY hemorrhagic lesion is noted, regardless of age or size."
        ),
    )
    NPWMR: Optional[SeverityCode] = Field(
        None,
        description=(
            "Severity of white matter rarefaction (myelin loss / leukoencephalopathy). "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "White matter rarefaction = pallor and loss of myelin with sparing of axons "
            "(unlike infarction), often periventricular. Associated with chronic ischemia. "
            "Look for: 'white matter rarefaction', 'white matter pallor', 'leukoaraiosis', "
            "'periventricular white matter changes', 'subcortical white matter disease', "
            "'myelin loss', 'white matter gliosis'. "
            "Grading: Mild=patchy or mild periventricular pallor; "
            "Moderate=confluent periventricular or multifocal changes; "
            "Severe=extensive, confluent throughout white matter. "
            "DISTINGUISH from infarcts: rarefaction is a diffuse process, not focal cavitation."
        ),
    )
    NACCARTE: Optional[SeverityCode] = Field(
        None,
        description=(
            "Severity of arteriolosclerosis (small vessel wall thickening). "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Arteriolosclerosis = concentric wall thickening, hyalinization, and luminal narrowing "
            "of small penetrating arteries and arterioles in basal ganglia, white matter, brainstem. "
            "DISTINCT from atherosclerosis (NACCAVAS, which affects large named arteries). "
            "Look for: 'arteriolosclerosis', 'hyaline arteriolosclerosis', 'arteriolar thickening', "
            "'small vessel disease', 'arteriolar hyalinization', 'concentric wall thickening'. "
            "Grading mirrors NACCAVAS: Mild=occasional thickened arterioles; "
            "Moderate=widespread thickening; Severe=extensive with near-obliteration of lumina."
        ),
    )
    NACCVASC: Optional[int] = Field(
        None,
        description=(
            "Derived summary: any vascular pathology present (yes/no). "
            "Codes: 0=No vascular pathology, 1=Vascular pathology present, 9=Unknown. "
            "Code 1 if ANY of the following are present: NACCAVAS≥1, NPLINF=1, NPLAC=1, "
            "NPHEM=1, NPWMR≥1, NACCARTE≥1. "
            "Code 0 only if all vascular fields are 0 or absent. "
            "This is a derived field — derive it from the other vascular findings."
        ),
    )
    NACCINF: Optional[int] = Field(
        None,
        description=(
            "Derived summary: any infarct or lacune present. "
            "Codes: 0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "Code 1 if NPLINF=1 (large infarct) OR NPLAC=1 (lacune). "
            "Code 0 if both NPLINF=2 and NPLAC=2 (both absent). "
            "This is a derived field."
        ),
    )
    NACCHEM: Optional[int] = Field(
        None,
        description=(
            "Derived summary: any hemorrhage or microbleed present. "
            "Codes: 0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "Code 1 if NPHEM=1. Code 0 if NPHEM=2. "
            "This is a derived field."
        ),
    )

    @field_validator("NACCVASC", "NACCINF", "NACCHEM")
    @classmethod
    def val_derived(cls, v):
        if v is not None and v not in {0, 1, 8, 9}:
            raise ValueError(f"Derived vascular field must be 0, 1, 8, or 9, got {v}")
        return v


class MicroscopicFindings(BaseModel):
    """Microscopic and immunohistochemical cellular findings."""

    model_config = ConfigDict(extra="forbid")

    NPNLOSS: Optional[SeverityCode] = Field(
        None,
        description=(
            "Severity of neuronal loss specifically in the substantia nigra (pars compacta). "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Neuronal loss in the SN is the hallmark of Parkinson's disease and related synucleinopathies. "
            "Look for: 'neuronal loss in substantia nigra', 'depopulation of SN neurons', "
            "'loss of dopaminergic neurons', 'reduced neuronal density in SN pars compacta', "
            "'neuronal depletion substantia nigra'. "
            "Note: this field is SPECIFICALLY for SN neuronal loss, not global or cortical neuronal loss. "
            "Grading: Mild=<25% reduction; Moderate=25–75% reduction; Severe=>75% reduction "
            "relative to expected neuronal density. "
            "Accompanying reactive gliosis is expected and does not change the code."
        ),
    )
    NPHIPSCL: Optional[int] = Field(
        None,
        description=(
            "Hippocampal sclerosis: neuronal loss and gliosis in CA1 and/or subiculum sectors. "
            "Codes: 0=None, 1=Unilateral, 2=Bilateral, 3=Present but laterality not specified (NOS), "
            "8=Not assessed, 9=Unknown. "
            "Hippocampal sclerosis (HS) is characterized by severe CA1 pyramidal cell loss "
            "with reactive astrogliosis, often with relative sparing of CA2. "
            "Common in TDP-43 pathology, epilepsy, and aging-related HS (ARTAG). "
            "Look for: 'hippocampal sclerosis', 'CA1 neuronal loss', 'hippocampal gliosis', "
            "'sclerosis of the hippocampus', 'HS-Aging'. "
            "If unilateral, code 1. If bilateral, code 2. If present but side not specified, code 3. "
            "Do not confuse with hippocampal atrophy (NPGRHA) — atrophy is a gross finding; "
            "hippocampal sclerosis is a microscopic finding."
        ),
    )
    NACCLEWY: Optional[int] = Field(
        None,
        description=(
            "Derived Lewy body distribution pattern summary. "
            "Codes: 0=No Lewy bodies, 1=Brainstem-predominant, 2=Limbic/transitional, "
            "3=Neocortical/diffuse, 4=Lewy bodies present but pattern unspecified, "
            "8=Not assessed, 9=Unknown. "
            "This is a derived field — derive it from NPLBOD (the raw observation). "
            "Brainstem-predominant=Lewy bodies confined to brainstem nuclei (SN, LC, dorsal vagal). "
            "Limbic=brainstem + limbic structures (amygdala, cingulate, entorhinal). "
            "Neocortical=widespread including neocortex."
        ),
    )
    NPLBOD: Optional[LewyBodyPattern] = Field(
        None,
        description=(
            "Lewy body pathology distribution pattern (raw observation). "
            "Codes: 0=No Lewy bodies identified; "
            "1=Brainstem-predominant (SN, LC, dorsal vagal nucleus, raphe — no limbic/cortical); "
            "2=Limbic/transitional (brainstem + amygdala, entorhinal, cingulate — some cortex); "
            "3=Neocortical/diffuse (widespread, including frontal/temporal/parietal neocortex); "
            "4=Amygdala-predominant (heavy amygdala involvement without proportionate brainstem); "
            "5=Olfactory bulb only; 8=Not assessed, 9=Unknown. "
            "Look for: 'Lewy bodies', 'Lewy neurites', 'alpha-synuclein inclusions', "
            "'synuclein IHC positive', 'Lewy body disease'. "
            "If the report gives a named pattern (e.g., 'neocortical Lewy body disease', "
            "'diffuse Lewy body disease') → map to the appropriate code. "
            "If Lewy bodies are present but distribution is not characterized → code 4 (Unspecified). "
            "NPLBOD is the raw observation; NACCLEWY is derived from it — keep them consistent."
        ),
    )


class ADPathology(BaseModel):
    """Alzheimer's disease ABC neuropathological scores (NIA-AA 2012 criteria)."""

    model_config = ConfigDict(extra="forbid")

    NPTHAL: Optional[ThalPhase] = Field(
        None,
        description=(
            "Thal phase of amyloid-beta (Aβ) plaque deposition — the 'A' score in ABC. "
            "Codes: 0=No Aβ plaques; "
            "1=Phase 1 (neocortex only — frontal, temporal, parietal, occipital); "
            "2=Phase 2 (neocortex + allocortex: entorhinal, hippocampus, cingulate, insula); "
            "3=Phase 3 (Phase 2 + subcortical: basal ganglia, cholinergic nuclei, thalamus); "
            "4=Phase 4 (Phase 3 + brainstem: SN, LC, raphe); "
            "5=Phase 5 (Phase 4 + cerebellum); "
            "8=Not assessed, 9=Unknown. "
            "Look for explicitly stated Thal phase ('Thal phase 3', 'A2', 'amyloid phase 2'). "
            "If not stated directly, infer from the distribution of amyloid plaques described. "
            "Note: Thal phase is based on amyloid/Aβ PLAQUES, not CAA (which is NACCAMY)."
        ),
    )
    NACCBRAA: Optional[BraakStage] = Field(
        None,
        description=(
            "Braak neurofibrillary tangle (NFT) stage — the 'B' score in ABC. "
            "Codes: 0=No NFTs; "
            "1=Stage I (NFTs in transentorhinal region / layer Pre-alpha of entorhinal cortex); "
            "2=Stage II (NFTs in entorhinal cortex + hippocampus CA1); "
            "3=Stage III (NFTs spread to association neocortex: temporal pole, insula, cingulate); "
            "4=Stage IV (NFTs throughout temporal, frontal, parietal association cortex); "
            "5=Stage V (NFTs in all association areas including prefrontal); "
            "6=Stage VI (NFTs in primary cortices: motor, sensory, visual); "
            "7=Other tauopathy (non-AD tau, e.g., PSP, CBD, Pick's — use when tangle pattern "
            "does not fit Braak staging for AD); "
            "8=Not assessed, 9=Unknown. "
            "Look for: 'Braak stage X', 'neurofibrillary stage', 'Braak and Braak stage X', "
            "or description of NFT distribution mapped to stages above. "
            "If the report gives a named AD stage ('Braak IV') → map directly."
        ),
    )
    NACCNEUR: Optional[CERADScore] = Field(
        None,
        description=(
            "CERAD score for neuritic (senile) plaques — the 'C' score in ABC. "
            "Neuritic plaques = amyloid cores with surrounding dystrophic neurites (tau-positive). "
            "Codes: 0=No neuritic plaques; "
            "1=Sparse (occasional neuritic plaques in neocortex); "
            "2=Moderate (easily found but not numerous); "
            "3=Frequent (numerous in multiple neocortical fields); "
            "8=Not assessed, 9=Unknown. "
            "Look for: 'CERAD score', 'neuritic plaques', 'senile plaques', 'NP score'. "
            "DISTINGUISH from diffuse plaques (NACCDIFF): neuritic plaques have a dense amyloid "
            "core and surrounding tau-positive neurites; diffuse plaques are pre-amyloid deposits "
            "without neuritic change. "
            "Pathologist language: 'sparse neuritic plaques' → 1; 'moderate' → 2; 'frequent/numerous' → 3."
        ),
    )
    NPADNC: Optional[ADNCScore] = Field(
        None,
        description=(
            "NIA-AA overall Alzheimer's Disease Neuropathological Change (ADNC) score — "
            "the final ABC composite. "
            "Codes: 0=Not AD / no/minimal ADNC; "
            "1=Low ADNC (Thal 1–2, Braak I–II, CERAD 0–1 — age-related changes only); "
            "2=Intermediate ADNC (Thal 3, Braak III–IV, CERAD 2 — some but not full AD); "
            "3=High ADNC (Thal 4–5, Braak V–VI, CERAD 3 — consistent with full AD diagnosis); "
            "8=Not assessed, 9=Unknown. "
            "Look for: 'ADNC', 'ABC score', 'low/intermediate/high AD neuropathological change', "
            "'meets criteria for AD', 'NIA-AA criteria'. "
            "If the report gives the ABC scores separately, derive NPADNC from them: "
            "all three must be high for NPADNC=3; if any one is low, the composite is lower."
        ),
    )
    NACCDIFF: Optional[CERADScore] = Field(
        None,
        description=(
            "CERAD score for diffuse (pre-amyloid) plaques. "
            "Diffuse plaques = Aβ deposits without dense core and without neuritic change (tau-negative). "
            "Codes: 0=None, 1=Sparse, 2=Moderate, 3=Frequent, 8=Not assessed, 9=Unknown. "
            "Look for: 'diffuse plaques', 'diffuse amyloid deposits', 'pre-amyloid plaques'. "
            "Often reported alongside neuritic plaques ('sparse neuritic and moderate diffuse plaques'). "
            "Diffuse plaques alone (without neuritic plaques) are less pathologically significant. "
            "Use same CERAD grading as NACCNEUR: sparse/moderate/frequent."
        ),
    )
    NACCAMY: Optional[SeverityCode] = Field(
        None,
        description=(
            "Severity of cerebral amyloid angiopathy (CAA) — amyloid deposition in vessel walls. "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "CAA = Aβ deposition in walls of leptomeningeal and cortical arteries/arterioles. "
            "DISTINCT from parenchymal amyloid plaques (NPTHAL). "
            "Look for: 'cerebral amyloid angiopathy', 'CAA', 'amyloid in vessel walls', "
            "'congophilic angiopathy', 'vascular amyloid'. "
            "Grading: Mild=occasional vessel wall deposits; "
            "Moderate=widespread but not all vessels affected; "
            "Severe=extensive, many vessels affected, may have associated microinfarcts or hemorrhage. "
            "If CAA is explicitly graded in the report, use that grade."
        ),
    )


class DiagnosticCodes(BaseModel):
    """Primary and contributing neuropathological final diagnoses."""

    model_config = ConfigDict(extra="forbid")

    NACCCBD: Optional[YesNoCode] = Field(
        None,
        description=(
            "Corticobasal degeneration (CBD) present as a neuropathological diagnosis. "
            "Codes: 0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "CBD is a 4-repeat tauopathy with astrocytic plaques, ballooned neurons, "
            "and tau inclusions in cortex and basal ganglia. "
            "Look for: 'corticobasal degeneration', 'CBD', 'corticobasal syndrome with pathology', "
            "'4-repeat tauopathy consistent with CBD', 'astrocytic plaques'. "
            "Note: CBS (clinical syndrome) ≠ CBD (pathological diagnosis); "
            "code 1 only when the PATHOLOGICAL diagnosis is CBD. "
            "CBD often co-occurs with AD pathology — code 1 regardless of co-pathology."
        ),
    )
    NPPVASC: Optional[int] = Field(
        None,
        description=(
            "Vascular disease listed as the PRIMARY (principal) neuropathological diagnosis. "
            "Codes: 1=Yes (vascular disease is the primary diagnosis), "
            "2=No (vascular disease is absent or only contributing). "
            "Code 1 ONLY if the report explicitly names vascular disease, stroke, "
            "or vascular dementia as the PRIMARY or PRINCIPAL cause of pathology. "
            "If vascular changes are present but secondary to another disease (e.g., AD with CAA) → code 2. "
            "Look for: 'primary diagnosis: vascular dementia', 'cerebrovascular disease as principal finding', "
            "'vascular disease accounting for cognitive decline'. "
            "DISTINGUISH from NPCVASC (contributing diagnosis)."
        ),
    )
    NPPAD: Optional[int] = Field(
        None,
        description=(
            "Alzheimer's disease listed as the PRIMARY neuropathological diagnosis. "
            "Codes: 1=Yes, 2=No. "
            "Code 1 if the report identifies AD (high ADNC, NPADNC=3) as the primary/principal finding. "
            "If AD is a contributing co-pathology only → code 2 (and set NPCAD=1 instead). "
            "Look for: 'primary diagnosis: Alzheimer's disease', 'Alzheimer's disease, NIA-AA high'."
        ),
    )
    NPCAD: Optional[int] = Field(
        None,
        description=(
            "Alzheimer's disease listed as a CONTRIBUTING (secondary) neuropathological diagnosis. "
            "Codes: 1=Yes, 2=No. "
            "Code 1 if AD pathology is present but is NOT the primary diagnosis "
            "(i.e., another disease is primary). Common in mixed dementia cases. "
            "If AD is the primary diagnosis → NPPAD=1 and NPCAD=2."
        ),
    )
    NPPLEWY: Optional[int] = Field(
        None,
        description=(
            "Lewy body disease listed as the PRIMARY neuropathological diagnosis. "
            "Codes: 1=Yes, 2=No. "
            "Encompasses PD, DLB, and PD with dementia when Lewy body pathology "
            "is the principal finding. "
            "Code 1 if the report identifies Lewy body disease / DLB / PD as primary. "
            "If Lewy bodies are incidental or contributing only → code 2 (set NPCLEWY=1)."
        ),
    )
    NPCLEWY: Optional[int] = Field(
        None,
        description=(
            "Lewy body disease listed as a CONTRIBUTING (secondary) neuropathological diagnosis. "
            "Codes: 1=Yes, 2=No. "
            "Code 1 when Lewy body pathology is present but is not the primary diagnosis. "
            "Common in mixed AD+Lewy body cases. "
            "If Lewy body disease is the primary diagnosis → NPPLEWY=1 and NPCLEWY=2."
        ),
    )
    NPCVASC: Optional[int] = Field(
        None,
        description=(
            "Vascular disease listed as a CONTRIBUTING (secondary) neuropathological diagnosis. "
            "Codes: 1=Yes, 2=No. "
            "Code 1 when vascular pathology contributes to the overall pathological picture "
            "but is NOT the primary diagnosis (see NPPVASC for primary). "
            "Common in mixed dementia (e.g., AD + vascular disease). "
            "Look for: 'contributing vascular disease', 'mixed AD and vascular pathology', "
            "'cerebrovascular disease as contributing factor'."
        ),
    )
    NPPFTLD: Optional[int] = Field(
        None,
        description=(
            "Frontotemporal lobar degeneration (FTLD) listed as the PRIMARY diagnosis. "
            "Codes: 1=Yes, 2=No. "
            "FTLD encompasses FTLD-tau (Pick's, PSP, CBD, AGD) and FTLD-TDP subtypes. "
            "Code 1 if any FTLD subtype is listed as the principal neuropathological diagnosis. "
            "Look for: 'FTLD', 'frontotemporal lobar degeneration', as primary diagnosis label."
        ),
    )
    NACCPROG: Optional[YesNoCode] = Field(
        None,
        description=(
            "Progressive supranuclear palsy (PSP) present as a neuropathological diagnosis. "
            "Codes: 0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "PSP = 4-repeat tauopathy with tufted astrocytes, coiled bodies, globose NFTs "
            "predominantly in basal ganglia, subthalamic nucleus, SN, brainstem. "
            "Look for: 'progressive supranuclear palsy', 'PSP', 'PSP-Richardson', "
            "'tufted astrocytes', 'globose tangles in STN'. "
            "Code 1 regardless of whether PSP is primary or contributing."
        ),
    )
    NACCPICK: Optional[YesNoCode] = Field(
        None,
        description=(
            "Pick's disease (PiD) present as a neuropathological diagnosis. "
            "Codes: 0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "Pick's disease = 3-repeat tauopathy with Pick bodies (round, tau-positive inclusions) "
            "and ballooned neurons, with frontotemporal predilection. "
            "Look for: 'Pick disease', 'Pick bodies', 'PiD', '3-repeat tauopathy consistent with PiD'. "
            "Code 1 regardless of whether it is primary or contributing."
        ),
    )
    NPFTDTDP: Optional[YesNoCode] = Field(
        None,
        description=(
            "FTLD-TDP (TDP-43 proteinopathy) present as a neuropathological diagnosis. "
            "Codes: 0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "TDP-43 pathology includes cytoplasmic inclusions, neurites, and intranuclear inclusions "
            "staining with TDP-43 IHC; subtypes A–E (Mackenzie classification). "
            "Look for: 'FTLD-TDP', 'TDP-43 pathology', 'TDP-43 inclusions', "
            "'TDP-43 immunoreactive inclusions', 'ALS-TDP'. "
            "Code 1 if TDP-43 pathology is present in any distribution or subtype. "
            "Note: hippocampal sclerosis in the elderly often has associated TDP-43 pathology."
        ),
    )
    NACCPRIO: Optional[YesNoCode] = Field(
        None,
        description=(
            "Prion disease present as a neuropathological diagnosis. "
            "Codes: 0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "Prion diseases include CJD (sporadic, familial, iatrogenic, variant), "
            "GSS, fatal insomnia, and kuru. "
            "Neuropathological hallmarks: spongiform change (vacuolation), neuronal loss, "
            "gliosis, PrP immunoreactivity, amyloid plaques (kuru-type or florid). "
            "Look for: 'Creutzfeldt-Jakob disease', 'CJD', 'prion disease', "
            "'spongiform encephalopathy', 'PrP-positive', 'spongiform change consistent with prion'. "
            "Code 1 if prion disease is diagnosed (confirmed or suspected on pathology)."
        ),
    )

    @field_validator(
        "NPPVASC", "NPPAD", "NPCAD", "NPPLEWY", "NPCLEWY", "NPCVASC", "NPPFTLD"
    )
    @classmethod
    def val_binary_diag(cls, v):
        if v is not None and v not in {1, 2}:
            raise ValueError(
                f"Primary/contributing diagnosis field must be 1 or 2, got {v}"
            )
        return v


# ---------------------------------------------------------------------------
# Top-level extraction model
# ---------------------------------------------------------------------------


class NeuropathologyExtraction(BaseModel):
    """Structured NACC neuropathology extraction — 40 variables + per-field annotations.

    Priority fields (extract with highest care):
      NPSEX, NPFIX, NPWBRWT, NPGRLA, NPGRHA, NPGRSNH, NPGRLCH,
      NACCAVAS, NPLINF, NPLAC, NPHEM, NPWMR, NPNLOSS, NACCCBD, NPPVASC
    """

    model_config = ConfigDict(extra="forbid")

    specimen_info: SpecimenInfo = Field(
        default_factory=SpecimenInfo,
        description="Specimen details — PRIORITY: NPSEX, NPFIX, NPWBRWT",
    )
    gross_findings: GrossFindings = Field(
        default_factory=GrossFindings,
        description="Gross findings — PRIORITY: NPGRLA, NPGRHA, NPGRSNH, NPGRLCH",
    )
    vascular_pathology: VascularPathology = Field(
        default_factory=VascularPathology,
        description="Vascular pathology — PRIORITY: NACCAVAS, NPLINF, NPLAC, NPHEM, NPWMR",
    )
    microscopic_findings: MicroscopicFindings = Field(
        default_factory=MicroscopicFindings,
        description="Microscopic findings — PRIORITY: NPNLOSS",
    )
    ad_pathology: ADPathology = Field(
        default_factory=ADPathology,
        description="AD ABC scores: Thal phase / Braak stage / CERAD / ADNC",
    )
    diagnostic_codes: DiagnosticCodes = Field(
        default_factory=DiagnosticCodes,
        description="Final diagnoses — PRIORITY: NACCCBD, NPPVASC",
    )

    field_annotations: Optional[Dict[str, FieldAnnotation]] = Field(
        None,
        description=(
            "Per-variable extraction audit, keyed by variable name (e.g. 'NPGRHA', 'NACCAVAS'). "
            "Provide an entry for EVERY non-null extracted variable. "
            "For null variables, add an entry ONLY when the absence itself was ambiguous "
            "(e.g. a finding is mentioned but cannot be coded). "
            "Omit entries for null variables that are simply absent from the report. "
            "Each entry is an object with three keys: "
            "(1) 'confidence': float 0.0–1.0 — certainty this value is correct "
            "(1.0=exact match; 0.8=clearly implied; 0.6=indirect inference; 0.4=ambiguous guess; "
            "below 0.4: prefer null instead of guessing); "
            "(2) 'evidence': string — verbatim phrase or sentence from the report that drove the decision "
            "(required for every non-null extraction; null only when a finding is unambiguously absent); "
            "(3) 'note': string or null — brief reasoning note ONLY when extraction was non-trivial "
            "(ambiguous language, had to choose between codes, conflicting statements); "
            "leave null for straightforward extractions. "
            'Example: {"NPGRHA": {"confidence": 0.9, "evidence": "moderate hippocampal atrophy bilaterally", "note": null}}'
        ),
    )
    extraction_confidence: Optional[str] = Field(
        None,
        description="Overall case-level confidence in this extraction: high, moderate, or low.",
    )
    extraction_notes: Optional[str] = Field(
        None,
        description=(
            "Case-level caveats not captured in field_annotations — e.g. illegible sections, "
            "report refers to addenda not provided, or systemic ambiguity affecting many variables."
        ),
    )

    @model_validator(mode="after")
    def confidence_is_valid(self) -> "NeuropathologyExtraction":
        if self.extraction_confidence and self.extraction_confidence not in {
            "high",
            "moderate",
            "low",
        }:
            raise ValueError(
                "extraction_confidence must be 'high', 'moderate', or 'low'"
            )
        return self

    @model_validator(mode="after")
    def validate_derived_consistency(self) -> "NeuropathologyExtraction":
        """Soft-check derived fields are consistent with their source fields."""
        vp = self.vascular_pathology
        # NACCINF should be 1 if NPLINF=1 or NPLAC=1
        if vp.NACCINF == 0 and (
            vp.NPLINF and vp.NPLINF.value == 1 or vp.NPLAC and vp.NPLAC.value == 1
        ):
            raise ValueError(
                "NACCINF=0 (no infarcts) but NPLINF or NPLAC is 1 (present) — inconsistent."
            )
        # NACCHEM should be 1 if NPHEM=1
        if vp.NACCHEM == 0 and vp.NPHEM and vp.NPHEM.value == 1:
            raise ValueError(
                "NACCHEM=0 (no hemorrhage) but NPHEM=1 (present) — inconsistent."
            )
        return self


# ---------------------------------------------------------------------------
# Public API — main.py imports only these two functions
# ---------------------------------------------------------------------------


def get_extraction_model() -> Type[BaseModel]:
    """Return the top-level extraction model class."""
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
    """Recursively describe a Pydantic field for the LLM prompt."""
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
    """Walk the Pydantic schema and emit LLM-facing format instructions."""
    if model is None:
        model = get_extraction_model()

    lines = [
        "Respond with a single JSON object conforming to this schema:",
        "",
        f"Root model: {model.__name__}",
        "",
        "PRIORITY fields — extract these with the highest care:",
        "  NPSEX, NPFIX, NPWBRWT, NPGRLA, NPGRHA, NPGRSNH, NPGRLCH,",
        "  NACCAVAS, NPLINF, NPLAC, NPHEM, NPWMR, NPNLOSS, NACCCBD, NPPVASC",
        "",
        "DERIVED fields — compute from other extracted values, do not re-read the report:",
        "  NACCVASC (from NACCAVAS/NPLINF/NPLAC/NPHEM/NPWMR/NACCARTE),",
        "  NACCINF (from NPLINF+NPLAC), NACCHEM (from NPHEM), NACCLEWY (from NPLBOD)",
        "",
    ]
    for name, info in model.model_fields.items():
        lines.extend(describe_field(name, info, indent=0))
    lines += [
        "",
        "Use null for any field whose value cannot be determined from the report.",
        "Do not invent information. Extract only what is explicitly stated or clearly implied.",
        "All integer codes must be exact values from the allowed set in each description.",
    ]
    return "\n".join(lines)
