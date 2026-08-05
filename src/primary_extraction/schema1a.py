"""schema1a.py — Pass 1a: Core staging, AD/ABC, Lewy body, and primary/contributing dx (35 vars).

Pass 1a variables:
  Specimen:      NPSEX, NPFIX, NPFIXX, NPWBRWT, NPWBRF, NPPMIH
  Gross:         NPGRLA, NPGRHA, NPGRSNH, NPGRLCH, NPGRCCA, NACCAVAS
  Legacy vasc:   NPLINF, NPLAC, NPHEM
  WM/arterio:    NPWMR, NACCARTE
  Microscopic:   NPNLOSS, NPHIPSCL
  Lewy:          NACCLEWY, NPLBOD
  AD/ABC:        NPTHAL, NACCBRAA, NACCNEUR, NPADNC, NACCDIFF, NACCAMY
  Primary dx:    NPPFTLD, NPCFTLD, NPPVASC, NPCVASC, NPPAD, NPCAD,
                 NPPLEWY, NPCLEWY
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

class SeverityCode(int, Enum):
    none = 0
    mild = 1
    moderate = 2
    severe = 3
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
    """Specimen and case-level details - 6 variables."""
    model_config = ConfigDict(extra="forbid")

    NPSEX: Optional[int] = Field(
        None,
        description=(
            "Subject biological sex recorded at autopsy. "
            "Codes: 1=Male, 2=Female. "
            "Look in the demographics or header section. "
            "Do not infer from pronouns alone - use null if not stated."
        ),
    )
    NPFIX: Optional[int] = Field(
        None,
        description=(
            "Fixative used for brain preservation before sectioning. "
            "Codes: 1=Formalin (most common; 'fixed in 10% formalin', 'formalin-fixed'), "
            "2=Paraformaldehyde ('PFA', '4% paraformaldehyde'), "
            "7=Other (must also populate NPFIXX with the fixative name). "
            "Look for phrases like 'brain was fixed in...', 'fixed overnight in...'."
        ),
    )
    NPFIXX: Optional[str] = Field(
        None,
        description=(
            "Free-text name of the fixative when NPFIX=7 (Other). "
            "Populate ONLY when NPFIX=7, otherwise null "
            "Examples: 'glutaraldehyde', 'Bouin's solution', 'methanol'."
        ),
    )
    NPWBRWT: float = Field(
        description=(
            "Whole brain weight in grams, measured before or after fixation. "
            "Valid range: 100-2500 g. Use 9999 if weight unknown. "
            "Use 9999 if weight unknown. "
            "Extract the numeric value only - strip units (e.g., '1250 g' → 1250). "
            "If weight is reported as a range, use the midpoint. "
            "Look for 'brain weight', 'cerebral weight', 'combined weight', 'weighs X grams'. "
            "If only cerebellum or brainstem weight is given separately, do not add them "
            "unless the report explicitly states 'total brain weight'. "
            "Normal adult range: 1100-1400 g (male), 1000-1300 g (female)."
        ),
    )
    NPWBRF: int = Field(
        description=(
            "Whether the recorded brain weight was measured fresh (before fixation) "
            "or after fixation. Fixation typically adds 5-10% weight. "
            "Codes: 1=Fresh (weighed at autopsy before fixation), "
            "2=Fixed (weighed after fixation period), "
            "8=Not applicable (weight not recorded or not relevant). "
            "Look for 'fresh weight', 'fixed weight', or context clues like 'weighed at autopsy'."
        ),
    )
    NPPMIH: float = Field(
        description=(
            "Postmortem interval (PMI): time from death to brain fixation or freezing, in hours. "
            "Valid range: 0.0-98.9 hours; "
            "Use 99.9 if unknown or not reported. "
            "Look for 'postmortem interval', 'PMI', 'time of death to autopsy'. "
            "Convert days to hours if needed (e.g., '2 days' → 48.0). "
            "If a range is given, use the midpoint."
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
        if v is not None and v not in {1, 2, 7}:
            raise ValueError(f"NPFIX must be 1, 2, or 7, got {v}")
        return v

    @field_validator("NPWBRWT")
    @classmethod
    def val_weight(cls, v):
        if v != 9999 and not (100 <= v <= 2500):
            raise ValueError(f"NPWBRWT must be 100-2500 or 9999, got {v}")
        return v

    @field_validator("NPWBRF")
    @classmethod
    def val_wbrf(cls, v):
        if v not in {1, 2, 8}:
            raise ValueError(f"NPWBRF must be 1, 2, or 8, got {v}")
        return v

    @field_validator("NPPMIH")
    @classmethod
    def val_pmih(cls, v):
        if v != 99.9 and not (0.0 <= v <= 98.9):
            raise ValueError(f"NPPMIH must be 0.0-98.9 or 99.9, got {v}")
        return v


class GrossAndVascular(BaseModel):
    """Gross findings + vascular flags + WM/arteriolosclerosis - 11 variables."""
    model_config = ConfigDict(extra="forbid")

    NPGRLA: int = Field(
        description=(
            "Presence of focal lobar atrophy on gross examination (any lobe). "
            "Codes: 0=None (no focal atrophy identified), 1=Yes (focal lobar atrophy present), "
            "8=Not assessed, 9=Unknown. "
            "Focal lobar atrophy means one or more lobes are selectively and disproportionately "
            "shrunken compared to the rest of the cortex. "
            "Look for: 'focal frontal atrophy', 'frontotemporal atrophy', 'parietal atrophy', "
            "'knife-edge gyri', 'asymmetric cortical atrophy'. "
            "Also check the final diagnoses - if neuronal loss or axonal loss is described as "
            "most severe in specific lobes, code 1 even if the gross description uses diffuse language. "
            "DISTINGUISH from diffuse cortical atrophy (see NPGRCCA): "
            "Code 1 if one or more lobes are selectively and disproportionately shrunken. "
            "Global diffuse atrophy without focal lobar emphasis → code 0."
        ),
    )
    NPGRHA: SeverityCode = Field(
        description=(
            "Severity of hippocampal atrophy on gross examination. "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "The hippocampus is assessed bilaterally on the medial temporal surface. "
            "Look for: 'hippocampal atrophy', 'hippocampal shrinkage', 'medial temporal atrophy', "
            "'small hippocampi', 'parahippocampal atrophy', 'reduced hippocampal volume'. "
            "Grading: Mild=subtle volume reduction compared to expected; "
            "Moderate=clearly shrunken, firm; Severe=markedly shrunken, leather-like. "
            "If only one side is affected but severity stated, code the severity as described. "
            "If 'mild to moderate' → 2; 'moderate to severe' → 3. "
            "If qualitative atrophy language accompanies a measurement, grade from the language. "
            "If only a raw measurement is given with no atrophy description → code 9. "
            "Reserve 0 only if the report explicitly states the hippocampus is normal or unremarkable."
        ),
    )
    NPGRSNH: SeverityCode = Field(
        description=(
            "Severity of substantia nigra (SN) hypopigmentation on gross examination. "
            "Codes: 0=None (normal dark pigmentation), 1=Mild, 2=Moderate, 3=Severe (near-complete pallor), "
            "8=Not assessed, 9=Unknown. "
            "The SN is normally a darkly pigmented band in the midbrain tegmentum. "
            "Hypopigmentation (pallor, depigmentation) is the gross hallmark of Parkinson-spectrum disease. "
            "Look for: 'pallor of substantia nigra', 'depigmentation of SN', 'hypopigmented SN', "
            "'loss of pigmentation in the substantia nigra', 'pale substantia nigra'. "
            "Grading guidance: Mild=slightly pale compared to expected; "
            "Moderate=clearly paler than normal; "
            "Severe requires near-complete or complete depigmentation - almost white. "
            "'Marked depigmentation', 'marked loss of pigmentation', or 'significant depigmentation' "
            "without explicit near-complete or complete loss → code 2 "
            "If depigmentation is described without any severity qualifier → code 2. "
            "Reserve 3 only for near-complete or complete depigmentation. "
            "If the report only notes SN pigmentation is 'normal' or 'intact' → code 0."
        ),
    )
    NPGRLCH: SeverityCode = Field(
        description=(
            "Severity of locus coeruleus (LC) hypopigmentation on gross examination. "
            "Codes: 0=None (normal blue-gray pigmentation), 1=Mild, 2=Moderate, 3=Severe, "
            "8=Not assessed, 9=Unknown. "
            "The LC is a pigmented nucleus in the dorsal pons; normally blue-gray ('coeruleus'=blue). "
            "Look for: 'locus coeruleus pallor', 'depigmentation of locus coeruleus', "
            "'hypopigmented locus ceruleus', 'pale LC'. "
            "Often assessed together with SN; both can be affected in Parkinson-spectrum disease. "
            "If the LC is not mentioned in the report, code 0 - LC pigmentation is a visually "
            "unavoidable gross finding when the pons is sectioned; silence after routine examination "
            "means normal pigmentation. "
            "Reserve 8 only when the pons was explicitly not available or not examined. "
            "If report says LC is 'normally pigmented' or 'intact' → code 0."
        ),
    )
    NPGRCCA: SeverityCode = Field(
        description=(
            "Severity of diffuse cerebral cortical atrophy on gross examination. "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Captures global/bilateral cortical volume loss. "
            "Note: 'diffuse' describes distribution, not severity - grade independently. "
            "If explicit severity words are absent, infer from structural markers: "
            "slight sulcal widening → 1; clearly widened sulci, mild ventricular enlargement → 2; "
            "prominent sulcal widening, markedly thinned gyri, significant ventricular enlargement → 3. "
            "If atrophy is described only in specific lobes, grade based on overall extent - "
            "focal/regional atrophy without global involvement grades lower than diffuse. "
            "0=No atrophy reported; 9=atrophy present but severity cannot be determined."
        ),
    )
    NACCAVAS: SeverityCode = Field(
        description=(
            "Severity of atherosclerosis of the circle of Willis and major cerebral arteries. "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Atherosclerosis = intimal plaque formation in large vessels (circle of Willis, "
            "basilar artery, MCA, ACA, PCA). Distinct from arteriolosclerosis (NACCARTE). "
            "Look for 'atherosclerosis', 'arteriosclerosis', 'calcified plaques', "
            "'atheromatous plaques', 'stenosis' of named cerebral arteries. "
            "Grading: Mild=single or few focal plaques, no significant luminal narrowing; "
            "Moderate=plaques in multiple vessels or with up to ~50% stenosis; "
            "Severe=extensive plaques, >50% stenosis or near-occlusion. "
            "A single plaque in one vessel without stenosis → code 1. "
            "If report says 'mild to moderate' → code 2. 'Moderate to severe' → code 3. "
            "If arteries described as patent or without atherosclerosis → code 0."
        ),
    )
    NPLINF: int = Field(
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
    NPLAC: int = Field(
        description=(
            "Lacunar infarcts present - small vessel infarcts <1.5 cm. "
            "Codes: 1=Present, 2=Absent, 3=Not assessed, 9=Unknown. "
            "Lacunes are small cavitated lesions from occlusion of penetrating arteries, "
            "typically in basal ganglia, thalamus, internal capsule, pons, or deep white matter. "
            "Look for: 'lacunar infarct', 'lacune', 'lacunae', 'small vessel infarct', "
            "'état lacunaire', 'cribriform change' (when referring to actual infarcts, not perivascular spaces). "
            "DISTINGUISH from dilated perivascular spaces (Virchow-Robin spaces): "
            "perivascular spaces are NOT infarcts - do not code those as lacunes. "
            "If no lacunes mentioned and report is complete → code 2 (Absent)."
        ),
    )
    NPHEM: int = Field(
        description=(
            "Hemorrhage(s) present - includes microbleeds, petechiae, macrohemorrhage. "
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
    NPWMR: SeverityCode = Field(
        description=(
            "Severity of white matter rarefaction: myelin loss, leukoencephalopathy, or "
            "white matter axonal loss. Search BOTH the gross description AND all final "
            "diagnoses - the relevant finding may appear under any diagnosis heading. "
            "Accept any description of white matter myelin or axonal pathology regardless "
            "of exact terminology: 'white matter rarefaction', 'white matter pallor', "
            "'myelin loss', 'white matter gliosis', 'leukoaraiosis', 'reduction in bulk/volume "
            "of white matter', 'periventricular white matter changes', 'leukoencephalopathy'. "
            "IMPORTANT: absence of the word 'rarefaction' does NOT mean the finding is absent. "
            "Grading (use the worst region described across the entire report): "
            "0=None: No white matter pathology reported anywhere in the report. "
            "1=Mild: patchy or focal, limited to isolated regions. "
            "2=Moderate: confluent periventricular or multifocal; or 'mild to moderate'. "
            "3=Severe: extensive or confluent throughout; diffuse; severe in multiple regions; "
            "'moderate to severe'; or any single region described as severe. "
            "8=Not assessed. 9=Unknown. "
            "Only output 0 after confirming no white matter pathology appears anywhere "
            "in the gross description or final diagnoses."
        ),
    )
    NACCARTE: SeverityCode = Field(
        description=(
            "Severity of arteriolosclerosis - small vessel wall thickening. "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Arteriolosclerosis = concentric wall thickening, hyalinization, and luminal narrowing "
            "of small penetrating arteries and arterioles in basal ganglia, white matter, brainstem. "
            "DISTINCT from atherosclerosis (NACCAVAS, which affects large named arteries). "
            "Look for: 'arteriolosclerosis', 'hyaline arteriolosclerosis', 'arteriolar thickening', "
            "'small vessel disease', 'arteriolar hyalinization', 'concentric wall thickening'. "
            "Grading mirrors NACCAVAS: Mild=occasional thickened arterioles; "
            "Moderate=widespread thickening; Severe=extensive with near-obliteration of lumina. "
            "If arteriolosclerosis is not mentioned → output 0."
        ),
    )

    @field_validator("NPGRLA")
    @classmethod
    def val_grla(cls, v):
        if v not in {0, 1, 8, 9}:
            raise ValueError(f"NPGRLA must be 0, 1, 8, or 9, got {v}")
        return v

    @field_validator("NPLINF", "NPLAC", "NPHEM")
    @classmethod
    def val_vasc(cls, v):
        if v not in {1, 2, 3, 9}:
            raise ValueError(f"Vascular flag must be 1, 2, 3, or 9, got {v}")
        return v


class MicroscopicFindings(BaseModel):
    """Microscopic findings - 2 variables."""
    model_config = ConfigDict(extra="forbid")

    NPNLOSS: SeverityCode = Field(
        description=(
            "Severity of neuronal loss specifically in the substantia nigra (SN) pars compacta only. "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Look for: 'neuronal loss in substantia nigra', 'depopulation of SN neurons', "
            "'loss of dopaminergic neurons', 'reduced neuronal density in SN pars compacta', "
            "'neuronal depletion substantia nigra'. "
            "If neuronal loss is described in a severity list (e.g. 'severe: X, Y; moderate: SN, Z'), "
            "Note: this field is SPECIFICALLY for SN neuronal loss "
            "use the severity qualifier assigned to the SN specifically, not the worst overall. "
            "0=No SN neuronal loss reported; 8=SN not assessed; 9=severity cannot be determined. "
            "Do NOT use gross depigmentation language ('depigmented', 'loss of pigmentation', "
            "'pale substantia nigra') as evidence - that is a gross finding captured in NPGRSNH. "
            "NPNLOSS requires microscopic evidence of neuronal loss from the final diagnoses section only. "
            "If no microscopic neuronal loss stated for SN → code 8."
        ),
    )
    NPHIPSCL: int = Field(
        description=(
            "Hippocampal sclerosis (HS): severe CA1/subiculum neuronal loss with gliosis. "
            "Codes: 0=None, 1=Unilateral, 2=Bilateral, 3=Present but laterality not assessed, "
            "8=Not assessed, 9=Unknown. "
            "Hippocampal sclerosis (HS) is characterized by severe CA1 pyramidal cell loss "
            "with reactive astrogliosis, often with relative sparing of CA2. "
            "Common in TDP-43 pathology, epilepsy, and aging-related HS (ARTAG). "
            "Look for: 'hippocampal sclerosis', 'CA1 neuronal loss', 'hippocampal gliosis', "
            "'sclerosis of the hippocampus', 'HS-Aging'. "
            "Do not confuse with hippocampal atrophy (NPGRHA) - this is a microscopic finding. "
            "hippocampal sclerosis is a microscopic finding. "
            "HS requires confirmed CA1 neuronal loss and gliosis. "
            "If hippocampal sclerosis not mentioned → code 0."
            "Use 8 only if the report explicitly states hippocampal sections were not examined. "
            "Use 9 only if hippocampal tissue was unavailable or assessment was equivocal."
        ),
    )

    @field_validator("NPHIPSCL")
    @classmethod
    def val_hipscl(cls, v):
        if v not in {0, 1, 2, 3, 8, 9}:
            raise ValueError(f"NPHIPSCL must be 0, 1, 2, 3, 8, or 9, got {v}")
        return v


class LewyPathology(BaseModel):
    """Lewy body pathology - 2 variables."""
    model_config = ConfigDict(extra="forbid")

    NPLBOD: LewyBodyPattern = Field(
        description=(
            "Lewy body pathology distribution pattern (raw observation). "
            "Codes: 0=No Lewy bodies identified; "
            "1=Brainstem-predominant (SN, LC, dorsal vagal nucleus, raphe - no limbic/cortical); "
            "2=Limbic/transitional (brainstem + amygdala, entorhinal, cingulate - some cortex); "
            "3=Neocortical/diffuse (widespread, including frontal/temporal/parietal neocortex); "
            "4=Amygdala-predominant (heavy amygdala involvement without proportionate brainstem); "
            "5=Olfactory bulb only; 8=Not assessed, 9=Unknown. "
            "Look for: 'Lewy bodies', 'Lewy neurites', 'alpha-synuclein inclusions', "
            "'synuclein IHC positive', 'Lewy body disease'. "
            "If the report gives a named pattern (e.g., 'neocortical Lewy body disease', "
            "'diffuse Lewy body disease') → map to the appropriate code. "
            "If Lewy bodies are present but distribution is not characterized → code 4 (Unspecified). "
            "If alpha-synuclein IHC was performed and the report does NOT mention Lewy bodies "
            "in the diagnoses or microscopic section → code 0 (No Lewy bodies). "
            "Reserve code 8 only when synuclein staining is explicitly stated as not performed or not assessed. "
            "NPLBOD is the raw observation; NACCLEWY is derived from it - keep them consistent."
        ),
    )
    NACCLEWY: int = Field(
        description=(
            "Derived Lewy body distribution pattern summary - derive from NPLBOD. "
            "Codes: 0=No Lewy bodies, 1=Brainstem-predominant, 2=Limbic/transitional or amygdala-predominant, "
            "3=Neocortical/diffuse, 4=Lewy bodies present but pattern unspecified or olfactory bulb only, "
            "8=Not assessed, 9=Unknown. "
            "Brainstem-predominant=Lewy bodies confined to brainstem nuclei (SN, LC, dorsal vagal). "
            "Limbic=brainstem + limbic structures (amygdala, cingulate, entorhinal). "
            "Neocortical=widespread including neocortex. "
            "When NPLBOD=4 (amygdala), code NACCLEWY=2. "
            "When NPLBOD=5 (olfactory bulb), code NACCLEWY=4."
        ),
    )

    @field_validator("NACCLEWY")
    @classmethod
    def val_lewy(cls, v):
        if v not in {0, 1, 2, 3, 4, 8, 9}:
            raise ValueError(f"NACCLEWY must be 0, 1, 2, 3, 4, 8, or 9, got {v}")
        return v


class ADPathology(BaseModel):
    """Alzheimer's disease ABC neuropathological scores - 6 variables."""
    model_config = ConfigDict(extra="forbid")

    NPTHAL: ThalPhase = Field(
        description=(
            "Thal phase of amyloid-beta (Aβ) plaque deposition - the 'A' score in ABC. "
            "Codes: 0=No Aβ plaques; "
            "1=Phase 1 (neocortex only - frontal, temporal, parietal, occipital); "
            "2=Phase 2 (neocortex + allocortex: entorhinal, hippocampus, cingulate, insula); "
            "3=Phase 3 (Phase 2 + subcortical: basal ganglia, cholinergic nuclei, thalamus); "
            "4=Phase 4 (Phase 3 + brainstem: SN, LC, raphe); "
            "5=Phase 5 (Phase 4 + cerebellum); "
            "8=Not assessed, 9=Unknown. "
            "NIA-AA A-score to Thal phase conversion - apply this when the report gives an A-score: "
            "A0 → code 0, A1 → code 1 or 2, A2 → code 3, A3 → code 4 or 5. "
            "WARNING: A3 does NOT mean phase 3 - it means phase 4 or 5. "
            "If not stated directly, infer from the distribution of amyloid plaques described. "
            "Note: Thal phase is based on amyloid/Aβ PLAQUES, not CAA (which is NACCAMY). "
            "When A-score maps to a range (A1→1-2, A3→4-5), default to the lower phase "
            "unless the report confirms the defining region for the higher phase. "
            "A1 → code 1 unless allocortex (entorhinal, hippocampus, cingulate, insula) is explicitly mentioned → code 2. "
            "A3 → code 4 unless cerebellum is explicitly mentioned → code 5. "
            "Use 0 only if Aβ plaques are explicitly absent or diagnosis excludes amyloid pathology. "
            "Use 8 if amyloid assessment was explicitly not performed."
        ),
    )
    NACCBRAA: BraakStage = Field(
        description=(
            "Braak neurofibrillary tangle (NFT) stage - the 'B' score in ABC. "
            "Codes: 0=No NFTs identified (NFTs explicitly absent or absent on tau staining); "
            "1=Stage I (NFTs confined to transentorhinal/entorhinal layer Pre-alpha); "
            "2=Stage II (NFTs in entorhinal cortex and hippocampus CA1); "
            "3=Stage III (NFTs in entorhinal + hippocampus + association neocortex: "
            "temporal pole, insula, cingulate); "
            "4=Stage IV (NFTs throughout temporal, frontal, parietal association cortex); "
            "5=Stage V (NFTs in all association areas including prefrontal); "
            "6=Stage VI (NFTs in primary cortices: motor, sensory, visual); "
            "7=The presence of a tauopathy (other than aging/AD) precludes Braak staging (e.g., PSP, CBD, Pick's, FTLD-tau); "
            "8=Staging not performed / not assessable; 9=Unknown. "
            "NIA-AA B-score: stages 0=B0, 1-2=B1, 3-4=B2, 5-6=B3. "
            "STEP 1 - If the report gives a B-score directly, convert it: "
            "B0 → 0, B1 → 1, B2 → 3, B3 → 5. "
            "These are the LOWER bound of each B-score range. Default to lower unless distribution confirms higher. "
            "STEP 2 - If no B-score but NFT distribution is described, infer stage: "
            "entorhinal/transentorhinal only → 1; adds hippocampus CA1 → 2; "
            "adds temporal/insula/cingulate → 3; widespread frontal/parietal → 4; "
            "all association cortex → 5; primary cortices (motor/sensory/visual) → 6. "
            "STEP 3 - Special cases: "
            "If PRIMARY diagnosis is CBD, PSP, or Pick's disease → code 7. "
            "If tau immunoreactivity described but NFTs not specifically mentioned → 8. "
            "Code 0 ONLY if NFTs explicitly stated absent. "
            "PART and CTE as secondary findings do not trigger code 7 - stage normally."
        ),
    )
    NACCNEUR: CERADScore = Field(
        description=(
            "CERAD score for neuritic (senile) plaques - the 'C' score in ABC. "
            "Neuritic plaques = amyloid cores with surrounding dystrophic neurites (tau-positive). "
            "Codes: 0=No neuritic plaques; "
            "1=Sparse (occasional neuritic plaques in neocortex); "
            "2=Moderate (easily found but not numerous); "
            "3=Frequent (numerous in multiple neocortical fields); "
            "8=Not assessed, 9=Unknown. "
            "Look for: 'CERAD score', 'neuritic plaques', 'senile plaques', 'C-score'. "
            "The ABC header C-score maps directly: C0=0, C1=1, C2=2, C3=3. "
            "CRITICAL: diffuse plaques (NACCDIFF) are NOT neuritic plaques - "
            "do not code NACCNEUR from diffuse plaque descriptions. "
            "Neuritic plaques require a dense amyloid core with tau-positive dystrophic neurites. "
            "If neuritic plaques are not mentioned and C-score is not given → output 0. "
            "Use 8 only if amyloid/tau staining was explicitly not performed."
        ),
    )
    NPADNC: ADNCScore = Field(
        description=(
            "NIA-AA overall Alzheimer's Disease Neuropathological Change (ADNC) score - "
            "the final ABC composite. "
            "Codes: 0=Not AD / no/minimal ADNC; "
            "1=Low ADNC (Thal 1-2, Braak I-II, CERAD 0-1 - age-related changes only); "
            "2=Intermediate ADNC (Thal 3, Braak III-IV, CERAD 2 - some but not full AD); "
            "3=High ADNC (Thal 4-5, Braak V-VI, CERAD 3 - consistent with full AD diagnosis); "
            "8=Not assessed, 9=Unknown. "
            "Look for: 'ADNC', 'ABC score', 'low/intermediate/high AD neuropathological change', "
            "'meets criteria for AD', 'NIA-AA criteria'. "
            "If the report gives the ABC scores separately: "
            "all high → 3; any low → lower composite. "
            "If no AD pathology or ADNC assessment is mentioned → code 0. "
            "Use 8 only if the report explicitly states AD assessment was not performed."
        ),
    )
    NACCDIFF: CERADScore = Field(
        description=(
            "CERAD score for diffuse (pre-amyloid) plaques. "
            "Diffuse plaques = Aβ deposits without dense core and without neuritic change (tau-negative). "
            "Codes: 0=None, 1=Sparse, 2=Moderate, 3=Frequent, 8=Not assessed, 9=Unknown. "
            "Look for: 'diffuse plaques', 'diffuse amyloid deposits', 'pre-amyloid plaques'. "
            "Often reported alongside neuritic plaques ('sparse neuritic and moderate diffuse plaques'). "
            "Diffuse plaques alone (without neuritic plaques) are less pathologically significant. "
            "Use same CERAD grading as NACCNEUR: sparse/moderate/frequent. "
            "If diffuse plaques not mentioned → code 0. "
            "Use 8 only if amyloid staining was explicitly not performed."
        ),
    )
    NACCAMY: SeverityCode = Field(
        description=(
            "Severity of cerebral amyloid angiopathy (CAA) - amyloid deposition in vessel walls. "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "CAA = Aβ deposition in walls of leptomeningeal and cortical arteries/arterioles. "
            "DISTINCT from parenchymal amyloid plaques (NPTHAL). "
            "Look for: 'cerebral amyloid angiopathy', 'CAA', 'amyloid in vessel walls', "
            "'congophilic angiopathy', 'vascular amyloid'. "
            "Grading: Mild=occasional vessel wall deposits; "
            "Moderate=widespread but not all vessels affected; "
            "Severe=extensive, many vessels affected, may have associated microinfarcts or hemorrhage. "
            "If CAA is explicitly graded in the report, use that grade. "
            "If CAA not mentioned → code 0. "
            "Code 0 if amyloid plaques are absent or not mentioned "
            "Use 8 only if amyloid staining was explicitly not performed."
        ),
    )


class PrimaryContribDx(BaseModel):
    """Primary and contributing diagnosis pairs - 8 variables.

    Coding convention: 1=Yes (diagnosis present in this role), 2=No.
    Zero is NOT a valid code for these fields.
    """
    model_config = ConfigDict(extra="forbid")

    NPPAD: int = Field(
        description=(
            "Alzheimer's disease listed as PRIMARY neuropathological diagnosis. "
            "Codes: 1=Yes, 2=No. "
            "Code 1 if the report identifies AD (high ADNC, NPADNC=3) as the primary/principal finding. "
            "If AD is a contributing co-pathology only → code 2 (and set NPCAD=1 instead). "
            "Look for: 'primary diagnosis: Alzheimer's disease', 'Alzheimer's disease, NIA-AA high'. "
            "Decision rule: identify the PRIMARY diagnosis label in the Final Diagnoses section. "
            "If AD/ADNC is listed first or explicitly labeled primary → NPPAD=1, NPCAD=2. "
            "If another disease is listed first and AD appears as a secondary finding → NPPAD=2, NPCAD=1. "
            "If there is no AD pathology at all → NPPAD=2, NPCAD=2."
        ),
    )
    NPCAD: int = Field(
        description=(
            "Alzheimer's disease listed as a CONTRIBUTING (secondary) neuropathological diagnosis. "
            "Codes: 1=Yes, 2=No. "
            "Code 1 if AD pathology is present but is NOT the primary diagnosis "
            "(i.e., another disease is primary). Common in mixed dementia cases. "
            "Decision rule: if AD is the primary diagnosis → NPPAD=1 and NPCAD=2. "
            "If AD is secondary to another primary disease → NPPAD=2 and NPCAD=1. "
            "If no AD pathology at all → both NPPAD=2 and NPCAD=2."
        ),
    )
    NPPLEWY: int = Field(
        description=(
            "Lewy body disease listed as the PRIMARY neuropathological diagnosis. "
            "Codes: 1=Yes, 2=No. "
            "Encompasses PD, DLB, and PD with dementia when Lewy body pathology "
            "is the principal finding. "
            "Code 1 if the report identifies Lewy body disease / DLB / PD as primary. "
            "If Lewy bodies are incidental or contributing only → code 2 (set NPCLEWY=1). "
            "Decision rule: identify the PRIMARY diagnosis in the Final Diagnoses section. "
            "If Lewy body disease / DLB / PD is listed first or explicitly labeled primary → NPPLEWY=1, NPCLEWY=2. "
            "If another disease is primary and Lewy body pathology is secondary → NPPLEWY=2, NPCLEWY=1. "
            "If no Lewy body pathology at all → NPPLEWY=2, NPCLEWY=2. "
            "If your reasoning leads you to code 0 (meaning No/Absent), output 2 instead."
        ),
    )
    NPCLEWY: int = Field(
        description=(
            "Lewy body disease listed as a CONTRIBUTING (secondary) neuropathological diagnosis. "
            "Codes: 1=Yes, 2=No. "
            "Code 1 when Lewy body pathology is present but is not the primary diagnosis. "
            "Common in mixed AD+Lewy body cases. "
            "Decision rule: if Lewy body disease is the primary diagnosis → NPPLEWY=1 and NPCLEWY=2. "
            "If Lewy body pathology is secondary to another primary disease → NPPLEWY=2 and NPCLEWY=1. "
            "If no Lewy body pathology at all → both NPPLEWY=2 and NPCLEWY=2. "
            "If your reasoning leads you to code 0 (meaning No/Absent), output 2 instead."
        ),
    )
    NPPVASC: int = Field(
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
    NPCVASC: int = Field(
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
    NPPFTLD: int = Field(
        description=(
            "Frontotemporal lobar degeneration (FTLD) listed as the PRIMARY diagnosis. "
            "Codes: 1=Yes, 2=No. "
            "FTLD encompasses FTLD-tau (Pick's, PSP, CBD, AGD) and FTLD-TDP subtypes. "
            "Code 1 if any FTLD subtype is and should be listed as the principal neuropathological diagnosis. "
            "Look for: 'FTLD', 'frontotemporal lobar degeneration', as primary diagnosis label. "
            "CRITICAL: This field uses 1=Yes and 2=No only. Code 0 is not allowed. "
            "If your reasoning leads you to code 0 (meaning No/Absent), output 2 instead."
        ),
    )
    NPCFTLD: int = Field(
        description=(
            "Frontotemporal lobar degeneration (FTLD) listed as the CONTRIBUTING diagnosis. "
            "Codes: 1=Yes, 2=No. "
            "Code 1 if FTLD present but not the primary diagnosis."
        ),
    )

    @field_validator(
        "NPPAD", "NPCAD", "NPPLEWY", "NPCLEWY",
        "NPPVASC", "NPCVASC", "NPPFTLD", "NPCFTLD"
    )
    @classmethod
    def val_primary_contrib(cls, v):
        if v not in {1, 2}:
            raise ValueError(
                f"Primary/contributing diagnosis field must be 1 or 2, got {v}"
            )
        return v

    @model_validator(mode="after")
    def validate_mutual_exclusion(self) -> "PrimaryContribDx":
        """Both primary and contributing cannot be 1 for the same disease."""
        pairs = [
            ("NPPAD", "NPCAD", "AD"),
            ("NPPLEWY", "NPCLEWY", "Lewy body"),
            ("NPPVASC", "NPCVASC", "Vascular"),
            ("NPPFTLD", "NPCFTLD", "FTLD"),
        ]
        for prim, contrib, name in pairs:
            if getattr(self, prim) == 1 and getattr(self, contrib) == 1:
                raise ValueError(
                    f"{name}: both {prim}=1 (primary) and {contrib}=1 (contributing) "
                    f"- a disease can only be primary OR contributing, not both."
                )
        return self


# ---------------------------------------------------------------------------
# Top-level Pass 1 model
# ---------------------------------------------------------------------------

class Pass1aExtraction(BaseModel):
    """Pass 1a extraction - 35 variables.

    Covers: specimen details, gross findings, legacy vascular flags,
    WM/arteriolosclerosis, microscopic findings, Lewy body pathology,
    AD/ABC scores, and primary/contributing diagnosis pairs.
    """
    model_config = ConfigDict(extra="forbid")

    specimen_info: SpecimenInfo = Field(
        description="Specimen details - NPSEX, NPFIX, NPFIXX, NPWBRWT, NPWBRF, NPPMIH"
    )
    gross_and_vascular: GrossAndVascular = Field(
        description=(
            "Gross findings and vascular flags - "
            "NPGRLA, NPGRHA, NPGRSNH, NPGRLCH, NPGRCCA, NACCAVAS, "
            "NPLINF, NPLAC, NPHEM, NPWMR, NACCARTE"
        )
    )
    microscopic_findings: MicroscopicFindings = Field(
        description="Microscopic findings - NPNLOSS, NPHIPSCL"
    )
    lewy_pathology: LewyPathology = Field(
        description="Lewy body pathology - NPLBOD, NACCLEWY"
    )
    ad_pathology: ADPathology = Field(
        description="AD ABC scores - NPTHAL, NACCBRAA, NACCNEUR, NPADNC, NACCDIFF, NACCAMY"
    )
    primary_contrib_dx: PrimaryContribDx = Field(
        description=(
            "Primary/contributing diagnosis pairs - "
            "NPPAD, NPCAD, NPPLEWY, NPCLEWY, NPPVASC, NPCVASC, NPPFTLD, NPCFTLD"
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
    def validate_confidence_value(self) -> "Pass1aExtraction":
        if self.extraction_confidence and self.extraction_confidence not in {
            "high", "moderate", "low"
        }:
            raise ValueError("extraction_confidence must be 'high', 'moderate', or 'low'")
        return self

    @model_validator(mode="after")
    def enforce_null_policy(self) -> "Pass1aExtraction":
        """Non-nullable fields must not be null - use 0, 8, or 9 instead."""
        nullable = {"NPSEX", "NPFIX", "NPFIXX"}
        sections = [
            self.specimen_info, self.gross_and_vascular,
            self.microscopic_findings, self.lewy_pathology,
            self.ad_pathology, self.primary_contrib_dx,
        ]
        for section in sections:
            for field_name in section.model_fields:
                if field_name not in nullable and getattr(section, field_name) is None:
                    raise ValueError(
                        f"{field_name} must not be null - use 0, 8, or 9, etc. instead (depending on variable)."
                    )
        return self


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_pass1a_model() -> Type[BaseModel]:
    """Return the top-level extraction model class."""
    return Pass1aExtraction


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


def build_pass1a_format_instructions() -> str:
    """Walk the Pydantic schema and emit LLM-facing format instructions."""
    model = Pass1aExtraction
    lines = [
        "PASS 1a - Extract ONLY the variables listed below.",
        "Respond with a single JSON object with EXACTLY these top-level keys:",
        "",
        "  specimen_info, gross_and_vascular, microscopic_findings,",
        "  lewy_pathology, ad_pathology,",
        "  primary_contrib_dx, field_annotations,",
        "  extraction_confidence, extraction_notes",
        "",
        "Each top-level key maps to a nested object containing its variables.",
        "NEVER output variables at the root level - they must always be nested.",
        "",
        "PRIORITY fields - extract these with the highest care:",
        "  NPSEX, NPFIX, NPWBRWT, NPWBRF, NPPMIH,",
        "  NPGRLA, NPGRCCA, NPGRHA, NPGRSNH, NPGRLCH,",
        "  NACCAVAS, NACCARTE, NPLINF, NPLAC, NPHEM, NPWMR,",
        "  NPNLOSS, NPHIPSCL,",
        "  NPLBOD, NPTHAL, NACCBRAA, NACCNEUR, NPADNC, NACCDIFF, NACCAMY,",
        "  NPPAD, NPCAD, NPPLEWY, NPCLEWY, NPPVASC, NPCVASC, NPPFTLD, NPCFTLD",
        "",
        "DERIVED field - compute from other extracted values, do not re-read the report:",
        "  NACCLEWY (derive from NPLBOD)",
        "",
        "NULL POLICY:",
        "  null is valid ONLY for: NPSEX, NPFIX, NPFIXX.",
        "  For every other variable: use an explicit numeric code - never null.",
        "  If a finding was assessed and absent → 0",
        "  If a structure was explicitly not examined or stain not performed → 8",
        "  If examined or mentioned but severity/result cannot be determined → 9",
        "",
        "SEVERITY LANGUAGE MAPPING:",
        "  'mild' → 1, 'moderate' → 2, 'severe' → 3, 'none/absent/no' → 0",
        "  'mild to moderate' → 2, 'moderate to severe' → 3",
        "",
        "PRIMARY/CONTRIBUTING DX FIELDS:"
        "  Use 1=Yes or 2=No only. Never 0.",
        "  For each disease pair, exactly one field may be 1 and the other must be 2.",
        "  A disease can be PRIMARY OR CONTRIBUTING, not both.",
        "  Do not output 1 for both members of any pair.",
        "  Pairs:",
        "    - AD: NPPAD / NPCAD",
        "    - Lewy body disease: NPPLEWY / NPCLEWY",
        "    - Vascular disease: NPPVASC / NPCVASC",
        "    - FTLD: NPPFTLD / NPCFTLD",
        "  Use the Final Diagnoses section to decide which role applies.",
        "  If the report clearly states the disease is primary, set the primary field to 1 and the contributing field to 2.",
        "  If the report clearly states the disease is contributing/secondary, set the contributing field to 1 and the primary field to 2.",
        "  If the disease is absent, set both fields to 2.",
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