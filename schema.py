"""schema.py — NACC Neuropathology extraction schema (43 variables).

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
            "A SINGLE string — never an array. "
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
            "Look for phrases like 'brain was fixed in...', 'fixed overnight in...'."
        ),
    )
    NPWBRWT: float = Field(
        description=(
            "Whole brain weight in grams, measured before or after fixation. "
            "Valid range: 100-2500 g. Special code: 9999=weight unknown. "
            "Extract the numeric value only — strip units (e.g., '1250 g' → 1250). "
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
            "Valid range: 0.0-98.9 hours; 99.9=unknown. "
            "Look for 'postmortem interval', 'PMI', 'time of death to autopsy'. "
            "Convert days to hours if needed (e.g., '2 days' → 48.0). "
            "If a range is given, use the midpoint. "
            "If PMI is not reported anywhere in the report → output 99.9."
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
        if v is not None and v not in {1, 2, 7}:
            raise ValueError(f"NPFIX must be 1, 2, or 7, got {v}")
        return v

    @field_validator("NPWBRWT")
    @classmethod
    def val_weight(cls, v):
        if v is None:
            raise ValueError("NPWBRWT must not be null — use 9999 if weight is unknown")
        if v != 9999 and not (100 <= v <= 2500):
            raise ValueError(f"NPWBRWT must be 100-2500 or 9999, got {v}")
        return v


    @field_validator("NPWBRF")
    @classmethod
    def val_wbrf(cls, v):
        if v is None:
            raise ValueError("NPWBRF must not be null — use 8 if not applicable")
        if v not in {1, 2, 8}:
            raise ValueError(f"NPWBRF must be 1, 2, or 8, got {v}")
        return v

    @field_validator("NPPMIH")
    @classmethod
    def val_pmih(cls, v):
        if v is None:
            raise ValueError("NPPMIH must not be null — use 99.9 if unknown")
        if v != 99.9 and not (0.0 <= v <= 98.9):
            raise ValueError(f"NPPMIH must be 0.0-98.9 or 99.9, got {v}")
        return v


class GrossFindings(BaseModel):
    """Macroscopic (gross) examination findings observed at autopsy."""

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
            "Also check the final diagnoses — if neuronal loss or axonal loss is described as "
            "most severe in specific lobes, code 1 even if the gross description uses diffuse language. "
            "DISTINGUISH from diffuse cortical atrophy (see NPGRCCA): NPGRLA=1 requires "
            "a specific lobe to be called out as preferentially affected. "
            "Global cortical thinning without focal emphasis → NPGRLA=0."
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
            "Severe requires near-complete or complete depigmentation — almost white. "
            "'Marked depigmentation', 'marked loss of pigmentation', or 'significant depigmentation' "
            "without explicit near-complete or complete loss → code 2, not 3. "
            "If depigmentation is described without any severity qualifier → code 2. "
            "Reserve 3 only for explicitly near-complete or complete depigmentation. "
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
            "If the LC is not mentioned in the report, code 0 — LC pigmentation is a visually "
            "unavoidable gross finding when the pons is sectioned; silence after routine examination "
            "means normal pigmentation. "
            "Reserve 8 only when the pons was explicitly not available or not examined. "
            "If report says LC is 'normally pigmented' or 'intact' → code 0."
        ),
    )
    NPGRCCA: SeverityCode = Field(
        description=(
            "Severity of DIFFUSE cerebral cortical atrophy on gross examination. "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Captures global/bilateral cortical volume loss. "
            "Note: 'diffuse' describes distribution, not severity — grade independently. "
            "If explicit severity words are absent, infer from structural markers: "
            "slight sulcal widening → 1; clearly widened sulci, mild ventricular enlargement → 2; "
            "prominent sulcal widening, markedly thinned gyri, significant ventricular enlargement → 3. "
            "If atrophy is described only in specific lobes, grade based on overall extent — "
            "focal/regional atrophy without global involvement grades lower than diffuse. "
            "0=No atrophy reported; 9=atrophy present but severity cannot be determined."
        ),
    )
    NACCBRNN: int = Field(
        description=(
            "Derived flag: NO significant neuropathological changes identified. "
            "Codes: 0=Neuropathological changes ARE present (normal = no), "
            "1=No significant changes found (essentially normal brain), "
            "8=Not assessed or missing. "
            "Code 1 only when the report concludes the examination is essentially unremarkable "
            "or 'within normal limits for age'. "
            "If ANY pathological finding is present elsewhere in this extraction, code 0. "
            "If the report contains ANY pathological diagnosis — AD, FTLD, vascular disease, "
            "Lewy body disease, or any other finding — code 0."
        ),
    )

    @field_validator("NPGRLA")
    @classmethod
    def val_grla(cls, v):
        if v not in {0, 1, 8, 9}:
            raise ValueError(f"NPGRLA must be 0, 1, 8, or 9, got {v}")
        return v

    @field_validator("NACCBRNN")
    @classmethod
    def val_brnn(cls, v):
        if v not in {0, 1, 8}:
            raise ValueError(f"NACCBRNN must be 0, 1, or 8, got {v}")
        return v


class VascularPathology(BaseModel):
    """Vascular and ischemic neuropathological findings."""

    model_config = ConfigDict(extra="forbid")

    NACCAVAS: SeverityCode = Field(
        description=(
            "Severity of atherosclerosis of the circle of Willis and major cerebral arteries. "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Atherosclerosis = intimal plaque formation in large vessels (circle of Willis, "
            "basilar artery, MCA, ACA, PCA). Distinct from arteriolosclerosis (NACCARTE). "
            "Look for: 'atherosclerosis', 'arteriosclerosis', 'calcified plaques', "
            "'atheromatous plaques', 'stenosis' of named cerebral arteries. "
            "Grading: Mild=single or few focal plaques, no significant luminal narrowing; "
            "Moderate=plaques in multiple vessels or with up to ~50% stenosis; "
            "Severe=extensive plaques, >50% stenosis or near-occlusion. "
            "A single plaque in one vessel without stenosis → code 1. "
            "If report says 'mild to moderate' → code 2. 'Moderate to severe' → code 3. "
            "If arteries described as 'patent' or 'without significant atherosclerosis' → code 0."
        ),
    )
    NPLINF: PresentAbsentCode = Field(
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
    NPLAC: PresentAbsentCode = Field(
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
    NPHEM: PresentAbsentCode = Field(
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
    NPWMR: SeverityCode = Field(
        description=(
            "Severity of white matter rarefaction: myelin loss, leukoencephalopathy, or "
            "white matter axonal loss. Search BOTH the gross description AND all final "
            "diagnoses — the relevant finding may appear under any diagnosis heading. "
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
            "Severity of arteriolosclerosis (small vessel wall thickening). "
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
    NACCVASC: int = Field(
        description=(
            "Derived summary: any vascular pathology present (yes/no). "
            "Codes: 0=No vascular pathology, 1=Vascular pathology present, 9=Unknown. "
            "Code 1 if ANY of the following are present: NACCAVAS≥1, NPLINF=1, NPLAC=1, "
            "NPHEM=1, NPWMR≥1, NACCARTE≥1. "
            "Code 0 only if all vascular fields are 0 or absent. "
            "This is a derived field — derive it from the other vascular findings."
        ),
    )
    NACCINF: int = Field(
        description=(
            "Derived summary: any infarct or lacune present. "
            "Codes: 0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "Code 1 if NPLINF=1 (large infarct) OR NPLAC=1 (lacune). "
            "Code 0 if both NPLINF=2 and NPLAC=2 (both absent). "
            "This is a derived field."
        ),
    )
    NACCHEM: int = Field(
        description=(
            "Derived summary: any hemorrhage or microbleed present. "
            "Codes: 0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "Code 1 if NPHEM=1. Code 0 if NPHEM=2. "
            "This is a derived field."
        ),
    )

    @field_validator("NACCVASC")
    @classmethod
    def val_naccvasc_derived(cls, v):
        if v not in {0, 1, 9}:
            raise ValueError(f"NACCVASC must be 0, 1, or 9, got {v}")
        return v

    @field_validator("NACCINF", "NACCHEM")
    @classmethod
    def val_derived(cls, v):
        if v not in {0, 1, 8, 9}:
            raise ValueError(f"Derived field must be 0, 1, 8, or 9, got {v}")
        return v


class MicroscopicFindings(BaseModel):
    """Microscopic and immunohistochemical cellular findings."""

    model_config = ConfigDict(extra="forbid")

    NPNLOSS: SeverityCode = Field(
        description=(
            "Severity of neuronal loss specifically in the substantia nigra (SN) pars compacta only. "
            "Codes: 0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Unknown. "
            "Look for: 'neuronal loss in substantia nigra', 'depopulation of SN neurons', "
            "'loss of dopaminergic neurons', 'reduced neuronal density in SN pars compacta', "
            "'neuronal depletion substantia nigra'. "
            "If neuronal loss is described in a severity list (e.g. 'severe: X, Y; moderate: SN, Z'), "
            "Note: this field is SPECIFICALLY for SN neuronal loss"
            "use the severity qualifier assigned to the SN specifically, not the worst overall. "
            "0=No SN neuronal loss reported; 8=SN not assessed; 9=severity cannot be determined."
            "Do NOT use gross depigmentation language ('depigmented', 'loss of pigmentation', "
            "'pale substantia nigra') as evidence — that is a gross finding captured in NPGRSNH. "
            "NPNLOSS requires microscopic evidence of neuronal loss from the final diagnoses section only. "
            "If no microscopic neuronal loss is stated for the SN → code 8."
        ),
    )
    NPHIPSCL: int = Field(
        description=(
            "Hippocampal sclerosis (HS): severe CA1/subiculum neuronal loss with gliosis. "
            "Codes: 0=None, 1=Unilateral, 2=Bilateral, 3=Present laterality unspecified, "
            "8=Not assessed, 9=Unknown. "
            "Hippocampal sclerosis (HS) is characterized by severe CA1 pyramidal cell loss "
            "with reactive astrogliosis, often with relative sparing of CA2. "
            "Common in TDP-43 pathology, epilepsy, and aging-related HS (ARTAG). "
            "Look for: 'hippocampal sclerosis', 'CA1 neuronal loss', 'hippocampal gliosis', "
            "'sclerosis of the hippocampus', 'HS-Aging'. "
            "Do not confuse with hippocampal atrophy (NPGRHA) — atrophy is a gross finding; "
            "hippocampal sclerosis is a microscopic finding. "
            "HS requires confirmed CA1 neuronal loss and gliosis. "
            "If the report does not mention hippocampal sclerosis → output 0. "
            "Use 8 only if the report explicitly states hippocampal sections were not examined. "
            "Use 9 only if hippocampal tissue was unavailable or assessment was equivocal."
        ),
    )
    NACCLEWY: int = Field(
        description=(
            "Derived Lewy body distribution pattern summary. "
            "Codes: 0=No Lewy bodies, 1=Brainstem-predominant, 2=Limbic/transitional or amygdala-predominant, "
            "3=Neocortical/diffuse, 4=Lewy bodies present but pattern unspecified or olfactory bulb only, "
            "8=Not assessed, 9=Unknown. "
            "This is a derived field — derive it from NPLBOD (the raw observation). "
            "Brainstem-predominant=Lewy bodies confined to brainstem nuclei (SN, LC, dorsal vagal). "
            "Limbic=brainstem + limbic structures (amygdala, cingulate, entorhinal). "
            "Neocortical=widespread including neocortex. "
            "When NPLBOD=4 (amygdala-predominant) → derive NACCLEWY=2 (limbic/transitional or amygdala-predominant). "
            "When NPLBOD=5 (olfactory bulb only) → derive NACCLEWY=4 (unspecified/olfactory). "
            "If alpha-synuclein IHC was performed and the report does NOT mention Lewy bodies "
            "in the diagnoses or microscopic section → code 0 (No Lewy bodies). "
            "Reserve code 8 only when synuclein staining is explicitly stated as not performed or not assessed."
        ),
    )
    NPLBOD: LewyBodyPattern = Field(
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
            "If alpha-synuclein IHC was performed and the report does NOT mention Lewy bodies "
            "in the diagnoses or microscopic section → code 0 (No Lewy bodies). "
            "Reserve code 8 only when synuclein staining is explicitly stated as not performed or not assessed. "
            "NPLBOD is the raw observation; NACCLEWY is derived from it — keep them consistent."
        ),
    )

    @field_validator("NPHIPSCL")
    @classmethod
    def val_hipscl(cls, v):
        if v is None:
            raise ValueError("NPHIPSCL must not be null — use 8 if not assessed, or 9 if Missing/Unknown")
        if v not in {0, 1, 2, 3, 8, 9}:
            raise ValueError(f"NPHIPSCL must be 0, 1, 2, 3, 8, or 9, got {v}")
        return v

    @field_validator("NACCLEWY")
    @classmethod
    def val_lewy(cls, v):
        if v is None:
            raise ValueError("NACCLEWY must not be null — use 8 if not assessed")
        if v not in {0, 1, 2, 3, 4, 8, 9}:
            raise ValueError(f"NACCLEWY must be 0, 1, 2, 3, 4, 8, or 9, got {v}")
        return v


class ADPathology(BaseModel):
    """Alzheimer's disease ABC neuropathological scores (NIA-AA 2012 criteria)."""

    model_config = ConfigDict(extra="forbid")

    NPTHAL: ThalPhase = Field(
        description=(
            "Thal phase of amyloid-beta (Aβ) plaque deposition — the 'A' score in ABC. "
            "Codes: 0=No Aβ plaques; "
            "1=Phase 1 (neocortex only — frontal, temporal, parietal, occipital); "
            "2=Phase 2 (neocortex + allocortex: entorhinal, hippocampus, cingulate, insula); "
            "3=Phase 3 (Phase 2 + subcortical: basal ganglia, cholinergic nuclei, thalamus); "
            "4=Phase 4 (Phase 3 + brainstem: SN, LC, raphe); "
            "5=Phase 5 (Phase 4 + cerebellum); "
            "8=Not assessed, 9=Unknown. "
            "NIA-AA A-score to Thal phase conversion — apply this when the report gives an A-score: "
            "A0 → code 0, A1 → code 1 or 2, A2 → code 3, A3 → code 4 or 5. "
            "WARNING: A3 does NOT mean phase 3 — it means phase 4 or 5. "
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
            "Braak neurofibrillary tangle (NFT) stage — the 'B' score in ABC. "
            "Codes: "
            "0=No NFTs identified (NFTs explicitly absent or absent on tau staining); "
            "1=Stage I (NFTs confined to transentorhinal/entorhinal layer Pre-alpha); "
            "2=Stage II (NFTs in entorhinal cortex and hippocampus CA1); "
            "3=Stage III (NFTs in entorhinal + hippocampus + association neocortex: "
            "temporal pole, insula, cingulate); "
            "4=Stage IV (NFTs throughout temporal, frontal, parietal association cortex); "
            "5=Stage V (NFTs in all association areas including prefrontal); "
            "6=Stage VI (NFTs in primary cortices: motor, sensory, visual); "
            "7=Non-AD tauopathy precludes Braak staging (e.g., PSP, CBD, Pick's, FTLD-tau); "
            "8=Staging not performed / not assessable; 9=Unknown. "
            "NIA-AA B-score: stages 0=B0, 1-2=B1, 3-4=B2, 5-6=B3. "
            "STEP 1 — If the report gives a B-score directly, convert it: "
            "B0 → 0, B1 → 1, B2 → 3, B3 → 5. "
            "These are the LOWER bound of each B-score range. Default to lower unless distribution confirms higher. "
            "STEP 2 — If no B-score but NFT distribution is described, infer stage: "
            "entorhinal/transentorhinal only → 1; adds hippocampus CA1 → 2; "
            "adds temporal/insula/cingulate → 3; widespread frontal/parietal → 4; "
            "all association cortex → 5; primary cortices (motor/sensory/visual) → 6. "
            "STEP 3 — Special cases: "
            "If PRIMARY diagnosis is CBD, PSP, or Pick's disease → 7. "
            "If tau immunoreactivity described but NFTs not specifically mentioned → 8. "
            "Code 0 ONLY if NFTs explicitly stated absent. "
            "PART and CTE as secondary findings do not trigger code 7 — stage normally."
        ),
    )
    NACCNEUR: CERADScore = Field(
        description=(
            "CERAD score for neuritic (senile) plaques — the 'C' score in ABC. "
            "Neuritic plaques = amyloid cores with surrounding dystrophic neurites (tau-positive). "
            "Codes: 0=No neuritic plaques; "
            "1=Sparse (occasional neuritic plaques in neocortex); "
            "2=Moderate (easily found but not numerous); "
            "3=Frequent (numerous in multiple neocortical fields); "
            "8=Not assessed, 9=Unknown. "
            "Look for: 'CERAD score', 'neuritic plaques', 'senile plaques', 'C-score'. "
            "The ABC header C-score maps directly: C0=0, C1=1, C2=2, C3=3. "
            "CRITICAL: diffuse plaques (NACCDIFF) are NOT neuritic plaques — "
            "do not code NACCNEUR from diffuse plaque descriptions. "
            "Neuritic plaques require a dense amyloid core with tau-positive dystrophic neurites. "
            "If neuritic plaques are not mentioned and C-score is not given → output 0. "
            "Use 8 only if amyloid/tau staining was explicitly not performed."
        ),
    )
    NPADNC: ADNCScore = Field(
        description=(
            "NIA-AA overall Alzheimer's Disease Neuropathological Change (ADNC) score — "
            "the final ABC composite. "
            "Codes: 0=Not AD / no/minimal ADNC; "
            "1=Low ADNC (Thal 1-2, Braak I-II, CERAD 0-1 — age-related changes only); "
            "2=Intermediate ADNC (Thal 3, Braak III-IV, CERAD 2 — some but not full AD); "
            "3=High ADNC (Thal 4-5, Braak V-VI, CERAD 3 — consistent with full AD diagnosis); "
            "8=Not assessed, 9=Unknown. "
            "Look for: 'ADNC', 'ABC score', 'low/intermediate/high AD neuropathological change', "
            "'meets criteria for AD', 'NIA-AA criteria'. "
            "If the report gives the ABC scores separately, derive NPADNC from them: "
            "all three must be high for NPADNC=3; if any one is low, the composite is lower. "
            "If no AD pathology or ADNC assessment is mentioned → output 0. "
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
            "If diffuse plaques are not mentioned in the report → code 0. "
            "Use 8 only if amyloid staining was explicitly not performed."
        ),
    )
    NACCAMY: SeverityCode = Field(
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
            "If CAA is explicitly graded in the report, use that grade. "
            "If CAA is not mentioned in the report → output 0. "
            "Code 0 if amyloid plaques are absent or not mentioned "
            "Use 8 only if amyloid staining was explicitly not performed."
        ),
    )


class DiagnosticCodes(BaseModel):
    """Primary and contributing neuropathological final diagnoses."""

    model_config = ConfigDict(extra="forbid")

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
            "CBD often co-occurs with AD pathology — code 1 regardless of co-pathology. "
            "If not mentioned in the report → code 0."
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
    NPPAD: int = Field(
        description=(
            "Alzheimer's disease listed as the PRIMARY neuropathological diagnosis. "
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
    NACCPROG: YesNoCode = Field(
        description=(
            "Progressive supranuclear palsy (PSP) present as a neuropathological diagnosis. "
            "Codes: 0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "PSP = 4-repeat tauopathy with tufted astrocytes, coiled bodies, globose NFTs "
            "predominantly in basal ganglia, subthalamic nucleus, SN, brainstem. "
            "Look for: 'progressive supranuclear palsy', 'PSP', 'PSP-Richardson', "
            "'tufted astrocytes', 'globose tangles in STN'. "
            "If not mentioned in the report → code 0. "
            "Code 1 regardless of whether PSP is primary or contributing."
        ),
    )
    NACCPICK: YesNoCode = Field(
        description=(
            "Pick's disease (PiD) present as a neuropathological diagnosis. "
            "Codes: 0=No, 1=Yes, 8=Not assessed, 9=Unknown. "
            "Pick's disease = 3-repeat tauopathy with Pick bodies (round, tau-positive inclusions) "
            "and ballooned neurons, with frontotemporal predilection. "
            "Look for: 'Pick disease', 'Pick bodies', 'PiD', '3-repeat tauopathy consistent with PiD'. "
            "If not mentioned in the report → code 0. "
            "Code 1 regardless of whether it is primary or contributing."
        ),
    )
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
            "If TDP-43 is not mentioned → output 0. "
            "Use 8 only if TDP-43 staining was explicitly not performed. "
            "Note: hippocampal sclerosis in the elderly often has associated TDP-43 pathology."
        ),
    )
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

    @field_validator(
        "NPPVASC", "NPPAD", "NPCAD", "NPPLEWY", "NPCLEWY", "NPCVASC", "NPPFTLD"
    )
    @classmethod
    def val_binary_diag(cls, v):
        if v not in {1, 2}:
            raise ValueError(
                f"Primary/contributing diagnosis field must be 1 or 2, got {v}"
            )
        return v


# ---------------------------------------------------------------------------
# Top-level extraction model
# ---------------------------------------------------------------------------

class NeuropathologyExtraction(BaseModel):
    """Structured NACC neuropathology extraction — 43 variables + per-field annotations.

    Priority fields (extract with highest care):
      NPSEX, NPFIX, NPWBRWT, NPGRLA, NPGRHA, NPGRSNH, NPGRLCH,
      NACCAVAS, NPLINF, NPLAC, NPHEM, NPWMR, NPNLOSS, NACCCBD, NPPVASC
    """
    model_config = ConfigDict(extra="forbid")

    specimen_info: SpecimenInfo = Field(
        description="Specimen details — PRIORITY: NPSEX, NPFIX, NPWBRWT",
    )
    gross_findings: GrossFindings = Field(
        description="Gross findings — PRIORITY: NPGRLA, NPGRHA, NPGRSNH, NPGRLCH",
    )
    vascular_pathology: VascularPathology = Field(
        description="Vascular pathology — PRIORITY: NACCAVAS, NPLINF, NPLAC, NPHEM, NPWMR",
    )
    microscopic_findings: MicroscopicFindings = Field(
        description="Microscopic findings — PRIORITY: NPNLOSS",
    )
    ad_pathology: ADPathology = Field(
        description="AD ABC scores: Thal phase / Braak stage / CERAD / ADNC",
    )
    diagnostic_codes: DiagnosticCodes = Field(
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
            "(1) 'confidence': float 0.0-1.0 — certainty this value is correct "
            "(1.0=exact match; 0.8=clearly implied; 0.6=indirect inference; 0.4=ambiguous guess; "
            "below 0.4: prefer null instead of guessing); "
            "(2) 'evidence': string — verbatim phrase or sentence from the report that drove the decision "
            "(required for every non-null extraction; null only when a finding is unambiguously absent); "
            "(3) 'note': string or null — brief reasoning note ONLY when extraction was non-trivial "
            "(ambiguous language, had to choose between codes, conflicting statements); "
            "leave null for straightforward extractions. "
            'Example: {"NPGRHA": {"confidence": 0.6, "evidence": "hippocampus measures 10 mm", "note": "raw measurement only, no explicit grade; coded 9"}}'
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
            (vp.NPLINF is not None and vp.NPLINF.value == 1) or 
            (vp.NPLAC is not None and vp.NPLAC.value == 1)
        ):
            raise ValueError(
                "NACCINF=0 (no infarcts) but NPLINF or NPLAC is 1 (present) — inconsistent."
            )
        # NACCHEM should be 1 if NPHEM=1
        if vp.NACCHEM == 0 and vp.NPHEM is not None and vp.NPHEM.value == 1:
            raise ValueError(
                "NACCHEM=0 (no hemorrhage) but NPHEM=1 (present) — inconsistent."
            )
        return self
    
    @model_validator(mode="after")
    def enforce_null_policy(self) -> "NeuropathologyExtraction":
        nullable = {"NPSEX", "NPFIX", "NPFIXX"}
        for section in [
            self.specimen_info, self.gross_findings, self.vascular_pathology,
            self.microscopic_findings, self.ad_pathology, self.diagnostic_codes
        ]:
            for field_name in section.model_fields:
                if field_name not in nullable and getattr(section, field_name) is None:
                    raise ValueError(
                        f"{field_name} must not be null — use 0, 8, or 9 instead."
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
        "Respond with a single JSON object with EXACTLY these top-level keys:",
        "",
        "  specimen_info, gross_findings, vascular_pathology,",
        "  microscopic_findings, ad_pathology, diagnostic_codes,",
        "  field_annotations, extraction_confidence, extraction_notes",
        "",
        "Each top-level key maps to a nested object containing its variables.",
        "NEVER output variables at the root level — they must always be nested.",
        "",
        "PRIORITY fields — extract these with the highest care:",
        "  NPSEX, NPFIX, NPWBRWT, NPGRLA, NPGRHA, NPGRSNH, NPGRLCH,",
        "  NACCAVAS, NPLINF, NPLAC, NPHEM, NPWMR, NPNLOSS, NACCCBD, NPPVASC",
        "",
        "DERIVED fields — compute from other extracted values, do not re-read the report:",
        "  NACCVASC (from NACCAVAS/NPLINF/NPLAC/NPHEM/NPWMR/NACCARTE),",
        "  NACCINF (from NPLINF+NPLAC), NACCHEM (from NPHEM), NACCLEWY (from NPLBOD)",
        "",
        "NULL POLICY — null is valid for ONLY these three variables:",
        "  NPSEX: null if sex is not stated anywhere in the report.",
        "  NPFIX: null if fixative is not mentioned in the report.",
        "  NPFIXX: null always EXCEPT when NPFIX=7 (Other), in which case populate it.",
        "For every other variable you MUST output an explicit numeric code — never null.",
        "  If a finding was assessed and absent → 0",
        "  If a structure was explicitly not examined or stain not performed → 8",
        "  If examined or mentioned but severity/result cannot be determined → 9",
        "",
    ]
    for name, info in model.model_fields.items():
        raw_ann = info.annotation
        inner, _ = unwrap_optional(raw_ann)
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            lines.append(f"[{name}]")
            for sub_name, sub_info in inner.model_fields.items():
                lines.extend(describe_field(sub_name, sub_info, indent=1))
            lines.append("")
        else:
            lines.extend(describe_field(name, info, indent=0))
    lines += [
        "",
        "Do not invent information. Extract only what is explicitly stated or clearly implied.",
        "All integer codes must be exact values from the allowed set in each description.",
    ]
    return "\n".join(lines)