"""extract.py — Stage 2: Structured residual variable extraction.

Extracts 49 residual neuropathology variables from each autopsy report.
These variables are not captured by the 199 NACC NP Form variables
and were identified through the Stage 1A/1B discovery and clustering pipeline.

Output format mirrors the primary extraction pipeline:
    {block_name: {variable: value, ...}, ...,
     field_annotations: {var: {confidence, evidence, note}},
     extraction_confidence: str, extraction_notes: str}

Runs 5 seeds per report (same as primary extraction) for majority voting.
Output: {report_id}_seed{seed}.residual.json per report per seed.

Usage (single report — SLURM loop):
    python extract.py -i report.pdf --vllm-url http://localhost:8000

Usage (full directory — local testing):
    python extract.py --reports-dir /path/to/reports --vllm-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from json_repair import repair_json
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from primary_extraction.main import load_report, strip_json_fences, _sanitize_for_json

logger = logging.getLogger("residual")

# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

CONT_UNKNOWN = 9999.0   # continuous variable not reported in the text
ORD_UNKNOWN  = 9        # ordinal variable not mentioned in the report

# ---------------------------------------------------------------------------
# Block names and metadata keys — used by normaliser
# ---------------------------------------------------------------------------

RESIDUAL_BLOCKS = {
    "bilateral_hemibrain_weights",
    "bilateral_cerebral_weights",
    "bilateral_cerebellar_weights",
    "bilateral_brainstem_weights",
    "corpus_callosum_measurements",
    "circle_of_willis_diameters",
    "structural_measurements",
    "structural_severity_markers",
    "regional_pathology_severity",
}

METADATA_KEYS = {"field_annotations", "extraction_confidence", "extraction_notes"}

# ---------------------------------------------------------------------------
# FieldAnnotation — mirrors primary pipeline (schema3.py)
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
            "below 0.4: leave field at its sentinel value."
        ),
    )
    evidence: Optional[str] = Field(
        None,
        description=(
            "Verbatim phrase or sentence from the report that supports this value. "
            "Must be a single string — never a list or array. "
            "Required for every non-sentinel extraction."
        ),
    )
    note: Optional[str] = Field(
        None,
        description=(
            "Reasoning note. Write ONLY when extraction was non-trivial "
            "(indirect language, ambiguous measurement, multiple values present). "
            "Leave null for clear-cut extractions."
        ),
    )


# ---------------------------------------------------------------------------
# Block sub-models — 9 blocks, 49 variables total
# ---------------------------------------------------------------------------

class BilateralHemibrainWeights(BaseModel):
    """Right and left hemibrain weights, fresh and fixed. (4 variables)

    FIXATION DETECTION — identify whether a weight is fresh or fixed:
      FRESH: appears before any fixation statement; OR described as 'at time of autopsy',
             'saved frozen at -70°C / -80°C', 'fresh weight', 'immediately weighed'.
      FIXED: appears after 'following formalin fixation', 'after fixation', 'formalin-fixed',
             'after X days in formalin', or any fixation duration statement.

    FULL-TEXT SCAN — weight values can appear ANYWHERE in the gross description,
    including mid-paragraph in the hemisection or fixation sections. Do not stop
    scanning after the first few paragraphs. Read the entire gross description.

    LATERALITY — extract right and left separately. Never assign the same measurement
    to both sides unless the report explicitly states the same value for both hemispheres.
    If only a combined total weight is given, use 9999.0 for both individual side variables.
    """
    model_config = ConfigDict(extra="ignore")

    hemibrain_weight_right_fresh_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Right hemibrain weight in grams measured FRESH (pre-fixation). "
            "The hemibrain includes right cerebral hemisphere + right cerebellar hemisphere + "
            "right brainstem half, all weighed together as one unit. "
            "Typical range 400-800 g. Range 0.0-2000.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'the right hemibrain weighs X grams', 'right half weighs X grams'. "
            "FRESH indicator: value appears before any 'following formalin fixation' statement, "
            "OR the brain is described as being frozen/stored immediately after. "
            "Do NOT use this field for the left hemibrain or for the whole brain weight."
        ),
    )
    hemibrain_weight_left_fresh_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Left hemibrain weight in grams measured FRESH (pre-fixation). "
            "Typical range 400-800 g. Range 0.0-2000.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'the left hemibrain weighs X grams'. "
            "FRESH indicator: value appears before any fixation statement."
        ),
    )
    hemibrain_weight_right_fixed_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Right hemibrain weight in grams measured after FORMALIN FIXATION. "
            "Range 0.0-2000.0. Use 9999.0 if not reported. "
            "FIXED indicator: value appears after 'following formalin fixation (X days)' or similar. "
            "COMMON PHRASING: 'following formalin fixation... the right hemibrain weighs X grams'. "
            "Many reports fix only the left side; right may remain 9999.0."
        ),
    )
    hemibrain_weight_left_fixed_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Left hemibrain weight in grams measured after FORMALIN FIXATION. "
            "Range 0.0-2000.0. Use 9999.0 if not reported. "
            "FIXED indicator: appears after fixation duration statement. "
            "COMMON PHRASING: 'following formalin fixation... the left hemibrain weighs X grams'. "
            "This is the most commonly reported fixed hemibrain weight."
        ),
    )


class BilateralCerebralWeights(BaseModel):
    """Right and left cerebral hemisphere weights, fresh and fixed. (4 variables)

    IMPORTANT: These are for the ISOLATED cerebral hemisphere ONLY, separate from
    cerebellum and brainstem. Do not confuse with hemibrain weights.

    FULL-TEXT SCAN: Cerebral hemisphere weights often appear mid-paragraph in the section
    describing hemisection. Scan the ENTIRE gross description, including paragraphs
    describing what was done to each side after separation.
    """
    model_config = ConfigDict(extra="ignore")

    cerebral_weight_right_fresh_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Right cerebral hemisphere ONLY (excluding cerebellum and brainstem) "
            "weight in grams measured FRESH. "
            "Typical range 350-700 g. Range 0.0-2000.0. Use 9999.0 if not reported separately. "
            "COMMON PHRASING: 'the right cerebral hemisphere weighs X grams'. "
            "Often appears in the fresh section after the brainstem and cerebellum are separated. "
            "FRESH indicator: value reported before fixation statement OR in the section "
            "describing frozen/fresh tissue processing. "
            "Use 9999.0 if only the hemibrain weight (not cerebral alone) is given for this side."
        ),
    )
    cerebral_weight_left_fresh_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Left cerebral hemisphere ONLY weight in grams measured FRESH. "
            "Range 0.0-2000.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'the left cerebral hemisphere weighs X grams' in pre-fixation section. "
            "Many reports weigh the left hemisphere only after fixation; this field may be 9999.0."
        ),
    )
    cerebral_weight_right_fixed_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Right cerebral hemisphere ONLY weight in grams measured after FORMALIN FIXATION. "
            "Range 0.0-2000.0. Use 9999.0 if not reported. "
            "FIXED indicator: appears after 'following formalin fixation' statement. "
            "COMMON PHRASING: 'following formalin fixation... the right cerebral hemisphere weighs X'. "
            "Note: the right hemisphere is often frozen fresh without fixation, making this 9999.0."
        ),
    )
    cerebral_weight_left_fixed_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Left cerebral hemisphere ONLY weight in grams measured after FORMALIN FIXATION. "
            "Range 0.0-2000.0. Use 9999.0 if not reported. "
            "FIXED indicator: appears after fixation statement in the left hemisphere description. "
            "COMMON PHRASING: 'the left cerebral hemisphere weighs X grams' in the post-fixation section. "
            "This is commonly reported since the left side is typically the fixed/sectioned side. "
            "IMPORTANT: This value often appears mid-paragraph embedded in a description of the "
            "hemisphere's cut surface. Read the entire left hemisphere description paragraph."
        ),
    )


class BilateralCerebellarWeights(BaseModel):
    """Right and left cerebellar hemisphere weights, fresh and fixed. (4 variables)"""
    model_config = ConfigDict(extra="ignore")

    cerebellar_weight_right_fresh_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Right cerebellar hemisphere weight in grams measured FRESH. "
            "Typical range 40-130 g. Range 0.0-200.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'the right cerebellar hemisphere weighs X grams', "
            "'right half of the cerebellum weighs X grams', 'right cerebellum weighs X'. "
            "IMPORTANT: If the report says 'a portion of the right half of the cerebellum weighs X' — "
            "this is a PARTIAL section (not the full hemisphere); still extract the value. "
            "FRESH indicator: appears in the pre-fixation section, alongside frozen tissue description."
        ),
    )
    cerebellar_weight_left_fresh_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Left cerebellar hemisphere weight in grams measured FRESH. "
            "Range 0.0-200.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'the left cerebellar hemisphere weighs X grams'. "
            "IMPORTANT: In many reports the left cerebellar weight appears LATER in the text, "
            "after the fixed section description — check whether this appears BEFORE or AFTER the "
            "fixation statement to determine fresh vs fixed. "
            "Scan the ENTIRE gross description including mid-paragraph mentions."
        ),
    )
    cerebellar_weight_right_fixed_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Right cerebellar hemisphere weight in grams measured after FORMALIN FIXATION. "
            "Range 0.0-200.0. Use 9999.0 if not reported. "
            "FIXED indicator: appears after fixation duration statement. "
            "CAUTION: A fixed cerebellar weight that is dramatically lower than the fresh weight "
            "for the same side (e.g., fresh 63g vs fixed 12g) may indicate a partial section — "
            "extract the value as reported but note in field_annotations."
        ),
    )
    cerebellar_weight_left_fixed_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Left cerebellar hemisphere weight in grams measured after FORMALIN FIXATION. "
            "Range 0.0-200.0. Use 9999.0 if not reported. "
            "FIXED indicator: appears after 'following formalin fixation' statement. "
            "COMMON PHRASING: 'the left half of the cerebellum weighs X grams' "
            "in the fixed section."
        ),
    )


class BilateralBrainstemWeights(BaseModel):
    """Right and left brainstem half weights, fresh and fixed. (4 variables)

    HEMISECTION EXCEPTION: Some reports state that the brainstem was NOT hemisected
    (e.g., 'with the exception of the brainstem, the brain is hemisected along the
    midsagittal plane'). In this case, ANY brainstem weight is for the WHOLE brainstem,
    NOT a half. Use 9999.0 for BOTH right and left variables — do not assign the
    whole brainstem weight to either side.

    LATERALITY: If the report gives brainstem weight without left/right qualifier
    AND the brainstem WAS hemisected, use context — reports typically describe
    right side first (in the fresh section) and left side in the fixed/left hemisphere section.
    """
    model_config = ConfigDict(extra="ignore")

    brainstem_weight_right_fresh_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Right HALF of brainstem weight in grams measured FRESH. "
            "Typical range 8-25 g. Range 0.0-100.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'the right half of the brainstem weighs X grams'. "
            "HEMISECTION RULE: Only extract if the brainstem WAS hemisected. "
            "If the report states the brainstem was excepted from hemisection, use 9999.0. "
            "FRESH indicator: appears in the pre-fixation right hemisphere description. "
            "LATERALITY: 'right half' or 'right half of brainstem' explicitly stated."
        ),
    )
    brainstem_weight_left_fresh_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Left HALF of brainstem weight in grams measured FRESH. "
            "Typical range 8-25 g. Range 0.0-100.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'the left half of brainstem weighs X grams', "
            "'the left half of the brainstem weighs X grams (including upper cervical cord)'. "
            "IMPORTANT: Some reports include a segment of upper cervical cord in the left brainstem "
            "half — extract the stated weight even if it includes cord."
        ),
    )
    brainstem_weight_right_fixed_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Right HALF of brainstem weight in grams measured after FORMALIN FIXATION. "
            "Range 0.0-100.0. Use 9999.0 if not reported. "
            "Fixed brainstem half weights are rarely reported — most reports only give fresh values. "
            "FIXED indicator: appears after fixation duration statement."
        ),
    )
    brainstem_weight_left_fixed_g: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Left HALF of brainstem weight in grams measured after FORMALIN FIXATION. "
            "Range 0.0-100.0. Use 9999.0 if not reported. "
            "FIXED indicator: appears after fixation duration statement. "
            "COMMON PHRASING: 'the left half of brainstem weighs X grams' in the fixed section. "
            "HEMISECTION RULE: If the brainstem was not hemisected, use 9999.0."
        ),
    )


class CorpusCallosumMeasurements(BaseModel):
    """Corpus callosum segment thicknesses in millimeters. (5 variables)

    SCAN LOCATION: Corpus callosum measurements appear in the fixed hemisphere section,
    typically after the fixation weight statement and before or during the coronal section
    description. They are often in a structured sentence listing all segments.

    COMMON REPORT FORMAT:
    'Its thickness varies: genu X mm; body anteriorly Y mm; mid-region Z mm;
    posteriorly W mm; splenium V mm.'
    OR: 'the body is X mm anteriorly, Y mm in the mid-region, and Z mm posteriorly'

    All five segments MUST be extracted if present in the report.
    Do not stop after extracting the genu — read the entire callosal measurement.
    """
    model_config = ConfigDict(extra="ignore")

    corpus_callosum_genu_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Corpus callosum GENU (anterior knee) thickness in millimeters. "
            "Typical range 8-20 mm. Range 0.0-30.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'at the level of the genu, it is X mm', 'genu X mm', "
            "'the genu measures X mm in thickness'. "
            "Also called 'head of corpus callosum' in some reports — 'head' = genu/anterior part."
        ),
    )
    corpus_callosum_body_anterior_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Corpus callosum ANTERIOR BODY thickness in millimeters. "
            "Typical range 3-12 mm. Range 0.0-30.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'the body is X mm anteriorly', 'body anteriorly X mm', "
            "'anterior body X mm', 'body (anterior) X mm'. "
            "This is the first body measurement — do not confuse with mid or posterior body."
        ),
    )
    corpus_callosum_body_mid_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Corpus callosum MID-BODY (middle portion) thickness in millimeters. "
            "Typical range 3-12 mm. Range 0.0-30.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'in the mid-region X mm', 'mid-region X mm', "
            "'midportion X mm', 'body (mid) X mm', 'middle portion X mm'. "
            "This is the second/middle body measurement."
        ),
    )
    corpus_callosum_body_posterior_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Corpus callosum POSTERIOR BODY thickness in millimeters. "
            "Typical range 1-12 mm. Range 0.0-30.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'posteriorly X mm', 'body posteriorly X mm', "
            "'posterior X mm (before splenium)', 'body (posterior) X mm'. "
            "This is the third body measurement, listed just before the splenium."
        ),
    )
    corpus_callosum_splenium_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Corpus callosum SPLENIUM (posterior knob) thickness in millimeters. "
            "Typical range 5-18 mm. Range 0.0-30.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'at the level of the splenium, it is X mm', 'splenium X mm'. "
            "Always the LAST measurement in the callosal sequence — listed after posterior body."
        ),
    )


class CircleOfWillisDiameters(BaseModel):
    """Circle of Willis vessel diameters in millimeters. (14 variables)

    CRITICAL SCANNING RULE: Reports present ALL CoW vessel diameters in a single structured
    paragraph or list. You MUST read the ENTIRE vessel list before filling any variable.
    Do NOT stop reading after the first 4-5 vessels — ACA, PCom, PCA, and ACom often
    appear at the END of the list and are commonly missed.

    BILATERAL LISTING WITHOUT EXPLICIT LATERALITY:
    When a report lists "posterior cerebral arteries, X mm" (plural, no L/R qualifier),
    apply the SAME value to BOTH cow_pca_right_mm AND cow_pca_left_mm.
    When a report lists "anterior cerebral arteries, X mm" (plural), apply to BOTH
    cow_aca_right_mm AND cow_aca_left_mm. Similarly for other bilateral vessels.

    SEGMENT SPECIFICATIONS:
    Some reports give P1 and P2 segment values for the PCA (e.g., 'left PCA 1 mm (P1) and
    2 mm (P2)'). Use the P1 (proximal/first) segment value for the cow_pca_* variable.

    VESSEL NAMING VARIANTS commonly seen in reports:
    - ICA: 'internal carotid artery', 'terminal segment of the internal carotid', 'ICA'
    - ACA: 'anterior cerebral artery', 'A1 segment of the anterior cerebral artery'
    - MCA: 'middle cerebral artery'
    - PCA: 'posterior cerebral artery', 'P1 segment', 'P2 segment'
    - PCom: 'posterior communicating artery'
    - ACom: 'anterior communicating artery'
    - Vertebral: 'vertebral artery'
    - Basilar: 'basilar artery'

    ABSENT / NON-PATENT VESSELS:
    'cannot be identified', 'not visualized', 'absent', 'non-patent', 'thread-like'
    → use 0.0 (not 9999.0). These are explicitly absent, not simply unreported.

    LESS-THAN VALUES: '<1 mm' → use 0.5; '<2 mm' → use 1.5. Do not skip these — they
    are present and measurably small.

    ATHEROMATOUS PLAQUE SIZE is NOT vessel diameter. 'A plaque measuring 5 mm' describes
    plaque length, not lumen diameter. Extract diameter only from explicit diameter statements.
    """
    model_config = ConfigDict(extra="ignore")

    cow_basilar_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "BASILAR ARTERY diameter in millimeters. Midline vessel — no laterality. "
            "Range 0.0-15.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'the basilar artery measures X mm', 'basilar artery X mm'. "
            "WARNING: Do NOT confuse atheromatous plaque size with vessel diameter. "
            "'The basilar artery has a plaque measuring 5 mm' → plaque size, NOT diameter → 9999.0."
        ),
    )
    cow_vertebral_right_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "RIGHT VERTEBRAL ARTERY diameter in millimeters. "
            "Range 0.0-15.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'right vertebral artery X mm', 'vertebral arteries: right X mm'. "
            "When 'vertebral arteries X mm' is listed without L/R, apply same value to both sides."
        ),
    )
    cow_vertebral_left_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "LEFT VERTEBRAL ARTERY diameter in millimeters. "
            "Range 0.0-15.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'left vertebral artery X mm'. "
            "When 'vertebral arteries X mm' without L/R, apply same value to both sides."
        ),
    )
    cow_ica_right_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "RIGHT INTERNAL CAROTID ARTERY (ICA) diameter in millimeters. "
            "Range 0.0-15.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'right internal carotid artery X mm', 'right ICA X mm', "
            "'internal carotid arteries X mm' (bilateral listing — apply same value to both). "
            "Also called 'terminal segment of the internal carotid artery' in some reports."
        ),
    )
    cow_ica_left_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "LEFT INTERNAL CAROTID ARTERY (ICA) diameter in millimeters. "
            "Range 0.0-15.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'left internal carotid artery X mm', "
            "'the internal carotid arteries measure X mm on the left and Y mm on the right' "
            "(extract left value here, right value in cow_ica_right_mm)."
        ),
    )
    cow_mca_right_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "RIGHT MIDDLE CEREBRAL ARTERY (MCA) diameter in millimeters. "
            "Range 0.0-15.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'right middle cerebral artery X mm', "
            "'the right and left middle cerebral arteries are both X mm' → same value both sides. "
            "'middle cerebral arteries X mm' (bilateral) → apply to both sides."
        ),
    )
    cow_mca_left_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "LEFT MIDDLE CEREBRAL ARTERY (MCA) diameter in millimeters. "
            "Range 0.0-15.0. Use 9999.0 if not reported."
        ),
    )
    cow_aca_right_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "RIGHT ANTERIOR CEREBRAL ARTERY (ACA) diameter in millimeters. "
            "Range 0.0-15.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'right anterior cerebral artery X mm', 'right ACA X mm', "
            "'A1 segment of the right anterior cerebral artery X mm'. "
            "BILATERAL LISTING: 'anterior cerebral arteries X mm' or 'anterior cerebral arteries, "
            "X mm' (without L/R) → apply the SAME value to BOTH cow_aca_right_mm and cow_aca_left_mm. "
            "ACA typically appears near the END of the vessel list — do not stop reading early."
        ),
    )
    cow_aca_left_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "LEFT ANTERIOR CEREBRAL ARTERY (ACA) diameter in millimeters. "
            "Range 0.0-15.0. Use 9999.0 if not reported. "
            "BILATERAL LISTING: 'anterior cerebral arteries X mm' → apply to BOTH sides. "
            "When separately stated: 'A1 segment of the left ACA X mm' → use that value."
        ),
    )
    cow_pca_right_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "RIGHT POSTERIOR CEREBRAL ARTERY (PCA) diameter in millimeters. "
            "Range 0.0-15.0. Use 9999.0 if not reported. "
            "BILATERAL LISTING: 'posterior cerebral arteries X mm' (no L/R) → apply to BOTH sides. "
            "P1/P2 SEGMENTS: When both P1 and P2 measurements are given "
            "(e.g., 'right PCA: 1 mm (P1) and 2 mm (P2)'), use the P1 segment value. "
            "THIN VALUES: 'P1 segments are thin and measure 0.5 mm' → extract 0.5, not 9999.0. "
            "Small but measurable values (even <1 mm) must be extracted — do not treat as absent."
        ),
    )
    cow_pca_left_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "LEFT POSTERIOR CEREBRAL ARTERY (PCA) diameter in millimeters. "
            "Range 0.0-15.0. Use 9999.0 if not reported. "
            "BILATERAL LISTING: 'posterior cerebral arteries X mm' → apply to BOTH sides. "
            "P1/P2 SEGMENTS: 'both P1 segments measure 0.5 mm' → cow_pca_left = 0.5. "
            "PCA typically appears mid-list — read past MCA to find it."
        ),
    )
    cow_pcom_right_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "RIGHT POSTERIOR COMMUNICATING ARTERY (PCom) diameter in millimeters. "
            "Range 0.0-15.0. Use 9999.0 if not reported. "
            "Use 0.0 if explicitly described as absent or non-patent. "
            "COMMON PHRASING: 'right posterior communicating artery X mm', "
            "'posterior communicating arteries X mm on both sides' → apply to both. "
            "ABSENT: 'right PCom cannot be identified', 'right PCom absent' → 0.0."
        ),
    )
    cow_pcom_left_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "LEFT POSTERIOR COMMUNICATING ARTERY (PCom) diameter in millimeters. "
            "Range 0.0-15.0. Use 9999.0 if not reported. "
            "Use 0.0 if explicitly described as absent or non-patent. "
            "ABSENT: 'left posterior communicating artery, cannot be identified' → 0.0."
        ),
    )
    cow_acom_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "ANTERIOR COMMUNICATING ARTERY (ACom) diameter in millimeters. "
            "Midline vessel — no laterality. Range 0.0-15.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'the diameter of the anterior communicating artery is X mm', "
            "'anterior communicating artery X mm', 'anterior communicating artery, X mm'. "
            "ACom appears at the END of the vessel list — always read to the end before stopping. "
            "Do not confuse with 'anterior cerebral artery' (ACA) — these are different vessels."
        ),
    )


class StructuralMeasurements(BaseModel):
    """Structural morphometric measurements. (1 variable)"""
    model_config = ConfigDict(extra="ignore")

    caudate_nucleus_head_width_mm: float = Field(
        default=CONT_UNKNOWN,
        description=(
            "Caudate nucleus HEAD width at its widest point in millimeters. "
            "Range 0.0-30.0. Use 9999.0 if not reported. "
            "COMMON PHRASING: 'the head of the caudate nucleus... measures X mm at its widest', "
            "'the head of the caudate nucleus is X mm at its widest point', "
            "'head of caudate nucleus measures X mm medio-laterally'. "
            "TYPICAL VALUES: Normal ~15-20 mm; atrophic ~4-8 mm. "
            "Appears in the coronal section description of the cerebral hemisphere. "
            "Do NOT extract the width of the corpus callosum body — that is a different variable."
        ),
    )


class StructuralSeverityMarkers(BaseModel):
    """Ordinal severity scales for gross structural findings. (8 variables)

    CRITICAL RULES FOR ALL ORDINAL VARIABLES:
    CODE DISAMBIGUATION — these codes have distinct meanings:
      0 = Finding is ABSENT or NONE: report explicitly states no atrophy, normal size,
          OR structure was examined and found normal ('unremarkable', 'not observed').
      1 = MILD: 'mild', 'slight', 'minimal', 'some', 'mild reduction in bulk', 'small'.
      2 = MODERATE: 'moderate', 'moderately', 'important reduction', 'prominent'.
      3 = SEVERE: 'severe', 'marked', 'marked atrophy', 'very atrophic', 'most severe'.
      8 = NOT ASSESSED: use ONLY when the report explicitly says finding could not be evaluated.
          NEVER use 8 for gross atrophy — atrophy severity is always assessable on gross exam.
      9 = NOT MENTIONED: finding not discussed anywhere in the ENTIRE report.
          Use 9 ONLY if the structure/finding is NEVER mentioned in gross OR microscopic section.

    KEY RULE: 9 means ABSENT FROM REPORT TEXT.
    Code 0 means PRESENT AND NORMAL. Do not use 9 when you simply did not observe it
    in one section — check the entire report including Final Diagnoses.

    SEVERITY MAPPING TABLE:
      'some atrophy' / 'slight reduction' → 1
      'mild atrophy' / 'mild reduction in bulk' → 1
      'mild to moderate' → 2
      'moderate' / 'moderately atrophic' → 2
      'important reduction in bulk' → 2
      'moderate to severe' → 3
      'severe' / 'marked' / 'very atrophic' / 'most severe' → 3
    """
    model_config = ConfigDict(extra="ignore")

    lateral_ventricle_enlargement_severity: int = Field(
        default=ORD_UNKNOWN,
        description=(
            "Severity of lateral ventricular enlargement on gross examination. "
            "0=None/normal, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Not mentioned. "
            "SCAN LOCATION: Look in the GROSS DESCRIPTION section — often appears mid-paragraph "
            "in the fixed hemisphere coronal section description, not as a standalone sentence. "
            "SEVERITY MAPPING: "
            "'enlargement' alone (no qualifier) → 2; "
            "'mild enlargement' → 1; "
            "'moderate enlargement' → 2; "
            "'marked enlargement' / 'significant enlargement' → 3; "
            "'severe enlargement' → 3; "
            "'enlarged with anterocaudate extension' → 3 (anterocaudate extension indicates severe dilation); "
            "'dilated' alone → 2; "
            "'most prominent at the level of the occipital horn' = additional description of existing "
            "enlargement, not a severity modifier. "
            "IMPORTANT: Check BOTH the right hemisphere section (fresh) AND the left hemisphere section "
            "(fixed/post-fixation) — reports often describe only the fixed hemisphere's ventricle. "
            "Code the maximum severity found across both descriptions."
        ),
    )
    amygdala_atrophy_severity: int = Field(
        default=ORD_UNKNOWN,
        description=(
            "Severity of amygdala atrophy on gross examination. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Not mentioned. "
            "SCAN LOCATION: Gross description, typically in the coronal section description. "
            "SEVERITY MAPPING: "
            "'amygdala is unremarkable' → 0; "
            "'mild atrophy of the amygdala' → 1; "
            "'mild to moderate atrophy of the amygdala' → 2; "
            "'the amygdala appears atrophic' (no qualifier) → 2; "
            "'very atrophic' / 'severely atrophic' → 3. "
            "MEASUREMENTS: Some reports give amygdala dimensions ('measuring X mm') — "
            "use these to infer severity only if no explicit severity language is present. "
            "Amygdala normally 15-20 mm; <12 mm = mild, <10 mm = moderate, <8 mm = severe."
        ),
    )
    dilated_perivascular_spaces: int = Field(
        default=ORD_UNKNOWN,
        description=(
            "Severity of perivascular space dilation in hemispheric white matter and basal ganglia. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Not mentioned. "
            "SCAN LOCATION: Gross description AND Final Diagnoses. "
            "COMMON TERMS: 'perivascular spaces', 'Virchow-Robin spaces', 'enlarged perivascular spaces'. "
            "SEVERITY MAPPING: "
            "'mild enlargement of perivascular spaces' → 1; "
            "'enlargement of perivascular spaces' (no qualifier) → 2; "
            "'dilatation of perivascular spaces, diffuse' → 2; "
            "'prominent / marked dilation' → 3. "
            "If mentioned in Final Diagnoses only without explicit gross description → still code from "
            "the severity language used ('diffuse' = 2). "
            "NOT mentioned anywhere → 9."
        ),
    )
    cerebellar_atrophy_severity: int = Field(
        default=ORD_UNKNOWN,
        description=(
            "Severity of cerebellar cortical and/or vermis atrophy on gross examination. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Not mentioned. "
            "SCAN LOCATION: Gross description of the cerebellum. "
            "CRITICAL DISTINCTION: "
            "'The cerebellum does not show any lesions on the external surfaces' or "
            "'no focal lesions noted in the cerebellum' → this means NO FOCAL LESIONS, "
            "NOT necessarily no atrophy. Do NOT code 0 based on 'no lesions' alone. "
            "Code 0 ONLY if the report explicitly says the cerebellum is NORMAL in bulk, "
            "size, or shows NO atrophy. "
            "Code 9 if cerebellar cortical atrophy is simply not mentioned. "
            "SEVERITY MAPPING: "
            "'mild atrophy of the anterior part of the cerebellum' → 1; "
            "'some reduction' / 'mild reduction' → 1; "
            "'moderate atrophy' → 2; "
            "'severe atrophy' → 3. "
            "NOTE: Dentate nucleus atrophy is a SEPARATE variable — do not confuse cerebellar "
            "cortical atrophy with dentate nucleus atrophy."
        ),
    )
    globus_pallidus_neuronal_loss_severity: int = Field(
        default=ORD_UNKNOWN,
        description=(
            "Severity of neuronal loss in the globus pallidus on microscopic examination. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Not mentioned. "
            "SCAN LOCATION: FINAL DIAGNOSES section (neuronal loss enumeration) AND any "
            "microscopic description. This variable is typically NOT coded from gross alone. "
            "COMMON FORMAT in Final Diagnoses: "
            "'Neuronal loss and gliosis: Severe: globus pallidus, thalamus...' → code 3 for GP. "
            "'Neuronal loss, moderate, globus pallidus' → code 2. "
            "SEVERITY MAPPING: "
            "'severe' neuronal loss in globus pallidus → 3; "
            "'moderate to severe' → 3; "
            "'moderate' → 2; "
            "'mild to moderate' → 2; "
            "'mild' → 1; "
            "Globus pallidus NOT listed in neuronal loss section → 9. "
            "IMPORTANT: Always check the Final Diagnoses section — neuronal loss is enumerated "
            "by region there. Some seeds miss this by scanning only the gross description."
        ),
    )
    basal_ganglia_atrophy: int = Field(
        default=ORD_UNKNOWN,
        description=(
            "Severity of basal ganglia atrophy on gross examination. "
            "Basal ganglia = caudate nucleus + putamen + globus pallidus. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=NEVER USE (gross atrophy is always assessable), "
            "9=Not mentioned anywhere. "
            "SCAN LOCATION: Fixed hemisphere coronal sections (after fixation weight statement) "
            "AND external examination paragraph — both may contain basal ganglia atrophy descriptions. "
            "CODE 8 IS FORBIDDEN: Gross atrophy is always assessable in these reports. "
            "Never code 8 for this variable. If uncertain between 1 and 2, code the lower value. "
            "SEVERITY MAPPING: "
            "'some atrophy' of caudate or putamen → 1; "
            "'mild atrophy' → 1; "
            "'mild reduction in bulk' → 1; "
            "'mild to moderate atrophy' → 2; "
            "'moderate atrophy' / 'moderate reduction in bulk' → 2; "
            "'the head is atrophic' without qualifier → 2; "
            "'severe atrophy' / 'markedly atrophic' → 3; "
            "'head of caudate is flat' / 'very atrophic' → 3. "
            "BILATERAL SCORING: If left and right sides differ, code the MAXIMUM severity found. "
            "If basal ganglia structures are described as normal or unremarkable → 0. "
            "If basal ganglia simply not mentioned in the gross description → 9."
        ),
    )
    thalamic_degeneration_severity: int = Field(
        default=ORD_UNKNOWN,
        description=(
            "Severity of thalamic neuronal loss and gliosis. "
            "Includes BOTH gross atrophy (from gross description) AND microscopic neuronal loss "
            "(from Final Diagnoses / microscopic section). "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Not mentioned. "
            "SCAN LOCATION: Check ALL of: (1) gross description for thalamic size/bulk, "
            "(2) Final Diagnoses neuronal loss enumeration, (3) any microscopic description. "
            "COMMON FORMAT in Final Diagnoses: "
            "'Neuronal loss and gliosis: Severe: thalamus...' → code 3. "
            "'Neuronal loss, moderate, thalamus' → code 2. "
            "SEVERITY MAPPING: "
            "'thalamus appears unremarkable' (gross) → 0 unless microscopic section says otherwise; "
            "'mild neuronal loss in thalamus' → 1; "
            "'moderate neuronal loss' → 2; "
            "'severe neuronal loss in thalamus' → 3; "
            "'thalamus is relatively better preserved' → 0 or 1. "
            "IMPORTANT: Thalamic neuronal loss is ALWAYS in the Final Diagnoses section in these "
            "reports. Always scan that section before coding 9."
        ),
    )
    brainstem_atrophy_severity: int = Field(
        default=ORD_UNKNOWN,
        description=(
            "Severity of brainstem atrophy on gross or microscopic examination. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Not mentioned. "
            "SCAN LOCATION: Gross description (pons and midbrain atrophy) AND Final Diagnoses. "
            "COMMON PHRASING: "
            "'the pons is mildly atrophic' → 1; "
            "'mild to moderate reduction in bulk of the midbrain and pons' → 2; "
            "'the midbrain appears atrophic' (no qualifier) → 2; "
            "'the midbrain appears slightly atrophic' → 1; "
            "'severe atrophy of the brainstem' → 3; "
            "'reduced in bulk' without qualifier → 2. "
            "DIMENSIONS: Some reports give pons dimensions ('pons measures X mm transversally '). "
            "Small pons (<25mm) → 2-3; borderline small (25-30mm) → 1. "
            "If brainstem is described as unremarkable or normal → 0."
        ),
    )


class RegionalPathologySeverity(BaseModel):
    """Ordinal severity scales for regional neuropathological findings. (5 variables)

    CRITICAL RULE FOR THIS BLOCK: All five variables are coded PRIMARILY from the
    FINAL DIAGNOSES section of the report, not the gross description.
    Reports enumerate neuropathological findings by name and severity in the Final Diagnoses.
    Always scan the Final Diagnoses before coding any variable in this block as 9.
    """
    model_config = ConfigDict(extra="ignore")

    purkinje_cell_loss_severity: int = Field(
        default=ORD_UNKNOWN,
        description=(
            "Severity of Purkinje cell loss in cerebellar cortex on microscopic examination. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Not mentioned. "
            "SCAN LOCATION: Final Diagnoses (neuronal loss section for cerebellar cortex) "
            "AND any microscopic description of cerebellum. "
            "COMMON FORMAT: "
            "'cerebellar cortex (molecular, granular, and Purkinje cell layers): moderate neuronal loss' "
            "→ code 2 for Purkinje cell loss. "
            "'few atrophic neurons in Purkinje cell layer' → code 1. "
            "'Purkinje cell loss, mild' → code 1. "
            "NOT mentioned in either Final Diagnoses or microscopic section → 9. "
            "IMPORTANT: Look for 'cerebellar cortex' in the neuronal loss enumeration — "
            "when cerebellar cortex is listed, Purkinje cells are implied to be affected."
        ),
    )
    dentate_nucleus_atrophy: int = Field(
        default=ORD_UNKNOWN,
        description=(
            "Severity of cerebellar dentate nucleus atrophy on gross or microscopic examination. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Not mentioned. "
            "SCAN LOCATION: Gross description of cerebellum AND microscopic/Final Diagnoses. "
            "COMMON GROSS PHRASING: "
            "'the dentate nucleus appears atrophic' → code 2 (atrophic without qualifier = moderate); "
            "'some atrophy of the dentate nucleus' → code 1; "
            "'mild atrophy of the dentate nucleus' → code 1; "
            "'thinning of the dentate nucleus' → code 1; "
            "'reduction in the thickness of the dentate nucleus' → code 1. "
            "MICROSCOPIC: 'dentate nucleus: moderate neuronal loss' → code 2. "
            "IMPORTANT: Do NOT code 0 (absent) unless the report explicitly states the dentate "
            "nucleus is NORMAL or shows NO atrophy. If not mentioned → 9. "
            "Do NOT code 0 based on 'no focal lesions in cerebellum' — that refers to infarcts/lesions."
        ),
    )
    artag_severity: int = Field(
        default=ORD_UNKNOWN,
        description=(
            "Severity of aging-related tau astrogliopathy (ARTAG) in subpial, perivascular, "
            "and subependymal regions on tau immunohistochemistry. "
            "0=None, 1=Mild, 2=Moderate, 3=Severe, 8=Not assessed, 9=Not mentioned. "
            "SCAN LOCATION: Final Diagnoses (tau proteinopathy section). "
            "COMMON PHRASING: "
            "'Aging-related tau astrogliopathy. Focal: subpial' → code 1; "
            "'ARTAG, mild' → code 1; "
            "'tau-immunoreactive astrocytes in subpial, perivascular regions, moderate' → code 2. "
            "IMPORTANT: ARTAG is only identified on tau immunostaining and MUST be explicitly "
            "mentioned by name or described as tau-immunoreactive astrocytes in specific locations. "
            "Do NOT infer ARTAG from general tau pathology. "
            "Most reports will have ARTAG = 9 (not mentioned). "
            "Code 0 only if a report explicitly states ARTAG is absent after tau immunostaining."
        ),
    )
    gvd_hippocampus_severity: int = Field(
        default=ORD_UNKNOWN,
        description=(
            "Severity of granulovacuolar degeneration (GVD) in hippocampal formation "
            "(Ammon's horn and/or subiculum) on microscopic examination. "
            "0=None, 1=Mild (occasional/focal), 2=Moderate, 3=Severe. "
            "8=Not assessed, 9=Not mentioned. "
            "SCAN LOCATION: FINAL DIAGNOSES section, typically listed under tau proteinopathy "
            "or as a standalone finding. Also check any microscopic hippocampal description. "
            "COMMON PHRASING: "
            "'Granulovacuolar degeneration. Focal: hippocampus, subiculum' → code 1; "
            "'occasional neurons with granulovacuolar degeneration' → code 1; "
            "'granulovacuolar degeneration, mild' → code 1; "
            "'granulovacuolar degeneration, moderate, hippocampus' → code 2; "
            "'granulovacuolar degeneration, severe' → code 3. "
            "NOT mentioned in Final Diagnoses or microscopic section → 9. "
            "ALWAYS scan the Final Diagnoses before coding 9."
        ),
    )
    hirano_bodies_hippocampus_severity: int = Field(
        default=ORD_UNKNOWN,
        description=(
            "Severity of Hirano bodies in hippocampal formation on microscopic examination. "
            "0=None, 1=Mild (rare/occasional/few), 2=Moderate, 3=Severe. "
            "8=Not assessed, 9=Not mentioned. "
            "SCAN LOCATION: FINAL DIAGNOSES section AND any microscopic hippocampal description. "
            "Reports list Hirano bodies as a standalone finding in the Final Diagnoses. "
            "COMMON PHRASING: "
            "'Hirano bodies. Rare: hippocampus' → code 1; "
            "'a small number of Hirano bodies' → code 1; "
            "'few Hirano bodies in CA1, CA2, CA3' → code 1; "
            "'Hirano bodies, mild' → code 1; "
            "'Hirano bodies, moderate, hippocampus' → code 2; "
            "'Hirano bodies, numerous/severe' → code 3. "
            "CRITICAL: Always check the FINAL DIAGNOSES section explicitly — "
            "Hirano bodies are only described microscopically and ONLY appear in Final Diagnoses "
            "or the microscopic description, NEVER in the gross description. "
            "Seeds that scan only the gross description will incorrectly code 9. "
            "NOT mentioned in Final Diagnoses OR microscopic section → 9."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level model — 49 variables across 9 blocks
# ---------------------------------------------------------------------------

class ResidualOutput(BaseModel):
    """Structured residual variable extraction for one autopsy report.

    Mirrors the primary extraction pipeline output format:
    block names at top level, variables nested within blocks,
    field_annotations per variable, and case-level metadata.
    """

    model_config = ConfigDict(extra="ignore")

    bilateral_hemibrain_weights: BilateralHemibrainWeights = Field(
        default_factory=BilateralHemibrainWeights,
        description="Right and left hemibrain weights, fresh and fixed.",
    )
    bilateral_cerebral_weights: BilateralCerebralWeights = Field(
        default_factory=BilateralCerebralWeights,
        description="Right and left cerebral hemisphere weights, fresh and fixed.",
    )
    bilateral_cerebellar_weights: BilateralCerebellarWeights = Field(
        default_factory=BilateralCerebellarWeights,
        description="Right and left cerebellar hemisphere weights, fresh and fixed.",
    )
    bilateral_brainstem_weights: BilateralBrainstemWeights = Field(
        default_factory=BilateralBrainstemWeights,
        description="Right and left brainstem half weights, fresh and fixed.",
    )
    corpus_callosum_measurements: CorpusCallosumMeasurements = Field(
        default_factory=CorpusCallosumMeasurements,
        description="Corpus callosum segment thicknesses in millimeters.",
    )
    circle_of_willis_diameters: CircleOfWillisDiameters = Field(
        default_factory=CircleOfWillisDiameters,
        description="Circle of Willis vessel diameters in millimeters.",
    )
    structural_measurements: StructuralMeasurements = Field(
        default_factory=StructuralMeasurements,
        description="Structural morphometric measurements.",
    )
    structural_severity_markers: StructuralSeverityMarkers = Field(
        default_factory=StructuralSeverityMarkers,
        description="Ordinal severity scales for gross structural findings.",
    )
    regional_pathology_severity: RegionalPathologySeverity = Field(
        default_factory=RegionalPathologySeverity,
        description="Ordinal severity scales for regional neuropathological findings.",
    )

    field_annotations: Optional[Dict[str, FieldAnnotation]] = Field(
        None,
        description=(
            "Per-variable extraction audit keyed by variable name. "
            "Provide an entry for every variable with a non-sentinel value. "
            "Each entry: confidence (float 0-1), evidence (verbatim quote), note (or null)."
        ),
    )
    extraction_confidence: Optional[str] = Field(
        None,
        description="Overall case-level confidence: 'high', 'moderate', or 'low'.",
    )
    extraction_notes: Optional[str] = Field(
        None,
        description="Case-level caveats — illegible sections, ambiguities. Null if none.",
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> "ResidualOutput":
        """Validate all continuous ranges and ordinal codes."""

        cont_checks = [
            (self.bilateral_hemibrain_weights, "hemibrain_weight_right_fresh_g",  0.0, 2000.0),
            (self.bilateral_hemibrain_weights, "hemibrain_weight_left_fresh_g",   0.0, 2000.0),
            (self.bilateral_hemibrain_weights, "hemibrain_weight_right_fixed_g",  0.0, 2000.0),
            (self.bilateral_hemibrain_weights, "hemibrain_weight_left_fixed_g",   0.0, 2000.0),
            (self.bilateral_cerebral_weights, "cerebral_weight_right_fresh_g",   0.0, 2000.0),
            (self.bilateral_cerebral_weights, "cerebral_weight_left_fresh_g",    0.0, 2000.0),
            (self.bilateral_cerebral_weights, "cerebral_weight_right_fixed_g",   0.0, 2000.0),
            (self.bilateral_cerebral_weights, "cerebral_weight_left_fixed_g",    0.0, 2000.0),
            (self.bilateral_cerebellar_weights, "cerebellar_weight_right_fresh_g", 0.0, 200.0),
            (self.bilateral_cerebellar_weights, "cerebellar_weight_left_fresh_g",  0.0, 200.0),
            (self.bilateral_cerebellar_weights, "cerebellar_weight_right_fixed_g", 0.0, 200.0),
            (self.bilateral_cerebellar_weights, "cerebellar_weight_left_fixed_g",  0.0, 200.0),
            (self.bilateral_brainstem_weights, "brainstem_weight_right_fresh_g",  0.0, 100.0),
            (self.bilateral_brainstem_weights, "brainstem_weight_left_fresh_g",   0.0, 100.0),
            (self.bilateral_brainstem_weights, "brainstem_weight_right_fixed_g",  0.0, 100.0),
            (self.bilateral_brainstem_weights, "brainstem_weight_left_fixed_g",   0.0, 100.0),
            (self.corpus_callosum_measurements, "corpus_callosum_genu_mm",          0.0, 30.0),
            (self.corpus_callosum_measurements, "corpus_callosum_body_anterior_mm",  0.0, 30.0),
            (self.corpus_callosum_measurements, "corpus_callosum_body_mid_mm",       0.0, 30.0),
            (self.corpus_callosum_measurements, "corpus_callosum_body_posterior_mm", 0.0, 30.0),
            (self.corpus_callosum_measurements, "corpus_callosum_splenium_mm",       0.0, 30.0),
            (self.circle_of_willis_diameters, "cow_basilar_mm",         0.0, 15.0),
            (self.circle_of_willis_diameters, "cow_vertebral_right_mm", 0.0, 15.0),
            (self.circle_of_willis_diameters, "cow_vertebral_left_mm",  0.0, 15.0),
            (self.circle_of_willis_diameters, "cow_ica_right_mm",       0.0, 15.0),
            (self.circle_of_willis_diameters, "cow_ica_left_mm",        0.0, 15.0),
            (self.circle_of_willis_diameters, "cow_mca_right_mm",       0.0, 15.0),
            (self.circle_of_willis_diameters, "cow_mca_left_mm",        0.0, 15.0),
            (self.circle_of_willis_diameters, "cow_aca_right_mm",       0.0, 15.0),
            (self.circle_of_willis_diameters, "cow_aca_left_mm",        0.0, 15.0),
            (self.circle_of_willis_diameters, "cow_pca_right_mm",       0.0, 15.0),
            (self.circle_of_willis_diameters, "cow_pca_left_mm",        0.0, 15.0),
            (self.circle_of_willis_diameters, "cow_pcom_right_mm",      0.0, 15.0),
            (self.circle_of_willis_diameters, "cow_pcom_left_mm",       0.0, 15.0),
            (self.circle_of_willis_diameters, "cow_acom_mm",            0.0, 15.0),
            (self.structural_measurements, "caudate_nucleus_head_width_mm", 0.0, 30.0),
        ]

        for block, field, lo, hi in cont_checks:
            val = getattr(block, field)
            if val != CONT_UNKNOWN and not (lo <= val <= hi):
                raise ValueError(
                    f"{field}={val} is outside allowable range [{lo}, {hi}]. "
                    f"Use {CONT_UNKNOWN} if unknown or not reported."
                )

        valid_ordinal = {0, 1, 2, 3, 8, 9}
        ord_checks = [
            (self.structural_severity_markers, "lateral_ventricle_enlargement_severity"),
            (self.structural_severity_markers, "amygdala_atrophy_severity"),
            (self.structural_severity_markers, "dilated_perivascular_spaces"),
            (self.structural_severity_markers, "cerebellar_atrophy_severity"),
            (self.structural_severity_markers, "globus_pallidus_neuronal_loss_severity"),
            (self.structural_severity_markers, "basal_ganglia_atrophy"),
            (self.structural_severity_markers, "thalamic_degeneration_severity"),
            (self.structural_severity_markers, "brainstem_atrophy_severity"),
            (self.regional_pathology_severity, "purkinje_cell_loss_severity"),
            (self.regional_pathology_severity, "dentate_nucleus_atrophy"),
            (self.regional_pathology_severity, "artag_severity"),
            (self.regional_pathology_severity, "gvd_hippocampus_severity"),
            (self.regional_pathology_severity, "hirano_bodies_hippocampus_severity"),
        ]

        for block, field in ord_checks:
            val = getattr(block, field)
            if val not in valid_ordinal:
                raise ValueError(
                    f"{field}={val} is not a valid ordinal code. "
                    f"Must be one of {valid_ordinal}."
                )

        return self

    @model_validator(mode="after")
    def validate_confidence(self) -> "ResidualOutput":
        if self.extraction_confidence and self.extraction_confidence not in {
            "high", "moderate", "low"
        }:
            raise ValueError(
                "extraction_confidence must be 'high', 'moderate', or 'low'."
            )
        return self


# ---------------------------------------------------------------------------
# Format instructions builder — mirrors build_pass3_format_instructions()
# ---------------------------------------------------------------------------

def build_residual_format_instructions() -> str:
    """Walk ResidualOutput schema and emit LLM-facing format instructions."""

    lines = [
        "RESIDUAL VARIABLE EXTRACTION — Extract exactly 49 variables across 9 blocks.",
        "Respond with a single JSON object with EXACTLY these top-level keys:",
        "",
        "  bilateral_hemibrain_weights, bilateral_cerebral_weights,",
        "  bilateral_cerebellar_weights, bilateral_brainstem_weights,",
        "  corpus_callosum_measurements, circle_of_willis_diameters,",
        "  structural_measurements, structural_severity_markers,",
        "  regional_pathology_severity,",
        "  field_annotations, extraction_confidence, extraction_notes",
        "",
        "Each block key maps to a nested object containing its variables.",
        "NEVER output variables at the root level — they must always be nested in their block.",
        "",
        "═══════════════════════════════════════════════════════════════",
        "SENTINEL VALUES",
        "═══════════════════════════════════════════════════════════════",
        "  Continuous (weights, mm): 9999.0 = value not reported anywhere in the report.",
        "  Ordinal (0-3 severity):   9 = finding not mentioned anywhere in the report.",
        "  Ordinal:                  0 = finding explicitly examined and found absent/normal.",
        "  CoW vessels only:         0.0 = vessel explicitly absent, non-patent, or not identified.",
        "",
        "═══════════════════════════════════════════════════════════════",
        "WEIGHT EXTRACTION RULES (applies to all 4 weight blocks)",
        "═══════════════════════════════════════════════════════════════",
        "1. FULL-TEXT SCAN: Weight values appear ANYWHERE in the gross description, including",
        "   mid-paragraph in hemisection or fixation sections. Scan the ENTIRE gross description.",
        "   Do not stop after the first paragraph or the structured header section.",
        "",
        "2. FIXATION STATE DETECTION:",
        "   FRESH = mentioned before any fixation statement, OR described as frozen/saved at -70°C/-80°C.",
        "   FIXED = mentioned after 'Following formalin fixation (X days)' — this covers ALL weight",
        "   values in that paragraph AND all subsequent paragraphs until end of gross description.",
        "   Never assign a fixed weight to a fresh field or vice versa.",
        "",
        "3. LATERALITY: Extract right and left separately. When bilateral = same value is given",
        "   without left/right split (e.g., 'both hemibrains weigh X'), use 9999.0 for both.",
        "",
        "4. BRAINSTEM HEMISECTION EXCEPTION: If the report states the brainstem was NOT hemisected",
        "   (e.g., 'with the exception of the brainstem, the brain is hemisected'), then any",
        "   brainstem weight mentioned = WHOLE brainstem, not a half.",
        "   Use 9999.0 for BOTH right and left brainstem weight variables in this case.",
        "",
        "5. PARTIAL SECTIONS: 'a portion of the right cerebellar hemisphere weighs X' — still",
        "   extract the value. Note it is a partial section in field_annotations.",
        "",
        "═══════════════════════════════════════════════════════════════",
        "CIRCLE OF WILLIS RULES",
        "═══════════════════════════════════════════════════════════════",
        "1. COMPLETE ENUMERATION: Read the ENTIRE vessel measurement paragraph before extracting",
        "   any value. ACA, PCom, and ACom appear at the END of the list and are easily missed.",
        "   Do not stop reading after ICA or MCA.",
        "",
        "2. BILATERAL LISTING WITHOUT L/R: When the report gives one value for both sides",
        "   (e.g., 'posterior cerebral arteries, 3 mm' or 'MCA both 2 mm'),",
        "   apply the SAME value to BOTH left and right variables.",
        "",
        "3. P1/P2 SEGMENTS: When PCA has P1 and P2 values listed separately",
        "   (e.g., 'left PCA 1 mm (P1) and 2 mm (P2)'), use the P1 value.",
        "   When 'P1 segments measure X mm' is given for both sides → fill both cow_pca_*_mm = X.",
        "",
        "4. SMALL BUT VALID VALUES: 'P1 segments are thin and measure 0.5 mm' → extract 0.5.",
        "   Do not treat small diameters (<1 mm) as absent — they are measurably present.",
        "",
        "5. ABSENT VESSELS: 'cannot be identified', 'absent', 'non-patent', 'not visualized' → 0.0.",
        "   '<1 mm' → use 0.5. '<2 mm' → use 1.5.",
        "",
        "6. PLAQUE SIZE ≠ VESSEL DIAMETER: 'The basilar artery has a plaque measuring 5 mm'",
        "   describes plaque LENGTH, not vessel diameter. Do not extract this as a diameter.",
        "",
        "═══════════════════════════════════════════════════════════════",
        "ORDINAL SEVERITY RULES (structural_severity_markers and regional_pathology_severity)",
        "═══════════════════════════════════════════════════════════════",
        "1. SCAN ALL SECTIONS: Check GROSS DESCRIPTION and FINAL DIAGNOSES and MICROSCOPIC",
        "   DESCRIPTION before coding any ordinal variable as 9 (not mentioned).",
        "   Many severity findings appear ONLY in the Final Diagnoses section.",
        "",
        "2. CODE MEANINGS:",
        "   0 = Examined and ABSENT (normal). Use when report says 'no atrophy', 'unremarkable'.",
        "   1 = Mild / Slight / Some / Minimal / Rare / Focal / Occasional.",
        "   2 = Moderate / Prominent / Important reduction / Mild-to-moderate.",
        "   3 = Severe / Marked / Very / Advanced / Most severe / Moderate-to-severe.",
        "   8 = Not assessable (rarely appropriate — NEVER use for gross atrophy variables).",
        "   9 = Not mentioned ANYWHERE in the report.",
        "",
        "3. KEY DISTINCTION: 9 means ABSENT FROM REPORT TEXT, not absent from the brain.",
        "   Code 0 means PRESENT AND NORMAL. Do not use 9 when the finding is described",
        "   as normal or absent — that is code 0.",
        "",
        "4. FINAL DIAGNOSES SCANNING: The neuronal loss enumeration in Final Diagnoses lists",
        "   regions with their severity. Example:",
        "   'Neuronal loss and gliosis: Severe: globus pallidus, thalamus; Moderate: caudate'",
        "   → globus_pallidus_neuronal_loss_severity = 3, thalamic_degeneration_severity = 3.",
        "   ALWAYS read this enumeration completely.",
        "",
        "5. CEREBELLAR ATROPHY: 'No lesions on external surfaces' means no FOCAL LESIONS,",
        "   NOT no atrophy. Code 0 ONLY if the report explicitly says 'no cerebellar atrophy'",
        "   or 'cerebellum is normal in size/bulk'. Otherwise use 9 if atrophy not mentioned.",
        "",
        "6. BASAL GANGLIA: Code 8 is FORBIDDEN for basal_ganglia_atrophy — gross atrophy",
        "   is always assessable. Use 0 if normal, 9 if not mentioned, 1-3 for severity.",
        "",
        "7. HIRANO BODIES AND GVD: These appear ONLY in Final Diagnoses or microscopic section,",
        "   NEVER in the gross description. Always check Final Diagnoses before coding 9.",
        "",
    ]

    # Walk the top-level model fields — emit block structure with variable descriptions
    for block_name, field_info in ResidualOutput.model_fields.items():
        if block_name in METADATA_KEYS:
            continue
        annotation = field_info.annotation

        if hasattr(annotation, "__mro__") and issubclass(annotation, BaseModel):
            block_cls = annotation
        else:
            try:
                from typing import get_args
                args = get_args(annotation)
                block_cls = args[0] if args else None
            except Exception:
                block_cls = None

        if block_cls is None or not issubclass(block_cls, BaseModel):
            continue

        lines.append(f"[{block_name}]")
        for var_name, var_info in block_cls.model_fields.items():
            desc = var_info.description or ""
            # Emit first sentence only — detailed rules are in the block-level
            # sections above. Full descriptions remain in Field() for code docs.
            first_dot = desc.find('. ')
            brief = (desc[:first_dot + 1] if first_dot != -1 else desc)[:180]
            lines.append(f"  - {var_name}: {brief}")
        lines.append("")

    lines += [
        "[field_annotations]",
        "  Per-variable extraction audit keyed by exact variable name.",
        "  Include an entry for EVERY variable with a non-sentinel value",
        "  (continuous != 9999.0, or ordinal != 9).",
        "  Each entry must have exactly three keys:",
        "    confidence: float 0.0-1.0 (1.0=exact; 0.8=implied; 0.6=inferred; 0.4=ambiguous)",
        "    evidence: single string — verbatim phrase from the report. Never a list.",
        "    note: string or null — reasoning note only for non-trivial extractions.",
        "  IMPORTANT: confidence must be a float between 0.0 and 1.0.",
        "  Do NOT put an ordinal severity code (0-9) in the confidence field.",
        "",
        "[extraction_confidence]",
        "  Overall case-level confidence: 'high', 'moderate', or 'low'.",
        "",
        "[extraction_notes]",
        "  Free-text case-level caveats. Null if none.",
        "",
        "Do not invent values. Extract only what is explicitly stated or clearly implied.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts — same template pattern as main.py
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a clinical NLP system that extracts structured residual neuropathology \
variables from autopsy reports.

{format_instructions}

GLOBAL EXTRACTION RULES:
- Extract only what is explicitly stated or clearly implied. Do not hallucinate values.
- All numeric codes must be exact values from the allowed set in each field description.
- OCR quality may be imperfect: if a word appears misspelled or garbled, infer the most \
likely intended term from context. Do not skip a field solely due to apparent OCR errors.

SEVERITY LANGUAGE MAPPING (applies to all ordinal variables):
  'some', 'slight', 'rare', 'focal', 'occasional', 'minimal' → 1
  'mild' → 1
  'mild to moderate' → 2
  'moderate', 'important', 'prominent', 'diffuse' (as severity) → 2
  'moderate to severe', 'moderately severe' → 3
  'severe', 'marked', 'very', 'most severe', 'advanced' → 3

ANNOTATION RULES:
- For every variable with a non-sentinel value, add an entry in field_annotations.
- evidence: exact phrase from the report — single string, never a list.
- note: reasoning note only for non-trivial extractions. Null for clear ones.
- confidence: float 0.0-1.0. NEVER put a severity code (0-9) in this field.

OUTPUT: valid JSON only — no markdown fences, no commentary before or after.\
"""

USER_PROMPT = "NEUROPATHOLOGY REPORT:\n\n{report_text}"


def build_messages(report_text: str) -> list[dict]:
    """Build the system + user message list for one extraction call."""
    fmt = build_residual_format_instructions()
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(format_instructions=fmt)},
        {"role": "user",   "content": USER_PROMPT.format(report_text=report_text)},
    ]


# ---------------------------------------------------------------------------
# Output normaliser — reroutes misplaced variables (mirrors reroute_fields)
# ---------------------------------------------------------------------------

def _build_residual_router() -> dict[str, str]:
    """Map variable_name -> block_name for all 49 residual variables."""
    router: dict[str, str] = {}
    for block_name, field_info in ResidualOutput.model_fields.items():
        if block_name in METADATA_KEYS:
            continue
        annotation = field_info.annotation
        try:
            from typing import get_args
            args = get_args(annotation)
            block_cls = args[0] if args else annotation
        except Exception:
            block_cls = annotation

        if isinstance(block_cls, type) and issubclass(block_cls, BaseModel):
            for var_name in block_cls.model_fields:
                router[var_name] = block_name
    return router


_RESIDUAL_ROUTER: dict[str, str] = {}


def normalize_residual_output(data: dict) -> dict:
    """Fix common LLM output issues before Pydantic validation."""
    global _RESIDUAL_ROUTER
    if not _RESIDUAL_ROUTER:
        _RESIDUAL_ROUTER = _build_residual_router()

    for var_name in list(data.keys()):
        if var_name in RESIDUAL_BLOCKS or var_name in METADATA_KEYS:
            continue
        correct_block = _RESIDUAL_ROUTER.get(var_name)
        if correct_block is not None:
            logger.debug("rerouting root variable %s -> %s", var_name, correct_block)
            target = data.setdefault(correct_block, {})
            if var_name not in target:
                target[var_name] = data.pop(var_name)
            else:
                data.pop(var_name)

    for block_name in list(data.keys()):
        if block_name not in RESIDUAL_BLOCKS:
            continue
        block = data[block_name]
        if not isinstance(block, dict):
            continue
        for var_name in list(block.keys()):
            correct_block = _RESIDUAL_ROUTER.get(var_name)
            if correct_block and correct_block != block_name:
                logger.debug(
                    "rerouting %s from %s -> %s", var_name, block_name, correct_block
                )
                target = data.setdefault(correct_block, {})
                if var_name not in target:
                    target[var_name] = block.pop(var_name)
                else:
                    block.pop(var_name)

    return data


# ---------------------------------------------------------------------------
# Programmatic delta computation
# ---------------------------------------------------------------------------

def compute_programmatic(result: ResidualOutput) -> dict:
    """Compute bilateral weight asymmetry deltas (right minus left).

    Prefers fresh bilateral pair. Falls back to fixed bilateral pair only if
    both sides have fixed weights. Mixed fresh/fixed → 9999.0 (unreliable).
    A delta of 9999.0 means the delta could not be computed.
    """
    hb = result.bilateral_hemibrain_weights
    cb = result.bilateral_cerebral_weights
    cr = result.bilateral_cerebellar_weights
    bs = result.bilateral_brainstem_weights

    def best_delta(r_fresh: float, l_fresh: float,
                   r_fixed: float, l_fixed: float) -> float:
        if r_fresh != CONT_UNKNOWN and l_fresh != CONT_UNKNOWN:
            return round(r_fresh - l_fresh, 2)
        if r_fixed != CONT_UNKNOWN and l_fixed != CONT_UNKNOWN:
            return round(r_fixed - l_fixed, 2)
        return CONT_UNKNOWN

    return {
        "hemibrain_weight_delta_g": best_delta(
            hb.hemibrain_weight_right_fresh_g, hb.hemibrain_weight_left_fresh_g,
            hb.hemibrain_weight_right_fixed_g, hb.hemibrain_weight_left_fixed_g,
        ),
        "cerebral_weight_delta_g": best_delta(
            cb.cerebral_weight_right_fresh_g, cb.cerebral_weight_left_fresh_g,
            cb.cerebral_weight_right_fixed_g, cb.cerebral_weight_left_fixed_g,
        ),
        "cerebellar_weight_delta_g": best_delta(
            cr.cerebellar_weight_right_fresh_g, cr.cerebellar_weight_left_fresh_g,
            cr.cerebellar_weight_right_fixed_g, cr.cerebellar_weight_left_fixed_g,
        ),
        "brainstem_weight_delta_g": best_delta(
            bs.brainstem_weight_right_fresh_g, bs.brainstem_weight_left_fresh_g,
            bs.brainstem_weight_right_fixed_g, bs.brainstem_weight_left_fixed_g,
        ),
    }


# ---------------------------------------------------------------------------
# Core extraction — mirrors main.py extract()
# ---------------------------------------------------------------------------

def run_residual_extraction(
    report_text: str,
    report_id: str,
    seed: int,
    client: OpenAI,
    model: str = "openai/gpt-oss-20b",
    max_new_tokens: int = 16384,
    temperature: float = 0.01,
    top_p: float = 0.9,
    reasoning_effort: str = "medium",
    max_retries: int = 3,
) -> ResidualOutput:
    """Run residual extraction for one report at one seed."""
    base_messages = build_messages(report_text)
    curr_messages = list(base_messages)
    last_exc: Optional[Exception] = None

    extra_body = {
        "repetition_penalty": 1.1,
        "reasoning_effort":   reasoning_effort,
    }

    for attempt in range(1, max_retries + 1):
        logger.info(
            "report %s seed %d — attempt %d/%d", report_id, seed, attempt, max_retries
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=curr_messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                extra_body=extra_body,
            )
            raw = response.choices[0].message.content or ""
            logger.debug("raw output (first 300): %r", raw[:300])

        except Exception as e:
            logger.error("API call failed: %s", e)
            raise

        cleaned = strip_json_fences(raw)

        try:
            content = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("JSON decode failed, attempting repair: %s", e)
            try:
                content = repair_json(cleaned, return_objects=True)
                if not isinstance(content, dict) or not content:
                    raise ValueError("repair produced empty or non-dict result")
                logger.info("JSON repaired successfully")
            except Exception as repair_err:
                last_exc = repair_err
                logger.error(
                    "JSON repair failed. Parse: %s | Repair: %s", e, repair_err
                )
                curr_messages = list(base_messages) + [
                    {"role": "assistant", "content": cleaned or ""},
                    {"role": "user",
                     "content": (
                         f"Your response was not valid JSON. Error: {e}. "
                         "Output valid JSON only, no markdown fences."
                     )},
                ]
                continue

        content = normalize_residual_output(content)

        try:
            return ResidualOutput.model_validate(content)

        except Exception as val_err:
            last_exc = val_err
            logger.warning("Validation failed: %s", val_err)
            curr_messages = list(base_messages) + [
                {"role": "assistant",
                 "content": json.dumps(_sanitize_for_json(content), indent=2)},
                {"role": "user",
                 "content": (
                     f"Your JSON had validation errors:\n{val_err}\n"
                     "Fix these errors and output corrected JSON only."
                 )},
            ]
            continue

    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Per-report processing
# ---------------------------------------------------------------------------

def process_report(
    report_path: Path,
    output_dir: Path,
    client: OpenAI,
    model: str,
    seeds: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    reasoning_effort: str,
    max_retries: int,
) -> bool:
    """Process one report across all seeds. Returns True if all seeds succeeded."""
    report_id   = report_path.stem
    failed_path = output_dir / f"{report_id}.residual.failed"

    all_existing = [
        output_dir / f"{report_id}_seed{s}.residual.json"
        for s in seeds
    ]
    if all(p.exists() for p in all_existing):
        logger.info("skipping %s — all %d seeds already done", report_id, len(seeds))
        return True

    if failed_path.exists():
        logger.info("clearing previous failure for %s — retrying", report_id)
        failed_path.unlink()

    logger.info("processing %s (%d seeds)", report_id, len(seeds))

    try:
        report_text = load_report(report_path)
    except Exception as e:
        logger.error("FAILED to load %s: %s", report_id, e)
        failed_path.write_text(str(e), encoding="utf-8")
        return False

    all_ok = True
    for seed in seeds:
        out_path = output_dir / f"{report_id}_seed{seed}.residual.json"
        if out_path.exists():
            logger.info("seed %d already done for %s — skipping", seed, report_id)
            continue

        try:
            result = run_residual_extraction(
                report_text=report_text,
                report_id=report_id,
                seed=seed,
                client=client,
                model=model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                reasoning_effort=reasoning_effort,
                max_retries=max_retries,
            )

            out_data = result.model_dump(mode="json")
            out_data = {
                "programmatic": compute_programmatic(result),
                **out_data,
            }

            out_path.write_text(
                json.dumps(_sanitize_for_json(out_data), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("wrote %s seed %d", report_id, seed)

        except Exception as e:
            logger.error("FAILED %s seed %d: %s", report_id, seed, e)
            all_ok = False

    if not all_ok:
        failed_path.write_text(
            f"One or more seeds failed for {report_id}",
            encoding="utf-8",
        )
    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Stage 2: Structured residual variable extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--input", type=Path,
        help="Single report file (.pdf or .txt) — used by SLURM loop.",
    )
    input_group.add_argument(
        "--reports-dir", type=Path,
        help="Directory of reports — used for local testing.",
    )

    ap.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write .residual.json files.",
    )
    ap.add_argument(
        "-m", "--model", type=str, default="openai/gpt-oss-20b",
        help="Model name as served by vLLM (default: openai/gpt-oss-20b).",
    )
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="One or more random seeds (e.g. --seeds 0 1 2 3 4).",
    )
    ap.add_argument(
        "--vllm-url", type=str, required=True,
        help="vLLM server URL e.g. http://localhost:8000.",
    )
    ap.add_argument("--temperature",      type=float, default=0.01)
    ap.add_argument("--top-p",            type=float, default=0.9)
    ap.add_argument(
        "--reasoning-effort", type=str, default="medium",
        choices=["low", "medium", "high"],
    )
    ap.add_argument(
        "--max-new-tokens", type=int, default=16384,
        help="Max new tokens to generate.",
    )
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else [0, 1, 2, 3, 4]

    client = OpenAI(
        base_url=f"{args.vllm_url}/v1",
        api_key="dummy",
    )

    shared_kwargs = dict(
        output_dir=args.output_dir,
        client=client,
        model=args.model,
        seeds=seeds,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        reasoning_effort=args.reasoning_effort,
        max_retries=args.max_retries,
    )

    if args.input is not None:
        if not args.input.exists():
            logger.error("file not found: %s", args.input)
            sys.exit(1)
        process_report(report_path=args.input, **shared_kwargs)
        sys.exit(0)

    reports = sorted([
        f for f in args.reports_dir.iterdir()
        if f.suffix.lower() in {".pdf", ".txt"}
    ])
    if not reports:
        logger.error("No reports found in %s", args.reports_dir)
        sys.exit(1)

    logger.info("Found %d reports", len(reports))
    completed = sum(
        process_report(report_path=r, **shared_kwargs) for r in reports
    )
    failed = len(reports) - completed
    logger.info(
        "Done. completed=%d  failed=%d  total=%d",
        completed, failed, len(reports),
    )


if __name__ == "__main__":
    main()