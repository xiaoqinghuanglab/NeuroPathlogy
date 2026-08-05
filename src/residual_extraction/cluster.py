"""cluster.py — Stage 1B: Embedding-based schema induction.

Reads all .discovery.json files from Stage 1A, embeds observations with
PubMedBERT, reduces dimensionality with UMAP, clusters with HDBSCAN, then
uses OSS-20B to label each cluster into a draft variable definition.

Outputs:
    residual_schema_draft.json  — machine-readable draft schema
    residual_schema_review.xlsx — human review sheet
                                  (KEEP / DROP / EDIT per variable)

Usage:
    python cluster.py --vllm-url http://localhost:PORT

Pipeline:
    1. Load all .discovery.json -> extract observations with metadata
    2. Embed (finding + clinical_significance) with PubMedBERT on GPU
    3. L2-normalize -> UMAP (768 -> 50 dims) -> HDBSCAN clustering
    4. LLM labels each cluster -> draft variable name, description,
       value_type, allowable_codes, nacc_overlap
    5. Deduplication pass -> merge clusters with same concept
    6. LLM labels noise category groups -> same ClusterLabel output
    7. Write residual_schema_draft.json + residual_schema_review.xlsx
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
from json_repair import repair_json
from openai import OpenAI
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from discovery import build_nacc_boundary

logger = logging.getLogger("cluster")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ClusterLabel(BaseModel):
    """LLM-proposed variable definition for one cluster or noise group."""

    model_config = ConfigDict(extra="ignore")

    variable_name: str = Field(
        description=(
            "Snake_case variable name, concise and descriptive. "
            "Examples: sn_pigmentation_right, cc_genu_thickness_mm, "
            "hemisphere_weight_right_g, biondi_bodies_present."
        )
    )
    description: str = Field(
        description=(
            "One sentence defining what this variable captures and why it "
            "matters clinically. Should be suitable for a data dictionary."
        )
    )
    value_type: str = Field(
        description=(
            "One of: continuous, ordinal_0_3, binary_0_1, categorical. "
            "continuous = numeric measurement (weight in grams, diameter in mm). "
            "ordinal_0_3 = severity scale 0=absent/none, 1=mild, 2=moderate, 3=severe. "
            "binary_0_1 = present/absent flag, 0=absent, 1=present. "
            "categorical = discrete unordered codes with no magnitude meaning."
        )
    )
    allowable_codes: dict = Field(
        description=(
            "For continuous: {\"unit\": \"grams\"} or {\"unit\": \"mm\"} etc. "
            "For ordinal_0_3: {\"0\": \"absent\", \"1\": \"mild\", "
            "\"2\": \"moderate\", \"3\": \"severe\"}. "
            "For binary_0_1: {\"0\": \"absent\", \"1\": \"present\"}. "
            "For categorical: map each code integer to its meaning."
        )
    )
    nacc_overlap: str = Field(
        description=(
            "Does any NACC variable partially capture this? "
            "Answer 'none' if no overlap. "
            "Otherwise name the NACC variable and explain what detail "
            "this residual variable adds beyond it. "
            "Example: 'NPGRSNH captures global SN atrophy but not laterality — "
            "this variable adds left vs right distinction.'"
        )
    )

    @field_validator("value_type")
    @classmethod
    def val_value_type(cls, v):
        allowed = {"continuous", "ordinal_0_3", "binary_0_1", "categorical"}
        if v not in allowed:
            raise ValueError(f"value_type must be one of {allowed}, got {v!r}")
        return v

    @model_validator(mode="after")
    def validate_codes_match_type(self) -> "ClusterLabel":
        """Ensure allowable_codes structure matches value_type."""
        if self.value_type == "continuous":
            if "unit" not in self.allowable_codes:
                raise ValueError(
                    "continuous value_type requires allowable_codes with a 'unit' key, "
                    f"got {self.allowable_codes}"
                )
        elif self.value_type == "ordinal_0_3":
            if not all(str(k) in self.allowable_codes for k in ["0", "1", "2", "3"]):
                raise ValueError(
                    "ordinal_0_3 requires allowable_codes with keys '0','1','2','3', "
                    f"got {self.allowable_codes}"
                )
        elif self.value_type == "binary_0_1":
            if not all(str(k) in self.allowable_codes for k in ["0", "1"]):
                raise ValueError(
                    "binary_0_1 requires allowable_codes with keys '0' and '1', "
                    f"got {self.allowable_codes}"
                )
        return self


class DuplicateGroup(BaseModel):
    """One group of duplicate clusters identified during deduplication pass."""

    model_config = ConfigDict(extra="ignore")

    keep_index: int = Field(
        description=(
            "Index (0-based) of the cluster to keep from the candidate list. "
            "Prefer the cluster with the highest prevalence_reports."
        )
    )
    drop_indices: List[int] = Field(
        description="Indices of the duplicate clusters to remove."
    )
    reason: str = Field(
        description="One sentence explaining why these clusters are duplicates."
    )


class DeduplicationResult(BaseModel):
    """Full deduplication result — list of duplicate groups to merge."""

    model_config = ConfigDict(extra="ignore")

    duplicate_groups: List[DuplicateGroup] = Field(
        description=(
            "List of duplicate groups. Each group identifies one canonical variable "
            "to keep and one or more duplicates to drop. "
            "If no duplicates are found, return an empty list."
        )
    )


# ---------------------------------------------------------------------------
# Step 1 — Load Stage 1A outputs
# ---------------------------------------------------------------------------

def load_observations(discovery_dir: Path) -> list[dict]:
    """Load all .discovery.json files and flatten to observation list.

    Each observation dict has:
        text        : finding + clinical_significance (for embedding)
        finding     : raw finding text
        category    : LLM-assigned category label from Stage 1A
        report_id   : source report
        obs_idx     : index within that report
    """
    observations = []
    files = sorted(discovery_dir.glob("*.discovery.json"))

    if not files:
        raise FileNotFoundError(
            f"No .discovery.json files found in {discovery_dir}"
        )

    logger.info("Loading observations from %d discovery files", len(files))

    for fpath in files:
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Skipping %s — could not parse: %s", fpath.name, e)
            continue

        report_id = data.get("report_id", fpath.stem)
        for idx, obs in enumerate(data.get("clinical_observations", [])):
            finding      = obs.get("finding", "").strip()
            significance = obs.get("clinical_significance", "").strip()
            if not finding:
                continue
            observations.append({
                "text":                  f"{finding} {significance}".strip(),
                "finding":               finding,
                "clinical_significance": significance,
                "category":              obs.get("category", ""),
                "report_id":             report_id,
                "obs_idx":               idx,
            })

    logger.info("Loaded %d total observations from %d reports",
                len(observations), len(files))
    return observations


# ---------------------------------------------------------------------------
# Step 2 — Embed with PubMedBERT
# ---------------------------------------------------------------------------

def embed_observations(
    observations: list[dict],
    model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    batch_size: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    """Embed observation texts with PubMedBERT.

    Mean-pools last hidden state across tokens.
    Returns float32 array of shape (N, 768).
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers and torch required. "
            "pip install transformers torch"
        )

    logger.info("Loading PubMedBERT model: %s on %s", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    texts          = [obs["text"] for obs in observations]
    all_embeddings = []

    logger.info("Embedding %d observations in batches of %d", len(texts), batch_size)

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch   = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)

            outputs          = model(**encoded)
            attention_mask   = encoded["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            mask_expanded    = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask       = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings     = (sum_embeddings / sum_mask).cpu().numpy()
            all_embeddings.append(embeddings)

            if (i // batch_size) % 5 == 0:
                logger.info(
                    "Embedded %d / %d observations",
                    min(i + batch_size, len(texts)), len(texts),
                )

    result = np.vstack(all_embeddings).astype(np.float32)
    logger.info("Embedding complete. Shape: %s", result.shape)
    return result


# ---------------------------------------------------------------------------
# Step 3 — L2-normalize -> UMAP -> HDBSCAN
# ---------------------------------------------------------------------------

def cluster_embeddings(
    embeddings: np.ndarray,
    umap_n_components: int = 50,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    hdbscan_min_cluster_size: int = 8,
    hdbscan_min_samples: int = 3,
    random_state: int = 42,
) -> tuple:
    """L2-normalize -> UMAP -> HDBSCAN. Returns (labels, reduced) arrays."""
    try:
        import umap
        import hdbscan as hdbscan_lib
    except ImportError:
        raise ImportError(
            "umap-learn and hdbscan required. "
            "pip install umap-learn hdbscan"
        )

    norms          = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms          = np.where(norms == 0, 1e-9, norms)
    embeddings_norm = embeddings / norms
    logger.info("L2 normalization complete")

    logger.info(
        "Running UMAP: %d dims -> %d dims (n_neighbors=%d)",
        embeddings_norm.shape[1], umap_n_components, umap_n_neighbors,
    )
    reducer = umap.UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=random_state,
        low_memory=False,
    )
    reduced = reducer.fit_transform(embeddings_norm)
    logger.info("UMAP complete. Reduced shape: %s", reduced.shape)

    logger.info(
        "Running HDBSCAN (min_cluster_size=%d, min_samples=%d)",
        hdbscan_min_cluster_size, hdbscan_min_samples,
    )
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int(np.sum(labels == -1))
    logger.info(
        "HDBSCAN complete. Clusters: %d  Noise observations: %d",
        n_clusters, n_noise,
    )
    return labels, reduced


# ---------------------------------------------------------------------------
# Step 4 — LLM cluster labeling (shared by clusters and noise groups)
# ---------------------------------------------------------------------------

LABEL_SYSTEM_PROMPT_TEMPLATE = """\
You are a clinical data architect designing a neuropathology data dictionary.

The following 199 variables are ALREADY captured by the NACC NP Form.
Use this list to accurately determine nacc_overlap for each candidate variable.

{nacc_boundary}

You will be given a group of clinical observations from brain autopsy reports \
that have been automatically clustered because they describe similar findings. \
Your task is to propose a structured variable definition for this cluster.

VALUE TYPE RULES — apply these exactly:
- If the observations contain numeric measurements with explicit units \
(grams, mm, cm, mL), the value_type MUST be continuous regardless of how \
the observations describe the finding. Never assign ordinal_0_3 or categorical \
to a variable that is measured numerically in the reports.
  CORRECT: hemisphere weights in grams -> continuous, unit: grams
  CORRECT: vessel diameters in mm -> continuous, unit: mm
  WRONG:   hemisphere weights -> ordinal_0_3

- If the observations describe multiple distinct protein types \
(e.g. tau AND TDP-43, or amyloid AND alpha-synuclein) without a single \
dominant protein type, name the variable after the MOST PREVALENT single \
protein type observed across the cluster. Do not create a combined variable \
that merges different protein types. Each protein type warrants its own variable.

OUTPUT FORMAT: valid JSON only — no markdown fences, no commentary.
The JSON must have EXACTLY these keys:
  - variable_name: string (snake_case)
  - description: string (one sentence)
  - value_type: string (one of: continuous, ordinal_0_3, binary_0_1, categorical)
  - allowable_codes: object
  - nacc_overlap: string

Rules for allowable_codes by value_type:
  continuous  -> {{"unit": "<unit of measurement>"}}
  ordinal_0_3 -> {{"0": "absent", "1": "mild", "2": "moderate", "3": "severe"}}
  binary_0_1  -> {{"0": "absent", "1": "present"}}
  categorical -> map each integer code to its label

For nacc_overlap: write "none" if no NACC variable captures this at all. \
Otherwise name the NACC variable and explain what additional detail \
this residual variable captures beyond it.
"""

LABEL_USER_PROMPT = """\
The following {n_obs} observations from {n_reports} autopsy reports were \
clustered together by semantic similarity. They all describe the same type \
of finding.

OBSERVATIONS:
{obs_text}

Propose a single structured variable definition that best represents \
what this cluster captures.
"""


def label_cluster(
    cluster_observations: list[dict],
    client: OpenAI,
    system_prompt: str,
    model: str = "openai/gpt-oss-20b",
    max_new_tokens: int = 1024,
    max_retries: int = 3,
) -> Optional[ClusterLabel]:
    """Ask LLM to propose a variable definition for one cluster or noise group.

    For clusters: caller passes centroid-sampled observations (top 30 by
    proximity to centroid) — ensures diverse representation regardless of
    cluster size. For noise groups: caller passes all observations (small).

    Input per call: ~3,500 (system prompt) + n_obs x ~80 tokens.
    Output per call: ~300 tokens.
    """
    n_reports = len(set(o["report_id"] for o in cluster_observations))

    obs_lines = []
    for i, obs in enumerate(cluster_observations, 1):
        obs_lines.append(
            f"{i}. [{obs['report_id']}] {obs['finding']}\n"
            f"   Significance: {obs.get('clinical_significance', '')}"
        )
    obs_text = "\n".join(obs_lines)

    user_content = LABEL_USER_PROMPT.format(
        n_obs=len(cluster_observations),
        n_reports=n_reports,
        obs_text=obs_text,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    extra_body = {"repetition_penalty": 1.1, "reasoning_effort": "medium"}
    last_exc   = None
    cleaned    = ""  # initialised so retry message is always well-defined

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.01,
                top_p=0.9,
                seed=0,
                extra_body=extra_body,
            )
            raw = response.choices[0].message.content or ""

            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

            try:
                content = json.loads(cleaned)
            except json.JSONDecodeError:
                content = repair_json(cleaned, return_objects=True)

            return ClusterLabel.model_validate(content)

        except Exception as e:
            last_exc = e
            logger.warning("Label attempt %d failed: %s", attempt, e)
            if attempt < max_retries:
                messages = messages[:2] + [
                    {"role": "assistant", "content": cleaned},
                    {"role": "user",
                     "content": f"Validation error: {e}. Fix and output valid JSON only."},
                ]
            continue

    logger.error("Labeling failed after %d attempts: %s", max_retries, last_exc)
    return None


# ---------------------------------------------------------------------------
# Step 4b — Deduplication pass
# ---------------------------------------------------------------------------

DEDUP_SYSTEM_PROMPT = """\
You are reviewing a list of candidate variable names proposed for a \
neuropathology data dictionary. Some variables may be duplicates — they \
capture the same clinical concept but were named differently because they \
came from different clusters. Your task is to identify duplicate groups \
and determine which variable to keep in each group.

A duplicate group is two or more variables that capture essentially the same \
clinical concept, even if the names or descriptions differ slightly. \
Examples of duplicates:
  - arteriolosclerosis_severity (49 reports) and arteriolosclerosis_severity \
(28 reports) — same concept, should be merged
  - substantia_nigra_depigmentation_severity and sn_pigmentation_severity \
— same concept, different names
  - regional_cortical_atrophy_severity (36 reports) and \
regional_cortical_atrophy_severity (4 reports) — exact same name

OUTPUT FORMAT: valid JSON only — no markdown fences, no commentary.
The JSON must have exactly one key:
  - duplicate_groups: list of objects, each with:
    - keep_index: integer (0-based index of the variable to keep; \
prefer the one with higher prevalence_reports)
    - drop_indices: list of integers (indices of variables to remove)
    - reason: string (one sentence explaining why these are duplicates)

If no duplicates are found, return: {"duplicate_groups": []}
"""

DEDUP_USER_PROMPT = """\
Here are the {n_vars} candidate variables from clustering. \
Identify all duplicate groups.

CANDIDATES:
{var_list}
"""


def deduplicate_clusters(
    cluster_results: list[dict],
    client: OpenAI,
    model: str = "openai/gpt-oss-20b",
    max_retries: int = 3,
) -> list[dict]:
    """Run a deduplication pass over labeled clusters.

    Sends the full list of proposed variable names and descriptions to the LLM,
    which identifies duplicate groups. Duplicates are removed, keeping the
    highest-prevalence variable from each group.
    """
    labeled = [r for r in cluster_results if r.get("label") is not None]
    if len(labeled) < 2:
        return cluster_results

    var_lines = []
    for i, entry in enumerate(cluster_results):
        label    = entry.get("label") or {}
        examples = entry.get("representative_examples", [])
        example  = examples[0][:80] if examples else ""
        var_lines.append(
            f"#{i}: {label.get('variable_name', 'unlabeled')} "
            f"({entry.get('prevalence_reports', 0)} reports) — "
            f"{label.get('description', '')[:100]}\n"
            f"   Example: {example}"
        )
    var_list = "\n".join(var_lines)

    user_content = DEDUP_USER_PROMPT.format(
        n_vars=len(cluster_results),
        var_list=var_list,
    )

    messages = [
        {"role": "system", "content": DEDUP_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    extra_body = {"repetition_penalty": 1.1, "reasoning_effort": "medium"}
    last_exc   = None
    cleaned    = ""  # initialised so retry message is always well-defined

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2048,
                temperature=0.01,
                top_p=0.9,
                seed=0,
                extra_body=extra_body,
            )
            raw = response.choices[0].message.content or ""

            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

            try:
                content = json.loads(cleaned)
            except json.JSONDecodeError:
                content = repair_json(cleaned, return_objects=True)

            result   = DeduplicationResult.model_validate(content)
            drop_set = set()

            for group in result.duplicate_groups:
                keep_idx = group.keep_index
                for drop_idx in group.drop_indices:
                    if drop_idx != keep_idx and 0 <= drop_idx < len(cluster_results):
                        drop_set.add(drop_idx)
                        logger.info(
                            "Dedup: dropping #%d (%s) in favour of #%d (%s) — %s",
                            drop_idx,
                            (cluster_results[drop_idx].get("label") or {}).get(
                                "variable_name", "?"
                            ),
                            keep_idx,
                            (cluster_results[keep_idx].get("label") or {}).get(
                                "variable_name", "?"
                            ),
                            group.reason,
                        )

            filtered = [
                entry for i, entry in enumerate(cluster_results)
                if i not in drop_set
            ]
            logger.info(
                "Deduplication complete: %d -> %d variables (%d removed)",
                len(cluster_results), len(filtered), len(drop_set),
            )
            return filtered

        except Exception as e:
            last_exc = e
            logger.warning("Dedup attempt %d failed: %s", attempt, e)
            if attempt < max_retries:
                messages = messages[:2] + [
                    {"role": "assistant", "content": cleaned},
                    {"role": "user",
                     "content": f"Error: {e}. Fix and output valid JSON only."},
                ]
            continue

    logger.error(
        "Deduplication failed after %d attempts: %s — returning undeduped results",
        max_retries, last_exc,
    )
    return cluster_results


# ---------------------------------------------------------------------------
# Step 4c — Noise group labeling
# ---------------------------------------------------------------------------

def label_noise_groups(
    noise_observations: list[dict],
    client: OpenAI,
    system_prompt: str,
    model: str = "openai/gpt-oss-20b",
    max_retries: int = 3,
) -> list[dict]:
    """Label each Stage 1A category group in the noise observations.

    Groups noise observations by their Stage 1A category string. For each
    unique category, runs label_cluster (capped at 15 observations) to produce
    a full ClusterLabel — same output as cluster labeling.

    Token budget per call:
        ~3,500 (system prompt) + 15 x 80 (observations) = ~4,700 input tokens
        ~300 output tokens
        Total: ~5,000 tokens per call
    Number of calls: ~15-20 (one per unique Stage 1A category in noise)
    Additional compute: ~5 minutes

    Returns list of dicts:
        {
            "category":        str,
            "n_observations":  int,
            "label":           ClusterLabel.model_dump() or None,
        }
    Sorted by n_observations descending.
    """
    # Group by Stage 1A category
    groups: dict[str, list[dict]] = defaultdict(list)
    for obs in noise_observations:
        cat = obs.get("category", "uncategorized").strip() or "uncategorized"
        groups[cat].append(obs)

    logger.info(
        "Labeling %d noise category groups (%d total noise observations)",
        len(groups), len(noise_observations),
    )

    results = []
    for category, obs_list in sorted(groups.items(),
                                     key=lambda x: len(x[1]), reverse=True):
        logger.info(
            "Labeling noise group '%s': %d observations",
            category, len(obs_list),
        )
        label = label_cluster(
            obs_list,
            client=client,
            system_prompt=system_prompt,
            model=model,
            max_retries=max_retries,
        )

        # Derive block_category from LLM variable name, fall back to Stage 1A category
        variable_name  = label.variable_name if label else ""
        block_category = derive_block_category(variable_name, category)

        results.append({
            "category":       category,
            "block_category": block_category,
            "n_observations": len(obs_list),
            "label":          label.model_dump() if label else None,
        })

    logger.info("Noise group labeling complete: %d groups", len(results))
    return results


# ---------------------------------------------------------------------------
# Step 5 — Write outputs
# ---------------------------------------------------------------------------

def write_draft_json(
    cluster_results: list[dict],
    noise_category_labels: list[dict],
    noise_observations: list[dict],
    output_path: Path,
) -> None:
    """Write machine-readable draft schema JSON."""
    output = {
        "candidate_variables":  cluster_results,
        "noise_category_labels": noise_category_labels,
        "noise_observations":   noise_observations,
        "metadata": {
            "n_candidates":         len(cluster_results),
            "n_noise_categories":   len(noise_category_labels),
            "n_noise_observations": len(noise_observations),
        },
    }
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote draft schema: %s", output_path)


def write_review_xlsx(
    cluster_results: list[dict],
    noise_category_labels: list[dict],
    output_path: Path,
) -> None:
    """Write human review Excel sheet for Stage 1C.

    Two sheets:
      1. Candidate Variables   — one row per cluster, full ClusterLabel
                                 pre-filled, decision column (KEEP/DROP/EDIT)
      2. Noise Categories      — one row per unique Stage 1A noise category,
                                 full ClusterLabel pre-filled by LLM,
                                 decision column (PROMOTE/SKIP)
    """
    wb = Workbook()

    bold_font  = Font(name="Arial", bold=True, size=10)
    body_font  = Font(name="Arial", size=9)
    wrap_align = Alignment(wrap_text=True, vertical="top", horizontal="left")
    top_align  = Alignment(vertical="top", horizontal="left")

    def set_headers(ws, headers):
        for col, h in enumerate(headers, 1):
            c = ws.cell(row=1, column=col, value=h)
            c.font = bold_font
            c.alignment = top_align

    def write_row(ws, r, vals):
        for col, val in enumerate(vals, 1):
            c = ws.cell(row=r, column=col, value=val)
            c.font = body_font
            c.alignment = wrap_align
        ws.row_dimensions[r].height = 60

    def set_col_widths(ws, widths):
        for col, w in enumerate(widths, 1):
            ws.column_dimensions[ws.cell(1, col).column_letter].width = w

    # ── Sheet 1: Candidate Variables ────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Candidate Variables"
    ws1.sheet_view.showGridLines = False

    headers1 = [
        "block_category", "variable_name", "description", "value_type",
        "allowable_codes", "prevalence_reports", "nacc_overlap",
        "example_1", "example_2", "example_3",
        "decision",   # KEEP / DROP / EDIT
        "notes",
    ]
    set_headers(ws1, headers1)

    for r, entry in enumerate(cluster_results, 2):
        label    = entry.get("label") or {}
        examples = entry.get("representative_examples", [])
        write_row(ws1, r, [
            entry.get("block_category", ""),
            label.get("variable_name", ""),
            label.get("description", ""),
            label.get("value_type", ""),
            json.dumps(label.get("allowable_codes", {})),
            entry.get("prevalence_reports", 0),
            label.get("nacc_overlap", ""),
            examples[0] if len(examples) > 0 else "",
            examples[1] if len(examples) > 1 else "",
            examples[2] if len(examples) > 2 else "",
            "",  # decision
            "",  # notes
        ])

    set_col_widths(ws1, [22, 25, 45, 15, 30, 12, 40, 35, 35, 35, 12, 25])
    ws1.freeze_panes = "A2"

    # ── Sheet 2: Noise Categories ────────────────────────────────────────
    # One row per unique Stage 1A category in noise.
    # Full ClusterLabel pre-filled by LLM — human decides PROMOTE or SKIP.
    ws2 = wb.create_sheet("Noise Categories")
    ws2.sheet_view.showGridLines = False

    headers2 = [
        "block_category", "category", "n_observations",
        "proposed_variable_name", "proposed_description", "proposed_value_type",
        "proposed_allowable_codes", "nacc_overlap",
        "decision",   # PROMOTE / SKIP
        "notes",
    ]
    set_headers(ws2, headers2)

    for r, entry in enumerate(noise_category_labels, 2):
        label = entry.get("label") or {}
        write_row(ws2, r, [
            entry.get("block_category", ""),
            entry.get("category", ""),
            entry.get("n_observations", 0),
            label.get("variable_name", ""),
            label.get("description", ""),
            label.get("value_type", ""),
            json.dumps(label.get("allowable_codes", {})),
            label.get("nacc_overlap", ""),
            "",  # decision
            "",  # notes
        ])

    set_col_widths(ws2, [22, 28, 12, 28, 45, 15, 30, 40, 12, 25])
    ws2.freeze_panes = "A2"

    wb.save(output_path)
    logger.info("Wrote review sheet: %s", output_path)


# ---------------------------------------------------------------------------
# Helper — representative examples
# ---------------------------------------------------------------------------

def get_representative_examples(
    cluster_observations: list[dict],
    cluster_embeddings_reduced: np.ndarray,
    cluster_indices: np.ndarray,
    n: int = 3,
) -> list[str]:
    """Return n observations closest to cluster centroid."""
    centroid    = cluster_embeddings_reduced[cluster_indices].mean(axis=0)
    dists       = np.linalg.norm(
        cluster_embeddings_reduced[cluster_indices] - centroid, axis=1
    )
    closest_idx = np.argsort(dists)[:n]
    return [cluster_observations[i]["finding"] for i in closest_idx]


# ---------------------------------------------------------------------------
# Helper — block category derivation from LLM variable name
# ---------------------------------------------------------------------------

# Maps keywords found in the LLM-proposed variable_name to canonical block
# categories. Keyword lookup is more reliable than trusting Stage 1A labels
# because the LLM names the variable from observation content, not from the
# category string that Stage 1A may have assigned incorrectly.
BLOCK_CATEGORY_MAP = {
    "hemisphere":        "hemisphere_weights",
    "hemibrain":         "hemisphere_weights",
    "cerebral_weight":   "hemisphere_weights",
    "cerebellar_weight": "cerebellar_weights",
    "cerebellar_hemi":   "cerebellar_weights",
    "brainstem_weight":  "brainstem_weights",
    "corpus_callosum":   "corpus_callosum_morphometry",
    "hippocampal":       "hippocampal_morphometry",
    "hippocampus":       "hippocampal_morphometry",
    "amygdala":          "amygdala_morphometry",
    "cow_":              "circle_of_willis_diameters",
    "vessel_diameter":   "circle_of_willis_diameters",
    "artery_diameter":   "circle_of_willis_diameters",
    "arterial_diameter": "circle_of_willis_diameters",
    "intracranial_art":  "circle_of_willis_diameters",
    "substantia_nigra":  "substantia_nigra_pigmentation",
    "sn_pigment":        "substantia_nigra_pigmentation",
    "sn_depigment":      "substantia_nigra_pigmentation",
    "locus_coeruleus":   "locus_coeruleus_pigmentation",
    "lc_pigment":        "locus_coeruleus_pigmentation",
    "brainstem_morph":   "brainstem_morphometry",
    "pons_dimension":    "brainstem_morphometry",
    "cortical_atrophy":  "regional_cortical_atrophy",
    "subcortical":       "subcortical_atrophy",
    "ventricular":       "ventricular_dilation",
    "white_matter":      "white_matter_changes",
    "myelin":            "white_matter_changes",
    "arteriolosclerosis":"vascular_pathology",
    "atherosclerosis":   "vascular_pathology",
    "vascular":          "vascular_pathology",
    "tau_":              "protein_pathology_spatial",
    "tdp":               "protein_pathology_spatial",
    "amyloid":           "protein_pathology_spatial",
    "lewy":              "protein_pathology_spatial",
    "prion":             "protein_pathology_spatial",
    "alpha_synuclein":   "protein_pathology_spatial",
    "neuronal_loss":     "neuronal_loss_regional",
    "neuron_loss":       "neuronal_loss_regional",
    "incidental":        "incidental_pathology",
    "iron_deposit":      "incidental_pathology",
    "biondi":            "incidental_pathology",
}


def derive_block_category(variable_name: str, fallback_category: str) -> str:
    """Derive block_category from the LLM-proposed variable_name.

    Uses keyword lookup against BLOCK_CATEGORY_MAP. Falls back to the most
    common Stage 1A category across the cluster's observations if no keyword
    matches — this handles novel variable types not in the map.
    """
    vn = variable_name.lower()
    for keyword, block in BLOCK_CATEGORY_MAP.items():
        if keyword in vn:
            return block
    return fallback_category


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_stage1b(
    discovery_dir: Path,
    output_dir: Path,
    rddnp_path: Path,
    vllm_url: str,
    model: str = "openai/gpt-oss-20b",
    pubmedbert_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    embed_device: str = "cuda",
    embed_batch_size: int = 64,
    umap_n_components: int = 50,
    umap_n_neighbors: int = 15,
    hdbscan_min_cluster_size: int = 8,
    hdbscan_min_samples: int = 3,
    max_label_retries: int = 3,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    nacc_boundary    = build_nacc_boundary(rddnp_path)
    label_sys_prompt = LABEL_SYSTEM_PROMPT_TEMPLATE.format(nacc_boundary=nacc_boundary)
    logger.info("Built NACC boundary for cluster labeling (%d chars)", len(label_sys_prompt))

    # Step 1 — Load
    observations = load_observations(discovery_dir)
    if not observations:
        raise ValueError("No observations loaded. Check discovery_dir.")

    # Step 2 — Embed
    embeddings = embed_observations(
        observations,
        model_name=pubmedbert_model,
        batch_size=embed_batch_size,
        device=embed_device,
    )

    # Step 3 — Cluster
    labels, reduced = cluster_embeddings(
        embeddings,
        umap_n_components=umap_n_components,
        umap_n_neighbors=umap_n_neighbors,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
    )

    cluster_ids    = sorted(set(labels[labels != -1]))
    noise_indices  = np.where(labels == -1)[0]
    noise_obs_list = [observations[i] for i in noise_indices]

    logger.info(
        "%d clusters found | %d noise observations",
        len(cluster_ids), len(noise_obs_list),
    )

    # Step 4 — LLM cluster labeling
    client = OpenAI(base_url=f"{vllm_url}/v1", api_key="dummy")

    cluster_results = []
    for cid in cluster_ids:
        indices     = np.where(labels == cid)[0]
        cluster_obs = [observations[i] for i in indices]
        report_ids  = set(o["report_id"] for o in cluster_obs)

        # stage1a_category used as fallback if keyword lookup finds nothing
        cats             = [o["category"] for o in cluster_obs if o["category"]]
        stage1a_category = Counter(cats).most_common(1)[0][0] if cats else ""

        logger.info(
            "Labeling cluster %d: %d obs from %d reports (category: %s)",
            cid, len(cluster_obs), len(report_ids), stage1a_category,
        )

        # Centroid-sample top 30 observations by proximity — diverse
        # representation regardless of cluster size. Well within token budget.
        cluster_reduced   = reduced[indices]
        centroid          = cluster_reduced.mean(axis=0)
        dists             = np.linalg.norm(cluster_reduced - centroid, axis=1)
        sampled_local_idx = np.argsort(dists)[:30]
        label_obs         = [cluster_obs[i] for i in sampled_local_idx]

        label    = label_cluster(
            label_obs,
            client=client,
            system_prompt=label_sys_prompt,
            model=model,
            max_retries=max_label_retries,
        )

        # Derive block_category from LLM variable name — more reliable than
        # Stage 1A label which may have mis-categorized some observations.
        variable_name  = label.variable_name if label else ""
        block_category = derive_block_category(variable_name, stage1a_category)
        examples = get_representative_examples(cluster_obs, reduced, indices, n=3)

        # cluster_id and source_report_ids intentionally excluded
        cluster_results.append({
            "block_category":          block_category,
            "prevalence_observations": len(cluster_obs),
            "prevalence_reports":      len(report_ids),
            "label":                   label.model_dump() if label else None,
            "representative_examples": examples,
        })

    # Sort by prevalence descending before deduplication
    cluster_results.sort(key=lambda x: x["prevalence_reports"], reverse=True)

    # Step 4b — Deduplication pass
    logger.info(
        "Running deduplication pass over %d labeled clusters...",
        len(cluster_results),
    )
    cluster_results = deduplicate_clusters(
        cluster_results, client=client, model=model
    )

    # Step 4c — Noise group labeling
    # Groups noise by Stage 1A category, runs label_cluster per group.
    # ~15-20 calls, ~5,000 tokens each, ~5 minutes total.
    noise_category_labels = label_noise_groups(
        noise_obs_list,
        client=client,
        system_prompt=label_sys_prompt,
        model=model,
        max_retries=max_label_retries,
    )

    # Step 5 — Write outputs
    write_draft_json(
        cluster_results,
        noise_category_labels,
        noise_obs_list,
        output_dir / "residual_schema_draft.json",
    )
    write_review_xlsx(
        cluster_results,
        noise_category_labels,
        output_dir / "residual_schema_review.xlsx",
    )

    logger.info(
        "Stage 1B complete. %d candidate variables, %d noise categories, "
        "%d noise observations.",
        len(cluster_results), len(noise_category_labels), len(noise_obs_list),
    )
    logger.info("Review sheet: %s", output_dir / "residual_schema_review.xlsx")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Stage 1B - Cluster: Embed, cluster, label, and deduplicate.",
    )
    ap.add_argument(
        "--discovery-dir", type=Path,
        default=Path("/N/project/ADRD/neuropathoroot/output/oss-20b/residual_discovery"),
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=Path("/N/project/ADRD/neuropathoroot/output/oss-20b/residual_schema"),
    )
    ap.add_argument(
        "--rddnp-path", type=Path,
        default=Path("/N/project/ADRD/neuropathoroot/csv_input/rdd-np.csv"),
    )
    ap.add_argument("--vllm-url", type=str, required=True)
    ap.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    ap.add_argument(
        "--pubmedbert-model", type=str,
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    )
    ap.add_argument("--embed-device",    type=str,  default="cuda",
                    choices=["cuda", "cpu"])
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--umap-n-components", type=int, default=50)
    ap.add_argument("--umap-n-neighbors",  type=int, default=15)
    ap.add_argument("--hdbscan-min-cluster-size", type=int, default=8)
    ap.add_argument("--hdbscan-min-samples",      type=int, default=3)
    ap.add_argument("--max-label-retries", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    run_stage1b(
        discovery_dir=args.discovery_dir,
        output_dir=args.output_dir,
        rddnp_path=args.rddnp_path,
        vllm_url=args.vllm_url,
        model=args.model,
        pubmedbert_model=args.pubmedbert_model,
        embed_device=args.embed_device,
        embed_batch_size=args.embed_batch_size,
        umap_n_components=args.umap_n_components,
        umap_n_neighbors=args.umap_n_neighbors,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        max_label_retries=args.max_label_retries,
    )


if __name__ == "__main__":
    main()