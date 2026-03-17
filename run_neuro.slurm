#!/bin/bash

# ── SLURM Directives ───────────────────────────────────────────────────────
#SBATCH -A r01042                                                        # Project allocation
#SBATCH -J neuropath_extractor                                           # Job name
#SBATCH -o /N/project/ADRD/neuropathoroot/logs/%x/neuropath_%j.out       # STDOUT (one file per job)
#SBATCH -e /N/project/ADRD/neuropathoroot/logs/%x/neuropath_%j.err       # STDERR
#SBATCH -t 02:00:00                                                      # 8 hours (4 PDFs x 5 seeds)
#SBATCH -c 16                                                            # CPU cores
#SBATCH --mem=256G                                                       # Memory
#SBATCH -p hopper                                                        # Hopper partition (H100s)
#SBATCH -q hopper                                                        # QOS for hopper
#SBATCH --gres=gpu:h100:1                                                # 1x H100 GPU
#SBATCH --mail-type=END,FAIL                                             # Email on finish or fail
#SBATCH --mail-user=aantappa@iu.edu                                      # Your email

set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ── Resolve project root ───────────────────────────────────────────────────
PROJECT=/N/project/ADRD/neuropathoroot
VENV=$PROJECT/.venv
REPORTS=$PROJECT/reports
OUTPUT=$PROJECT/output/qwen2.5-14b

mkdir -p "$OUTPUT"

# ── Load system modules ────────────────────────────────────────────────────
module load python/3.12
module load cudatoolkit/12.6

# ── Virtual environment ────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "[setup] Creating virtual environment at $VENV"
    python -m venv "$VENV"
    source "$VENV/bin/activate"
    echo "[setup] Installing dependencies..."
    pip install --upgrade pip -q
    pip install -r "$PROJECT/NeuroPathlogy/requirements.txt" -q
else
    source "$VENV/bin/activate"
    echo "[setup] Syncing dependencies..."
    pip install --upgrade pip -q
    pip install -r "$PROJECT/NeuroPathlogy/requirements.txt" -q
fi

# ── HuggingFace token ──────────────────────────────────────────────────────
export HF_TOKEN=$(cat ~/.hf_token)

# ── HuggingFace cache ──────────────────────────────────────────────────────
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# ── Confirm GPU ────────────────────────────────────────────────────────────
echo "[gpu] GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Model selection ────────────────────────────────────────────────────────
MODEL_ALIAS="qwen2.5-14b"
NUM_GPUS=1

# ── Seeds ──────────────────────────────────────────────────────────────────
SEEDS=(0 1 2 3 4)

# ── Run extraction ─────────────────────────────────────────────────────────
echo "[run] Starting batch extraction at $(date)"
echo "[run] Model: $MODEL_ALIAS | Seeds: ${SEEDS[*]}"

for REPORT in "$REPORTS"/*.pdf "$REPORTS"/*.txt; do
    [ -f "$REPORT" ] || continue

    BASENAME=$(basename "$REPORT")

    echo "[extract] $BASENAME all seeds -> $OUTPUT/${BASENAME%.*}_seed*.extracted.json"
    python "$PROJECT/NeuroPathlogy/main.py" \
        --input             "$REPORT"      \
        --output-dir        "$OUTPUT"      \
        --model             "$MODEL_ALIAS" \
        --seeds             ${SEEDS[@]}    \
        --num-gpus          "$NUM_GPUS"    \
        --dtype             auto           \
        --temperature       0.01           \
        --top-p             0.9            \
        --reasoning-effort  medium         \
        --max-new-tokens    16384          \
        --max-retries       3
done

echo "[run] Done at $(date)"