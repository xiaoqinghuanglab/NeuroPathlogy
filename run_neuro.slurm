#!/bin/bash
#SBATCH -A r00939                         # Project allocation
#SBATCH -J neuropath_extract              # Job name
#SBATCH -o logs/neuropath_%j.out          # STDOUT (one file per job)
#SBATCH -e logs/neuropath_%j.err          # STDERR
#SBATCH -t 2:00:00                        # 2 hours (testing as per PI)
#SBATCH -c 16                             # CPU cores
#SBATCH --mem=256G                        # Memory (256G+ as per PI)
#SBATCH -p hopper                         # Hopper partition (H100s)
#SBATCH -q hopper                         # QOS for hopper
#SBATCH --gres=gpu:h100:1                 # 1x H100 GPU
#SBATCH --mail-type=END,FAIL              # Email on finish or fail
#SBATCH --mail-user=ankisure@iu.edu       # Your email

set -euo pipefail

# ── Resolve project root ───────────────────────────────────────────────────
PROJECT=/N/project/SingleCell_Image/Ankita/neuropath
VENV=$PROJECT/.venv
REPORTS=$PROJECT/reports
OUTPUT=$PROJECT/output

mkdir -p "$OUTPUT" "$PROJECT/logs"

# ── Load system modules ────────────────────────────────────────────────────
module load python/3.12
module load cudatoolkit/12.2

# ── Virtual environment ────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "[setup] Creating virtual environment at $VENV"
    python -m venv "$VENV"
fi
source "$VENV/bin/activate"

# ── Dependencies ───────────────────────────────────────────────────────────
echo "[setup] Installing/updating dependencies..."
pip install --upgrade pip -q
pip install -r "$PROJECT/requirements.txt" -q

# ── HuggingFace token ──────────────────────────────────────────────────────
export HF_TOKEN=$(cat ~/.hf_token)

# ── HuggingFace cache → scratch ───────────────────────────────────────────
export HF_HOME=/N/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# ── Confirm GPU ────────────────────────────────────────────────────────────
echo "[gpu] GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Model selection ────────────────────────────────────────────────────────
MODEL_ALIAS="oss-20b"      # → openai/gpt-oss-20b
NUM_GPUS=1

# ── Run extraction ─────────────────────────────────────────────────────────
echo "[run] Starting batch extraction at $(date)"
echo "[run] Model: $MODEL_ALIAS"

for REPORT in "$REPORTS"/*.pdf "$REPORTS"/*.txt; do
    [ -f "$REPORT" ] || continue

    BASENAME=$(basename "$REPORT")
    OUT_FILE="$OUTPUT/${BASENAME%.*}.extracted.json"

    if [ -f "$OUT_FILE" ]; then
        echo "[skip] $BASENAME already extracted"
        continue
    fi

    echo "[extract] $BASENAME → $OUT_FILE"
    python "$PROJECT/main.py" \
        --input        "$REPORT"      \
        --output       "$OUT_FILE"    \
        --model        "$MODEL_ALIAS" \
        --num-gpus     "$NUM_GPUS"    \
        --dtype        auto           \
        --temperature  0.3            \
        --top-p        0.9            \
        --max-new-tokens 4096         \
        --max-retries  3
done

echo "[run] Done at $(date)"

