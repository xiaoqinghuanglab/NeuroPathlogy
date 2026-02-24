#!/bin/bash
#SBATCH -A r00939                         # Project allocation
#SBATCH -J neuropath_extract              # Job name
#SBATCH -o logs/neuropath_%j.out          # STDOUT (one file per job)
#SBATCH -e logs/neuropath_%j.err          # STDERR
#SBATCH -t 8:00:00                        # 8 hours (4 PDFs x 5 seeds)
#SBATCH -c 16                             # CPU cores
#SBATCH --mem=256G                        # Memory
#SBATCH -p hopper                         # Hopper partition (H100s)
#SBATCH -q hopper                         # QOS for hopper
#SBATCH --gres=gpu:h100:1                 # 1x H100 GPU
#SBATCH --mail-type=END,FAIL              # Email on finish or fail
#SBATCH --mail-user=ankisure@iu.edu       # Your email

set -euo pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True

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
    source "$VENV/bin/activate"
    echo "[setup] Installing dependencies..."
    pip install --upgrade pip -q
    pip install -r "$PROJECT/requirements.txt" -q
else
    source "$VENV/bin/activate"
fi

# ── HuggingFace token ──────────────────────────────────────────────────────
export HF_TOKEN=$(cat ~/.hf_token)

# ── HuggingFace cache → Slate ─────────────────────────────────────────────
export HF_HOME=/N/project/SingleCell_Image/Ankita/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# ── Confirm GPU ────────────────────────────────────────────────────────────
echo "[gpu] GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Model selection ────────────────────────────────────────────────────────
MODEL_ALIAS="oss-20b"
NUM_GPUS=1

# ── Seeds ─────────────────────────────────────────────────────────────────
SEEDS=(0 1 2 3 4)

# ── Run extraction ─────────────────────────────────────────────────────────
echo "[run] Starting batch extraction at $(date)"
echo "[run] Model: $MODEL_ALIAS | Seeds: ${SEEDS[*]}"

for REPORT in "$REPORTS"/*.pdf "$REPORTS"/*.txt; do
    [ -f "$REPORT" ] || continue

    BASENAME=$(basename "$REPORT")

    for SEED in "${SEEDS[@]}"; do
        OUT_FILE="$OUTPUT/${BASENAME%.*}_seed${SEED}.extracted.json"

        if [ -f "$OUT_FILE" ]; then
            echo "[skip] $BASENAME seed=$SEED already done"
            continue
        fi

        echo "[extract] $BASENAME seed=$SEED → $OUT_FILE"
        python "$PROJECT/main.py" \
            --input             "$REPORT"      \
            --output            "$OUT_FILE"    \
            --model             "$MODEL_ALIAS" \
            --seed              $SEED          \
            --num-gpus          "$NUM_GPUS"    \
            --dtype             auto           \
            --temperature       0.01           \
            --top-p             0.9            \
            --reasoning-effort  medium         \
            --max-new-tokens    16384          \
            --max-retries       3
    done
done

echo "[run] Done at $(date)"
