#!/bin/bash

# ── SLURM Directives ───────────────────────────────────────────────────────
#SBATCH -A r01042                                                        # Project allocation
#SBATCH -J neuropath_extractor                                           # Job name
#SBATCH -o /N/project/ADRD/neuropathoroot/logs/%x/neuropath_%j_%a.out    # STDOUT (one file per job)
#SBATCH -e /N/project/ADRD/neuropathoroot/logs/%x/neuropath_%j_%a.err    # STDERR
#SBATCH -t 24:00:00                                                      # x hours (n PDFs x 5 seeds)
#SBATCH -c 16                                                            # CPU cores
#SBATCH --mem=256G                                                       # Memory
#SBATCH -p hopper                                                        # Hopper partition (H100s)
#SBATCH -q hopper                                                        # QOS for hopper
#SBATCH --gres=gpu:h100:1                                                # 1x H100 GPU
#SBATCH --mail-type=END,FAIL                                             # Email on finish or fail
#SBATCH --mail-user=aantappa@iu.edu                                      # Your email


# ── Run Configuration ──────────────────────────────────────────────────────
RUN_MODE="prod"   # "gt" = 4-report validation | "prod" = n-report production

PROJECT=/N/project/ADRD/neuropathoroot
MODEL="openai/gpt-oss-20b"
MODEL_ALIAS="oss-20b"
#MODEL="meta-llama/Llama-3.1-8B-Instruct"
#MODEL_ALIAS="llama3.1-8b"
#MODEL="Qwen/Qwen2.5-14B-Instruct"
#MODEL_ALIAS="qwen2.5-14b"
SEEDS=(0 1 2 3 4)
VENV=$PROJECT/.venv
SIF=$PROJECT/containers/vllm-gptoss.sif

if [ "$MODEL_ALIAS" = "oss-20b" ]; then
    VLLM_CONFIG=$PROJECT/NeuroPathlogy/config/GPT-OSS_Hopper.yaml
else
    VLLM_CONFIG=$PROJECT/NeuroPathlogy/config/Llama_Qwen_Hopper.yaml
fi

VLLM_CONFIG_CONTAINER="/NeuroPathlogy/config/$(basename $VLLM_CONFIG)"

if [ "$RUN_MODE" = "gt" ]; then
    REPORTS=$PROJECT/reports_gt
    OUTPUT=$PROJECT/output_gt/$MODEL_ALIAS
    LOGS=$PROJECT/logs/neuropath_extractor_gt
    CHUNK_SIZE=4
elif [ "$RUN_MODE" = "prod" ]; then
    REPORTS=$PROJECT/reports
    OUTPUT=$PROJECT/output/$MODEL_ALIAS
    LOGS=$PROJECT/logs/neuropath_extractor
    CHUNK_SIZE=18
else
    echo "[error] Unknown RUN_MODE: $RUN_MODE. Must be 'gt' or 'prod'."
    exit 1
fi

echo "[config] RUN_MODE=$RUN_MODE | MODEL_ALIAS=$MODEL_ALIAS | CHUNK_SIZE=$CHUNK_SIZE"
echo "[config] OUTPUT=$OUTPUT"

mkdir -p "$OUTPUT"
mkdir -p "$PROJECT/cache_vllm"

# ── Load system modules ────────────────────────────────────────────────────
module load python/3.12
module load apptainer

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

# ── Environment ────────────────────────────────────────────────────────────
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16

# ── Confirm GPU ────────────────────────────────────────────────────────────
echo "[gpu] GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Find a free port for vLLM server ──────────────────────────────────────
VLLM_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
VLLM_URL="http://localhost:$VLLM_PORT"
echo "[vllm] Using port: $VLLM_PORT"

# ── Start vLLM server ──────────────────────────────────────────────────────
echo "[vllm] Starting vLLM server at $(date)..."

apptainer exec --nv \
    --cleanenv \
    --bind $PROJECT/cache_hf:/cache_hf \
    --bind $PROJECT/cache_vllm:/cache_vllm \
    --bind $PROJECT/NeuroPathlogy:/NeuroPathlogy \
    --env HF_HOME=/cache_hf \
    --env HF_HUB_OFFLINE=1 \
    --env PYTHONNOUSERSITE=1 \
    --env VLLM_CACHE_ROOT=/cache_vllm \
    $SIF \
    vllm serve $MODEL \
    --config $VLLM_CONFIG_CONTAINER \
    --port $VLLM_PORT \
    > $LOGS/vllm_server_${SLURM_JOB_ID}.out \
    2> $LOGS/vllm_server_${SLURM_JOB_ID}.err &

VLLM_PID=$!
echo "[vllm] Server PID: $VLLM_PID"

# ── Wait for vLLM server to be ready (up to 10 mins) ──────────────────────
echo "[vllm] Waiting for server to be ready..."
READY=0
for i in $(seq 1 120); do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        READY=1
        echo "[vllm] Server ready after $((i*5)) seconds at $(date)"
        break
    fi
    sleep 5
done

if [ $READY -eq 0 ]; then
    echo "[error] vLLM server did not start within 10 minutes"
    echo "Last 50 lines of server log"
    tail -50 $LOGS/vllm_server_${SLURM_JOB_ID}.err
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# ── Build report chunk for this array task ─────────────────────────────────
ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

ALL_REPORTS=()
for f in "$REPORTS"/*.pdf "$REPORTS"/*.txt; do
    [ -f "$f" ] && ALL_REPORTS+=("$f")
done

TOTAL=${#ALL_REPORTS[@]}
START=$((ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))
if [ $END -gt $TOTAL ]; then
    END=$TOTAL
fi

echo "[run] Array task $ARRAY_TASK_ID | Reports $START-$((END-1)) of $TOTAL total"
echo "[run] Model: $MODEL | Seeds: ${SEEDS[*]} | Starting at $(date)"

if [ $START -ge $TOTAL ]; then
    echo "[run] No reports for this array task, exiting."
    kill $VLLM_PID 2>/dev/null
    exit 0
fi

CHUNK=("${ALL_REPORTS[@]:$START:$((END - START))}")

# ── Run extraction ─────────────────────────────────────────────────────────
for REPORT in "${CHUNK[@]}"; do
    [ -f "$REPORT" ] || continue

    BASENAME=$(basename "$REPORT")
    echo "[extract] $BASENAME -> $OUTPUT/${BASENAME%.*}_seed*.extracted.json"

    python "$PROJECT/NeuroPathlogy/src/primary_extraction/main.py" \
        --input             "$REPORT"          \
        --output-dir        "$OUTPUT"          \
        --model             "$MODEL_ALIAS"     \
        --seeds             ${SEEDS[@]}        \
        --vllm-url          "$VLLM_URL"        \
        --temperature       0.01               \
        --top-p             0.9                \
        --reasoning-effort  medium             \
        --max-new-tokens    16384              \
        --max-retries       3                  \
        --logprobs
done

echo "[run] Done at $(date)"

# ── Shutdown vLLM server ───────────────────────────────────────────────────
echo "[vllm] Shutting down server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
echo "[vllm] Server shut down at $(date)"