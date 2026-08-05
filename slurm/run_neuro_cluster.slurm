#!/bin/bash

# ── SLURM Directives ───────────────────────────────────────────────────────
#SBATCH -A r01042                                                         # Project allocation
#SBATCH -J neuropath_cluster                                              # Job name
#SBATCH -o /N/project/ADRD/neuropathoroot/logs/%x/cluster_%j.out          # STDOUT (one file per job)
#SBATCH -e /N/project/ADRD/neuropathoroot/logs/%x/cluster_%j.err          # STDERR
#SBATCH -t 2:00:00                                                        # 2 hours
#SBATCH -c 16                                                             # CPU cores
#SBATCH --mem=256G                                                        # Memory
#SBATCH -p hopper                                                         # Hopper partition (H100s)
#SBATCH -q hopper                                                         # QOS for hopper
#SBATCH --gres=gpu:h100:1                                                 # 1x H100 GPU
#SBATCH --mail-type=END,FAIL                                              # Email on finish or fail
#SBATCH --mail-user=aantappa@iu.edu                                       # Your email



# ── Configuration ──────────────────────────────────────────────────────────
PROJECT=/N/project/ADRD/neuropathoroot
MODEL="openai/gpt-oss-20b"
MODEL_ALIAS="oss-20b"
#MODEL="meta-llama/Llama-3.1-8B-Instruct"
#MODEL_ALIAS="llama3.1-8b"
#MODEL="Qwen/Qwen2.5-14B-Instruct"
#MODEL_ALIAS="qwen2.5-14b"
DISCOVERY_DIR=$PROJECT/output/$MODEL_ALIAS/residual_discovery
OUTPUT_DIR=$PROJECT/output/$MODEL_ALIAS/residual_schema

VENV=$PROJECT/.venv
SIF=$PROJECT/containers/vllm-gptoss.sif
RDDNP=$PROJECT/csv_input/rdd-np.csv
LOGS=$PROJECT/logs/neuropath_cluster

if [ "$MODEL_ALIAS" = "oss-20b" ]; then
    VLLM_CONFIG=$PROJECT/NeuroPathlogy/config/GPT-OSS_Hopper.yaml
else
    VLLM_CONFIG=$PROJECT/NeuroPathlogy/config/Llama_Qwen_Hopper.yaml
fi

VLLM_CONFIG_CONTAINER="/NeuroPathlogy/config/$(basename $VLLM_CONFIG)"

echo "[config] MODEL_ALIAS=$MODEL_ALIAS"
echo "[config] DISCOVERY_DIR=$DISCOVERY_DIR"
echo "[config] OUTPUT_DIR=$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT/cache_vllm"

# ── Load modules ───────────────────────────────────────────────────────────
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

# ── Find a free port ───────────────────────────────────────────────────────
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
    --bind $PROJECT/csv_input:/csv_input \
    --bind $PROJECT/output:/output \
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

# ── Wait for vLLM server ───────────────────────────────────────────────────
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
    tail -50 $LOGS/vllm_server_${SLURM_JOB_ID}.err
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# ── Run Cluster ────────────────────────────────────────────────
echo "[run] Starting Cluster at $(date)..."

python "$PROJECT/NeuroPathlogy/src/residual_extraction/cluster.py" \
    --discovery-dir         "$DISCOVERY_DIR"     \
    --output-dir            "$OUTPUT_DIR"        \
    --rddnp-path            "$RDDNP"             \
    --vllm-url              "$VLLM_URL"          \
    --model                 "$MODEL"             \
    --embed-device          cuda                 \
    --embed-batch-size      64                   \
    --umap-n-components     50                   \
    --umap-n-neighbors      15                   \
    --hdbscan-min-cluster-size  8                \
    --hdbscan-min-samples       3                \
    --max-label-retries     3

echo "[run] Cluster done at $(date)"

# ── Shutdown vLLM ─────────────────────────────────────────────────────────
echo "[vllm] Shutting down server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
echo "[vllm] Server shut down at $(date)"