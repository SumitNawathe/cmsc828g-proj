#!/bin/bash
#SBATCH --job-name=sched_inf
#SBATCH --output=/cmlscratch/snawathe/cmsc828g-proj/logs/sched_inf_%j.log
#SBATCH --error=/cmlscratch/snawathe/cmsc828g-proj/logs/sched_inf_%j.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4          # 4 CPU cores dedicated per GPU worker
#SBATCH --mem=64G
#SBATCH --qos=cml-high
#SBATCH --account=cml-sfeizi
#SBATCH --partition=cml-sfeizi

module purge
module load cmake/4.0.3
module load clang/12.0.0
module load gmp/6.3.0
module load gcc/14.2.0
module load cuda/13.1.1

# Source environment variables
source env_vars.sh
cd $PROJECT_ROOT

# ─── Parameters (edit these) ─────────────────────────────────────────────────
MODEL="qwen3-4b"
DATASET="math100"
NUM_GPUS=2
NUM_EPOCHS=8
GROUP_SIZE=4
MAX_NEW_TOKENS=32768
STRATEGY="baseline"           # "baseline", "logt_max", or "logt_sum"
BATCH_GROUPING=2              # Number of prompts per batch in baseline
GEN_STRATEGY="recursive_retry"  # GPU generation strategy (see GPU_GENERATE_REGISTRY)
MEM_LIMIT=250000              # KV-cache token budget (empirically determined for L40s)
ALPHA=0.001                    # OOM risk threshold (logt only)

# ─── Run ─────────────────────────────────────────────────────────────────────
srun $PYTHON_EXEC run_inference.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --num-gpus "$NUM_GPUS" \
    --num-epochs "$NUM_EPOCHS" \
    --group-size "$GROUP_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --strategy "$STRATEGY" \
    --batch-grouping "$BATCH_GROUPING" \
    --gen-strategy "$GEN_STRATEGY" \
    --mem-limit "$MEM_LIMIT" \
    --alpha "$ALPHA"
