#!/bin/bash
#SBATCH --job-name=inf
#SBATCH --output /cmlscratch/snawathe/cmsc828g-proj/logs/inf_%j.log
#SBATCH --error /cmlscratch/snawathe/cmsc828g-proj/logs/inf_%j.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --qos=cml-medium
#SBATCH --account=cml-sfeizi
#SBATCH --partition=cml-sfeizi
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

module purge
module load cmake/4.0.3
module load clang/12.0.0
module load gmp/6.3.0
module load gcc/14.2.0
module load cuda/13.1.1

# Source environment variables
source env_vars.sh
cd $PROJECT_ROOT

# Define the chunks you want to process in this job
CHUNKS=(0 1 2 3)
NUM_GPUS=4

for i in "${!CHUNKS[@]}"; do
    GPU_ID=$((i % NUM_GPUS))
    CHUNK_IDX=${CHUNKS[$i]}
    
    echo "Starting chunk $CHUNK_IDX on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_EXEC basic_inference_hf.py \
        --model qwen3-4b \
        --dataset math500 \
        --num-chunks 100 \
        --chunk-idx $CHUNK_IDX \
        -B 32768 \
        -N 4 \
        -G 4 &
    
    # Stagger startups to avoid simultaneous model loading spikes
    sleep 20
done

# Wait for all background processes to finish
wait
echo "All chunks completed."
