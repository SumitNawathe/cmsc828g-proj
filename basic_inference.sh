#!/bin/bash
#SBATCH --job-name=inf
#SBATCH --output /cmlscratch/snawathe/cmsc828g-proj/logs/inf_%A_%a.log
#SBATCH --error /cmlscratch/snawathe/cmsc828g-proj/logs/inf_%A_%a.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --mem=32G
#SBATCH --array=0-99
#SBATCH --cpus-per-task=2

module purge
module load cmake/4.0.3
module load clang/12.0.0
module load gmp/6.3.0
module load gcc/14.2.0
module load cuda/12.8.1

# Source environment variables
source env_vars.sh
cd $PROJECT_ROOT

sleep $((SLURM_ARRAY_TASK_ID * 30))  # Stagger the start times of the jobs
# sleep $(( (SLURM_ARRAY_TASK_ID - 220) * 30))  # Stagger the start times of the jobs

# chunk 22 of 50
# => chunks 220 to 229 of 500

srun $PYTHON_EXEC basic_inference.py \
    --model qwen3-4b \
    --dataset math500 \
    --num-chunks 100 \
    --chunk-idx $SLURM_ARRAY_TASK_ID \
    -B 32768 \
    -N 32 \
	-G 4