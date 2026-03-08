#!/bin/bash
#SBATCH --job-name=chronos-ft-cpu
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ----------------------------
# Environment setup
# ----------------------------
module purge
module load python/3.11.0-icl

if [ -f "$SLURM_SUBMIT_DIR/d200/bin/activate" ]; then
  source "$SLURM_SUBMIT_DIR/d200/bin/activate"
elif [ -f "$HOME/venvs/media/bin/activate" ]; then
  source "$HOME/venvs/media/bin/activate"
else
  echo "No virtual environment found (checked $SLURM_SUBMIT_DIR/d200 and $HOME/venvs/media)."
  exit 1
fi

python -c "import sys; print(sys.executable); print(sys.version)"

# Keep caches off HOME
export HF_HOME=/home/rjl202/media/.hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=/home/rjl202/media/.torch_cache
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME"

# Avoid CPU oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

# ----------------------------
# Run Chronos fine-tuning
# ----------------------------
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python cronos_finetune.py
