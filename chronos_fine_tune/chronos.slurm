#!/bin/bash
#SBATCH --job-name=chronos-ft-cpu
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=chronos_fine_tune/%x-%j.out
#SBATCH --error=chronos_fine_tune/%x-%j.err

set -euo pipefail

# ----------------------------
# Environment setup
# ----------------------------
module purge
module load python/3.11.0-icl

if [ -n "${D200_VENV:-}" ] && [ -f "$D200_VENV/bin/activate" ]; then
  source "$D200_VENV/bin/activate"
elif [ -f "$SLURM_SUBMIT_DIR/d200/bin/activate" ]; then
  source "$SLURM_SUBMIT_DIR/d200/bin/activate"
elif [ -f "$HOME/venvs/media/bin/activate" ]; then
  source "$HOME/venvs/media/bin/activate"
else
  echo "No virtual environment found (checked D200_VENV, $SLURM_SUBMIT_DIR/d200, and $HOME/venvs/media)."
  exit 1
fi

python -c "import sys; print(sys.executable); print(sys.version)"

# Keep caches out of $HOME without hardcoding a user-specific path.
CACHE_ROOT="${TMPDIR:-$SLURM_SUBMIT_DIR/.cache}"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME"

# Avoid CPU oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

# ----------------------------
# Run Chronos fine-tuning
# ----------------------------
cd "$SLURM_SUBMIT_DIR"

python "$SLURM_SUBMIT_DIR/chronos_fine_tune/cronos_finetune.py"
