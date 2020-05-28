#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name="ML_ms"

ARGS=('concurrent' 'parallel')

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/cond_pkgs

/scratch-ssd/oatml/scripts/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f my_env.yml
source /scratch-ssd/oatml/miniconda3/bin/activate lisa_margliknew


srun python linear_combo_extended.py\
 --seed 64\
 --reps 100\
 --train_type ${ARGS[$SLURM_ARRAY_TASK_ID]} 