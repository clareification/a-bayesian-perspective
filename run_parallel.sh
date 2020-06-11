srun python dnn_experiment.py\
 --seed ${ARGS[$SLURM_ARRAY_TASK_ID]}\
 --reps 1\
 --train_type 'parallel'
