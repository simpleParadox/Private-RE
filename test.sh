#!/bin/bash
#SBATCH --mail-user=kundu2@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
# SBATCH --array=0-2
#SBATCH --gres=gpu:a100:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=6G       # memory per node
#SBATCH --time=2:30:00      # time (DD-HH:MM)
#SBATCH --job-name=testing_semeval_epoch_0_100_Adam_batch10_nonprivate
#SBATCH --output=out_files/%x-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-afyshe-ab

module load cuda cudnn 
source ~/DP_RE/bin/activate
# python store_embeds.py
# python project_622_semeval.py 5 $SLURM_ARRAY_TASK_ID
python project_622_semeval.py 10.0 5 $SLURM_ARRAY_TASK_ID

#### For testing purposes only.
###### salloc --cpus-per-task=1 --account=def-afyshe-ab --time=00:30:10 --mem=6G --gres=gpu:a100:1