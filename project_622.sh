#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-4
#SBATCH --gres=gpu:a100:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=15G       # memory per node
#SBATCH --time=6:00:00      # time (DD-HH:MM)
#SBATCH --job-name=dp_re_all_seeds_epoch_0_5_table_data_small_80_20_split_tf_encoding_Adam_001_private
#SBATCH --output=out_files/%x-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-afyshe-ab

module load cuda cudnn
source ~/DP_RE/bin/activate
# python store_embeds.py
# Order of command-line args -> Noise_multiplier, epochs, seed
python project_622.py 20.0 5 $SLURM_ARRAY_TASK_ID

#### For testing purposes only.
###### salloc --cpus-per-task=1 --account=def-afyshe-ab --time=00:30:10 --mem=6G --gres=gpu:a100:1