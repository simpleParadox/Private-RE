#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:a100:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=60G       # memory per node
#SBATCH --time=5:30:00      # time (DD-HH:MM)
#SBATCH --job-name=seed_0_epoch_10_table_data_small_80_20_split_tf_encoding_Adam_0.001
#SBATCH --output=%x-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-afyshe-ab

module load cuda cudnn 
source ~/DP_RE/bin/activate
# python store_embeds.py
python project_622.py

#### For testing purposes only.
###### salloc --cpus-per-task=1 --account=def-afyshe-ab --time=00:30:10 --mem=10G --gres=gpu:a100:1 