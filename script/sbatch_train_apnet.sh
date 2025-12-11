#!/bin/bash --login
#SBATCH --job-name=apnet_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=2
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --qos=dw87
# SBATCH --partition=cs,cs2


mamba activate agnet

python /home/ssgardin/nobackup/autodelete/AggNet/script/train_apnet.py /home/ssgardin/nobackup/autodelete/AggNet/data/100K/massive_exp__aggregation__thompson.xlsx