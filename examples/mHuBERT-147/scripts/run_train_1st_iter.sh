#!/bin/bash
#SBATCH -n 4
#SBATCH -N 4
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --mem=1900G
#SBATCH --output=train-mhubert-%j.log
#SBATCH --time=360:00:00

FAIRSEQ=<DIR>
LOG_OUT=<DIR>
DATA=<DIR>

CONFIG_DIR="$FAIRSEQ/examples/mHuBERT-147/config/pretrain/"
CONFIG="mhubert147_base"
RATE=100

conda activate <environment>


srun python $FAIRSEQ/fairseq_cli/hydra_train.py \
--config-dir $CONFIG_DIR \
--config-name $CONFIG \
checkpoint.save_dir=$LOG_OUT/checkpoints/ \
distributed_training.distributed_world_size=32 \
optimization.update_freq=[1] \
common.tensorboard_logdir=$LOG_OUT/tensorboard/ \
task.data=$DATA \
task.label_dir=$DATA \
task.numpy_folder=$DATA \
task.check_alignment=True \ #change this to False after first launch (slow alignment check)
task.labels='["km"]' \
model.label_rate=$RATE
