#!/bin/bash

# set time zone to Japan/Tokyo
export TZ=Asia/Tokyo

# create log folder if it doesn't exist
current_date=$(date +'%Y_%m_%d')
log_folder="logs/${current_date}"
mkdir -p $log_folder

# create log file
current_datetime=$(date +'%Y_%m_%d_%H_%M')
log_filename="${log_folder}/logs_${current_datetime}.txt"

# Weights and biases (wandb) related config. Set use_wandb=none if you don't want to use wandb.
use_wandb=none # Set to "none" to disable wandb tracking, or "wandb" to enable it.
export DISPLAY_NAME=PromptRMT-bart-base
export RUN_ID=1
export WANDB_MODE=online
export LINEAGE=PromptRMT # This is just a tag on wandb to make tracking runs easier
export WANDB_PROJECT_NAME="<ORG>/<PROJECT_NAME>" # IMPORTANT: set this to your own wandb project

export MODEL_NAME=facebook/bart-base
export DATASET_NAME=cnn_dailymail
checkpoint_dir=saved/$DATASET_NAME/$WANDB_NAME/ 

batch_size=4
eval_batch_size=8

# 执行命令并将输出重定向到日志文件
nohup python3 run.py \
--model_name_or_path "$MODEL_NAME" \
--dataset_name "$DATASET_NAME" \
--output_dir "$checkpoint_dir" \
--do_train true \
--do_eval true \
--do_predict true \
--per_device_train_batch_size $batch_size \
--per_device_eval_batch_size $eval_batch_size \
--predict_epoch 2 \
--num_train_epochs 10 \
--dataset_percentage 0.1 \
"$@" > $log_filename 2>&1 &
