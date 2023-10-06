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
use_wandb="wandb" # Set to "none" to disable wandb tracking, or "wandb" to enable it.
export DISPLAY_NAME=BaseModel-CnnDailymail
export RUN_ID=1
export WANDB_MODE=online
export LINEAGE=BaseModel-CnnDailymail # This is just a tag on wandb to make tracking runs easier
export WANDB_PROJECT_NAME="kaifan-li/BaseModel-CnnDailymail" # IMPORTANT: set this to your own wandb project

export MODEL_NAME=facebook/bart-base
export DATASET_NAME=cnn_dailymail
export MODEL_DIR_NAME=bart-base
checkpoint_dir=saved/$DATASET_NAME/$MODEL_DIR_NAME/$WANDB_NAME/ 

# 执行命令并将输出重定向到日志文件
# nohup accelerate launch run.py \
nohup python3 run_summarization.py \
--model_name_or_path "$MODEL_NAME" \
--dataset_name "$DATASET_NAME" \
--dataset_config_name "3.0.0" \
--output_dir "$checkpoint_dir" \
--do_train true \
--do_eval true \
--do_predict true \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--num_train_epochs 5 \
--max_train_samples 10000 \
--max_eval_samples 1000 \
--max_predict_samples 1000 \
--max_source_length 1024 \
--max_target_length 128 \
--pre_seq_len 0 \
--post_seq_len 0 \
--training_strategy "BaseModel" \
--predict_with_generate \
"$@" > $log_filename 2>&1 &
