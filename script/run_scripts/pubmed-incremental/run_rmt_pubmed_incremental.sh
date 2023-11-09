#!/bin/bash

# set time zone to Japan/Tokyo
export TZ=Asia/Tokyo
export MODEL_NAME=facebook/bart-base
export DATASET_NAME=pubmed-incremental
export MODEL_DIR_NAME=bart-base
export MODEL_TYPE=BaseModelWithRMT
checkpoint_dir=saved/$DATASET_NAME/$MODEL_DIR_NAME/$MODEL_TYPE/$WANDB_NAME

# create log folder if it doesn't exist
current_date=$(date +'%Y_%m_%d')
log_folder="logs/$DATASET_NAME/$MODEL_DIR_NAME/$MODEL_TYPE/${current_date}"
mkdir -p $log_folder

# create log file
current_datetime=$(date +'%Y_%m_%d_%H_%M')
log_filename="${log_folder}/logs_${current_datetime}.txt"

# # Weights and biases (wandb) related config. Set use_wandb=none if you don't want to use wandb.
# use_wandb="wandb" # Set to "none" to disable wandb tracking, or "wandb" to enable it.
# export DISPLAY_NAME=BaseModel-CnnDailymail
# export RUN_ID=1
# export WANDB_MODE=online
# export LINEAGE=BaseModel-CnnDailymail # This is just a tag on wandb to make tracking runs easier
# export WANDB_PROJECT_NAME="kaifan-li/BaseModel-CnnDailymail" # IMPORTANT: set this to your own wandb project

# 执行命令并将输出重定向到日志文件
# nohup accelerate launch run.py \
nohup python3 run_summarization.py \
--model_name_or_path "$MODEL_NAME" \
--dataset_name "$DATASET_NAME" \
--output_dir "$checkpoint_dir" \
--overwrite_output_dir \
--do_train true \
--do_eval true \
--do_predict true \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--num_train_epochs 5 \
--max_train_samples 100000 \
--max_eval_samples 5000 \
--max_predict_samples 5000 \
--max_source_length 1024 \
--max_target_length 300 \
--pre_seq_len 20 \
--post_seq_len 300 \
--generation_num_beams 4 \
--save_total_limit 1 \
--evaluation_strategy epoch \
--save_strategy epoch \
--load_best_model_at_end True \
--model_type "$MODEL_TYPE" \
--task_type "Segment" \
--predict_with_generate \
"$@" > $log_filename 2>&1 &
