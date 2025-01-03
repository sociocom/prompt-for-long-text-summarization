#!/bin/bash

# set time zone to Japan/Tokyo
export TZ=Asia/Tokyo
export MODEL_NAME=facebook/bart-base
export DATASET_NAME=pubmed-incremental
export MODEL_DIR_NAME=bart-base
export MODEL_TYPE=BaseModel
checkpoint_dir=saved/$DATASET_NAME/$MODEL_DIR_NAME/$MODEL_TYPE/$WANDB_NAME

# create log folder if it doesn't exist
current_date=$(date +'%Y_%m_%d')
log_folder="logs/$DATASET_NAME/$MODEL_DIR_NAME/$MODEL_TYPE/${current_date}"
mkdir -p $log_folder

# create log file
current_datetime=$(date +'%Y_%m_%d_%H_%M')
log_filename="${log_folder}/logs_${current_datetime}.txt"

# # Weights and biases (wandb) related config. Set use_wandb=none if you don't want to use wandb.
use_wandb=none # Set to "none" to disable wandb tracking, or "wandb" to enable it.
export DISPLAY_NAME=BartBase-RMT
export RUN_ID=1
export WANDB_MODE=online
export LINEAGE=BartBase-RMT # This is just a tag on wandb to make tracking runs easier
export WANDB_PROJECT_NAME="kaifan-li/Incremental_Summarization" # IMPORTANT: set this to your own wandb project

# 执行命令并将输出重定向到日志文件
# nohup accelerate launch run.py \
# --length_penalty=2.0 \

# for pre_seq_len in 0 20 32 64 128 150 200 212 256 300 
# do
#     for post_seq_len in 0 300  
#     do

pre_seq_len=0
post_seq_len=0
max_source_length=$((pre_seq_len + post_seq_len + 2048))
export WANDB_NAME=$DISPLAY_NAME-$pre_seq_len-$post_seq_len
# export log_filename="${log_folder}/logs_${current_datetime}_${pre_seq_len}_${post_seq_len}.txt"

CUDA_VISIBLE_DEVICES=1 nohup python3 run_summarization.py \
--model_name_or_path "$MODEL_NAME" \
--dataset_name "$DATASET_NAME" \
--output_dir "$checkpoint_dir" \
--overwrite_output_dir \
--do_train true \
--do_eval false \
--do_predict true \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--num_train_epochs 1 \
--max_train_samples 100000 \
--max_eval_samples 10000 \
--max_predict_samples 10000 \
--max_source_length $max_source_length \
--max_n_segments 4 \
--max_target_length 300 \
--pre_seq_len $pre_seq_len \
--post_seq_len $post_seq_len \
--generation_num_beams 4 \
--save_total_limit 2 \
--evaluation_strategy epoch \
--save_strategy epoch \
--metric_for_best_model rouge1 \
--load_best_model_at_end True \
--model_type "$MODEL_TYPE" \
--task_type "Normal" \
--rouge_type "Accumulation" \
--predict_with_generate \
--freeze_model False \
--skip_memory_metrics False \
"$@" > $log_filename 2>&1 &

#     done
# done

