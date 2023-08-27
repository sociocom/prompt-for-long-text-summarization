#!/bin/bash

# 设置时区为东京时区
export TZ=Asia/Tokyo

# 获取当前日期
current_date=$(date +'%Y_%m_%d')
# 创建日志文件夹
log_folder="logs/${current_date}"
mkdir -p $log_folder

# 获取当前日期时间
current_datetime=$(date +'%Y_%m_%d_%H_%M')
# 定义日志文件名
log_filename="${log_folder}/logs_${current_datetime}.txt"

# 执行命令并将输出重定向到日志文件
nohup python3 run.py > $log_filename 2>&1 &
