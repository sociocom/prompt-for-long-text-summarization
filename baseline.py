import gc
import os
import sys
import threading

import numpy as np
import psutil
import torch

from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader

from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from peft import PrefixTuningConfig, TaskType, get_peft_model

# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)

# 用于跟踪和记录代码的内存使用情况
class TorchTracemalloc:
    # 收集垃圾和清空GPU缓存。
    # 重置 GPU 的最大内存使用量（peak memory allocated）计数器，将其设为零。
    # 记录进入上下文管理器时的 GPU 内存使用情况和 CPU 内存使用情况，并启动一个线程用于监控 CPU 内存的峰值。
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self
    
    # 获取进入上下文管理器时的 GPU 内存使用情况和 CPU 内存使用情况，并停止监控 CPU 内存的峰值。
    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    # 停止监控 CPU 内存的峰值。
    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    # 停止监控 CPU 内存的线程。
    # 再次收集垃圾和清空GPU缓存。
    # 记录退出上下文管理器时的 GPU 内存使用情况和 CPU 内存使用情况，并计算内存使用量的差值，得到实际的内存消耗和峰值内存消耗。
    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")
        
        return False
    
def main():
    # ================================== 1. 定义模型的超参数 ================================== 
    accelerator = Accelerator() # device_placement="cuda:0"
    model_name_or_path = "facebook/bart-large"
    dataset_name = "cnn-daily-mail"
    peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        num_virtual_tokens=20,
    )
    text_column = 'article'
    label_column = 'highlights'
    lr = 1e-4
    num_epochs = 10
    batch_size = 8
    seed = 42
    do_test = False
    set_seed(seed)
    
    # ================================== 2. 加载数据集 =======================================
    cnn_dataset = load_dataset(dataset_name, "3.0.0")
    
    # ================================== 2.1 加载tokenizer ======================================
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    target_max_length = max([len(tokenizer(x, truncation=True, padding='max_length')['input_ids']) for x in cnn_dataset['train'][label_column]])
    
    # ================================== 2.2 数据预处理 ======================================
    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[label_column]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True) # 这里暂时不padding
        targets = tokenizer(
            targets,
            max_length=target_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        targets = targets['input_ids']
        targets[targets == tokenizer.pad_token_id] = -100
        model_inputs['labels'] = targets
        
        return model_inputs
    
    with accelerator.main_process_context():
        cnn_dataset = cnn_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=cnn_dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()
    
    # ================================== 2.3 数据加载器 ======================================
    train_dataset = cnn_dataset["train"]
    eval_dataset = cnn_dataset["validation"]
    test_dataset = cnn_dataset["test"]
    
    def collate_fn(examples):
        return tokenizer.pad(examples, padding='longest', return_tensors='pt')
    
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        pin_memory=True, # 将数据加载到固定的内存中，可以加速数据加载
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        pin_memory=True,
    )        
    
    # ================================== 3. 加载模型 ======================================
    
    # TODO: 写一个rouge的评估函数