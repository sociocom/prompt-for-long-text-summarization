import logging
import torch
import os
# import sys
import numpy as np
# from typing import Dict
from tqdm import tqdm
from torch.utils.data import DataLoader

# from transformers import Trainer, EarlyStoppingCallback
# from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoConfig, AutoTokenizer
from transformers import get_linear_schedule_with_warmup, set_seed

from peft import PrefixTuningConfig, TaskType, get_peft_model
from accelerate import Accelerator
from datasets import load_dataset
import evaluate

import wandb
import os
os.environ['WANDB_DIR'] = os.getcwd() + '/wandb/'
os.environ['WANDB_CACHE_DIR'] = os.getcwd() + '/wandb/.cache/'
os.environ['WANDB_CONFIG_DIR'] = os.getcwd() + '/wandb/.config/'

logger = logging.getLogger(__name__)

from arguments import get_args

from model.summarization import BartPrefixForConditionalGeneration
from config.custom_config import PromptBartConfig
from utils import evaluate_utils, trace_malloc

def main():
    # args = get_args()
    # print(args)
    # ================================== 1. 定义模型的超参数 ================================== 
    accelerator = Accelerator() # device_placement="cuda:0"
    model_name_or_path = "facebook/bart-base"
    dataset_name = "cnn_dailymail"
    peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        num_virtual_tokens=20,
    )
    text_column = 'article'
    label_column = 'highlights'
    lr = 3e-3
    num_epochs = 10
    batch_size = 8
    seed = 42
    do_test = True
    set_seed(seed)
    # ================================== 2. 加载数据集 =======================================
    cnn_dataset = load_dataset(dataset_name, "3.0.0")

    # ================================== 2.1 加载tokenizer ======================================
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    target_max_length = 256
    # ================================== 2.2 数据预处理 ======================================
    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[label_column]
        model_inputs = tokenizer(
            inputs, 
            max_length=2048,
            padding=False,
            truncation=True
        ) 
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
    
    with accelerator.main_process_first():
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
    # 计算需要取出的样本数量
    train_size = int(len(cnn_dataset["train"]) * 0.001)
    eval_size = int(len(cnn_dataset["validation"]) * 0.01)
    test_size = int(len(cnn_dataset["test"]) * 0.01)

    # 从打乱后的数据集中随机抽取指定数量的数据
    train_dataset = cnn_dataset["train"].shuffle(seed=42).select(range(train_size))
    eval_dataset = cnn_dataset["validation"].shuffle(seed=42).select(range(eval_size))
    test_dataset = cnn_dataset["test"].shuffle(seed=42).select(range(test_size))
    
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
    bart_config = AutoConfig.from_pretrained(model_name_or_path)
    custom_config = PromptBartConfig(**bart_config.to_dict())
    model = BartPrefixForConditionalGeneration(
        checkpoint=model_name_or_path,
        config=custom_config,
        peft_config=peft_config,
        accelerator=accelerator,
    )
    model.model.print_trainable_parameters()
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * num_epochs,
    )
    
    # accelerator
    model, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, train_dataloader, eval_dataloader, test_dataloader
    )
    accelerator.print(model)
    
    # ================================== 可以省略 ======================================
    # 如果不设置 deepspeed 参数，则 Accelerator 默认使用 PyTorch 的原生分布式训练机制
    """
    zero_stage == 0: 模型参数的梯度更新是全局同步的，即所有设备在计算完梯度后会同步梯度并更新模型参数。
    zero_stage == 1: 模型参数的梯度更新是本地同步的，每个设备在计算完梯度后先更新本地模型参数，然后再全局同步更新模型参数。
    zero_stage == 2: 模型参数的梯度更新是本地异步的，每个设备在计算完梯度后立即更新本地模型参数，不进行全局同步更新。
    zero_stage == 3: 模型参数的梯度更新是本地异步的，每个设备在计算完梯度后立即更新本地模型参数，并与其他设备进行全局异步同步。
    """
    is_ds_zero3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None): 
        # zero_stage==3 表示
        is_ds_zero3 = accelerator.state.deepspeed_plugin.zero_stage == 3    
         
    # ================================== 4. 训练模型 ======================================
    print('Start training ...')
    print(accelerator.device)  
    
    for epoch in range(num_epochs):
        print("================================== epoch {} ==================================".format(epoch))
        with trace_malloc.TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss) 
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()    
                
        # ================================== 可以省略: 计算存储消耗 ======================================     
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the train : {}".format(trace_malloc.b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + trace_malloc.b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the train : {}".format(trace_malloc.b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + trace_malloc.b2mb(tracemalloc.cpu_begin)
            )
        )               
        # ================================== 4.1 评估训练集 ======================================
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")    
                       
        # ================================== 4.2 评估验证集 ======================================
        # TODO: 加上清空缓存
        model.eval()
        print("\n")
        print("Start evaluating ...")
        with trace_malloc.TorchTracemalloc() as tracemalloc:
            with torch.no_grad():
                eval_total_loss = 0
                for step, batch in enumerate(tqdm(eval_dataloader)):
                    outputs = model(**batch)
                    loss = outputs.loss
                    eval_total_loss += loss.detach().float()
        eval_epoch_loss = eval_total_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        accelerator.print(f"{epoch=}: {eval_ppl=} {eval_epoch_loss=}")      
          
        # ================================== 5. 评估测试集 ======================================
        if do_test := True:
            if (epoch+1) % 1 == 0:
                model.eval()
                rouge_metric = evaluate.load("rouge")
                summarization_metric = evaluate_utils.SummarizationMetric()
                summarization_metric.calculate_metrics(
                    dataloader=test_dataloader,
                    metric=rouge_metric,
                    model=model,
                    tokenizer=tokenizer,
                    accelerator=accelerator,
                    target_max_length=target_max_length,
                ) 
                summarization_metric.show_metrics()
                # summarization_metric.show_samples()
            
    # ================================== 6. 保存模型 ======================================
    accelerator.wait_for_everyone()
    model.push_to_hub(
        "kaifanli/"
        + f"prefix_rmt_bart_cnn-dm",
        state_dict=accelerator.get_state_dict(model),
        use_auth_token=True,
    )
    accelerator.wait_for_everyone()        
    
if __name__ == "__main__":
    main()