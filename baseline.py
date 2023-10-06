import gc
import os
import sys
import threading

import numpy as np
import psutil
import torch
import random

from accelerate import Accelerator
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from peft import PrefixTuningConfig, TaskType, get_peft_model

# from model import summarization
from utils import trace_malloc, evaluate_utils

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

# 用于计算 ROUGE 指标的函数
class SummarizationMetric():
    def __init__(self):
        self.summary_samples = {
            "predict": [],
            "highlights": []
        }
        self.rouge_metrics = evaluate.load("rouge")
        self.bleu_metrics = evaluate.load("sacrebleu")
        
    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # ROUGE expects a newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
    
    def calculate_metrics(self,
                          dataloader, metric, 
                          model, tokenizer,
                          accelerator,
                          target_max_length,
                          batch_size=32,
                          column_text="article",
                          column_summary="highlights"):
        
        with TorchTracemalloc() as tracemalloc:
            for step, batch in enumerate(tqdm(dataloader)):
                labels = batch["labels"]
                # 对于peft的模型直接收**batch的形式
                # 不能单独传batch和attention_mask
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        **batch,
                        # sysnced_gpus=is_ds_zero3,
                        length_penalty=0.8,
                        num_beams=8,
                        max_length=target_max_length,
                    )
                # 当使用分布式训练时，不同设备或节点上的模型生成的输出可能有不同的长度。
                # 为了进行后续的评估和计算指标，我们需要将这些输出统一为相同的长度。
                # dim=1的维度是token的维度，这里的pad_index是tokenizer的pad_token_id
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1,
                    pad_index=tokenizer.pad_token_id,
                )
                
                labels = accelerator.pad_across_processes(
                    labels, dim=1, pad_index=tokenizer.pad_token_id
                )
                
                # 将分布式计算的输出结果收集到主进程中
                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()
                
                # Replace -100 in the labels as we can't decode them
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[-1]
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                decoded_preds, decoded_labels = self.postprocess_text(
                    decoded_preds, decoded_labels
                )
                print(f'{decoded_preds=}')
                print(f'{decoded_labels=}')
                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            
                # # 解码pred成自然语言文本
                # decoded_summaries = [tokenizer.decode(
                # seq, skip_special_tokens=True,
                # clean_up_tokenization_spaces=True)
                # for seq in preds]   
                                 
                # # 空字符转为空格, 将文本调整成自然语言的格式
                # decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
                
                # # reference解码
                # # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                # reference = np.where(reference != -100, reference, tokenizer.pad_token_id).cpu().numpy()
                # reference_texts = [tokenizer.decode(reference, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                #    for tokens in reference]
                # decoded_reference = [d.replace("", " ") for d in reference_texts]
                
                # metric.add_batch(predictions=decoded_summaries, references=decoded_reference)
                
                # # 每个batch添加2个sample
                # self.summary_samples["predict"].extend(decoded_summaries[:2])
                # self.summary_samples["highlights"].extend(decoded_reference[:2])
                
                # print(f"predict: {decoded_summaries[0]}")
                # print(f"highlights: {decoded_reference[0]}")
                
        # TODO: 包装成函数形式
        # ================================== 可以省略: 计算存储消耗 ======================================
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the eval : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )
        
        score = metric.compute()
        if isinstance(metric, type(self.rouge_metrics)):
            self.rouge_metrics = score
        elif isinstance(metric, type(self.rouge_metrics)):
            self.bleu_metrics = score
        return score

    def show_metrics(self):
        print(self.rouge_metrics)
        # print(self.bleu_metrics)
        
    # def show_samples(self):
    #     print(self.summary_samples[:10])
        
def main():
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
    num_epochs = 1
    batch_size = 1
    seed = 42
    do_test = True
    set_seed(seed)
    
    # ================================== 2. 加载数据集 =======================================
    cnn_dataset = load_dataset(dataset_name, "3.0.0")

    # ================================== 2.1 加载tokenizer ======================================
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    target_max_length = 142
    # target_max_length = max([len(tokenizer(x, truncation=True, padding='max_length')['input_ids']) for x in cnn_dataset['train'][label_column]])
    
    # ================================== 2.2 数据预处理 ======================================
    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[label_column]
        model_inputs = tokenizer(inputs, truncation=True) # 这里暂时不padding
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
    eval_size = int(len(cnn_dataset["validation"]) * 0.001)
    test_size = int(len(cnn_dataset["test"]) * 0.001)

    # 从打乱后的数据集中随机抽取指定数量的数据
    train_dataset = cnn_dataset["train"].shuffle(seed=42).select(range(100))
    eval_dataset = cnn_dataset["validation"].shuffle(seed=42).select(range(50))
    test_dataset = cnn_dataset["test"].shuffle(seed=42).select(range(50))
    
    # NOTE
    # 这里是为了动态填充batch，因为每个batch的样本长度不一样
    # 比如batch0最大长度为100，batch1最大长度为200，那么batch0的样本会被pad到200
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
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    accelerator.print(model)
    
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
        with TorchTracemalloc() as tracemalloc:
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
        accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
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
        with TorchTracemalloc() as tracemalloc:
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
        # TODO: 测试初始的rouge值
        if (epoch+1) % 1 == 0:
            model.eval()
            rouge_metric = evaluate.load("rouge")
            summarization_metric = SummarizationMetric()
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
        + f"prefix-tuning_cnn_bart-base",
        state_dict=accelerator.get_state_dict(model),
        use_auth_token=True,
    )
    accelerator.wait_for_everyone()
    
if __name__ == "__main__":
    main()
