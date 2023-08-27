import evaluate
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from .trace_malloc import * 

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
                          batch_size=16,
                          column_text="article",
                          column_summary="highlights"):
        
        with TorchTracemalloc() as tracemalloc:
            for step, batch in enumerate(tqdm(dataloader)):
                labels = batch["labels"]
                print('labels: ', labels.shape, type(labels))
                # 对于peft的模型直接收**batch的形式
                # 不能单独传batch和attention_mask
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        **batch,
                        # sysnced_gpus=is_ds_zero3,
                        length_penalty=0.8,
                        num_beams=4,
                        max_length=target_max_length,
                    )
                    print(f"generated_tokens: {generated_tokens}")
                # 当使用分布式训练时，不同设备或节点上的模型生成的输出可能有不同的长度。
                # 为了进行后续的评估和计算指标，我们需要将这些输出统一为相同的长度。
                # dim=1的维度是token的维度，这里的pad_index是tokenizer的pad_token_id
                
                generated_tokens = torch.stack([s for s in generated_tokens if s is not None])
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, 
                    dim=1,
                    pad_index=tokenizer.pad_token_id,
                )
                
                labels = accelerator.pad_across_processes(
                    labels, 
                    dim=1, 
                    pad_index=tokenizer.pad_token_id
                )
                
                # 将分布式计算的输出结果收集到主进程中
                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()
                
                # Replace -100 in the labels as we can't decode them
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                
                print('last_segments output: ', generated_tokens[-1])
                print('last_segment_output type: ', type(generated_tokens[-1]))
                print('labels:', labels)
                print('labels type: ', type(labels))
                
                generated_tokens = generated_tokens[-1]                
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                
                decoded_preds, decoded_labels = self.postprocess_text(
                    decoded_preds, decoded_labels
                )
                
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
        