# prompt-for-long-text-summarization

## project structure

```
project/
|-- config/
|   |-- config.py
|-- datasets/
|-- logs/
|-- models/
|   |-- base.py                   -> RMT base model
|   |-- modeling_bart             -> custom bart model (PrefixProp) 
|   |-- summarization             -> custom downstream class for our task
|   |-- prefix_encoder             -> for prompt generation
|-- utils/
|-- script/                       -> task specific run script
|-- run_summarization.py          -> training
|-- ...
|-- arguments.py                  -> training_args
|-- requirements.txt  
```
## How to use
### Important:
* due to evaluate.rouge don't support other language 
  * need to add a tokenizer=nltk.word_tokenize

## Dataset Statistic

### **Pubmed statistic:**

* Before processing
  * | split | samples |
    | :-: | :-: |
    | train | 119924 |
    | eval  | 6633 |
    | test  | 6658 |
* after processing
  * | split | samples | avg_token_size_section | avg_token_size_abstract |
    | :-: | :-: | :-: | :-: |
    | train | 24843 | 2740 | 299 |
    | eval | 1399 | 2752 | 300 |
    | test | 1431 | 2732 | 303 |

## Result:

### PubMed-Incremental-User_wise
> This is a mistake that concat all segments output to calculate rouge score

| Model | pre_seq_len | post_seq_len | max_source_length | max_target_length | rouge1 | rouge2 | rougeL | batch_size |  lr  | 
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Bart-base | 0 | 0 | 512 | 300 | 49.35 | 19.38 | 48.05 | 2 * 3 | 5e-5 |
| RMT | 20 | 0 | 512 | 300 | 48.94 | 18.76 | 47.66 | 2 * 3 | 5e-5 |
| RMT | 32 | 0 | 512 | 300 | 49.44 | 19.16 | 48.14 | 2 * 3 | 5e-5 |
| RMT | 64 | 0 | 512 | 300 | 49.14 | 18.79 | 47.85 | 2 * 3 | 5e-5 |  
| RMT | 100 | 0 | 512 | 300 | 49.83 | 19.66 | 48.52 | 2 * 3 | 5e-5 |
| RMT | 128 | 0 | 512 | 300 | 50.04 | 19.75 | 48.72 | 2 * 3 | 5e-5 | 
| RMT | 150 | 0 | 512 | 300 | 50.15 | 19.91 | 48.85 | 2 * 3 | 5e-5 | 
| RMT | 200 | 0 | 512 | 300 | 50.59 | 20.24 | 49.31 | 2 * 3 | 5e-5 |
| RMT | 212 | 0 | 512 | 300 | 50.25 | 20.03 | 48.95 | 2 * 3 | 5e-5 |
| RMT | 256 | 0 | 512 | 300 | 50.25 | 20.03 | 48.92 | 2 * 3 | 5e-5 |  

### PubMed-Incremental-Pair_wise

| Model | pre_seq_len | post_seq_len | max_source_length | max_target_length | rouge1 | rouge2 | rougeL | batch_size | lr |  avg_gen_len_seg_1 | avg_gen_len_seg_2 | avg_gen_len_seg_3 | avg_gen_len_seg_4 | seg1_rouge1/2/L | seg2_rouge1/2/L | seg3_rouge1/2/L | seg4_rouge1/2/L | 
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |:-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Bart-base | 0 | 0 | 512 | 300 | 42.06 | 17.42 | 38.80 | 2 * 3 | 5e-5 | 61.41 | 139.81 | 197.47 | 223.08 | 43.06/22.68/38.39 | 41.72/17.00/38.01  | 41.66/15.26/39.33 | 41.80/14.73/39.37 |
| RMT | 20 | 0 | 512 | 300 | 46.00 | 20.22 | 42.45 | 2 * 3 | 5e-5 | 64.93 | 142.16 | 215.83 | 248.98 | 45.90/25.39/41.017 | 45.4236/20.19/41.41 | 46.47/18.46/43.82 | 46.21/16.87/43.54 |
| RMT | 32 | 0 | 512 | 300 |  |  |  | 2 * 3 | 5e-5 |:--: |:--: |:--: |:--: |:--: |:--: |:--: |:--: |
| RMT | 64 | 0 | 512 | 300 |  |  |  |   2 * 3   | 5e-5 |:--: |:--: |:--: |:--: |:--: |:--: |:--: |:--: |
| RMT | 100 | 0 |  512 | 300 |  |  |  | 2 * 3 | 5e-5 |:--: |:--: |:--: |:--: |:--: |:--: |:--: |:--: |
| RMT | 128 | 0 |  512 | 300 |  |  |  | 2 * 3 | 5e-5 |:--: |:--: |:--: |:--: |:--: |:--: |:--: |:--: |
| RMT | 150 | 0 |  512 | 300 |  |  |  | 2 * 3 | 5e-5 |:--: |:--: |:--: |:--: |:--: |:--: |:--: |:--: |
| RMT | 200 | 0 |  512 | 300 | 45.61 | 19.91 | 42.08 | 2 * 3 | 5e-5 | 61.41 | 139.81 | 197.47 | 223.08 | 45.62/25.49/40.67 | 45.03/19.80/41.08 | 46.31/18.20/43.68 | 45.50/16.18/42.87 |
| RMT | 212 | 0 |  512 | 300 |  |  |  | 2 * 3 | 5e-5 | :--: |:--: |:--: |:--: |:--: |:--: |:--: |:--: |
| RMT | 256 | 0 |  512 | 300 |  |  |  | 2 * 3 | 5e-5 |:--: |:--: |:--: |:--: |:--: |:--: |:--: |:--: |

### NLP_JP_CORPUS-User_wise

| Model | pre_seq_len | post_seq_len | max_source_length | max_target_length | rouge1 | rouge2 | rougeL | batch_size | lr |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Bart-base | 0 | 0 | 512 | 300 | 51.62 | 19.84 | 50.49 | 1 * 3 | 3e-6 |
| RMT | 20 | 0 | 512 | 300 | 50.75 | 20.18 | 49.02 | 1 * 3 | 3e-6 |
| RMT | 32 | 0 | 512 | 300 | 50.89 | 20.21 | 49.01 | 1 * 3 | 3e-6 |
| RMT | 64 | 0 | 512 | 300 | 50.66 | 20.17 | 48.85 | 1 * 3 | 3e-6 |
| RMT | 100 | 0 | 512 | 300 | 51.44 | 19.51 | 49.58 | 1 * 3 | 3e-6 |
| RMT | 128 | 0 | 512 | 300 | 51.70 | 19.53 | 49.67 | 1 * 3 | 3e-6 |
| RMT | 150 | 0 | 512 | 300 | 52.05 | 19.89 | 49.94 | 1 * 3 | 3e-6 |
| RMT | 200 | 0 | 512 | 300 | 51.16 | 19.87 | 49.31 | 1 * 3 | 3e-6 |
| RMT | 256 | 0 | 512 | 300 | 50.82 | 19.53 | 48.98 | 1 * 3 | 3e-6 |

### NLP_JP_CORPUS-Pair_wise

| Model | pre_seq_len | post_seq_len | max_source_length | max_target_length | rouge1 | rouge2 | rougeL | batch_size | lr | avg_gen_len_seg_1 | avg_gen_len_seg_2 | avg_gen_len_seg_3 | seg1_rouge1/2/L | seg2_rouge1/2/L | seg3_rouge1/2/L |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Bart-base | 0 | 0 | 512 | 300 | 40.05 | 15.30 | 36.58 | 2 * 3 | 3e-6 | 291.13 | 298.95 | 300.0 | 28.51/14.30/25.88 | 44.00/16.39/40.04 | 47.53/15.30/43.78 |
| RMT | 20 | 0 | 512 | 300 |  |  |  | 2 * 3 | 3e-6 | :-: | :-: | :-: | :-: | :-: | :-: |
| RMT | 32 | 0 | 512 | 300 |  |  |  | 2 * 3 | 3e-6 | :-: | :-: | :-: | :-: | :-: | :-: |
| RMT | 64 | 0 | 512 | 300 |  |  |  | 2 * 3 | 3e-6 | :-: | :-: | :-: | :-: | :-: | :-: |
| RMT | 100 | 0 | 512 | 300 |  |  |  | 2 * 3 | 3e-6 | :-: | :-: | :-: | :-: | :-: | :-: |
| RMT | 128 | 0 | 512 | 300 |  |  |  | 2 * 3 | 3e-6 | :-: | :-: | :-: | :-: | :-: | :-: |
| RMT | 150 | 0 | 512 | 300 |  |  |  | 2 * 3 | 3e-6 | :-: | :-: | :-: | :-: | :-: | :-: |
| RMT | 200 | 0 | 512 | 300 |  |  |  | 2 * 3 | 3e-6 | :-: | :-: | :-: | :-: | :-: | :-: |
| RMT | 256 | 0 | 512 | 300 |  |  |  | 2 * 3 | 3e-6 | :-: | :-: | :-: | :-: | :-: | :-: |


### Tobyoki

> All experiments on a single A100 40GB
> About memory usage : https://huggingface.co/docs/transformers/main_classes/trainer
>   we calculate (*_mem_gpu_alloc_delta + *_mem_gpu_peaked_delta)

### Train Memory Usage
| Model | Seq_len | Mem |
| :--: | :--: | :--: |
| Baseline BART | 512 |    |
| Baseline BART | 1024 |    |
| Baseline BART | 2048 |    |
| Baseline BART | 4096 |    |
| RMT-Summ | 512 | 5939 |
| RMT-Summ | 1024 | 10118 |
| RMT-Summ | 2048 | 19180 |
| RMT-Summ | 4096 | 37249 |

### Inference Memory Usage
| Model | Seq_len | Mem |
| :--: | :--: | :--: |
| Baseline BART | 512 | |
| Baseline BART | 1024 | |
| Baseline BART | 2048 | |
| Baseline BART | 4096 | |
| RMT-Summ | 512 | 1345 |
| RMT-Summ | 1024 | 1350 |
| RMT-Summ | 2048 | 2179 |
| RMT-Summ | 4096 | 3996 |

### Train Iteration Time (iter/s)
| Model | Seq_len | Iter Time |
| :--: | :--: | :--: |
| Baseline BART | 512 | |
| Baseline BART | 1024 | |
| Baseline BART | 2048 | |
| Baseline BART | 4096 | |
| RMT-Summ | 512 | 2.481 |
| RMT-Summ | 1024 | 1.029 |
| RMT-Summ | 2048 | 0.464 |
| RMT-Summ | 4096 | 0.243 |

### Inference Iteration Time (iter/s)
| Model | Seq_len | Iter Time |
| :--: | :--: | :--: |
| Baseline BART | 512 | |
| Baseline BART | 1024 | |
| Baseline BART | 2048 | |
| Baseline BART | 4096 | |
| RMT-Summ | 512 | 0.11 |
| RMT-Summ | 1024 | 0.121 |
| RMT-Summ | 2048 | 0.043 |
| RMT-Summ | 4096 | 0.022 |

