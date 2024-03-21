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
| Bart-base | 0 | 0 | 512 | 300 | 42.06 | 17.42 | 38.80 | 6 | 5e-5 | 61.41 | 139.81 | 197.47 | 223.08 | 43.06/22.68/38.39 | 41.72/17.00/38.01  | 41.66/15.26/39.33 | 41.80/14.73/39.37 |
| RMT | 20 | 0 | 512 | 300 | 44.79 | 19.17 | 41.35 | 6 | 5e-5 | 63.77 | 141.50 | 213.74 | 246.10 | 45.33/24.82/40.48 | 44.28/19.07/40.36 | 44.76/17.02/42.24 | 44.82/15.77/42.32 |
| RMT | 32 | 0 | 512 | 300 | 44.03 | 18.54 | 40.66 | 6 | 5e-5 | 63.02 | 144.92 | 211.32 | 247.60 | 43.87/23.43/39.16 | 43.57/18.50/39.75 | 43.90/16.35/41.40 | 44.85/15.87/42.27 |
| RMT | 64 | 0 | 512 | 300 | 44.57 | 19.01 | 41.15 | 6 | 5e-5 | 66.05 | 147.11 | 214.30 | 249.73 | 44.54/24.15/39.70 | 44.24/19.14/40.42 | 44.59/16.91/42.10 | 44.92/15.83/42.35 |
| RMT | 100 | 0 |  512 | 300 | 45.71 | 20.03 | 42.19 | 6 | 5e-5 | 67.96 | 149.57 | 220.55 | 249.69 | 45.54/25.51/40.65 | 45.07/19.82/41.09 | 46.65/18.63/44.93 | 45.52/16.20/42.93 |
| RMT | 128 | 0 |  512 | 300 | 45.55 | 19.96 | 42.06 | 6 | 5e-5 | 67.68 | 146.62 | 217.09 | 250.61 | 45.47/25.51/40.60 | 44.92/19.85/41.00 | 46.33/18.23/43.67 | 45.55/16.28/42.95 |
| RMT | 150 | 0 |  512 | 300 | 45.81 | 20.09 | 42.24 | 6 | 5e-5 | 64.74 | 149.35 | 224.53 | 253.12 | 45.77/25.75/40.78 | 45.06/19.87/41.10 | 46.78/18.47/44.07 | 45.63/16.27/43.01 |
| RMT | 200 | 0 |  512 | 300 | 45.61 | 19.91 | 42.08 | 6 | 5e-5 | 61.41 | 139.81 | 197.47 | 223.08 | 45.62/25.49/40.67 | 45.03/19.80/41.08 | 46.31/18.20/43.68 | 45.50/16.18/42.87 |
| RMT | 256 | 0 |  512 | 300 | 45.81 | 20.15 | 42.25 | 6 | 5e-5 | 64.63 | 148.71 | 233.86 | 254.25 | 45.86/25.86/40.88 | 45.00/19.82/41.00 | 46.59/18.43/43.9 | 45.82/16.48/43.19 |

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
| RMT | 20 | 0 | 512 | 300 | 40.00 | 14.94 | 36.52 | 2 * 3 | 3e-6 | 264.33 | 268.62 | 277.99 | 29.50/14.12/26.56 | 43.85/15.95/39.99 | 46.56/14.75/43.07 |
| RMT | 32 | 0 | 512 | 300 | 39.61 | 15.04 | 36.26 | 2 * 3 | 3e-6 | 275.67 | 273.42 | 273.20 | 29.41/14.41/26.53 | 43.30/15.83/39.44 | 46.08/14.86/42.78 |
| RMT | 64 | 0 | 512 | 300 | 40.06 | 15.13 | 36.58 | 2 * 3 | 3e-6 | 263.64 | 267.14 | 274.98 | 29.72/14.79/26.89 | 43.61/15.84/39.59 | 46.82/14.81/43.22 |
| RMT | 100 | 0 | 512 | 300 | 39.88 | 15.20 | 36.48 | 2 * 3 | 3e-6 | 268.57 | 268.95 | 277.15 | 29.32/14.50/26.72 | 43.57/16.31/39.59 | 46.56/14.82/43.17 |
| RMT | 128 | 0 | 512 | 300 | 40.05 | 15.15 | 36.49 | 2 * 3 | 3e-6 | 268.37 | 261.89 | 265.98 | 29.98/14.43/27.11 | 43.55/16.15/39.54 | 46.37/14.97/42.98 |
| RMT | 150 | 0 | 512 | 300 | 40.22 | 15.25  | 36.61 | 2 * 3 | 3e-6 | 246.44 | 253.39 | 245.14 | 31.10/14.81/28.08 | 44.09/16.33/39.92 | 45.32/14.64/41.94 |
| RMT | 200 | 0 | 512 | 300 | 40.08 | 14.85 | 36.51 | 2 * 3 | 3e-6 | 210.81 | 240.44 | 238.31 | 31.98/13.97/28.54 | 43.11/16.17/39.34 | 45.07/14.43/41.57 |
| RMT | 256 | 0 | 512 | 300 | 39.74 | 15.04 | 36.18 | 2 * 3 | 3e-6 | 191.36 | 228.13 | 222.49 | 32.61/14.60/20.07 | 42.66/16.38/39.32 | 43.75/14.20/40.32 |


### Tobyoki (with space)
> training on first 10 segment

| Model | pre_seq_len | post_seq_len | max_source_length | max_target_length | rouge1 | rouge2 | rougeL | batch_size | lr | 
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| Bart-base-full-segment | 0 | 0 | 512 | 300 | 15.96 | 2.63 | 13.49 | 1 | 5e-5 | 
| Bart-base-first10 | 0 | 0 | 512 | 300 |  |  |  | 1 | 3e-6 | 
| RMT-first10 | 20 | 0 | 512 | 300 | 4.56 | 0.78 | 4.39 | 1 | 5e-5 | 
| RMT-first10 | 20 | 0 | 512 | 300 | 7.98 | 0.45 | 7.67 | 1 | 3e-6 | 
| RMT-first10 | 32 | 0 | 512 | 300 | 5.04 | 0.91 | 4.81 | 1 | 5e-5 | 
| RMT-first10 | 32 | 0 | 512 | 300 | 8.11 | 0.43 | 7.84 | 1 | 3e-6 | 
| RMT-first10 | 64 | 0 | 512 | 300 | 7.55 | 0.42 | 7.32 | 1 | 3e-6 | 
| RMT-first10 | 100 | 0 | 512 | 300 | 39.88 | 15.20 | 36.48 | 1 | 3e-6 | 
| RMT-first10 | 128 | 0 | 512 | 300 | 40.05 | 15.15 | 36.49 | 1 | 3e-6 | 
| RMT-first10 | 150 | 0 | 512 | 300 | 40.22 | 15.25  | 36.61 | 1 | 3e-6 | 
| RMT-first10 | 200 | 0 | 512 | 300 | 40.08 | 14.85 | 36.51 | 1 | 3e-6 | 
| RMT-first10 | 256 | 0 | 512 | 300 | 39.74 | 15.04 | 36.18 | 1 | 3e-6 | 

### Tobyoki (w/o space | first)

| Model | pre_seq_len | post_seq_len | max_source_length | max_target_length | rouge1 | rouge2 | rougeL | batch_size | lr |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| Bart-base | 0 | 0 | 512 | 300 | 40.05 | 15.30 | 36.58 | 2 * 3 | 3e-6 | 
| RMT | 20 | 0 | 512 | 300 | 40.00 | 14.94 | 36.52 | 2 * 3 | 3e-6 | 
| RMT | 32 | 0 | 512 | 300 | 39.61 | 15.04 | 36.26 | 2 * 3 | 3e-6 | 
| RMT | 64 | 0 | 512 | 300 | 40.06 | 15.13 | 36.58 | 2 * 3 | 3e-6 | 
| RMT | 100 | 0 | 512 | 300 | 39.88 | 15.20 | 36.48 | 2 * 3 | 3e-6 | 
| RMT | 128 | 0 | 512 | 300 | 40.05 | 15.15 | 36.49 | 2 * 3 | 3e-6 | 
| RMT | 150 | 0 | 512 | 300 | 40.22 | 15.25  | 36.61 | 2 * 3 | 3e-6 |
| RMT | 200 | 0 | 512 | 300 | 40.08 | 14.85 | 36.51 | 2 * 3 | 3e-6 |
| RMT | 256 | 0 | 512 | 300 | 39.74 | 15.04 | 36.18 | 2 * 3 | 3e-6 |

> All experiments on a single A100 40GB
> About memory usage : https://huggingface.co/docs/transformers/main_classes/trainer
>   we calculate (*_mem_gpu_alloc_delta + *_mem_gpu_peaked_delta)
> batch_size = 2

### Train Memory Usage
| Model | Seq_len | Mem |
| :--: | :--: | :--: |
| Baseline BART | 512 | 2679 |
| Baseline BART | 1024 | 4195 |
| Baseline BART | 2048 | 9182 |
| Baseline BART | 4096 | 27200 |
| Baseline BART | 8192 | OOM |
| RMT-Summ | 512 | 2685 |
| RMT-Summ | 1024 | 4072 |
| RMT-Summ | 2048 | 7051 |
| RMT-Summ | 4096 | 12997 |
| RMT-Summ | 8192 | 24881 |

### Inference Memory Usage
| Model | Seq_len | Mem |
| :--: | :--: | :--: |
| Baseline BART | 512 | 425 |
| Baseline BART | 1024 | 579 |
| Baseline BART | 2048 | 890 |
| Baseline BART | 4096 | 3350 |
| Baseline BART | 8192 | OOM |
| RMT-Summ | 512 | 447 |
| RMT-Summ | 1024 | 455 |
| RMT-Summ | 2048 | 755 |
| RMT-Summ | 4096 | 1382 |
| RMT-Summ | 8192 | 2640 |

### Train Iteration Time (iter/s)
| Model | Seq_len | Iter Time | s/iter |
| :--: | :--: | :--: | :--: |
| Baseline BART | 512 | 2.407 | 0.415 |
| Baseline BART | 1024 | 2.242 | 0.446 |
| Baseline BART | 2048 | 1.796 | 0.557 |
| Baseline BART | 4096 | 1.105 | 0.905 |
| Baseline BART | 8192 | OOM | OOM |
| RMT-Summ | 512 | 4.699 | 0.213 |
| RMT-Summ | 1024 | 2.228 | 0.449 |
| RMT-Summ | 2048 | 0.944 | 1.059 |
| RMT-Summ | 4096 | 0.479 | 2.088 |
| RMT-Summ | 8192 | 0.242 | 4.132 |

### Inference Iteration Time (iter/s)
| Model | Seq_len | Iter Time |
| :--: | :--: | :--: |
| Baseline BART | 512 | 0.199 |
| Baseline BART | 1024 | 0.201 | 
| Baseline BART | 2048 | 0.198 |
| Baseline BART | 4096 | 0.185 |
| Baseline BART | 8192 | OOM |
| RMT-Summ | 512 | 0.717 |
| RMT-Summ | 1024 | 0.24 |
| RMT-Summ | 2048 | 0.081 |
| RMT-Summ | 4096 | 0.04 |
| RMT-Summ | 8192 | 0.02 |
