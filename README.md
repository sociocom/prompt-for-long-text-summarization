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
    | ----- | ------- |
    | train | 119924  |
    | eval  | 6633    |
    | test  | 6658    |
* after processing
  * | split | samples | avg_token_size_section | avg_token_size_abstract |
    | :---: | :-----: | :--------------------: | :---------------------: |
    | train |  24843  |          2740          |           299           |
    | eval |  1399  |          2752          |           300           |
    | test |  1431  |          2732          |           303           |

## Result:
### PubMed-Incremental

|   Model   | pre_seq_len | post_seq_len | max_source_length | max_target_length | rouge1 | rouge2 | rougeL | batch_size |  lr  |  |
| :-------: | :---------: | :----------: | :---------------: | :---------------: | :----: | :----: | :----: | :--------: | :--: | :-: |
| Bart-base |      0      |      0      |        512        |        300        | 49.35 | 19.38 | 48.05 |   2 * 3   | 5e-5 |  |
|    RMT    |     20     |      0      |        512        |        300        | 48.94 | 18.76 | 47.66 |   2 * 3   | 5e-5 |  |
|    RMT    |     32     |      0      |        512        |        300        | 49.44 | 19.16 | 48.14 |   2 * 3   | 5e-5 |  |
|    RMT    |     64     |      0      |        512        |        300        | 49.14 | 18.79 | 47.85 |   2 * 3   | 5e-5 |  |
|    RMT    |     100     |      0      |        512        |        300        | 49.83 | 19.66 | 48.52 |   2 * 3   | 5e-5 |  |
|    RMT    |     128     |      0      |        512        |        300        | 50.04 | 19.75 | 48.72 |   2 * 3   | 5e-5 |  |
|    RMT    |     150     |      0      |        512        |        300        | 50.15 | 19.91 | 48.85 |   2 * 3   | 5e-5 |  |
|    RMT    |     200     |      0      |        512        |        300        | 50.59 | 20.24 | 49.31 |   2 * 3   | 5e-5 |  |
|    RMT    |     212     |      0      |        512        |        300        | 50.25 | 20.03 | 48.95 |   2 * 3   | 5e-5 |  |
|    RMT    |     256     |      0      |        512        |        300        | 50.25 | 20.03 | 48.92 |   2 * 3   | 5e-5 |  |


### NLP_JP_CORPUS

|   Model   | pre_seq_len | post_seq_len | max_source_length | max_target_length | rouge1 | rouge2 | rougeL | batch_size | lr   |
| :-------: | :---------: | :----------: | :---------------: | :---------------: | :----: | :----: | :----: | :--------: | ---- |
| Bart-base |      0      |      0      |        512        |        300        | 51.62 | 19.84 | 50.49 |   1 * 3   | 3e-6 |
|    RMT    |     20     |      0      |        512        |        300        | 50.75 | 20.18 | 49.02 |   1 * 3   | 3e-6 |
|    RMT    |     32     |      0      |        512        |        300        | 50.89 | 20.21 | 49.01 |   1 * 3   | 3e-6 |
|    RMT    |     64     |      0      |        512        |        300        | 50.66 | 20.17 | 48.85 |   1 * 3   | 3e-6 |
|    RMT    |     100     |      0      |        512        |        300        | 51.44 | 19.51 | 49.58 |   1 * 3   | 3e-6 |
|    RMT    |     128     |      0      |        512        |        300        | 51.70 | 19.53 | 49.67 |   1 * 3   | 3e-6 |
|    RMT    |     150     |      0      |        512        |        300        | 52.05 | 19.89 | 49.94 |   1 * 3   | 3e-6 |
|    RMT    |     200     |      0      |        512        |        300        | 51.16 | 19.87 | 49.31 |   1 * 3   | 3e-6 |
|    RMT    |     256     |      0      |        512        |        300        | 50.82 | 19.53 | 48.98 |   1 * 3   | 3e-6 |


### Tobyoki

### Train Memory Usage
| Model | Seq_len | Mem |
| :--: | :--: | :--: |
| Baseline BART | 512 |    |
| Baseline BART | 1024 |    |
| Baseline BART | 2048 |    |
| Baseline BART | 4096 |    |
| RMT-Summ | 512 | 5939 |
| RMT-Summ | 1024 |    |
| RMT-Summ | 2048 | 19180 |
| RMT-Summ | 4096 |    |

### Eval Memory Usage
| Model | Seq_len | Mem |
| :--: | :--: | :--: |
| Baseline BART | 512 |    |
| Baseline BART | 1024 |    |
| Baseline BART | 2048 |    |
| Baseline BART | 4096 |    |
| RMT-Summ | 512 | 805 |
| RMT-Summ | 1024 |    |
| RMT-Summ | 2048 | 2151 |
| RMT-Summ | 4096 |    |

### Train Iteration Time
| Model | Seq_len | Iter Time |
| :--: | :--: | :--: |
| Baseline BART | 512 |    |
| Baseline BART | 1024 |    |
| Baseline BART | 2048 |    |
| Baseline BART | 4096 |    |
| RMT-Summ | 512 |    |
| RMT-Summ | 1024 |    |
| RMT-Summ | 2048 | 0.147 |
| RMT-Summ | 4096 |    |

### Eval Iteration Time 
| Model | Seq_len | Iter Time |
| :--: | :--: | :--: |
| Baseline BART | 512 |    |
| Baseline BART | 1024 |    |
| Baseline BART | 2048 |    |
| Baseline BART | 4096 |    |
| RMT-Summ | 512 |    |
| RMT-Summ | 1024 |    |
| RMT-Summ | 2048 | 0.041 |
| RMT-Summ | 4096 |    |

