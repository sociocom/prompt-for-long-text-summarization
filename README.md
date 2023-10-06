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
|   |-- prefix_encoder            -> for prompt generation
|-- utils/
|-- script/                       -> task specific run script
|-- run_summarization.py          -> training
|-- ...
|-- requirements.txt              -> training_args
```


## Result Table 
| Model | pre_seq_len| post_seq_len| Model fixed | rouge1 | rouge2 | rougeL | batch_size | train_sample | eval_sample | pred_sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bart-base | 0 | 0 | None | 31.97 | 12.91 | 21.51 |
| Bart-base Prefix-tuning | 20 | 0 | Fixed | 
| Bart-base Prefix-Prop | 20 | 0 | Fixed | 30.47 | 13.09 | 27.73 |
| Bart-base Prefix-Prop | 20 | 0 | None | 32.17 | 13.50 | 29.24 |


