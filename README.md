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
| Model | pre_seq_len| post_seq_len| Model fixed | train sample | eval/pred sample | rouge1 | rouge2 | rougeL | batch_size | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| Bart-base | 0 | 0 | None | 100 | 50 | 31.97 | 12.91 | 21.51 | 1 |
| Bart-base Prefix-tuning | 20 | 0 | Fixed | 100 | 50 | 19.48 | 5.68 | 18.55 | 1 | 
| Bart-base Prefix-Prop | 20 | 0 | None | 100 | 50 | 32.17 | 13.50 | 29.24 | 1 | 
| Bart-base Prefix-Prop | 20 | 0 | Fixed | 100 | 50 | 30.47 | 13.09 | 27.73 | 1 |

## BUG List
1. Due to unknown reason, the Prefix-tuning from peft library can't be trained by trainer, pls try to use accelerator.
