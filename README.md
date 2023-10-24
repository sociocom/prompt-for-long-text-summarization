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

## Dataset Statistic
* **Pubmed statistic:**
  | Split | samples | 
  | :-: | :-: |
  | Train | 119924 | 
  | Val | 6633 |
  | Test | 6658 |

* **Final statistic after preprocessing**
  | Split | samples | section_avg_token_size | abstract_avg_token_size |
  | :-: | :-: | :-: | :-: |
  | Train | 24843 | 2740 | 299 |
  | Val | 1399 | 2752 | 300 |
  | Test | 1431 | 2732 | 303 |

* **a sample of vanilla dataset**
  * labels seems to used for classification task, the labels of vanilla datasets are all None, i didn't do any processing.
  
  | article_id | article_text | abstract_text | labels | section_names | sections |
  | :-: | :-: | :-: | :-: | :-: | :-: |
  | str | List[str] | List[str] | None | List[str] | List[List[str]] |
  | 'PMC3872579' | split by sentence<br>(似乎按照句子进行分割了, 如果要使用需要进行join) | ['<BOS> background : the present study was carried out to assess the effects of community nutrition intervention based on advocacy approach on malnutrition status among school - aged children in shiraz , iran.materials and methods : this case - control nutritional intervention has been done between 2008 and 2009 on 2897 primary and secondary school boys and girls ( 7 - 13 years old ) based on advocacy approach in shiraz , iran . </EOS>', <br> '<BOS> the project provided nutritious snacks in public schools over a 2-year period along with advocacy oriented actions in order to implement and promote nutritional intervention . for evaluation of effectiveness of the intervention growth monitoring indices of pre- and post - intervention were statistically compared.results:the frequency of subjects with body mass index lower than 5% decreased significantly after intervention among girls ( p = 0.02 ) . </EOS>', <br> '<BOS> however , there were no significant changes among boys or total population . </EOS>', <br> '<BOS> the mean of all anthropometric indices changed significantly after intervention both among girls and boys as well as in total population . </EOS>', <br> "<BOS> the pre- and post - test education assessment in both groups showed that the student 's average knowledge score has been significantly increased from 12.5  3.2 to 16.8  4.3 ( p < 0.0001).conclusion : this study demonstrates the potential success and scalability of school feeding programs in iran . </EOS>", <br> '<BOS> community nutrition intervention based on the advocacy process model is effective on reducing the prevalence of underweight specifically among female school aged children . </EOS>'] <br> 分割方式不明, 需要进行join并重新分割 | None | ['INTRODUCTION', <br>'MATERIALS AND METHODS', <br>'Participants'Instruments', <br>'Procedure', <br>'First step', <br>'Second step', <br>'Third step', <br>'Forth step', <br>'Interventions', <br>'Fifth step (assessment)', <br>'Data analysis', <br>'RESULTS', <br>'DISCUSSION', <br>'CONCLUSION'] <br><br>请注意METHODS包含了从Participants'Instruments到Data analysis的部分 | [<br>section[seq1, seq2, ...], <br>section[seq1, seq2, ...], <br>...] |
  | reformat之后的格式 | :-: | :-: | :-: | :-: | :-: |
  | str | List[str] | list[str] | None | List[str] | List[str]|

## Result Table 
* CNN-DailyMail
    | Model | pre_seq_len| post_seq_len| Model fixed | train sample | eval/pred sample | rouge1 | rouge2 | rougeL | batch_size | 
    | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
    | Bart-base | 0 | 0 | True | 100 | 50 | 31.97 | 12.91 | 21.51 | 1 * 4 |
    | Bart-base Prefix-tuning | 20 | 0 | True | 100 | 50 | 19.48 | 5.68 | 18.55 | 1 * 4 | 
    | Bart-base Prefix-Prop | 20 | 0 | False | 100 | 50 | 32.17 | 13.50 | 29.24 | 1 * 4 | 
    | Bart-base Prefix-Prop | 20 | 0 | False | 287113 | 13368/11490 | 42.86 | 20.09 | 36.71 | 8 * 4 | 
    | Bart-base Prefix-Prop | 20 | 0 | True | 100 | 50 | 30.47 | 13.09 | 27.73 | 1 * 4 |
    | Bart-large Prefix-Prop | 20 | 0 | False | 287113 | 13368/11490 | 44.01 | 20.94 | 41.06 | 8 * 4 | 

* PubMed
    | Model | pre_seq_len| post_seq_len| max_source_length | Model fixed | train sample | eval/pred sample | rouge1 | rouge2 | rougeL | batch_size | 
    | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |  
    | Bart-base | 0 | 0 | 861 | False | 24843 | 1399/1431 | 53.29 | 25.78 | 49.06 | 4 * 3 |     
    | Bart-base | 0 | 0 | 1003 | False | 24843 | 1399/1431 | :-: | :-: | :-: | 4 * 3 |  
    | Bart-base | 0 | 0 | 1024 | False | 24843 | 1399/1431 | 53.80 | 26.55 | 49.70 | 4 * 3 |   
    | Bart-base-RMT | 20 | 0 | 1003 <br>(1024 - 20 - 0) <br> -1 (bos)| False | 24843 | 1399/1431 | 46.03 | 20.49 | 42.02 | 4 * 3 |
    | Bart-base-RMT | 20 | 142 | 861 <br>(1024 - 20 - 142) <br> -1 (bos)| False | 24843 | 1399/1431 | 46.43 | 20.82 | 42.52 | 4 * 3 |


## Model Architecture
![Alt text](image.png){:height="1000" width="1000"}