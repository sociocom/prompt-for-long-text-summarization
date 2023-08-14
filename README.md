# prompt-for-long-text-summarization
## project structure
```
project/
|-- config/
|   |-- config.py
|-- data/
|-- logs/
|-- models/
|   |-- base.py                   -> RMT base model
|   |-- conditional_generation.py -> RMT cond_gen task
|   |-- token_classification.py   -> RMT token_cls task
|   |-- prefix_encoder            -> for prompt generation
|   |-- summarization             -> summarization task
|-- utils/
|-- baseline.py
|-- main.py
|-- ...
|-- requirements.txt
```

> our code is in model/summarization.py


