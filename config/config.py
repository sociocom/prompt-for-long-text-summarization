from dataclasses import dataclass
from transformers import BartConfig, T5Config

class PromptBartConfig(BartConfig):
    def __init__(self, config):
        super().__init__()
        self.pre_seq_len = config.pre_seq_len

class PromptT5Config():
    raise NotImplementedError