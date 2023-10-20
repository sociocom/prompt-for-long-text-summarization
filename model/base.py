import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer

class RMTBaseModel(nn.Module):
    
    def __init__(self, base_model, rmt_config, **kwargs):
        super().__init__()
        self.base_model = base_model
        self.rmt_config = rmt_config
        
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['tokenizer_name_or_path'])
        self.extract_special_tokens(self.tokenizer)
        self.extend_word_embeddings(self.rmt_config.pre_seq_len, self.tokenizer)
        
        if rmt_config.freeze_model:
            for name, param in self.base_model.named_parameters():
                param.requires_grad = False
    
    def extract_special_tokens(self, tokenizer):
        """Extract special tokens from tokenizer.
        """
        self.pad_token_id = tokenizer.pad_token_id
        self.special_token_ids = [tokenizer.pad_token_id]
        for token in ['cls_token', 'sep_token', 'eos_token', 'bos_token']:
            token_id = getattr(tokenizer, f'{token}_id')
            if token_id is not None:
                self.register_buffer(token, torch.tensor([token_id]))
                self.special_token_ids.append(token_id)
            else:
                setattr(self, token, None)
    
    def extend_word_embeddings(self, num_mem_tokens, tokenizer):
        if num_mem_tokens != 0:   
            vocab_size = self.base_model.config.vocab_size
            extended_vocab_size = vocab_size + num_mem_tokens
            self.num_mem_tokens = num_mem_tokens
            self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
            self.base_model.resize_token_embeddings(extended_vocab_size)

            special_tokens = tokenizer.special_tokens_map
            mem_start_ind = int('cls_token' in special_tokens or 'bos_token' in special_tokens)
            self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)
        self.base_model.embeddings = self.base_model.get_input_embeddings()
        
    def set_memory(self, batch_size):
        memory = self.base_model.embeddings(self.mem_token_ids)
        memory = memory.repeat(batch_size, 1, 1)
        return memory 