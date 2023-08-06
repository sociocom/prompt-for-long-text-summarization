import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import logger
import copy
from typing import List, Optional, Tuple, Union

from transformers import (
    BartForConditionalGeneration, 
    T5ForConditionalGeneration,
    GPT2ForConditionalGeneration,
)
from transformers import BartConfig, T5Config, GPT2Config
from transformers.modeling_outputs import Seq2SeqLMOutput

from model.prefix_encoder import PrefixEncoder

# copied from transformers.modeling_bart.py
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

# prefix-tuning/p-tuning v2 version
class BartPrefixForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        # self.model = BartModel(config)
        # self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        
        # MODIFIED
        # Start
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # End
        
        # https://github.com/huggingface/transformers/issues/4701
        # if we use BartPrefixForConditionalGeneration.from_pretrained() to load the model, 
        # it will not overwrite the pretrained weights of the model
        # Initialize weights and apply final processing
        # self.post_init()
        
        # MODIFIED
        # Start
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)
        
        bart_param = 0
        all_param = 0
        
        # count the number of trainable parameters in bart
        for name, param in self.model.named_parameters():
            bart_param += param.numel() # numel() returns the total number of elements in the input tensor
        
        for name, param in self.named_parameters():
            all_param += param.numel()
            
        trainable_param = all_param - bart_param
        
        print("Total parameters: {:,}".format(all_param))
        print("Trainable parameters: {:,} {:,%}".format((trainable_param), trainable_param/all_param))
        # End

    def get_prompt(self):
        raise NotImplementedError
    
    def pad_and_segment(self, input_ids):
        raise NotImplementedError
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """ 
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                
        # MODIFIED: add prefix encoder
       
class BartPrefixPropForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

 
# ============================================
# MODIFIED from transformers.modeling_t5.py
# ============================================
