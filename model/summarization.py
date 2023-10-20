import torch 
import torch.nn as nn
import torch.nn.functional as F

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

from transformers.modeling_outputs import Seq2SeqLMOutput

from model.base import RMTBaseModel

logger = logging.getLogger(__name__)


class BartForPubmed(RMTBaseModel):
    """
    Using BartForConditionalGeneration as base model
    Without any additional modification
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def prepare_kwargs(self, sec_input_ids, kwargs):
        sec_kwargs = dict(**kwargs)
    
    def forward(
        self,
        input_ids: List[torch.LongTensor] = None, # our model input_ids is different from BartForConditionalGeneration torch.LongTensor
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
        
        kwargs = {
            # 'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'head_mask': head_mask,
            'decoder_head_mask': decoder_head_mask,
            'cross_attn_head_mask': cross_attn_head_mask,
            'encoder_outputs': encoder_outputs,
            'past_key_values': past_key_values,
            'input_emebeds': inputs_embeds,
            'decoder_inputs_embeds': decoder_inputs_embeds,
            'labels': labels,
            'use_cache': use_cache,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict
        }
        
        base_model_outputs = []
        for sec_num, sec_input_ids in enumerate(input_ids):
            if self.rmt_config.bptt_depth != -1:
                raise NotImplementedError
            
            sec_kwargs = self.prepare_kwargs(sec_input_ids, kwargs)
            
        for sec_index in range(len(input_ids.shape[1])):
            pass