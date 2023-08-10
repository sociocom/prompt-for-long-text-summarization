import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import logging
import copy
import math
from typing import List, Optional, Tuple, Union

from transformers import (
    BartForConditionalGeneration, 
    T5ForConditionalGeneration,
    # GPT2ForConditionalGeneration,
)
from transformers import BartConfig, T5Config, GPT2Config
from transformers.modeling_outputs import Seq2SeqLMOutput

from model.prefix_encoder import PrefixEncoder

logger = logging.getLogger(__name__)

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
    def __init__(self, config: BartConfig, tokenizer):
        super().__init__(config)
        # self.model = BartModel(config)
        # self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        
        # MODIFIED
        # Start
        self.set_params(
            tokenizer=tokenizer,
            config=config
        )
        
        # TODO: forget some part of long range memory and add new memory
        # 
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
        
        for param in self.lm_head.parameters():
            param.requires_grad = False
            
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
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

    # MODIFIED
    # Start
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.model.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz,
            seqlen,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )        
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4])
        return past_key_values
    # End
    
    # MODIFIED
    # Start
    def pad_and_segment(self, input_ids, attention_mask=None, labels=None):
        """
        segment input_ids into segments
        
        input sample:
        segmented_batch = [
            [sample1_seg1, sample1_seg2, sample1_seg3],
            [sample2_seg1, sample2_seg2],
            [sample3_seg1, sample3_seg2, sample3_seg3, sample3_seg4]
        ]
                   
        output sample:
        segmented_batch = [
            [sample1_seg1, sample2_seg1, sample3_seg1],
            [sample1_seg2, sample2_seg2, sample3_seg2],
            [sample1_seg3, None, sample3_seg3],
            [None, None, sample3_seg4]
        ]
        """
        segmented_batch = []
        segmented_batch_attention_masks = []
        segmented_batch_labels = []
        
        if attention_mask is None:
            attention_mask = [None] * input_ids.shape[0]
        batch_attention_mask = attention_mask
            
        # inference mode
        if labels is None:
            labels = [None] * input_ids.shape[0]
        batch_labels = labels
        
        # input_ids: [batch_size, seq_len]
        for seq, attn_mask, label in zip(input_ids, batch_attention_mask, batch_labels):
            
            # pytorch syntax: element-wise operation
            drop_mask = sum([seq == t for t in self.special_token_ids])
            
            # bool type slice for tensor type
            # remove special tokens
            seq = seq[(1 - drop_mask).bool()]
            
            # truncate the sequence to the maximum length
            # TODO: config is NotImplemented  dict or dataclass?
            seq = seq[:self.config.segment_size * self.config.max_n_segments]
            
            if att_mask is not None:
                att_mask = att_mask[(1-drop_mask).bool()]
                att_mask = att_mask[:self.config.segment_size * self.config.max_n_segments]
            if label is not None:
                label = label[(1-drop_mask).bool()]
                label = label[:self.config.segment_size * self.config.max_n_segments]
            
            
            align = self.config.segment_alignment
            if align in {'right', None}:
                split_inds = (list(range(len(seq), 0, -self.config.segment_size)) + [0])[::-1]
            elif align == 'left':
                split_inds = list(range(0, len(seq), self.config.segment_size)) + [len(seq)]
            elif align == 'center':
                n_seg = math.ceil(len(seq) / self.config.segment_size)
                split_inds = list(range(0, len(seq), math.ceil(len(seq) / n_seg))) + [len(seq)]
            else:
                raise NotImplementedError

            input_segments = [seq[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
            input_segments = [self.pad_add_special_tokens(t, self.config.input_size) for t in input_segments]
            
            # add empty segment markers if needed
            n_empty_segments = self.config.max_n_segments - len(input_segments)
            # input_segments:
            input_segments = [None] * n_empty_segments + input_segments
            
            # segmented_batch: 
            segmented_batch.append(input_segments)
            
            if attn_mask is not None:
                attn_mask_segments = [att_mask[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
                attn_mask_segments = [self.pad_add_special_tokens(t, self.config.input_size, add_to='attention_mask') for t in attn_mask_segments]
                attn_mask_segments = [None] * n_empty_segments + attn_mask_segments
                segmented_batch_attention_masks.append(attn_mask_segments)
            
            if labels is not None:
                labels_segments = [labels[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
                labels_segments = [self.pad_add_special_tokens(t, self.config.input_size, add_to='labels') for t in labels_segments]
                labels_segments = [None] * n_empty_segments + labels_segments
                segmented_batch_labels.append(labels_segments)
                
        segmented_batch = [[sample[seg_num] for sample in segmented_batch] 
                            for seg_num in range(self.config.max_n_segments)]
        segmented_batch_attention_masks = [[sample[seg_num] for sample in segmented_batch_attention_masks]
                                           for seg_num in range(self.config.max_n_segments)]
        segmented_batch_labels = [[sample[seg_num] for sample in segmented_batch_labels]
                                  for seg_num in range(self.config.max_n_segments)]
        return segmented_batch, segmented_batch_attention_masks, segmented_batch_labels
    # End
    
    def set_params(self, tokenizer, config):
        self.config = config 
        self.extract_special_tokens(tokenizer)
        # self.extend_word_embeddings(config['pre_seq_len'], tokenizer)
        
        # tokenizer.num_special_tokens_to_add()cal the number of special tokens needed to add except [SEP]
        self.segment_size = config['input_size'] - self.pre_seq_len - tokenizer.num_special_tokens_to_add()
        if 'sep_token' in tokenizer.special_tokens_map:
            self.segment_size -= 1
        
    def extract_special_tokens(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.special_token_ids = [tokenizer.pad_token_id]
        for token in ['cls_token', 'sep_token', 'eos_token', 'bos_token']:
            token_id = getattr(tokenizer, f'{token}_id')
            if token_id is not None:
                self.register_buffer(token, torch.tensor([token_id]))
                self.special_token_ids.append(token_id)
            else:
                setattr(self, token, None)
                
    # def extend_word_embeddings(self, tokenizer):
    #     vocab_size = self.model.config.vocab_size
    #     # NOTE: Really necessary???
    #     extended_vocab_size = vocab_size + self.config.pre_seq_len
    #     self.pre_seq_len = self.config.pre_seq_len

    # Memory mechanism like RNN
    def forget_and_memory(self,):
        raise NotImplementedError
    
    #  prefix-tuning don't need to concat prefix and input sequence
    def pad_add_special_tokens(self, tensor, segment_size, 
                               prompts=None, prompt_attention_mask=None, # maybe better to use pre_seq_len and generate prompts attention mask?
                               add_to='input_ids'):
        """
        bart tokenizer:
        {'bos_token': '<s>', 0
         'eos_token': '</s>', 2
         'unk_token': '<unk>', 3
         'sep_token': '</s>', 0
         'pad_token': '<pad>', 1
         'cls_token': '<s>', 0
         'mask_token': '<mask>' 50264
        }
        """
        input_elements = []
        # Add special tokens: <s> and </s> to the input sequence
        # For prefix-prop
        if prompts is not None:
            if add_to == 'inputs':
                input_elements += [self.cls_token, prompts, self.sep_token, tensor, self.sep_token]
            # For Bart, only the pad token is 0 in attention_mask
            elif add_to == 'attention_mask':
                mask_value = torch.ones((1), device=tensor.device)
                input_elements += [mask_value, prompt_attention_mask, mask_value, tensor, mask_value]
            # As a encoder-decoder model：is not needed to add prompt to labels
            elif add_to == 'labels':
                input_elements += [self.cls_token, tensor, self.sep_token]
        # For prefix-tuning/p-tuning v2
        else:
            if add_to == 'input_ids':
                input_elements += [self.cls_token, tensor, self.sep_token]
            elif add_to == 'attention_mask':
                mask_value = torch.ones((1), device=tensor.device)
                input_elements += [mask_value, tensor, mask_value]
            elif add_to == 'labels':
                input_elements += [self.cls_token, tensor, self.sep_token]
        tensor = torch.cat(input_elements)
        
        # Add padding tokens
        # TODO: implement summary module
        #       now self.config.sum_size default = 0
        pad_size = segment_size - tensor.shape[0] - self.config.sum_size
        if pad_size > 0:
            if add_to == 'input_ids':
                tensor = F.pad(tensor, (0, pad_size), value=self.pad_token_id)
            elif add_to == 'attention_mask':
                tensor = F.pad(tensor, (0, pad_size), value=0)
            elif add_to == 'labels':
                # for Seq2Seq labels need to be pad by -100
                tensor = F.pad(tensor, (0, pad_size), value=-100)
                pass
        return tensor

        # TODO: this implementation just add <s> and </s> to the input sequence
        #       maybe need to add other special tokens
    
    def prepare_kwargs(self, segment_input_ids, kwargs):
        seg_kwargs = dict(**kwargs)
        # [sample1_seg1, sample2_seg1, sample3_seg1,....] up to batch_size
        # Some of the segments are None like: [sample1_seg3, None, sample3_seg3]
        non_empty_mask = [s is not None for s in segment_input_ids]
        
        # all the segments are None, due to the max_n_segments >> the number of segments
        if sum(non_empty_mask) == 0:
            return None, non_empty_mask
        
        input_ids = torch.stack([s for s in segment_input_ids if s is not None])
        # embedding layer
        input_embeds = self.model.shared(input_ids)
        
        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = input_embeds
        if seg_kwargs.get('labels') is not None:
            seg_kwargs['labels'] = seg_kwargs['labels'][non_empty_mask]
        seg_kwargs['attention_mask'] = self.get_attention_mask(input_ids)
    
    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask
    
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
        
        kwargs = {
            'attention_mask': attention_mask, 
            # 'token_type_ids': token_type_ids,
            # 'position_ids': position_ids, 
            'inputs_embeds': inputs_embeds,
            'labels': labels, 'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
        }
        
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
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len)
        attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        
        # segmented: [max_n_segments, batch_size, segment_size]
        segmented = self.pad_and_segment(input_ids)
        
        # NOTE: why???
        # if self.pre_seq_len == 0:
        #     segmented = segmented[-1:]
        
        model_outputs = []
        for seg_num, segment_input_ids in enumerate(segmented):
            if self.config.bptt_depth != -1:
                raise NotImplementedError

    def generate(self):
        raise NotImplementedError
    
    
class BartPrefixPropForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)


# ============================================
# MODIFIED from transformers.modeling_t5.py
# ============================================
