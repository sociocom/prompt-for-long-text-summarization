import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import logging
import copy
import math
from typing import List, Optional, Tuple, Union, Iterable

from transformers import (
    BartModel,
    BartPreTrainedModel,
    BartForConditionalGeneration,
    BartTokenizer,
)
from transformers import BartConfig
from peft import PrefixTuningConfig, TaskType, get_peft_model
from transformers.modeling_outputs import Seq2SeqLMOutput

from model.prefix_encoder import PrefixEncoder
from model.base import RMTBaseModel
from config import PromptBartConfig

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

# ============================================
# =============== BART model =================
# ============================================
class BartRMTForConditionalGeneration(BartPreTrainedModel):
    # base_model_prefix = "model"
    # _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    # _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    config_class = PromptBartConfig
    
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = BartForConditionalGeneration(config)
        # self.model = BartModel(config)
        # self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.tokenizer = BartTokenizer.from_pretrained(kwargs['tokenizer_name_or_path'])
        
        self.config = config
        self.segment_alignment = config.segment_alignment
        self.extract_special_tokens(self.tokenizer)
        # self.extend_word_embeddings(config.pre_seq_len, self.tokenizer)
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads   
        self.segment_size = config.input_size - self.pre_seq_len - self.tokenizer.num_special_tokens_to_add()
        
        if 'sep_token' in self.tokenizer.special_tokens_map:
            self.segment_size -= 1      
        
        for name, param in self.model.named_parameters():
                param.requires_grad = False
        
        # Initialize weights and apply final processing
        # 这里不会重写from_pretrained的参数, 这里保证了有些非pretrained的参数被随机初始化
        self.post_init()
        
    def pad_and_segment(self, input_ids, labels=None):

        segmented_batch = []
        segmented_batch_labels = []
            
        # inference mode
        if labels is None:
            labels = [None] * input_ids.shape[0]
        batch_labels = labels
        
        # input_ids: [batch_size, seq_len]
        for seq, label in zip(input_ids, batch_labels):
            
            # pytorch syntax: element-wise operation
            drop_mask = sum([seq == t for t in self.special_token_ids])
            # Convert non-zero elements to 1
            drop_mask = torch.tensor([1 if t != 0 else 0 for t in drop_mask])
            
            # bool type slice for tensor type
            # remove special tokens
            seq = seq[(1 - drop_mask).bool()]
            
            # truncate the sequence to the maximum length
            seq = seq[:self.segment_size * self.config.max_n_segments]
            
            if label is not None:
                label_drop_mask = sum([label == t for t in self.special_token_ids + [-100]])
                label_drop_mask = torch.tensor([1 if t != 0 else 0 for t in label_drop_mask])
                label = label[(1-label_drop_mask).bool()]
                # TODO：label = label[:self.config.sum_max_size * self.config.max_n_segments]
                label = label[:self.segment_size * self.config.max_n_segments]
            
            align = self.segment_alignment
            if align in {'right', None}:
                split_inds = (list(range(len(seq), 0, -self.segment_size)) + [0])[::-1]
            elif align == 'left':
                split_inds = list(range(0, len(seq), self.segment_size)) + [len(seq)]
            elif align == 'center':
                n_seg = math.ceil(len(seq) / self.segment_size)
                split_inds = list(range(0, len(seq), math.ceil(len(seq) / n_seg))) + [len(seq)]
            else:
                raise NotImplementedError

            input_segments = [seq[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
            
            # TODO: just a test
            input_segments[-1] = seq[-self.segment_size:]
            input_segments = [self.pad_add_special_tokens(t, self.config.input_size) for t in input_segments]
            
            # add empty segment markers if needed
            n_empty_segments = self.config.max_n_segments - len(input_segments)
            # input_segments:
            input_segments = input_segments + [self.get_full_padding_segment(add_to='input_ids')] * n_empty_segments
            
            # segmented_batch: 
            segmented_batch.append(input_segments)

            if label is not None:
                # TODO : do test
                full_segment_size = len(input_segments) - n_empty_segments
                end_index = math.ceil(len(label) // full_segment_size)
                labels_segments = [label[:(end_index*(i+1))] for i in range(full_segment_size)]
                labels_segments = [self.pad_add_special_tokens(t, self.config.input_size, add_to='labels') for t in labels_segments]
                labels_segments = labels_segments + [self.get_full_padding_segment(add_to='label')] * n_empty_segments
                segmented_batch_labels.append(labels_segments)
                
                # full_segment_size = len(input_segments) - n_empty_segments
                # labels_segments = [label[:] for i in range(full_segment_size)]
                # labels_segments = [self.pad_add_special_tokens(t, self.config.input_size, add_to='labels') for t in labels_segments]
                # labels_segments = labels_segments + [self.get_full_padding_segment()] * n_empty_segments
                # segmented_batch_labels.append(labels_segments)
                
        segmented_batch = [[sample[seg_num] for sample in segmented_batch] 
                            for seg_num in range(self.config.max_n_segments)]

        segmented_batch_labels = [[sample[seg_num] for sample in segmented_batch_labels]
                                  for seg_num in range(self.config.max_n_segments)]
        return segmented_batch, segmented_batch_labels
        
    def get_full_padding_segment(self, add_to):
        if add_to == 'input_ids':
            padding_segment = torch.tensor([self.pad_token_id for _ in range(self.config.input_size)])
        elif add_to == 'label':
            padding_segment = torch.tensor([-100 for _ in range(self.config.input_size)])
        return padding_segment
    
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
                
    def extend_word_embeddings(self, num_mem_tokens, tokenizer):  
        if num_mem_tokens != 0:   
            vocab_size = self.model.config.vocab_size
            extended_vocab_size = vocab_size + num_mem_tokens
            # self.num_mem_tokens = num_mem_tokens
            self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
            self.model.resize_token_embeddings(extended_vocab_size)

            special_tokens = tokenizer.special_tokens_map
            mem_start_ind = int('cls_token' in special_tokens or 'bos_token' in special_tokens)
            self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)
            
            # TODO : just test
            # for param in self.model.lm_head.parameters():
            #     param.requires_grad = False
        self.model.embeddings = self.model.get_input_embeddings()
        self.model.embeddings.weight.requires_grad = True
        
    def set_memory(self, input_shape):
        memory = self.model.embeddings(self.mem_token_ids)
        memory = memory.repeat(input_shape[0], 1, 1)
        return memory        
    
    # Memory mechanism like RNN
    def forget_and_memory(self,):
        raise NotImplementedError    
           
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
            # As a encoder-decoder model：is not needed to add prompt to labels
            elif add_to == 'labels':
                # NOTE: for Seq2Seq Models labels are used for decoder_input_ids in training
                # and decoder_input_ids must start with the eos_token
                input_elements += [self.eos_token, tensor, self.sep_token]
        # For prefix-tuning/p-tuning v2
        else:
            if add_to == 'input_ids':
                if self.pre_seq_len != 0:
                    input_elements += [self.cls_token, self.mem_token_ids, self.sep_token, tensor, self.sep_token]
                else:
                    input_elements += [self.cls_token, tensor, self.sep_token]
            elif add_to == 'labels':
                # NOTE: just test
                # pad_value = -100
                # masked_labels = torch.ones((1), device=tensor.device, dtype=tensor.dtype) * pad_value
                # input_elements += [masked_labels, masked_labels.repeat(self.pre_seq_len), self.eos_token, tensor, masked_labels]
                # input_elements += [masked_labels, masked_labels.repeat(self.pre_seq_len), masked_labels, tensor, masked_labels]
                input_elements += [self.eos_token, tensor, self.sep_token]
                
        tensor = torch.cat(input_elements)
        # print(f'{tensor[:11]=}')
        # Add padding tokens
        # TODO: implement summary module
        #       now config.sum_token_size default = 0
        pad_size = segment_size - tensor.shape[0] - self.config.sum_token_size
        if pad_size > 0:
            if add_to == 'input_ids':
                tensor = F.pad(tensor, (0, pad_size), value=self.pad_token_id)
            elif add_to == 'labels':
                # for Seq2Seq labels need to be pad by -100
                tensor = F.pad(tensor, (0, pad_size), value=-100)
        return tensor  
    
    def prepare_kwargs(self, segment, kwargs):
        segment_input_ids, segment_label = segment
        seg_kwargs = dict(**kwargs)
            
        non_empty_mask = [not self.is_padding(s) for s in segment_input_ids]
        # all the segments are None, due to the max_n_segments >> the number of segments        
        if sum(non_empty_mask) == 0:
            return None, non_empty_mask
        
        # convert list to tensor
        segment_input_ids = [tensor.to(self.model.device) if tensor is not None else None for tensor in segment_input_ids]
        input_ids = torch.stack([s for s in segment_input_ids if s is not None])

        seg_kwargs['inputs_embeds'] = self.model.embeddings(input_ids)
        # print(f"{seg_kwargs['inputs_embeds'].shape=}")
        seg_kwargs['input_ids'] = None
        
        seg_kwargs['attention_mask'] = self.get_attention_mask(input_ids)
        
        if seg_kwargs['labels'] is not None:
            segment_label = [tensor.to(self.model.device) if tensor is not None else None for tensor in segment_label]
            seg_kwargs['labels'] = torch.stack([l for l in segment_label if l is not None])
        
        return seg_kwargs, non_empty_mask   
      
    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask
    
    def is_padding(self, tensor):
        return tensor[0] == self.pad_token_id
    
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
        
        kwargs = {
            'attention_mask': attention_mask, 
            'inputs_embeds': inputs_embeds,
            'labels': labels, 
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states, 
            'return_dict': return_dict,
        }
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print('requires_grad = True')
        #         print(f'{name}: {param.shape}')

        # print(f'{self.model.embeddings=}')
        # segmented: [max_n_segments, batch_size, segment_size]
        # !!! Note: the batch_size is not the same as the input batch_size
        segmented = self.pad_and_segment(
            input_ids=input_ids,
            labels=labels,
        )
        
        model_outputs = []
        if self.config.pre_seq_len != 0:
            memory = self.set_memory(input_ids.shape)
        else:
            memory = None
        # print(f'{self.model.embeddings.weight=}')
        for seg_num, segment in enumerate(zip(*segmented)):
            in_ids, l = segment
            # print(f'{seg_num=}, {memory=}')
            if self.config.bptt_depth != -1:
                raise NotImplementedError
            
            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            
            if memory is not None:
                seg_kwargs['inputs_embeds'][:, self.memory_position][non_empty_mask] = memory[non_empty_mask]
                # print(f'{seg_num=}, {seg_kwargs["inputs_embeds"]=}')
                # print(f'{seg_kwargs["inputs_embeds"].shape=}')
                out = self.model(**seg_kwargs)
                # (batch_size, sequence_length, hidden_size) 
                memory = out.encoder_last_hidden_state[:, self.memory_position]
                # print(f'{seg_num=}, {out.loss=}')
            else:
                out = self.model(**seg_kwargs)
                # print(f'{seg_num=}, {out.loss=}')
            # print(f'{seg_num}')
            # print(f'{out.loss.shape=}, {out.loss=}')
            model_outputs.append(out)
        
        out = self.process_outputs(model_outputs, output_attentions, output_hidden_states)
        return out    
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_specific_kwargs
    ) -> torch.LongTensor:

        kwargs = {
            'input_ids': input_ids,
            'num_beams': num_beams,
            'min_length': min_length,
            'max_length': max_length,
            'labels': None,
            'attention_mask': None,
        }
        
        segmented = self.pad_and_segment(
            input_ids=input_ids,
        )
        
        model_outputs = []
        final_index = []
        if self.config.pre_seq_len != 0:
            memory = self.set_memory(input_ids.shape)
        else:
            memory = None

        for seg_num, segment in enumerate(zip(*segmented)):
            # print(f'{seg_num=}, {memory=}')
            in_ids, l = segment
            
            if self.config.bptt_depth != -1:
                raise NotImplementedError
            
            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            
            if memory is not None:
                seg_kwargs['inputs_embeds'][:, self.memory_position][non_empty_mask] = memory[non_empty_mask] 
                if seg_num == len(segmented) - 1:
                    out = self.model.generate(**seg_kwargs)
                else:
                    # print(f'{seg_kwargs["inputs_embeds"]} after memory=') 
                    out = self.model.generate(**seg_kwargs) 
                    for param in ['min_length', 'max_length', 'num_beams', 'labels']:
                        if param in seg_kwargs:
                            seg_kwargs.pop(param)
                    # print(f"{seg_num=} {seg_kwargs['inputs_embeds'][:,self.memory_position]=}")      
                    encoder_out = self.model.model.encoder(
                        **seg_kwargs,
                    )
                    memory = encoder_out.last_hidden_state[:, self.memory_position]
            else:
                out = self.model.generate(**seg_kwargs)
            
            # print('out: ', out)
            model_outputs.append(out)

            if not len(final_index):
                for index, non_pad in enumerate(non_empty_mask):
                    final_index.append(seg_num)
                    
            else:
                for index, non_pad in enumerate(non_empty_mask):
                    if non_pad:
                        final_index[index] = seg_num
        # print(f'model_outputs: {model_outputs}')
        final_outputs = []
        for idx, _ in enumerate(non_empty_mask):
            final_outputs.append(model_outputs[(final_index[idx])][idx])
        
        # print(f'final_outputs: {final_outputs}')
        return final_outputs    
    
    def process_outputs(self, model_outputs, output_attentions, output_hidden_states):
        rmt_out = model_outputs[-1]

        segment_keys = ['loss']
        if output_attentions:
            segment_keys.append('attentions')
        if output_hidden_states:
            segment_keys.append('hidden_states')

        extracted = {}
        for seg_num, out in enumerate(model_outputs):
            for key, value in out.items():
                if any([sk in key for sk in segment_keys]):
                    extracted[f'{key}_{seg_num}'] = value

        # if self.rmt_config['sum_loss']:
        losses = [out['loss'] for out in model_outputs]
        extracted['loss'] = torch.stack(losses).mean(dim=0)

        for key, value in extracted.items():
            rmt_out[key] = value
        
        # drop unnecessary hiddens to save memory
        if not output_hidden_states:
            for key in rmt_out.keys():
                if 'hidden_state' in key:
                    rmt_out[key] = None

        return rmt_out 
    
# prefix-tuning/p-tuning v2 version
class BartPrefixForConditionalGeneration(BartPreTrainedModel):
    config_class = PromptBartConfig
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = BartForConditionalGeneration(config)
        self.model = BartModel(config)
        self.registerbuffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.tokenizer = BartTokenizer.from_pretrained(kwargs['tokenizer_name_or_path'])
        
        self.config = config
        self.segment_alignment = config.segment_alignment
        self.extract_special_tokens(self.tokenizer)
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        
        # tokenizer.num_special_tokens_to_add()cal the number of special tokens needed to add except [SEP]
        self.segment_size = config.input_size - self.pre_seq_len - self.tokenizer.num_special_tokens_to_add()
        if 'sep_token' in self.tokenizer.special_tokens_map:
            self.segment_size -= 1
        
        # TODO: forget some part of long range memory and add new memory
    
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
            [sample1_seg3, full_padding, sample3_seg3],
            [full_padding, full_padding, sample3_seg4]
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
            # Convert non-zero elements to 1
            drop_mask = torch.tensor([1 if t != 0 else 0 for t in drop_mask])
            
            # bool type slice for tensor type
            # remove special tokens
            seq = seq[(1 - drop_mask).bool()]
            
            # truncate the sequence to the maximum length
            seq = seq[:self.segment_size * self.config.max_n_segments]
            
            if attn_mask is not None:
                attn_mask_drop_mask = sum([attn_mask == self.pad_token_id])
                attn_mask = attn_mask[attn_mask_drop_mask.bool()]
                attn_mask = attn_mask[:self.segment_size * self.config.max_n_segments]
            if label is not None:
                label_drop_mask = sum([label == t for t in self.special_token_ids + [-100]])
                label_drop_mask = torch.tensor([1 if t != 0 else 0 for t in label_drop_mask])
                label = label[(1-label_drop_mask).bool()]
                # TODO：label = label[:self.config.sum_max_size * self.config.max_n_segments]
                label = label[:self.segment_size * self.config.max_n_segments]
            
            align = self.segment_alignment
            if align in {'right', None}:
                split_inds = (list(range(len(seq), 0, -self.segment_size)) + [0])[::-1]
            elif align == 'left':
                split_inds = list(range(0, len(seq), self.segment_size)) + [len(seq)]
            elif align == 'center':
                n_seg = math.ceil(len(seq) / self.segment_size)
                split_inds = list(range(0, len(seq), math.ceil(len(seq) / n_seg))) + [len(seq)]
            else:
                raise NotImplementedError

            input_segments = [seq[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
            input_segments = [self.pad_add_special_tokens(t, self.config.input_size) for t in input_segments]
            
            # add empty segment markers if needed
            n_empty_segments = self.config.max_n_segments - len(input_segments)
            # input_segments:
            input_segments = input_segments + [self.get_full_padding_segment()] * n_empty_segments
            
            # segmented_batch: 
            segmented_batch.append(input_segments)
            
            if attn_mask is not None:
                attn_mask_segments = [attn_mask[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
                attn_mask_segments = [self.pad_add_special_tokens(t, self.config.input_size, add_to='attention_mask') for t in attn_mask_segments]
                attn_mask_segments = attn_mask_segments + [self.get_full_padding_segment()] * n_empty_segments
                segmented_batch_attention_masks.append(attn_mask_segments)
            
            # TODO: labels need to be segmented by other rules
            if label is not None:
                # full_segment_size = len(input_segments)
                end_index = math.ceil(len(label) // (full_segment_size := len(input_segments)))
                labels_segments = [label[:(end_index*(i+1))] for i in range(full_segment_size)]
                labels_segments = [self.pad_add_special_tokens(t, self.config.input_size, add_to='labels') for t in labels_segments]
                labels_segments = labels_segments + [self.get_full_padding_segment()] * n_empty_segments
                segmented_batch_labels.append(labels_segments)
                
        segmented_batch = [[sample[seg_num] for sample in segmented_batch] 
                            for seg_num in range(self.config.max_n_segments)]
        segmented_batch_attention_masks = [[sample[seg_num] for sample in segmented_batch_attention_masks]
                                           for seg_num in range(self.config.max_n_segments)]
        segmented_batch_labels = [[sample[seg_num] for sample in segmented_batch_labels]
                                  for seg_num in range(self.config.max_n_segments)]
        return segmented_batch, segmented_batch_attention_masks, segmented_batch_labels
        
    def get_full_padding_segment(self,):
        padding_segment = torch.tensor([self.pad_token_id for _ in range(self.config.input_size)])
        return padding_segment
    
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
                # NOTE: for Seq2Seq Models labels are used for decoder_input_ids in training
                # and decoder_input_ids must start with the eos_token
                input_elements += [self.eos_token, tensor, self.sep_token]
        # For prefix-tuning/p-tuning v2
        else:
            if add_to == 'input_ids':
                input_elements += [self.cls_token, tensor, self.sep_token]
            elif add_to == 'attention_mask':
                mask_value = torch.ones((1), device=tensor.device)
                input_elements += [mask_value, tensor, mask_value]
            elif add_to == 'labels':
                input_elements += [self.eos_token, tensor, self.sep_token]
        tensor = torch.cat(input_elements)
        
        # Add padding tokens
        # TODO: implement summary module
        #       now config.sum_token_size default = 0
        pad_size = segment_size - tensor.shape[0] - self.config.sum_token_size
        if pad_size > 0:
            if add_to == 'input_ids':
                tensor = F.pad(tensor, (0, pad_size), value=self.pad_token_id)
            elif add_to == 'attention_mask':
                tensor = F.pad(tensor, (0, pad_size), value=0)
            elif add_to == 'labels':
                # for Seq2Seq labels need to be pad by -100
                tensor = F.pad(tensor, (0, pad_size), value=-100)
        return tensor

        # TODO: this implementation just add <s> and </s> to the input sequence
        #       maybe need to add other special tokens
    
    def prepare_kwargs(self, segment, kwargs):
        segment_input_ids, segment_attention_mask, segment_label = segment
        seg_kwargs = dict(**kwargs)
        
        # [sample1_seg1, sample2_seg1, sample3_seg1,....] up to batch_size
        # Some of the segments are None like: [sample1_seg3, None, sample3_seg3]
        # TODO: need another method to deal with this situation
        # segment_input_ids : batch_size * seq_len
            
        non_empty_mask = [not self.is_padding(s) for s in segment_input_ids]
        # all the segments are None, due to the max_n_segments >> the number of segments        
        if sum(non_empty_mask) == 0:
            return None, non_empty_mask
        
        # convert list to tensor
        segment_input_ids = [tensor.to(self.model.device) if tensor is not None else None for tensor in segment_input_ids]
        input_ids = torch.stack([s for s in segment_input_ids if s is not None])
        seg_kwargs['input_ids'] = input_ids
        
        if segment_attention_mask is not None:
            seg_kwargs['attention_mask'] = self.get_attention_mask(input_ids)
        
        if seg_kwargs['labels'] is not None:
            seg_kwargs['labels'] = torch.stack([l for l in segment_label if l is not None])
        
        # # generate prompts 
        # batch_size = input_ids.shape[0]
        # past_key_values = self.get_prompt(batch_size)
        # prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len)
        # if segment_attention_mask is not None:
            # attention_mask = torch.stack([s for s in segment_attention_mask if s is not None])
            # attn_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        # seg_kwargs['attention_mask'] = attn_mask
        # seg_kwargs['past_key_values'] = past_key_values
        
        return seg_kwargs, non_empty_mask
        
    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask
    
    def is_padding(self, tensor):
        return tensor[0] == self.pad_token_id
        
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
        
        kwargs = {
            'attention_mask': attention_mask, 
            'inputs_embeds': inputs_embeds,
            'labels': labels, 
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states, 
            'return_dict': return_dict,
        }

        # segmented: [max_n_segments, batch_size, segment_size]
        # !!! Note: the batch_size is not the same as the input batch_size
        segmented = self.pad_and_segment(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        # NOTE: why???
        # if self.pre_seq_len == 0:
        #     segmented = segmented[-1:]
        
        model_outputs = []
        for seg_num, segment in enumerate(zip(*segmented)):
            in_ids, attn_mask, l = segment
            # TODO: can't control the number of gradient accumulation steps now
            if self.config.bptt_depth != -1:
                raise NotImplementedError
            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            out = self.model(**seg_kwargs)
            model_outputs.append(out)
            # memory_tokens = out.past_key_values
            # print("memory_tokens: ", memory_tokens.shape)
        
        out = self.process_outputs(input_ids, model_outputs, output_attentions, output_hidden_states)
        return out
        
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_specific_kwargs
    ) -> torch.LongTensor:

        kwargs = {
            'input_ids': input_ids,
            'num_beams': num_beams,
            'min_length': min_length,
            'max_length': max_length,
            'labels': None,
            'attention_mask': None
        }
        
        segmented = self.pad_and_segment(
            input_ids=input_ids,
        )
        
        model_outputs = []
        final_index = []
        for seg_num, segment in enumerate(zip(*segmented)):
            
            in_ids, attn_mask, l = segment
            
            if self.config.bptt_depth != -1:
                raise NotImplementedError
            
            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            out = self.model.generate(**seg_kwargs)
            
            # just save the last non-padding segment output

            model_outputs.append(out)

            if not len(final_index):
                for index, non_pad in enumerate(non_empty_mask):
                    final_index.append(seg_num)
                    
            else:
                for index, non_pad in enumerate(non_empty_mask):
                    if non_pad:
                        final_index[index] = seg_num
                        
        final_outputs = []
        for idx, _ in enumerate(non_empty_mask):
            final_outputs.append(model_outputs[(final_index[idx])][idx])
        return final_outputs
        
    def process_outputs(self, input_ids, model_outputs, output_attentions, output_hidden_states):
        out = model_outputs[-1] # get the last segment output
        
        bs, seq_len = input_ids.shape
        
        losses = []
        logits = []
        labels_segm = []
        
        for out in model_outputs:
            losses.append(out['loss'])
            logits.append(out['logits'].detach())
            # if out['seg_kwargs'] is not None:
            #     labels_segm += [out['seg_kwargs']['labels']]
        
        if not output_hidden_states:
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None
                    
        for i, l in enumerate(losses):
            out[f'loss_{i}'] = l.mean()
            
        out['loss'] = torch.stack(losses).mean()
        
        # TODO: need to be fixed | out of index error
        # for i in range(len(logits)):
        #     logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
        #     labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, bs - labels_segm[i].shape[0]), value=-100)
        
        out['logits'] = torch.cat(logits, dim=1)
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        
        return out
    
# prefix-propagation version
class BartPrefixPropForConditionalGeneration(BartPreTrainedModel):
    # base_model_prefix = "model"
    # _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    # _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    config_class = PromptBartConfig
    
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = BartForConditionalGeneration(config)
        
        self.tokenizer = BartTokenizer.from_pretrained(kwargs['tokenizer_name_or_path'])
        
        self.config = config
        self.segment_alignment = config.segment_alignment
        self.extract_special_tokens(self.tokenizer)
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        
        # tokenizer.num_special_tokens_to_add()cal the number of special tokens needed to add except [SEP]
        self.segment_size = config.input_size - self.pre_seq_len - self.tokenizer.num_special_tokens_to_add()
        if 'sep_token' in self.tokenizer.special_tokens_map:
            self.segment_size -= 1

        for param in self.model.parameters():
            param.requires_grad = False

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config, propagate_prefix=config.propagate_prefix)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        model_param = 0
        all_param = 0
        
        # count the number of trainable parameters in bart
        for name, param in self.model.named_parameters():
            model_param += param.numel() # numel() returns the total number of elements in the input tensor
            
        for name, param in self.named_parameters():
            all_param += param.numel()
            
        trainable_param = all_param - model_param
        
        print("Total parameters: {:,}".format(all_param))
        print("Trainable parameters: {:,} {:,%}".format((trainable_param), trainable_param/all_param))

        # Initialize weights and apply final processing
        # 这里不会重写from_pretrained的参数, 这里保证了有些非pretrained的参数被随机初始化
        self.post_init()
        
    def get_prompt(self, batch_size, memory=None):
        if memory is not None:
            prompts = self.prefix_encoder(memory)
        else: 
            prefix_tokens = (
                self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.model.device)
            )
            prompts = self.prefix_encoder(prefix_tokens)
        return prompts
    
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


class BartForPubmed(RMTBaseModel):
    raise NotImplementedError

class RMTForPubmed(RMTBaseModel):
    raise NotImplementedError
    
    def generate(self,):
        """
        class transformers.generation.BeamSearchEncoderDecoderOutput: 
        encoder_hidden_states (tuple(torch.FloatTensor), optional, 
        returned when output_hidden_states=True is passed 
        or when config.output_hidden_states=True) 
        — Tuple of torch.FloatTensor 
        (one for the output of the embeddings + 
        one for the output of each layer) of shape 
        (batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)."
        """
        
        # if we want to take memory from each step
        # need to pass output_hidden_states=True
        raise NotImplementedError
    