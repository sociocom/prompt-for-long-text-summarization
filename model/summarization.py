import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import logging
import copy
import math
from typing import List, Optional, Tuple, Union, Iterable

from transformers import (
    BartPretrainedModel,
    BartForConditionalGeneration,
    BartTokenizer,
)
from transformers import BartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput

from model.prefix_encoder import PrefixEncoder
from config import config

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

# prefix-tuning/p-tuning v2 version
class BartPrefixForConditionalGeneration(BartPretrainedModel):
    def __init__(self, config, checkpoint):
        super().__init__(config)

        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.config = config
        self.segment_alignment = config.segment_alignment
        self.extract_special_tokens(self.tokenizer)
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        # self.extend_word_embeddings(config.pre_seq_len, tokenizer)
        
        # tokenizer.num_special_tokens_to_add()cal the number of special tokens needed to add except [SEP]
        self.segment_size = config.input_size - self.pre_seq_len - self.tokenizer.num_special_tokens_to_add()
        if 'sep_token' in self.tokenizer.special_tokens_map:
            self.segment_size -= 1
        
        # TODO: forget some part of long range memory and add new memory

        for param in self.model.parameters():
            param.requires_grad = False

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)
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

    def get_prompt(self, batch_size):
        # get last_hidden_state as prompt
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
        # (2,batch_size,n_head,seq_len,head_dim)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
    
    # TODO：labels按照比例切分
    # TODO: 25% -> 50% -> 75% -> 100% -> 100% -> 100% -> 100% -> 100% -> 100% -> 100%
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
                labels_segments = [label[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
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
        non_empty_mask = [s is not None for s in segment_input_ids]
        # all the segments are None, due to the max_n_segments >> the number of segments        
        if sum(non_empty_mask) == 0:
            return None, non_empty_mask
        
        # convert list to tensor
        input_ids = torch.stack([s for s in segment_input_ids if s is not None])
        seg_kwargs['input_ids'] = input_ids
        
        if segment_attention_mask is not None:
            seg_kwargs['attention_mask'] = self.get_attention_mask(input_ids)
        
        if seg_kwargs['labels'] is not None:
            seg_kwargs['labels'] = torch.stack([el for el, m in zip(segment_label, non_empty_mask) if m])
        
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
            
            # out -> Seq2SeqLMOutput
            # loss: Optional[torch.FloatTensor] = None
            # logits: torch.FloatTensor = None
            # past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
            # decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
            # decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
            # cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
            # encoder_last_hidden_state: Optional[torch.FloatTensor] = None
            # encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
            # encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
            out = self.model(**seg_kwargs)
            
            memory_tokens = out.past_key_values
            print("memory_tokens: ", memory_tokens.shape)
            
            model_outputs.append(out)
        
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
        for seg_num, segment in enumerate(zip(*segmented)):
            in_ids, attn_mask, l = segment
            
            if self.config.bptt_depth != -1:
                raise NotImplementedError
            
            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            
            out = self.model.generate(**seg_kwargs)
            model_outputs.append(out)

        print("model_outputs: ", self.tokenizer.decode(model_outputs[-1][0], skip_special_tokens=True))
        
    def process_outputs(self, input_ids, model_outputs, output_attentions, output_hidden_states):
        out = model_outputs[-1] # get the last segment output
        
        bs, seq_len = input_ids.shape
        
        losses = []
        logits = []
        labels_segm = []
        
        for out in model_outputs:
            losses.append(out['loss'])
            logits.append(out['logits'].detach())
            labels_segm += [out['seg_kwargs']['labels']]
        
        if not output_hidden_states:
            for key in out.keys():
                if 'hidden_state' in key:
                    out[key] = None
                    
        for i, l in enumerate(losses):
            out[f'loss_{i}'] = l.mean()
            
        out['loss'] = torch.stack(losses).mean()
        
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, bs - labels_segm[i].shape[0]), value=-100)
        
        out['logits'] = torch.cat(logits, dim=1)
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        out['logits_segm'] = [logits]
        out['labels_segm'] = [labels_segm]
        
        return out
# prefix-propagation version
class BartPrefixPropForConditionalGeneration(BartPretrainedModel):
    def __init__(self, config, checkpoint):
        super().__init__(config)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        
        self.config = config
        self.segment_alignment = config.segment_alignment
        self.extract_special_tokens(self.tokenizer)
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        # self.extend_word_embeddings(config.pre_seq_len, tokenizer)
        
        # tokenizer.num_special_tokens_to_add()cal the number of special tokens needed to add except [SEP]
        self.segment_size = config.input_size - self.pre_seq_len - self.tokenizer.num_special_tokens_to_add()
        if 'sep_token' in self.tokenizer.special_tokens_map:
            self.segment_size -= 1
        
        # TODO: forget some part of long range memory and add new memory

        for param in self.model.parameters():
            param.requires_grad = False

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)
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
    
    def get_prompt(self, batch_size, memory=None):
        if memory is not None:
            prompts = self.prefix_encoder(memory)
        else: 
            prefix_tokens = (
                self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.model.device)
            )
            prompts = self.prefix_encoder(prefix_tokens)
        return prompts
# ============================================
# ================ T5 model ==================
# ============================================
