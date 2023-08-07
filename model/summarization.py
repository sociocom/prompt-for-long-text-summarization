import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import logger
import copy
import math
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
    def pad_and_segment(self, input_ids):
        """
        segment input_ids into segments
        be careful that all the segments are treated as one input sequence 
        and dealed with incurrence 
        """
        segmented_batch = []
        # input_ids: [batch_size, seq_len]
        for seq in input_ids:
            # pytorch syntax: element-wise operation
            drop_mask = sum([seq == t for t in self.special_token_ids])
            # bool type slice for tensor type
            # remove special tokens
            seq = seq[(1 - drop_mask).bool()]
            # truncate the sequence to the maximum length
            seq = seq[:self.config.segment_size * self.config.max_n_segments]
            
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
            # TODO: do the implementation
            # input_segments = [self.pad_add_special_tokens(t, self.config.input_size) for t in input_segments]
            
            # add empty segment markers if needed
            n_empty_segments = self.config.max_n_segments - len(input_segments)
            # input_segments:
            input_segments = [None] * n_empty_segments + input_segments
            
            # segmented_batch: 
            segmented_batch.append(input_segments)
        
        segmented_batch = [[sample[seg_num] for sample in segmented_batch] \
                            for seg_num in range(self.config.max_n_segments)]
        return segmented_batch
    # End
    
    # TODO: add [SEG] between input and summary
    # Maybe we should use BartDataCollatorForSeq2Seq
    def pad_add_special_tokens(self, **kwargs):
        
        """
        Copied from transformers.BartTokenizer.py

        You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
        call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

        <Tip>

        When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

        </Tip>

        This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
        this superclass for more information regarding those methods.

        Args:
            vocab_file (`str`):
                Path to the vocabulary file.
            merges_file (`str`):
                Path to the merges file.
            errors (`str`, *optional*, defaults to `"replace"`):
                Paradigm to follow when decoding bytes to UTF-8. See
                [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
            bos_token (`str`, *optional*, defaults to `"<s>"`):
                The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

                <Tip>

                When building a sequence using special tokens, this is not the token that is used for the beginning of
                sequence. The token used is the `cls_token`.

                </Tip>

            eos_token (`str`, *optional*, defaults to `"</s>"`):
                The end of sequence token.

                <Tip>

                When building a sequence using special tokens, this is not the token that is used for the end of sequence.
                The token used is the `sep_token`.

                </Tip>

            sep_token (`str`, *optional*, defaults to `"</s>"`):
                The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
                sequence classification or for a text and a question for question answering. It is also used as the last
                token of a sequence built with special tokens.
            cls_token (`str`, *optional*, defaults to `"<s>"`):
                The classifier token which is used when doing sequence classification (classification of the whole sequence
                instead of per-token classification). It is the first token of the sequence when built with special tokens.
            unk_token (`str`, *optional*, defaults to `"<unk>"`):
                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
                token instead.
            pad_token (`str`, *optional*, defaults to `"<pad>"`):
                The token used for padding, for example when batching sequences of different lengths.
            mask_token (`str`, *optional*, defaults to `"<mask>"`):
                The token used for masking values. This is the token used when training this model with masked language
                modeling. This is the token which the model will try to predict.
            add_prefix_space (`bool`, *optional*, defaults to `False`):
                Whether or not to add an initial space to the input. This allows to treat the leading word just as any
                other word. (BART tokenizer detect beginning of words by the preceding space).
            """
            
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
