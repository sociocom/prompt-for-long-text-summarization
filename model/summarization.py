import torch 
import torch.nn as nn
import torch.nn.functional as F

import logging
import math
from typing import Dict, List, Callable, Optional, Tuple, Union

from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.generation.utils import GenerateOutput
from transformers.generation.streamers import BaseStreamer

from transformers import (
    LogitsProcessorList, 
    StoppingCriteriaList,
    PreTrainedModel,
    GenerationConfig,
) 

from model.base import RMTBaseModel

logger = logging.getLogger(__name__)

class BartForPubmed(RMTBaseModel):
    """
    Using BartForConditionalGeneration as base model
    Without any additional modification
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generation_config = self.model.generation_config
        # self.generation_config.max_length = self.generation_config.max_length * 4
    
    def forward(
        self,
        input_ids: torch.LongTensor = None, # our model input_ids is different from BartForConditionalGeneration torch.LongTensor
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
            'use_cache': use_cache,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict
        }
        
        base_model_outputs = []
        
        # reshape input_ids, attention_mask, labels
        input_ids, attention_mask, labels = self._prepare_batch_inputs(input_ids, attention_mask, labels)
        
        for sec_num, sec_input_ids in enumerate(input_ids):
            if self.rmt_config.bptt_depth != -1:
                raise NotImplementedError
            
            sec_attention_mask = attention_mask[sec_num]
            sec_labels = labels[sec_num]
            sec_kwargs = self._prepare_kwargs(
                sec_input_ids=sec_input_ids,
                sec_attention_mask=sec_attention_mask,
                sec_labels=sec_labels,
                kwargs=kwargs)
            
            sec_outputs = self.model(**sec_kwargs)
            base_model_outputs.append(sec_outputs)
            
        model_outputs = self._process_outputs(base_model_outputs, output_attentions, output_hidden_states)
        return model_outputs
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
                `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
                generating before other GPUs. Otherwise it'll be set to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchDecoderOnlyOutput`],
                    - [`~generation.SampleDecoderOnlyOutput`],
                    - [`~generation.BeamSearchDecoderOnlyOutput`],
                    - [`~generation.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchEncoderDecoderOutput`],
                    - [`~generation.SampleEncoderDecoderOutput`],
                    - [`~generation.BeamSearchEncoderDecoderOutput`],
                    - [`~generation.BeamSampleEncoderDecoderOutput`]
        """    

        sec_kwargs = {
            'generation_config': generation_config,
            'logits_processor': logits_processor,
            'stopping_criteria': stopping_criteria,
            'prefix_allowed_tokens_fn': prefix_allowed_tokens_fn,
            'synced_gpus': synced_gpus,
            'assistant_model': assistant_model,
            'streamer': streamer,
            'negative_prompt_ids': negative_prompt_ids,
            'negative_prompt_attention_mask': negative_prompt_attention_mask,
        }
        
        if kwargs is not None:
            for key, values in kwargs.items():
                sec_kwargs[key] = values
        for param in ["attention_mask", "labels"]:
            if param in sec_kwargs:
                sec_kwargs.pop(param)
        
        base_model_outputs = []
        input_ids, attention_mask, labels = self._prepare_batch_inputs(sec_kwargs['input_ids'])
        
        for idx, sec_inputs in enumerate(input_ids):
            if self.rmt_config.bptt_depth != -1:
                raise NotImplementedError
            
            sec_kwargs['input_ids'] = sec_inputs
            sec_outputs = self.model.generate(**sec_kwargs)
            base_model_outputs.append(sec_outputs)
        
        base_model_outputs = self._process_generation_outputs(base_model_outputs)

        return base_model_outputs
    
class BartRMTForPubmed(RMTBaseModel):
    """
    Using BartForConditionalGeneration as base model
    Without RMT memory structure
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generation_config = self.model.generation_config
        # self.generation_config.max_length = self.generation_config.max_length * 4    
        
    # def _process_generation_outputs(self, model_outputs):
        
    #     outputs = []
    #     for batch_idx in range(len(model_outputs[0])):
    #         batch_outputs = []
    #         for sample in model_outputs:
    #             batch_outputs.append(sample[batch_idx])
    #         batch_outputs = torch.concat(batch_outputs)
    #         outputs.append(batch_outputs)
            
    #     outputs = torch.stack([o for o in outputs])
            
    #     return outputs

    def _process_generation_outputs(self, model_outputs):
        outputs = []
        print(f'length of model_outputs: {len(model_outputs)}')
        
        for batch_idx in range(len(model_outputs[0])):
            batch_outputs = []
            for sample in model_outputs:
                batch_outputs.append(sample[batch_idx])
            
            # Add print statements to debug
            print(f"Batch Index: {batch_idx}")
            print(f"Sizes of individual tensors in batch_outputs: {[o.size() for o in batch_outputs]}")

            max_size = self.rmt_config.max_target_length
            batch_outputs_padded = [torch.nn.functional.pad(o, (0, max_size - o.size(0)), value=-100) for o in batch_outputs]
            
            batch_outputs_padded = torch.stack([o for o in batch_outputs_padded])
            outputs.append(batch_outputs_padded)
        
        # Add print statement to debug
        print(f"Sizes of individual tensors in outputs: {[o.size() for o in outputs]}")
        
        outputs = torch.stack([o for o in outputs])
        return outputs
    
    def _pad_generation_output(self, tensor):
        return F.pad(tensor, (0, self.rmt_config.post_seq_len-tensor.shape[1]), value=self.pad_token_id)
        
    def forward(
        self,
        input_ids: torch.LongTensor = None, # our model input_ids is different from BartForConditionalGeneration torch.LongTensor
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
            'use_cache': use_cache,
            'output_attentions': output_attentions,
            'output_hidden_states': True,
            'return_dict': return_dict
        }
        
        base_model_outputs = []
        
        if self.rmt_config.pre_seq_len != 0:
            memory = self._set_memory(input_ids.shape[0])
        summary_embeds = None
        
        input_ids, attention_mask, labels = self._prepare_batch_inputs(input_ids, attention_mask, labels)
        input_ids, attention_mask = self._init_prefix_postfix(input_ids, attention_mask)
        
        for sec_num, sec_input_ids in enumerate(input_ids):
            if self.rmt_config.bptt_depth != -1:
                raise NotImplementedError
            
            sec_attention_mask = attention_mask[sec_num]
            sec_labels = labels[sec_num]
            
            sec_kwargs = self._prepare_kwargs(
                sec_input_ids=sec_input_ids,
                sec_attention_mask=sec_attention_mask,
                sec_labels=sec_labels,
                kwargs=kwargs
            )
            
            if self.rmt_config.pre_seq_len != 0:
                sec_kwargs['inputs_embeds'][:, self.memory_position] = memory
            
            if summary_embeds is not None:
                sec_kwargs['inputs_embeds'][:, self.summary_position] = summary_embeds
            
            sec_outputs = self.model(**sec_kwargs)
            base_model_outputs.append(sec_outputs)
            
            if self.rmt_config.pre_seq_len != 0:
                memory = sec_outputs.encoder_last_hidden_state[:, self.memory_position]
            
            if self.rmt_config.post_seq_len != 0:
                summary_embeds = sec_outputs.decoder_hidden_states[-1]
            
        model_outputs = self._process_outputs(base_model_outputs, output_attentions, output_hidden_states)
        
        return model_outputs
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        sec_kwargs = {
            'generation_config': generation_config,
            'logits_processor': logits_processor,
            'stopping_criteria': stopping_criteria,
            'prefix_allowed_tokens_fn': prefix_allowed_tokens_fn,
            'synced_gpus': synced_gpus,
            'assistant_model': assistant_model,
            'streamer': streamer,
            'negative_prompt_ids': negative_prompt_ids,
            'negative_prompt_attention_mask': negative_prompt_attention_mask,
        }
        # if self.rmt_config.post_seq_len != 0:
        #     sec_kwargs['output_hidden_states'] = True
        #     sec_kwargs['return_dict_in_generate'] = True,
        
        encoder_sec_kwargs = {
            'output_hidden_states': True,
        }
        
        if kwargs is not None:
            for key, values in kwargs.items():
                sec_kwargs[key] = values
        
        base_model_outputs = []        
        
        memory = self._set_memory(sec_kwargs['input_ids'].shape[0])
        summary_embeds = None
        
        input_ids, attention_mask, labels = self._prepare_batch_inputs(
            input_ids=sec_kwargs['input_ids'],
            attention_mask=sec_kwargs['attention_mask'],
        )
        input_ids, attention_mask = self._init_prefix_postfix(
            input_ids,
            attention_mask=attention_mask,
        )
        
        # generate() has _prepare_attention_mask_for_generation 
        # so we don't need to pass it here
        for param in ["attention_mask", "labels"]:
            if param in sec_kwargs:
                sec_kwargs.pop(param) 
                       
        for sec_num, sec_inputs in enumerate(input_ids):
            
            if self.rmt_config.bptt_depth != -1:
                raise NotImplementedError        

            encoder_sec_kwargs = self._prepare_kwargs(
                sec_input_ids=sec_inputs,
                kwargs=encoder_sec_kwargs,
            )     
               
            encoder_sec_kwargs['inputs_embeds'][:, self.memory_position] = memory
            if summary_embeds is not None:
                encoder_sec_kwargs['inputs_embeds'][:, self.summary_position] = summary_embeds            

            sec_attention_mask = torch.ones_like(sec_inputs)
            sec_attention_mask[sec_inputs == self.pad_token_id] = 0
            encoder_sec_kwargs['attention_mask'] = sec_attention_mask

            encoder_outputs = self.model.get_encoder()(**encoder_sec_kwargs)
            memory = encoder_outputs.last_hidden_state[:, self.memory_position]
            
            sec_kwargs['input_ids'] = None
            sec_kwargs['encoder_outputs'] = encoder_outputs
            
            sec_outputs = self.model.generate(**sec_kwargs)
            if self.rmt_config.post_seq_len != 0:
                summary_embeds = self._pad_generation_output(sec_outputs)
                summary_embeds = self.model.embeddings(summary_embeds)
                
            base_model_outputs.append(sec_outputs)
            
        base_model_outputs = self._process_generation_outputs(base_model_outputs)

        return base_model_outputs        
        
            