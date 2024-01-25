import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

class RMTBaseModel(nn.Module):
    
    def __init__(self, base_model, rmt_config, **kwargs):
        super().__init__()
        self.rmt_config = rmt_config
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'])
        base_model = get_peft_model(base_model, peft_config)
        self.model = base_model
        self.model.print_trainable_parameters()
        
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['tokenizer_name_or_path'])
        self._extract_special_tokens(self.tokenizer)
        self._extend_word_embeddings(self.rmt_config.pre_seq_len, self.tokenizer)
        
        if rmt_config.freeze_model:
            for name, param in self.model.named_parameters():
                if name != 'model.shared.weight':
                    param.requires_grad = False
                    print(name, param.requires_grad)
                elif name == 'model.shared.weight':
                    print(name, param.requires_grad)
    
    def _extract_special_tokens(self, tokenizer):
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
    
    def _extend_word_embeddings(self, num_mem_tokens, tokenizer):
        # if num_mem_tokens != 0:   
        vocab_size = self.model.config.vocab_size
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.model.resize_token_embeddings(extended_vocab_size)

        special_tokens = tokenizer.special_tokens_map
        mem_start_ind = int('cls_token' in special_tokens or 'bos_token' in special_tokens)
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)
        self.summary_position = range(1 + num_mem_tokens + self.rmt_config.max_source_length, self.rmt_config.max_section_length)
        self.model.embeddings = self.model.get_input_embeddings()
        
    def _set_memory(self, batch_size):
        memory = self.model.embeddings(self.mem_token_ids)
        memory = memory.repeat(batch_size, 1, 1)
        return memory 

    def _prepare_kwargs(
        self, 
        sec_input_ids, 
        kwargs,
        sec_attention_mask=None,
        sec_labels=None,
    ):
        sec_kwargs = dict(**kwargs)
        
        sec_kwargs['input_ids'] = None
        sec_kwargs['inputs_embeds'] = self.model.embeddings(sec_input_ids)
        
        if sec_attention_mask is not None:
            sec_kwargs['attention_mask'] = sec_attention_mask
        if sec_labels is not None:
            sec_kwargs['labels'] = sec_labels

        return sec_kwargs

    def _prepare_batch_inputs(self, input_ids, attention_mask=None, labels=None):
        batch_input_ids = None # batch_size, section_num, seq_len 
        batch_attention_mask = None
        batch_labels = None
        
        batch_input_ids = torch.stack([
            torch.stack([sample[sec_num] for sample in input_ids])
            for sec_num in range(input_ids.shape[1])
        ])
    
        if attention_mask is not None:
            batch_attention_mask = torch.stack([
                torch.stack([sample[sec_num] for sample in attention_mask])
                for sec_num in range(attention_mask.shape[1])
            ])
            
        if labels is not None:
            batch_labels = torch.stack([
                torch.stack([sample[sec_num] for sample in labels])
                for sec_num in range(labels.shape[1])
            ])
            
        return batch_input_ids, batch_attention_mask, batch_labels

    def _init_prefix_postfix(self, input_ids, attention_mask=None):
        
        processed_input_ids = []
        for sec_num, sec_input_ids in enumerate(input_ids):
            sec_input_ids = torch.cat([
                self.cls_token.expand(sec_input_ids.shape[0], -1),
                self.mem_token_ids.expand(sec_input_ids.shape[0], -1),
                sec_input_ids,
                self._get_postfix_padding().to(self.model.device).expand(sec_input_ids.shape[0], -1)],
                # self.sep_token.expand(sec_input_ids.shape[0], -1)], 
                dim=1,
            )
            processed_input_ids.append(sec_input_ids)
        processed_input_ids = torch.stack(processed_input_ids)
        
        if attention_mask is not None:
            processed_attention_mask = []
            for sec_num, sec_attention_mask in enumerate(attention_mask):
                sec_attention_mask = torch.cat([
                    torch.ones(sec_attention_mask.shape[0], 1, dtype=torch.long).to(self.model.device),
                    torch.ones(sec_attention_mask.shape[0], self.rmt_config.pre_seq_len, dtype=torch.long).to(self.model.device),
                    sec_attention_mask,
                    self._get_postfix_attention_mask().to(self.model.device).expand(sec_attention_mask.shape[0], -1)],
                    # torch.ones(sec_attention_mask.shape[0], 1, dtype=torch.long).to(self.model.device)], 
                    dim=1,
                )
                processed_attention_mask.append(sec_attention_mask)
            processed_attention_mask = torch.stack(processed_attention_mask)
        else:
            processed_attention_mask = None
            
        return processed_input_ids, processed_attention_mask
    
    def _get_postfix_padding(self,):
        return torch.ones(self.rmt_config.post_seq_len, dtype=torch.long) * self.pad_token_id
        
    def _get_postfix_attention_mask(self,):
        return torch.zeros(self.rmt_config.post_seq_len, dtype=torch.long)  
     
    def _process_generation_outputs(self, model_outputs):
        outputs = []
        for batch_idx in range(len(model_outputs[0])):
            batch_outputs = []
            for sample in model_outputs:
                batch_outputs.append(sample[batch_idx])
            batch_outputs = torch.concat(batch_outputs)
            outputs.append(batch_outputs)
        
        outputs = torch.stack([o for o in outputs])
        return outputs

    def _process_outputs(self, model_outputs, output_attentions, output_hidden_states):
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
             
        if self.rmt_config.sum_loss:       
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