import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

@dataclass
class RMTDataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"    
    
    def __call__(self, features, return_tensors=None):
        # features: list[dict], batch_size = len(features)
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature['labels'] for feature in features] if "labels" in features[0].keys() else None
        print(labels)
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = 0
            for sample in labels:
                max_label_length = max(max_label_length, max(len(s) for s in sample))
            
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    max_label_length + self.pad_to_multiple_of - 1
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            
            padding_side = self.tokenizer.padding_side
            for feature in features:
                for index, seg_label in enumerate(feature['labels']):
                    remainder = [self.label_pad_token_id] * (max_label_length - len(seg_label))
                    if isinstance(seg_label, list):
                        feature['labels'][index] = seg_label + remainder if padding_side == "right" else remainder + seg_label
                    elif padding_side == "right":
                        feature['labels'][index] = np.concatenate(seg_label, remainder).astype(np.int64)
                    else:
                        feature['labels'][index] = np.concatenate(remainder, seg_label).astype(np.int64)
            
        inputs = {
            'input_ids': None,
            'attention_mask': None,
            'labels': None,
        }
        
        max_source_length = 0
        for sample in features:
            max_source_length = max(max_source_length, max(len(s) for s in sample['input_ids']))
        
        for idx, feature in enumerate(features):
            # feature: dict of one sample in batch
            feature = self.tokenizer.pad(
                feature,
                padding='max_length',
                max_length=max_source_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
                
            if idx == 0:
                inputs['input_ids'] = feature['input_ids'].unsqueeze(0)
                inputs['attention_mask'] = feature['attention_mask'].unsqueeze(0)
                inputs['labels'] = feature['labels'].unsqueeze(0)
            else:
                inputs['input_ids'] = torch.cat((inputs['input_ids'], feature['input_ids'].unsqueeze(0)), dim=0)
                inputs['attention_mask'] = torch.cat((inputs['attention_mask'], feature['attention_mask'].unsqueeze(0)), dim=0)
                inputs['labels'] = torch.cat((inputs['labels'], feature['labels'].unsqueeze(0)), dim=0)
        
        # batch_size, sec_len, seq_len
        return inputs
            