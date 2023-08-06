# Modified from RMT
# Modified parts are marked with "MODIFIED"
# https://arxiv.org/abs/2304.11062
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptRMTBaseModel(nn.Module):
    # MODIFIED
    def __init__(self, base_model, config):
        super().__init__()
        self.model = base_model
        self.set_params(config)
        
    def forward(self, config):
        raise NotImplementedError
    
    def pad_and_segment(self, input_ids, sum_incurrence: bool = False):
        segmented_batch = []
        for seq in input_ids:
            pass
        