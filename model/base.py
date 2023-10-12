import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMTBaseModel(nn.Module):
    
    def __init__(self, base_model, config):
        super().__init__()
        self.model = base_model
        self.set_params(config)
        
    def forward(self, config):
        raise NotImplementedError