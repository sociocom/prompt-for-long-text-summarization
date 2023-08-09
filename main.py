import logging
import torch
import os
import sys
import numpy as np
from typing import Dict

import datasets
import transformers
from transformers import set_seed, Trainer, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

from arguments import get_args

import wandb
import os
os.environ['WANDB_DIR'] = os.getcwd() + '/wandb/'
os.environ['WANDB_CACHE_DIR'] = os.getcwd() + '/wandb/.cache/'
os.environ['WANDB_CONFIG_DIR'] = os.getcwd() + '/wandb/.config/'

logger = logging.getLogger(__name__)

kwargs = {
    "pre_seq_len": 20,
    "input_size": 512,
    "segment_size": 512,
    "max_n_segments": 3,
    "bptt_depth": 2,
    "prefix_projection" : False,
}