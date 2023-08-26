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

from model.summarization import BartPrefixForConditionalGeneration
from config.custom_config import *
from utils.utils import *

def main():
    args = get_args()
    print(args)

if __name__ == "__main__":
    main()