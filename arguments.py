from transformers import HfArgumentParser, TrainingArguments

from typing import Optional, Literal
from dataclasses import dataclass, field

DATASETS = ["cnn_dailymail", "xsum", "NYT"]

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.training_args
    """
    
    dataset_name: str = field(
        metadata={
            'help': "The name of the dataset to use: " + ", ".join(DATASETS),
            "choices": DATASETS,
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    early_stopping_patience: Optional[int] = field(
        default=-1,
        metadata={
            "help": "If default or less than 0, no early stopping."
            "Metric to monitor defaults to first in eval dictionary"
        },
    )
    
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    peft_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained peft_config name or path"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    prefix: bool = field(
        default=False, 
        metadata={"help": "Will use P-tuning v2 during training"}
    )
    prompt: bool = field(
        default=False, 
        metadata={"help": "Will use prompt tuning during training"}
    )
    finetune: bool = field(
        default=False, 
        metadata={"help": "Will use regular finetune during training"}
    )
    propagate_prefix: str = field(
        default="none",
        metadata={
            "help": "Will propagate query of prefix (increases parameter count)."
            "Set to `only` for prefix propagation only, `none` to disable, "
            "or `combine` to combine it with prefix tuning (half of the prefix length will "
            "be used for actual prefix tuning, and half for propagated tokens)",
            "choices": ["none", "only", "combine"],
        },
    )
    pre_seq_len: int = field(
        default=20, 
        metadata={"help": "The length of prompt"}
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Apply a two-layer MLP head over the prefix embeddings"},
    )
    # prefix_hidden_size: int = field(
    #     default=512,
    #     metadata={
    #         "help": "The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used"
    #     },
    # )

def get_args():
    """Parse all the args."""
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
        )
    )
    
    args = parser.parse_args_into_dataclasses()
    
    return args