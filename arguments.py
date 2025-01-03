from transformers import HfArgumentParser, TrainingArguments, Seq2SeqTrainingArguments

from typing import Optional, Literal
from dataclasses import dataclass, field

MODEL_TYPE = [
    "BaseModel", 
    "BaseModelWithPrefixTuning", 
    "BaseModelWithPrefixProp",
    
    "BaseModelWithRMT",
    "BaseModelWithRMTAndPrefixProp",
]

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
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    peft_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained peft_config name or path"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )    
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )        
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    
    # For PrefixTuning & RMT
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
    post_seq_len: int = field(
        default=128,
        metadata={"help": "The length of the summary cell"}
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Apply a two-layer MLP head over the prefix embeddings"},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})
    
    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    
    # If you want to use your own dataset, you can pass the path to it here
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_section_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization with prefix and postfix. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )    
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )   
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    max_n_segments: Optional[int] = field(
        default=4, metadata={"help": "Define the max segments of training data"}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    max_position_embeddings: Optional[int] = field (
        default=1024, metadata={"help": "Define the max input length that model can handle with."}
    )
                 
    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
@dataclass
class CustomTrainingArguments(Seq2SeqTrainingArguments):
    do_hyper_search: bool = field(
        default=False, metadata={"help": "Run a hyperparameter search"}
    )
    predict_epoch: int = field(
        default=5, metadata={"help": "Ever n epochs to run a predict"}
    )
    model_type: str = field(
        default="Normal",
        metadata={
            "help": "The training strategy to use",
            "choices": MODEL_TYPE,
        },
    )
    freeze_model: bool = field(
        default=False, metadata={"help": "Fix model parameters"}
    )
    task_type: str = field(
        default='Normal', metadata={
            "help": "The task type to use",
            "choices": ["Normal", "Segment"]
        },
    )
    rouge_type: str = field(
        default="Accumulation", metadata={
            "help": "The Rouge type to use, Accumulation means calculate rouge score for each section output."
            "Final means calculate rouge score for the last section output",
            "choices": ["Accumulation", "Final"]
        }
    )
    
def get_args():
    """Parse all the args."""
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            CustomTrainingArguments,
        )
    )
    
    args = parser.parse_args_into_dataclasses()
    
    return args