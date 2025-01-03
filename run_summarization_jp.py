#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import torch
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import spacy
import pandas as pd
import datasets
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, Dataset
from filelock import FileLock

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.generation import GenerationConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
from peft import (
    PrefixTuningConfig, 
    LoraConfig,
    TaskType, 
    get_peft_model
)

from arguments import get_args
from config import RMTBartConfig
from utils import RMTDataCollatorForSeq2Seq
from model.modeling_bart import BartPrefixPropForConditionalGeneration
from model.summarization import BartForPubmed, BartRMTForPubmed

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)
        
# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
    # TODO:
    "pubmed": ("sections", "abstract_text"),
    "NLP_JP_CORPUS_INCREMENTAL_JUMAN": ("sections", "abs_incremental"),
    # Add arXiv
    # Add BookSum
}

import wandb
wandb.init(project="RMT", entity='lkf1013606100')

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    model_args, data_args, training_args = get_args()

    if model_args.use_auth_token is not None:
        warnings.warn("The `use_auth_token` argument is deprecated and will be removed in v4.34.", FutureWarning)
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token
        
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization", model_args, data_args)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )    
    
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
        
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()    

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
        
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )    
            
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # TODO: todo from here
    if data_args.dataset_name == "pubmed":
        raw_datasets = load_dataset(
            "json", 
            data_dir='datasets/pubmed-dataset-processed-final',
        )
    elif data_args.dataset_name == "pubmed-incremental":
        raw_datasets = load_dataset(
            "json",
            data_dir="datasets/pubmed-dataset-incremental",
        )
    elif data_args.dataset_name == "NLP_JP_CORPUS_INCREMENTAL_JUMAN":
        data_frame = pd.read_json('datasets/NLP_JP_CORPUS_INCREMENTAL_JUMAN/NLP_JP_CORPUS_INCREMENTAL_JUMAN.json', orient='records', encoding='utf-8')
        raw_datasets = Dataset.from_pandas(data_frame)
        raw_datasets = raw_datasets.train_test_split(test_size=0.25, seed=42)
        temp = raw_datasets['train'].train_test_split(test_size=0.125/(0.625+0.125), seed=42)
        raw_datasets['train'], raw_datasets['validation'] = temp['train'], temp['test']
        
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            # token=model_args.token,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            # token=model_args.token,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.  
    
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        # token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        # token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if training_args.model_type == "BaseModel":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            # token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )  
        if training_args.task_type == "Segment":
            # prepare rmt parameters
            rmt_config = RMTBartConfig(
                pre_seq_len=model_args.pre_seq_len if model_args.pre_seq_len is not None else 0,
                post_seq_len=model_args.post_seq_len if model_args.post_seq_len is not None else 0,
                freeze_model=training_args.freeze_model,
                max_section_length=data_args.max_source_length,
                max_source_length=data_args.max_source_length-model_args.pre_seq_len-model_args.post_seq_len-1,
                max_target_length=data_args.max_target_length,
                **config.to_dict()
            )
            data_args.max_source_length = data_args.max_source_length - model_args.pre_seq_len - model_args.post_seq_len
            # load rmt model
            if data_args.dataset_name == "pubmed" or data_args.dataset_name == "pubmed-incremental":
                model = BartForPubmed(
                    base_model=model,
                    rmt_config=rmt_config,
                    tokenizer_name_or_path=model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                )
            else:
                raise NotImplementedError
    elif training_args.model_type == "BaseModelWithRMT":
        # load base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            # token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )  
        # prepare rmt parameters
        rmt_config = RMTBartConfig(
            pre_seq_len=model_args.pre_seq_len if model_args.pre_seq_len is not None else 0,
            post_seq_len=model_args.post_seq_len if model_args.post_seq_len is not None else 0,
            freeze_model=training_args.freeze_model,
            max_section_length=data_args.max_source_length,
            max_source_length=data_args.max_source_length-model_args.pre_seq_len-model_args.post_seq_len-1,
            max_target_length=data_args.max_target_length,
            **config.to_dict()
        )
        data_args.max_source_length = data_args.max_source_length - model_args.pre_seq_len - model_args.post_seq_len
        # load rmt model
        if data_args.dataset_name == "pubmed" or data_args.dataset_name == "pubmed-incremental" or data_args.dataset_name == "NLP_JP_CORPUS_INCREMENTAL_JUMAN":
            model = BartRMTForPubmed(
                base_model=base_model,
                rmt_config=rmt_config,
                tokenizer_name_or_path=model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    print(model)
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    if training_args.task_type == "Normal":
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
        
        # For Multi-lingual summarization, we need to set the decoder_start_token_id.
        if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            if isinstance(tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
            else:
                model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)
                
        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")    
    elif training_args.task_type == "Segment":
        embedding_size = model.model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.model.resize_token_embeddings(len(tokenizer))
            
        # For Multi-lingual summarization, we need to set the decoder_start_token_id.
        if model.rmt_config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            if isinstance(tokenizer, MBartTokenizer):
                model.rmt_config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
            else:
                model.rmt_config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)
        
        if model.rmt_config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        
    # Only resize position embedding for baseline models Normal task
    # We have implemented memory mechanism for long documents, so don't need to resize
    if training_args.task_type == "Normal":
        if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
        ):
            if model_args.resize_position_embeddings is None:
                logger.warning(
                    "Increasing the model's number of position embedding vectors from"
                    f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
                )
                model.resize_position_embeddings(data_args.max_source_length)
            elif model_args.resize_position_embeddings:
                model.resize_position_embeddings(data_args.max_source_length)
            else:
                raise ValueError(
                    f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                    f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                    f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                    " model's position encodings by passing `--resize_position_embeddings`."
                )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id
        
    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )    
            
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False    

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if training_args.task_type == "Normal":
        def preprocess_function(examples):
            # remove pairs where at least one record is None
            
            inputs, targets = [], []
            for i in range(len(examples[text_column])):
                if examples[text_column][i] and examples[summary_column][i]:
                    inputs.append(examples[text_column][i])
                    targets.append(examples[summary_column][i])

            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
            
            # Tokenize targets with the `text_target` keyword argument
            labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and data_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
    elif training_args.task_type == "Segment":
        def preprocess_function(examples):
            
            inputs = examples['sections']
            targets = examples['abs_incremental']
            
            model_inputs = {
                'input_ids': [],
                'attention_mask': [],
                'labels': [],
            }
            
            for sample_input, sample_targets in zip(inputs, targets):
                section_input_ids = []
                section_attention_mask = []
                section_labels = []
                for section, target in zip(sample_input, sample_targets):
                    sample_input_ids = tokenizer(
                        section, 
                        max_length=data_args.max_source_length,
                        padding=padding,
                        truncation=True,
                    )
                    section_input_ids.append(sample_input_ids['input_ids'])
                    section_attention_mask.append(sample_input_ids['attention_mask'])
                    
                    sample_targets = tokenizer(
                        target,
                        max_length=max_target_length,
                        padding=padding,
                        truncation=True,
                    )
                    sample_targets = sample_targets['input_ids']
                    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
                        sample_targets[sample_targets == tokenizer.pad_token_id] = -100
                    section_labels.append(sample_targets)
                    
                model_inputs['input_ids'].append(section_input_ids)
                model_inputs['attention_mask'].append(section_attention_mask)
                model_inputs['labels'].append(section_labels)
                    
            return model_inputs       
    
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
      
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )    
            
    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    
    if training_args.task_type == "Normal":
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    elif training_args.task_type == "Segment":
        data_collator = RMTDataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            max_target_length=max_target_length,
        )
    
    # Metric
    metric = evaluate.load('rouge')
    def postprocess_text(preds, labels):
        # sent_detector = nltk.RegexpTokenizer(u'[^　！？。]*[！？。.\n]')
        
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # preds = ["\n".join(sent_detector.tokenize(pred)) for pred in preds]
        # labels = ["\n".join(sent_detector.tokenize(label)) for label in labels]
        
        # rougeLSum expects newline after each sentence
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        
        import functools
        from ja_sentence_segmenter.common.pipeline import make_pipeline
        from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
        from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
        from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation

        split_punc2 = functools.partial(split_punctuation, punctuations=r"．。!?.")
        # concat_tail_no = functools.partial(concatenate_matching, former_matching_rule=r"^(?P<result>.+)(の)$", remove_former_matched=False)
        segmenter = make_pipeline(split_punc2)

        preds = ["\n".join(list(segmenter(pred))) for pred in preds]
        labels = ["\n".join(list(segmenter(label))) for label in labels]
        
        # print(f'{preds=}')
        # print(f'{labels=}')
        return preds, labels
    
    if training_args.task_type == "Normal":
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple): 
                preds = preds[0]
            # Replace -100s used for padding as we can't decode them
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            # print(f'decoded_preds: {decoded_preds}')
            # print(f'decoded_labels: {decoded_labels}')
            
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, 
                                    tokenizer=lambda x: x.split(), use_stemmer=True)
            result = {k: round(v * 100, 4) for k, v in result.items()}
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            return result
        
    elif training_args.task_type == "Segment":
        print(f'{training_args.rouge_type=}')
        if training_args.rouge_type == "Accumulation":
            def compute_metrics(eval_preds):
                
                # format: [batch_size, section, seq_len]
                preds, labels = eval_preds
                
                from nltk.tokenize import word_tokenize
                
                # calculate rouge for each segment
                for index in range(preds.shape[1]):
                    pred = preds[:, index, :]
                    label = labels[:, index, :]
                    
                    pred = np.where(pred != -100, pred, tokenizer.pad_token_id)
                    decoded_pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
                    # print(f'{decoded_pred=}')
                    
                    label = np.where(label != -100, label, tokenizer.pad_token_id)
                    decoded_label = tokenizer.batch_decode(label, skip_special_tokens=True)
                    # print(f'{decoded_label=}')
                    
                    # Some simple post-processing
                    decoded_pred, decoded_label = postprocess_text(decoded_pred, decoded_label)
                    # print(f'{decoded_pred=}')
                    # print(f'{decoded_label=}')
                    
                    result = metric.compute(predictions=decoded_pred, references=decoded_label, use_stemmer=True, tokenizer=word_tokenize)
                    result = {k: round(v * 100, 4) for k, v in result.items()}
                    predicton_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in pred]
                    result["gen_len"] = np.mean(predicton_lens)
                    print(f'-'*50)
                    print(f'result for {index+1} segment:')
                    print(result)
                    print(f'-'*50)
                    print(f'\n')
                    
                # calculate rouge for the whole document
                preds = preds.reshape(-1, preds.shape[-1])
                labels = labels.reshape(-1, labels.shape[-1])
                
                if isinstance(preds, tuple): 
                    preds = preds[0]
                
                # Replace -100s used for padding as we can't decode them
                preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                # Some simple post-processing
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                # print(f'decoded_preds: {decoded_preds}')
                # print(f'decoded_labels: {decoded_labels}')
                
                result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, tokenizer=word_tokenize)
                result = {k: round(v * 100, 4) for k, v in result.items()}
                prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
                result["gen_len"] = np.mean(prediction_lens)
                
                return result
             
        elif training_args.rouge_type == "Final":
            raise NotImplementedError
        
        
    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    
    # generation_config = GenerationConfig(
    #     bos_token_id=0,
    #     decoder_start_token_id=2,        
    #     early_stopping=False,
    #     eos_token_id=2,
    #     forced_bos_token_id=0,
    #     forced_eos_token_id=2,
    #     no_repeat_ngram_size=3,
    #     num_beams=4,
    #     pad_token_id=1,
    #     length_penalty=2.0,
    #     max_length=300,
    #     min_length=200,
    # )

    # training_args.generation_config = generation_config
    
    # training_args.generation_config = AutoConfig.from_pretrained('ku-nlp/bart-base-japanese')
    
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )   
    print(f'{trainer.model.generation_config=}')
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_ds_name, eval_ds in eval_dataset.items():
                dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        
        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                predictions = predictions.reshape(-1, predictions.shape[-1])
                predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))
                    
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
