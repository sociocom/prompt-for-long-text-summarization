# config.segment_size
# config.max_n_segments
# config.segment_alignment
from transformers import HfArgumentParser

kwargs = {
    "pre_seq_len": 20,
    "input_size": 512,
    "segment_size": 512,
    "max_n_segments": 3,
    "bptt_depth": 2,
    "prefix_projection" : False,
    "dropout": 0.1,
}

def set_args(args, kwargs):
    raise NotImplementedError

def set_config(config, kwargs):
    if config is not None:
        config.pre_seq_len = kwargs['pre_seq_len']
        config.segment_size = kwargs['segment_size']
        config.input_size = kwargs['input_size']
        config.max_n_segments = kwargs['max_n_segments']
        config.bptt_depth = kwargs['bptt_depth']
        config.prefix_projection = kwargs['prefix_projection']
        config.dropout = kwargs['dropout']
    else:
        raise Exception('config is None!')
    return config