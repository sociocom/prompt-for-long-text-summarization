# config.segment_size
# config.max_n_segments
# config.segment_alignment
from transformers import HfArgumentParser
from transformers import BartConfig
kwargs = {
    "pre_seq_len": 20,
    "input_size": 512,
    "segment_size": 512,
    "max_n_segments": 3,
    "bptt_depth": 2,
    "prefix_projection" : False,
}

def init_custom_config(custom_kwargs):
    bart_config = BartConfig.from_pretrained('facebook/bart-large-cnn')
    custom_config = BartConfig(**bart_config.to_dict(), **custom_kwargs)
    return custom_config
# from transformers import BartConfig, BartForConditionalGeneration

# # 现有的 BartConfig 对象
# existing_config = BartConfig.from_pretrained("facebook/bart-large-cnn")

# # 自定义参数作为 kwargs
# custom_kwargs = {
#     "my_custom_parameter": "abc",
# }

# # 创建新的配置对象并传递自定义参数
# new_config = BartConfig(**existing_config.to_dict(), **custom_kwargs)

# # 输出新配置对象的属性
# print(new_config.test)
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn', config=new_config)
# model, model.config

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