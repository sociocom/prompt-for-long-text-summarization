from dataclasses import dataclass
from transformers import BartConfig

# ============================================
# =============== BART model =================
# ============================================

# CustomBartConfig is used to add some new parameters to BartConfig
# usage:
# pretrained_config = AutoConfig.from_pretrained('facebook/bart-base')
# custom_config = CustomBartConfig(
#   pre_sqe_len = 10,
#   ...
#   **pretrained_config.to_dict(),
# )
# ...
# TODO: print(cumtom_config) will raise Error
#       but print(custom_config.xxx)| print(custom_config.to_dict()) is ok
# TODO：need to rewrite __str__ method
class RMTBartConfig(BartConfig):
    def __init__(self,
                 pre_seq_len=0,
                 post_seq_len=0,
                 max_section_length=1024,
                 max_source_length=1024,
                 max_n_segments=4,
                 bptt_depth=-1,
                 prefix_projection=False, 
                 propagate_prefix='only',
                 hidden_dropout_prob=0.1,
                 segment_alignment='left',
                 sum_token_size=0,
                 label_max_size=256,
                 sum_loss=True,
                 propagate_prefix_scalar=False,
                 freeze_model=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.pre_seq_len = pre_seq_len
        self.post_seq_len = post_seq_len
        self.max_section_length = max_section_length
        self.max_source_lenght = max_source_length
        self.max_n_segments = max_n_segments
        self.bptt_depth = bptt_depth
        self.prefix_projection = prefix_projection # whether to use reparametrization trick
        self.propagate_prefix = propagate_prefix # whether to propagate the prefix between layers
        self.hidden_dropout_prob = hidden_dropout_prob # dropout for prefix encoder
        self.segment_alignment = segment_alignment # how to segment the input sequence
        self.sum_token_size = sum_token_size # the size of summary tokens
        self.label_max_size = label_max_size # the max size of labels
        self.sum_loss = sum_loss # whether to use summary loss
        self.propagate_prefix = propagate_prefix
        self.propagate_prefix_scalar = propagate_prefix_scalar
        self.freeze_model = freeze_model # whether to freeze the model parameters
        
# Another method to custom BartConfig
# existing_config = BartConfig.from_pretrained("facebook/bart-large-cnn")

# 自定义参数作为 kwargs
# custom_kwargs = {
#     "my_custom_parameter": "custom",
# }

# 创建新的配置对象并传递自定义参数
# new_config = BartConfig(**existing_config.to_dict(), **custom_kwargs)

# 输出新配置对象的属性
# print(new_config.test)
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn', config=new_config)
# model, model.config
