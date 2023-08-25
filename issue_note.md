# Debug记录

## 环境问题
### 一：WANDB
1. 没有权限写入/tmp
* 解决： 
    ```python
    # 在每个项目的根目录下创建wandb/文件夹
    # 在run.py文件内加入以下代码
    import wandb
    import os
    os.environ['WANDB_DIR'] = os.getcwd() + '/wandb/'
    os.environ['WANDB_CACHE_DIR'] = os.getcwd() + '/wandb/.cache/'
    os.environ['WANDB_CONFIG_DIR'] = os.getcwd() + '/wandb/.config/'

    # 此外需要在wandb.init()的时候指定entity参数: entity是组织和机构的名字, 目前默认为kaifan-li, project参数任意
    ```

## Transformer库学习记录
### 一：Trainer
1. trainer类可以`evaluate_during_training=True`, 这样就可以在训练的时候进行评估, 而不需要单独调用trainer.evaluate()。 注意evaluate默认只在eval数据集上测试一次。

### 二：XXPretrainedModel | XXModel | XXForXX
1. XXPretrainedModel: 抽象基类(Abstrct Base Model), 无法被实例化。
    * 继承自PretrainedModel基类(包含一些from_pretrained)
    * 主要负责：
        * 存储模型的config
        * 加载｜下载｜保存模型(包含一些from_pretrained|save_pretrained...)
    * 注意: 虽然写作PretrainedModel, 但是并没有加载模型, 所以无法实例化, 也不包含任何预训练参数。
2. XXModel: 原始的模型XX模型基类
    ```python
    # 以Bart为例
    class BartModel(BartPreTrainedModel):
        _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

        def __init__(self, config: BartConfig):
            super().__init__(config)

            padding_idx, vocab_size = config.pad_token_id, config.vocab_size
            self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

            self.encoder = BartEncoder(config, self.shared)
            self.decoder = BartDecoder(config, self.shared)

            # Initialize weights and apply final processing
            self.post_init()
    ...
    ```
3. XXForXX: 针对下游任务做了适配, 继承自XXPretrainedModel, 一般来说里面包含了XXModel和一个lm_head。重写了prepare_inputs和forward方法
    ```python
    class BartForConditionalGeneration(BartPreTrainedModel):
        base_model_prefix = "model"
        _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
        _keys_to_ignore_on_load_missing = ["final_logits_bias"]

        def __init__(self, config: BartConfig):
            super().__init__(config)
            self.model = BartModel(config)
            self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
            self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

            # Initialize weights and apply final processing
            self.post_init()
        ...
    ```

### 三：模型加载方法
```python
from transformers import BertConfig, BertModel

# from_pretrained -> transformers.modeling_utils
# def from_pretrained(
#     cls,
#     pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
#     *model_args,
#     config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
#     cache_dir: Optional[Union[str, os.PathLike]] = None,
#     ignore_mismatched_sizes: bool = False,
#     force_download: bool = False,
#     local_files_only: bool = False,
#     token: Optional[Union[str, bool]] = None,
#     revision: str = "main",
#     use_safetensors: bool = None,
#     **kwargs,
# ):

# 1. Download model and configuration from huggingface.co and cache.
model = BertModel.from_pretrained("bert-base-uncased")

# 2. Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
model = BertModel.from_pretrained("./test/saved_model/")

# 3. Update configuration during loading.
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
assert model.config.output_attentions == True

# 4. Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)

# 5. Loading from a Flax checkpoint file instead of a PyTorch model (slower)
model = BertModel.from_pretrained("bert-base-uncased", from_flax=True)

# 6. Custom model config
from transformers import BartConfig, BartForConditionalGeneration

# 现有的 BartConfig 对象
existing_config = BartConfig.from_pretrained("facebook/bart-large-cnn")

# 自定义参数作为 kwargs
custom_kwargs = {
    "my_custom_parameter": "abc",
}

# 创建新的配置对象并传递自定义参数
new_config = BartConfig(**existing_config.to_dict(), **custom_kwargs)

# 输出新配置对象的属性
print(new_config.test)
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn', config=new_config)
model, model.config
```

#### .from_pretrained()方法
> 不会覆盖预训练参数！！！
1. 找到正确的基础模型类进行初始化
2. 使用伪随机初始化来初始化该类（通过使用_init_weights您提到的函数）
3. 找到具有预训练权重的文件
4. 在适用的情况下使用预先训练的权重覆盖我们刚刚创建的模型的权重，在初始化参数时，如果模型结构与预训练模型不同，那么只有与预训练模型相同的部分才会被初始化。

#### 自定义config的问题
```python
kwargs = {
    "pre_seq_len": 20,
    "input_size": 512,
    "segment_size": 512,
    "max_n_segments": 3,
    "bptt_depth": 2,
    "prefix_projection" : False,
}

# NOTE: 这里似乎不能在from_pretrained的时候就传入**kwargs
bart_config = BartConfig.from_pretrained('facebook/bart-large-cnn')
custom_config = BartConfig(**bart_config.to_dict(), **kwargs)
# BartConfig支持写入自定义字段
# Additional attributes without default values
# >>>for key, value in kwargs.items():
# >>>   try:
# >>>       setattr(self, key, value)
# >>>   except AttributeError as err:
# >>>       logger.error(f"Can't set {key} with value {value} for {self}")
# >>>       raise err
custom_config
>>> BartConfig {
    ...
    "pre_seq_len": 20
    ...
}
```

```python
# 自定义config类的方法
class CustomBartConfig(BartConfig):
    def __init__(self,
                 pre_seq_len=20,
                 segment_size=512,
                 input_size=,
                 max_n_segments=4,
                 bptt_depth=-1,
                 prefix_projection,
                 hidden_dropout_prob,
                 hidden_size,
                 **kwargs):
        super().__init__(**kwargs)
        self.pre_seq_len = pre_seq_len
        self.segment_size = segment_size
        self.input_size = input_size
        self.max_n_segments = max_n_segments
        self.bptt_depth = bptt_depth
        self.prefix_projection = prefix_projection
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size

custom_config = CustomBartConfig(
    pre_seq_len=
)
```

#### past_key_values()
* 如果要求past_key_values + input_ids 的长度和attention_mask相等就很麻烦了
* 到底是怎么实现的呢
* past_key_values在encoder-decoder模型里同时需要考虑self-attention&cross-attention

    * self.attention: q|k|v 来自同一个序列(encoder/decoder都有self.attention)
    * cross.attention: q来自decoder的输入, k|v来自encoder的最后一层
    ```python
    # `past_key_value[0].shape[2] == key_value_states.shape[1]`
    # is checking that the `sequence_length` of the `past_key_value` is the same as
    # the provided `key_value_states` to support prefix tuning
    if (
        is_cross_attention
        and past_key_value is not None
        and past_key_value[0].shape[2] == key_value_states.shape[1]
    ):
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
        value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    ```

#### decoder_input_ids
> see: https://github.com/huggingface/transformers/issues/7865

> and: https://github.com/facebookresearch/fairseq/issues/1389#issuecomment-565947058

* 一般需要使用labels 而且注意一定要以eos_token开头

## Model Architecture
### 一：Auto Regressive & Teacher Forcing
> 引发了exposure bias： [Bridging the Gap between Training and Inference for Neural Machine Translation](https://aclanthology.org/P19-1426/)


## Debug记录
1. decoder_input_ids出现out of index: decoder_input_ids必须以eos开头
2. decoder_input_ids不需要手动生成, 模型会自动从labels转换