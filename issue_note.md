# Debug记录

## 语法问题
### Python
#### __init.py__
> https://muyuuuu.github.io/2021/07/11/python-init-file/
```python
├─ main.py
└─ network
       ├─ __init__.py
       ├─ msg
       │    └─ info
       │           └─ send.py(send_msg function)
       └─ parse.py
如果在__init__.py里写 from .msg.info.send import *
则可以直接在main.py里
from network import send_msg
```
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

### 四：自定义模型
> https://huggingface.co/docs/transformers/custom_models
#### 1. 自定义config
**有三件必须要做的事情**
1. 需要从PretrainedConfig类继承: 只有这样才能使用huggingface的通用方法
2. 必须要接收kwargs参数
3. 需要把kwargs参数传递给super类的__init__参数

## 五: Model Architecture
### 一：Auto Regressive & Teacher Forcing
> 引发了exposure bias： [Bridging the Gap between Training and Inference for Neural Machine Translation](https://aclanthology.org/P19-1426/)
### 二：Custom Generate
> https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#utilities-for-generation


## Debug记录
1. decoder_input_ids出现out of index: decoder_input_ids必须以eos开头
2. decoder_input_ids不需要手动生成, 模型会自动从labels转换
3. past_key_values只能用在decoder部分 (去年的版本？？现在是否已经更新了呢)
    > https://github.com/huggingface/transformers/issues/15591
4. 一个模型的max_size取决于max_position_embeddings: 
    > https://github.com/huggingface/transformers/issues/4224
```bash
nohup python run.py >logs/2023_08_26/logs_2023_08_26_08.txt 1>&1 &
```
5. 如何删除已经被git追踪的文件：git rm -r --cached "filename"
6. using prefix-tuning in BART
> https://blog.csdn.net/weixin_42953627/article/details/125586001

7. 无法正确学习, 很可能是因为label的切片策略出了问题。
8. 在generate里不需要手动传入attention_mask
9. 对一个encoder-decoder模型, generate方法会首先调用encoder, 然后会调用model.forward()注意此时encoder已经被pass了