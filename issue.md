# Debug记录

## 环境问题
### 一：WANDB
1. 没有权限写入/tmp
* 解决： 
    ```
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

### 二：XXXPretrainedModel
1. 例如BertPretrainedModel, 其实并不是已经预训练好的模型。这个基类 BertPretrainedModel 包含了 BERT 模型的通用结构和方法，但它的参数并没有进行初始化。