# prompt-for-long-text-summarization

## NOTE
1. 虽然我们可以直接调用transformers的api, 但是多个segment直接传递的时候, 直觉上还是应该使用relative positional encoding
2. 如果在多个segment直接传递resnet
3. 实际上并不是将输入拼接了进去, 而是使用了past_key_values参数，这样就不需要修改原始模型架构，但是问题是prefix部分并没有query矩阵传进去，这样prefix部分就无法显示的保存后续序列的信息（只能通过反向传递的时候保存）。为什么没有直接拼接的原因是：如果直接拼接了, prefix部分也成了被冻结的参数，很难不改网络结构的前提下，只冻结一部分。
4. past_key_value和past_key_values的定义在bert源码591行
5. 如何确定学习率等超参数呢
6. 总结一下BART更类似于传统的编码器解码器模型，即编码器和解码器之间有一个cell在传递信息。
   而T5更类似于transformer

## TODO
1. ~~复现一份简单的代码完成Summarization任务~~
2. 自己在transformer的库里添加一个past_query参数
3. 如何让prefix部分各层之间相连 (是否可以通过添加MLP呢)
4. 该如何考虑长文本的批次划分问题呢


## Note in 07-24
1. 关于划分摘要的事情, yada建议rule-based的方法手动为每一段做切分
2. 由于tobyoki的文本非常特殊, 可以sentence by sentence去切割
3. 或者可以找到time-relation的数据集, 根据时间做切分
4. Lis提到可以类似于LSTM的方法, 关注一下遗忘比例和记忆比例
5. 在实验部分, yada认为可以用base模型来试一下RMT的效果
6. yada说summarize部分是自然语言文本, 他觉得可以当作是长期记忆，而prompt部分当作短期记忆
7. 初始的summarize部分是用[SEP]分割, 也就说第一段摘要为空
**anyway** : talk is cheap, show me your code
           : 需要反复大量尝试各种方法

### TODO: next week
1. 读今天ACL上一篇相关idea
2. 读一下RetNet和RWKV
3. 实际上在segment之间也可以传播梯度，只在prefix部分传导就可以了，prefix部分的参数量很小，不会带来很大的内存开销
4. 我可能需要加一个字典来记录超参数, 用以多轮查询最优的超参数



### TODO: 07-26 
1. 读一下lora代码
2. 跑一遍lora代码

3. 了解一下hfargumentparser logging类

## NOTE: 07-26
1. fine-tuning灾难性遗忘问题（记忆的重要性）
2. 为什么bart模型在推理的时候反而内存消耗比训练的时候大：
   * 可能原因：对于t5和bart模型，针对生成类任务，在训练阶段是使用teacher forcing方法而不是beam search
3. 我可能可以试着先去尝试一下别的长文本任务


### 
1. 读一下RNN和LSTM的代码


## NOTE: 08-01
1. 需要解决一下rel postion的问题：或许可以参照longformer

## NOTE：08-03
1. Pytorch中很多代码之所以加unsqueeze是为了让数组进行广播操作
```
import torch

# 定义两个形状不完全匹配的张量
a = torch.tensor([[1, 2, 3]])
b = torch.tensor([10])

# 逐元素相加，b会自动广播成与a相同形状
c = a + b

print("a shape:", a.shape)  # 输出: a shape: torch.Size([1, 3])
print("b shape:", b.shape)  # 输出: b shape: torch.Size([1])
print("c:", c)  # 输出: c: tensor([[11, 12, 13]])
```