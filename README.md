# prompt-for-long-text-summarization

## NOTE
1. 虽然我们可以直接调用transformers的api, 但是多个segment直接传递的时候, 直觉上还是应该使用relative positional encoding
2. 如果在多个segment直接传递resnet
3. 实际上并不是将输入拼接了进去, 而是使用了past_key_values参数，这样就不需要修改原始模型架构，但是问题是prefix部分并没有query矩阵传进去，这样prefix部分就无法显示的保存后续序列的信息（只能通过反向传递的时候保存）。为什么没有直接拼接的原因是：如果直接拼接了, prefix部分也成了被冻结的参数，很难不改网络结构的前提下，只冻结一部分。
4. past_key_value和past_key_values的定义在bert源码591行
5. 如何确定学习率等超参数呢

## TODO
1. 复现一份简单的代码完成Summarization任务
2. 自己在transformer的库里添加一个past_query参数
3. 如何让prefix部分各层之间相连
4. 该如何考虑长文本的批次划分问题呢