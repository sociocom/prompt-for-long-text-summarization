## 2023.08.06
1. 完成prefix encoder
2. 完成base model
3. 数据处理
4. 保存模型 earlystopping
5. RMT如何处理batch的: 重写了forward函数, 将一个长文本切成多个片段, 多个片段包装在一个输入里, 多个输入组成一个batch
6. 我可能不需要写那么复杂的代码, 基于prefix encoder和base model重写一个新的bart/t5 model然后还是调用api就可以