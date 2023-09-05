## TODO List 
* ~~finish prefix encoder~~
* earlystopping
* ~~segment process~~ / ~~batch process~~
* control bptt_depth 
* relative position encoding
* residual connection
* control forget and memory 
* for now, the model parameters are updated after all segments processed, which means now grad_update for prefix-encoder in a single sequence.
* results in training and inference contains some full padding segments
  and the generated result is unable to be evaluated
    * try to use no-emoty-mask 
* due to the reprocess of the batch data, the data is not on the same device, and need to be moved by hand
* prefix-prop trick for memory cell
* to add a summarization part in decoder structure is a little bit hard
* 试一下不使用input_ids 而是input_embeddings
  * 对于prefix_encoder, 只在开始的时候过第一层embedding, 后续使用第二层embeddings, 这样就可以使用
  * !!!!!!!!!!
* 记录测试开始的时间和结束的时间
* 利用prefix- propagation的思路, 在第一段seg生成一个n-layer的prefix tokens(直接插入到hidden里的), 然后第二段开始是直接取第一段的每一层的hidden状态，放入到模型。
* 测试一下LLaMa2和其他模型的表现, 不是RMT结构也可以
