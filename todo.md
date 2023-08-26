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