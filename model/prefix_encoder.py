# Copy from prefix-propagation
# https://aclanthology.org/2023.acl-short.120.pdf
import torch
import torch.nn as nn

class PrefixEncoder(nn.Module):
    r"""
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """
    
    def __init__(self, config, propagate_prefix: bool = False):
        """
        propagate_prefix: bool
            Whether to propagate the prefix between layers
            If False: follow the original prefix-tuning/p-tuning v2
            If True: follow the prefix-propagation
        """
        super().__init__()
        # whether to use reparametrization trick: Use a two-layer MLP to encode the prefix
        # Better to not use it in NLU tasks(follow by p-tuning v2 paper)
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            self.embedding = nn.Embedding(config.pre_seq_len, config.hidden_size)
            # directly concat the prefix prompt with the input sequence
            if propagate_prefix:
                # Use a two-layer MLP to encode the prefix
                self.trans = nn.Sequential(
                    nn.Linear(config.hidden_size, config.prefix_hidden_size),
                    nn.Tanh(),
                    nn.Linear(
                        config.prefix_hidden_size,
                        # just one matrix(concat the prefix prompt with the input sequence)
                        config.num_hidden_layers * config.hidden_size,
                    ),
                )
            # use past_key_values as a trick to insert in the K/V of the attention
            else:
                # Use a two-layer MLP to encode the prefix
                self.trans = nn.Sequential(
                    nn.Linear(config.hidden_size, config.prefix_hidden_size),
                    nn.Tanh(),
                    nn.Linear(
                        config.prefix_hidden_size,
                        # distributed to K/V
                        config.num_hidden_layers * 2 * config.hidden_size,
                    ),
                )
        elif not propagate_prefix:
            self.embedding = nn.Embedding(
                config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size
            )
            # Useless as author said
            # self.trainable_embedding = None
            # if config.add_pre_seq_len:
            #     self.trainable_embedding = nn.Embedding(
            #         config.add_pre_seq_len,
            #         config.num_hidden_layers * 2 * config.hidden_size,
            #     )
            #     self.embedding.requires_grad = False
        elif propagate_prefix:
            self.embedding = nn.Embedding(
                config.pre_seq_len, config.num_hidden_layers * config.hidden_size * (2 if config.propagate_prefix_scalar else 1)
            )        
            # self.trainable_embedding = None
            
    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
            
        return past_key_values