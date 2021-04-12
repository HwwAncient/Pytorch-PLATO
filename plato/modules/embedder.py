"""
Embedder class.
"""

import torch
import torch.nn as nn


class Embedder(nn.Module):
    """
    Composite embedding layer.
    """

    def __init__(self,
                 hidden_dim,
                 num_token_embeddings,
                 num_pos_embeddings,
                 num_type_embeddings,
                 num_turn_embeddings,
                 padding_idx=None,
                 dropout=0.1,
                 pos_trainable=False):
        super(Embedder, self).__init__()

        self.token_embedding = nn.Embedding(num_token_embeddings, hidden_dim)
        self.pos_embedding = nn.Embedding(num_pos_embeddings, hidden_dim)
        self.pos_embedding.weight.requires_grad = pos_trainable
        self.type_embedding = nn.Embedding(num_type_embeddings, hidden_dim)
        self.turn_embedding = nn.Embedding(num_turn_embeddings, hidden_dim)
        self.dropout_layer = nn.Dropout(p=dropout)

        # follow the default xavier_uniform initializer in paddle version
        # otherwise, there are bugs for dec_probs computation in weight typing setting
        # default norm initializer in nn.Embedding in pytorch, which samples larger values
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        nn.init.xavier_uniform_(self.type_embedding.weight)
        nn.init.xavier_uniform_(self.turn_embedding.weight)
        return

    def forward(self, token_inp, pos_inp, type_inp, turn_inp):
        embed = self.token_embedding(token_inp) + \
            self.pos_embedding(pos_inp) + \
            self.type_embedding(type_inp) + \
            self.turn_embedding(turn_inp)
        embed = self.dropout_layer(embed)
        return embed


def main():
    import numpy as np

    model = Embedder(10, 20, 20, 20, 20)
    token_inp = torch.tensor(np.random.randint(0, 19, [10, 10]).astype("int64"))
    pos_inp = torch.tensor(np.random.randint(0, 19, [10, 10]).astype("int64"))
    type_inp = torch.tensor(np.random.randint(0, 19, [10, 10]).astype("int64"))
    turn_inp = torch.tensor(np.random.randint(0, 19, [10, 10]).astype("int64"))
    out = model(token_inp, pos_inp, type_inp, turn_inp)
    print(out)


if __name__ == "__main__":
    main()
