# All code here stolen from https://github.com/facebookresearch/XLM/blob/master/PKM-layer.ipynb

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_uniform_keys(n_keys, dim, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    keys = rng.uniform(-bound, bound, (n_keys, dim))
    return keys.astype(np.float32)


class HashingMemory(nn.Module):

    def __init__(self, input_dim, output_dim, k_dim=256, heads=4, knn=32, n_keys=512, query_batchnorm=True):

        super().__init__()

        # global parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_dim = k_dim
        self.v_dim = output_dim
        self.n_keys = n_keys
        self.size = self.n_keys ** 2
        self.heads = heads
        self.knn = knn
        assert self.k_dim >= 2 and self.k_dim % 2 == 0

        # dropout
        self.input_dropout = 0.
        self.query_dropout = 0.
        self.value_dropout = 0.

        # initialize keys / values
        self.initialize_keys()
        self.values = nn.EmbeddingBag(self.size, self.v_dim, mode='sum', sparse=False)
        nn.init.normal_(self.values.weight, mean=0, std=self.v_dim ** -0.5)

        # query network
        self.query_proj = nn.Sequential(*filter(None, [
            nn.Linear(self.input_dim, self.heads * self.k_dim, bias=True),
            nn.BatchNorm1d(self.heads * self.k_dim) if query_batchnorm else None
        ]))

        if query_batchnorm:
            print("WARNING: Applying batch normalization to queries improves the performance "
                  "and memory usage. But if you use it, be sure that you use batches of "
                  "sentences with the same size at training time (i.e. without padding). "
                  "Otherwise, the padding token will result in incorrect mean/variance "
                  "estimations in the BatchNorm layer.\n")

    def initialize_keys(self):
        """
        Create two subkey sets per head.
        `self.keys` is of shape (heads, 2, n_keys, k_dim // 2)
        """
        half = self.k_dim // 2
        keys = nn.Parameter(torch.from_numpy(np.array([
            get_uniform_keys(self.n_keys, half, seed=(2 * i + j))
            for i in range(self.heads)
            for j in range(2)
        ])).view(self.heads, 2, self.n_keys, half))
        self.keys = nn.Parameter(keys)

    def _get_indices(self, query, subkeys):
        """
        Generate scores and indices for a specific head.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        bs = query.size(0)
        knn = self.knn
        half = self.k_dim // 2
        n_keys = len(subkeys[0])

        # split query for product quantization
        q1 = query[:, :half]                                          # (bs,half)
        q2 = query[:, half:]                                          # (bs,half)

        # compute indices with associated scores
        scores1 = F.linear(q1, subkeys[0], bias=None)                 # (bs,n_keys)
        scores2 = F.linear(q2, subkeys[1], bias=None)                 # (bs,n_keys)
        scores1, indices1 = scores1.topk(knn, dim=1)                  # (bs,knn)
        scores2, indices2 = scores2.topk(knn, dim=1)                  # (bs,knn)

        # cartesian product on best candidate keys
        all_scores = (
            scores1.view(bs, knn, 1).expand(bs, knn, knn) +
            scores2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)                                                # (bs,knn**2)
        all_indices = (
            indices1.view(bs, knn, 1).expand(bs, knn, knn) * n_keys +
            indices2.view(bs, 1, knn).expand(bs, knn, knn)
        ).view(bs, -1)                                                # (bs,knn**2)

        # select best scores with associated indices
        scores, best_indices = torch.topk(all_scores, k=knn, dim=1)   # (bs,knn)
        indices = all_indices.gather(1, best_indices)                 # (bs,knn)

        assert scores.shape == indices.shape == (bs, knn)
        return scores, indices

    def get_indices(self, query):
        """
        Generate scores and indices.
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        query = query.view(-1, self.heads, self.k_dim)
        bs = len(query)
        outputs = [self._get_indices(query[:, i], self.keys[i]) for i in range(self.heads)]
        s = torch.cat([s.view(bs, 1, self.knn) for s, _ in outputs], 1)  # (bs,heads,knn)
        i = torch.cat([i.view(bs, 1, self.knn) for _, i in outputs], 1)  # (bs,heads,knn)
        return s.view(-1, self.knn), i.view(-1, self.knn)

    def forward(self, input):
        """
        Read from the memory.
        """
        # input dimensions
        assert input.shape[-1] == self.input_dim
        prefix_shape = input.shape[:-1]
        bs = np.prod(prefix_shape)

        # compute query
        input = F.dropout(input, p=self.input_dropout, training=self.training)  # (...,i_dim)
        query = self.query_proj(input.contiguous().view(-1, self.input_dim))    # (bs,heads*k_dim)
        query = query.view(bs * self.heads, self.k_dim)                         # (bs*heads,k_dim)
        query = F.dropout(query, p=self.query_dropout, training=self.training)  # (bs*heads,k_dim)
        assert query.shape == (bs * self.heads, self.k_dim)

        # retrieve indices and scores
        scores, indices = self.get_indices(query)                               # (bs*heads,knn)
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)              # (bs*heads,knn)

        # merge heads / knn (since we sum heads)
        indices = indices.view(bs, self.heads * self.knn)                       # (bs,heads*knn)
        scores = scores.view(bs, self.heads * self.knn)                         # (bs,heads*knn)

        # weighted sum of values
        output = self.values(indices, per_sample_weights=scores)                # (bs,v_dim)
        output = F.dropout(output, p=self.value_dropout, training=self.training)# (bs,v_dim)

        # reshape output
        if len(prefix_shape) >= 2:
            output = output.view(prefix_shape + (self.v_dim,))                  # (...,v_dim)

        return output