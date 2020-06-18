import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization


class HashingMemory(tf.Module):
    def __init__(self, input_dim, output_dim, k_dim=256, heads=4, knn=32, n_keys=512, query_batchnorm=True,
                 embeddingbag_method='reduce_sum', seed=12345):

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
        half = self.k_dim // 2
        key_bound = 1 / math.sqrt(half)
        rng = np.random.RandomState(seed)

        # initialize keys / values
        keys = rng.uniform(size=(self.heads, 2, self.n_keys, half), low=-key_bound, high=key_bound)
        self.keys = tf.Variable(initial_value=keys, dtype=tf.float32)
        self.values = tf.Variable(dtype=tf.float32,
                                  initial_value=np.random.normal(loc=0, scale=self.v_dim ** -0.5,
                                                                 size=(self.size, self.v_dim)))

        # query network
        if query_batchnorm:
            self.batchnorm = BatchNormalization(momentum=0.1)  # More similar to Pytorch batchnorm
        else:
            self.batchnorm = None
        self.query_proj = Dense(self.heads * self.k_dim, use_bias=True)

        if query_batchnorm:
            print("WARNING: Applying batch normalization to queries improves the performance "
                  "and memory usage. But if you use it, be sure that you use batches of "
                  "sentences with the same size at training time (i.e. without padding). "
                  "Otherwise, the padding token will result in incorrect mean/variance "
                  "estimations in the BatchNorm layer.\n")
            print("SECOND WARNING: I haven't verified that the TF layer with batchnorm is "
                  "equivalent to the original PyTorch implementation.")

        if embeddingbag_method not in ('reduce_sum', 'einsum'):
            raise NotImplementedError("Unrecognized embeddingbag method!")
        self.embeddingbag_method = embeddingbag_method


    def _get_indices(self, query, subkeys):
        """
        Generate scores and indices for a specific head.
        """
        assert len(query.shape) == 2 and query.shape[1] == self.k_dim
        bs = query.shape[0]
        knn = self.knn
        half = self.k_dim // 2
        n_keys = len(subkeys[0])

        # split query for product quantization
        q1 = query[:, :half]                                          # (bs,half)
        q2 = query[:, half:]                                          # (bs,half)

        # compute indices with associated scores
        scores1 = tf.matmul(q1, subkeys[0], transpose_b=True)                         # (bs,n_keys)
        scores2 = tf.matmul(q2, subkeys[1], transpose_b=True)                         # (bs,n_keys)
        scores1, indices1 = tf.math.top_k(scores1, knn, sorted=True)                  # (bs,knn)
        scores2, indices2 = tf.math.top_k(scores2, knn, sorted=True)                  # (bs,knn)

        # The original code used torch.repeat, we can just use broadcasting
        all_scores = tf.expand_dims(scores1, 2) + tf.expand_dims(scores2, 1)              # (bs, knn, knn)
        all_scores = tf.reshape(all_scores, (bs, -1))                                     # (bs,knn**2)
        all_indices = tf.expand_dims(indices1, 2) * n_keys + tf.expand_dims(indices2, 1)  # (bs, knn, knn)
        all_indices = tf.reshape(all_indices, (bs, -1))                                   # (bs,knn**2)

        # select best scores with associated indices
        scores, best_indices = tf.math.top_k(all_scores, k=knn, sorted=True)   # (bs,knn)
        # Note the use of 'batch_dims=-1' to ensure compatibility with torch.gather
        indices = tf.gather(all_indices, best_indices, axis=1, batch_dims=-1)             # (bs,knn)
        assert scores.shape == indices.shape == (bs, knn)
        return scores, indices

    def get_indices(self, query):
        """
        Generate scores and indices.
        """
        # TODO Get indices for all heads at once for speed?
        assert len(query.shape) == 2 and query.shape[1] == self.k_dim
        query = tf.reshape(query, (-1, self.heads, self.k_dim))
        bs = len(query)
        outputs = [self._get_indices(query[:, i], self.keys[i]) for i in range(self.heads)]
        s = tf.concat([tf.reshape(s, (bs, 1, self.knn)) for s, _ in outputs], 1)  # (bs,heads,knn)
        i = tf.concat([tf.reshape(i, (bs, 1, self.knn)) for _, i in outputs], 1)  # (bs,heads,knn)
        return tf.reshape(s, (-1, self.knn)), tf.reshape(i, (-1, self.knn))

    def __call__(self, input):
        """
        Read from the memory.
        """
        # input dimensions
        assert input.shape[-1] == self.input_dim
        prefix_shape = input.shape[:-1]
        bs = np.prod(prefix_shape)

        # compute query
        query = self.query_proj(tf.reshape(input, (-1, self.input_dim)))    # (bs,heads*k_dim)
        if self.batchnorm is not None:
            query = self.batchnorm(query)
        query = tf.reshape(query, (bs * self.heads, self.k_dim))             # (bs*heads,k_dim)
        assert query.shape == (bs * self.heads, self.k_dim)

        # retrieve indices and scores
        scores, indices = self.get_indices(query)                               # (bs*heads,knn)
        scores = tf.nn.softmax(scores, axis=-1)                          # (bs*heads,knn)

        # merge heads / knn (since we sum heads)
        indices = tf.reshape(indices, (bs, self.heads * self.knn))                       # (bs,heads*knn)
        scores = tf.reshape(scores, (bs, self.heads * self.knn))                         # (bs,heads*knn)

        # weighted sum of values
        values = tf.gather(self.values, indices)
        # The following two calls are equivalent
        if self.embeddingbag_method == 'einsum':
            output = tf.einsum('ijk, ij -> ik', values, scores)  # (bs,v_dim)
        else:
            output = tf.reduce_sum(values * tf.expand_dims(scores, -1), axis=1)  # (bs,v_dim)

        # reshape output
        if len(prefix_shape) >= 2:
            output = tf.reshape(output, (prefix_shape + (self.v_dim,)))                  # (...,v_dim)

        return output