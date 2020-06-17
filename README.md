# Product Key Memory layers for Tensorflow

This repo contains a straightforward reimplementation of the code for 
[Product Key Memory layers](https://arxiv.org/abs/1907.05242) (PKMs) from the Facebook
[XLM repo](https://github.com/facebookresearch/XLM/blob/master/PKM-layer.ipynb), translated from PyTorch to Tensorflow. 
In addition, the original PyTorch code is included (lightly edited to 
replace the `params` object with function arguments, but otherwide identical). Since trusting random single-script repos
on GitHub is usually a bad idea, I've also included a `test_equivalence.py` that will initialize both the original 
PyTorch and my TF version, synchronize weights between them and show that the outputs are equivalent (with some 
inevitable tiny float32 deviations).

When batchnorm is enabled, the outputs are no longer equivalent - I think this is because of some differences in the
specifics of how TF and PyTorch compute batchnorm. If anyone can figure out how to make the two line up, please let
me know! I'm pretty confident that even if they're not exactly equivalent the TF code is still valid.

## Usage

#### Using PKM in your own projects

```
from tf_memory_layer import HashingMemory

pkm_layer = HashingMemory(input_dim=256, output_dim=256, query_batchnorm=False)
pkm_output = pkm_layer(my_input_tensor)
```

#### Testing equivalence
Note that this requires both Tensorflow and PyTorch to be installed in the same environment.

```
$ python test_equivalence.py
Mean absolute difference between torch and tf is 4.676983245133215e-09
```

## Benchmarking versus original PyTorch function

All benchmarks performed on an RTX 2080 Ti, Ubuntu 18.04, float32 precision, CUDA 10.1, TF 2.2 and Torch 1.5.0.

The benchmark consists of a network with 6 product-key memory layers, with `input_dim=768` and `output_dim=768` to match
BERT-base and `query_batchnorm=False`. The input data is of size `(16, 256, 768)` to simulate a normal BERT
input with a batch size of 16 and a sequence length of 256 tokens. The reported time is the time taken for
100 runs through this memory-only network.

In general, the manual implementations to match Torch's `EmbeddingBag` layer were much slower, but Tensorflow's
XLA compiler (accessed by setting the `experimental_compile` argument to `tf.function()` to `True`) seemed to be able to
optimize things back to almost-identical performance.

| framework | compiled     | EmbeddingBag method | Time (100 runs) |
|-----------|--------------|---------------------|-----------------|
| tf        | no           | einsum              | 28.69s           |
| tf        | no           | reduce_sum          | 32.32s           |
| tf        | yes          | reduce_sum          | 26.97s           |
| tf        | yes          | einsum              | 23.17s           |
| tf        | XLA          | einsum              | 6.59s            |
| tf        | XLA          | reduce_sum          | 6.06s            |
| torch     | no           | EmbeddingBag        | 4.94s            |
| torch     | yes (traced) | EmbeddingBag        | 4.91s            |



## Todo

The code currently feels very like translated PyTorch, because it is. The more idiomatic TF/Keras (Kerasic?) way
to do things is not to require explicitly defined input dimensions, and instead to finish constructing layers when the
module is first called and the shapes are known. I might get around to that.

Also, a key issue in the translation is that Tensorflow does not have the `EmbeddingBag` layer that PyTorch does. I
replaced it with `tf.gather()` followed by `tf.einsum()`, though `tf.gather()` followed by a broadcast multiplication
and a `tf.reduce_sum()` is also equivalent and might be easier to understand. This is a very fast layer, so performance
isn't an issue either way, but one variant might have better memory usage, and I haven't tested it 
thoroughly yet. If anyone can find a better low-memory implementation of `EmbeddingBag` in TF, let me know!