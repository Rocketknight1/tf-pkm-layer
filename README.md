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

##Usage

### Using PKM in your own projects

```
from tf_memory_layer import HashingMemory

pkm_layer = HashingMemory(input_dim=256, output_dim=256, query_batchnorm=False)
pkm_output = pkm_layer(my_input_tensor)
```

### Testing equivalence
Note that this requires both Tensorflow and PyTorch to be installed in the same environment.

```
$ python test_equivalence.py
Mean absolute difference between torch and tf is 4.676983245133215e-09
```

## Todo

The code currently feels very like translated PyTorch, because it is. The more idiomatic TF/Keras (Kerasic?) way
to do things is not to require explicitly defined input dimensions, and instead to finish constructing layers when the
module is first called and the shapes are known. I might get around to that.

Also, a key issue in the translation is that Tensorflow does not have the `EmbeddingBag` layer that PyTorch does. I
replaced it with `tf.gather()` followed by `tf.einsum()`, though `tf.gather()` followed by a broadcast multiplication
and a `tf.reduce_sum()` is also equivalent and might be easier to understand. This is a very fast layer, so performance
isn't an issue either way, but one variant might have better memory usage, and I haven't tested it 
thoroughly yet. If anyone can find a better low-memory implementation of `EmbeddingBag` in TF, let me know!