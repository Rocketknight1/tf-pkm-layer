import numpy as np
import tensorflow as tf
import torch
from tf_memory_layer import HashingMemory as TFHashingMemory
from memory_layer import HashingMemory as TorchHashingMemory

# Initialize and run the TF network
network = TFHashingMemory(input_dim=256, output_dim=256, query_batchnorm=False)
test_input = np.random.random(size=(128, 256))
tf_input = tf.convert_to_tensor(test_input, dtype=tf.float32)
tf_out = network(tf_input)

# Initialize the torch network
torch_layer = TorchHashingMemory(input_dim=256, output_dim=256, query_batchnorm=False)

# Copy the relevant parameters (except batchnorm)
torch_layer.keys[:] = torch.tensor(network.keys.numpy())
torch_layer.values.weight[:] = torch.tensor(network.values.numpy())
torch_layer.query_proj[0].weight[:] = torch.tensor(np.transpose((network.query_proj.get_weights()[0])))
torch_layer.query_proj[0].bias[:] = torch.tensor(network.query_proj.get_weights()[1])

# Run the torch network and get the difference in outputs
torch_input = torch.from_numpy(test_input).to(torch.float32)
torch_out = torch_layer(torch_input)
diff = torch_out.detach().numpy() - tf_out.numpy()

print(f"Mean absolute difference between torch and tf is {np.mean(np.abs(diff))}")