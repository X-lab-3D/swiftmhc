import torch
import torch.nn as nn

# Define the batch size, sequence length, and feature dimensions
batch_size = 2
seq_length = 4
feature_dim = 6  # Updated to be divisible by nhead
nhead = 2

# Create dummy input and mask tensors
input_tensor = torch.randn(batch_size, seq_length, feature_dim)
mask_tensor = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])  # Example mask data per batch

# Define the TransformerEncoderLayer
encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, batch_first=True)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

# Generate the source mask
mask = torch.zeros((batch_size * nhead, seq_length, seq_length))
for i in range(seq_length):
    for j in range(seq_length):
        if mask_tensor[:, i].sum() > 0 and mask_tensor[:, j].sum() > 0:
            mask[:, i, j] = float('-inf')

print("input", input_tensor.shape)
print("mask", mask.shape)

# Apply the source mask to the input tensor
output = transformer_encoder(input_tensor, mask=mask)

# Print the output
print(output)
