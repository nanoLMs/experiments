
import torch
from config import TrainConfig
from model_moe import NanoMoEModel

# Instantiate the configuration
cfg = TrainConfig()

# Instantiate the model
model = NanoMoEModel(cfg)

# Calculate the number of parameters
num_params = model.num_parameters()

# Calculate model size in MB
# Assuming fp16 as per config (2 bytes per parameter)
model_size_mb = num_params * 2 / (1024 * 1024)

print(f"Total number of parameters: {num_params:,}")
print(f"Estimated model size (fp16): {model_size_mb:.2f} MB")
