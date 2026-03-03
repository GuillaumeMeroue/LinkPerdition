import os
import random
import numpy as np
import torch
import wandb

# Used to verify if the generated numbers are the same on different machines

# Initialisation wandb
wandb.init(project="random_test", config={"server": os.uname().nodename}, name=f"random_test_{os.uname().nodename}")

# Fix a seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Generate random numbers
rand_py = [random.random() for _ in range(20)]
rand_np = np.random.rand(20).tolist()
rand_torch_gpu = torch.rand(20, device=device).tolist()
rand_torch_cpu = torch.rand(20).tolist()

# Log the generated numbers sur wandb
print("Random python:", rand_py)
print("Random numpy :", rand_np)
print("Random torch GPU:", rand_torch_gpu)
print("Random torch CPU:", rand_torch_cpu)

wandb.finish()
