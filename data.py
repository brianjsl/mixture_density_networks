import numpy as np
import torch
from torch.utils.data import Dataset

N = 300
rng = np.random.default_rng(19)
x = rng.random(N)
t = x + 0.3*np.sin(2*np.pi*x) + (0.2*rng.random(x.size) - 0.1)

class train_dataset(Dataset):
    def __init__(self, x , t):
        self.x = torch.from_numpy(x).type(torch.float32)
        self.t = torch.from_numpy(t).type(torch.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx].unsqueeze(0), self.t[idx].unsqueeze(0)
    