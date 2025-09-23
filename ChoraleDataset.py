import torch
import numpy as np
import pickle
from torch.utils.data import Dataset

class ChoraleDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(np.expand_dims(self.data[idx]['input'], axis=0)).float()
        output_tensors = [torch.from_numpy(np.expand_dims(roll, axis=0)).float() for roll in self.data[idx]['output']]
        return input_tensor, output_tensors[0], output_tensors[1], output_tensors[2]