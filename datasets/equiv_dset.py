import torch
from torch.utils.data import DataLoader
import numpy as np



class EquivDataset(torch.utils.data.Dataset):
    def __init__(self, PATH, greyscale = False):
        self.data = np.load(PATH + 'equiv_data.npy',  mmap_mode = 'r+')
        self.lbls = np.load(PATH + 'equiv_lbls.npy',  mmap_mode = 'r+')
        self.greyscale = greyscale

    def __getitem__(self, index):
        if self.greyscale:
            return torch.FloatTensor(self.data[index, 0]).unsqueeze(0), torch.FloatTensor(self.data[index, 1]).unsqueeze(0), torch.FloatTensor((self.lbls[index],))
        else:
            return torch.FloatTensor(self.data[index, 0]), torch.FloatTensor(self.data[index, 1]), torch.FloatTensor((self.lbls[index],))
    def __len__(self):
        return len(self.data)
