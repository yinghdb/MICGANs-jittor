import jittor as jt
import jittor.transform as transform
from jittor.dataset.mnist import MNIST
from jittor.dataset import Dataset
import numpy as np

import os
import random

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        img, label = self.dataset[index]

        return img, label, index

    def __len__(self):
        return len(self.dataset)

def get_dataset(data_dir, size=64):
                
    transforms = transform.Compose([
        transform.Resize(size),
        transform.Gray(),
        transform.ImageNormalize(mean=[0.5],std=[0.5]),
    ])

    dataset = MNIST(
        data_root=data_dir, 
        train=True,
        download=True,
        transform=transforms
    )

    dataset = IndexedDataset(dataset)  # provide data with indexes 

    return dataset