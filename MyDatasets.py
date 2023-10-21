# define dataset class

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, sentence1: list, sentence2: list, labels: list):
        super(Dataset, self).__init__()
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.labels = labels

    def __getitem__(self, index):
        dict = {'sentence1': self.sentence1[index], 'sentence2': self.sentence2[index], 'label': self.labels[index], 'id': index}
        return dict 

    def __len__(self):
        return len(self.labels)
    
    def get_list_sentence1(self):
        return self.sentence1.tolist()
    
    def get_list_sentence2(self):
        return self.sentence2.tolist()
    
    def get_list_labels(self):
        return self.labels.tolist()