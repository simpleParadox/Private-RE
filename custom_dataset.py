import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding


class TableDataset(Dataset):
    def __init__(self, tokenized_data, labels):
        self.tokenized_data = tokenized_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        class_label = self.labels[idx]
        tokens_info = self.tokenized_data[idx]
        return [tokens_info, class_label]


class SemevalDataset(Dataset):
    def __init__(self, tokenized_data, labels):
        self.tokenized_data = tokenized_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        class_label = self.labels[idx]
        tokens_info = self.tokenized_data[idx]
        return [tokens_info, class_label]