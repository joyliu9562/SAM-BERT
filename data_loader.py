# -*- coding: utf-8 -*-
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch
from utils import get_attention_mask


class MyOwnDataset_Train(Dataset):
    def __init__(self, root, start_idx, end_idx, file_dic, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.dataset = file_dic

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        return [f"graph_{idx}.pt" for idx in range(self.start_idx, self.end_idx)]

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        idx = idx + self.start_idx
        info = self.dataset[str(idx)]
        sentence = info['sentence']
        human = info['human']
        time = info['time']
        location = info['location']
        input_ids, custom_attention_mask, attention_mask = get_attention_mask(sentence, (human, time, location))

        graph = torch.load(f'./data/unbalance_aug/processed/graph_{idx}.pt')
        sub_graph = torch.load(f'./data/unbalance_aug/processed/sub_graph_{idx}.pt')


        return graph, sub_graph, input_ids, custom_attention_mask, attention_mask

class MyOwnDataset_Test(Dataset):
    def __init__(self, root, start_idx, end_idx, file_dic, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.dataset = file_dic

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        return [f"graph_{idx}.pt" for idx in range(self.start_idx, self.end_idx)]

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        idx = idx + self.start_idx
        info = self.dataset[str(idx)]
        sentence = info['sentence']
        human = info['human']
        time = info['time']
        location = info['location']
        input_ids, custom_attention_mask, attention_mask = get_attention_mask(sentence, (human, time, location))

        # tokens = tokenizer.encode_plus(sentence, padding='max_length', max_length=86, truncation=True, return_tensors='pt')
        # input_ids = tokens['input_ids']
        # attention_mask = tokens['attention_mask']
        graph = torch.load(f'./data/unbalance_aug/processed/graph_{idx}.pt')
        sub_graph = torch.load(f'./data/unbalance_aug/processed/sub_graph_{idx}.pt')

        return graph, sub_graph, input_ids, custom_attention_mask, attention_mask
