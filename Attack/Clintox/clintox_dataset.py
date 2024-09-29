# Note that the graph tensors have been saved to disk already. Process() function is not
# overridden.

from typing import List, Tuple, Union
import torch
from torch_geometric.data import Dataset
from dotenv import load_dotenv
import os
import os.path as osp


class ClintoxDataset(Dataset):
    def __init__(self, fold_key, root, start, stop):
        self.fold_key = fold_key
        self.root = root
        self.start = start
        self.stop = stop

        super().__init__(root)

    @property
    def processed_file_names(self):
        load_dotenv(".env")

        processed_names = list()
        for n in range(self.start, self.stop+1):
            processed_names.append('data_'+str(n)+'.pt')

        return processed_names

    @property
    def processed_paths(self):
        load_dotenv(".env")

        directory = os.getenv(self.fold_key)

        return [os.path.join(directory, file) for file in os.listdir(directory)[self.start:self.stop+1]]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir,
                                   f'data_{idx+self.start}.pt'))

        return data
