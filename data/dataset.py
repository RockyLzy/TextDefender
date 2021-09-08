from typing import List, Any
from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self, data: List[Any]):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
