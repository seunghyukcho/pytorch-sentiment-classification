import pandas as pd
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, file_name, tokenizer, label=False):
        super().__init__()

        self.data = pd.read_csv(file_name, header=0, index_col=0)
        self.label = label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data['Sentence'].iloc[idx]
        x = self.tokenizer.tokenize(x)
        if self.label:
            return x, self.data['Category'].iloc[idx]
        else:
            return x

