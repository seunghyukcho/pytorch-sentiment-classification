import torch
import pandas as pd
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence

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


class PadBatch:
    def __init__(self, pad_token_id, inference=False):
        super(PadBatch, self).__init__()
        self.inference = inference
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        sentences, lens = [], []
        if self.inference:
            for item in batch:
                sentences.append(torch.LongTensor(item))
                lens.append(len(item))
            return pad_sequence(sentences, batch_first=True, padding_value=self.pad_token_id), torch.LongTensor(lens)
        else:
            labels = []
            for item in batch:
                x, y = item
                sentences.append(torch.LongTensor(x))
                lens.append(len(x))
                labels.append(y)
            return pad_sequence(sentences, batch_first=True, padding_value=self.pad_token_id), torch.LongTensor(labels), torch.LongTensor(lens)

