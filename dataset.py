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
        word, sentence = self.tokenizer.tokenize(x)
        if self.label:
            return word, sentence, self.data['Category'].iloc[idx]
        else:
            return word, sentence


class PadBatch:
    def __init__(self, inference=False):
        super(PadBatch, self).__init__()
        self.inference = inference

    def __call__(self, batch):
        words, sentences, lens = [], [], []
        if self.inference:
            for word, sentence in batch:
                words.append(torch.LongTensor(word))
                sentences.append(torch.LongTensor(sentence))
                lens_word.append(len(word))
                lens_sentence.append(len(sentence))
            return pad_sequence(words, batch_first=True), pad_sequence(sentences, batch_first=True), torch.LongTensor(lens_word), torch.LongTensor(lens_sentence)
        else:
            labels = []
            for item in batch:
                word, sentence, y = item
                words.append(torch.LongTensor(word))
                sentences.append(torch.LongTensor(sentences))
                lens_word.append(len(word))
                lens_sentence.append(len(sentence))
                labels.append(y)
            return pad_sequence(words, batch_first=True), pad_sequence(sentences, batch_first=True), torch.LongTensor(lens_word), torch.LongTensor(lens_sentence), torch.LongTensor(labels)

