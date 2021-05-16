import sys
sys.path = sys.path[1:] + sys.path[:1]
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name, tokenizer):
        data = pd.read_csv(file_name, header=0, index_col=0)
        
        self.data_size = len(data['Sentence'])
        self.texts = tokenizer(data['Sentence'].tolist(), truncation=True, padding=True)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.texts.items()}
        return item


tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
test_data = Dataset('data/eval_final_open.csv', tokenizer)

model = AutoModelForSequenceClassification.from_pretrained('./bert-basev2/checkpoint-380', num_labels=5)

trainer = Trainer(model=model)
preds = trainer.predict(test_data).predictions

preds = np.array(preds)
preds = np.argmax(preds, axis=1)

with open('submission.csv', 'w') as f:
    f.write('Id,Category\n')
    for i in range(len(preds)):
        f.write(str(i) + ',' + str(int(preds[i])) + '\n')

