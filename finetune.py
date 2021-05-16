import sys
sys.path = sys.path[1:] + sys.path[:1]
import torch
import pandas as pd
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name, tokenizer):
        data = pd.read_csv(file_name, header=0, index_col=0)
        
        self.texts = tokenizer(data['Sentence'].tolist(), truncation=True, padding=True)
        self.labels = data['Category'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.texts.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item


tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
train_data = Dataset('data/train.csv', tokenizer)
valid_data = Dataset('data/valid.csv', tokenizer)
batch_size=64
training_args = TrainingArguments(
        output_dir='./finetunev2',
        num_train_epochs=50,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir='./logs/bert',
        weight_decay=0.1,
        evaluation_strategy="steps",
        logging_steps=100,
        save_steps=100,
        )

model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=5)

trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=valid_data)
trainer.train()

