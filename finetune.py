import sys
sys.path = sys.path[1:] + sys.path[:1]
import torch
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name, tokenizer):
        data = pd.read_csv(file_name, header=0, index_col=0)
        
        self.data_size = len(data['Sentence'])
        self.texts = tokenizer(data['Sentence'].tolist(), truncation=True, padding=True)
        self.labels = data['Category'].tolist()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.texts.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item


model_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

for param in model.bert.parameters():
    param.requires_grad = False

train_data = Dataset('data/train.csv', tokenizer)
valid_data = Dataset('data/valid.csv', tokenizer)

batch_size=512
training_args = TrainingArguments(
        output_dir='./bert-base',
        learning_rate=1e-3,
        num_train_epochs=100,
        lr_scheduler_type='constant',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 4,
        logging_dir='./logs/bert-base',
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        )

trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=valid_data)
trainer.train(resume_from_checkpoint=True)

