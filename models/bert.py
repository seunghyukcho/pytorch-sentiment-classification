import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification


def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument('--model_name', type=str,
            help='Name of pretrained model')
    group.add_argument('--n_hids', type=int, default=768,
            help='Dimension of hiddens')


class Model(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.classifier = nn.Sequential(nn.Linear(self.encoder.config.hidden_size, args.n_hids, bias=False), nn.ReLU(), nn.Linear(args.n_hids, args.n_classes, bias=False))
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        for param in self.classifier.parameters():
            nn.init.kaiming_uniform_(param.data)

    def forward(self, x, lens):
        x = self.encoder(x)
        x = x[0][:, 1, :]
        x = self.classifier(x)

        return x

