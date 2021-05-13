import torch
from torch import nn


def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument('--n_hids', type=int, default=128,
            help='Number of hidden units')
    group.add_argument('--n_layers', type=int, default=1,
            help='Number of layers')
    group.add_argument('--n_heads', type=int, default=12,
            help='Number of heads in transformer')
    group.add_argument('--embd_dim', type=int, default=1,
            help='Embedding dimension')


class Model(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()

        self.n_hids = args.n_hids
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.n_classes = args.n_classes
        self.embd_dim = args.embd_dim

        self.embd = nn.Embedding(vocab_size, self.embd_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embd_dim, nhead=self.n_heads, dim_feedforward=self.n_hids)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layers)
        self.decoder = nn.Linear(self.embd_dim, self.n_classes)

    def forward(self, x, lens):
        x = self.embd(x)
        x = self.encoder(x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = x.squeeze(dim=1)
        lens = lens.unsqueeze(dim=1)
        x = x / lens
        x = self.decoder(x)

        return x

