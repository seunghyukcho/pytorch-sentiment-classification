import torch
import math
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
    group.add_argument('--glove_embd', type=str,
            help='directory to embdding matrix')
    group.add_argument('--dropout', type=float,
            help='dropout prob to positional encoding')
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class Model(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()

        self.n_hids = args.n_hids
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.n_classes = args.n_classes
        self.embd_dim = args.embd_dim
        self.dropout = args.dropout
#         self.embd = nn.Embedding(vocab_size, self.embd_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(self.n_hids, self.dropout)
        t = torch.load(args.glove_embd)
        self.embd = nn.Embedding.from_pretrained(t, freeze=True, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embd_dim, nhead=self.n_heads, dim_feedforward=self.n_hids, batch_first= True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layers)
        self.decoder = nn.Linear(self.embd_dim, self.n_classes)
        
    def forward(self, x, lens):
        x = self.embd(x) * math.sqrt(self.embd_dim)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = x.squeeze(dim=1)
        lens = lens.unsqueeze(dim=1)
        x = x / lens
        x = self.decoder(x)

        return x

