from torch import nn


def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument('--n_hids', type=int, default=128,
            help='Number of hidden units')
    group.add_argument('--n_layers', type=int, default=1,
            help='Number of layers')
    group.add_argument('--embd_dim', type=int, default=1,
            help='Embedding dimension')


class Model(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()
        self.n_hids = args.n_hids
        self.n_layers = args.n_layers
        self.n_classes = args.n_classes
        self.embd_dim = args.embd_dim
        self.embd = nn.Embedding(vocab_size, self.embd_dim, padding_idx=0)
        self.rnn = nn.GRU(self.embd_dim, self.n_hids, self.n_layers, batch_first=True, bidirectional = True)
        self.decoder = nn.Linear(self.n_hids*2, self.n_classes)




    def forward(self, x, lens):
        self.rnn.flatten_parameters()
        x = self.embd(x)
        x, _ = self.rnn(x)
        lens = lens.unsqueeze(-1).repeat(1, self.n_hids*2).unsqueeze(1) - 1
        x = x.gather(1, lens)
        x = x.squeeze(1)
        x = self.decoder(x)

        return x

