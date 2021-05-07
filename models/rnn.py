from torch import nn


def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument('--n_hids', type=int, default=128,
            help='Number of hidden units')
    group.add_argument('--n_layers', type=int, default=1,
            help='Number of layers')


class Model(nn.Module):
    def __init__(self, args, embd_dim):
        super().__init__()

        self.n_hids = args.n_hids
        self.n_layers = args.n_layers
        self.n_classes = args.n_classes

        self.rnn = nn.RNN(embd_dim, self.n_hids, self.n_layers, batch_first=True)
        self.decoder = nn.Linear(self.n_hids, self.n_classes)

    def forward(self, x):
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        x = self.decoder(x)

        return x

