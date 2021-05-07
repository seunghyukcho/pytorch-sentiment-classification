from torch import nn


class Model(nn.Module):
    def __init__(self, embd_dim, n_classes, n_hids, n_layers):
        super().__init__()

        self.rnn = nn.RNN(embd_dim, n_hids, n_layers, batch_first=True)
        self.decoder = nn.Linear(n_hids, n_classes)

    def forward(self, x):
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        x = self.decoder(x)

        return x

