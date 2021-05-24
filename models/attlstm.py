from torch import nn


def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument('--n_hids', type=int, default=128,
            help='Number of hidden units')
    group.add_argument('--n_layers', type=int, default=1,
            help='Number of layers')
    group.add_argument('--embd_dim', type=int, default=1,
            help='Embedding dimension')
    group.add_argument('--k', type=int, default= 100,
            help='the number of aspects defined by the user')


class Model(nn.Module):
    def __init__(self, args, vocab_size, sentence_size):
        super().__init__()

        self.n_hids = args.n_hids
        self.n_layers = args.n_layers
        self.n_classes = args.n_classes
        self.embd_dim = args.embd_dim
        self.k = args.k 

        #todo
        # embedding layer
        self.word_embd = nn.Embedding(vocab_size, self.embd_dim, padding_idx=0)
        self.sentence_embd = nn.Embedding(sentence_size, self.embd_dim, padding_idx=0)
        
        # rnn structure
        self.rnn = nn.LSTM(self.embd_dim, self.n_hids, self.n_layers, batch_first=True, dropout= 0.3)
        
        # AE structure
        self.Wt = nn.Linear(self.embd_dim, self.k)
        self.softmax = nn.Softmax(dim=1)
        self.T = nn.Parameter(torch.randn((self.k,self.emb_dim), requires_grad=True))

        self.decoder = nn.Linear(self.n_hids, self.n_classes)

    def forward(self, w, s, lw, ls):
        self.rnn.flatten_parameters()
        # embedding layer
        w, s = self.word_embd(w), self.sentence_embd(s) # (batch, n, self.embd_dim), (batch, m, self.embd_dim)

        # 1) word --> lstm
        h,_ = self.rnn(w) # h: (batch, n, self.n_hids)
        # 2) sentence --> Auto encoder
        w, s = torch.mean(w, dim=1), torch.mean(s, dim=1) # w, s: (batch, self.embd_dim)
        cs = (w+s)/2.0 # AVERAGE, captures both target information and context information.
        qt = self.softmax(self.Wt(cs))
        ts = self.T.T@
        
        
        # x = self.embd(x)
        # x, _ = self.rnn(x)
        # lens = lens.unsqueeze(-1).repeat(1, self.n_hids).unsqueeze(1) - 1
        # x = x.gather(1, lens)
        # x = x.squeeze(1)
        # x = self.decoder(x)

        return x
