from torch import nn
import torch

def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument('--n_hids', type=int, default=300,
            help='Number of hidden units')
    group.add_argument('--n_layers', type=int, default=1,
            help='Number of layers')
    group.add_argument('--embd_dim', type=int, default=1,
            help='Embedding dimension')
    group.add_argument('--k', type=int, default= 8,
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
        self.rnn = nn.LSTM(self.embd_dim, self.n_hids, self.n_layers, batch_first=True, dropout= 0.5)
        
        # AE structure
        self.Wt = nn.Linear(self.embd_dim, self.k)
        self.softmax = nn.Softmax(dim=1)
        self.T = nn.Parameter(torch.randn((self.k,self.embd_dim), requires_grad=True))
        self.Wa = nn.Parameter(torch.randn((self.embd_dim,self.embd_dim), requires_grad=True))
        self.tanh = nn.Tanh()
        
        # decoder
        self.decoder = nn.Linear(self.n_hids, self.n_classes)


    def forward(self, w, s, lw, ls, is_train= True):
        self.rnn.flatten_parameters()
        # embedding layer
        w, s = self.word_embd(w), self.sentence_embd(s) # (batch, n, self.embd_dim), (batch, m, self.embd_dim)

        # 1) word --> lstm
        h,_ = self.rnn(w) # h: (batch, n, self.n_hids)
        # 2) sentence --> Auto encoder
        w, s = torch.mean(w, dim=1), torch.mean(s, dim=1) # w, s: (batch, self.embd_dim)
        c_s = (w+s)/2.0 # AVERAGE, captures both target information and context information. (batch, self.embd_dim)
        q_t = self.softmax(self.Wt(c_s)) # (batch, self.k)
        t_s = q_t@self.T # (batch, self.k) (self.k, self.embd_dim) = (batch, self.embd_dim)
        # t_s = (self.T.T@q_t.unsqueeze(-1)).squeeze() # (embd_dim, k) (batch, k, 1) = (batch, self.embd_dim)

        # (batch,n,self.embd_dim) (self.embd_dim, self.embd_dim) = (batch, n, self.embd_dim)
        # print(f'h\'s shape: {h.shape}')
        # print(f'Wa\'s shape: {self.Wa.shape}')
        d = h@(self.Wa)
        # print(f'h@self.Wa\'s shape: {d.shape}')
        # print(f't_s.unsqueeze(-1) shape: {t_s.unsqueeze(-1).shape}')
        d = self.tanh((d@(t_s.unsqueeze(-1))))  # (batch, n, self.embd_dim) (batch, self.embd_dim,1) != (batch, n, 1)
        p = self.softmax(d) # (batch, n, 1)

        # obtain sentence representation
        z_s = torch.sum(p*h, dim=1) # (batch, self.embd_dim)
        prob = self.decoder(z_s) # (batch, n_classes)
        prob = self.softmax(prob)

        if is_train:
            return prob, self.T, t_s, c_s
        else:
            return prob
