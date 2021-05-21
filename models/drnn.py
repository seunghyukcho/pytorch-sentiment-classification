from torch import nn
import torch

def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument('--n_hids', type=int, default=128,
            help='Number of hidden units')
    group.add_argument('--n_layers', type=int, default=1,
            help='Number of layers')
    group.add_argument('--embd_dim', type=int, default=1,
            help='Embedding dimension')
    group.add_argument('--k', type=int, default=15,
            help='window size')
    
class Model(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()

        self.n_hids = args.n_hids
        self.n_layers = args.n_layers
        self.n_classes = args.n_classes
        self.embd_dim = args.embd_dim
        self.k = args.k 

        self.embd = nn.Embedding(vocab_size, self.embd_dim, padding_idx=0)
        self.rnn = nn.GRU(self.embd_dim, self.n_hids, self.n_layers, batch_first=True, dropout= 0.1)
        self.drop = nn.Dropout(p=0.05)
        self.mlp = nn.Sequential(
                nn.BatchNorm1d(self.k),
                nn.Linear(self.n_hids, self.n_hids),
                nn.ReLU()
            )
#         self.maxpool = nn.AdaptiveMaxPool1d(self.n_hids)

        self.decoder = nn.Sequential(
            nn.Linear(self.k, self.n_hids),
            nn.ReLU(),
            nn.Linear(self.n_hids, self.n_classes)
        )

    def forward(self, x, lens):
        # lens: list(a length of every sentence)
        self.rnn.flatten_parameters()
        x = self.embd(x)
        
        # DGRU and MLP
        n_tokens = x.shape[1] # including paddings.(# = k-1)
        seq_len = n_tokens-self.k
        h = torch.tensor([],device= x.device)
        # print(f'x\'s shape: {x.shape}') # batch, n_token, embd_dim: ([64, 58, 256])
        for i in range(seq_len):

            # print(x[:,i:i+self.k,:].shape)
            h_t, _ = self.rnn(x[:,i:i+self.k,:]) # output from last layer, the others.
            # print(f'seq_len: {seq_len}')
            # print(f'self.k: {self.k}')
            # print(f'n_hids: {self.n_hids}')
            # print(f'hidden vector\'s dimension: {h_t.shape}') # batch, k(window size), n_hids
            # print(f'{},{},{}')
            h_t = self.drop(h_t)
            h_t = self.mlp(h_t)
            h= torch.cat((h,h_t),dim= 1)
            
        # Max Pool
        # todo.
        bs= h.shape[0]
        h_reshape = torch.reshape(h,(bs,-1,seq_len,self.n_hids))
#         h_reshape = torch.unsqueeze(h_reshape, -1) # to apply maxpool1d
        maxpool= nn.MaxPool2d(kernel_size= (seq_len,1),stride=(1,1))
        pooled= maxpool(h_reshape)
        pooled = torch.reshape(pooled, (bs,-1,self.n_hids))
        # print(f'pooled size: {pooled.shape}')
        maxpool1d = nn.MaxPool1d(self.n_hids)
        pooled = maxpool1d(pooled).squeeze()
        # print(f'pooled size: {pooled.shape}')
        # lens = lens.unsqueeze(-1).repeat(1, self.n_hids).unsqueeze(1) - 1
        # print('lens', lens.shape)
        # h = h.gather(1, lens)
        # print('after gather ', h.shape)
        # h = h.squeeze(1)
        # print('after squeeze ',h.shape)

        # decoder
        # pooled: (batch, N, n_hids)
        output = self.decoder(pooled)
        # print(f'output\'s size: {output.shape}')
        return output
