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
        self.rnn = nn.GRU(self.embd_dim, self.n_hids, self.n_layers, batch_first=True, dropout= 0.2)
        # self.drop = nn.Dropout(p=0.2)
        self.mlp = nn.Sequential(
                # nn.BatchNorm1d(self.k),
                nn.Linear(self.n_hids, self.n_hids),
                nn.Dropout(p=0.5),
                nn.ReLU()
            )

        self.decoder = nn.Sequential(
            # nn.Linear(self.k, self.k),
            # nn.ReLU(),
            nn.Linear(self.n_hids, self.n_classes)
        )



    def forward(self, x, lens):
        # lens: list(a length of every sentence)
        self.rnn.flatten_parameters()
        # print(x.shape)
        zeros = torch.zeros((x.shape[0],self.k),device=x.device,dtype=torch.long)
        x = torch.cat((zeros,x),dim=1)
        x = self.embd(x)

        # DGRU and MLP
        n_tokens = x.shape[1] # including paddings.(# = k-1)
        seq_len = n_tokens-self.k
        h = torch.tensor([],device= x.device)
        # print(f'x\'s shape: {x.shape}') # batch, n_token, embd_dim: ([64, 58, 256])
        for i in range(seq_len):

            # print(x[:,i:i+self.k,:].shape)
            h_t, _ = self.rnn(x[:,i:i+self.k,:]) # output from last layer, the others.
            # h_t: (B, self.k, self.embd_dim)
            # print(f'seq_len: {seq_len}')
            # print(f'self.k: {self.k}')
            # print(f'n_hids: {self.n_hids}')
            # print(f'hidden vector\'s dimension: {h_t.shape}') # batch, k(window size), n_hids
            # print(f'{},{},{}')
            # h_t = self.drop(h_t)
            h_t = h_t[:,-1,:] # (B, self.n_hids)
            h_t = self.mlp(h_t) # (B, self.n_hids)
            h= torch.cat((h,h_t),dim= 1)          
        
        # todo.
        bs= h.shape[0] # h: batch, 1, seq_len * n_hids
        h_reshape = torch.reshape(h,(bs,seq_len,self.n_hids))
#         h_reshape = torch.unsqueeze(h_reshape, -1) # to apply maxpool1d
        # maxpool= nn.MaxPool2d(kernel_size= (seq_len,self.n_hids),stride=(1,1))
        maxpool= nn.MaxPool1d(kernel_size= seq_len)
        pooled= maxpool(torch.transpose(h_reshape,1,2))
        pooled = pooled.squeeze() # batch, self.n_hids
        # print(f'pooled\'s shape = {pooled.shape}')

        # decoder
        # pooled: (batch, N, n_hids)
        output = self.decoder(pooled)
        # print(f'output\'s size: {output.shape}')
        return output
