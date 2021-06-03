import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification


def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument('--model_name', type=str,
            help='Name of pretrained model')
    group.add_argument('--n_hids', type=int, default=768,
            help='Dimension of hiddens')


class SICModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_1 = nn.Linear(hidden_size, hidden_size)
        self.W_2 = nn.Linear(hidden_size, hidden_size)
        self.W_3 = nn.Linear(hidden_size, hidden_size)
        self.W_4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, start_indices, end_indices):
        W1_h = self.W_1(hidden_states)  # (bs, length, hidden_size)
        W2_h = self.W_2(hidden_states)
        W3_h = self.W_3(hidden_states)
        W4_h = self.W_4(hidden_states)

        W1_hi_emb = torch.index_select(W1_h, 1, start_indices)  # (bs, span_num, hidden_size)
        W2_hj_emb = torch.index_select(W2_h, 1, end_indices)
        W3_hi_start_emb = torch.index_select(W3_h, 1, start_indices)
        W3_hi_end_emb = torch.index_select(W3_h, 1, end_indices)
        W4_hj_start_emb = torch.index_select(W4_h, 1, start_indices)
        W4_hj_end_emb = torch.index_select(W4_h, 1, end_indices)

        span = W1_hi_emb + W2_hj_emb + (W3_hi_start_emb - W3_hi_end_emb) + torch.mul(W4_hj_start_emb, W4_hj_end_emb)
        h_ij = torch.tanh(span)
        return h_ij


class InterpretationModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.h_t = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h_ij, span_masks):
        o_ij = self.h_t(h_ij).squeeze(-1)  # (ba, span_num)
        o_ij = o_ij - span_masks
        a_ij = self.softmax(o_ij)
        H = (a_ij.unsqueeze(-1) * h_ij).sum(dim=1)  # (bs, hidden_size)

        return H, a_ij


class Model(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()

        self.config = AutoConfig.from_pretrained(args.model_name, output_hidden_states=False)
        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.span_info_collect = SICModel(self.config.hidden_size)
        self.interpretation = InterpretationModel(self.config.hidden_size)
        self.output = nn.Linear(self.config.hidden_size, args.n_classes)
        
    def forward(self, x, lens):
        start_indices = []
        end_indices = []
        max_len = int(torch.max(lens).item())
        for i in range(1, max_len - 1):
            for j in range(i, max_len - 1):
                start_indices.append(i)
                end_indices.append(j)
    
        span_masks = []
        for i in range(x.size(0)):
            sentence = x[i].tolist()
            middle_index = sentence.index(2)
            span_mask = []
            for start_index, end_index in zip(start_indices, end_indices):
                if 1 <= start_index <= lens[i].item() - 2 and 1 <= end_index <= lens[i].item() - 2 and (start_index > middle_index or end_index < middle_index):
                    span_mask.append(0)
                else:
                    span_mask.append(1e6)
            span_masks.append(span_mask)

        start_indices, end_indices, span_masks = torch.cuda.LongTensor(start_indices, device='cuda:1'), torch.cuda.LongTensor(end_indices, device='cuda:1'), torch.cuda.LongTensor(span_masks, device='cuda:1')

        # print(start_indices, end_indices, span_masks)

        attention_mask = (x != 1).long()
        hidden_states, _ = self.encoder(x, attention_mask=attention_mask)  # output.shape = (bs, length, hidden_size)
        h_ij = self.span_info_collect(hidden_states, start_indices, end_indices)
        H, a_ij = self.interpretation(h_ij, span_masks)
        out = self.output(H)

        reg_loss = a_ij.pow(2).sum(dim=1).mean()
        return out, 0.01 * reg_loss

