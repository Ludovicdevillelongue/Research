import torch
import torch.nn as nn
import numpy as np


class CustomLSTM(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int
    ):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_i = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_i = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(self.hidden_size))

        self.w_f = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_f = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(self.hidden_size))

        self.w_c = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_c = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(self.hidden_size))

        self.w_o = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.u_o = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(self.hidden_size))

        self.init()

    def init(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(
            self,
            X: torch.Tensor

    ):
        bs, seq_size, _ = X.size()
        hidden_seq = []

        h_t, c_t = (
            torch.zeros(bs, self.hidden_size),
            torch.zeros(bs, self.hidden_size)
        )
        for t in range(seq_size):
            x_t = X[:, t, :]
            i_t = torch.sigmoid(x_t @ self.w_i + h_t @ self.u_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.w_f + h_t @ self.u_f + self.b_f)
            g_t = torch.tanh(x_t @ self.w_c + h_t @ self.u_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.w_o + h_t @ self.u_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class AFTFull(nn.Module):
    def __init__(self, max_seqlen, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)
        self.wbias = nn.Parameter(torch.Tensor(max_seqlen, max_seqlen))
        nn.init.xavier_uniform_(self.wbias)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x).view(B, T, self.hidden_dim)
        V = self.to_v(x).view(B, T, self.hidden_dim)
        temp_wbias = self.wbias[:T, :T].unsqueeze(0)

        Q_sig = torch.sigmoid(Q)
        temp = torch.exp(temp_wbias) @ torch.mul(torch.exp(K), V)
        weighted = temp / (torch.exp(temp_wbias) @ torch.exp(K))
        Yt = torch.mul(Q_sig, weighted)

        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)

        return Yt


class AF_LSTM(nn.Module):
    # todo implémenter un AF-LSTM à l'aide des modules LSTM et AFTFull
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            max_seqlen: int,
            output_size: int

    ):
        super(AF_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_seqlen = max_seqlen
        self.output_size = output_size
        self.fc = nn.Linear(input_size, hidden_size)
        self.lstm = CustomLSTM(self.hidden_size, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.attention = AFTFull(self.max_seqlen, self.hidden_size, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size,self.output_size)
        self.init()


    def init(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def filter(self, x):
        x_ = self.fc(x)
        x_ = self.attention(x_)
        self.norm(x_)
        x_filter = torch.relu(x_)
        return x_filter

    def forward(self,
                X_3D: torch.Tensor):
        filter_ = self.filter(X_3D)
        attention_free_X = self.norm1(self.attention(self.fc(X_3D)))
        prod_ = self.norm2(filter_.mul(attention_free_X))
        return self.lstm(prod_)


