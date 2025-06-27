import torch.nn as nn
import torch
import torch.nn.functional as F


class ETSEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                              kernel_size=3, padding=2, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x, ):
        x = self.conv(x.permute(0, 2, 1))[..., :-2]
        return self.dropout(x.transpose(1, 2))


class StatusEmbedding(nn.Module):
    def __init__(self, status_in, out_length,in_length, dropout=0.1):
        super().__init__()
        self.Lin = nn.Linear(status_in, 512)
        self.Conv = nn.Conv2d(in_channels=in_length, out_channels=out_length, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.kaiming_normal_(self.Conv.weight)

    def forward(self, x):
        if (x==None):
            print("x:None")
        x = torch.cat(x, dim=2)
        x = self.Lin(x)#.transpose(dim0=0,dim1=1)
        x = self.dropout(x)
        # x = nn.ReLU()(x)
        x = self.Conv(x.transpose(dim0=0, dim1=1)).transpose(dim0=0,dim1=1)
        return self.dropout(x)

class Feedforward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1, activation='sigmoid'):
        # Implementation of Feedforward model
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x):
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)
