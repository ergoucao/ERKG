import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from torch import Tensor


class DampingLayer(nn.Module):

    def __init__(self, pred_len, nhead, dropout=0.1, output_attention=False):
        super().__init__()
        self.pred_len = pred_len
        self.nhead = nhead
        self.output_attention = output_attention
        self._damping_factor = nn.Parameter(torch.randn(1, nhead))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = repeat(x, 'b 1 d -> b t d', t=self.pred_len)
        b, t, d = x.shape

        powers = torch.arange(self.pred_len, device=self._damping_factor.device) + 1
        powers = powers.view(self.pred_len, 1)
        damping_factors = self.damping_factor ** powers
        damping_factors = damping_factors.cumsum(dim=0)
        x = x.view(b, t, self.nhead, -1)
        x = self.dropout(x) * damping_factors.unsqueeze(-1)
        x = x.view(b, t, d)
        if self.output_attention:
            return x, damping_factors
        return x, None

    @property
    def damping_factor(self):
        return torch.sigmoid(self._damping_factor)


class DecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, c_out, pred_len, dropout=0.1, output_attention=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.c_out = c_out
        self.pred_len = pred_len
        self.output_attention = output_attention

        self.growth_damping = DampingLayer(pred_len, nhead, dropout=dropout, output_attention=output_attention)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, growth, season):
        growth_horizon, growth_damping = self.growth_damping(growth[:, -1:])
        growth_horizon = self.dropout1(growth_horizon)

        seasonal_horizon = season[:, -self.pred_len:]

        if self.output_attention:
            return growth_horizon, seasonal_horizon, growth_damping
        return growth_horizon, seasonal_horizon, None


class Status_classifier(nn.Module):

    def __init__(self, d_model, status_num, level_num, pred_len) -> None:
        super().__init__()
        self.level_regress = nn.Linear(1, pred_len)
        self.regress_layer = nn.Linear(2 * d_model + level_num, status_num)

    def forward(self,growth_repr : Tensor ,season_repr : Tensor ,level : Tensor):
        l = self.level_regress(torch.transpose(level, dim0=1, dim1=2)).transpose(dim0=1,dim1=2)
        out = self.regress_layer(torch.cat((growth_repr, season_repr, l),dim=2))
        return out


class Decoder(nn.Module):

    def __init__(self, layers, statusLayers, status=[3, 3, 3, 3, 3,
                                                     3, 3, 3, 3, 3,
                                                     3, 3, 3, 3, 3,
                                                     3, 3, 3, 3, 3,
                                                     3, 3, 3]):
        super().__init__()
        self.d_model = layers[0].d_model
        self.c_out = layers[0].c_out
        self.pred_len = layers[0].pred_len
        self.nhead = layers[0].nhead
        self.status_num = sum(status)
        self.status = status
        self.layers = nn.ModuleList(layers)
        self.pred = nn.Linear(self.d_model, self.c_out)
        self.statusLayer = nn.ModuleList(statusLayers)
        self.fuse_layer = nn.ModuleList(
            nn.Conv1d(in_channels=512 * 2, out_channels=512, kernel_size=1)
            for _ in range(len(self.layers))
        )
        self.status_classifier =  Status_classifier(self.d_model, self.status_num, len(self.status), self.pred_len )# nn.Sequential(
        #     # nn.Linear(2*self.d_model+ 23, 2*self.d_model+ 23),
        #     # nn.BatchNorm1d(2*self.d_model+ 23),
        #     # nn.ReLU(),
        #     nn.Linear(2 * self.d_model + 23, self.status_num))

    def forward(self, growths, seasons, level, status_growths_emb, status_seasons_emb):
        growth_repr = []
        season_repr = []
        growth_dampings = []

        for idx, layer in enumerate(self.layers):
            growth_horizon, season_horizon, growth_damping = layer(growths[idx], seasons[idx])
            if status_seasons_emb != None and status_seasons_emb != None:
                status_growth_horizon, statsu_season_horizon, status_growth_damping = self.statusLayer[idx](
                    status_growths_emb, status_seasons_emb)

                growth_horizon = self.fuse_layer[idx](
                    torch.cat([status_growth_horizon, growth_horizon], dim=2).transpose(dim0=1, dim1=2)).transpose(dim0=1,
                                                                                                                   dim1=2)
                season_horizon = self.fuse_layer[idx](
                    torch.cat([statsu_season_horizon, season_horizon], dim=2).transpose(dim0=1, dim1=2)).transpose(dim0=1,
                                                                                                                   dim1=2)
                if status_growth_damping != None:
                    growth_damping = self.fuse_layer[idx](
                        torch.cat([status_growth_damping, growth_damping], dim=2).transpose(dim0=1, dim1=2)).transpose(dim0=1,
                                                                                                                       dim1=2)
            growth_repr.append(growth_horizon)
            season_repr.append(season_horizon)
            growth_dampings.append(growth_damping)
        growth_repr = sum(growth_repr)
        season_repr = sum(season_repr)
        return self.pred(growth_repr), self.pred(season_repr), growth_dampings, self.status_classifier(
            growth_repr, season_repr, level)

if __name__=='__main__':
    sc=Status_classifier(512,23,1,96)
    growth_repr = torch.randn((128,96,512))
    season_repr = torch.randn((128, 96, 512))
    level = torch.randn((128,1,23))
    print("shape{}".format(sc(growth_repr, season_repr, level).shape[0]))