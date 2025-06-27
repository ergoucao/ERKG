import torch
from torch import nn

from STID_arch.mlp import  MultiLayerPerceptron


class STID_DSC(nn.Module):
    """
    Base On The implementation of CIKM 2022 short paper
        "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
    Link: https://arxiv.org/abs/2208.05233
    """

    #     "num_nodes" : 336,
    #     'input_len' : 168,
    #     'input_dim' : 3,
    #     'embed_dim' : 32,
    #     'output_len': 12,
    #     'num_layer' : 3,
    #     "if_node"   : True,
    #     'node_dim'  : 32,
    #     "if_T_i_D"  : True,
    #     "if_D_i_W"  : True,
    #     'temp_dim_tid'  : 32,
    #     'temp_dim_diw'  : 32,

    def __init__(self):
        super().__init__()
        # attributes
        self.num_nodes = 321
        self.node_dim = 32
        self.input_len = 168
        self.input_dim = 3
        self.embed_dim = 32
        self.output_len = 1
        self.num_layer = 6
        self.temp_dim_tid = 32
        self.temp_dim_diw = 32

        self.if_time_in_day = True
        self.if_day_in_week = True
        self.if_spatial = True

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(288, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(7, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_day_in_week) + \
            self.temp_dim_diw*int(self.if_time_in_day)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # align
        self.hidden_align_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=64, kernel_size=(1, 1), bias=True)

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)


    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """

        # prepare data
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(
                t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[(
                d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        # encoding
        hidden = self.encoder(hidden)
        hidden_align = self.hidden_align_layer(hidden)
        # regression
        prediction = self.regression_layer(hidden)

        return prediction, hidden_align

class DeepWise_pointWise_Conv(nn.Module):
    def _init_(self, in_ch, out_ch,kernel_size):
        super(DeepWise_pointWise_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels = in_ch,
            out_channels = out_ch,
            kernel_size = 1,
            groups=1,
        )

    def forward(self, input):
        x=self.depth_conv(input)
        x=self.point_conv(x)
        return x