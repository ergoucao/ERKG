import torch
from torch import nn

from .mlp import MultiLayerPerceptron
from .mlp import MultiLayerPerceptron_One_Fc


class STID(nn.Module):
    """
    The implementation of CIKM 2022 short paper
        "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
    Link: https://arxiv.org/abs/2208.05233
    """

    def __init__(self, dim=32, num_layer=6, node=737, input_dim=6, input_len=168, args=None, status=[3, 3, 3, 3, 3,
                                                                                                     3, 3, 3, 3, 3,
                                                                                                     3, 3, 3, 3, 3,
                                                                                                     3, 3, 3, 3, 3,
                                                                                                     3, 3, 3]):
        super().__init__()
        # attributes
        self.num_nodes = node
        self.node_dim = dim
        self.input_len = input_len
        self.input_dim = input_dim
        self.embed_dim = dim
        self.output_len = 1
        self.num_layer = num_layer
        self.temp_dim_tid = dim
        self.temp_dim_diw = dim
        self.status_num = sum(status)

        # 'if_load':True, 'if_temperature':False, 'if_humidity':False,'if_time_in_day':True,'if_day_in_week':True,'if_location':True
        self.if_time_in_day = False  # args.feature_dict['if_time_in_day']
        self.if_day_in_week = False  # args.feature_dict['if_day_in_week']
        self.if_spatial = False  # args.feature_dict['if_location']

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(24, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(7, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim + self.node_dim * \
                          int(self.if_spatial) + self.temp_dim_tid * int(self.if_day_in_week) + \
                          self.temp_dim_diw * int(self.if_time_in_day)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # align
        self.hidden_align_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=64, kernel_size=(1, 1), bias=True)

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=64, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

        # classifier
        self.classifier = nn.Linear(in_features=64 * self.num_nodes, out_features=self.status_num)

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
            t_i_d_data = input_data[..., -2]
            time_in_day_emb = self.time_in_day_emb[(
                    t_i_d_data[:, -1, :] * 23).type(torch.LongTensor)]  # 时间划分为24份
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = input_data[..., -1]
            day_in_week_emb = self.day_in_week_emb[(  # 时间划分为7份
                    d_i_w_data[:, -1, :] * 6).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)  # Conv2d(1176, 32, kernel_size=(1, 1), stride=(1, 1))

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
        hidden_align = self.hidden_align_layer(hidden)  # 对齐便于与teacher的hint层计算loss。
        # regression
        prediction = self.regression_layer(hidden_align)  # Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))

        # # classfier
        status = self.classifier(torch.flatten(hidden_align, start_dim=1))

        return prediction, status


class STID_Student(nn.Module):
    """
    The implementation of CIKM 2022 short paper
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

    def __init__(self, args, node=737, input_dim=6):
        super().__init__()
        # attributes
        self.num_nodes = node
        self.node_dim = 32
        self.input_len = 168
        self.input_dim = input_dim
        self.embed_dim = 32
        self.output_len = 1
        self.num_layer = 3
        self.temp_dim_tid = 32
        self.temp_dim_diw = 32

        self.if_time_in_day = True
        self.if_day_in_week = True
        self.if_spatial = True
        self.hidden_align_dim = args.rep_dim
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
        self.hidden_dim = self.embed_dim + self.node_dim * \
                          int(self.if_spatial) + self.temp_dim_tid * int(self.if_day_in_week) + \
                          self.temp_dim_diw * int(self.if_time_in_day)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron_One_Fc(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

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
            t_i_d_data = history_data[..., -2]
            time_in_day_emb = self.time_in_day_emb[(
                    t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., -1]
            day_in_week_emb = self.day_in_week_emb[(
                d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)  # Conv2d(1176, 32, kernel_size=(1, 1), stride=(1, 1))

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
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb,
                           dim=1)  # hidden=torch.Size([32, 128, 737, 1]) [32,32,737,1] [32,32,737,1]

        # encoding
        hidden = self.encoder(hidden)
        hidden_align = self.hidden_align_layer(hidden)  # 对齐便于与teacher的hint 层计算loss。
        # regression
        hidden_align = torch.relu(hidden_align)
        prediction = self.regression_layer(hidden_align)  # Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))

        return prediction, hidden_align


# class Weight_Net(nn.Module):
#     def __init__(self,node_dim,hint_one_node_dim,models_num):
#         super().__init__()
#         self.teacher_data_batch_shape_size=4
#         self.w=nn.Parameter(torch.randn(32,models_num,1,hint_one_node_dim*node_dim),requires_grad=True)
#
#     def print_w(self):
#         return (self.w)
#
#     def forward(self, x): # [32, models_num, 64, 737]
#         if (len(x.shape)<self.teacher_data_batch_shape_size):
#             x=torch.unsqueeze(x,dim=1)
#         x = torch.flatten(x, start_dim=2) # [32, models_num, 64*737]
#         x = torch.unsqueeze(x,dim=-1) # [32, models_num, 64*737, 1]
#         x = torch.matmul(self.w,x) # [32, models_num, 1, 1]
#         return torch.squeeze(torch.squeeze(x,dim=-1),dim=-1)

class STID_Attention(nn.Module):
    def __init__(self, args, num_nodes=737, input_dim=6, num_layer=3):
        super().__init__()
        # attributes
        self.num_nodes = num_nodes
        self.node_dim = 128
        self.input_len = 168
        self.input_dim = input_dim
        self.embed_dim = 128
        self.output_len = 1
        self.num_layer = num_layer
        self.temp_dim_tid = 128
        self.temp_dim_diw = 128
        self.state_dim = 128
        self.state_num = 50  # 对于 residential 数据集中美国总共有50个州

        self.if_time_in_day = args.feature_dict['if_time_in_day']
        self.if_day_in_week = args.feature_dict['if_day_in_week']
        self.if_spatial = True
        self.if_state = False
        self.if_load = args.feature_dict['if_load']
        self.if_temperature = args.feature_dict['if_temperature']
        self.if_humidity = args.feature_dict['if_humidity']
        self.hidden_align_dim = args.rep_dim
        self.weight = None
        self.is_attention = True
        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Conv2d(
                in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
            # self.time_in_day_emb = nn.Parameter(
            #     torch.empty(24, self.temp_dim_tid))
            # nn.init.kaiming_uniform(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Conv2d(
                in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
            # self.day_in_week_emb = nn.Parameter(
            #     torch.empty(7, self.temp_dim_diw))
            # nn.init.kaiming_uniform(self.day_in_week_emb)

        # state embeddings
        self.state_emb_layer = nn.Parameter(torch.empty(self.state_num, self.state_dim))
        nn.init.xavier_uniform_(self.state_emb_layer)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        self.load_emb_layer = nn.Conv2d(in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1),
                                        bias=True)
        self.temperature_emb_layer = nn.Conv2d(in_channels=self.input_len, out_channels=self.embed_dim,
                                               kernel_size=(1, 1), bias=True)
        self.humidity_emb_layer = nn.Conv2d(in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1),
                                            bias=True)

        # encoding
        self.hidden_dim = self.embed_dim
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron_One_Fc(self.hidden_dim, self.hidden_dim, args.mlp_drop_rate) for _ in
              range(self.num_layer)])

        # align
        self.hidden_align_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=64, kernel_size=(1, 1), bias=True)

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=64, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def feature_ensemble(self, all_feature, node_emb, is_attention):
        all_feature = all_feature.transpose(1, 2)  # 32,737,128,6
        node_emb = node_emb.transpose(1, 2).transpose(2, 3)
        weight = torch.matmul(node_emb, all_feature)  # 32.737,1,6
        weight = nn.Softmax(dim=-1)(weight)
        if is_attention:
            ensemble_feature = torch.sum(torch.mul(weight, all_feature), dim=-1)
        else:
            ensemble_feature = torch.mean(all_feature, dim=-1)
        return ensemble_feature, weight

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
            # t_i_d_data = history_data[..., -2]
            # time_in_day_emb = self.time_in_day_emb[(
            #     t_i_d_data[:, -1, :] * 24).type(torch.LongTensor)]
            time_in_day_emb = self.time_in_day_emb(input_data[..., [-2]])
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            # d_i_w_data = history_data[..., -1]
            # day_in_week_emb = self.day_in_week_emb[(
            #     d_i_w_data[:, -1, :]).type(torch.LongTensor)]
            day_in_week_emb = self.day_in_week_emb(input_data[..., [-1]])
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, dim = input_data.shape
        all_feature = []
        if self.if_load:
            load_emb = self.load_emb_layer(input_data[..., [0]])
            all_feature.append(load_emb.squeeze())
        if self.if_temperature:
            temperature_emb = self.temperature_emb_layer((input_data[..., 1]).unsqueeze(-1))
            all_feature.append(temperature_emb.squeeze())
        if self.if_humidity:
            humidity_emb = self.humidity_emb_layer((input_data[..., 2]).unsqueeze(-1))
            all_feature.append(humidity_emb.squeeze())

        if self.if_state == True:
            state_emb = self.state_emb_layer[(input_data[:, -1, :, 3]).type(torch.LongTensor)]
            all_feature(state_emb.squeeze())
            # all_feature = torch.stack([load_emb.squeeze(),
            #                            temperature_emb.squeeze(),
            #                            state_emb.transpose(1, 2),
            #                            humidity_emb.squeeze(),
            #                            time_in_day_emb.transpose(1, 2),
            #                            day_in_week_emb.transpose(1, 2)],
            #                           dim=-1
            #                           )  # 32 128 737 6
        # else:
        # all_feature = torch.stack([load_emb.squeeze(),
        #                            temperature_emb.squeeze(),
        #                            humidity_emb.squeeze(),
        #                            time_in_day_emb.transpose(1, 2),
        #                            day_in_week_emb.transpose(1, 2)],
        #                           dim=-1
        #                           )  # 32 128 737 6
        if self.if_time_in_day:
            # all_feature.append(time_in_day_emb.transpose(1, 2))
            all_feature.append(time_in_day_emb.squeeze())
        if self.if_day_in_week:
            # all_feature.append(day_in_week_emb.transpose(1, 2))
            all_feature.append(day_in_week_emb.squeeze())
        all_feature = torch.stack(all_feature, dim=-1)
        node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(
            -1)  # torch.Size([32, 128, 737, 1])
        hidden, self.weight = self.feature_ensemble(all_feature, node_emb,
                                                    self.is_attention)  # torch.Size([32, 737, 128])  torch.Size([32, 737, 1, 6])
        hidden = torch.unsqueeze(torch.transpose(hidden, 1, 2), dim=-1)
        # input_data = input_data.transpose(1, 2).contiguous()
        # input_data = input_data.view(
        #     batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        # time_series_emb = self.time_series_emb_layer(input_data)  # Conv2d(1176, 32, kernel_size=(1, 1), stride=(1, 1))

        # temporal embeddings
        # tem_emb = []
        # if time_in_day_emb is not None:
        #     tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        # if day_in_week_emb is not None:
        #     tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        # hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        # add by attention weight
        # hidden =

        # encoding
        hidden = self.encoder(hidden)
        hidden_align = self.hidden_align_layer(hidden)  # 对齐便于与teacher的hint 层计算loss。
        hidden_align = torch.relu(hidden_align)
        # regression
        prediction = self.regression_layer(hidden_align)  # Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
        return prediction, hidden_align


class STID_Attention_simple(nn.Module):
    def __init__(self, args, num_nodes=737, input_dim=6, num_layer=3):
        super().__init__()
        # attributes
        dim = 64
        self.num_nodes = num_nodes
        self.node_dim = dim
        self.input_len = 168
        self.input_dim = input_dim
        self.embed_dim = dim
        self.output_len = 1
        self.num_layer = num_layer
        self.temp_dim_tid = dim
        self.temp_dim_diw = dim
        self.state_dim = dim
        self.state_num = 50  # 对于 residential 数据集中美国总共有50个州

        self.if_time_in_day = args.feature_dict['if_time_in_day']
        self.if_day_in_week = args.feature_dict['if_day_in_week']
        self.if_spatial = True
        self.if_state = False
        self.if_load = args.feature_dict['if_load']
        self.if_temperature = args.feature_dict['if_temperature']
        self.if_humidity = args.feature_dict['if_humidity']
        self.hidden_align_dim = args.rep_dim
        self.weight = None
        self.is_attention = True
        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Conv2d(
                in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
            # self.time_in_day_emb = nn.Parameter(
            #     torch.empty(24, self.temp_dim_tid))
            # nn.init.kaiming_uniform(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Conv2d(
                in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
            # self.day_in_week_emb = nn.Parameter(
            #     torch.empty(7, self.temp_dim_diw))
            # nn.init.kaiming_uniform(self.day_in_week_emb)

        # state embeddings
        self.state_emb_layer = nn.Parameter(torch.empty(self.state_num, self.state_dim))
        nn.init.xavier_uniform_(self.state_emb_layer)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        self.load_emb_layer = nn.Conv2d(in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1),
                                        bias=True)
        self.temperature_emb_layer = nn.Conv2d(in_channels=self.input_len, out_channels=self.embed_dim,
                                               kernel_size=(1, 1), bias=True)
        self.humidity_emb_layer = nn.Conv2d(in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1),
                                            bias=True)

        # # BN层
        # self.feature_bn=nn.BatchNorm1d(self.num_nodes)

        # encoding
        self.hidden_dim = self.embed_dim
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron_One_Fc(self.hidden_dim, self.hidden_dim, args.mlp_drop_rate) for _ in
              range(self.num_layer)])

        # align
        self.hidden_align_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=64, kernel_size=(1, 1), bias=True)

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def freeze(self, layer):
        for param in layer.parameters():
            param.requires_grad = False
        # for child in layer.children():
        #     for param in child.parameters():
        #         param.requires_grad = False

    def unfreeze(self, layer):
        for param in layer.parameters():
            param.requires_grad = True

    def feature_ensemble(self, all_feature, node_emb, is_attention):
        all_feature = all_feature.transpose(1, 2)  # 32,737,128,6
        node_emb = node_emb.transpose(1, 2).transpose(2, 3)
        weight = torch.matmul(node_emb, all_feature)  # 32.737,1,6
        weight = nn.Softmax(dim=-1)(weight)
        if is_attention:
            ensemble_feature = torch.sum(torch.mul(weight, all_feature), dim=-1)
        else:
            ensemble_feature = torch.mean(all_feature, dim=-1)
        # ensemble_feature = self.feature_bn(ensemble_feature)
        # ensemble_feature=self.feature_bn(torch.transpose(ensemble_feature,dim0=2,dim1=1))
        # ensemble_feature=torch.transpose(ensemble_feature,dim0=2,dim1=1)
        return ensemble_feature, weight

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
            # t_i_d_data = history_data[..., -2]
            # time_in_day_emb = self.time_in_day_emb[(
            #     t_i_d_data[:, -1, :] * 24).type(torch.LongTensor)]
            time_in_day_emb = self.time_in_day_emb(input_data[..., [-2]])
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            # d_i_w_data = history_data[..., -1]
            # day_in_week_emb = self.day_in_week_emb[(
            #     d_i_w_data[:, -1, :]).type(torch.LongTensor)]
            day_in_week_emb = self.day_in_week_emb(input_data[..., [-1]])
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, dim = input_data.shape
        all_feature = []
        if self.if_load:
            load_emb = self.load_emb_layer(input_data[..., [0]])
            all_feature.append(load_emb.squeeze())
        if self.if_temperature:
            temperature_emb = self.temperature_emb_layer((input_data[..., 1]).unsqueeze(-1))
            all_feature.append(temperature_emb.squeeze())
        if self.if_humidity:
            humidity_emb = self.humidity_emb_layer((input_data[..., 2]).unsqueeze(-1))
            all_feature.append(humidity_emb.squeeze())

        if self.if_state == True:
            state_emb = self.state_emb_layer[(input_data[:, -1, :, 3]).type(torch.LongTensor)]
            all_feature(state_emb.squeeze())
            # all_feature = torch.stack([load_emb.squeeze(),
            #                            temperature_emb.squeeze(),
            #                            state_emb.transpose(1, 2),
            #                            humidity_emb.squeeze(),
            #                            time_in_day_emb.transpose(1, 2),
            #                            day_in_week_emb.transpose(1, 2)],
            #                           dim=-1
            #                           )  # 32 128 737 6
        # else:
        # all_feature = torch.stack([load_emb.squeeze(),
        #                            temperature_emb.squeeze(),
        #                            humidity_emb.squeeze(),
        #                            time_in_day_emb.transpose(1, 2),
        #                            day_in_week_emb.transpose(1, 2)],
        #                           dim=-1
        #                           )  # 32 128 737 6
        if self.if_time_in_day:
            # all_feature.append(time_in_day_emb.transpose(1, 2))
            all_feature.append(time_in_day_emb.squeeze())
        if self.if_day_in_week:
            # all_feature.append(day_in_week_emb.transpose(1, 2))
            all_feature.append(day_in_week_emb.squeeze())
        all_feature = torch.stack(all_feature, dim=-1)
        node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(
            -1)  # torch.Size([32, 128, 737, 1])
        hidden, self.weight = self.feature_ensemble(all_feature, node_emb,
                                                    self.is_attention)  # torch.Size([32, 737, 128])  torch.Size([32, 737, 1, 6])
        hidden = torch.unsqueeze(torch.transpose(hidden, 1, 2), dim=-1)

        # input_data = input_data.transpose(1, 2).contiguous()
        # input_data = input_data.view(
        #     batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        # time_series_emb = self.time_series_emb_layer(input_data)  # Conv2d(1176, 32, kernel_size=(1, 1), stride=(1, 1))

        # temporal embeddings
        # tem_emb = []
        # if time_in_day_emb is not None:
        #     tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        # if day_in_week_emb is not None:
        #     tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # concate all embeddings
        # hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        # add by attention weight
        # hidden =

        # encoding
        hidden = self.encoder(hidden)
        hidden_align = self.hidden_align_layer(hidden)  # 对齐便于与teacher的hint 层计算loss。
        hidden_align = torch.relu(hidden_align)
        # regression
        prediction = self.regression_layer(hidden)  # Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
        return prediction, hidden_align


class DeepWise_pointWise_Conv(nn.Module):
    def _init_(self, in_ch, out_ch, kernel_size):
        super(DeepWise_pointWise_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            groups=1,
        )

    def forward(self, input):
        x = self.depth_conv(input)
        x = self.point_conv(x)
        return x
