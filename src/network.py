import torch
import torch.nn as nn
import multiprocessing

from src.etsformer.model import Transform
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class Normalization_Linear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs):
        super(Normalization_Linear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]


class Classifier(BaseNetwork):

    def __init__(self, pred_length, status_num, input_length):
        super().__init__()
        self.precition_length = pred_length
        self.status_num = status_num
        self.feature_dim = (int)(1 * input_length / 2 ** 7)
        self.emb = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2),

            # nn.Dropout(p=0.5),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2)
        )
        self.regression_layer_prediction_length = nn.Sequential(
            # nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
            # nn.Dropout(p=0.5),
            nn.Conv1d(in_channels=512, out_channels=self.precition_length, kernel_size=1, stride=1)
        )
        self.regression_layer_staus = nn.Sequential(
            # nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=1, stride=1),
            # nn.Dropout(p=0.5),
            nn.Conv1d(in_channels=self.feature_dim, out_channels=status_num, kernel_size=1, stride=1)
        )

    def forward(self, input: torch.Tensor):
        out = self.emb(input)
        out = self.regression_layer_prediction_length(out)
        out = torch.transpose(out, dim0=1, dim1=2)
        return torch.transpose(self.regression_layer_staus(out), dim0=1, dim1=2)


class RegressionLayer(BaseNetwork):

    def __init__(self, pred_length, status_num, input_length):
        super().__init__()
        self.precition_length = pred_length
        self.status_num = status_num
        self.feature_dim = (int)(1 * input_length / 2 ** 7)
        self.emb = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm1d(256),
            # nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(512),
            # nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2)
        )
        self.regression_layer_prediction_length = nn.Conv1d(in_channels=512, out_channels=self.precition_length,
                                                            kernel_size=1, stride=1)

        self.regression_layer_staus = nn.Conv1d(in_channels=self.feature_dim, out_channels=1, kernel_size=1, stride=1)

    def forward(self, input: torch.Tensor):
        out = self.emb(input)
        out = self.regression_layer_prediction_length(out)
        out = torch.transpose(out, dim0=1, dim1=2)
        return torch.transpose(self.regression_layer_staus(out), dim0=1, dim1=2)


class StatusPredictor(BaseNetwork):
    def __init__(self, init_weights=True, status=[3, 3, 3, 3, 3,
                                                  3, 3, 3, 3, 3,
                                                  3, 3, 3, 3, 3,
                                                  3, 3, 3, 3, 3,
                                                  3, 3, 3], length=336, pred_length=72):
        super(StatusPredictor, self).__init__()
        self.in_dim = len(status)
        self.feature_dim = (int)(1 * length / 2 ** 7)
        self.status = list(status)
        self.out_status = list(status)
        self.status_num = sum(status)
        self.pred_length = pred_length
        

        self.ShareConv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_dim, out_channels=512, kernel_size=3, stride=2, padding=1),
            # 64 23  336 -> 64 512
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2),
        )

        # ablation study 
        # self.in_dim=1
        # self.status[0]=self.status_num


        self.classifiers = nn.ModuleList()
        self.regression = nn.ModuleList()
        for i in range(self.in_dim):
            self.classifiers.append(Classifier(pred_length, self.status[i], input_length=length))
        for i in range(self.in_dim):
            self.regression.append(RegressionLayer(pred_length, self.status[i], input_length=length))

        self.Fusion = nn.Sequential(
            nn.Linear(in_features=self.status_num, out_features=self.status_num),
        )

        # self.StatusEncoder=nn.ModuleList()

        # for idx, status_num in enumerate(status):
        #     self.StatusEncoder.add_module(
        #         'encoder{}'.format(idx),
        #         nn.Linear(in_features=512, out_features=status_num),
        #         nn.ReLU(True),
        #         nn.Dropout(0.5),
        #         )
        # self.Fusion=nn.Sequential(
        #     nn.Linear(in_features=self.in_dim,out_features=self.in_dim)
        # )
        self.transform = Transform(sigma=0.2)
        if init_weights:
            self.init_weights()

        self.lag_linear = nn.Linear(in_features=len(self.status), out_features=self.status_num)

    def forward(self, x, history_status):
        # with torch.no_grad():
        #     if self.training:
        #         x = self.transform.transform(x)
        # print("transforme ! ! ! ")
        lag = x[:, [-1], :]
        if (len(x.shape) == 4):
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        # print("statusPredictor device ******"+str(self.ShareConv.device)+str(x.deivce))
        out = self.ShareConv(x.transpose(dim0=2, dim1=1))
        # out_status = self.ShareConv(history_status[:, [-1], :].expand(x.shape).transpose(dim0=2, dim1=1))
        hidden = out
        logits = []
        regression_out = []

        for i in range(self.in_dim):
            # logits.append(self.classifiers[i](out)+self.classifiers[i](out_status))
            logits.append(self.classifiers[i](out))
            regression_out.append(self.regression[i](out))
        regression_out = torch.cat(regression_out, dim=2)
        regression_out = regression_out  # + lag
        out = self.Fusion(torch.cat(logits, dim=2))
        # x=self.encoder(x) # 64x23x336  ->  64x512x2
        # hidden=x
        # out=self.classifier(x)
        # status_logit 23 32 3
        # 32 38
        # out = self.Fusion(torch.cat(status_logit,dim=1))
        
        # ablation study 
        out += self.lag_linear(history_status[:, [-1], :]).expand(out.shape)
        out = torch.split(out, self.out_status, dim=2)
        hs = history_status[:, [-1], :].expand(history_status.shape[0],self.pred_length ,history_status.shape[2]).type(torch.int64)
        out = list(out)
        for i in range(len(self.status)):
            element = out[i].gather(2,torch.max(torch.abs(out[i]), dim=2)[1].unsqueeze(dim=-1))
        # #     # element = out[i].gather(2, hs[:, :, [i]])
            out[i] = out[i].scatter(2, hs[:, :, [i]], torch.abs(element)) # out[i].scatter(2, hs[:, :, [i]], torch.abs(element * 10))
        # torch.nn.CrossEntropyLoss()()
        # out = torch.sigmoid(out)
        return out, hidden, regression_out


class Regressor(BaseNetwork):
    class STID_Attention(nn.Module):
        def __init__(self, args, num_nodes=737, input_dim=6, num_layer=3):
            super().__init__()
            # attributes
            self.num_nodes = num_nodes
            self.forecast_num = 1
            self.node_dim = 64
            self.input_len = 168
            self.input_dim = input_dim
            self.embed_dim = 64
            self.output_len = 1
            self.num_layer = num_layer
            self.temp_dim_tid = 64
            self.temp_dim_diw = 64

            self.if_time_in_day = args.feature_dict['if_time_in_day']
            self.if_day_in_week = args.feature_dict['if_day_in_week']
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

            nn.init.xavier_uniform_(self.state_emb_layer)

            # embedding layer
            self.time_series_emb_layer = nn.Conv2d(
                in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
            self.load_emb_layer = nn.Conv2d(in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1),
                                            bias=True)

            # encoding
            self.hidden_dim = self.embed_dim
            self.encoder = nn.Sequential(
                *[MultiLayerPerceptron_One_Fc(self.hidden_dim, self.hidden_dim, args.mlp_drop_rate) for _ in
                  range(self.num_layer)])

            # regression
            self.regression_layer = nn.Sequential(
                nn.Linear(in_features=self.node_dim * 4, out_features=512),
                nn.ReLU(True),
                nn.Dropout(0.5),

                nn.Linear(in_features=512, out_features=self.forecast_num),
                nn.ReLU(True),
                nn.Dropout(0.5),
            )

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
                time_in_day_emb = self.time_in_day_emb(input_data[..., [-2]])
            else:
                time_in_day_emb = None
            if self.if_day_in_week:
                day_in_week_emb = self.day_in_week_emb(input_data[..., [-1]])
            else:
                day_in_week_emb = None

            # time series embedding
            batch_size, _, num_nodes, dim = input_data.shape
            all_feature = []
            load_emb = self.load_emb_layer(input_data[..., [0]])
            all_feature.append(load_emb.squeeze())

            if self.if_time_in_day:
                all_feature.append(time_in_day_emb.squeeze())
            if self.if_day_in_week:
                all_feature.append(day_in_week_emb.squeeze())
            all_feature = torch.stack(all_feature, dim=-1)
            node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(
                -1)  # torch.Size([32, 128, 737, 1])
            all_feature.append(load_emb)
            hidden = torch.concatenate(all_feature, dim=1)  # torch.Size([32, 737, 128])  torch.Size([32, 737, 1, 6])
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


if __name__ == '__main__':
    input = torch.randn((32, 23, 168, 1))
    status_predictor = StatusPredictor()
    print(status_predictor(input).shape[0])
