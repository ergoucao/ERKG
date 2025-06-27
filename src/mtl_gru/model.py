import torch
import torch.nn as nn
import torch.optim as optim

# 定义MTL-GRU模型
class MTL_GRU(nn.Module):
    def __init__(self, data_dim, output_size, forecast_steps, gru_units=[256, 128], dropout_rates=[0.2, 0.1]):
        super(MTL_GRU, self).__init__()

        self.forecast_steps = forecast_steps

        # 硬参数共享层的GRU
        self.shared_gru = nn.GRU(data_dim, gru_units[0], batch_first=True)
        self.shared_dropout = nn.Dropout(dropout_rates[0])

        # 各任务的单独GRU层和Dropout
        self.task_gru_layers = nn.ModuleList([
            nn.GRU(gru_units[0], gru_units[1], batch_first=True) for _ in range(output_size)
        ])
        self.task_dropouts = nn.ModuleList([
            nn.Dropout(dropout_rates[1]) for _ in range(output_size)
        ])

        # 每个任务的输出层：每个任务输出 forecast_steps 个时间步的预测
        self.output_layers = nn.ModuleList([
            nn.Linear(gru_units[1], forecast_steps) for _ in range(output_size)
        ])

    def forward(self, x):
        # 共享层的前向传播
        shared_out, _ = self.shared_gru(x)
        shared_out = self.shared_dropout(shared_out)

        outputs = []
        # 为每个任务计算多步输出
        for i in range(len(self.task_gru_layers)):
            task_out, _ = self.task_gru_layers[i](shared_out)
            task_out = self.task_dropouts[i](task_out)
            # 输出多步预测结果，取 GRU 的最后一个时间步作为输出
            task_out = self.output_layers[i](task_out[:, -1, :])  # 输出为 [batch_size, forecast_steps]
            outputs.append(task_out.unsqueeze(1))  # 增加维度以便与其他任务的输出对齐

        # 将各任务输出堆叠为 [batch_size, output_size, forecast_steps]
        return torch.cat(outputs, dim=1).transpose(1,2)