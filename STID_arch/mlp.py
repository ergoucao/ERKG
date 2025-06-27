import torch
from torch import nn


class DeepWise_pointWise_Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=1,
            bias=bias
        )

    def forward(self, input):
        x = self.depth_conv(input)
        x = self.point_conv(x)
        return x


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        # self.fc1 = DeepWise_pointWise_Conv(input_dim,  hidden_dim, 1, True)
        # self.fc2 = DeepWise_pointWise_Conv(hidden_dim, hidden_dim, 1, True)
        self.fc1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        # hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = self.drop(self.act(self.fc1(input_data)))  # 改造MLP2
        hidden = hidden + input_data  # residual
        return hidden


class MultiLayerPerceptron_One_Fc(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim, p=0.15) -> None:
        super().__init__()
        # self.fc1 = DeepWise_pointWise_Conv(input_dim,  hidden_dim, 1, True)
        # self.fc2 = DeepWise_pointWise_Conv(hidden_dim, hidden_dim, 1, True)
        self.fc1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=p)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        # hidden = self.drop(self.act(self.fc1(input_data))) # 改造MLP
        hidden = hidden + input_data  # residual
        return hidden

# class MultiLayerPerceptron_linear(nn.Module):
#     """Multi-Layer Perceptron with residual links."""
#
#     def __init__(self, input_dim, hidden_dim, p=0.15) -> None:
#         super().__init__()
#         # self.fc1 = DeepWise_pointWise_Conv(input_dim,  hidden_dim, 1, True)
#         # self.fc2 = DeepWise_pointWise_Conv(hidden_dim, hidden_dim, 1, True)
#         self.fc1 = nn.Linear(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1,1), bias=True)
#         self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1,1), bias=True)
#         self.act = nn.ReLU()
#         self.drop = nn.Dropout(p=p)
#
#     def forward(self, input_data: torch.Tensor) -> torch.Tensor:
#         """Feed forward of MLP.
#
#         Args:
#             input_data (torch.Tensor): input data with shape [B, D, N]
#
#         Returns:
#             torch.Tensor: latent repr
#         """
#
#         hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
#         # hidden = self.drop(self.act(self.fc1(input_data))) # 改造MLP
#         hidden = hidden + input_data                           # residual
#         return hidden
