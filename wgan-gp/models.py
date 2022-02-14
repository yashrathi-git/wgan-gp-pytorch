import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim: int, img_size: int, n_channels: int, hidden_dim: int):
        super(Generator, self).__init__()
        assert img_size % 16 == 0, "img_size has to be a multiple of 16"

        start_dim = int(img_size / 16)
        self.start_dim = start_dim
        self.z_dim = z_dim

        self.init_block = nn.Sequential(
            nn.Linear(z_dim, start_dim * start_dim * 3),
            nn.ReLU(),
        )
        self.gen = nn.Sequential(
            self._generator_block(3, hidden_dim * 2),
            self._generator_block(hidden_dim * 2, hidden_dim * 4),
            self._generator_block(hidden_dim * 4, hidden_dim * 2),
            self._generator_block(hidden_dim * 2, n_channels, final_layer=True),
        )

    def _generator_block(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=1,
        padding="same",
        final_layer=False,
    ):
        if not final_layer:
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.Tanh(),
            )

    def forward(self, z):
        out = self.init_block(z)
        out = out.view(z.size(0), -1, self.start_dim, self.start_dim)
        out = self.gen(out)
        return out

    def sample_noise_vector(self, batch_size):
        return torch.randn(batch_size, self.z_dim)


class Critic(nn.Module):
    def __init__(self, img_size: int, n_channels: int, hidden_dim: int):
        super(Critic, self).__init__()
        assert img_size % 16 == 0, "img_size has to be a multiple of 16"

        end_dim = int(img_size / 16)

        self.critic = nn.Sequential(
            self._critic_block(n_channels, hidden_dim),
            self._critic_block(hidden_dim, hidden_dim * 2),
            self._critic_block(hidden_dim * 2, hidden_dim * 4),
            self._critic_block(hidden_dim * 4, hidden_dim * 8),
        )
        # Each of the conv block halves the image size by a factor of 2
        self.final_fc = nn.Sequential(
            nn.Linear(hidden_dim * 8 * end_dim * end_dim, 1),
        )

    def _critic_block(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
    ):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        out = self.critic(x)
        out = out.view(x.size(0), -1)
        out = self.final_fc(out)
        return out
