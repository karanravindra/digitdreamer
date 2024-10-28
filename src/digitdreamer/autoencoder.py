from torch import nn

from digitdreamer.modules import Block, Down, Up


def down_block(in_channels, out_channels):
    return [
        Block(in_channels, out_channels),
        Down(out_channels, out_channels),
    ]


def up_block(in_channels, out_channels):
    return [
        Up(in_channels, out_channels),
        Block(out_channels, out_channels, reverse=True),
    ]


class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 1),
            *down_block(4, 8),
            *down_block(8, 16),
            *down_block(16, 32),
            *down_block(32, 32),
            nn.Conv2d(32, 8, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 32, 1),
            *up_block(32, 32),
            *up_block(32, 16),
            *up_block(16, 8),
            *up_block(8, 4),
            nn.Conv2d(4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
