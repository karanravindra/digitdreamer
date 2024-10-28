from torch import Tensor, nn


class Down(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, scale_factor: int = 2
    ) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels // (scale_factor**2), 1)
        self.px_unshuffle = nn.PixelUnshuffle(scale_factor)

    def forward(self, x: Tensor) -> Tensor:
        return self.px_unshuffle(self.reduce(x))


class Up(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, scale_factor: int = 2
    ) -> None:
        super().__init__()
        self.expand = nn.Conv2d(in_channels, out_channels * (scale_factor**2), 1)
        self.px_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: Tensor) -> Tensor:
        return self.px_shuffle(self.expand(x))
