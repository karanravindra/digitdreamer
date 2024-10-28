from torch import nn


class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        reverse=False,
    ) -> None:
        super().__init__()

        if reverse:
            self.norm1 = nn.GroupNorm(1, in_channels)
            self.norm2 = nn.GroupNorm(1, out_channels)
            self.conv1 = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
            )
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding)

            self.skip1 = nn.Identity()
            self.skip2 = (
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                )
                if in_channels != out_channels
                else nn.Identity()
            )

        else:
            self.norm1 = nn.GroupNorm(1, out_channels)
            self.norm2 = nn.GroupNorm(1, out_channels)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding)
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
            )

            self.skip1 = (
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                )
                if in_channels != out_channels
                else nn.Identity()
            )
            self.skip2 = nn.Identity()

    def forward(self, x):
        x = nn.functional.silu(self.norm1(self.conv1(x))) + self.skip1(x)
        return nn.functional.silu(self.norm2(self.conv2(x))) + self.skip2(x)
