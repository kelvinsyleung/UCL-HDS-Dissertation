import torch
import torch.nn as nn

class Conv_block(nn.Module):
    """
    A convolutional block consisting of two convolutional layers with batch normalization and ReLU activation.
    
    Parameters
    ----------
        in_channels: int
            The number of input channels.
        out_channels: int
            The number of output channels.
        kernel_size: int
            The size of the kernel.
        stride: int
            The stride of the kernel.
        padding: str
            The padding to use.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding="same"):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class Downsample_block(nn.Module):
    """
    A downsampling block consisting of a convolutional block followed by a max pooling layer and dropout.

    Parameters
    ----------
        in_channels: int
            The number of input channels.
        out_channels: int
            The number of output channels.
        kernel_size: int
            The size of the kernel.
        stride: int
            The stride of the kernel.
        padding: str
            The padding to use.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding="same"):
        super(Downsample_block, self).__init__()
        self.conv_block = Conv_block(in_channels, out_channels, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        x = self.dropout(x)
        return x, skip

class Upsample_block(nn.Module):
    """
    An upsampling block consisting of a transposed convolutional layer followed by a convolutional block.

    Parameters
    ----------
        in_channels: int
            The number of input channels.
        out_channels: int
            The number of output channels.
        kernel_size: int
            The size of the kernel.
        stride: int
            The stride of the kernel.
        padding: str
            The padding to use.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding="same"):
        super(Upsample_block, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.3)
        self.conv_block = Conv_block(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        x = self.dropout(x)
        x = self.conv_block(x)
        return x

class UNet(nn.Module):
    """
    A UNet model consisting of a downsampling path, bottleneck, and upsampling path.

    Parameters
    ----------
        in_channels: int
            The number of input channels. i.e. the number of bands. RGB = 3, BW = 1.
        out_channels: int
            The number of output channels. i.e. the number of classes.
        features: list
            A list of the number of features in each layer.
    """
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        # Downsample 64 -> 128 -> 256 -> 512
        self.down1 = Downsample_block(in_channels, features[0])
        self.down2 = Downsample_block(features[0], features[1])
        self.down3 = Downsample_block(features[1], features[2])
        self.down4 = Downsample_block(features[2], features[3])
        # Bottleneck 512 -> 1024
        self.bottleneck = Conv_block(features[3], features[3]*2, kernel_size=3, stride=1, padding="same")
        # Upsample 512 -> 256 -> 128 -> 64
        self.up1 = Upsample_block(features[3]*2, features[3])
        self.up2 = Upsample_block(features[3], features[2])
        self.up3 = Upsample_block(features[2], features[1])
        self.up4 = Upsample_block(features[1], features[0])
        self.out = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # Downsample 64 -> 128 -> 256 -> 512
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        # Bottleneck 512 -> 1024
        x = self.bottleneck(x)
        # Upsample 512 -> 256 -> 128 -> 64 with concatenation and skip connections
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        x = self.out(x)
        return x
