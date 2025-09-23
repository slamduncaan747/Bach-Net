import torch

class BachNet(torch.nn.Module):
    def __init__(self):
        super(BachNet, self).__init__()
        self.double_conv_1 = DoubleConvBlock(1, 64)
        self.encoder_1 = EncoderBlock(64, 128)
        self.encoder_2 = EncoderBlock(128, 256)
        self.encoder_3 = EncoderBlock(256, 512)

        self.bottleneck = DoubleConvBlock(512, 1024)

        self.decoder_1 = DecoderBlock(1024, 512)
        self.decoder_2 = DecoderBlock(512, 256)
        self.decoder_3 = DecoderBlock(256, 128)
        self.decoder_4 = DecoderBlock(128, 64)

        self.output_1 = torch.nn.Conv2d(64, 1, kernel_size=1)
        self.output_2 = torch.nn.Conv2d(64, 1, kernel_size=1)
        self.output_3 = torch.nn.Conv2d(64, 1, kernel_size=1)
    def forward(self, x):
        skip1 = self.double_conv_1(x)
        downsampled_1, skip2 = self.encoder_1(skip1)
        downsampled_2, skip3 = self.encoder_2(downsampled_1)
        downsampled_3, skip4 = self.encoder_3(downsampled_2)

        bottleneck = self.bottleneck(downsampled_3)

        upsampled_1 = self.decoder_1(bottleneck, skip4)
        upsampled_2 = self.decoder_2(upsampled_1, skip3)
        upsampled_3 = self.decoder_3(upsampled_2, skip2)
        decoder_output = self.decoder_4(upsampled_3, skip1)

        output_1 = self.output_1(decoder_output)
        output_2 = self.output_2(decoder_output)
        upsampled_output_3 = torch.nn.functional.interpolate(decoder_output, size=(28, 64), mode='bilinear', align_corners=False)
        output_3 = self.output_3(upsampled_output_3)
        return output_1, output_2, output_3

        



class DoubleConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.double_conv = DoubleConvBlock(in_channels, out_channels)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        skip_connection_tensor = self.double_conv(x)
        downsampled_tensor = self.pool(skip_connection_tensor)
        return downsampled_tensor, skip_connection_tensor

class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConvBlock(in_channels*2, out_channels)
    def forward(self, x, skip_connection_tensor):
        upsampled_tensor = self.conv1(x)
        combined_tensor = torch.cat((upsampled_tensor, skip_connection_tensor), dim=1)
        feature_map = self.double_conv(combined_tensor)
        return feature_map

