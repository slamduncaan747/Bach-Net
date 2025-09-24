import torch

class BachNet(torch.nn.Module):
    def __init__(self, dropout_rate=.2):
        super(BachNet, self).__init__()
        self.dropout_rate = dropout_rate
        self.double_conv_1 = DoubleConvBlock(1, 64, dropout_rate=self.dropout_rate)
        self.encoder_1 = EncoderBlock(64, 128, dropout_rate=self.dropout_rate)
        self.encoder_2 = EncoderBlock(128, 256, dropout_rate=self.dropout_rate)
        self.encoder_3 = EncoderBlock(256, 512, dropout_rate=self.dropout_rate)

        self.bottleneck = DoubleConvBlock(512, 1024)

        self.decoder_1 = DecoderBlock(1024, 512, dropout_rate=self.dropout_rate)
        self.decoder_2 = DecoderBlock(512, 256, dropout_rate=self.dropout_rate)
        self.decoder_3 = DecoderBlock(256, 128, dropout_rate=self.dropout_rate)
        self.decoder_4 = DecoderBlock(128, 64, dropout_rate=self.dropout_rate)

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

        resized_for_1_2 = torch.nn.functional.interpolate(decoder_output, size=(21, 64), mode='bilinear', align_corners=False)
        output_1 = self.output_1(resized_for_1_2)
        output_2 = self.output_2(resized_for_1_2)

        resized_for_3 = torch.nn.functional.interpolate(decoder_output, size=(28, 64), mode='bilinear', align_corners=False)
        output_3 = self.output_3(resized_for_3)

        return output_1, output_2, output_3


class DoubleConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(DoubleConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = torch.nn.Dropout2d(p=dropout_rate)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(EncoderBlock, self).__init__()
        self.double_conv = DoubleConvBlock(in_channels, out_channels, dropout_rate=dropout_rate)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connection_tensor = self.double_conv(x)
        downsampled_tensor = self.pool(skip_connection_tensor)
        return downsampled_tensor, skip_connection_tensor


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(DecoderBlock, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.double_conv = DoubleConvBlock(in_channels, out_channels, dropout_rate=dropout_rate)
    
    def forward(self, x, skip_connection_tensor):
        upsampled_tensor = self.conv1(x)
        resized_skip = torch.nn.functional.interpolate(skip_connection_tensor, size=upsampled_tensor.shape[2:])
        combined_tensor = torch.cat((upsampled_tensor, resized_skip), dim=1)
        feature_map = self.double_conv(combined_tensor)
        return feature_map

