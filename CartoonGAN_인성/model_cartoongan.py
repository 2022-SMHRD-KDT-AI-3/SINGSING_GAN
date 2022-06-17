import torch
from torch import nn
import torchvision


# class Encoder1(nn.Module):
#     def __init__(self):
#         super(Encoder1, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(3, 64, 7, 1, 3, bias=False),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.layers(x)


# class Encoder2(nn.Module):
#     def __init__(self, channel_input):
#         super(Encoder2, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(channel_input, channel_input * 2, 3, 2, 1),
#             nn.Conv2d(channel_input * 2, channel_input * 2, 3, 2, 1, bias=False),
#             nn.InstanceNorm2d(channel_input * 2),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.layers(x)


# class ResidualBlock(nn.Module):
#     def __init__(self):
#         super(ResidualBlock, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(256, 256, 3, 1, 1, bias=False),
#             nn.InstanceNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, 1, 1, bias=False),
#             nn.InstanceNorm2d(256)
#         )

#     def forward(self, x):
#         return x + self.layers(x)


# class Decoder(nn.Module):
#     def __init__(self, channel_input):
#         super(Decoder, self).__init__()
#         self.layers = nn.Sequential(
#             nn.ConvTranspose2d(channel_input, channel_input // 2, 3, 2, 1, output_padding=1),
#             nn.Conv2d(channel_input // 2, channel_input // 2, 3, 2, 1, bias=False),
#             nn.InstanceNorm2d(channel_input / 2),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.layers(x)


# class DiscriminatorLayer(nn.Module):
#     def __init__(self, channel_input, channel_middle):
#         super(DiscriminatorLayer, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(channel_input, channel_middle, 3, 2, 1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(channel_middle, channel_middle * 2, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(channel_middle * 2),
#             nn.LeakyReLU(0.2, inplace=True)
#         )

#     def forward(self, x):
#         return self.layers(x)


# class CartoonGANGenerator(nn.Module):
#     def __init__(self):
#         super(CartoonGANGenerator, self).__init__()
#         self.layers = nn.Sequential(
#             Encoder1(),
#             Encoder2(64),
#             Encoder2(128),
#             *[ResidualBlock() for i in range(8)],
#             Decoder(256),
#             Decoder(128),
#             nn.Conv2d(64, 3, 7, 1, 3),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         return self.layers(x)


# class CartoonGANDiscriminator(nn.Module):
#     def __init__(self):
#         super(CartoonGANDiscriminator, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(3, 32, 3, 1, 1),
#             nn.LeakyReLU(0.2, inplace=True),
#             DiscriminatorLayer(32, 64),
#             DiscriminatorLayer(128, 128),
#             nn.Conv2d(256, 256, 3, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 1, 3, 1, 1)
#         )

#     def forward(self, x):
#         return self.layers(x)


# class VGG19(nn.Module):
#     def __init__(self):
#         super().__init__()
#         vgg = torchvision.models.vgg19_bn(pretrained=True)
#         self.feature_extractor = vgg.features[:37]

#         for child in self.feature_extractor.children():
#             for param in child.parameters():
#                 param.requires_grad = False

#     def forward(self, input):
#         return self.feature_extractor(input)
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid

class ResidualBlock(nn.Module):
  def __init__(self):
    super(ResidualBlock, self).__init__()
    self.conv_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.conv_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.norm_1 = nn.BatchNorm2d(256)
    self.norm_2 = nn.BatchNorm2d(256)

  def forward(self, x): 
    output = self.norm_2(self.conv_2(F.relu(self.norm_1(self.conv_1(x)))))
    return output + x #ES

class Generator(nn.Module):
    def __init__(self):
      super(Generator, self).__init__()
      self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)
      self.norm_1 = nn.BatchNorm2d(64)
      
      # down-convolution #
      self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
      self.conv_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
      self.norm_2 = nn.BatchNorm2d(128)
      
      self.conv_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
      self.conv_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
      self.norm_3 = nn.BatchNorm2d(256)
      
      # residual blocks #
      residualBlocks = []
      for l in range(8):
        residualBlocks.append(ResidualBlock())
      self.res = nn.Sequential(*residualBlocks)
      
      # up-convolution #
      self.conv_6 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
      self.conv_7 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
      self.norm_4 = nn.BatchNorm2d(128)

      self.conv_8 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
      self.conv_9 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
      self.norm_5 = nn.BatchNorm2d(64)
      
      self.conv_10 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
      x = F.relu(self.norm_1(self.conv_1(x)))
      
      x = F.relu(self.norm_2(self.conv_3(self.conv_2(x))))
      x = F.relu(self.norm_3(self.conv_5(self.conv_4(x))))
      
      x = self.res(x)
      x = F.relu(self.norm_4(self.conv_7(self.conv_6(x))))
      x = F.relu(self.norm_5(self.conv_9(self.conv_8(x))))

      x = self.conv_10(x)

      x = sigmoid(x)

      return x