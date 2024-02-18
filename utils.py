import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as F

from PIL import Image
from typing import Optional
from functools import partial
from torch import Tensor
from torchvision import transforms

from canny_tools import Canny  # canny edge detection
from transformers import DPTForDepthEstimation  # depth estimation

from torchvision.transforms import RandomCrop


def get_reward_model(task='segmentation', model_path='mmseg::upernet/upernet_r50_4xb4-160k_ade20k-512x512.py'):
    """Return reward model for different tasks.

    Args:
        task (str, optional): Task name. Defaults to 'segmentation'.
        model_path (str, optional): Model name or pre-trained path.

    """
    if task == 'canny':
        return Canny()
    elif task == 'line_drawing':
        model = LineDrawingModel()
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_path, map_location=torch.device('cpu')))
        return model
    elif task == 'hed':
        return HEDdetector(model_path)
    else:
        return None



norm_layer = nn.InstanceNorm2d
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class LineDrawingModel(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, n_residual_blocks=3, sigmoid=True):
        super(LineDrawingModel, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out



class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        for i in range(1, layer_number):
            self.convs.append(torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = torch.nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class HEDdetector(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        state_dict = torch.hub.load_state_dict_from_url(model_path, map_location=torch.device('cpu'))

        self.netNetwork = ControlNetHED_Apache2()
        self.netNetwork.load_state_dict(state_dict)

    def __call__(self, input_image):
        H, W = input_image.shape[2], input_image.shape[3]

        edges = self.netNetwork((input_image * 255).clip(0, 255))
        edges = [torch.nn.functional.interpolate(edge, size=(H, W), mode='bilinear') for edge in edges]
        edges = torch.stack(edges, dim=1)
        edge = 1 / (1 + torch.exp(-torch.mean(edges, dim=1)))
        edge = (edge * 255.0).clip(0, 255).to(torch.uint8)

        return edge / 255.0

        # with torch.no_grad():
        #     image_hed = torch.from_numpy(input_image.copy()).float().cuda()
        #     image_hed = rearrange(image_hed, 'h w c -> 1 c h w')
        #     edges = self.netNetwork(image_hed)
        #     edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
        #     edges = [cv2.resize(e, (W, H)) for e in edges]
        #     edges = np.stack(edges, axis=2)
        #     edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
        #     if safe:
        #         edge = safe_step(edge)
        #     edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
        #     return edge