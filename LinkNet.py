import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np
from dataclasses import dataclass
import random

@dataclass
class SystemConfig:
    seed: int = 42                                               # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = False                        # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True                             # make cudnn deterministic (reproducible training)

def setup_system(system_config: SystemConfig) -> None:
    torch.manual_seed(system_config.seed)
    np.random.seed(system_config.seed)
    random.seed(system_config.seed)
    torch.set_printoptions(precision=10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic


class DecoderBlock(nn.Module):                                   # create decoder block inherited from nn.Module
    
    def __init__(self, channels_in, channels_out):
        super().__init__()

        # 1x1 projection module to reduce channels
        self.proj = nn.Sequential(
            nn.Conv2d(channels_in, channels_in // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_in // 4),
            nn.ReLU()
        )

        # fully convolutional module
        self.deconv = nn.Sequential(
            # deconvolution
            nn.ConvTranspose2d(channels_in // 4, channels_in // 4, kernel_size=4, stride=2, padding=1, output_padding=0, groups=channels_in // 4, bias=False),
            nn.BatchNorm2d(channels_in // 4),
            nn.ReLU()
        )

        # 1x1 unprojection module to increase channels
        self.unproj = nn.Sequential(
            nn.Conv2d(channels_in // 4, channels_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU()
        )

    def forward(self, x):                                        # stack layers and perform a forward pass
        proj = self.proj(x) 
        deconv = self.deconv(proj)
        unproj = self.unproj(deconv)

        return unproj



class LinkNet(nn.Module):                                        # create LinkNet model with ResNet18 encoder
    
    def __init__(self, num_classes, encoder="resnet18"):
        super().__init__()
        assert hasattr(models, encoder), "Undefined encoder type"
        
        feature_extractor = getattr(models, encoder)(pretrained=True)

        # Init block: get configured Conv2d, BatchNorm2d layers and ReLU from torch ResNet class
        self.init = nn.Sequential(
            feature_extractor.conv1, 
            feature_extractor.bn1, 
            feature_extractor.relu
        )
        self.maxpool = feature_extractor.maxpool

        # Encoder's blocks: torch ResNet-18 blocks initialization
        self.layer1 = feature_extractor.layer1
        self.layer2 = feature_extractor.layer2
        self.layer3 = feature_extractor.layer3
        self.layer4 = feature_extractor.layer4

        # Decoder's block: DecoderBlock module
        self.up4 = DecoderBlock(self._num_channels(self.layer4), self._num_channels(self.layer3))
        self.up3 = DecoderBlock(self._num_channels(self.layer3), self._num_channels(self.layer2))
        self.up2 = DecoderBlock(self._num_channels(self.layer2), self._num_channels(self.layer1))
        self.up1 = DecoderBlock(self._num_channels(self.layer1), self._num_channels(self.layer1))

        # Classification block: define a classifier module
        self.classifier = nn.Sequential(
            nn.ConvTranspose2d(self._num_channels(self.layer1), 32, 3, stride=2, bias=False),
            nn.BatchNorm2d(32),                                  # num_features = 32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=2, padding=0)
        )


    # get a compatible number of channels to stack all of the LinkNet's blocks together
    @staticmethod
    def _num_channels(block):
        """
           Extract batch-norm num_features from the input block.
        """
        # check whether the input block is models.resnet.BasicBlock type
        if isinstance(block[-1], models.resnet.BasicBlock):
            return block[-1].bn2.weight.size(0)
        # if not extract the spatial characteristic of batch-norm weights from input block
        return block[-1].bn3.weight.size(0)

    def forward(self, x):

        # output size = (64, 160, 160)
        init = self.init(x)

        # output size = (64, 80, 80)
        maxpool = self.maxpool(init)

        # output size = (64, 80, 80)
        layer1 = self.layer1(maxpool)

        # output size = (128, 40, 40)
        layer2 = self.layer2(layer1)

        # output size = (256, 20, 20)
        layer3 = self.layer3(layer2)

        # output size = (512, 10, 10)
        layer4 = self.layer4(layer3)

        # output size = (256, 20, 20)
        up4 = self.up4(layer4) + layer3

        # output size = (128, 40, 40)
        up3 = self.up3(up4) + layer2

        # output size = (64, 80, 80)
        up2 = self.up2(up3) + layer1

        # output size = (64, 160, 160)
        up1 = self.up1(up2)

        # output size = (5, 320, 320), where 5 is the predefined number of classes
        output = self.classifier(up1)

        return output

class ModelProfiler(nn.Module):
    """ Profile PyTorch models.
    Compute FLOPs (FLoating OPerations) and number of trainable parameters of model.

    Arguments:
        model (nn.Module): model which will be profiled.

    Example:
        model = torchvision.models.resnet50()
        profiler = ModelProfiler(model)
        var = torch.zeros(1, 3, 224, 224)
        profiler(var)
        print("FLOPs: {0:.5}; #Params: {1:.5}".format(profiler.get_flops('G'), profiler.get_params('M')))

    Warning:
        Model profiler doesn't work with models, wrapped by torch.nn.DataParallel.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.flops = 0
        self.units = {'K': 10.**3, 'M': 10.**6, 'G': 10.**9}
        self.hooks = None
        self._remove_hooks()

    def get_flops(self, units='G'):
        """ Get number of floating operations per inference.

        Arguments:
            units (string): units of the flops value ('K': Kilo (10^3), 'M': Mega (10^6), 'G': Giga (10^9)).

        Returns:
            Floating operations per inference at the choised units.
        """
        assert units in self.units
        return self.flops / self.units[units]

    def get_params(self, units='G'):
        """ Get number of trainable parameters of the model.

        Arguments:
            units (string): units of the flops value ('K': Kilo (10^3), 'M': Mega (10^6), 'G': Giga (10^9)).

        Returns:
            Number of trainable parameters of the model at the choised units.
        """
        assert units in self.units
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if units is not None:
            params = params / self.units[units]
        return params

    def forward(self, *args, **kwargs):
        self.flops = 0
        self._init_hooks()
        output = self.model(*args, **kwargs)
        self._remove_hooks()
        return output

    def _remove_hooks(self):
        if self.hooks is not None:
            for hook in self.hooks:
                hook.remove()
        self.hooks = None

    def _init_hooks(self):
        self.hooks = []

        def hook_compute_flop(module, _, output):
            self.flops += module.weight.size()[1:].numel() * output.size()[1:].numel()

        def add_hooks(module):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.hooks.append(module.register_forward_hook(hook_compute_flop))

        self.model.apply(add_hooks)


def profile_model(model, input_size, cuda):
    """ Compute FLOPS and #Params of the CNN.

    Arguments:
        model (nn.Module): model which should be profiled.
        input_size (tuple): size of the input variable.
        cuda (bool): if True then variable will be upload to the GPU.

    Returns:
        dict:
            dict["flops"] (float): number of GFLOPs.
            dict["params"] (int): number of million parameters.
    """
    profiler = ModelProfiler(model)
    var = torch.zeros(input_size)
    if cuda:
        var = var.cuda()
    profiler(var)
    return {"flops": profiler.get_flops('G'), "params": profiler.get_params('M')}


setup_system(SystemConfig)

# input data for model check
input_tensor = torch.zeros(1, 3, 320, 320)

# LinkNet architecture
model = LinkNet(num_classes=5, encoder="resnet18")

# examining the prediction size
pred = model(input_tensor)
print('Prediction Size: {}'.format(pred.size()))


input_tensor = torch.zeros(1, 3, 640, 320)

flops, params = profile_model(model, input_tensor.size(), False).values()

print('GFLOPs:\t\t\t\t{}\nNo. of params (in million):\t{}'.format(flops, params))