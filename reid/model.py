import os
import sys
import torch
import timm
# import open_clip
import numpy as np
import torch.nn as nn
import torchvision.models as models

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
# from module.cbam import SAM, CAM, CBAM
# from torchvision.models.squeezenet import SqueezeNet1_1_Weights
      
class MobileNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenetv3 = timm.create_model('mobilenetv3_large_100.miil_in21k_ft_in1k')
    
    def forward(self, x):
        embedding_feature = self.mobilenetv3(x)
        return embedding_feature