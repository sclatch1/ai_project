import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class SpatialPyramidPooling(nn.Module):
    def __init__(self, levels=[1, 2, 4], pool_type='max'):
        super().__init__()
        self.levels = levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        outputs = []

        for L in self.levels:
            if self.pool_type == 'max':
                # (L x L) map, each cell exactly
                tensor = F.adaptive_max_pool2d(x, output_size=(L, L))
            else:
                tensor = F.adaptive_avg_pool2d(x, output_size=(L, L))
            # flatten each pooled feature-map
            outputs.append(tensor.view(bs, -1))

        # concatenate all levels
        return torch.cat(outputs, dim=1)



class ResNet18_SPP(nn.Module):
    """
    ResNet-18 with Spatial Pyramid Pooling and two FC layers (fc6, fc7) before final classifier,
    following "Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition".
    """
    def __init__(self, num_classes, spp_levels=[1, 2, 4], pretrained=False, pool_type='max', fc_dim=1024, dropout=0.5):
        super().__init__()
        # Load ResNet18 backbone, remove avgpool & fc
        encoder = resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(encoder.children())[:-2])
        feat_dim = encoder.fc.in_features

        # SPP module
        self.spp = SpatialPyramidPooling(levels=spp_levels, pool_type=pool_type)
        spp_bins = sum([lvl * lvl for lvl in spp_levels])
        spp_dim = feat_dim * spp_bins

        # Interleaved FC layers as in SPP-Net
        self.fc6 = nn.Linear(spp_dim, fc_dim)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout(p=dropout)
        self.norm6 = nn.LayerNorm(spp_dim)
        self.fc7 = nn.Linear(fc_dim, fc_dim)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout(p=dropout)
        
        self.norm7 = nn.LayerNorm(fc_dim)
        # Final classification layer
        self.classifier = nn.Linear(fc_dim, num_classes)

        # Weight initialization for FC layers
        for m in [self.fc6, self.fc7, self.classifier]:
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.spp(x)
        x = self.norm6(x)
        x = self.drop6(self.relu6(self.fc6(x)))
        x = self.drop7(self.relu7(self.fc7(x)))
        x = self.norm7(x)
        x = self.classifier(x)
        return x


def build_model_spp(num_classes, spp_levels=[1, 2, 4], pretrained=False, pool_type='max', fc_dim=128, dropout=0.3):
    """
    Utility to build the SPP-ResNet18 model for scene classification.
    """
    model = ResNet18_SPP(
        num_classes=num_classes,
        spp_levels=spp_levels,
        pretrained=pretrained,
        pool_type=pool_type,
        fc_dim=fc_dim,
        dropout=dropout
    )

    return model