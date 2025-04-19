# TODO - implement the training loop incorporating the forgery augmentation methods and the monotonic curriculum
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Backbone network fα using SwinTransformerV2-Base
class Backbone(nn.Module):
    def __init__(self, checkpoint_path=None):
        super(Backbone, self).__init__()
        self.model = timm.create_model('swinv2_base_window12_192_22k', pretrained=False, num_classes=0)
        # REFERENCE : https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md#simmim-pretrained-swin-v2-models
        # We are loading the checkpoint_path - SwinTransformerV2-Base; 87M Parameter ; ImageNet-22k;
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)

# hγ: three-layer MLP head with softmax
class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return F.softmax(self.mlp(x), dim=1)

# Combined model
class FullModel(nn.Module):
    def __init__(self, backbone_ckpt_path, hidden_dim, output_dim):
        super(FullModel, self).__init__()
        self.backbone = Backbone(checkpoint_path=backbone_ckpt_path)
        backbone_output_dim = self.backbone.model.num_features
        self.head = MLPHead(backbone_output_dim, hidden_dim, output_dim)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


model = FullModel(
    backbone_ckpt_path='swinv2_base.pth',
    hidden_dim=512,
    output_dim=10 
)

optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50)
