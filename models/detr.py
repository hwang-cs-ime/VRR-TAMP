# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)

from .backbone import build_backbone
from .transformer import build_vis_transformer, build_transformer
from .position_encoding import build_position_encoding

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, img, mask):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        samples = NestedTensor(img, mask)
        # pos: position encoding
        features, pos = self.backbone(samples)  # pos:list, pos[-1]: [64, 256, 20, 20]

        src, mask = features[-1].decompose()  # src:[64, 2048, 20, 20]  mask:[64,20,20]
        assert mask is not None
        out = self.transformer(self.input_proj(src), mask, pos[-1])
        
        return out
    
class VLFusion(nn.Module):
    def __init__(self, transformer, pos):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: no use
            """
        super().__init__()
        #self.num_queries = num_queries
        self.transformer = transformer
        self.pos = pos
        hidden_dim = transformer.d_model
        self.pr = nn.Embedding(1, hidden_dim)
        self.v_proj = torch.nn.Sequential(
          nn.Linear(256, 256),
          nn.ReLU(),)
        self.l_proj = torch.nn.Sequential(
          nn.Linear(768, 256),
          nn.ReLU(),)

    def forward(self, fv, fl):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        bs, c, h, w = fv.shape
        _, _, l = fl.shape

        pv = self.v_proj(fv.view(bs, c, -1).permute(0,2,1))
        pl = self.l_proj(fl)
        pv = pv.permute(0,2,1)
        pl = pl.permute(0,2,1)

        pr = self.pr.weight
        pr = pr.expand(bs,-1).unsqueeze(2)

        x0 = torch.cat((pv, pl), dim=2)
        x0 = torch.cat((x0, pr), dim=2)
        
        pos = self.pos(x0).to(x0.dtype)
        mask = torch.zeros([bs, x0.shape[2]]).cuda()
        mask = mask.bool()
        
        out = self.transformer(x0, mask, pos)
        
        return out[-1]



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_detr(args):

    device = torch.device(args.device)
    backbone = build_backbone(args) # ResNet 50
    transformer = build_vis_transformer(args)

    model = DETR(
        backbone,
        transformer,
    )
    return model

def build_VLFusion(args):

    device = torch.device(args.device)
    
    transformer = build_transformer(args)

    pos = build_position_encoding(args, position_embedding = 'learned')

    model = VLFusion(
        transformer,
        pos,
    )
    return model
