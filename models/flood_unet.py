"""Compact matched-capacity U-Net for the first controlled modality experiment."""
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.SiLU(),
                                  nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.SiLU())
    def forward(self,x):return self.layers(x)

class FloodUNet(nn.Module):
    def __init__(self,modality='fusion',width=16):
        super().__init__()
        if modality not in ['sar','fusion','optical','rgb_fusion']:raise ValueError(modality)
        self.modality=modality
        self.enc=nn.ModuleList([Block(6,width),Block(width,width*2),Block(width*2,width*4),Block(width*4,width*8)])
        self.bridge=Block(width*8,width*16)
        self.dec=nn.ModuleList([Block(width*24,width*8),Block(width*12,width*4),Block(width*6,width*2),Block(width*3,width)])
        self.output=nn.Conv2d(width,1,1)
        mask=torch.ones(1,6,1,1)
        if modality=='sar':mask[:,2:]=0
        if modality=='optical':mask[:,:2]=0
        if modality=='rgb_fusion':mask[:,5:]=0
        self.register_buffer('input_mask',mask)
    def forward(self,x):
        # Same six-channel first layer and parameter count for every modality ablation.
        x=x*self.input_mask;skips=[]
        for layer in self.enc:
            x=layer(x);skips.append(x);x=F.max_pool2d(x,2)
        x=self.bridge(x)
        for layer,skip in zip(self.dec,reversed(skips)):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=layer(torch.cat([x,skip],dim=1))
        return self.output(x).squeeze(1)
