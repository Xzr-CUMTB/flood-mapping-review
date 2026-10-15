"""TerraMind tiny adaptation for uint8 S2 RGB plus new GF3 HH/HV and NIR inputs."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU())


class TerraMindFlood(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from experiments.foundation_environment import configure
        configure()
        from terratorch.registry import BACKBONE_REGISTRY
        from terratorch.models.backbones.terramind.model.terramind_register import (
            PRETRAINED_BANDS, v1_pretraining_mean, v1_pretraining_std)
        # The pinned implementation and current register source both declare R,G,B.
        # The prose guide lists B,G,R; this discrepancy is retained in the model audit.
        rgb_key = 'untok_sen2rgb@224'
        if PRETRAINED_BANDS[rgb_key] != ['RED', 'GREEN', 'BLUE']:
            raise ValueError('Review RGB band ordering for this TerraTorch version')
        self.backbone = BACKBONE_REGISTRY.build('terramind_v1_tiny', pretrained=False,
            modalities=['RGB', {'GF3_HH_HV_uint8': 2}, {'S2_NIR_uint8': 1}],
            img_size=224, merge_method='mean', modality_drop_rate=0.)
        self.register_buffer('rgb_mean', torch.tensor(v1_pretraining_mean[rgb_key]).reshape(1, 3, 1, 1))
        self.register_buffer('rgb_std', torch.tensor(v1_pretraining_std[rgb_key]).reshape(1, 3, 1, 1))
        self.pretrained_loading = None
        if pretrained:
            path = ROOT/'models/pretrained/terramind_tiny/TerraMind_v1_tiny.pt'
            source = torch.load(path, map_location='cpu', weights_only=True)
            target = self.backbone.state_dict()
            pretrained_keys = [key for key in target if key.startswith('encoder.') or key.startswith('encoder_norm.')
                               or key.startswith('encoder_embeddings.'+rgb_key+'.')]
            for key in pretrained_keys:
                if key not in source or source[key].shape != target[key].shape:
                    raise ValueError('Missing or incompatible pretrained weight: '+key)
            accepted = {key: value for key, value in source.items() if key in target and value.shape == target[key].shape}
            incompatible = self.backbone.load_state_dict(accepted, strict=False)
            if incompatible.unexpected_keys:
                raise ValueError('Unexpected pretrained keys')
            allowed = ('encoder_embeddings.GF3_HH_HV_uint8.', 'encoder_embeddings.S2_NIR_uint8.')
            if any(not key.startswith(allowed) for key in incompatible.missing_keys):
                raise ValueError('An expected pretrained layer was left random')
            self.pretrained_loading = {'matched_tensors': len(accepted), 'required_pretrained_tensors': len(pretrained_keys),
                'new_input_tensors': incompatible.missing_keys, 'unused_source_tensors': len(source)-len(accepted),
                'checkpoint': str(path), 'rgb_order': ['RED', 'GREEN', 'BLUE'],
                'rgb_order_source': 'PRETRAINED_BANDS in pinned terratorch 1.2.6, also present in current register source',
                'guide_discrepancy': 'The prose guide lists BLUE,GREEN,RED; not used to override executable PRETRAINED_BANDS'}
        self.project = nn.Sequential(nn.Conv2d(192*4, 128, 1, bias=False), nn.GroupNorm(8, 128), nn.SiLU())
        self.decode = nn.ModuleList([conv_block(128, 64), conv_block(64, 32), conv_block(32, 16)])
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        # Cached channels are GF3 HH, HV, S2 R, G, B, NIR, each divided by 255.
        rgb = (x[:, 2:5]*255.-self.rgb_mean)/self.rgb_std
        inputs = {'RGB': rgb, 'GF3_HH_HV_uint8': x[:, :2]*2.-1., 'S2_NIR_uint8': x[:, 5:6]*2.-1.}
        layers = self.backbone(inputs)
        height, width = x.shape[-2]//16, x.shape[-1]//16
        maps = [layers[i].transpose(1, 2).reshape(x.shape[0], 192, height, width) for i in [2, 5, 8, 11]]
        z = self.project(torch.cat(maps, dim=1))
        for block, scale in zip(self.decode, [4, 2, 2]):
            z = block(F.interpolate(z, scale_factor=scale, mode='bilinear', align_corners=False))
        return self.output(z).squeeze(1)

    def optimizer_groups(self, backbone_lr=1e-4, adaptation_lr=1e-3):
        slow, fast = [], []
        for name, value in self.named_parameters():
            if name.startswith('backbone.encoder.') or name.startswith('backbone.encoder_norm.'):
                slow.append(value)
            else:
                fast.append(value)
        return [{'params': slow, 'lr': backbone_lr}, {'params': fast, 'lr': adaptation_lr}]
