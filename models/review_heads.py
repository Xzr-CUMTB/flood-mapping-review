"""Small spatial heads for event-cross-fitted evidence value and omission ranking."""
import torch
from torch import nn


def entropy(probability):
    p = probability.float().clamp(1e-6, 1-1e-6)
    return -(p*p.log() + (1-p)*torch.log1p(-p))


def evidence_features(x, sar_probability, fusion_probability, coverage):
    """Raw six-band inputs plus inference-available probabilities and coverage.

    No coordinates, event identity, labels, or unverified acquisition age enter the head.
    coverage has two channels: SAR and optical input support.
    """
    s, f = sar_probability[:, None], fusion_probability[:, None]
    return torch.cat([x, s, f, (s-f).abs(), entropy(s), entropy(f), coverage], dim=1)


class SpatialHead(nn.Module):
    def __init__(self, channels=13, width=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1), nn.GroupNorm(6, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GroupNorm(6, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.GroupNorm(6, width), nn.SiLU(),
            nn.Conv2d(width, 1, 1))

    def forward(self, features):
        return self.net(features).squeeze(1)


def route(logits, sar_probability, fusion_probability, optical_available, kind='value', temperature=.05):
    if temperature <= 0:
        raise ValueError('Positive routing temperature required')
    if kind == 'value':
        value = logits.float().tanh()
        gate = torch.sigmoid(value / temperature)
    elif kind == 'ordinary_gate':
        value = None
        gate = logits.float().sigmoid()
    else:
        raise ValueError(kind)
    gate = torch.where(optical_available.bool(), gate, torch.zeros_like(gate))
    fusion_safe = torch.where(optical_available.bool(), fusion_probability, sar_probability)
    probability = sar_probability + gate * (fusion_safe - sar_probability)
    return probability, gate, value


def omission_features(base_features, system_probability, gate):
    return torch.cat([base_features, system_probability[:, None], entropy(system_probability)[:, None], gate[:, None]], dim=1)


def omission_probability(logits, system_probability):
    # The prediction decision is available at inference; predicted-positive pixels cannot be false negatives.
    return logits.float().sigmoid() * (system_probability < .5)
