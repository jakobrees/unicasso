"""Toggleable LoRA/FiLM adapters for CLIP's ModifiedResNet visual tower.

Dual-path contract: ONE model serves both sides of every comparison. Adapters ENABLED =
the ascii/render path (trainable); adapters DISABLED = bit-exact base CLIP for the
parent/target path. All base weights stay frozen; every adapter is zero-initialized, so
an untrained adapter is an exact no-op (step-0 eval = base-model numbers for free).

What gets adapted (RN101):
  FiLM        per-channel scale/shift after every BatchNorm, stem -> layer3. The conv-net
              equivalent of "train the BN affines", but as a residual delta on a frozen
              BN so the disabled path stays untouched.
  LoRAConv1x1 rank-r update on every 1x1 conv in layer1..3 (bottleneck compress/expand +
              downsample projections). A 1x1 conv IS a Linear over channels -- textbook
              matrix LoRA, no kernel reshaping.
  LoRALinear  (OFF by default) attnpool q/k/v/c_proj. open_clip's AttentionPool2d reads
              `self.q_proj.weight` directly into F.multi_head_attention_forward instead
              of calling the module, so wrapping is a no-op that breaks attribute lookup
              -- only enable if attnpool is ever refactored. The 0.1-weight semantic
              term still adapts indirectly through the trunk below.
Layer4's convs stay fully frozen (no geometric readout there; its features only feed
attnpool).
"""
from contextlib import contextmanager

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """y = x * (1 + g) + b, per channel, zero-init. Wraps a frozen BN (delta AFTER it)."""

    def __init__(self, bn):
        super().__init__()
        self.bn = bn
        c = bn.num_features
        self.g = nn.Parameter(torch.zeros(c))
        self.b = nn.Parameter(torch.zeros(c))
        self.enabled = True

    def forward(self, x):
        x = self.bn(x)
        if self.enabled:
            x = x * (1.0 + self.g.view(1, -1, 1, 1)) + self.b.view(1, -1, 1, 1)
        return x


class LoRAConv1x1(nn.Module):
    def __init__(self, conv, rank=8, scale=1.0):
        super().__init__()
        self.base = conv
        self.down = nn.Conv2d(conv.in_channels, rank, 1, stride=conv.stride, bias=False)
        self.up = nn.Conv2d(rank, conv.out_channels, 1, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / rank)
        nn.init.zeros_(self.up.weight)                       # zero-init: exact no-op at start
        self.scale = scale
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + self.scale * self.up(self.down(x))
        return y


class LoRALinear(nn.Module):
    def __init__(self, lin, rank=8, scale=1.0):
        super().__init__()
        self.base = lin
        self.down = nn.Linear(lin.in_features, rank, bias=False)
        self.up = nn.Linear(rank, lin.out_features, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / rank)
        nn.init.zeros_(self.up.weight)
        self.scale = scale
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + self.scale * self.up(self.down(x))
        return y


def inject_adapters(visual, rank=8, scale=1.0, max_stage=3, attnpool=False):
    """Wrap the target modules in-place. Returns {name: adapter} (insertion order stable).
    Call ONCE on a freshly-loaded frozen tower."""
    adapters = {}

    def wrap(owner, key, mod):
        adapters[f"{prefix}.{key}" if prefix else key] = mod
        owner._modules[key] = mod                            # works for Sequential digit keys too

    prefix = "stem"
    for key in ("bn1", "bn2", "bn3"):                        # stem convs are 3x3 -> FiLM only
        wrap(visual, key, FiLM(getattr(visual, key)))
    for i in range(1, max_stage + 1):
        stage = getattr(visual, f"layer{i}")
        for mname, m in list(stage.named_modules()):
            for key, ch in list(m._modules.items()):
                prefix = f"layer{i}" + (f".{mname}" if mname else "")
                if isinstance(ch, nn.Conv2d) and ch.kernel_size == (1, 1):
                    wrap(m, key, LoRAConv1x1(ch, rank, scale))
                elif isinstance(ch, nn.BatchNorm2d):
                    wrap(m, key, FiLM(ch))
    if attnpool:
        prefix = "attnpool"
        for key in ("q_proj", "k_proj", "v_proj", "c_proj"):
            wrap(visual.attnpool, key, LoRALinear(getattr(visual.attnpool, key), rank, scale))
    return adapters


def adapter_parameters(adapters):
    """(film_params, lora_params) -- separate groups (FiLM tolerates a higher LR)."""
    film, lora = [], []
    for m in adapters.values():
        if isinstance(m, FiLM):
            film += [m.g, m.b]
        else:
            lora += list(m.down.parameters()) + list(m.up.parameters())
    return film, lora


def set_enabled(adapters, flag):
    for m in adapters.values():
        m.enabled = flag


@contextmanager
def adapters_disabled(adapters):
    """Frozen-base pass: the parent/target side of every comparison."""
    set_enabled(adapters, False)
    try:
        yield
    finally:
        set_enabled(adapters, True)


def save_adapters(path, adapters, rank, scale, extra=None):
    state = {name: {k: v for k, v in m.state_dict().items() if not k.startswith("base.")
                    and not k.startswith("bn.")}             # adapter deltas only, never base
             for name, m in adapters.items()}
    torch.save(dict(state=state, rank=rank, scale=scale, extra=extra or {}), path)


def load_adapters(visual, path, device="cpu"):
    """Inject + load. Returns (adapters dict, extra)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    adapters = inject_adapters(visual, rank=ck["rank"], scale=ck["scale"])
    for name, m in adapters.items():
        m.load_state_dict(ck["state"][name], strict=False)
    return adapters, ck.get("extra", {})
