"""Bounded, synchronous model streaming for cards that cannot hold a trunk.

Uses ordinary PyTorch device transfers in Forge's process. Each block is
uploaded just before its forward and returned to RAM immediately afterwards,
including when its forward raises. No casting, quantisation or step skipping.
"""
from __future__ import annotations

import torch


class BlockStreamer:
    def __init__(self, modules, device):
        self.device = torch.device(device)
        self._handles = []
        self._modules = list(modules)

    def install(self):
        if self._handles:
            return
        for module in self._modules:
            self._handles.append(module.register_forward_pre_hook(self._upload))
            self._handles.append(module.register_forward_hook(self._park, always_call=True))

    def _upload(self, module, args):
        module.to(device=self.device)

    def _park(self, module, args, output):
        # Blocking transfer: weights cannot leave CUDA while their kernels
        # still read them. Parameters retain dtype and their object identity.
        module.to(device='cpu')

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._modules.clear()


def transformer_modules(model):
    """Non-overlapping forward units; ModuleList itself is never called."""
    for name, child in model.named_children():
        if name == 'layers':
            yield from child
        else:
            yield child


def text_encoder_modules(model):
    # The visual tower is never called for Ideogram text prompts. Leave it
    # on CPU; only the language tower's actually called modules need CUDA.
    lm = model.language_model
    yield lm.embed_tokens
    yield lm.rotary_emb
    yield from lm.layers
