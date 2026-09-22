"""Portable compute policy. ROCm exposes AMD devices through torch.cuda too."""
import torch


def resolve_device(device='cuda'):
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA/ROCm device is available. Install a PyTorch build supported by your GPU and driver, or use CPU explicitly.')
        device = torch.device('cuda', torch.cuda.current_device())
    if device.type not in ('cpu', 'cuda'):
        raise ValueError(f'Ideogram supports PyTorch CUDA/ROCm or CPU; {device.type} is not implemented.')
    return device


def compute_dtype(device='cuda', requested=torch.bfloat16):
    device = resolve_device(device)
    if requested != torch.bfloat16:
        return requested
    if device.type == 'cuda':
        try:
            with torch.cuda.device(device):
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
        except (AttributeError, RuntimeError):
            pass
    # FP16 has a narrower exponent range and is not a safe blanket substitute
    # for BF16-trained weights. FP32 is slower/larger but preserves that range.
    return torch.float32
