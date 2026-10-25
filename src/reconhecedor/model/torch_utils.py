"""Utilitarios compartilhados pelas implementacoes de Model baseadas em PyTorch."""

from __future__ import annotations

import numpy as np
import torch

from ..seed import set_seed, DEFAULT_SEED, fix_seed, seed_everything


def to_batch(image: np.ndarray, device: "torch.device") -> "torch.Tensor":
    """Converte a saida do PreProcessor em um batch NCHW pronto para a rede.

    Aceita tanto CHW (saida do ``TorchPreProcessor``) quanto HWC; entradas em
    escala de cinza (2D) sao replicadas em 3 canais para compatibilidade com
    backbones RGB (ResNet50/FaceNet).
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=0)
    elif array.ndim == 3 and array.shape[-1] == 3 and array.shape[0] != 3:
        array = np.transpose(array, (2, 0, 1))
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"embedding espera entrada com 3 canais, recebeu shape {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).to(device)


__all__ = ["to_batch", "set_seed", "DEFAULT_SEED", "fix_seed", "seed_everything"]
