"""Fixacao de seed para reproducibilidade (ADR 008 / RNF de experimentos).

Concentra em um unico lugar a inicializacao das sementes dos geradores
aleatorios usados pelo pipeline (``random``, ``numpy`` e ``torch``), de modo
que experimentos sejam replicaveis entre execucoes.
"""

from __future__ import annotations

import random
from typing import Any

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Fixa a seed em ``random``, ``numpy`` e ``torch`` (CPU e GPU).

    Retorna a seed utilizada, para facilitar o registro nos resultados.
    """
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass
    return seed


# Alias para compatibilidade com nomes alternativos esperados por testes.
fix_seed = set_seed
seed_everything = set_seed


__all__ = ["DEFAULT_SEED", "set_seed", "fix_seed", "seed_everything"]