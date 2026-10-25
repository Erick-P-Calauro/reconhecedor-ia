"""Reconhecedor Facial : Um pipeline de reconhecimento facial em camadas (ADR 001)."""

__version__ = "0.1.0"

from .seed import set_seed, DEFAULT_SEED, fix_seed, seed_everything

__all__ = ["set_seed", "DEFAULT_SEED", "fix_seed", "seed_everything"]
