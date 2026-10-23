"""Camada Dataloader (ADR 001): carregamento de imagens dos datasets."""

from .base import Dataloader
from .folder_dataloader import FolderDataloader
from . import lfw_dataset
from .lfw_dataloader import LFWDataloader, build_lfw_loaders

__all__ = [
    "Dataloader",
    "FolderDataloader",
    "LFWDataloader",
    "build_lfw_loaders",
    "lfw_dataset",
]