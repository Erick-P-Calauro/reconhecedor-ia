"""Camada Dataloader (ADR 001): carregamento de imagens dos datasets."""

from .base import Dataloader
from .folder_dataloader import FolderDataloader
from . import lfw_dataset
from .lfw_dataloader import LFWDataloader, build_lfw_loaders, build_lfw_splits, build_lfw_splits_finetune, _ListDataloader

__all__ = [
    "Dataloader",
    "FolderDataloader",
    "LFWDataloader",
    "build_lfw_loaders",
    "build_lfw_splits",
    "_ListDataloader",
    "lfw_dataset",
]