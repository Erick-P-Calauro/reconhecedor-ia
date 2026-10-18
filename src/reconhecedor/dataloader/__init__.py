"""Camada Dataloader (ADR 001): carregamento de imagens dos datasets."""

from .base import Dataloader
from .folder_dataloader import FolderDataloader, read_image

__all__ = ["Dataloader", "FolderDataloader", "read_image"]
