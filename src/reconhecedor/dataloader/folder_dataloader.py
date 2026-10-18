"""Dataloader de pasta (layout tipico de datasets publicos, ex.: LFW)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..types import Sample
from .base import Dataloader

#: Formatos aceitos pela Spec 002 RNF-01.
DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg")


def read_image(path: str | Path) -> np.ndarray | None:
    """Le uma imagem colorida (BGR) de ``path``.

    Usa ``np.fromfile`` + ``cv2.imdecode`` para funcionar com caminhos que
    contem acentos no Windows (``cv2.imread`` falha nesses casos).
    Retorna ``None`` quando o arquivo nao e uma imagem valida.
    """
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class FolderDataloader(Dataloader):
    """Carrega datasets com layout ``root/<identidade>/*.png|jpg|jpeg``.

    Funciona com arquivos de imagem soltos (Spec 003 RF-02) e com datasets
    publicos organizados por pastas de identidade (Spec 003 RF-01).
    Arquivos fora dos formatos aceitos sao ignorados (Spec 002 RNF-01).
    """

    def __init__(
        self,
        root: str | Path,
        extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    ) -> None:
        self.root = Path(root)
        self.extensions = tuple(ext.lower() for ext in extensions)

    def load(self) -> list[Sample]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset nao encontrado: {self.root}")

        samples: list[Sample] = []
        for identity_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            label = identity_dir.name
            for path in sorted(identity_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in self.extensions:
                    continue
                image = read_image(path)
                if image is None:
                    print(f"[dataloader] ignorando arquivo ilegivel: {path}")
                    continue
                samples.append(Sample(path=path, image=image, label=label))
        return samples
