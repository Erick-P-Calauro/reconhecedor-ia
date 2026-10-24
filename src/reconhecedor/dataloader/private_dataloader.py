from __future__ import annotations
import json
from pathlib import Path
from pathlib import Path
from typing import Any
import numpy as np
from .base import Dataloader
from .folder_dataloader import FolderDataloader
from ..types import Sample

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_DEFAULT_MIN_FACES = 4
_DEFAULT_IDENTITIES = 15
_DEFAULT_KNOWN = 7
_DEFAULT_GALLERY_RATIO = 0.7
_DEFAULT_MAX_PER_IDENTITY = 25
_DEFAULT_RESIZE = 1.0
DEFAULT_DATASET = "data/private"
FULL_SLICE = (slice(0, 250), slice(0, 250))

def _has_images(path: Path) -> bool:
    return path.is_dir() and any(
        entry.is_file() and entry.suffix.lower() in _IMAGE_SUFFIXES for entry in path.rglob("*")
    )

def resolve_data_home(root: str | Path | None = None, data_home: str | Path | None = None) -> str | Path:
    return "private"

def ensure_dataset(
    dataset_root: str | Path = DEFAULT_DATASET,
    *,
    rematerialize: bool = False,
    force: bool = False,
    **kwargs,
) -> dict:
    print("[Dataset] Inicializando Private Dataset")
    root = Path(dataset_root)
    info_path = root / "dataset_info.json"
    if _has_images(root / "gallery") and _has_images(root / "probe"):
        if info_path.is_file():
            return json.loads(info_path.read_text(encoding="utf-8"))
    return {"dataset_root": str(root), "status": "ja_materializado"}

class PrivateDataloader(Dataloader):

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "gallery",
        *,
        direct: bool = False,
        mode: str | None = None,
        rematerialize: bool = False,
        force: bool = False,
        min_faces_per_person: int = _DEFAULT_MIN_FACES,
        n_identities: int = _DEFAULT_IDENTITIES,
        n_known: int = _DEFAULT_KNOWN,
        gallery_ratio: float = _DEFAULT_GALLERY_RATIO,
        max_per_identity: int = _DEFAULT_MAX_PER_IDENTITY,
        resize: float = _DEFAULT_RESIZE,
        data_home: str | Path | None = None,
    ) -> None:
        print("[Dataloader] inicializando")
        self.root = Path(root) if root is not None else Path(DEFAULT_DATASET)
        self.split = split
        self.direct = bool(direct) or (mode == "direct")
        self.rematerialize = bool(rematerialize or force)
        self.force = bool(force)
        self.min_faces_per_person = int(min_faces_per_person)
        self.n_identities = int(n_identities)
        self.n_known = int(n_known)
        self.gallery_ratio = float(gallery_ratio)
        self.max_per_identity = int(max_per_identity)
        self.resize = float(resize)
        self.data_home = data_home
        self._params = {
            "min_faces_per_person": self.min_faces_per_person,
            "n_identities": self.n_identities,
            "n_known": self.n_known,
            "gallery_ratio": self.gallery_ratio,
            "max_per_identity": self.max_per_identity,
            "resize": self.resize,
        }
        print("[Dataloader] inicializado")

    def _prepare_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self._params)
        if self.data_home is not None:
            kwargs["data_home"] = self.data_home
        return kwargs

    def load(self) -> list[Sample]:
        # materializado
        kwargs = self._prepare_kwargs()
        ensure_dataset(
            self.root,
            rematerialize=self.rematerialize,
            force=self.force,
            **kwargs,
        )
        folder = self.root / self.split
        samples = FolderDataloader(folder).load()
        print(f"[Dataloader] {len(samples)} samples carregados")
        return samples


def build_private_loaders(
    root: str | Path | None = None,
    *,
    direct: bool = False,
    mode: str | None = None,
    rematerialize: bool = False,
    force: bool = False,
    **kwargs: Any,
) -> tuple[PrivateDataloader, PrivateDataloader]:
    """Factory que retorna (gallery_loader, probe_loader)."""
    common = dict(
        root=root,
        direct=direct or (mode == "direct"),
        mode=mode,
        rematerialize=rematerialize,
        force=force,
        **kwargs,
    )
    gallery = PrivateDataloader(split="gallery", **common)
    probe = PrivateDataloader(split="probe", **common)
    return gallery, probe


class _ListDataloader(Dataloader):
    """Dataloader leve que devolve uma lista pre-carregada (sem reler disco)."""

    def __init__(self, samples: list[Sample] | None = None) -> None:
        self._samples = list(samples or [])

    def load(self) -> list[Sample]:
        print(f"[Dataloader] carregando: lista em memoria ({len(self._samples)})")
        return list(self._samples)


def _split_probe_samples(probe_samples, *, validation_ratio: float, test_ratio: float = None):
    """Divide amostras de probe em validacao/teste por identidade (deterministico)."""
    if test_ratio is None:
        test_ratio = 1.0 - validation_ratio

    by_label: dict[str, list[Sample]] = {}
    for sample in probe_samples:
        by_label.setdefault(sample.label, []).append(sample)

    val_ids: set[int] = set()
    for label, group in by_label.items():
        ordered = sorted(group, key=lambda s: str(s.path))
        n = len(ordered)
        if n <= 1:
            n_val = 0
        else:
            n_val = int(round(n * validation_ratio))
            if n_val < 1:
                n_val = 1
            if n_val > n - 1:
                n_val = n - 1
        for sample in ordered[:n_val]:
            val_ids.add(id(sample))

    validation = [s for s in probe_samples if id(s) in val_ids]
    test = [s for s in probe_samples if id(s) not in val_ids]
    return validation, test

split_probe = _split_probe_samples

def build_private_splits(
    root=None,
    *,
    sample_size,
    n_classes,
    min_photos_per_class,
    validation_ratio,
    test_ratio,
    rematerialize=False,
    force=False,
    **kwargs: Any,
) -> tuple[_ListDataloader, _ListDataloader, _ListDataloader]:
    """Factory que retorna (gallery_loader, validation_loader, test_loader).

    Carrega gallery e probe uma unica vez (materializado, reaproveitando o que
    ja existe) e divide o probe deterministicamente em validacao/teste.
    """
    common = dict(kwargs)
    common.update(
        {
            "min_faces_per_person": int(min_photos_per_class),
            "n_identities": int(n_classes),
            "max_per_identity": int(sample_size),
            "rematerialize": bool(rematerialize),
            "force": bool(force),
        }
    )
    gallery_loader_raw, probe_loader_raw = build_private_loaders(root, **common)
    gallery_samples = gallery_loader_raw.load()
    probe_samples = probe_loader_raw.load()

    validation_samples, test_samples = _split_probe_samples(
        probe_samples, validation_ratio=float(validation_ratio), test_ratio=float(test_ratio)
    )

    return (
        _ListDataloader(gallery_samples),
        _ListDataloader(validation_samples),
        _ListDataloader(test_samples),
    )

__all__ = [
    "PrivateDataloader",
    "build_private_loaders",
    "build_private_splits",
    "_ListDataloader",
    "split_probe"
]
