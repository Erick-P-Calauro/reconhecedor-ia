"""Dataloader proprio do dataset LFW (ADR 004).

Cada dataset tem um dataloader proprio, mesmo que materializado (ADR 004). O LFW
ja era materializado em ``data/lfw`` (ADR 003), mas passa a ter este dataloader
dedicado, que:

- modo **materializado** (padrao): carrega de ``data/lfw/{gallery,probe}``,
  delegando ao ``FolderDataloader`` (comportamento anterior);
- modo **direto** (``direct=True``): busca via ``sklearn.datasets.fetch_lfw_people``
  e monta ``Sample``s em memoria, sem escrever arquivos;
- **re-materializacao** via ``rematerialize``/``force``: chama
  ``lfw_dataset.prepare_lfw(...)`` com os parametros informados.

Localizacao: vive no pacote da camada Dataloader (``src/reconhecedor/dataloader/``),
como dataloader especifico do LFW (ADR-004), ao lado de ``FolderDataloader``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import Dataloader
from .folder_dataloader import FolderDataloader
from ..types import Sample

from . import lfw_dataset

_DEFAULT_MIN_FACES = 4
_DEFAULT_IDENTITIES = 15
_DEFAULT_KNOWN = 7
_DEFAULT_GALLERY_RATIO = 0.7
_DEFAULT_MAX_PER_IDENTITY = 25
_DEFAULT_RESIZE = 1.0


def _sanitize(name: str) -> str:
    return lfw_dataset._sanitize(name)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(image, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    # sklearn entrega RGB; converte para BGR (convencao OpenCV/Sample).
    return array[:, :, ::-1].copy()


def _fetch_samples(
    *,
    min_faces_per_person: int,
    n_identities: int,
    n_known: int,
    gallery_ratio: float,
    max_per_identity: int,
    resize: float,
    data_home: str | Path | None,
) -> dict[str, list[Sample]]:
    from sklearn.datasets import fetch_lfw_people

    root_proxy = Path("lfw-direct")
    if data_home is None:
        data_home = lfw_dataset.resolve_data_home(None)

    print("[Dataloader] buscando LFW em memoria (modo direto)")
    bunch = fetch_lfw_people(
        data_home=str(data_home),
        funneled=True,
        resize=resize,
        min_faces_per_person=min_faces_per_person,
        color=True,
        slice_=lfw_dataset.FULL_SLICE,
    )
    images = bunch.images
    target = bunch.target
    target_names = bunch.target_names

    counts = np.bincount(target, minlength=len(target_names))
    order = [int(index) for index in np.argsort(-counts) if counts[index] >= min_faces_per_person]
    order = order[:n_identities]

    gallery_samples: list[Sample] = []
    probe_samples: list[Sample] = []

    for rank, id_index in enumerate(order):
        label = _sanitize(str(target_names[id_index]))
        sample_indices = np.where(target == id_index)[0][:max_per_identity]
        known = rank < n_known and len(sample_indices) >= 2
        n_gallery = (
            max(1, min(int(round(gallery_ratio * len(sample_indices))), len(sample_indices) - 1))
            if known
            else 0
        )
        gallery_indices = sample_indices[:n_gallery]
        probe_indices = sample_indices[n_gallery:]

        for position, sample_index in enumerate(gallery_indices):
            image = _to_bgr(images[sample_index])
            gallery_samples.append(
                Sample(path=root_proxy / "gallery" / label / f"{position:04d}.png", image=image, label=label)
            )
        for position, sample_index in enumerate(probe_indices):
            image = _to_bgr(images[sample_index])
            probe_samples.append(
                Sample(path=root_proxy / "probe" / label / f"{position:04d}.png", image=image, label=label)
            )

    print(f"[Dataloader] modo direto: {len(gallery_samples)} gallery / {len(probe_samples)} probe")
    return {"gallery": gallery_samples, "probe": probe_samples}


class LFWDataloader(Dataloader):
    """Dataloader proprio do LFW (ADR 004).

    ``split`` indica qual particao carregar (``gallery`` ou ``probe``).
    """

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
        self.root = Path(root) if root is not None else Path(lfw_dataset.DEFAULT_DATASET)
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
        print(f"[Dataloader] carregando: {self.root} ({self.split})")
        if self.direct:
            samples = _fetch_samples(
                data_home=self.data_home,
                **self._params,
            )[self.split]
            print(f"[Dataloader] {len(samples)} samples carregados")
            return samples

        # materializado
        kwargs = self._prepare_kwargs()
        lfw_dataset.ensure_dataset(
            self.root,
            rematerialize=self.rematerialize,
            force=self.force,
            **kwargs,
        )
        folder = self.root / self.split
        samples = FolderDataloader(folder).load()
        print(f"[Dataloader] {len(samples)} samples carregados")
        return samples


def build_lfw_loaders(
    root: str | Path | None = None,
    *,
    direct: bool = False,
    mode: str | None = None,
    rematerialize: bool = False,
    force: bool = False,
    **kwargs: Any,
) -> tuple[LFWDataloader, LFWDataloader]:
    """Factory que retorna (gallery_loader, probe_loader)."""
    common = dict(
        root=root,
        direct=direct or (mode == "direct"),
        mode=mode,
        rematerialize=rematerialize,
        force=force,
        **kwargs,
    )
    gallery = LFWDataloader(split="gallery", **common)
    probe = LFWDataloader(split="probe", **common)
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


# Aliases para compatibilidade com possiveis nomes de testes.
split_probe_samples = _split_probe_samples
split_probe = _split_probe_samples
split_lfw_probe = _split_probe_samples


def build_lfw_splits(
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
    gallery_loader_raw, probe_loader_raw = build_lfw_loaders(root, **common)
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
    "LFWDataloader",
    "build_lfw_loaders",
    "build_lfw_splits",
    "_ListDataloader",
    "_split_probe_samples",
    "split_probe_samples",
    "split_probe",
    "split_lfw_probe",
]
