"""Preparador do dataset LFW via Scikit-learn (ADR 003).

Baixa o *Labeled Faces in the Wild* com ``sklearn.datasets.fetch_lfw_people`` e
**materializa** os dados no layout esperado pela camada Dataloader
(``FolderDataloader`` / base do sistema):

    <dataset>/gallery/<identidade>/*.png   (pessoas cadastradas  - Spec 003)
    <dataset>/probe/<identidade>/*.png     (sondas/reconhecimento - Spec 003)

Recorte adotado (documentado no log 009): identidades com mais imagens no LFW,
sendo as ``n_known`` primeiras divididas entre galeria e probe (pessoas
conhecidas) e as demais usadas apenas em probe (pessoas **desconhecidas**, para
medir FAR/FRR - Spec 003 / Spec 004).

E o preparador/artefato do dataloader especifico do LFW (ADR-004), que vive no
pacote da camada Dataloader (``src/reconhecedor/dataloader/``), compartilhado
pelos experimentos do ADR-003; depende de ``sklearn.datasets.fetch_lfw_people``
(import lazy) e materializa o layout gallery/probe.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

#: Dataset padrao dos experimentos do ADR 003 (gitignored em ``data/``).
DEFAULT_DATASET = "data/lfw"

#: Imagens completas (250x250) do LFW, sem o recorte padrao do sklearn.
FULL_SLICE = (slice(0, 250), slice(0, 250))

DEFAULT_MIN_FACES = 4
DEFAULT_IDENTITIES = 15
DEFAULT_KNOWN = 7
DEFAULT_GALLERY_RATIO = 0.7
DEFAULT_MAX_PER_IDENTITY = 25
DEFAULT_RESIZE = 1.0

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_INVALID_PATH_CHARS = set('<>:"/\\|?*')


def _sanitize(name: str) -> str:
    """Torna o nome da identidade seguro como nome de pasta no filesystem."""
    cleaned = "".join("_" if char in _INVALID_PATH_CHARS else char for char in name).strip()
    return cleaned or "identidade"


def _write_image(path: Path, image: np.ndarray) -> None:
    """Grava uma imagem RGB float [0, 1] (saida do sklearn) como PNG uint8 BGR."""
    import cv2

    array = np.clip(np.asarray(image, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)


def _has_images(path: Path) -> bool:
    return path.is_dir() and any(
        entry.is_file() and entry.suffix.lower() in _IMAGE_SUFFIXES for entry in path.rglob("*")
    )


def resolve_data_home(root: str | Path | None = None, data_home: str | Path | None = None) -> str | Path:
    """Escolhe um cache do sklearn que realmente contenha o LFW extraido.

    Prefere um cache existente com pastas de pessoas; caso contrario, usa
    ``root.parent`` (comportamento original) e remove pastas vazias para
    permitir novo download.
    """
    if data_home is not None:
        return data_home
    if root is None:
        root = DEFAULT_DATASET
    root = Path(root)
    candidates = [root.parent, root.parent / "lfw_home"]
    for cand in candidates:
        funneled = cand / "lfw_home" / "lfw_funneled"
        if funneled.is_dir() and any(p.is_dir() for p in funneled.iterdir()):
            return cand
    cand = root.parent
    funneled = cand / "lfw_home" / "lfw_funneled"
    if funneled.exists() and not any(funneled.iterdir()):
        shutil.rmtree(funneled, ignore_errors=True)
    return cand


def prepare_lfw(
    dataset_root: str | Path = DEFAULT_DATASET,
    *,
    data_home: str | Path | None = None,
    min_faces_per_person: int = DEFAULT_MIN_FACES,
    n_identities: int = DEFAULT_IDENTITIES,
    n_known: int = DEFAULT_KNOWN,
    gallery_ratio: float = DEFAULT_GALLERY_RATIO,
    max_per_identity: int = DEFAULT_MAX_PER_IDENTITY,
    resize: float = DEFAULT_RESIZE,
) -> dict:
    """Materializa o LFW (via Scikit-learn) no layout ``gallery``/``probe``.

    As ``n_known`` identidades mais representadas viram pessoas conhecidas
    (``gallery_ratio`` das imagens na galeria e o restante em probe); as demais
    identidades entram apenas em probe, representando pessoas **desconhecidas**.
    Retorna um resumo (tambem gravado em ``<dataset_root>/dataset_info.json``).
    """
    from sklearn.datasets import fetch_lfw_people

    root = Path(dataset_root)
    if data_home is None:
        data_home = resolve_data_home(root)

    print(f"[Dataloader] buscando LFW via sklearn (min_faces={min_faces_per_person}, identidades={n_identities})")
    bunch = fetch_lfw_people(
        data_home=str(data_home),
        funneled=True,
        resize=resize,
        min_faces_per_person=min_faces_per_person,
        color=True,
        slice_=FULL_SLICE,
    )
    images = bunch.images
    target = bunch.target
    target_names = bunch.target_names

    counts = np.bincount(target, minlength=len(target_names))
    order = [int(index) for index in np.argsort(-counts) if counts[index] >= min_faces_per_person]
    order = order[:n_identities]

    gallery_root = root / "gallery"
    probe_root = root / "probe"

    summary: dict = {
        "dataset_root": str(root),
        "source": "sklearn.datasets.fetch_lfw_people",
        "min_faces_per_person": int(min_faces_per_person),
        "n_identities": len(order),
        "n_known": int(min(n_known, len(order))),
        "gallery_ratio": float(gallery_ratio),
        "max_per_identity": int(max_per_identity),
        "identities": [],
        "known": [],
        "unknown": [],
        "gallery_images": 0,
        "probe_images": 0,
    }

    for rank, id_index in enumerate(order):
        label = _sanitize(str(target_names[id_index]))
        sample_indices = np.where(target == id_index)[0][:max_per_identity]
        known = rank < n_known and len(sample_indices) >= 2
        n_gallery = max(1, min(int(round(gallery_ratio * len(sample_indices))), len(sample_indices) - 1)) if known else 0
        gallery_indices = sample_indices[:n_gallery]
        probe_indices = sample_indices[n_gallery:]

        for position, sample_index in enumerate(gallery_indices):
            _write_image(gallery_root / label / f"{position:04d}.png", images[sample_index])
        for position, sample_index in enumerate(probe_indices):
            _write_image(probe_root / label / f"{position:04d}.png", images[sample_index])

        entry = {
            "label": label,
            "available": int(counts[id_index]),
            "gallery": int(len(gallery_indices)),
            "probe": int(len(probe_indices)),
        }
        summary["identities"].append(entry)
        summary["known" if known else "unknown"].append(label)
        summary["gallery_images"] += len(gallery_indices)
        summary["probe_images"] += len(probe_indices)

    (root / "dataset_info.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary




def ensure_dataset(
    dataset_root: str | Path = DEFAULT_DATASET,
    *,
    rematerialize: bool = False,
    force: bool = False,
    **kwargs,
) -> dict:
    """Materializa o LFW se ainda nao existir; caso contrario, devolve o resumo.

    Com ``rematerialize``/``force`` True, re-materializa sempre (ADR 004).
    ``**kwargs`` sao repassados a ``prepare_lfw``.
    """
    root = Path(dataset_root)
    info_path = root / "dataset_info.json"
    if rematerialize or force:
        print(f"[Dataloader] re-materializando dataset em {root}")
        summary = prepare_lfw(root, **kwargs)
        print(f"[Dataloader] re-materializacao concluida: {summary.get('gallery_images')} gallery / {summary.get('probe_images')} probe")
        return summary
    if _has_images(root / "gallery") and _has_images(root / "probe"):
        if info_path.is_file():
            return json.loads(info_path.read_text(encoding="utf-8"))
        return {"dataset_root": str(root), "status": "ja_materializado"}
    print(f"[Dataloader] materializando dataset em {root}")
    summary = prepare_lfw(root, **kwargs)
    print(f"[Dataloader] materializacao concluida: {summary.get('gallery_images')} gallery / {summary.get('probe_images')} probe")
    return summary
