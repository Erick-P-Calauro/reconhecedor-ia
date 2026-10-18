"""Montagem do dataset demo (imagens publicas) em ``data/demo``.

Ferramenta de preparacao de dataset do **experimento baseline** (ADR 002):
e a implementacao por tras do subcomando ``build-demo`` do ``main.py`` e o
dataset que ela monta e exatamente o declarado em
``src/experiments/baseline/config.json``. Como faz parte da preparacao do
experimento, vive na subpasta dedicada dele e nao em ``src/reconhecedor/``
(pasta base das camadas — ADR 002, ultima caracteristica). As imagens sao de
fontes publicas (scikit-image e amostras do proprio OpenCV) e sao baixadas
para ``data/demo``; nada aqui e dado privado de participantes. A pasta
``data/`` nao e versionada (Spec 003 RNF-02).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from reconhecedor.dataloader.folder_dataloader import DEFAULT_EXTENSIONS

URLS = {
    "astronaut.png": (
        "https://raw.githubusercontent.com/scikit-image/scikit-image/"
        "v0.19.0/skimage/data/astronaut.png"
    ),
    "lena.jpg": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg",
    "coffee.png": (
        "https://raw.githubusercontent.com/scikit-image/scikit-image/"
        "v0.19.0/skimage/data/coffee.png"
    ),
    "chelsea.png": (
        "https://raw.githubusercontent.com/scikit-image/scikit-image/"
        "v0.19.0/skimage/data/chelsea.png"
    ),
}


def _download(url: str) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=60) as response:
        data = np.frombuffer(response.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"falha ao decodificar imagem publica: {url}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    """Grava imagem de forma segura para caminhos com acento no Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower()
    if extension not in DEFAULT_EXTENSIONS:
        raise ValueError(f"formato nao suportado: {extension}")
    ok, buffer = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"falha ao codificar imagem: {path}")
    buffer.tofile(str(path))


def _brighter(image: np.ndarray, factor: float = 1.3) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _noisier(image: np.ndarray, sigma: float = 8.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _blurred(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (5, 5), 0)


def build_demo_dataset(dest: str | Path = Path("data/demo")) -> Path:
    """Cria ``data/demo`` com galeria, probe e imagens sem face.

    Layout gerado::

        data/demo/
          gallery/astronaut/001.png     # pessoa cadastrada (RF-03)
          probe/astronaut/{brilho,ruido,desfoque}.png   # genuine
          probe/lena/001.jpg             # fora do dominio -> desconhecido (RF-04)
          probe/_no_face/{coffee,chelsea}.png            # sem face -> descarte (RF-02)
    """
    dest = Path(dest)
    if (dest / "gallery").is_dir() and (dest / "probe").is_dir():
        print(f"[demo] dataset ja existe em {dest} (nenhum download necessario)")
        return dest

    print("[demo] baixando imagens publicas...")
    images = {name: _download(url) for name, url in URLS.items()}

    astronaut = images["astronaut.png"]
    _write_image(dest / "gallery" / "astronaut" / "001.png", astronaut)
    _write_image(dest / "probe" / "astronaut" / "brilho.png", _brighter(astronaut))
    _write_image(dest / "probe" / "astronaut" / "ruido.png", _noisier(astronaut))
    _write_image(dest / "probe" / "astronaut" / "desfoque.png", _blurred(astronaut))
    _write_image(dest / "probe" / "lena" / "001.jpg", images["lena.jpg"])
    _write_image(dest / "probe" / "_no_face" / "coffee.png", images["coffee.png"])
    _write_image(dest / "probe" / "_no_face" / "chelsea.png", images["chelsea.png"])

    print(f"[demo] dataset criado em {dest.resolve()}")
    return dest
