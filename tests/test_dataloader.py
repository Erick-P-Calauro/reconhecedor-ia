"""Testes do Dataloader: formatos aceitos e layout por pastas (Spec 002/003)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from reconhecedor.dataloader import FolderDataloader


def _write(path: Path, ext: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((24, 24, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(path.with_suffix(ext)), image)


def test_loads_png_jpg_jpeg_and_labels_by_folder(tmp_path: Path):
    """Spec 002 RNF-01 (.png/.jpg/.jpeg) + RF-01 (identidade da pasta)."""
    _write(tmp_path / "alice" / "a1", ".png")
    _write(tmp_path / "alice" / "a2", ".jpg")
    _write(tmp_path / "bob" / "b1", ".jpeg")
    _write(tmp_path / "bob" / "ignorado", ".bmp")  # formato fora do escopo

    samples = FolderDataloader(tmp_path).load()

    assert len(samples) == 3
    labels = sorted(sample.label for sample in samples)
    assert labels == ["alice", "alice", "bob"]
    assert all(sample.image.ndim == 3 for sample in samples)
    assert {sample.path.suffix for sample in samples} == {".png", ".jpg", ".jpeg"}


def test_skips_corrupt_files(tmp_path: Path):
    _write(tmp_path / "alice" / "ok", ".png")
    broken = tmp_path / "alice" / "quebrada.png"
    broken.write_bytes(b"isto nao e uma imagem")

    samples = FolderDataloader(tmp_path).load()

    assert [sample.path.name for sample in samples] == ["ok.png"]


def test_missing_root_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        FolderDataloader(tmp_path / "nao-existe").load()


def test_empty_root_returns_no_samples(tmp_path: Path):
    assert FolderDataloader(tmp_path).load() == []
