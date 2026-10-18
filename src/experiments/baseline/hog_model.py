"""Model do experimento baseline: HOG classico com numpy (sem DL).

Implementacao da camada ``Model`` (ADR 001) usada **por este experimento**
(``face_recognition_model: "baseline-hog"`` no ``config.json``); portanto fica
na subpasta dedicada do experimento e nao em ``src/reconhecedor/model/``
(pasta base — ADR 002, ultima caracteristica). Mantem a interface ``Model``
pronta para plugar redes neurais depois (Spec 003 RNF-01): basta implementar
``Model.embed`` em uma nova classe, na base ou em outro experimento.
"""

from __future__ import annotations

import numpy as np

from reconhecedor.model import Model


class BaselineHogModel(Model):
    """Embedding baseado em HOG leve + histograma de intensidade.

    Caracteristicas:
    - gradientes da face pre-processada (escala de cinza em [0, 1]);
    - histogramas de orientacao por celula (grade 8x8), normalizados por celula;
    - histograma global de intensidade;
    - vetor final normalizado em L2 (comparacao por cosseno na camada
      FaceRecognition).
    """

    def __init__(self, cells: int = 8, orientations: int = 9, intensity_bins: int = 16) -> None:
        if cells <= 0 or orientations <= 0 or intensity_bins <= 0:
            raise ValueError("cells, orientations e intensity_bins devem ser positivos")
        self.cells = int(cells)
        self.orientations = int(orientations)
        self.intensity_bins = int(intensity_bins)

    @property
    def name(self) -> str:
        return "baseline-hog"

    @property
    def output_dim(self) -> int:
        return self.cells * self.cells * self.orientations + self.intensity_bins

    def embed(self, image: np.ndarray) -> np.ndarray:
        gray = np.asarray(image, dtype=np.float32)
        if gray.ndim == 3:
            gray = gray.mean(axis=2)
        if gray.ndim != 2 or gray.size == 0:
            raise ValueError(f"entrada invalida para a Model: shape={gray.shape}")

        magnitude, orientation = self._gradients(gray)
        features = self._cell_orientation_histograms(magnitude, orientation)
        features = np.concatenate([features, self._intensity_histogram(gray)])

        norm = float(np.linalg.norm(features))
        if norm > 0:
            features = features / norm
        return features.astype(np.float32)

    def _gradients(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        magnitude = np.hypot(grad_x, grad_y)
        # Orientacao sem sinal em [0, 180) graus.
        orientation = np.degrees(np.arctan2(grad_y, grad_x)) % 180.0
        return magnitude, orientation

    def _cell_orientation_histograms(self, magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        height, width = orientation.shape
        cell_h = max(height // self.cells, 1)
        cell_w = max(width // self.cells, 1)
        bin_width = 180.0 / self.orientations
        bin_index = np.clip((orientation / bin_width).astype(np.int32), 0, self.orientations - 1)

        histograms: list[np.ndarray] = []
        for row in range(self.cells):
            for col in range(self.cells):
                ys = slice(row * cell_h, min((row + 1) * cell_h, height))
                xs = slice(col * cell_w, min((col + 1) * cell_w, width))
                counts = np.bincount(
                    bin_index[ys, xs].ravel(),
                    weights=magnitude[ys, xs].ravel(),
                    minlength=self.orientations,
                ).astype(np.float32)
                norm = float(np.linalg.norm(counts))
                if norm > 0:
                    counts /= norm
                histograms.append(counts)
        return np.concatenate(histograms)

    def _intensity_histogram(self, gray: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(gray, bins=self.intensity_bins, range=(0.0, 1.0))
        counts = counts.astype(np.float32)
        total = float(counts.sum())
        if total > 0:
            counts /= total
        return counts
