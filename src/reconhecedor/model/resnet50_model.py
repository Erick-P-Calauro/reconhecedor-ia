"""Model ResNet50 (torchvision) como extrator de embeddings (ADR 001 / ADR 003).

Implementacao **generica** da camada Model: usa a ResNet50 pre-treinada em
ImageNet (pesos ``torchvision``) sem a camada de classificacao. O vetor de 2048
dimensoes do pooling global e usado como embedding facial. Reutilizavel por
qualquer experimento cujo ``face_recognition_model`` seja ``resnet50`` (ADR 002,
ADR 003), sem alterar o pipeline (Spec 003).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from .base import Model
from .torch_utils import to_batch

#: Pesos ImageNet usados por padrao.
DEFAULT_WEIGHTS = ResNet50_Weights.IMAGENET1K_V2


class ResNet50Model(Model):
    """ResNet50 (ImageNet) como extrator de caracteristicas faciais."""

    def __init__(self, weights: ResNet50_Weights = DEFAULT_WEIGHTS, device: str | None = None) -> None:
        print("[Model] inicializando")
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        network = resnet50(weights=weights, progress=False)
        network.fc = nn.Identity()  # descarta a classificacao: mantem as features
        network.eval()
        self._network = network.to(self._device)
        print("[Model] inicializado")

    @property
    def name(self) -> str:
        return "resnet50"

    @property
    def output_dim(self) -> int:
        return 2048

    @property
    def is_fine_tunable(self) -> bool:
        return True

    def embed(self, image: np.ndarray) -> np.ndarray:
        batch = to_batch(image, self._device)
        with torch.no_grad():
            features = self._network(batch)
        return features.reshape(-1).cpu().numpy().astype(np.float32)

    def fine_tune(self, train_samples, *, detector, preprocessor, crop_padding, epochs=5, lr=1e-4) -> None:
        print("[Model] fine-tuning iniciado")
        images: list[np.ndarray] = []
        labels: list[int] = []
        label_to_index: dict[str, int] = {}

        for sample in train_samples:
            detection = detector.detect(sample.image)
            if detection is None:
                continue
            face = detector.crop(sample.image, detection, padding=crop_padding)
            if face is None or (hasattr(face, "size") and face.size == 0):
                continue
            processed = preprocessor.process(face)
            if sample.label not in label_to_index:
                label_to_index[sample.label] = len(label_to_index)
            images.append(np.asarray(processed, dtype=np.float32))
            labels.append(label_to_index[sample.label])

        if not images or len(set(labels)) < 2:
            print("[Model] fine-tuning sem dados suficientes; ignorado")
            return

        x = torch.from_numpy(np.stack(images)).to(self._device)
        y = torch.tensor(labels, dtype=torch.long, device=self._device)

        network = self._network
        n_classes = len(label_to_index)
        network.fc = nn.Linear(2048, n_classes).to(self._device)

        for param in network.parameters():
            param.requires_grad = False
        for param in network.layer4.parameters():
            param.requires_grad = True
        for param in network.fc.parameters():
            param.requires_grad = True

        network.train()
        optimizer = torch.optim.Adam(
            (p for p in network.parameters() if p.requires_grad), lr=float(lr)
        )
        criterion = nn.CrossEntropyLoss()

        batch_size = 8
        n = x.shape[0]
        for epoch in range(int(epochs)):
            permutation = torch.randperm(n, device=self._device)
            for start in range(0, n, batch_size):
                indices = permutation[start : start + batch_size]
                xb = x[indices]
                yb = y[indices]
                optimizer.zero_grad()
                logits = network(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
            print(f"[Model] fine-tuning epoch {epoch + 1}/{epochs}")

        network.fc = nn.Identity()
        for param in network.parameters():
            param.requires_grad = False
        network.eval()
        print("[Model] fine-tuning concluido")