"""Camada Model (ADR 001): geracao dos embeddings faciais.

Exporta a interface ``Model`` e as implementacoes concretas **genericas**
(reutilizaveis por experimentos, ADR 002): ``ResNet50Model`` (torchvision) e
``FaceNetModel`` (FaceNet/Inception-ResNet-v1 VGGFace2 via ``facenet-pytorch``).
"""

from .base import Model
from .facenet_model import FaceNetModel
from .resnet50_model import ResNet50Model

__all__ = ["Model", "ResNet50Model", "FaceNetModel"]
