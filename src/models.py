from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from ultralytics import YOLO

from src.config import CLASSES


class HarshNeuralFeatureExtractor(nn.Module):
    """
    Módulo de Rede Neural PyTorch nativo (nn.Module) para extração de features
    e classificação auxiliar da qualidade da carga (bed / coal).
    Demonstra a conformidade e integração com o framework profundo PyTorch.
    """

    def __init__(self, num_classes: int = len(CLASSES), in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa a passagem direta (forward pass) do tensor de entrada."""
        feats = self.features(x)
        return self.classifier(feats)


def get_yolo_model(model_name: str = "yolo11n-seg.pt") -> YOLO:
    """
    Instancia o modelo YOLOv11 de segmentação/detecção.
    O YOLOv11 é construído nativamente em PyTorch (torch.nn.Module).

    Args:
        model_name: Nome do checkpoint (ex: 'yolo11n-seg.pt', 'yolo11s-seg.pt').

    Returns:
        YOLO: Instância do modelo carregado.
    """
    print(f"  Instanciando modelo YOLOv11 ({model_name})...")
    model = YOLO(model_name)
    return model
