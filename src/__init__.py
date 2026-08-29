"""
Pacote src - Engenharia de Software para IA e Frameworks Profundos
Pipeline modularizado para detecção e segmentação de carga em caminhões com YOLOv11 & PyTorch.
"""

from src.config import (
    API_KEY,
    BASE_DIR,
    CLASSES,
    CLASS_NAMES,
    DATA_DIR,
    DOWNLOAD_DIR,
    RESULTS_DIR,
    SEED,
    SPLIT_DIR,
    YOLO_DIR,
)

__all__ = [
    "SEED",
    "API_KEY",
    "CLASSES",
    "CLASS_NAMES",
    "BASE_DIR",
    "DATA_DIR",
    "DOWNLOAD_DIR",
    "YOLO_DIR",
    "SPLIT_DIR",
    "RESULTS_DIR",
]
