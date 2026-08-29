import random
from pathlib import Path
import numpy as np
import torch

# =============================================================================
# Configurações Globais e Reprodutibilidade
# =============================================================================
SEED = 42

def set_seed(seed: int = SEED) -> None:
    """Configura sementes para garantir reprodutibilidade experimental."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# Credenciais e Projeto Roboflow
# =============================================================================
API_KEY = "Md5T3krCkUBImQf9Df53"
WORKSPACE = "rapis-soft-technologies"
PROJECT = "harsh-0i6vw"
VERSION = 1

# =============================================================================
# Estrutura de Diretórios
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOWNLOAD_DIR = DATA_DIR / "harsh_dataset"
YOLO_DIR = DATA_DIR / "yolo_harsh"
SPLIT_DIR = DATA_DIR / "splits"
RESULTS_DIR = BASE_DIR / "resultados"
DOCS_DIR = BASE_DIR / "docs"

# Garantir criação de diretórios padrão
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
YOLO_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Classes do Domínio
# =============================================================================
CLASSES = ["bed", "coal"]
CLASS_NAMES = {0: "bed", 1: "coal"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
