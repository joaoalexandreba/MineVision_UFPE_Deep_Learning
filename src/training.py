from pathlib import Path
from typing import Optional

from src.config import RESULTS_DIR, SEED
from src.models import get_yolo_model


def train_model(
    data_yaml_path: Path,
    results_dir: Path = RESULTS_DIR,
    model_name: str = "yolo11n-seg.pt",
    epochs: int = 50,
    batch_size: int = 8,
    imgsz: int = 640,
    seed: int = SEED,
) -> Path:
    """
    Executa o treinamento do YOLOv11 com PyTorch e aceleração por GPU (se disponível).

    Args:
        data_yaml_path: Caminho do arquivo data.yaml.
        results_dir: Diretório onde serão gravados os resultados do treino.
        model_name: Checkpoint do modelo (padrão 'yolo11n-seg.pt').
        epochs: Quantidade de épocas de treinamento.
        batch_size: Tamanho do lote.
        imgsz: Resolução das imagens.
        seed: Semente pseudo-aleatória.

    Returns:
        Path: Caminho dos melhores pesos salvos ('best.pt').
    """
    print(f"[5/6] Iniciando treinamento do YOLOv11 ({epochs} épocas, batch={batch_size})...")
    
    model = get_yolo_model(model_name)
    
    resultados = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        seed=seed,
        project=str(results_dir),
        name="treino_yolov11_harsh",
        exist_ok=True,
        verbose=True,
    )

    best_weights = results_dir / "treino_yolov11_harsh" / "weights" / "best.pt"
    print(f"  Treinamento concluído com sucesso!")
    print(f"  Melhor modelo salvo em: {best_weights}")
    return best_weights
