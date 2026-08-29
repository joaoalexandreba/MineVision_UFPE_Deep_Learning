from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.config import RESULTS_DIR


def predict_image(
    image_path: Union[str, Path],
    model_weights: Union[str, Path],
    output_dir: Optional[Path] = None,
    conf_threshold: float = 0.25,
) -> Path:
    """
    Executa inferência de detecção/segmentação em uma nova imagem avulsa (RF05).

    Args:
        image_path: Caminho para a imagem de entrada.
        model_weights: Caminho dos pesos do modelo (ex: 'best.pt' ou 'yolo11n-seg.pt').
        output_dir: Diretório para salvar a imagem com as predições desenhadas.
        conf_threshold: Limiar de confiança para detecção.

    Returns:
        Path: Caminho da imagem salva com as predições anotadas.
    """
    img_path = Path(image_path)
    weights_path = Path(model_weights)
    
    if not img_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Arquivo de pesos não encontrado: {weights_path}")

    if output_dir is None:
        output_dir = RESULTS_DIR / "inferencias"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Inferência] Processando imagem: {img_path.name}...")
    model = YOLO(str(weights_path))
    results = model.predict(source=str(img_path), conf=conf_threshold, save=False, verbose=False)

    # Plotar os resultados na imagem
    result = results[0]
    im_array = result.plot()  # Retorna array BGR/RGB com boxes e máscaras desenhadas
    im = Image.fromarray(im_array[..., ::-1])  # Converte BGR para RGB PIL

    out_path = output_dir / f"pred_{img_path.stem}.jpg"
    im.save(out_path)
    print(f"  Resultado salvo com sucesso em: {out_path}")
    return out_path
