import argparse
import sys
from pathlib import Path

from src.config import CLASSES, DOWNLOAD_DIR, RESULTS_DIR, SEED, YOLO_DIR, set_seed
from src.data import load_dataset
from src.evaluation import evaluate_model
from src.inference import predict_image
from src.preprocessing import prepare_yolo_structure, preprocess_data, split_dataset
from src.training import train_model


def run_pipeline(
    epochs: int = 50,
    batch_size: int = 8,
    skip_download: bool = False,
    skip_train: bool = False,
    infer_image: str = "",
) -> None:
    """
    Orquestrador principal do pipeline de Visão Computacional / YOLOv11 & PyTorch.
    Integrando: Carregamento -> Pré-processamento -> Treino -> Avaliação -> Inferência.
    """
    set_seed(SEED)
    print("=" * 75)
    print("🚛 Pipeline de Visão Computacional - YOLOv11 & PyTorch (Dataset Harsh)")
    print("=" * 75)

    # 1. Carregamento / Download dos dados
    if skip_download:
        raiz_dataset = DOWNLOAD_DIR
        data_yaml_path = YOLO_DIR / "data.yaml"
        print(f"[1/6] Reutilizando dados locais em: {raiz_dataset}")
    else:
        raiz_dataset, _ = load_dataset()

    # 2. Pré-processamento & Dataset PyTorch
    df = preprocess_data(raiz_dataset)

    # 3. Divisão Estratificada (80% Treino / 20% Teste)
    df_train, df_test = split_dataset(df, train_ratio=0.8, seed=SEED)

    # 4. Estruturação YOLO e data.yaml
    data_yaml_path = prepare_yolo_structure(df_train, df_test, yolo_dir=YOLO_DIR, classes=CLASSES)

    # 5. Treinamento
    best_weights = RESULTS_DIR / "treino_yolov11_harsh" / "weights" / "best.pt"
    if skip_train and best_weights.exists():
        print(f"[5/6] Pulando treino. Utilizando melhor modelo existente em: {best_weights}")
    else:
        best_weights = train_model(
            data_yaml_path=data_yaml_path,
            results_dir=RESULTS_DIR,
            model_name="yolo11n-seg.pt",
            epochs=epochs,
            batch_size=batch_size,
            seed=SEED,
        )

    # 6. Avaliação Estatística (Métricas, IC 95% t-Student, Gráficos)
    df_metricas, df_stats = evaluate_model(
        best_weights=best_weights,
        data_yaml_path=data_yaml_path,
        results_dir=RESULTS_DIR,
        classes=CLASSES,
        seed=SEED,
    )

    # 7. Inferência para nova imagem (se fornecida)
    if infer_image:
        predict_image(image_path=infer_image, model_weights=best_weights)

    print("\n" + "=" * 75)
    print("✅ Pipeline executado com sucesso!")
    print(f"📁 Resultados salvos em: {RESULTS_DIR.resolve()}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Visão Computacional com YOLOv11 & PyTorch (Trabalho Final)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas de treino (padrão: 50)")
    parser.add_argument("--batch-size", type=int, default=8, help="Tamanho do batch (padrão: 8)")
    parser.add_argument("--skip-download", action="store_true", help="Pula download e usa dados locais existentes")
    parser.add_argument("--skip-train", action="store_true", help="Pula treino e executa avaliação no best.pt existente")
    parser.add_argument("--infer", type=str, default="", help="Caminho de uma imagem para realizar inferência (RF05)")

    args = parser.parse_args()

    run_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_download=args.skip_download,
        skip_train=args.skip_train,
        infer_image=args.infer,
    )


if __name__ == "__main__":
    main()
