import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import CLASSES, SEED, SPLIT_DIR, YOLO_DIR


class HarshDataset(Dataset):
    """
    Classe customizada PyTorch Dataset para carregar imagens e labels.
    Converte as imagens em tensores PyTorch (torch.Tensor) e aplica transformações.
    """

    def __init__(self, df: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        img_path = self.df.iloc[idx]["image_path"]
        label_path = self.df.iloc[idx]["label_path"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label_path


def get_transforms() -> transforms.Compose:
    """Retorna o pipeline padrão de transformações e normalização PyTorch."""
    return transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def preprocess(raiz_dataset: Path) -> pd.DataFrame:
    """
    Varre o dataset, extrai anotações de labels, mapeia amostras e instancia o HarshDataset (PyTorch).

    Args:
        raiz_dataset: Diretório contendo as pastas train/valid/test.

    Returns:
        pd.DataFrame: Metadados consolidados de todas as imagens.
    """
    print("[2/6] Pré-processamento - Mapeando amostras e preparando para PyTorch...")
    registros = []
    padroes_img = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]

    for split_name in ["train", "valid", "test", "val"]:
        img_dir = raiz_dataset / split_name / "images"
        lbl_dir = raiz_dataset / split_name / "labels"
        if not img_dir.exists():
            continue

        for padrao in padroes_img:
            for img_path in img_dir.glob(padrao):
                stem = img_path.stem
                label_path = lbl_dir / f"{stem}.txt"

                classes_presentes = set()
                if label_path.exists():
                    with open(label_path, "r", encoding="utf-8") as f:
                        for linha in f:
                            linha = linha.strip()
                            if linha:
                                partes = linha.split()
                                if partes and partes[0].isdigit():
                                    classes_presentes.add(int(partes[0]))

                strat_key = "_".join(sorted([str(c) for c in classes_presentes]))
                if not strat_key:
                    strat_key = "sem_classe"

                registros.append({
                    "image_path": str(img_path),
                    "label_path": str(label_path),
                    "split_original": split_name,
                    "classes_presentes": sorted(classes_presentes),
                    "n_classes": len(classes_presentes),
                    "strat_key": strat_key,
                })

    df = pd.DataFrame(registros)
    transformacoes = get_transforms()
    pytorch_dataset = HarshDataset(df=df, transform=transformacoes)
    print(f"  [PyTorch] HarshDataset instanciado: {len(pytorch_dataset)} amostras em formato Tensor.")
    print(f"  Total de amostras: {len(df)}")
    return df


preprocess_data = preprocess  # Alias de compatibilidade


def split_dataset(
    df: pd.DataFrame,
    split_dir: Path = SPLIT_DIR,
    train_ratio: float = 0.8,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Realiza a divisão estratificada 80/20 (treino/teste) usando NumPy com semente fixa.

    Args:
        df: DataFrame com as amostras e a coluna 'strat_key'.
        split_dir: Diretório onde serão gravados os arquivos train.txt e test.txt.
        train_ratio: Proporção de treino (padrão 0.8).
        seed: Semente para reprodutibilidade experimental.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (df_train, df_test)
    """
    print(f"[3/6] Divisão Estratificada - Split {int(train_ratio*100)}/{int((1-train_ratio)*100)} com NumPy...")
    rng = np.random.RandomState(seed)
    train_idx = []
    test_idx = []

    for strat_key, grupo in df.groupby("strat_key"):
        indices = grupo.index.to_numpy(copy=True)
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(np.round(train_ratio * n_total))
        if n_total > 1 and n_train == n_total:
            n_train = n_total - 1
        if n_train < 0:
            n_train = 0

        train_idx.extend(indices[:n_train].tolist())
        test_idx.extend(indices[n_train:].tolist())

    train_idx = np.array(train_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test = df.loc[test_idx].reset_index(drop=True)

    print(f"  Treino: {len(df_train)} amostras ({100*len(df_train)/len(df):.1f}%)")
    print(f"  Teste:  {len(df_test)} amostras ({100*len(df_test)/len(df):.1f}%)")

    split_dir.mkdir(parents=True, exist_ok=True)
    train_txt = split_dir / "train.txt"
    test_txt = split_dir / "test.txt"
    df_train["image_path"].to_csv(train_txt, index=False, header=False)
    df_test["image_path"].to_csv(test_txt, index=False, header=False)
    print(f"  Arquivos gerados: {train_txt.name}, {test_txt.name}")

    return df_train, df_test


def model_structure_configuration(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    yolo_dir: Path = YOLO_DIR,
    classes: list[str] = CLASSES,
) -> Path:
    """
    Estrutura os diretórios de imagens e labels para o modelo YOLO e gera o arquivo data.yaml.

    Args:
        df_train: DataFrame com dados de treino.
        df_test: DataFrame com dados de teste.
        yolo_dir: Diretório de destino da estrutura YOLO.
        classes: Lista com os nomes das classes.

    Returns:
        Path: Caminho do arquivo data.yaml gerado.
    """
    print("[4/6] Estruturação do Modelo - Preparando diretórios YOLO e data.yaml...")

    for split_name, df_split in [("train", df_train), ("test", df_test)]:
        img_out = yolo_dir / "images" / split_name
        lbl_out = yolo_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for _, row in df_split.iterrows():
            src_img = Path(row["image_path"])
            src_lbl = Path(row["label_path"])

            if not src_img.exists():
                continue
            dst_img = img_out / src_img.name
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            if src_lbl.exists():
                dst_lbl = lbl_out / src_lbl.name
                if not dst_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)
            else:
                dst_lbl = lbl_out / f"{src_img.stem}.txt"
                if not dst_lbl.exists():
                    dst_lbl.touch()

    data_yaml_path = yolo_dir / "data.yaml"
    conteudo_yaml = f"""path: {yolo_dir.resolve()}
train: images/train
val: images/test
test: images/test

nc: {len(classes)}
names: {classes}
"""
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(conteudo_yaml.strip() + "\n")

    print(f"  Estrutura criada em: {yolo_dir}")
    print(f"  data.yaml salvo em: {data_yaml_path}")
    return data_yaml_path


prepare_yolo_structure = model_structure_configuration  # Alias de compatibilidade
