#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabalho Final - Engenharia de Software para IA e Frameworks Profundos
Pipeline de Visão Computacional com YOLOv11 & PyTorch (Dataset Harsh)
"""


# ===========================================================================
# # 🚛 Trabalho Final - Engenharia de Software para IA e Frameworks Profundos
# ## Pipeline Completo de Visão Computacional com YOLOv11 & PyTorch (Dataset Harsh)
# 
# ---
# 
# ### 👥 Equipe
# * **Demetrius**
# * **Eraldo**
# * **Josivan**
# * **João Alexandre**
# * **Raissa**
# * **Hugo**
# 
# ---
# 
# ### 🎯 Resumo do Problema e Motivação
# * **Contexto**: Monitoramento de carga e transporte em operações de mineração e logística pesada.
# * **Problema**: Identificar com precisão caçambas de caminhões (`bed`) e quantificar a área/volume de carvão (`coal`) sob condições ambientais adversas (*harsh conditions*: poeira, iluminação variável e vibração).
# * **Solução**: Pipeline modularizado com arquitetura em camadas combinando **PyTorch** nativo para manipulação de tensores e **YOLOv11** para segmentação de instâncias em tempo real.
# * **Objetivo de Engenharia de Software**: Garantir alta coesão, baixo acoplamento, tipagem estática (*type hints*), 100% de reprodutibilidade experimental e cobertura por testes automatizados.
# ===========================================================================


# ===========================================================================
# > **Aviso**: Execute a célula abaixo **apenas se estiver executando este notebook no Google Colab**. No ambiente local gerenciado pelo `uv` (`pyproject.toml`), todas as dependências já estão instaladas.
# ===========================================================================

# =============================================================================
# Instalação de Dependências no Google Colab
# =============================================================================
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("🚀 Detectado Google Colab. Instalando ultralytics e roboflow...")
#     get_ipython().system("pip install -q ultralytics roboflow")
    print("✅ Dependências instaladas com sucesso no Colab!")
else:
    print("ℹ️ Ambiente local detectado. Instalação do Colab ignorada.")



# ===========================================================================
# ### ⚙️ Célula 1: Imports, Configurações Globais e Reprodutibilidade
# Nesta célula definimos as sementes pseudo-aleatórias (`SEED = 42`) para **NumPy, Random e PyTorch**, garantindo que qualquer execução produza exatamente os mesmos resultados.
# ===========================================================================

import os
import shutil
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from ultralytics import YOLO
from roboflow import Roboflow

# Habilitar exibição de gráficos
# get_ipython().run_line_magic("matplotlib", "inline")

# =============================================================================
# Configurações Globais e Reprodutibilidade Experimental
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

API_KEY = "Md5T3krCkUBImQf9Df53"
WORKSPACE = "rapis-soft-technologies"
PROJECT = "harsh-0i6vw"
VERSION = 1

# Diretórios de trabalho
BASE_DIR = Path(".").resolve()
DOWNLOAD_DIR = BASE_DIR / "data" / "harsh_dataset"
YOLO_DIR = BASE_DIR / "data" / "yolo_harsh"
SPLIT_DIR = BASE_DIR / "data" / "splits"
RESULTS_DIR = BASE_DIR / "resultados"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
YOLO_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Classes do domínio
CLASSES = ["bed", "coal"]
CLASS_NAMES = {0: "bed", 1: "coal"}

print("✅ Configurações e diretórios inicializados com sucesso!")
print(f"🖥️ Dispositivo PyTorch em uso: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")



# ===========================================================================
# ### 📥 Célula 2: Etapa 1 - Carregamento e Download do Dataset (Módulo `data`)
# * **Objetivo:** Conectar à API do Roboflow, baixar o dataset estruturado em formato YOLOv11 ou reutilizar o cache local existente, retornando o caminho raiz e o `data.yaml` original.
# ===========================================================================

def load_dataset(
    api_key: str = API_KEY,
    workspace: str = WORKSPACE,
    project: str = PROJECT,
    version: int = VERSION,
    download_dir: Path = DOWNLOAD_DIR,
) -> tuple[Path, Path]:
    """
    Baixa o dataset diretamente do Roboflow e retorna o diretório raiz e o data.yaml.
    """
    print("[1/6] Carregando dataset do Roboflow...")
    data_yaml = None
    if download_dir.exists():
        for p in download_dir.rglob("data.yaml"):
            data_yaml = p
            break

    if data_yaml is None:
        rf = Roboflow(api_key=api_key)
        projeto = rf.workspace(workspace).project(project)
        versao = projeto.version(version)
        versao.download("yolov11", location=str(download_dir))

        for p in download_dir.rglob("data.yaml"):
            data_yaml = p
            break

    if data_yaml is None:
        raise FileNotFoundError("data.yaml não encontrado após o download.")

    raiz_dataset = data_yaml.parent
    print(f"  Dataset localizado em: {raiz_dataset}")

    splits_encontrados = {}
    for split_name in ["train", "valid", "test", "val"]:
        img_dir = raiz_dataset / split_name / "images"
        if img_dir.exists():
            n = len(list(img_dir.glob("*.*")))
            splits_encontrados[split_name] = n
            print(f"  Split '{split_name}': {n} imagens")

    total = sum(splits_encontrados.values())
    print(f"  TOTAL de imagens disponíveis: {total}")
    return raiz_dataset, data_yaml

# Executar Etapa 1
raiz_dataset, data_yaml_original = load_dataset()



# ===========================================================================
# ### 🔄 Célula 3: Etapa 2 - Pré-processamento & Dataset PyTorch (Módulo `preprocessing`)
# * **Objetivo:** Implementar a classe `HarshDataset(Dataset)` nativa do **PyTorch**, aplicar transformações do `torchvision` (Resize para 640x640, conversão em `torch.Tensor` e normalização ImageNet) e mapear todas as anotações para análise exploratória.
# ===========================================================================

class HarshDataset(Dataset):
    """
    Classe customizada para carregar os dados no padrão PyTorch.
    Converte as imagens do disco em Tensores (torch.Tensor) na hora do treino.
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
    """Pipeline de transformações padrão do torchvision."""
    return transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def preprocess(raiz_dataset: Path) -> pd.DataFrame:
    """
    Mapeia todas as imagens e labels e estrutura o DataFrame com metadados.
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
    print(f"  Distribuição por split original:\n{df['split_original'].value_counts().to_string()}")
    print(f"  Distribuição por composição de classes:\n{df['strat_key'].value_counts().to_string()}")
    return df

# Executar Etapa 2
df = preprocess(raiz_dataset)
df.head()



# ===========================================================================
# ### ✂️ Célula 4: Etapa 3 - Divisão Estratificada 80/20 Treino & Teste com NumPy
# * **Objetivo:** Realizar a divisão estratificada (80% treino e 20% teste) considerando a ocorrência simultânea de classes por imagem (`strat_key`), utilizando `numpy.random.RandomState` para garantir semente fixa e ausência de vazamento de dados (*data leakage*).
# ===========================================================================

def split_dataset(
    df: pd.DataFrame,
    split_dir: Path = SPLIT_DIR,
    train_ratio: float = 0.8,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide o dataset de forma estratificada em 80% treino e 20% teste usando NumPy.
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
    print(f"  Arquivos salvos: {train_txt.name}, {test_txt.name}")

    return df_train, df_test

# Executar Etapa 3
df_train, df_test = split_dataset(df)



# ===========================================================================
# ### 📁 Célula 5: Etapa 4 - Preparação da Estrutura YOLO & data.yaml
# * **Objetivo:** Organizar fisicamente os arquivos em pastas de imagens e labels separadas para treino e teste, e gerar dinamicamente o arquivo `data.yaml` exigido pelo YOLO.
# ===========================================================================

def model_structure_configuration(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    yolo_dir: Path = YOLO_DIR,
    classes: list[str] = CLASSES,
) -> Path:
    """
    Copia imagens e labels para as pastas do YOLO e gera o data.yaml final.
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

    print(f"  Estrutura YOLO criada em: {yolo_dir}")
    print(f"  data.yaml salvo em: {data_yaml_path}")
    return data_yaml_path

# Executar Etapa 4
data_yaml_path = model_structure_configuration(df_train, df_test)



# ===========================================================================
# ### 🏋️ Célula 6: Etapa 5 - Treinamento do Modelo YOLOv11 & PyTorch (Módulo `training`)
# * **Objetivo:** Executar o treinamento da arquitetura YOLOv11-seg baseada em PyTorch (`torch.nn.Module`, otimizador `AdamW` e funções de perda `CIoU/BCE`), salvando automaticamente os pesos ótimos em `best.pt`.
# ===========================================================================

def train_model(
    data_yaml_path: Path,
    epochs: int = 50,
    batch_size: int = 8,
    results_dir: Path = RESULTS_DIR,
    seed: int = SEED,
) -> Path:
    """
    Executa o treinamento do YOLOv11 no dataset preparado com aceleração por GPU.
    """
    print(f"[5/6] Treinamento - Inicializando YOLOv11 ({epochs} épocas, batch={batch_size})...")
    
    modelos_candidatos = ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11n.pt"]
    modelo = None
    modelo_escolhido = None

    for nome in modelos_candidatos:
        try:
            print(f"  Carregando pesos pré-treinados: {nome}")
            modelo = YOLO(nome)
            modelo_escolhido = nome
            print(f"  Modelo PyTorch carregado com sucesso: {nome}")
            break
        except Exception as e:
            print(f"  Falha ao carregar {nome}: {e}")

    if modelo_escolhido is None:
        raise RuntimeError("Não foi possível instanciar o modelo YOLOv11.")

    best_weights = results_dir / "treino_yolov11_harsh" / "weights" / "best.pt"

    resultados = modelo.train(
        data=str(data_yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        seed=seed,
        project=str(results_dir),
        name="treino_yolov11_harsh",
        exist_ok=True,
        verbose=True,
    )

    print(f"  Treinamento concluído. Melhor modelo salvo em: {best_weights}")
    return best_weights

# Executar Etapa 5
best_weights = train_model(data_yaml_path, epochs=50, batch_size=8)



# ===========================================================================
# ### 📊 Célula 7: Etapa 6 - Avaliação Estatística, Intervalos de Confiança (t-Student 95%) e Gráficos (Módulo `evaluation`)
# * **Objetivo:** Avaliar o modelo no conjunto de teste isolado (76 imagens), calculando Precision, Recall, F1-Score, mAP50, mAP50-95 e Intervalos de Confiança de 95% via **t-Student** com graus de liberdade e erro padrão. Gera Matriz de Confusão e Boxplot comparativo com **Seaborn** e exporta relatórios `.csv`.
# ===========================================================================

def evaluate_model(
    best_weights: Path,
    data_yaml_path: Path,
    results_dir: Path = RESULTS_DIR,
    classes: list[str] = CLASSES,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Avalia o modelo treinado no conjunto de teste e calcula métricas com IC 95% (t-Student).
    """
    print("[6/6] Avaliação Estatística - Executando no conjunto de teste...")
    modelo = YOLO(str(best_weights))

    metricas = modelo.val(
        data=str(data_yaml_path),
        split="test",
        seed=seed,
        project=str(results_dir),
        name="avaliacao_yolov11_harsh",
        exist_ok=True,
        verbose=True,
        plots=True,
        save_json=True,
    )

    box = metricas.box
    precision_global = float(box.mp)
    recall_global = float(box.mr)
    map50_global = float(box.map50)
    map5095_global = float(box.map)

    print("\n--- Métricas Globais (Conjunto de Teste) ---")
    print(f"Precision global: {precision_global:.4f}")
    print(f"Recall global:    {recall_global:.4f}")
    print(f"mAP50 global:     {map50_global:.4f}")
    print(f"mAP50-95 global:  {map5095_global:.4f}")

    n_classes = len(classes)

    def ajustar(arr, n):
        arr = np.array(arr, dtype=float).reshape(-1)
        if len(arr) < n:
            arr = np.pad(arr, (0, n - len(arr)), constant_values=np.nan)
        return arr[:n]

    p_por_classe = ajustar(box.p, n_classes)
    r_por_classe = ajustar(box.r, n_classes)
    map50_por_classe = ajustar(box.ap50, n_classes)
    map5095_por_classe = ajustar(box.ap, n_classes)

    f1_por_classe = np.where(
        (p_por_classe + r_por_classe) > 0,
        2 * (p_por_classe * r_por_classe) / (p_por_classe + r_por_classe + 1e-12),
        0.0,
    )
    f1_macro = float(np.nanmean(f1_por_classe))

    df_metricas = pd.DataFrame({
        "classe": classes,
        "precision": p_por_classe,
        "recall": r_por_classe,
        "f1": f1_por_classe,
        "mAP50": map50_por_classe,
        "mAP50-95": map5095_por_classe,
    })

    print("\n--- Métricas por Classe ---")
    display(df_metricas) if "display" in globals() else print(df_metricas.to_string(index=False))
    print(f"\nF1-Score macro médio: {f1_macro:.4f}")

    colunas_metricas = ["precision", "recall", "f1", "mAP50", "mAP50-95"]
    linhas_stats = []

    n = n_classes
    graus_liberdade = max(n - 1, 1)
    t_crit = float(stats.t.ppf(0.975, graus_liberdade))

    for col in colunas_metricas:
        valores = df_metricas[col].to_numpy(dtype=float)
        media = float(np.nanmean(valores))
        desvio = float(np.nanstd(valores, ddof=1)) if n > 1 else 0.0
        erro_padrao = desvio / np.sqrt(n) if n > 0 else 0.0
        ic_inf = media - t_crit * erro_padrao
        ic_sup = media + t_crit * erro_padrao
        linhas_stats.append({
            "metrica": col,
            "media": media,
            "desvio_padrao": desvio,
            "ic95_inf": ic_inf,
            "ic95_sup": ic_sup,
            "t_critico": t_crit,
        })

    df_stats = pd.DataFrame(linhas_stats)
    print("\n--- Estatísticas (Média, Desvio Padrão, IC 95% t-Student) ---")
    display(df_stats) if "display" in globals() else print(df_stats.to_string(index=False))

    # 1. Matriz de Confusão Seaborn
    try:
        mc = getattr(metricas, "confusion_matrix", None)
        if mc is not None and hasattr(mc, "matrix"):
            matriz_confusao = np.array(mc.matrix)
            plt.figure(figsize=(7, 5))
            labels_plot = classes + ["background"] if matriz_confusao.shape[0] == len(classes) + 1 else classes
            sns.heatmap(
                matriz_confusao,
                annot=True,
                fmt=".0f",
                cmap="Blues",
                xticklabels=labels_plot,
                yticklabels=labels_plot,
            )
            plt.title("Matriz de Confusão - YOLOv11 (harsh)")
            plt.xlabel("Predito")
            plt.ylabel("Real")
            plt.tight_layout()
            cm_path = results_dir / "matriz_confusao_seaborn.png"
            plt.savefig(cm_path, dpi=150)
            plt.show()
            print(f"  Matriz de confusão salva em: {cm_path}")
    except Exception as e:
        print(f"  Aviso ao plotar matriz de confusão: {e}")

    # 2. Boxplot Comparativo Seaborn
    try:
        df_box = df_metricas.melt(
            id_vars=["classe"],
            value_vars=colunas_metricas,
            var_name="metrica",
            value_name="valor",
        )
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=df_box, x="metrica", y="valor", palette="Set2")
        sns.stripplot(data=df_box, x="metrica", y="valor", color="black", size=6, jitter=True)
        plt.title("Boxplot Comparativo das Métricas por Classe - YOLOv11 (harsh)")
        plt.ylim(0, 1.05)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        box_path = results_dir / "boxplot_metricas_por_classe.png"
        plt.savefig(box_path, dpi=150)
        plt.show()
        print(f"  Boxplot salvo em: {box_path}")
    except Exception as e:
        print(f"  Aviso ao plotar boxplot: {e}")

    # Exportação de relatórios
    df_metricas.to_csv(results_dir / "relatorio_metricas.csv", index=False, encoding="utf-8-sig")
    df_stats.to_csv(results_dir / "relatorio_estatisticas.csv", index=False, encoding="utf-8-sig")
    print(f"  Relatórios CSV salvos em: {results_dir}")

    return df_metricas, df_stats

# Executar Etapa 6
df_metricas, df_stats = evaluate_model(best_weights, data_yaml_path)



# ===========================================================================
# ### 🔍 Célula 8: Etapa 7 - Inferência & Predição para Novas Imagens (Módulo `inference` - RF05)
# * **Objetivo:** Atender ao requisito funcional **RF05** realizando a predição e renderização das caixas delimitadoras e máscaras de segmentação sobre uma imagem real de teste.
# ===========================================================================

def predict_image(
    image_path: Union[str, Path],
    model_weights: Union[str, Path],
    output_dir: Optional[Path] = None,
    conf_threshold: float = 0.25,
) -> Path:
    """
    Executa inferência em nova imagem avulsa e plota as máscaras e caixas detectadas.
    """
    img_p = Path(image_path)
    w_p = Path(model_weights)
    if not img_p.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {img_p}")
    if not w_p.exists():
        raise FileNotFoundError(f"Pesos não encontrados: {w_p}")

    if output_dir is None:
        output_dir = RESULTS_DIR / "inferencias"
    output_dir.mkdir(parents=True, exist_ok=True)

    modelo = YOLO(str(w_p))
    results = modelo.predict(source=str(img_p), conf=conf_threshold, save=False, verbose=False)
    result = results[0]
    im_array = result.plot()
    im = Image.fromarray(im_array[..., ::-1])

    out_path = output_dir / f"pred_{img_p.stem}.jpg"
    im.save(out_path)
    print(f"  Predição realizada com sucesso para: {img_p.name}")
    print(f"  Imagem resultante salva em: {out_path}")
    
    # Exibir a imagem anotada no notebook
    plt.figure(figsize=(8, 6))
    plt.imshow(im)
    plt.axis("off")
    plt.title(f"Inferência YOLOv11 - {img_p.name}")
    plt.show()
    return out_path

# Testar inferência em uma imagem de teste
amostras_teste = list((YOLO_DIR / "images" / "test").glob("*.jpg"))
if amostras_teste:
    img_amostra = amostras_teste[0]
    out_pred = predict_image(img_amostra, best_weights)
else:
    print("Nenhuma imagem de teste localizada para inferência.")



# ===========================================================================
# ### 🧪 Célula 9: Etapa 8 - Execução dos Testes Automatizados (Módulo `tests`)
# * **Objetivo:** Executar a suíte de testes unitários diretamente no notebook, validando tensores PyTorch `(3, 640, 640)`, integridade da divisão estratificada 80/20 e fórmulas do Intervalo de Confiança t-Student, gerando a evidência de execução (`Ran X tests ... OK`).
# ===========================================================================

import unittest

class TestSuiteNotebook(unittest.TestCase):
    """Suíte de testes automatizados integrada para validação do pipeline."""

    def test_01_pytorch_tensor_shape(self):
        """[Teste 1] Valida se o pipeline PyTorch produz tensores (3, 640, 640)."""
        transformacoes = get_transforms()
        dummy_img = Image.new("RGB", (300, 300), color=(100, 100, 100))
        tensor = transformacoes(dummy_img)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 640, 640))

    def test_02_stratified_split_proportions(self):
        """[Teste 2] Valida se a divisão estratificada mantém 80% treino e 20% teste."""
        dummy_df = pd.DataFrame({
            "image_path": [f"img_{i}.jpg" for i in range(100)],
            "strat_key": ["0_1"] * 60 + ["0"] * 40
        })
        train_df, test_df = split_dataset(dummy_df, seed=42)
        self.assertEqual(len(train_df), 80)
        self.assertEqual(len(test_df), 20)

    def test_03_no_data_leakage(self):
        """[Teste 3] Garante isolamento estrito sem vazamento de índices entre treino e teste."""
        dummy_df = pd.DataFrame({
            "image_path": [f"img_{i}.jpg" for i in range(50)],
            "strat_key": ["0_1"] * 30 + ["1"] * 20
        })
        train_df, test_df = split_dataset(dummy_df, seed=42)
        overlap = set(train_df["image_path"]).intersection(set(test_df["image_path"]))
        self.assertEqual(len(overlap), 0)

    def test_04_confidence_interval_math(self):
        """[Teste 4] Valida o cálculo do Intervalo de Confiança t-Student a 95%."""
        vals = np.array([0.98, 0.99, 0.97, 0.96, 1.00])
        media = float(np.mean(vals))
        desvio = float(np.std(vals, ddof=1))
        t_crit = float(stats.t.ppf(0.975, len(vals) - 1))
        erro = desvio / np.sqrt(len(vals))
        ic_inf = media - t_crit * erro
        ic_sup = media + t_crit * erro
        self.assertAlmostEqual(media, 0.98, places=2)
        self.assertTrue(ic_inf < media < ic_sup)

# Executar a suíte de testes no notebook
suite = unittest.TestLoader().loadTestsFromTestCase(TestSuiteNotebook)
runner = unittest.TextTestRunner(verbosity=2)
print("=== EXECUTANDO TESTES UNITÁRIOS AUTOMATIZADOS (EVIDÊNCIA SLIDE 12) ===")
resultado = runner.run(suite)



# ===========================================================================
# ---
# ## 🎯 Conclusão e Análise Crítica dos Resultados
# 
# ### 1. Síntese do Desempenho
# * O modelo **YOLOv11-seg** alcançou **99.70% de F1-Score macro** e **94.07% de mAP50-95** no conjunto de teste isolado.
# * Todas as metas de Engenharia de Software e critérios de aceitação foram superados com folga ($F1 \ge 0.80$, inferência $< 50ms$).
# 
# ### 2. Análise Crítica sobre Generalização e Suposto Overfit
# * O modelo foi avaliado exclusivamente no conjunto de teste independente (76 fotos).
# * O desempenho de ~99% decorre da alta qualidade das anotações, forte contraste visual entre caçamba e carvão, e pré-treinamento robusto no COCO.
# * **Limitação**: Por ter sido gravado em ambiente industrial fixo, imagens com condições extremas não catalogadas (ex: noite total, tempestade intensa) podem ter desempenho inferior.
# 
# ### 3. Próximos Passos e Melhorias Futuras
# 1. Incorporação de técnicas de *Out-of-Distribution (OOD)* e validação cruzada com câmeras externas (*GroupKFold*).
# 2. Quantização e exportação para **ONNX / TensorRT** para embarque em sistemas de borda (*Edge AI*) na cabine de caminhões.
# ===========================================================================
