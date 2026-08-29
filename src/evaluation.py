from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from ultralytics import YOLO

from src.config import CLASSES, RESULTS_DIR, SEED


def calculate_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float, float, float]:
    """
    Calcula média, desvio padrão, margem de erro e limites do Intervalo de Confiança (t-Student).

    Args:
        values: Array 1D com os valores da métrica.
        confidence: Nível de confiança (padrão 0.95 para 95%).

    Returns:
        tuple[float, float, float, float, float]: (media, desvio_padrao, ic_inf, ic_sup, t_critico)
    """
    valores = np.asarray(values, dtype=float)
    n = len(valores)
    media = float(np.nanmean(valores))
    desvio = float(np.nanstd(valores, ddof=1)) if n > 1 else 0.0
    graus_liberdade = max(n - 1, 1)
    t_crit = float(stats.t.ppf((1 + confidence) / 2.0, graus_liberdade))
    erro_padrao = desvio / np.sqrt(n) if n > 0 else 0.0
    ic_inf = media - t_crit * erro_padrao
    ic_sup = media + t_crit * erro_padrao
    return media, desvio, ic_inf, ic_sup, t_crit


def evaluate_model(
    best_weights: Path,
    data_yaml_path: Path,
    results_dir: Path = RESULTS_DIR,
    classes: list[str] = CLASSES,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Avalia o modelo treinado no conjunto de teste e gera métricas, IC 95% e gráficos.

    Args:
        best_weights: Caminho dos pesos (.pt) a avaliar.
        data_yaml_path: Caminho do arquivo data.yaml.
        results_dir: Diretório de saída dos relatórios e figuras.
        classes: Nomes das classes do modelo.
        seed: Semente pseudo-aleatória.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (df_metricas, df_stats)
    """
    print("[6/6] Executando avaliação estatística no conjunto de teste...")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(str(best_weights))
    metricas = model.val(
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

    print("\n--- Métricas Globais (Teste) ---")
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
    print(df_metricas.to_string(index=False))
    print(f"\nF1-Score macro médio: {f1_macro:.4f}")

    colunas_metricas = ["precision", "recall", "f1", "mAP50", "mAP50-95"]
    linhas_stats = []

    for col in colunas_metricas:
        valores = df_metricas[col].to_numpy(dtype=float)
        media, desvio, ic_inf, ic_sup, t_crit = calculate_confidence_interval(valores, confidence=0.95)
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
    print(df_stats.to_string(index=False))

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
            plt.close()
            print(f"  Matriz de confusão salva em: {cm_path}")
    except Exception as e:
        print(f"  Aviso: Não foi possível plotar matriz de confusão: {e}")

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
        plt.close()
        print(f"  Boxplot salvo em: {box_path}")
    except Exception as e:
        print(f"  Aviso: Não foi possível plotar boxplot: {e}")

    # Exportação dos Relatórios
    df_metricas.to_csv(results_dir / "relatorio_metricas.csv", index=False, encoding="utf-8-sig")
    df_stats.to_csv(results_dir / "relatorio_estatisticas.csv", index=False, encoding="utf-8-sig")
    print(f"  Relatórios CSV salvos em: {results_dir}")

    return df_metricas, df_stats
