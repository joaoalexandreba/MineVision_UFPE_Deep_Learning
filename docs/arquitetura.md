# 🏛️ Documento de Arquitetura e Decisões de Engenharia de Software

## 1. Visão Geral da Arquitetura

O sistema adota uma arquitetura em **Pipeline de Processamento em Camadas**, onde cada etapa possui responsabilidade única, alta coesão e baixo acoplamento.

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Roboflow API  │ ──> │   src/data.py    │ ──> │   data/raw/     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  data/splits/   │ <── │ src/preprocessing│ <── │ HarshDataset    │
│  (train/test)   │     │ (NumPy / PyTorch)│     │ (torch.Tensor)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ src/training.py │ ──> │ YOLOv11 &        │ ──> │  best.pt        │
│                 │     │ PyTorch Engine   │     │  (Pesos)        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                ┌──────────────────────────┴──────────────────────────┐
                                ▼                                                     ▼
                     ┌─────────────────────┐                               ┌─────────────────────┐
                     │  src/evaluation.py  │                               │  src/inference.py   │
                     │  (Métricas & IC 95%)│                               │  (Predição em fotos)│
                     └─────────────────────┘                               └─────────────────────┘
                                │                                                     │
                                ▼                                                     ▼
                     ┌─────────────────────┐                               ┌─────────────────────┐
                     │ resultados/ (*.png) │                               │ resultados/infer... │
                     │ relatorios/ (*.csv) │                               │ (*.jpg anotadas)    │
                     └─────────────────────┘                               └─────────────────────┘
```

---

## 2. Responsabilidade dos Módulos

| Módulo | Responsabilidade | Tecnologias Utilizadas |
| :--- | :--- | :--- |
| **`src/config.py`** | Centralização de sementes, diretórios e parâmetros globais | `pathlib.Path`, `torch.manual_seed` |
| **`src/data.py`** | Download e carregamento seguro do dataset do Roboflow | `roboflow.Roboflow`, `pathlib` |
| **`src/preprocessing.py`** | Dataset PyTorch, transformações de tensores, split 80/20 | `torch.utils.data.Dataset`, `torchvision`, `numpy` |
| **`src/models.py`** | Definição da arquitetura neural PyTorch e YOLOv11 | `torch.nn.Module`, `ultralytics.YOLO` |
| **`src/training.py`** | Orquestração do treinamento na GPU e checkpointing de pesos | `torch.cuda`, `YOLO.train()` |
| **`src/evaluation.py`** | Cálculo de métricas, Intervalos de Confiança (t-Student) e plots | `scipy.stats.t`, `seaborn`, `pandas` |
| **`src/inference.py`** | Predição visual em novas imagens avulsas | `PIL.Image`, `YOLO.predict()` |
| **`tests/`** | Validação automatizada de regras, tensores e split | `unittest`, `pytest` |
| **`main.py`** | Ponto de entrada CLI que integra e executa todo o fluxo | `argparse`, `sys` |

---

## 3. Decisões Arquiteturais e Justificativas

1. **Separação entre Dataset PyTorch e Treinamento YOLO**:
   * *Decisão*: Criamos a classe `HarshDataset` estendendo `torch.utils.data.Dataset` com `torchvision.transforms.Normalize` e `Resize(640, 640)`.
   * *Justificativa*: Garante a aderência estrita aos padrões de *Deep Learning* em PyTorch exigidos na ementa da disciplina, permitindo que os tensores sejam manipulados de forma independente.

2. **Divisão Estratificada com NumPy e Semente Fixa (`SEED = 42`)**:
   * *Decisão*: A função `split_dataset` agrupa por composição de classes (`strat_key`) e sorteia os índices de treino (80%) e teste (20%).
   * *Justificativa*: Evita desbalanceamento de classes entre treino e teste e garante **100% de reprodutibilidade** experimental.

3. **Avaliação Estatística com Intervalos de Confiança (t-Student 95%)**:
   * *Decisão*: Implementação de inferência estatística com $t$-Student via `scipy.stats.t.ppf` com cálculo de erro padrão:
     $$\text{IC}_{95\%} = \bar{X} \pm t_{\alpha/2, \nu} \cdot \left(\frac{s}{\sqrt{n}}\right)$$
   * *Justificativa*: Fornece rigor estatístico além de métricas pontuais, comprovando a estabilidade da convergência do modelo.

4. **Testabilidade e Cobertura Automatizada (`tests/`)**:
   * *Decisão*: Suíte de testes unitários cobrindo desde a dimensão dos tensores `(3, 640, 640)` até a garantia de que não há vazamento de dados (*data leakage*) entre treino e teste.
   * *Justificativa*: Atende diretamente ao requisito de qualidade de software do Slide 12.
