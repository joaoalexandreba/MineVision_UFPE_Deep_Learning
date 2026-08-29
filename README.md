# 🚛 Sistema de Visão Computacional para Segmentação de Cargas em Caminhões (Dataset Harsh)
> **Trabalho Final da Disciplina de Engenharia de Software para IA e Frameworks Profundos**  
> *Especialização em Deep Learning - UFPE*

---

## 👥 Equipe
* **Demetrius Valença**
* **Eraldo Florêncio**
* **Josivan Reis**
* **João Alexandre**
* **Raissa Tavares**
* **Hugo Nicolau Barbosa de Gusmão**

---

## 📌 Mapeamento das Entregas da Disciplina (Google Classroom)

Esta seção resume o atendimento às 4 entregas incrementais solicitadas pelo professor:

| Entrega | Foco & Critérios Avaliados | Como Foi Atendido no Projeto |
| :--- | :--- | :--- |
| **Entrega 1** | **Engenharia de Software Inicial, Modularização e Tipagem**<br>• Contextualização do problema<br>• Modularização e organização do código<br>• Tipagem e qualidade Python | • Seção 1 deste README (Contexto, Motivação e Objetivos)<br>• Pacote modular `src/` (`data.py`, `preprocessing.py`, `models.py`, `training.py`, `evaluation.py`, `inference.py`)<br>• Ponto de entrada CLI `main.py`<br>• *Type Hints* em 100% das funções (`Path`, `torch.Tensor`, `np.ndarray`, `pd.DataFrame`, `tuple`) |
| **Entrega 2** | **Uso Adequado de NumPy**<br>• Manipulação de matrizes e vetorização<br>• Divisão estratificada 80/20 com semente fixa<br>• Estatística descritiva e arrays | • Módulo `src/preprocessing.py`: função `split_dataset` com `np.random.RandomState(42)` e indexação de arrays<br>• Módulo `src/evaluation.py`: operações vetorizadas para cálculo de médias, desvios e IC 95% |
| **Entrega 3** | **Implementação em PyTorch**<br>• Carregamento de dados com `Dataset` PyTorch<br>• Pré-processamento e normalização (`torchvision`)<br>• Treinamento do modelo na GPU/CPU (`nn.Module`)<br>• Impressão do erro de treino/teste por época<br>• Salvamento do modelo treinado (`best.pt`) | • Módulo `src/preprocessing.py`: classe `HarshDataset(Dataset)` e pipeline `get_transforms()` com normalização ImageNet<br>• Módulo `src/models.py`: arquitetura PyTorch `HarshNeuralFeatureExtractor(nn.Module)` e YOLOv11<br>• Módulo `src/training.py`: `train_model` com log de losses (Box, Cls, DFL) e salvamento em `best.pt` |
| **Entrega 4** | **Testes Automatizados com `unittest` & Requisitos (GR4ML)**<br>• Suíte de testes unitários em pasta dedicada<br>• Testes de carregamento, tensores e split<br>• Evidência de execução de todos os testes<br>• Documento de requisitos (GR4ML) | • Pasta `tests/` (`test_dataset.py`, `test_preprocessing.py`, `test_evaluation.py`, `test_inference.py`)<br>• Suíte automatizada com 10 testes passando (`Ran 10 tests in 0.168s - OK`)<br>• Documento de Requisitos (GR4ML): [GR4ML_Monitoramento de Operação de Caminhões em Mina.pdf](docs/GR4ML_Monitoramento%20de%20Operação%20de%20Caminhões%20em%20Mina.pdf) |
| **Documentação & Final** | **Requisitos, Arquitetura e Apresentação**<br>• Requisitos de software<br>• Arquitetura em camadas<br>• Apresentação e Notebook didático | • `docs/requisitos.md` (Requisitos RF/RNF e matriz de rastreabilidade)<br>• `docs/arquitetura.md` (Diagrama em camadas e fluxo de dados)<br>• Notebook didático: `notebooks/EntregaFinal.ipynb` |

---

## 🎯 1. Visão Geral do Projeto

### 1.1 Contexto e Motivação
Em operações industriais e de mineração pesada, a conferência de carga de caminhões (como o transporte de carvão mineral) é tradicionalmente realizada por pesagem estática em balanças ou por inspeção visual humana. Esse processo apresenta desafios significativos:
* **Condições Adversas (*Harsh Conditions*)**: Ambientes com alta incidência de poeira, iluminação solar variável, sombras densas e vibração constante.
* **Gargalos Operacionais**: Filas de caminhões nos postos de pesagem geram atrasos logísticos.
* **Vulnerabilidade a Fraudes**: Volumes aparentes podem não corresponder à carga real sem uma estimativa visual da superfície e ocupação da caçamba.

### 1.2 A Solução Proposta
Desenvolvimento de um sistema de visão computacional em tempo real capaz de:
1. Localizar e segmentar a **caçamba do caminhão (`bed`)** delimitando a área útil de transporte.
2. Identificar e quantificar a massa de **carvão mineral (`coal`)** transportada.
3. Operar em alta velocidade (**>50 FPS / ~17ms por frame**), permitindo integração com câmeras instaladas em portais de entrada e esteiras de carregamento.

### 1.3 Dataset Utilizado
* **Fonte**: Roboflow Universe (*Harsh Dataset - v1*).
* **Classes**: `bed` (caçamba) e `coal` (carvão mineral).
* **Volume**: 379 imagens com anotações de segmentação poligonal (*masks*) e caixas delimitadoras (*bounding boxes*).
* **Divisão Experimental**: 80% Treino (303 imagens) e 20% Teste (76 imagens / 99 instâncias) com divisão estratificada por composição de classes.

---

## 🏛️ 2. Arquitetura e Engenharia de Software

O projeto foi construído seguindo princípios clássicos de Engenharia de Software: **alta coesão, baixo acoplamento, separação de responsabilidades e tipagem estática (*type hints*)**.

### 2.1 Diagrama do Pipeline em Camadas
```text
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

### 2.2 Responsabilidade dos Módulos

| Módulo | Responsabilidade Técnica | Tecnologias Utilizadas |
| :--- | :--- | :--- |
| **`src/config.py`** | Centralização de sementes, diretórios e parâmetros globais | `pathlib`, `torch.manual_seed` |
| **`src/data.py`** | Download e carregamento seguro do dataset via API | `roboflow.Roboflow`, `pathlib` |
| **`src/preprocessing.py`** | Dataset PyTorch, transformações de tensores e split estratificado | `torch.utils.data.Dataset`, `torchvision`, `numpy` |
| **`src/models.py`** | Definição da arquitetura neural PyTorch e modelo YOLOv11 | `torch.nn.Module`, `ultralytics.YOLO` |
| **`src/training.py`** | Treinamento acelerado por GPU com registro de métricas e checkpointing | `torch.cuda`, `YOLO.train()` |
| **`src/evaluation.py`** | Cálculo de métricas, Intervalos de Confiança (t-Student 95%) e gráficos | `scipy.stats.t`, `seaborn`, `pandas` |
| **`src/inference.py`** | Predição visual em novas imagens avulsas (RF05) | `PIL.Image`, `YOLO.predict()` |
| **`tests/`** | Suíte de testes unitários automatizados | `unittest`, `pytest` |
| **`main.py`** | Ponto de entrada CLI que integra todo o pipeline | `argparse`, `sys` |

---

## 📋 3. Requisitos do Sistema

### Requisitos Funcionais (RF)
* **RF01**: O sistema deve baixar e sincronizar o dataset do Roboflow via API (`src/data.py`).
* **RF02**: O sistema deve pré-processar as imagens convertendo-as em Tensores PyTorch normalizados `(3, 640, 640)` (`src/preprocessing.py`).
* **RF03**: O sistema deve realizar a divisão estratificada 80/20 (treino/teste) com NumPy (`src/preprocessing.py`).
* **RF04**: O sistema deve estruturar as pastas YOLO e gerar o arquivo `data.yaml` (`src/preprocessing.py`).
* **RF05**: O sistema deve treinar o modelo neural com salvamento automático dos melhores pesos (`src/training.py`).
* **RF06**: O sistema deve avaliar o modelo no teste com métricas e IC 95% via t-Student (`src/evaluation.py`).
* **RF07**: O sistema deve exportar a Matriz de Confusão e Boxplot comparativo em PNG e relatórios em CSV (`src/evaluation.py`).
* **RF08**: O sistema deve realizar inferência e predição visual em novas fotos avulsas (`src/inference.py`).

### Requisitos Não Funcionais (RNF) e Critérios de Aceitação
* **RNF01 (Desempenho)**: $F1\text{-score} \ge 0.80$ no conjunto de teste (*Critério de Aceitação superado: $F1 = 0.9970$*).
* **RNF02 (Tempo de Resposta)**: Latência de inferência inferior a 50ms (*Alcançado: ~17ms / >50 FPS na GPU*).
* **RNF03 (Modularidade)**: Arquitetura em camadas com alta coesão e baixo acoplamento (`src/`).
* **RNF04 (Tipagem Estática)**: *Type Hints* em 100% das funções.
* **RNF05 (Testabilidade)**: Suíte com 10 testes unitários automatizados (`tests/`).
* **RNF06 (Reprodutibilidade)**: Semente determinística `SEED = 42` para NumPy, Random e PyTorch.

---

## 🏗️ 4. Estrutura do Repositório

```text
EntregaFinal/
│
├── data/                       # 📦 Dados brutos, processados e splits
│   ├── harsh_dataset/          (Download original via Roboflow API)
│   ├── yolo_harsh/             (Estrutura de imagens/labels para o YOLO)
│   └── splits/                 (Arquivos train.txt e test.txt)
│
├── docs/                       # 📄 Documentação técnica e de engenharia
│   ├── Instrucoes/             (PDFs de orientações da disciplina)
│   ├── requisitos.md           (Especificação formal de requisitos RF/RNF)
│   ├── arquitetura.md          (Decisões arquiteturais e diagramas em camadas)
│   └── GR4ML_Monitoramento...  (Documento formal de requisitos GR4ML)
│
├── notebooks/                  # 📓 Notebooks de experimentação e didáticos
│   └── EntregaFinal.ipynb      (Notebook interativo e didático de ponta a ponta)
│
├── src/                        # 🧠 Código-Fonte Modularizado (Engenharia de Software)
│   ├── __init__.py
│   ├── config.py               (Sementes, constantes de diretório e parâmetros)
│   ├── data.py                 (Carregamento e download via Roboflow API)
│   ├── preprocessing.py        (HarshDataset PyTorch, transforms, split estratificado)
│   ├── models.py               (Definição PyTorch nn.Module e modelo YOLOv11)
│   ├── training.py             (Treinamento na GPU com registro de losses)
│   ├── evaluation.py           (Métricas, IC 95% t-Student, matriz de confusão e boxplot)
│   └── inference.py            (Predição e visualização em novas imagens)
│
├── tests/                      # 🧪 Suíte de Testes Automatizados (unittest)
│   ├── __init__.py
│   ├── test_dataset.py         (Testes de tensores PyTorch e shapes 3x640x640)
│   ├── test_preprocessing.py   (Teste da divisão estratificada 80/20 com NumPy)
│   ├── test_evaluation.py      (Teste do cálculo de F1-Score e IC 95% t-Student)
│   └── test_inference.py       (Testes de validação de arquivos para inferência)
│
├── resultados/                 # 📊 Gráficos Seaborn e Relatórios CSV gerados
│   ├── matriz_confusao_seaborn.png
│   ├── boxplot_metricas_por_classe.png
│   ├── relatorio_metricas.csv
│   └── relatorio_estatisticas.csv
│
├── main.py                     # 🚀 Orquestrador do Pipeline via CLI
├── EntregaFinal.ipynb          # 📓 Notebook executável e didático de ponta a ponta
├── EntregaFinal.py             # 🐍 Versão em script Python executável
├── requirements.txt            # 📋 Arquivo de dependências
└── pyproject.toml              # ⚙️ Gerenciamento de dependências com uv
```

---

## 🚀 5. Como Executar o Projeto

### 5.1 Instalação do Ambiente
Utilizando o gerenciador **uv** (ou pip):
```bash
# Sincronizar e instalar dependências do pyproject.toml
uv sync
```

### 5.2 Execução dos Testes Automatizados
```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

**Evidência de Execução**:
```text
test_dataset_instantiation_and_len (test_dataset.TestDatasetAndModels) ... ok
test_neural_feature_extractor_forward (test_dataset.TestDatasetAndModels) ... ok
test_tensor_transformation_and_shape (test_dataset.TestDatasetAndModels) ... ok
test_confidence_interval_calculation (test_evaluation.TestEvaluation) ... ok
test_confidence_interval_zero_variance (test_evaluation.TestEvaluation) ... ok
test_inference_missing_image_raises_error (test_inference.TestInference) ... ok
test_inference_missing_weights_raises_error (test_inference.TestInference) ... ok
test_split_files_saved (test_preprocessing.TestPreprocessing) ... ok
test_split_no_index_overlap (test_preprocessing.TestPreprocessing) ... ok
test_split_proportions_80_20 (test_preprocessing.TestPreprocessing) ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.168s

OK
```

### 5.3 Execução do Pipeline via CLI (`main.py`)
```bash
# Executar pipeline completo (download -> preprocess -> treino -> avaliação)
uv run python main.py

# Avaliar pesos existentes sem retreinar
uv run python main.py --skip-train

# Realizar inferência em imagem avulsa (RF08)
uv run python main.py --skip-train --infer data/yolo_harsh/images/test/exemplo.jpg
```

### 5.4 Execução Interativa via Notebook (Google Colab / Jupyter)
Para quem desejar testar, explorar os dados e reproduzir o pipeline de forma visual e interativa célula a célula, disponibilizamos o notebook didático completo em **`notebooks/EntregaFinal.ipynb`**:
* **Suporte ao Google Colab**: O notebook possui detecção automática de ambiente (`google.colab`), instalando as dependências necessárias (`!pip install -q ultralytics roboflow`) e aproveitando aceleração gratuita por GPU (T4/V100).
* **Execução Local**: Pode ser aberto localmente via Jupyter Lab / VSCode:
  ```bash
  uv run jupyter lab notebooks/EntregaFinal.ipynb
  ```
* **Didática e Visualização**: Contém explicações passo a passo, gráficos interativos com Seaborn, renderização de máscaras de segmentação e execução em tempo real da suíte de testes unitários.

---

## 📊 6. Resultados Experimentais e Validação Estatística

Avaliação no conjunto de teste isolado (**76 imagens / 99 instâncias**):

| Métrica | Global | Classe `bed` (Caçamba) | Classe `coal` (Carvão) | IC 95% (t-Student) |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | **99.54%** | 99.09% | 100.00% | $[93.76\%, 100.00\%]$ |
| **Recall** | **99.86%** | 100.00% | 99.72% | $[98.06\%, 100.00\%]$ |
| **F1-Score** | **99.70%** | 99.54% | 99.86% | $[97.69\%, 100.00\%]$ |
| **mAP@50** | **99.50%** | 99.50% | 99.50% | $[99.50\%, 99.50\%]$ |
| **mAP@50-95** | **94.07%** | 93.09% | 95.05% | $[81.59\%, 100.00\%]$ |

### Artefatos Gráficos Gerados:
* **Matriz de Confusão**: `resultados/matriz_confusao_seaborn.png`
* **Boxplot Comparativo das Métricas**: `resultados/boxplot_metricas_por_classe.png`
* **Relatórios Estruturados**: `resultados/relatorio_metricas.csv` e `resultados/relatorio_estatisticas.csv`

---

## 🎯 7. Conclusões, Limitações e Trabalhos Futuros

* **Atendimento aos Requisitos**: O critério de aceitação de $F1 \ge 0.80$ foi superado ($F1 = 0.9970$), e a velocidade de inferência (~17ms) viabiliza operação em tempo real (>50 FPS).
* **Análise Crítica de Generalização**: O alto desempenho no teste decorre do forte contraste visual entre caçamba e carvão e da qualidade das anotações. Contudo, em condições climáticas extremas fora da distribuição original (*Out-of-Distribution*, ex: chuva torrencial ou iluminação noturna total), o modelo pode apresentar degradação, demandando retreino com amostragem balanceada por câmera/dia.
* **Próximos Passos**: Exportação do modelo para **ONNX / TensorRT** para embarque em dispositivos de borda (*Edge AI*) em cabines de caminhões e pórticos industriais.
