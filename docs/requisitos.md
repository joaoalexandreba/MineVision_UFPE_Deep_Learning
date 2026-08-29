# 📋 Especificação de Requisitos de Software (SRS)
> **Projeto Integrador: Sistema de Visão Computacional para Detecção e Segmentação de Cargas (Dataset Harsh)**  
> *Disciplina: Engenharia de Software para IA e Frameworks Profundos - UFPE*

---

## 1. Visão do Produto
* **Nome do Sistema**: Sistema de Segmentação de Cargas em Caminhões (*Harsh Cargo Vision*).
* **Problema Resolvido**: Dificuldade de mensuração e fiscalização automatizada de carga de carvão mineral e detecção de caçambas de caminhões em condições de poeira e iluminação adversas.
* **Público-Alvo / Usuários**: Operadores de pátio de mineração, equipes de logística e sistemas automatizados de pesagem/volumetria.
* **Valor Entregue**: Redução de tempo de inspeção, eliminação de fraudes volumétricas e operação contínua em tempo real (>50 FPS).

---

## 2. Requisitos Funcionais (RF)

| ID | Nome do Requisito | Descrição | Módulo Responsável |
| :--- | :--- | :--- | :--- |
| **RF01** | **Download e Carga de Dados** | O sistema deve conectar à API do Roboflow e baixar/carregar o dataset estruturado no formato YOLOv11. | `src/data.py` (`load_dataset`) |
| **RF02** | **Pré-processamento e Tensores PyTorch** | O sistema deve transformar imagens brutas em Tensores PyTorch normalizados no formato `(3, 640, 640)`. | `src/preprocessing.py` (`preprocess`, `HarshDataset`) |
| **RF03** | **Divisão Estratificada 80/20** | O sistema deve realizar split balanceado 80% treino / 20% teste com NumPy e semente fixa (`SEED = 42`), gerando `train.txt` e `test.txt`. | `src/preprocessing.py` (`split_dataset`) |
| **RF04** | **Estruturação de Pastas e Configuração** | O sistema deve estruturar as imagens/labels e gerar o arquivo de configuração `data.yaml`. | `src/preprocessing.py` (`model_structure_configuration`) |
| **RF05** | **Treinamento do Modelo de IA** | O sistema deve treinar o modelo neural YOLOv11 com otimizador AdamW, registrando métricas de perda e salvando `best.pt`. | `src/training.py` (`train_model`), `src/models.py` |
| **RF06** | **Avaliação Estatística e Intervalos de Confiança** | O sistema deve avaliar o modelo no conjunto de teste, calcular Precision, Recall, F1, mAP e IC 95% via t-Student. | `src/evaluation.py` (`evaluate_model`) |
| **RF07** | **Geração de Gráficos e Relatórios** | O sistema deve salvar a Matriz de Confusão e Boxplot comparativo em PNG e exportar relatórios estruturados em CSV. | `src/evaluation.py` |
| **RF08** | **Inferência em Imagens Avulsas** | O sistema deve receber novas fotos avulsas, processar na rede e salvar a imagem com as máscaras e caixas anotadas. | `src/inference.py` (`predict_image`) |

---

## 3. Requisitos Não Funcionais (RNF)

| ID | Categoria | Descrição | Critério de Aceitação |
| :--- | :--- | :--- | :--- |
| **RNF01** | **Desempenho da IA** | O modelo deve atingir desempenho preditivo de alta acurácia no teste. | $F1\text{-score} \ge 0.80$ no conjunto de teste (*Superado: $F1 = 0.9970$*). |
| **RNF02** | **Tempo de Resposta** | O tempo de inferência por frame deve viabilizar operação em tempo real. | Latência $< 50\text{ms}$ por imagem (*Alcançado: ~17ms / >50 FPS*). |
| **RNF03** | **Modularidade** | O código deve seguir princípios de alta coesão e baixo acoplamento. | Código dividido em pacotes com responsabilidade única (`src/`). |
| **RNF04** | **Tipagem Estática** | O código Python deve utilizar type hints claros e contratos explícitos. | 100% das funções com anotações de tipo de entrada e retorno. |
| **RNF05** | **Testabilidade** | O sistema deve possuir suíte de testes unitários automatizados. | Testes com `unittest` cobrindo tensores, split e métricas com 100% de sucesso. |
| **RNF06** | **Reprodutibilidade** | Qualquer execução com a mesma semente deve gerar resultados idênticos. | Fixação de semente determinística `SEED = 42` para NumPy, Random e PyTorch. |
| **RNF07** | **Portabilidade** | O pipeline deve rodar tanto localmente (via `uv`/`venv`) quanto no Google Colab. | Detecção automática de ambiente e `pyproject.toml` configurado. |

---

## 4. Matriz de Rastreabilidade

| Requisito | Código de Implementação | Teste Unitário Associado |
| :--- | :--- | :--- |
| **RF01** | `src/data.py:load_dataset` | `tests/test_dataset.py:test_dataset_instantiation_and_len` |
| **RF02** | `src/preprocessing.py:HarshDataset` | `tests/test_dataset.py:test_tensor_transformation_and_shape` |
| **RF03** | `src/preprocessing.py:split_dataset` | `tests/test_preprocessing.py:test_split_proportions_80_20` |
| **RF04** | `src/preprocessing.py:model_structure_configuration` | `tests/test_preprocessing.py:test_split_files_saved` |
| **RF05** | `src/training.py:train_model` | `tests/test_dataset.py:test_neural_feature_extractor_forward` |
| **RF06 / RF07** | `src/evaluation.py:evaluate_model` | `tests/test_evaluation.py:test_confidence_interval_calculation` |
| **RF08** | `src/inference.py:predict_image` | `tests/test_inference.py:test_inference_missing_image_raises_error` |
