# 🌞 Detecção de Defeitos em Módulos Fotovoltaicos por Imagens Termográficas

> Estudo comparativo de redes neurais convolucionais leves com quantização pós-treinamento para deploy em sistema embarcado (Raspberry Pi), com rastreamento completo de experimentos via MLflow.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.x-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![TFLite](https://img.shields.io/badge/TensorFlow-Lite-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4%20Model%20B-A22846?logo=raspberrypi&logoColor=white)](https://www.raspberrypi.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Sumário

- [Sobre o projeto](#-sobre-o-projeto)
- [Resultados principais](#-resultados-principais)
- [Pipeline](#-pipeline)
- [Experiment Tracking (MLflow)](#-experiment-tracking-mlflow)
- [Estrutura do repositório](#-estrutura-do-repositório)
- [Dataset](#-dataset)
- [Instalação](#-instalação)
- [Como usar](#-como-usar)
- [Decisões técnicas](#-decisões-técnicas)
- [Resultados detalhados](#-resultados-detalhados)
- [Hardware utilizado](#-hardware-utilizado)
- [Limitações e trabalhos futuros](#-limitações-e-trabalhos-futuros)
- [Referências](#-referências)
- [Autores](#-autores)
- [Licença](#-licença)

---

## 🔍 Sobre o projeto

A geração distribuída solar fotovoltaica é hoje a **segunda maior fonte da matriz elétrica brasileira**, mas a degradação progressiva dos módulos — hotspots, falhas de diodo, sombreamento, rachaduras, sujidade — reduz a eficiência e pode causar danos irreversíveis ao sistema.

Este projeto desenvolve um **pipeline completo de classificação automática de defeitos** em módulos fotovoltaicos a partir de imagens termográficas, com foco explícito em **viabilidade de deploy em hardware embarcado de baixo custo**:

```
Dataset → Pré-processamento → Treinamento (3 arquiteturas) → Quantização → Benchmark na Raspberry Pi
```

Diferente da maior parte da literatura — que maximiza acurácia em ambiente computacional irrestrito — este trabalho avalia explicitamente o **trade-off entre acurácia, tamanho do modelo e latência de inferência** em um Raspberry Pi 4, validando a aplicação real em inspeção de campo.

---

## 🏆 Resultados principais

| Configuração | Tamanho | Acurácia (Raspberry Pi) | Latência | FPS |
|---|---|---|---|---|
| **MobileNetV2 — INT8** ⭐ | 3,1 MB | 80,30% | **65,8 ms** | **15,21** |
| MobileNetV2 — Float16 | 5,2 MB | 80,73% | 100,6 ms | 9,94 |
| EfficientNetB0 — Float16 | 8,1 MB | **81,37%** | 201,7 ms | 4,96 |
| EfficientNetB0 — INT8 | 4,9 MB | 73,13% | 242,5 ms | 4,12 |

> ⭐ **MobileNetV2 INT8** é a configuração recomendada para inspeção de campo em tempo real — melhor latência e FPS com apenas 3,1 MB e queda de acurácia de apenas 1,6 p.p. em relação ao modelo completo treinado no PC (81,90%).

---

## 🔄 Pipeline

```mermaid
flowchart LR
    A[Dataset<br/>20.000 imagens] --> B[Split<br/>70/15/15]
    B --> C[Pré-processamento<br/>INFERNO + Eq. Histograma]
    C --> D[Treinamento<br/>MobileNetV2 / EfficientNetB0 / MobileNetV3Small]
    D --> E[Avaliação<br/>Acurácia, Recall, F1]
    E --> F[Quantização<br/>Float16 / INT8]
    F --> G[Benchmark<br/>Raspberry Pi 4]
```

**1. Pré-processamento** — leitura em escala de cinza → resize bicúbico 224×224 → normalização min-max → equalização de histograma → pseudocolorização com colormap **INFERNO** (converte gradiente térmico em gradiente de cor compatível com pesos ImageNet)

**2. Treinamento em 2 fases** — congelamento da base + treino do head → fine-tuning das camadas superiores com LR reduzido (AdamW + label smoothing)

**3. Quantização pós-treinamento** — conversão para TensorFlow Lite em Float16 e INT8 com dataset representativo de calibração

**4. Benchmark embarcado** — inferência real na Raspberry Pi 4, medindo latência, FPS, acurácia e uso de RAM

---

## 🧪 Experiment Tracking (MLflow)

O projeto utiliza **MLflow** para documentar todo o processo de experimentação com rigor de engenharia de ML:

- **Histórico de hipóteses** — Cada iteração (v1 → v2 → v3 → final) registrada com hipótese, conclusão e próximo passo
- **Ablation studies** — Impacto isolado do colormap INFERNO (+4%), equalização de histograma (+2%) e interpolação bicúbica (+0.5%)
- **Comparativo de arquiteturas** — Métricas lado a lado com hiperparâmetros completos
- **Métricas por época** — Curvas de loss e accuracy logadas automaticamente durante treino
- **Métricas de quantização** — Perda de acurácia e compressão por formato (F16, INT8)
- **Artifacts** — Confusion matrices, modelos `.keras` e `.tflite`, históricos de treino

### Visualização

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Acesse: http://localhost:5000
```

### Estrutura dos Runs

```
pv-fault-detection (experiment)
├── historico_experimentos_completo (parent run)
│   ├── mobilenetv2_v1_baseline
│   ├── mobilenetv2_v2_head_melhorado
│   ├── mobilenetv2_v3_more_finetuning
│   ├── efficientnetb0_v1_baseline
│   ├── efficientnetb0_v2_conservative
│   ├── mobilenetv3small_v1_standard_aug
│   ├── mobilenetv3small_v2_light_aug
│   ├── ablation_no_colormap
│   ├── ablation_no_equalize_hist
│   └── ablation_bilinear_vs_bicubic
└── pipeline_completo_YYYYMMDD (parent run)
    ├── mobilenetv2_v_final (nested run)
    ├── efficientnetb0_v_final (nested run)
    └── mobilenetv3small_v_final (nested run)
```

---

## 📁 Estrutura do repositório

```
pv-fault-detection-edge-ai/
├── src/
│   ├── prepare_dataset.py              # Organiza imagens em data/normal e data/defect
│   ├── pipeline_final.py              # Pipeline de treino original (sem tracking)
│   ├── pipeline_mlflow.py             # Pipeline de treino com MLflow tracking
│   ├── evaluate_and_quantize_mlflow.py # Avaliação + quantização (sem retreinar)
│   ├── register_experiment_history.py # Registra histórico de experimentos no MLflow
│   ├── inference_raspberry.py         # Benchmark de inferência (Raspberry Pi)
│   └── predict_individual.py         # Predição/visualização individual
├── notebooks/
│   ├── pipeline_final.ipynb
│   └── pipeline_final_notebook_estruturado.ipynb
├── models/                            # Modelos treinados (.keras, .tflite) — não versionado
├── results/                           # Gráficos e CSVs de resultados
├── results_raspberry/                 # Benchmark de inferência embarcada
├── data/                              # Dataset organizado (não versionado)
│   ├── normal/
│   └── defect/
├── data_split/                        # Split 70/15/15 (não versionado)
├── mlflow.db                          # Database local do MLflow (não versionado)
├── requirements.txt                   # Dependências PC (treino)
├── requirements-raspberry.txt         # Dependências Raspberry Pi (inferência)
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🗂️ Dataset

**[Infrared Solar Modules Dataset](https://www.kaggle.com/datasets/marcosgabriel/infrared-solar-modules)** (Kaggle) — 20.000 imagens termográficas, resolução nativa de **40×24 pixels**.

| Classe original | Quantidade | Classe binária |
|---|---|---|
| No-Anomaly | 10.000 | **Normal** |
| Cell, Cell-Multi, Cracking, Diode, Diode-Multi, Hot-Spot, Hot-Spot-Multi, Offline-Module, Shadowing, Soiling, Vegetation | 10.000 | **Defeito** |

Divisão estratificada: **70% treino / 15% validação / 15% teste** (seed fixo = 42 para reprodutibilidade).

> ⚠️ O dataset não está incluído neste repositório por questões de tamanho. Baixe do Kaggle e organize com `python src/prepare_dataset.py`.

---

## ⚙️ Instalação

### No PC (treino, avaliação e quantização)

```bash
git clone https://github.com/seu-usuario/pv-fault-detection-edge-ai.git
cd pv-fault-detection-edge-ai

python -m venv env
source env/bin/activate        # Linux/Mac
env\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### Na Raspberry Pi (inferência embarcada)

```bash
pip install -r requirements-raspberry.txt
```

---

## 🚀 Como usar

### 1. Preparar dataset

Baixe o dataset do Kaggle e coloque em `InfraredSolarModules/`, depois:

```bash
python src/prepare_dataset.py
```

### 2. Registrar histórico de experimentos (opcional, rápido)

```bash
python src/register_experiment_history.py
```

### 3. Treinar com MLflow tracking

```bash
python src/pipeline_mlflow.py
```

Executa: split → pré-processamento → treino (3 arquiteturas) → avaliação → quantização → avaliação quantizada — com logging completo no MLflow.

### 4. Visualizar experimentos

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### 5. Copiar modelos para Raspberry Pi

```bash
scp models/*.tflite pi@<ip-da-raspberry>:~/pv-fault-detection-edge-ai/models/
```

### 6. Benchmark na Raspberry Pi

```bash
python src/inference_raspberry.py --test_dir data_split/test
```

### 7. Predição individual

```bash
python src/predict_individual.py --image caminho/imagem.jpg --model models/mobilenetv2.keras
```

---

## 🧠 Decisões técnicas

| Decisão | Justificativa |
|---------|---------------|
| Colormap INFERNO | Imagens térmicas grayscale → RGB permite aproveitar features ImageNet. Ablation: +4% vs. grayscale puro |
| Equalização de histograma | Faixa dinâmica estreita (9-255) nas imagens térmicas. Ablation: +2% |
| Interpolação bicúbica | Upscale extremo (5.6×) se beneficia de interpolação suave. Ablation: +0.5% vs. bilinear |
| Augmentation leve para MobileNetV3Small | Rede menor + imagens 40×24: augmentation agressivo destrói informação |
| Head melhorado para MobileNetV2 | Dense 256+BN+Dense 64 melhora capacidade discriminativa vs. head simples |
| Early stopping conservador para EfficientNet | Tende a overfitting — patience menor controla o gap treino-val |
| Label smoothing 0.1 | Reduz overconfidence e melhora generalização em dataset com possível ruído nos labels |
| AdamW no fine-tuning | Weight decay desacoplado previne overfitting nas camadas descongeladas |
| Quantização INT8 com fallback TFLITE_BUILTINS | Resolve incompatibilidade XNNPACK mantendo precisão |

---

## 📊 Resultados detalhados

### Treinamento (conjunto de teste, PC)

| Arquitetura | Acurácia | Precisão Defeito | Recall Defeito | F1 Macro |
|---|---|---|---|---|
| **MobileNetV2** | 81,90% | 75,88% | 93,53% | 81,65% |
| **EfficientNetB0** | 81,40% | 75,08% | 94,00% | 81,10% |
| MobileNetV3Small | 50,00%¹ | — | — | — |

¹ Não convergiu em nenhuma das configurações testadas — excluída das etapas de quantização e benchmark.

### Quantização pós-treinamento

| Modelo / Formato | Tamanho | Redução | Acurácia | Δ vs. Full |
|---|---|---|---|---|
| MobileNetV2 — Full | 27,3 MB | — | 81,90% | — |
| MobileNetV2 — Float16 | 5,2 MB | 81% | 82,17% | **+0,27%** |
| MobileNetV2 — INT8 | 3,1 MB | 89% | 80,50% | −1,40% |
| EfficientNetB0 — Full | 32,6 MB | — | 81,40% | — |
| EfficientNetB0 — Float16 | 8,1 MB | 75% | 81,40% | 0,00% |
| EfficientNetB0 — INT8 | 4,9 MB | 85% | 77,33% | −4,07% |

### Benchmark Raspberry Pi 4 (n = 3.000 imagens)

| Modelo / Formato | Acurácia | Latência média | FPS |
|---|---|---|---|
| MobileNetV2 — Float16 | 80,73% | 100,6 ms | 9,94 |
| **MobileNetV2 — INT8** | 80,30% | **65,8 ms** | **15,21** |
| **EfficientNetB0 — Float16** | **81,37%** | 201,7 ms | 4,96 |
| EfficientNetB0 — INT8 | 73,13% | 242,5 ms | 4,12 |

> 📌 Achado relevante: o EfficientNetB0 INT8 apresentou comportamento atípico na Raspberry Pi — latência *maior* que o Float16, sugerindo que o mecanismo de Squeeze-and-Excitation não se beneficia das otimizações de inteiro do processador ARM Cortex-A72 da mesma forma que o MobileNetV2.

---

## 💻 Hardware utilizado

| | Treinamento | Inferência embarcada |
|---|---|---|
| **Dispositivo** | PC | Raspberry Pi 4 Model B |
| **Processador** | Intel Core i5-1135G7 | ARM Cortex-A72 quad-core |
| **RAM** | 16 GB | 4 GB |
| **Armazenamento** | SSD NVMe | microSD |
| **Sistema** | Windows | Raspberry Pi OS (64-bit) |
| **Runtime** | TensorFlow / Keras 3.x | TensorFlow Lite Runtime |

---

## 🔭 Limitações e trabalhos futuros

**Limitações identificadas:**
- Resolução nativa das imagens de apenas 40×24 pixels — limita a informação espacial disponível
- MobileNetV3Small não convergiu em nenhuma configuração testada
- Dataset público único — sem validação com captura térmica própria em campo
- Sistema opera sobre imagens pré-capturadas, sem aquisição térmica em tempo real

**Direções futuras:**
- [ ] Classificação multiclasse (identificar o tipo específico de defeito entre as 11 classes)
- [ ] Integração com câmera térmica real (FLIR Lepton / AMG8833) na Raspberry Pi
- [ ] Avaliação de arquiteturas adicionais (EfficientNet-Lite, MobileOne)
- [ ] Conversão para TensorRT / ONNX Runtime para redução adicional de latência
- [ ] Coleta de dataset próprio em instalações fotovoltaicas brasileiras
- [ ] Implementar AUC-ROC e análise de threshold para otimizar recall de defeito

---

## 📚 Referências

- KORKMAZ, D.; ACIKGOZ, H. An efficient fault classification method in solar photovoltaic modules using transfer learning and multi-scale convolutional neural network. **Engineering Applications of Artificial Intelligence**, v. 113, 2022.
- SANDLER, M. et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks. **CVPR**, 2018.
- TAN, M.; LE, Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. **ICML**, 2019.
- VIEIRA, R. G. Aplicação de técnicas de inteligência artificial para identificação de faltas em módulos fotovoltaicos. Tese (Doutorado) — UFRN, 2021.
- Dataset: [Infrared Solar Modules Dataset](https://www.kaggle.com/datasets/marcosgabriel/infrared-solar-modules) (Kaggle)

---

## 👥 Autores

Trabalho de Conclusão de Curso — Engenharia Elétrica — Faculdade Matias Machline (FMM), Manaus-AM

- **Isaac Davi da Silva Martins**
- **Sandoval Marques de Oliveira Filho**

Orientador: Prof. Emerson Leão Brito do Nascimento

---

## 📄 Licença

Este projeto está licenciado sob a licença MIT — veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

<p align="center">
  Feito com 🔆 para um setor solar mais inteligente e acessível.
</p>
