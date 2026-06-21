"""
inference_raspberry.py — Inferência e benchmark na Raspberry Pi
Projeto: Classificação de Defeitos em Módulos Fotovoltaicos

Este script deve ser executado NA RASPBERRY PI.
Realiza inferência com os modelos .tflite e mede:
    - Latência por imagem (ms)
    - FPS estimado
    - Uso de memória RAM
    - Temperatura da CPU
    - Acurácia no conjunto de teste

Pré-requisitos na Raspberry Pi:
    pip install tflite-runtime opencv-python-headless numpy pandas scikit-learn

    Instalar tflite-runtime (mais leve que TensorFlow completo):
    pip install tflite-runtime

Uso:
    python inference_raspberry.py --test_dir data_split/test

Saída:
    results_raspberry/benchmark_completo.csv
    results_raspberry/comparativo_raspberry.png
"""

import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

# Tenta importar tflite_runtime (Raspberry) ou tensorflow (PC para debug)
try:
    import tflite_runtime.interpreter as tflite
    RUNTIME = 'tflite_runtime'
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    RUNTIME = 'tensorflow'

print(f"Runtime: {RUNTIME}")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────────────────────────────────────

IMG_SIZE    = (224, 224)
MODELS_DIR  = 'models'
RESULTS_DIR = 'results_raspberry'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Modelos a avaliar: nome → arquivo .tflite
TFLITE_MODELS = {
    'mobilenetv2_full_f16':   'models/mobilenetv2_f16.tflite',
    'mobilenetv2_int8':       'models/mobilenetv2_int8.tflite',
    'efficientnetb0_f16':     'models/efficientnetb0_f16.tflite',
    'efficientnetb0_int8':    'models/efficientnetb0_int8.tflite',
    'shufflenet_f16':         'models/shufflenet_f16.tflite',
    'shufflenet_int8':        'models/shufflenet_int8.tflite',
}

# ──────────────────────────────────────────────────────────────────────────────
# PRÉ-PROCESSAMENTO (mesmo pipeline do treino)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path):
    """
    Carrega e pré-processa uma imagem térmica para inferência.
    Aplica o mesmo pipeline usado no treinamento:
        1. Leitura em escala de cinza
        2. Resize bicúbico para 224×224
        3. Normalização min-max
        4. Equalização de histograma
        5. Colormap INFERNO
        6. Expansão de dimensão para batch
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Imagem não encontrada: {image_path}")

    # Resize bicúbico (mesmo do treino)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_CUBIC)

    # Normalização + equalização + colormap
    img_norm  = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_eq    = cv2.equalizeHist(img_norm)
    img_color = cv2.applyColorMap(img_eq, cv2.COLORMAP_INFERNO)
    img_rgb   = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Aplica o mesmo Rescaling do treino: x / 127.5 - 1.0  → range [-1, 1]
    # O modelo .tflite não inclui essa camada — deve ser feito aqui
    img_rgb = img_rgb / 127.5 - 1.0

    # Batch dimension
    return np.expand_dims(img_rgb, axis=0)

# ──────────────────────────────────────────────────────────────────────────────
# LEITURA DE TEMPERATURA DA CPU (Raspberry Pi)
# ──────────────────────────────────────────────────────────────────────────────

def get_cpu_temperature():
    """Lê a temperatura da CPU da Raspberry Pi."""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return float(f.read()) / 1000.0
    except FileNotFoundError:
        return None  # Não é Raspberry Pi


def get_ram_usage_mb():
    """Retorna uso atual de RAM em MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DO CONJUNTO DE TESTE
# ──────────────────────────────────────────────────────────────────────────────

def load_test_images(test_dir):
    """
    Carrega todas as imagens do conjunto de teste.
    Retorna lista de (image_path, label) onde label: 0=normal, 1=defect
    """
    test_dir = Path(test_dir)
    items = []

    for class_name in ['normal', 'defect']:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            print(f"⚠️  Pasta não encontrada: {class_dir}")
            continue

        label = 0 if class_name == 'normal' else 1
        images = list(class_dir.glob('*.*'))
        items.extend([(img, label) for img in images])

    print(f"Total de imagens de teste carregadas: {len(items)}")
    print(f"  Normal:  {sum(1 for _, l in items if l == 0)}")
    print(f"  Defeito: {sum(1 for _, l in items if l == 1)}")
    return items

# ──────────────────────────────────────────────────────────────────────────────
# BENCHMARK DE UM MODELO TFLITE
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_model(model_name, model_path, test_items, n_warmup=10):
    """
    Executa inferência em todo o conjunto de teste e mede:
        - Latência por imagem
        - FPS
        - Temperatura CPU (início e fim)
        - RAM usada
        - Acurácia
    """
    if not os.path.exists(model_path):
        print(f"⚠️  Modelo não encontrado: {model_path} — pulando.")
        return None

    print(f"\n{'─'*55}")
    print(f"  Benchmark: {model_name}")
    print(f"  Arquivo:   {model_path}")
    print(f"  Tamanho:   {os.path.getsize(model_path)/1e6:.2f} MB")
    print(f"{'─'*55}")

    # Carrega intérprete
    if RUNTIME == 'tflite_runtime':
        interpreter = tflite.Interpreter(model_path=model_path)
    else:
        interpreter = tflite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Warmup — aquece o intérprete antes de medir
    print(f"  Warmup ({n_warmup} imagens)...")
    warmup_img = preprocess_image(test_items[0][0])
    for _ in range(n_warmup):
        interpreter.set_tensor(input_details[0]['index'], warmup_img)
        interpreter.invoke()

    # Benchmark completo
    latencies = []
    y_true    = []
    y_pred    = []

    temp_inicio = get_cpu_temperature()
    ram_inicio  = get_ram_usage_mb()

    print(f"  Inferindo {len(test_items)} imagens...")
    for img_path, label in test_items:
        try:
            img = preprocess_image(img_path)
        except Exception as e:
            print(f"  Erro ao carregar {img_path}: {e}")
            continue

        # Medição de latência
        t_start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        t_end = time.perf_counter()

        latency_ms = (t_end - t_start) * 1000
        latencies.append(latency_ms)

        # Predição
        prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
        pred = 1 if prob > 0.5 else 0

        y_true.append(label)
        y_pred.append(pred)

    temp_fim = get_cpu_temperature()
    ram_fim  = get_ram_usage_mb()

    # Métricas
    latencies  = np.array(latencies)
    acc        = accuracy_score(y_true, y_pred) * 100
    lat_mean   = np.mean(latencies)
    lat_std    = np.std(latencies)
    lat_p95    = np.percentile(latencies, 95)
    fps        = 1000 / lat_mean

    print(f"\n  ✅ Resultados:")
    print(f"     Acurácia:          {acc:.2f}%")
    print(f"     Latência média:    {lat_mean:.1f} ms ± {lat_std:.1f} ms")
    print(f"     Latência P95:      {lat_p95:.1f} ms")
    print(f"     FPS estimado:      {fps:.1f}")
    if temp_inicio and temp_fim:
        print(f"     Temp CPU início:   {temp_inicio:.1f}°C")
        print(f"     Temp CPU fim:      {temp_fim:.1f}°C")
    if ram_inicio and ram_fim:
        print(f"     RAM usada:         {ram_fim:.1f} MB")

    print(f"\n  Relatório de classificação:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Defeito']))

    return {
        'modelo':           model_name,
        'arquivo':          model_path,
        'tamanho_mb':       round(os.path.getsize(model_path) / 1e6, 2),
        'acuracia_pct':     round(acc, 2),
        'latencia_media_ms':round(lat_mean, 2),
        'latencia_std_ms':  round(lat_std, 2),
        'latencia_p95_ms':  round(lat_p95, 2),
        'fps':              round(fps, 2),
        'temp_inicio_c':    temp_inicio,
        'temp_fim_c':       temp_fim,
        'ram_mb':           ram_fim,
        'n_imagens':        len(latencies),
    }

# ──────────────────────────────────────────────────────────────────────────────
# PREDIÇÃO EM IMAGEM INDIVIDUAL
# ──────────────────────────────────────────────────────────────────────────────

def predict_single(model_path, image_path, threshold=0.5):
    """
    Faz predição em uma única imagem com um modelo .tflite.
    Útil para testes rápidos em campo.
    """
    if RUNTIME == 'tflite_runtime':
        interpreter = tflite.Interpreter(model_path=model_path)
    else:
        interpreter = tflite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = preprocess_image(image_path)

    t_start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    latency_ms = (time.perf_counter() - t_start) * 1000

    prob  = interpreter.get_tensor(output_details[0]['index'])[0][0]
    label = 'DEFEITO' if prob > threshold else 'NORMAL'
    conf  = prob if prob > threshold else 1 - prob

    print(f"\nResultado:  {label}")
    print(f"Confiança:  {conf:.1%}")
    print(f"Latência:   {latency_ms:.1f} ms")
    return label, float(prob), latency_ms

# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICOS DE BENCHMARK
# ──────────────────────────────────────────────────────────────────────────────

def plot_benchmark(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    modelos = results_df['modelo'].tolist()
    colors  = plt.cm.tab10(np.linspace(0, 1, len(modelos)))

    # ── Latência média ────────────────────────────────────────────
    ax1 = axes[0]
    bars = ax1.bar(modelos, results_df['latencia_media_ms'], color=colors)
    ax1.errorbar(modelos, results_df['latencia_media_ms'],
                 yerr=results_df['latencia_std_ms'],
                 fmt='none', color='black', capsize=4)
    ax1.set_title('Latência Média por Imagem')
    ax1.set_ylabel('Latência (ms)')
    ax1.set_xticklabels(modelos, rotation=20, ha='right', fontsize=8)
    ax1.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, results_df['latencia_media_ms']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.0f}ms', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ── Acurácia ──────────────────────────────────────────────────
    ax2 = axes[1]
    bars2 = ax2.bar(modelos, results_df['acuracia_pct'], color=colors)
    min_acc = results_df['acuracia_pct'].min()
    ax2.set_title('Acurácia no Conjunto de Teste')
    ax2.set_ylabel('Acurácia (%)')
    ax2.set_ylim([max(0, min_acc - 5), 100])
    ax2.set_xticklabels(modelos, rotation=20, ha='right', fontsize=8)
    ax2.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars2, results_df['acuracia_pct']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ── Tamanho dos modelos ───────────────────────────────────────
    ax3 = axes[2]
    bars3 = ax3.bar(modelos, results_df['tamanho_mb'], color=colors)
    ax3.set_title('Tamanho dos Modelos')
    ax3.set_ylabel('Tamanho (MB)')
    ax3.set_xticklabels(modelos, rotation=20, ha='right', fontsize=8)
    ax3.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars3, results_df['tamanho_mb']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.1f}MB', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle('Benchmark Raspberry Pi — Inferência com Modelos TFLite',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'comparativo_raspberry.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nGráfico salvo em: {output_path}")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Benchmark TFLite na Raspberry Pi')
    parser.add_argument('--test_dir',   default='data_split/test', help='Pasta com imagens de teste')
    parser.add_argument('--models_dir', default='models',          help='Pasta com modelos .tflite')
    parser.add_argument('--predict',    default=None,              help='Caminho para predição em imagem única')
    parser.add_argument('--model',      default=None,              help='Modelo para predição única')
    args = parser.parse_args()

    # ── Modo predição única ───────────────────────────────────────
    if args.predict and args.model:
        print(f"\nModo: predição única")
        predict_single(args.model, args.predict)

    # ── Modo benchmark completo ───────────────────────────────────
    else:
        print(f"\nModo: benchmark completo")
        print(f"Diretório de teste: {args.test_dir}")
        print(f"Runtime: {RUNTIME}\n")

        test_items = load_test_images(args.test_dir)

        if not test_items:
            print("Nenhuma imagem encontrada. Verifique o diretório de teste.")
            exit(1)

        all_results = []
        for model_name, model_path in TFLITE_MODELS.items():
            result = benchmark_model(model_name, model_path, test_items)
            if result:
                all_results.append(result)

        if all_results:
            df = pd.DataFrame(all_results)
            csv_path = os.path.join(RESULTS_DIR, 'benchmark_completo.csv')
            df.to_csv(csv_path, index=False)

            print(f"\n{'='*70}")
            print("  RESUMO FINAL — BENCHMARK RASPBERRY PI")
            print(f"{'='*70}")
            print(f"{'Modelo':<28} {'Tamanho':>9} {'Acurácia':>10} {'Latência':>10} {'FPS':>7}")
            print("-" * 70)
            for _, row in df.iterrows():
                print(
                    f"  {row['modelo']:<26} "
                    f"{row['tamanho_mb']:>7.1f}MB "
                    f"{row['acuracia_pct']:>9.2f}% "
                    f"{row['latencia_media_ms']:>8.1f}ms "
                    f"{row['fps']:>6.1f}"
                )

            print(f"\nCSV salvo em: {csv_path}")
            plot_benchmark(df)
