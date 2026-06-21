"""
pipeline_final.py — Pipeline completo de treinamento, avaliação e quantização
Projeto: Classificação de Defeitos em Módulos Fotovoltaicos por Imagens Termográficas
TCC — Engenharia Elétrica

Arquiteturas: MobileNetV2 | EfficientNetB0 | MobileNetV3Small (ShuffleNet)
Configurações individuais por arquitetura (baseado nos experimentos v1/v2/v3)

Correções aplicadas nesta versão final:
    [1] Rescaling nativa em vez de preprocess_input — elimina erro TrueDivide
    [2] Formato SavedModel em vez de .h5 — elimina incompatibilidade do Keras >= 3.x
    [3] save_weights_only=True no checkpoint — sem problemas de serialização
    [4] Quantização integrada no mesmo script — sem arquivos auxiliares
    [5] Avaliação TFLite com fallback para diferentes versões do TF

Uso:
    python pipeline_final.py

Saída:
    models/<arch>/                  → SavedModel (treino)
    models/<arch>_f16.tflite        → Float16
    models/<arch>_int8.tflite       → INT8
    results/confusion_matrix_*.png
    results/comparativo_acuracia.png
    results/comparativo_quantizacao.png
    results/resumo_comparativo.csv
    results/resumo_quantizacao.csv
"""

import os
import cv2
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)

# ──────────────────────────────────────────────────────────────────────────────
# SEEDS E CONFIGURAÇÕES GLOBAIS
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
AUTOTUNE    = tf.data.AUTOTUNE
SOURCE_DIR  = 'data'          # data/normal/  e  data/defect/
DATA_DIR    = 'data_split'
MODELS_DIR  = 'models'
RESULTS_DIR = 'results'

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f'TensorFlow: {tf.__version__}')
print(f'Keras:      {tf.keras.__version__}')
print(f'GPU:        {len(tf.config.list_physical_devices("GPU")) > 0}\n')

# ──────────────────────────────────────────────────────────────────────────────
# [1] CONFIGURAÇÕES INDIVIDUAIS POR ARQUITETURA
# ──────────────────────────────────────────────────────────────────────────────

ARCH_CONFIGS = {
    'mobilenetv2': {
        'epochs_phase1':   10,
        'epochs_phase2':   30,      # mais épocas — respondeu bem ao fine-tuning
        'unfreeze_layers': 40,
        'head':            'melhorado',
        'dropout_1':       0.4,
        'dropout_2':       0.3,
        'lr_phase2':       1e-5,
        'weight_decay':    1e-4,
        'es_patience':     6,
        'augment_level':   'normal',
    },
    'efficientnetb0': {
        'epochs_phase1':   10,
        'epochs_phase2':   25,      # conservador — tende a overfitting
        'unfreeze_layers': 35,
        'head':            'simples',
        'dropout_1':       0.3,
        'dropout_2':       None,
        'lr_phase2':       1e-5,
        'weight_decay':    1e-4,
        'es_patience':     5,
        'augment_level':   'normal',
    },
    'shufflenet': {
        'epochs_phase1':   10,
        'epochs_phase2':   20,      # igual v1 — comprovadamente estável
        'unfreeze_layers': 20,      # rede menor, menos camadas
        'head':            'simples',
        'dropout_1':       0.3,
        'dropout_2':       None,
        'lr_phase2':       1e-5,
        'weight_decay':    5e-5,    # weight decay menor para rede menor
        'es_patience':     4,
        'augment_level':   'leve',  # augmentation leve — preserva info de imagens pequenas
    },
}

ARCHITECTURES = list(ARCH_CONFIGS.keys())

# ──────────────────────────────────────────────────────────────────────────────
# [2] DIVISÃO DO DATASET (70/15/15)
# ──────────────────────────────────────────────────────────────────────────────

def split_dataset(source_dir, output_dir, splits=(0.70, 0.15, 0.15)):
    random.seed(SEED)
    for class_name in ['normal', 'defect']:
        images  = list(Path(source_dir, class_name).glob('*.*'))
        random.shuffle(images)
        n       = len(images)
        n_train = int(n * splits[0])
        n_val   = int(n * splits[1])
        subsets = {
            'train': images[:n_train],
            'val':   images[n_train:n_train + n_val],
            'test':  images[n_train + n_val:]
        }
        for subset, files in subsets.items():
            dest = Path(output_dir, subset, class_name)
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, dest / f.name)
        print(f'  {class_name:8s} → {n_train:5d} treino | {n_val:5d} val | {len(subsets["test"]):5d} teste')
    print('Split concluído!\n')


if not os.path.exists(DATA_DIR):
    print('Dividindo dataset...')
    split_dataset(SOURCE_DIR, DATA_DIR)
else:
    print(f'Pasta "{DATA_DIR}" já existe — split ignorado.\n')

# ──────────────────────────────────────────────────────────────────────────────
# [3] PRÉ-PROCESSAMENTO E DATASET
# ──────────────────────────────────────────────────────────────────────────────

# Dois níveis de augmentation calibrados por arquitetura
augment_normal = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomZoom(0.12),
    tf.keras.layers.RandomBrightness(0.15),
    tf.keras.layers.RandomContrast(0.15),
], name='augmentation_normal')

augment_leve = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomBrightness(0.1),
], name='augmentation_leve')


def apply_colormap(img_np):
    """
    Converte imagem térmica (escala de cinza) para RGB com colormap INFERNO.
    Aplica equalização de histograma para melhorar contraste em imagens
    com faixa dinâmica estreita (dataset: 40x24px, faixa variável de 9-255).
    """
    img_gray  = img_np[:, :, 0].astype(np.uint8)
    img_norm  = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
    img_eq    = cv2.equalizeHist(img_norm)
    img_color = cv2.applyColorMap(img_eq, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB).astype(np.float32)


def preprocess_thermal(image, label):
    """Aplica colormap por imagem dentro do batch via tf.numpy_function."""
    def process_single(img):
        img_rgb = tf.numpy_function(apply_colormap, [img], tf.float32)
        img_rgb.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
        return img_rgb
    image = tf.map_fn(process_single, image, fn_output_signature=tf.float32)
    return image, label


def load_dataset(subset, augment_level=None):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, subset),
        labels='inferred',
        label_mode='binary',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(subset == 'train'),
        seed=SEED,
        color_mode='grayscale',   # lê 1 canal — correto para imagens térmicas
        interpolation='bicubic'   # melhor qualidade no upscale 40x24 → 224x224
    )
    ds = ds.map(preprocess_thermal, num_parallel_calls=AUTOTUNE)
    if augment_level == 'normal':
        ds = ds.map(
            lambda x, y: (augment_normal(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
    elif augment_level == 'leve':
        ds = ds.map(
            lambda x, y: (augment_leve(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
    return ds.prefetch(AUTOTUNE)


print('Carregando datasets de validação e teste...')
val_ds  = load_dataset('val')
test_ds = load_dataset('test')
print('Datasets carregados.\n')

# Verificação visual do pipeline
print('Gerando verificação visual do pipeline...')
train_check = load_dataset('train', augment_level='normal')
plt.figure(figsize=(14, 3))
for images, labels in train_check.take(1):
    for i in range(min(8, len(images))):
        plt.subplot(1, 8, i + 1)
        plt.imshow(images[i].numpy().astype(np.uint8))
        plt.title('Defeito' if int(labels[i]) == 1 else 'Normal', fontsize=7)
        plt.axis('off')
plt.suptitle('Pipeline — INFERNO + Equalização de Histograma + Augmentation',
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'verificacao_pipeline.png'), dpi=150)
plt.close()
del train_check
print('Verificação salva.\n')

# ──────────────────────────────────────────────────────────────────────────────
# [4] DEFINIÇÃO DAS ARQUITETURAS
# Rescaling nativa substitui preprocess_input — elimina erro de serialização
# ──────────────────────────────────────────────────────────────────────────────

def build_model(arch_name, cfg):
    """
    Constrói o modelo com Transfer Learning.

    Decisão de design: preprocess_input substituído por Rescaling nativa.
    - MobileNetV2 / MobileNetV3: divide por 127.5, subtrai 1 → [-1, 1]
    - EfficientNetB0: sem rescaling externo (normaliza internamente)

    Head adaptativo por arquitetura:
    - 'melhorado': Dense(256) + BN + Dropout + Dense(64) → MobileNetV2
    - 'simples':   GlobalAvgPool + Dropout → EfficientNet e ShuffleNet
    """
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name='input_image')

    if arch_name == 'mobilenetv2':
        x    = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1.0)(inputs)
        base = tf.keras.applications.MobileNetV2(
            input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet'
        )
    elif arch_name == 'efficientnetb0':
        x    = inputs   # EfficientNet normaliza internamente
        base = tf.keras.applications.EfficientNetB0(
            input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet'
        )
    elif arch_name == 'shufflenet':
        x    = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1.0)(inputs)
        base = tf.keras.applications.MobileNetV3Small(
            input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet'
        )
    else:
        raise ValueError(f'Arquitetura desconhecida: {arch_name}')

    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    if cfg['head'] == 'melhorado':
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(cfg['dropout_1'])(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(cfg['dropout_2'])(x)
    else:
        x = tf.keras.layers.Dropout(cfg['dropout_1'])(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs, name=arch_name), base

# ──────────────────────────────────────────────────────────────────────────────
# [5] TREINAMENTO EM 2 FASES
# ──────────────────────────────────────────────────────────────────────────────

def train_model(arch_name):
    global val_ds, test_ds

    cfg = ARCH_CONFIGS[arch_name]

    print(f"\n{'='*65}")
    print(f"  Treinando: {arch_name.upper()}")
    print(f"  Head: {cfg['head']} | Aug: {cfg['augment_level']} | "
          f"Épocas: {cfg['epochs_phase1']}+{cfg['epochs_phase2']} | "
          f"Unfreeze: {cfg['unfreeze_layers']}")
    print(f"{'='*65}")

    train_ds = load_dataset('train', augment_level=cfg['augment_level'])
    model, base = build_model(arch_name, cfg)

    # Caminhos de saída
    save_path = os.path.join(MODELS_DIR, f'{arch_name}.keras')  # formato nativo Keras 3.x
    best_weights_path = os.path.join(MODELS_DIR,
                                     f'{arch_name}_best.weights.h5')  # só pesos

    callbacks = [
        # [2] save_weights_only=True — salva só os pesos, sem o grafo
        # Elimina problemas de serialização no checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            best_weights_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=cfg['es_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # ── FASE 1: Head com base congelada ───────────────────────────
    print(f'\n[FASE 1] Head — base congelada | LR=1e-3')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=cfg['epochs_phase1'], callbacks=callbacks
    )

    # ── FASE 2: Fine-tuning com AdamW ────────────────────────────
    print(f"\n[FASE 2] Fine-tuning — últimas {cfg['unfreeze_layers']} camadas | "
          f"LR={cfg['lr_phase2']}")
    base.trainable = True
    for layer in base.layers[:-cfg['unfreeze_layers']]:
        layer.trainable = False

    n_trainable = sum(tf.size(w).numpy() for w in model.trainable_variables)
    print(f'  Parâmetros treináveis: {n_trainable:,}')

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=cfg['lr_phase2'],
            weight_decay=cfg['weight_decay']
        ),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    h2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=cfg['epochs_phase2'], callbacks=callbacks
    )

    # ── [2] Salva no formato SavedModel ──────────────────────────
    # SavedModel inclui grafo + pesos de forma estável entre versões do TF
    model.save(save_path)
    print(f'\nModelo salvo em: {save_path}')

    # Histórico completo (fase 1 + fase 2)
    history = {}
    for key in h1.history:
        history[key] = h1.history[key] + h2.history[key]
    pd.DataFrame(history).to_csv(
        os.path.join(RESULTS_DIR, f'history_{arch_name}.csv'), index=False
    )

    return model, history

# ──────────────────────────────────────────────────────────────────────────────
# [6] AVALIAÇÃO NO CONJUNTO DE TESTE
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(arch_name):
    """Carrega SavedModel e avalia no conjunto de teste."""
    print(f'\n[{arch_name}] Avaliando no conjunto de teste...')

    model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, f'{arch_name}.keras')
    )

    y_true, y_pred_prob = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred_prob.extend(probs.flatten())
        y_true.extend(labels.numpy().flatten())

    y_true      = np.array(y_true, dtype=int)
    y_pred_prob = np.array(y_pred_prob)
    y_pred      = (y_pred_prob > 0.5).astype(int)

    report   = classification_report(y_true, y_pred,
                                      target_names=['Normal', 'Defeito'],
                                      output_dict=True)
    accuracy = report['accuracy']

    print(f"\n{'─'*50}")
    print(f"  {arch_name.upper()} — Acurácia: {accuracy*100:.2f}%")
    print(f"{'─'*50}")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Defeito']))

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Defeito'],
                yticklabels=['Normal', 'Defeito'])
    plt.title(f'Matriz de Confusão — {arch_name}')
    plt.ylabel('Real'); plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_{arch_name}.png'),
                dpi=150)
    plt.close()

    del model
    tf.keras.backend.clear_session()
    return accuracy, report

# ──────────────────────────────────────────────────────────────────────────────
# [7] QUANTIZAÇÃO PARA TFLITE
# ──────────────────────────────────────────────────────────────────────────────

def get_representative_dataset():
    """Dataset representativo para calibração da quantização INT8."""
    def generator():
        count = 0
        cal_ds = load_dataset('val')
        for images, _ in cal_ds:
            if count >= 100:
                break
            yield [tf.cast(images, tf.float32)]
            count += 1
    return generator


def quantize_model(arch_name):
    """
    Converte SavedModel para Float16 e INT8.
    Carrega do SavedModel — sem nenhum problema de compatibilidade.
    """
    model_path = os.path.join(MODELS_DIR, f'{arch_name}.keras')
    print(f"\n{'─'*55}")
    print(f"  Quantizando: {arch_name.upper()}")

    model = tf.keras.models.load_model(model_path)

    size_full = os.path.getsize(model_path) / 1e6

    # ── Float16 ──────────────────────────────────────────────────
    print(f'  [Float16] Convertendo...')
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tflite_f16 = conv.convert()
    path_f16   = os.path.join(MODELS_DIR, f'{arch_name}_f16.tflite')
    open(path_f16, 'wb').write(tflite_f16)
    print(f'  [Float16] Salvo em: {path_f16}')

    # ── INT8 ─────────────────────────────────────────────────────
    print(f'  [INT8]    Convertendo...')
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = get_representative_dataset()
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,   # fallback — resolve erro XNNPACK
    ]
    conv.inference_input_type  = tf.float32
    conv.inference_output_type = tf.float32
    tflite_int8 = conv.convert()
    path_int8   = os.path.join(MODELS_DIR, f'{arch_name}_int8.tflite')
    open(path_int8, 'wb').write(tflite_int8)
    print(f'  [INT8]    Salvo em: {path_int8}')

    size_f16  = len(tflite_f16)  / 1e6
    size_int8 = len(tflite_int8) / 1e6

    print(f'\n  Tamanhos:')
    print(f'    Full:    {size_full:.2f} MB')
    print(f'    Float16: {size_f16:.2f} MB  ({size_f16/size_full*100:.0f}% do original)')
    print(f'    INT8:    {size_int8:.2f} MB  ({size_int8/size_full*100:.0f}% do original)')

    del model
    tf.keras.backend.clear_session()
    return path_f16, path_int8, size_full, size_f16, size_int8

# ──────────────────────────────────────────────────────────────────────────────
# [8] AVALIAÇÃO DOS MODELOS TFLITE
# ──────────────────────────────────────────────────────────────────────────────

def make_interpreter(model_path):
    """Cria intérprete TFLite com fallback para diferentes versões do TF."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_path=model_path)
    except ImportError:
        pass
    try:
        return tf.lite.Interpreter(model_path=model_path, num_threads=4)
    except TypeError:
        return tf.lite.Interpreter(model_path=model_path)


def evaluate_tflite(model_path, label):
    """Avalia modelo .tflite no conjunto de teste."""
    print(f'  Avaliando [{label}]...')
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    y_true, y_pred = [], []
    eval_ds = load_dataset('test')
    for images, labels in eval_ds:
        for i in range(len(images)):
            img = np.expand_dims(images[i].numpy(), 0).astype(np.float32)
            interpreter.set_tensor(inp[0]['index'], img)
            interpreter.invoke()
            prob = float(interpreter.get_tensor(out[0]['index'])[0][0])
            y_pred.append(1 if prob > 0.5 else 0)
            y_true.append(int(labels[i].numpy()))

    acc = accuracy_score(y_true, y_pred) * 100
    print(f'  Acurácia [{label}]: {acc:.2f}%')
    return acc

# ──────────────────────────────────────────────────────────────────────────────
# [9] GRÁFICOS COMPARATIVOS
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_comparison(eval_results):
    """Curvas de acurácia + barras de acurácia final."""
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for i, (name, data) in enumerate(eval_results.items()):
        epochs = range(1, len(data['history']['accuracy']) + 1)
        ax1.plot(epochs, data['history']['accuracy'],
                 label=f'{name} (treino)', color=colors[i], linewidth=2)
        ax1.plot(epochs, data['history']['val_accuracy'],
                 label=f'{name} (val)', color=colors[i],
                 linewidth=2, linestyle='--')
    ax1.set_title('Acurácia por Época')
    ax1.set_xlabel('Época'); ax1.set_ylabel('Acurácia')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    accs = [eval_results[n]['accuracy'] * 100 for n in ARCHITECTURES]
    bars = ax2.bar(ARCHITECTURES, accs, color=colors, width=0.5, edgecolor='white')
    ax2.set_title('Acurácia Final — Conjunto de Teste')
    ax2.set_ylabel('Acurácia (%)')
    ax2.set_ylim([max(0, min(accs) - 5), 100])
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    plt.suptitle(
        'Comparativo de Arquiteturas — Classificação de Defeitos em Painéis Fotovoltaicos',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparativo_acuracia.png'), dpi=150)
    plt.show()
    print(f'Gráfico salvo em: {RESULTS_DIR}/comparativo_acuracia.png')


def plot_quantization_comparison(df_quant):
    """Tamanho dos modelos + acurácia por formato de quantização."""
    colors = {'Full': '#1565C0', 'F16': '#2E7D32', 'INT8': '#E65100'}
    x, w   = range(len(ARCHITECTURES)), 0.25
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    b1 = ax1.bar([i-w for i in x], df_quant['tamanho_full_mb'],  w, label='Full',    color=colors['Full'])
    b2 = ax1.bar([i   for i in x], df_quant['tamanho_f16_mb'],   w, label='Float16', color=colors['F16'])
    b3 = ax1.bar([i+w for i in x], df_quant['tamanho_int8_mb'],  w, label='INT8',    color=colors['INT8'])
    for bars in [b1, b2, b3]:
        for bar in bars:
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                     f'{bar.get_height():.1f}MB', ha='center', fontsize=8, fontweight='bold')
    ax1.set_title('Tamanho dos Modelos por Formato')
    ax1.set_ylabel('Tamanho (MB)')
    ax1.set_xticks(list(x)); ax1.set_xticklabels(ARCHITECTURES)
    ax1.legend(); ax1.grid(True, axis='y', alpha=0.3)

    ax2 = axes[1]
    b4 = ax2.bar([i-w for i in x], df_quant['acuracia_full_pct'],  w, label='Full',    color=colors['Full'])
    b5 = ax2.bar([i   for i in x], df_quant['acuracia_f16_pct'],   w, label='Float16', color=colors['F16'])
    b6 = ax2.bar([i+w for i in x], df_quant['acuracia_int8_pct'],  w, label='INT8',    color=colors['INT8'])
    for bars in [b4, b5, b6]:
        for bar in bars:
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                     f'{bar.get_height():.1f}%', ha='center', fontsize=8, fontweight='bold')
    min_acc = df_quant[['acuracia_full_pct','acuracia_f16_pct','acuracia_int8_pct']].min().min()
    ax2.set_title('Acurácia por Formato de Quantização')
    ax2.set_ylabel('Acurácia (%)')
    ax2.set_ylim([max(0, min_acc - 5), 100])
    ax2.set_xticks(list(x)); ax2.set_xticklabels(ARCHITECTURES)
    ax2.legend(); ax2.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Comparativo de Quantização — Full vs Float16 vs INT8',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparativo_quantizacao.png'), dpi=150)
    plt.show()
    print(f'Gráfico salvo em: {RESULTS_DIR}/comparativo_quantizacao.png')

# ──────────────────────────────────────────────────────────────────────────────
# [10] PREDIÇÃO EM IMAGEM INDIVIDUAL
# ──────────────────────────────────────────────────────────────────────────────

def predict_single(model_path, image_path, threshold=0.5, use_tflite=False):
    """
    Predição em imagem individual.
    model_path: pasta SavedModel  ou  arquivo .tflite
    use_tflite: True para usar modelo quantizado
    """
    img_raw   = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img_resz  = cv2.resize(img_raw, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    img_norm  = cv2.normalize(img_resz, None, 0, 255, cv2.NORM_MINMAX)
    img_eq    = cv2.equalizeHist(img_norm)
    img_color = cv2.applyColorMap(img_eq, cv2.COLORMAP_INFERNO)
    img_rgb   = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_batch = np.expand_dims(img_rgb, axis=0)

    if use_tflite:
        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()
        out = interpreter.get_output_details()
        interpreter.set_tensor(inp[0]['index'], img_batch)
        interpreter.invoke()
        prob = float(interpreter.get_tensor(out[0]['index'])[0][0])
    else:
        model = tf.keras.models.load_model(model_path)  # aceita .keras ou SavedModel
        prob  = float(model.predict(img_batch, verbose=0)[0][0])

    label = 'DEFEITO' if prob > threshold else 'NORMAL'
    conf  = prob if prob > threshold else 1 - prob
    color_hex = '#e53935' if label == 'DEFEITO' else '#43a047'

    plt.figure(figsize=(4, 4))
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'{label}  |  {conf:.1%}', fontsize=13,
              fontweight='bold', color=color_hex)
    plt.tight_layout()
    plt.show()

    print(f'Resultado:  {label}')
    print(f'Confiança:  {conf:.1%}')
    print(f'Prob raw:   {prob:.4f}')
    return label, prob

# ──────────────────────────────────────────────────────────────────────────────
# MAIN — executa o pipeline completo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    eval_results  = {}
    quant_results = {}

    # ── ETAPA 1: Treinamento ─────────────────────────────────────
    print('\n' + '='*65)
    print('  ETAPA 1 — TREINAMENTO')
    print('='*65)

    trained_history = {}
    for arch in ARCHITECTURES:
        _, history = train_model(arch)
        trained_history[arch] = history
        tf.keras.backend.clear_session()
        val_ds  = load_dataset('val')
        test_ds = load_dataset('test')

    # ── ETAPA 2: Avaliação no conjunto de teste ───────────────────
    print('\n' + '='*65)
    print('  ETAPA 2 — AVALIAÇÃO NO CONJUNTO DE TESTE')
    print('='*65)

    for arch in ARCHITECTURES:
        val_ds  = load_dataset('val')
        test_ds = load_dataset('test')
        acc, report = evaluate_model(arch)
        eval_results[arch] = {
            'accuracy': acc,
            'history':  trained_history[arch],
            'report':   report
        }

    # CSV resumo de treinamento
    summary = pd.DataFrame([
        {
            'arquitetura':      arch,
            'acuracia_teste':   round(data['accuracy'] * 100, 2),
            'precisao_normal':  round(data['report']['Normal']['precision'] * 100, 2),
            'recall_normal':    round(data['report']['Normal']['recall'] * 100, 2),
            'precisao_defeito': round(data['report']['Defeito']['precision'] * 100, 2),
            'recall_defeito':   round(data['report']['Defeito']['recall'] * 100, 2),
        }
        for arch, data in eval_results.items()
    ])
    summary.to_csv(os.path.join(RESULTS_DIR, 'resumo_comparativo.csv'), index=False)
    print('\nResumo de treinamento:')
    print(summary.to_string(index=False))

    plot_training_comparison(eval_results)

    # ── ETAPA 3: Quantização ─────────────────────────────────────
    print('\n' + '='*65)
    print('  ETAPA 3 — QUANTIZAÇÃO PARA TFLITE')
    print('='*65)

    for arch in ARCHITECTURES:
        pf16, pint8, sf, sf16, si8 = quantize_model(arch)
        quant_results[arch] = {
            'path_f16':   pf16,  'path_int8':  pint8,
            'size_full':  sf,    'size_f16':   sf16,  'size_int8': si8
        }

    # ── ETAPA 4: Avaliação dos modelos quantizados ───────────────
    print('\n' + '='*65)
    print('  ETAPA 4 — AVALIAÇÃO DOS MODELOS QUANTIZADOS')
    print('='*65)

    # Garante que eval_results está populado mesmo se a Etapa 2
    # foi executada em uma sessão anterior (lê do CSV salvo)
    if not eval_results:
        print('  eval_results vazio — carregando do CSV salvo...')
        csv_path = os.path.join(RESULTS_DIR, 'resumo_comparativo.csv')
        if os.path.exists(csv_path):
            df_prev = pd.read_csv(csv_path)
            for _, row in df_prev.iterrows():
                eval_results[row['arquitetura']] = {
                    'accuracy': row['acuracia_teste'] / 100,
                    'history':  {},
                    'report': {
                        'Normal':  {'precision': row['precisao_normal']/100,
                                    'recall':    row['recall_normal']/100},
                        'Defeito': {'precision': row['precisao_defeito']/100,
                                    'recall':    row['recall_defeito']/100},
                    }
                }
            print(f'  Carregado: {list(eval_results.keys())}')
        else:
            print('  ⚠️  CSV não encontrado — execute a Etapa 2 primeiro.')

    # Reconstrói quant_results a partir dos arquivos .tflite no disco
    # (necessário quando a Etapa 3 foi executada em sessão anterior)
    if not quant_results:
        print('  quant_results vazio — reconstruindo a partir dos arquivos no disco...')
        for arch in ARCHITECTURES:
            path_f16  = os.path.join(MODELS_DIR, f'{arch}_f16.tflite')
            path_int8 = os.path.join(MODELS_DIR, f'{arch}_int8.tflite')
            model_path = os.path.join(MODELS_DIR, f'{arch}.keras')
            if not os.path.exists(path_f16) or not os.path.exists(path_int8):
                print(f'  ⚠️  TFLite de {arch} não encontrado — execute a Etapa 3 primeiro.')
                continue
            size_full = os.path.getsize(model_path) / 1e6 if os.path.exists(model_path) else 0
            quant_results[arch] = {
                'path_f16':  path_f16,
                'path_int8': path_int8,
                'size_full': size_full,
                'size_f16':  os.path.getsize(path_f16)  / 1e6,
                'size_int8': os.path.getsize(path_int8) / 1e6,
            }
        print(f'  Reconstruído: {list(quant_results.keys())}')

    quant_rows = []
    for arch in ARCHITECTURES:
        if arch not in quant_results:
            print(f'  ⚠️  {arch} não encontrado em quant_results — pulando.')
            continue
        print(f"\n  {arch.upper()}")
        acc_full = eval_results.get(arch, {}).get('accuracy', 0) * 100
        acc_f16  = evaluate_tflite(quant_results[arch]['path_f16'],  'Float16')
        acc_int8 = evaluate_tflite(quant_results[arch]['path_int8'], 'INT8')
        qr = quant_results[arch]
        quant_rows.append({
            'arquitetura':       arch,
            'tamanho_full_mb':   round(qr['size_full'], 2),
            'tamanho_f16_mb':    round(qr['size_f16'],  2),
            'tamanho_int8_mb':   round(qr['size_int8'], 2),
            'reducao_f16_pct':   round((1 - qr['size_f16']  / qr['size_full']) * 100, 1) if qr['size_full'] > 0 else 0,
            'reducao_int8_pct':  round((1 - qr['size_int8'] / qr['size_full']) * 100, 1) if qr['size_full'] > 0 else 0,
            'acuracia_full_pct': round(acc_full, 2),
            'acuracia_f16_pct':  round(acc_f16,  2),
            'acuracia_int8_pct': round(acc_int8, 2),
            'perda_f16_pct':     round(acc_full - acc_f16,  2),
            'perda_int8_pct':    round(acc_full - acc_int8, 2),
        })

    df_quant = pd.DataFrame(quant_rows)
    df_quant.to_csv(os.path.join(RESULTS_DIR, 'resumo_quantizacao.csv'), index=False)

    print('\n' + '='*65)
    print('  RESUMO FINAL — QUANTIZAÇÃO')
    print('='*65)
    print(df_quant.to_string(index=False))

    plot_quantization_comparison(df_quant)

    # ── RESUMO GERAL ─────────────────────────────────────────────
    print('\n' + '='*65)
    print('  PIPELINE CONCLUÍDO')
    print('='*65)
    print(f'\n  Modelos SavedModel: {MODELS_DIR}/<arch>/')
    print(f'  Modelos TFLite:     {MODELS_DIR}/<arch>_f16.tflite  /  <arch>_int8.tflite')
    print(f'  Resultados:         {RESULTS_DIR}/')
    print(f'\n  Próximo passo: copie os .tflite para a Raspberry Pi')
    print(f'  e rode: python inference_raspberry.py --test_dir data_split/test')
