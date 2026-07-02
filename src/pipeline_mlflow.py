"""
pipeline_mlflow.py — Pipeline com MLflow Tracking integrado
Projeto: Classificação de Defeitos em Módulos Fotovoltaicos por Imagens Termográficas

Este script executa o mesmo pipeline do pipeline_final.py, mas com tracking
completo via MLflow:
    - Parent run por execução completa
    - Nested runs por arquitetura
    - Logging de hiperparâmetros, métricas por época, métricas finais
    - Artifacts: confusion matrix, curvas de treino, modelos .keras e .tflite

Uso:
    python src/pipeline_mlflow.py

Visualização:
    mlflow ui --port 5000
    Acesse: http://localhost:5000

Saída:
    mlruns/               → Dados do MLflow
    models/               → Modelos salvos
    results/              → Gráficos e CSVs
"""

import os
import sys
import cv2
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, precision_score,
                              recall_score)

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
SOURCE_DIR  = 'data'
DATA_DIR    = 'data_split'
MODELS_DIR  = 'models'
RESULTS_DIR = 'results'

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DO MLFLOW
# ──────────────────────────────────────────────────────────────────────────────

MLFLOW_EXPERIMENT_NAME = "pv-fault-detection"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

print(f'TensorFlow: {tf.__version__}')
print(f'MLflow:     {mlflow.__version__}')
print(f'GPU:        {len(tf.config.list_physical_devices("GPU")) > 0}')
print(f'Experiment: {MLFLOW_EXPERIMENT_NAME}\n')

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES INDIVIDUAIS POR ARQUITETURA (versão final — v_final)
# ──────────────────────────────────────────────────────────────────────────────

ARCH_CONFIGS = {
    'mobilenetv2': {
        'epochs_phase1':   10,
        'epochs_phase2':   30,
        'unfreeze_layers': 40,
        'head':            'melhorado',
        'dropout_1':       0.4,
        'dropout_2':       0.3,
        'lr_phase1':       1e-3,
        'lr_phase2':       1e-5,
        'weight_decay':    1e-4,
        'es_patience':     6,
        'augment_level':   'normal',
        'label_smoothing': 0.1,
        'version':         'v_final',
        'hypothesis':      'Head melhorado (Dense 256+64) com mais épocas de fine-tuning e augmentation normal',
    },
    'efficientnetb0': {
        'epochs_phase1':   10,
        'epochs_phase2':   25,
        'unfreeze_layers': 35,
        'head':            'simples',
        'dropout_1':       0.3,
        'dropout_2':       None,
        'lr_phase1':       1e-3,
        'lr_phase2':       1e-5,
        'weight_decay':    1e-4,
        'es_patience':     5,
        'augment_level':   'normal',
        'label_smoothing': 0.1,
        'version':         'v_final',
        'hypothesis':      'Head simples com early stopping conservador — controle de overfitting',
    },
    'shufflenet': {
        'epochs_phase1':   10,
        'epochs_phase2':   20,
        'unfreeze_layers': 20,
        'head':            'simples',
        'dropout_1':       0.3,
        'dropout_2':       None,
        'lr_phase1':       1e-3,
        'lr_phase2':       1e-5,
        'weight_decay':    5e-5,
        'es_patience':     4,
        'augment_level':   'leve',
        'label_smoothing': 0.1,
        'version':         'v_final',
        'hypothesis':      'Augmentation leve + weight decay menor para rede compacta (edge deployment)',
    },
}

ARCHITECTURES = list(ARCH_CONFIGS.keys())

# ──────────────────────────────────────────────────────────────────────────────
# DIVISÃO DO DATASET (70/15/15)
# ──────────────────────────────────────────────────────────────────────────────

def split_dataset(source_dir, output_dir, splits=(0.70, 0.15, 0.15)):
    random.seed(SEED)
    counts = {}
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
        counts[class_name] = {'train': n_train, 'val': n_val, 'test': len(subsets['test'])}
        print(f'  {class_name:8s} → {n_train:5d} treino | {n_val:5d} val | {len(subsets["test"]):5d} teste')
    print('Split concluído!\n')
    return counts


if not os.path.exists(DATA_DIR):
    print('Dividindo dataset...')
    dataset_counts = split_dataset(SOURCE_DIR, DATA_DIR)
else:
    print(f'Pasta "{DATA_DIR}" já existe — split ignorado.\n')
    dataset_counts = None

# ──────────────────────────────────────────────────────────────────────────────
# PRÉ-PROCESSAMENTO E DATASET
# ──────────────────────────────────────────────────────────────────────────────

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
    """Converte imagem térmica (cinza) para RGB com colormap INFERNO + equalização."""
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
        color_mode='grayscale',
        interpolation='bicubic'
    )
    ds = ds.map(preprocess_thermal, num_parallel_calls=AUTOTUNE)
    if augment_level == 'normal':
        ds = ds.map(lambda x, y: (augment_normal(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    elif augment_level == 'leve':
        ds = ds.map(lambda x, y: (augment_leve(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

# ──────────────────────────────────────────────────────────────────────────────
# DEFINIÇÃO DAS ARQUITETURAS
# ──────────────────────────────────────────────────────────────────────────────

def build_model(arch_name, cfg):
    """Constrói modelo com Transfer Learning e head adaptativo."""
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name='input_image')

    if arch_name == 'mobilenetv2':
        x    = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1.0)(inputs)
        base = tf.keras.applications.MobileNetV2(
            input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    elif arch_name == 'efficientnetb0':
        x    = inputs
        base = tf.keras.applications.EfficientNetB0(
            input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    elif arch_name == 'shufflenet':
        x    = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1.0)(inputs)
        base = tf.keras.applications.MobileNetV3Small(
            input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
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
# MLFLOW CALLBACK — Log de métricas por época
# ──────────────────────────────────────────────────────────────────────────────

class MLflowEpochLogger(tf.keras.callbacks.Callback):
    """Callback que loga métricas por época no MLflow."""

    def __init__(self, phase_name="phase1", epoch_offset=0):
        super().__init__()
        self.phase_name = phase_name
        self.epoch_offset = epoch_offset

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        global_epoch = self.epoch_offset + epoch + 1
        metrics = {}
        for key, value in logs.items():
            metrics[f"{self.phase_name}_{key}"] = value
            metrics[f"epoch_{key}"] = value  # métrica unificada
        metrics["global_epoch"] = global_epoch
        mlflow.log_metrics(metrics, step=global_epoch)

# ──────────────────────────────────────────────────────────────────────────────
# TREINAMENTO COM MLFLOW TRACKING
# ──────────────────────────────────────────────────────────────────────────────

def train_model_mlflow(arch_name, parent_run_id):
    """Treina um modelo e registra tudo no MLflow como nested run."""
    cfg = ARCH_CONFIGS[arch_name]

    with mlflow.start_run(
        run_name=f"{arch_name}_{cfg['version']}",
        nested=True,
        tags={
            "architecture": arch_name,
            "version": cfg['version'],
            "hypothesis": cfg['hypothesis'],
            "phase": "training",
            "head_type": cfg['head'],
            "augment_level": cfg['augment_level'],
        }
    ) as run:
        # Log de hiperparâmetros
        mlflow.log_params({
            "architecture": arch_name,
            "version": cfg['version'],
            "img_size": str(IMG_SIZE),
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "epochs_phase1": cfg['epochs_phase1'],
            "epochs_phase2": cfg['epochs_phase2'],
            "total_epochs_max": cfg['epochs_phase1'] + cfg['epochs_phase2'],
            "unfreeze_layers": cfg['unfreeze_layers'],
            "head_type": cfg['head'],
            "dropout_1": cfg['dropout_1'],
            "dropout_2": cfg['dropout_2'] or 0,
            "lr_phase1": cfg['lr_phase1'],
            "lr_phase2": cfg['lr_phase2'],
            "weight_decay": cfg['weight_decay'],
            "early_stopping_patience": cfg['es_patience'],
            "augment_level": cfg['augment_level'],
            "label_smoothing": cfg['label_smoothing'],
            "optimizer_phase1": "Adam",
            "optimizer_phase2": "AdamW",
            "preprocessing": "grayscale → equalize_hist → INFERNO colormap",
            "interpolation": "bicubic",
            "transfer_learning_base": "ImageNet",
        })

        print(f"\n{'='*65}")
        print(f"  Treinando: {arch_name.upper()} [{cfg['version']}]")
        print(f"  Hipótese: {cfg['hypothesis']}")
        print(f"{'='*65}")

        train_ds = load_dataset('train', augment_level=cfg['augment_level'])
        val_ds   = load_dataset('val')
        model, base = build_model(arch_name, cfg)

        save_path = os.path.join(MODELS_DIR, f'{arch_name}.keras')
        best_weights_path = os.path.join(MODELS_DIR, f'{arch_name}_best.weights.h5')

        callbacks_phase1 = [
            tf.keras.callbacks.ModelCheckpoint(
                best_weights_path, monitor='val_accuracy',
                save_best_only=True, save_weights_only=True, verbose=1),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=cfg['es_patience'],
                restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
            MLflowEpochLogger(phase_name="phase1", epoch_offset=0),
        ]

        # ── FASE 1: Head com base congelada
        print(f'\n[FASE 1] Head — base congelada | LR={cfg["lr_phase1"]}')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['lr_phase1']),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=cfg['label_smoothing']),
            metrics=['accuracy']
        )
        h1 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=cfg['epochs_phase1'], callbacks=callbacks_phase1
        )

        actual_epochs_phase1 = len(h1.history['accuracy'])
        mlflow.log_metric("actual_epochs_phase1", actual_epochs_phase1)

        # ── FASE 2: Fine-tuning
        print(f"\n[FASE 2] Fine-tuning — últimas {cfg['unfreeze_layers']} camadas")
        base.trainable = True
        for layer in base.layers[:-cfg['unfreeze_layers']]:
            layer.trainable = False

        n_trainable = sum(tf.size(w).numpy() for w in model.trainable_variables)
        mlflow.log_metric("trainable_params_phase2", n_trainable)
        print(f'  Parâmetros treináveis: {n_trainable:,}')

        callbacks_phase2 = [
            tf.keras.callbacks.ModelCheckpoint(
                best_weights_path, monitor='val_accuracy',
                save_best_only=True, save_weights_only=True, verbose=1),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=cfg['es_patience'],
                restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
            MLflowEpochLogger(phase_name="phase2", epoch_offset=actual_epochs_phase1),
        ]

        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=cfg['lr_phase2'],
                weight_decay=cfg['weight_decay']),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=cfg['label_smoothing']),
            metrics=['accuracy']
        )
        h2 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=cfg['epochs_phase2'], callbacks=callbacks_phase2
        )

        actual_epochs_phase2 = len(h2.history['accuracy'])
        total_epochs = actual_epochs_phase1 + actual_epochs_phase2
        mlflow.log_metric("actual_epochs_phase2", actual_epochs_phase2)
        mlflow.log_metric("total_epochs_trained", total_epochs)

        # Salvar modelo
        model.save(save_path)
        print(f'\nModelo salvo em: {save_path}')

        # Log best val metrics
        best_val_acc = max(h1.history['val_accuracy'] + h2.history['val_accuracy'])
        best_val_loss = min(h1.history['val_loss'] + h2.history['val_loss'])
        mlflow.log_metrics({
            "best_val_accuracy": best_val_acc,
            "best_val_loss": best_val_loss,
        })

        # Salvar histórico completo
        history = {}
        for key in h1.history:
            history[key] = h1.history[key] + h2.history[key]
        history_df = pd.DataFrame(history)
        history_path = os.path.join(RESULTS_DIR, f'history_{arch_name}.csv')
        history_df.to_csv(history_path, index=False)
        mlflow.log_artifact(history_path, artifact_path="training_history")

        # Log modelo como artifact
        mlflow.log_artifact(save_path, artifact_path="models")

        return model, history, run.info.run_id

# ──────────────────────────────────────────────────────────────────────────────
# AVALIAÇÃO COM MLFLOW
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model_mlflow(arch_name):
    """Avalia modelo no conjunto de teste e loga métricas no MLflow."""
    print(f'\n[{arch_name}] Avaliando no conjunto de teste...')

    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, f'{arch_name}.keras'))
    test_ds = load_dataset('test')

    y_true, y_pred_prob = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred_prob.extend(probs.flatten())
        y_true.extend(labels.numpy().flatten())

    y_true      = np.array(y_true, dtype=int)
    y_pred_prob = np.array(y_pred_prob)
    y_pred      = (y_pred_prob > 0.5).astype(int)

    # Métricas
    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall    = recall_score(y_true, y_pred, average='weighted')

    report = classification_report(y_true, y_pred,
                                    target_names=['Normal', 'Defeito'],
                                    output_dict=True)

    # Log no MLflow
    mlflow.log_metrics({
        "test_accuracy": acc,
        "test_f1_weighted": f1,
        "test_precision_weighted": precision,
        "test_recall_weighted": recall,
        "test_precision_normal": report['Normal']['precision'],
        "test_recall_normal": report['Normal']['recall'],
        "test_f1_normal": report['Normal']['f1-score'],
        "test_precision_defeito": report['Defeito']['precision'],
        "test_recall_defeito": report['Defeito']['recall'],
        "test_f1_defeito": report['Defeito']['f1-score'],
    })

    print(f"\n  {arch_name.upper()} — Acurácia: {acc*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Defeito']))

    # Confusion Matrix como artifact
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Defeito'],
                yticklabels=['Normal', 'Defeito'])
    plt.title(f'Matriz de Confusão — {arch_name}')
    plt.ylabel('Real'); plt.xlabel('Predito')
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{arch_name}.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="evaluation")

    del model
    tf.keras.backend.clear_session()
    return acc, report

# ──────────────────────────────────────────────────────────────────────────────
# QUANTIZAÇÃO COM MLFLOW
# ──────────────────────────────────────────────────────────────────────────────

def get_representative_dataset():
    """Dataset representativo para calibração INT8."""
    def generator():
        count = 0
        cal_ds = load_dataset('val')
        for images, _ in cal_ds:
            if count >= 100:
                break
            yield [tf.cast(images, tf.float32)]
            count += 1
    return generator


def quantize_model_mlflow(arch_name):
    """Quantiza modelo e loga métricas de tamanho no MLflow."""
    model_path = os.path.join(MODELS_DIR, f'{arch_name}.keras')
    print(f"\n  Quantizando: {arch_name.upper()}")

    model = tf.keras.models.load_model(model_path)
    size_full = os.path.getsize(model_path) / 1e6

    # Float16
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tflite_f16 = conv.convert()
    path_f16 = os.path.join(MODELS_DIR, f'{arch_name}_f16.tflite')
    open(path_f16, 'wb').write(tflite_f16)

    # INT8
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = get_representative_dataset()
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    conv.inference_input_type  = tf.float32
    conv.inference_output_type = tf.float32
    tflite_int8 = conv.convert()
    path_int8 = os.path.join(MODELS_DIR, f'{arch_name}_int8.tflite')
    open(path_int8, 'wb').write(tflite_int8)

    size_f16  = len(tflite_f16)  / 1e6
    size_int8 = len(tflite_int8) / 1e6

    # Log métricas de quantização
    mlflow.log_metrics({
        "model_size_full_mb": size_full,
        "model_size_f16_mb": size_f16,
        "model_size_int8_mb": size_int8,
        "compression_ratio_f16": round(size_f16 / size_full, 3),
        "compression_ratio_int8": round(size_int8 / size_full, 3),
    })

    # Log artifacts TFLite
    mlflow.log_artifact(path_f16, artifact_path="tflite_models")
    mlflow.log_artifact(path_int8, artifact_path="tflite_models")

    print(f'    Full:    {size_full:.2f} MB')
    print(f'    Float16: {size_f16:.2f} MB ({size_f16/size_full*100:.0f}%)')
    print(f'    INT8:    {size_int8:.2f} MB ({size_int8/size_full*100:.0f}%)')

    del model
    tf.keras.backend.clear_session()
    return path_f16, path_int8, size_full, size_f16, size_int8

# ──────────────────────────────────────────────────────────────────────────────
# AVALIAÇÃO TFLITE COM MLFLOW
# ──────────────────────────────────────────────────────────────────────────────

def make_interpreter(model_path):
    """Cria intérprete TFLite com fallback."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_path=model_path)
    except ImportError:
        pass
    try:
        return tf.lite.Interpreter(model_path=model_path, num_threads=4)
    except TypeError:
        return tf.lite.Interpreter(model_path=model_path)


def evaluate_tflite_mlflow(model_path, label, arch_name):
    """Avalia modelo .tflite e loga acurácia no MLflow."""
    print(f'  Avaliando [{label}]...')
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    test_ds = load_dataset('test')
    y_true, y_pred = [], []
    for images, labels_batch in test_ds:
        for i in range(len(images)):
            img = np.expand_dims(images[i].numpy(), 0).astype(np.float32)
            interpreter.set_tensor(inp[0]['index'], img)
            interpreter.invoke()
            prob = float(interpreter.get_tensor(out[0]['index'])[0][0])
            y_pred.append(1 if prob > 0.5 else 0)
            y_true.append(int(labels_batch[i].numpy()))

    acc = accuracy_score(y_true, y_pred) * 100
    metric_name = f"tflite_accuracy_{label.lower()}"
    mlflow.log_metric(metric_name, acc)
    print(f'  Acurácia [{label}]: {acc:.2f}%')
    return acc

# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICOS COMPARATIVOS
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_comparison(eval_results):
    """Curvas de acurácia + barras comparativas."""
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for i, (name, data) in enumerate(eval_results.items()):
        if 'history' in data and data['history']:
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
    bars = ax2.bar(ARCHITECTURES, accs, color=colors, width=0.5)
    ax2.set_title('Acurácia Final — Conjunto de Teste')
    ax2.set_ylabel('Acurácia (%)')
    ax2.set_ylim([max(0, min(accs) - 5), 100])
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Comparativo de Arquiteturas — Classificação de Defeitos PV',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'comparativo_acuracia.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# ──────────────────────────────────────────────────────────────────────────────
# MAIN — Pipeline completo com MLflow
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print('\n' + '='*65)
    print('  PIPELINE COM MLFLOW TRACKING')
    print('  Projeto: Classificação de Defeitos em Módulos Fotovoltaicos')
    print('='*65)

    # Parent run — engloba toda a execução
    with mlflow.start_run(
        run_name=f"pipeline_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags={
            "project": "pv-fault-detection",
            "pipeline_type": "full_training",
            "dataset": "InfraredSolarModules",
            "task": "binary_classification",
            "target": "edge_deployment_raspberry_pi",
        }
    ) as parent_run:

        # Log de parâmetros globais do pipeline
        mlflow.log_params({
            "n_architectures": len(ARCHITECTURES),
            "architectures": str(ARCHITECTURES),
            "img_size": str(IMG_SIZE),
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "split_ratio": "70/15/15",
            "preprocessing_pipeline": "grayscale→equalize_hist→INFERNO→RGB",
            "quantization_formats": "Float16, INT8",
            "training_strategy": "2-phase (frozen_head + fine_tuning)",
        })

        eval_results  = {}
        quant_results = {}

        # ── ETAPA 1: Treinamento ─────────────────────────────────
        print('\n' + '='*65)
        print('  ETAPA 1 — TREINAMENTO')
        print('='*65)

        trained_models = {}
        for arch in ARCHITECTURES:
            model, history, run_id = train_model_mlflow(arch, parent_run.info.run_id)
            trained_models[arch] = {'model': model, 'history': history, 'run_id': run_id}
            del model
            tf.keras.backend.clear_session()

        # ── ETAPA 2: Avaliação ───────────────────────────────────
        print('\n' + '='*65)
        print('  ETAPA 2 — AVALIAÇÃO NO CONJUNTO DE TESTE')
        print('='*65)

        for arch in ARCHITECTURES:
            # Retoma o nested run da arquitetura para logar avaliação
            with mlflow.start_run(
                run_id=trained_models[arch]['run_id']
            ):
                acc, report = evaluate_model_mlflow(arch)
                eval_results[arch] = {
                    'accuracy': acc,
                    'history': trained_models[arch]['history'],
                    'report': report
                }

        # Gráfico comparativo como artifact do parent
        comp_path = plot_training_comparison(eval_results)
        mlflow.log_artifact(comp_path, artifact_path="comparisons")

        # Log da melhor arquitetura no parent run
        best_arch = max(eval_results, key=lambda x: eval_results[x]['accuracy'])
        mlflow.log_metrics({
            "best_test_accuracy": eval_results[best_arch]['accuracy'],
        })
        mlflow.set_tag("best_architecture", best_arch)

        # ── ETAPA 3: Quantização ─────────────────────────────────
        print('\n' + '='*65)
        print('  ETAPA 3 — QUANTIZAÇÃO PARA TFLITE')
        print('='*65)

        for arch in ARCHITECTURES:
            with mlflow.start_run(run_id=trained_models[arch]['run_id']):
                pf16, pint8, sf, sf16, si8 = quantize_model_mlflow(arch)
                quant_results[arch] = {
                    'path_f16': pf16, 'path_int8': pint8,
                    'size_full': sf, 'size_f16': sf16, 'size_int8': si8
                }

        # ── ETAPA 4: Avaliação dos modelos quantizados ───────────
        print('\n' + '='*65)
        print('  ETAPA 4 — AVALIAÇÃO DOS MODELOS QUANTIZADOS')
        print('='*65)

        quant_rows = []
        for arch in ARCHITECTURES:
            with mlflow.start_run(run_id=trained_models[arch]['run_id']):
                print(f"\n  {arch.upper()}")
                acc_full = eval_results[arch]['accuracy'] * 100
                acc_f16  = evaluate_tflite_mlflow(
                    quant_results[arch]['path_f16'], 'F16', arch)
                acc_int8 = evaluate_tflite_mlflow(
                    quant_results[arch]['path_int8'], 'INT8', arch)

                # Log de perda por quantização
                mlflow.log_metrics({
                    "accuracy_loss_f16": round(acc_full - acc_f16, 2),
                    "accuracy_loss_int8": round(acc_full - acc_int8, 2),
                })

                qr = quant_results[arch]
                quant_rows.append({
                    'arquitetura': arch,
                    'tamanho_full_mb': round(qr['size_full'], 2),
                    'tamanho_f16_mb': round(qr['size_f16'], 2),
                    'tamanho_int8_mb': round(qr['size_int8'], 2),
                    'acuracia_full_pct': round(acc_full, 2),
                    'acuracia_f16_pct': round(acc_f16, 2),
                    'acuracia_int8_pct': round(acc_int8, 2),
                    'perda_f16_pct': round(acc_full - acc_f16, 2),
                    'perda_int8_pct': round(acc_full - acc_int8, 2),
                })

        # Salvar resumo quantização
        df_quant = pd.DataFrame(quant_rows)
        quant_csv = os.path.join(RESULTS_DIR, 'resumo_quantizacao.csv')
        df_quant.to_csv(quant_csv, index=False)
        mlflow.log_artifact(quant_csv, artifact_path="comparisons")

        # Resumo final
        summary = pd.DataFrame([{
            'arquitetura': arch,
            'acuracia_teste': round(eval_results[arch]['accuracy'] * 100, 2),
            'precisao_defeito': round(eval_results[arch]['report']['Defeito']['precision'] * 100, 2),
            'recall_defeito': round(eval_results[arch]['report']['Defeito']['recall'] * 100, 2),
        } for arch in ARCHITECTURES])
        summary_csv = os.path.join(RESULTS_DIR, 'resumo_comparativo.csv')
        summary.to_csv(summary_csv, index=False)
        mlflow.log_artifact(summary_csv, artifact_path="comparisons")

        print('\n' + '='*65)
        print('  PIPELINE CONCLUÍDO COM MLFLOW TRACKING')
        print('='*65)
        print(f'\n  MLflow UI: mlflow ui --port 5000')
        print(f'  Acesse: http://localhost:5000')
        print(f'  Experiment: {MLFLOW_EXPERIMENT_NAME}')
        print(f'  Parent Run: {parent_run.info.run_id}')
        print(f'\n  Melhor arquitetura: {best_arch} ({eval_results[best_arch]["accuracy"]*100:.2f}%)')
