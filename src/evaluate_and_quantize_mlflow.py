"""
evaluate_and_quantize_mlflow.py — Avaliação e quantização com MLflow (sem retreinar)

Usa os modelos .keras já salvos em models/ para:
    - Avaliar no conjunto de teste
    - Quantizar para Float16 e INT8
    - Avaliar modelos quantizados
    - Logar tudo no MLflow

Uso:
    python src/evaluate_and_quantize_mlflow.py
"""

import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import mlflow
from datetime import datetime
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, precision_score,
                              recall_score)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
AUTOTUNE    = tf.data.AUTOTUNE
DATA_DIR    = 'data_split'
MODELS_DIR  = 'models'
RESULTS_DIR = 'results'

os.makedirs(RESULTS_DIR, exist_ok=True)

# MLflow
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "pv-fault-detection"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Arquiteturas com modelos já treinados
ARCH_CONFIGS = {
    'mobilenetv2': {
        'epochs_phase1': 10, 'epochs_phase2': 30, 'unfreeze_layers': 40,
        'head': 'melhorado', 'dropout_1': 0.4, 'dropout_2': 0.3,
        'lr_phase1': 1e-3, 'lr_phase2': 1e-5, 'weight_decay': 1e-4,
        'es_patience': 6, 'augment_level': 'normal', 'label_smoothing': 0.1,
        'version': 'v_final',
        'hypothesis': 'Head melhorado (Dense 256+64) com mais épocas de fine-tuning',
    },
    'efficientnetb0': {
        'epochs_phase1': 10, 'epochs_phase2': 25, 'unfreeze_layers': 35,
        'head': 'simples', 'dropout_1': 0.3, 'dropout_2': None,
        'lr_phase1': 1e-3, 'lr_phase2': 1e-5, 'weight_decay': 1e-4,
        'es_patience': 5, 'augment_level': 'normal', 'label_smoothing': 0.1,
        'version': 'v_final',
        'hypothesis': 'Head simples com early stopping conservador',
    },
    'mobilenetv3small': {
        'epochs_phase1': 10, 'epochs_phase2': 20, 'unfreeze_layers': 20,
        'head': 'simples', 'dropout_1': 0.3, 'dropout_2': None,
        'lr_phase1': 1e-3, 'lr_phase2': 1e-5, 'weight_decay': 5e-5,
        'es_patience': 4, 'augment_level': 'leve', 'label_smoothing': 0.1,
        'version': 'v_final',
        'hypothesis': 'Augmentation leve + weight decay menor para rede compacta',
    },
}

ARCHITECTURES = list(ARCH_CONFIGS.keys())

print(f'TensorFlow: {tf.__version__}')
print(f'MLflow:     {mlflow.__version__}')
print(f'GPU:        {len(tf.config.list_physical_devices("GPU")) > 0}\n')

# ──────────────────────────────────────────────────────────────────────────────
# PRÉ-PROCESSAMENTO (mesmo do pipeline de treino)
# ──────────────────────────────────────────────────────────────────────────────

def apply_colormap(img_np):
    img_gray  = img_np[:, :, 0].astype(np.uint8)
    img_norm  = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
    img_eq    = cv2.equalizeHist(img_norm)
    img_color = cv2.applyColorMap(img_eq, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB).astype(np.float32)


def preprocess_thermal(image, label):
    def process_single(img):
        img_rgb = tf.numpy_function(apply_colormap, [img], tf.float32)
        img_rgb.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
        return img_rgb
    image = tf.map_fn(process_single, image, fn_output_signature=tf.float32)
    return image, label


def load_dataset(subset):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, subset),
        labels='inferred',
        label_mode='binary',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        color_mode='grayscale',
        interpolation='bicubic'
    )
    ds = ds.map(preprocess_thermal, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

# ──────────────────────────────────────────────────────────────────────────────
# AVALIAÇÃO
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(arch_name, test_ds):
    """Avalia modelo .keras no conjunto de teste."""
    model_path = os.path.join(MODELS_DIR, f'{arch_name}.keras')
    if not os.path.exists(model_path):
        print(f'  ⚠️  {model_path} não encontrado — pulando.')
        return None, None

    print(f'\n  [{arch_name}] Avaliando no conjunto de teste...')
    model = tf.keras.models.load_model(model_path)

    y_true, y_pred_prob = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred_prob.extend(probs.flatten())
        y_true.extend(labels.numpy().flatten())

    y_true      = np.array(y_true, dtype=int)
    y_pred_prob = np.array(y_pred_prob)
    y_pred      = (y_pred_prob > 0.5).astype(int)

    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall    = recall_score(y_true, y_pred, average='weighted')

    report = classification_report(y_true, y_pred,
                                    target_names=['Normal', 'Defeito'],
                                    output_dict=True)

    # Log MLflow
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

    print(f"  Acurácia: {acc*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Defeito']))

    # Confusion matrix
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
# QUANTIZAÇÃO
# ──────────────────────────────────────────────────────────────────────────────

def get_representative_dataset():
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
    """Quantiza modelo para Float16 e INT8."""
    model_path = os.path.join(MODELS_DIR, f'{arch_name}.keras')
    if not os.path.exists(model_path):
        print(f'  ⚠️  {model_path} não encontrado — pulando.')
        return None, None, 0, 0, 0

    print(f'  Quantizando: {arch_name}...')
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

    mlflow.log_metrics({
        "model_size_full_mb": size_full,
        "model_size_f16_mb": size_f16,
        "model_size_int8_mb": size_int8,
        "compression_ratio_f16": round(size_f16 / size_full, 3),
        "compression_ratio_int8": round(size_int8 / size_full, 3),
    })
    mlflow.log_artifact(path_f16, artifact_path="tflite_models")
    mlflow.log_artifact(path_int8, artifact_path="tflite_models")

    print(f'    Full: {size_full:.2f} MB | F16: {size_f16:.2f} MB | INT8: {size_int8:.2f} MB')

    del model
    tf.keras.backend.clear_session()
    return path_f16, path_int8, size_full, size_f16, size_int8

# ──────────────────────────────────────────────────────────────────────────────
# AVALIAÇÃO TFLITE
# ──────────────────────────────────────────────────────────────────────────────

def make_interpreter(model_path):
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
    print(f'    Avaliando [{label}]...')
    try:
        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
    except RuntimeError as e:
        print(f'    ⚠️  Erro ao carregar modelo [{label}]: {e}')
        print(f'    Pulando avaliação TFLite [{label}].')
        mlflow.log_metric(f"tflite_accuracy_{label.lower()}", 0)
        mlflow.set_tag(f"tflite_{label.lower()}_error", str(e)[:200])
        return None

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
            y_true.append(int(labels_batch[i].numpy().item()))

    acc = accuracy_score(y_true, y_pred) * 100
    mlflow.log_metric(f"tflite_accuracy_{label.lower()}", acc)
    print(f'    Acurácia [{label}]: {acc:.2f}%')
    return acc

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print('='*65)
    print('  AVALIAÇÃO + QUANTIZAÇÃO COM MLFLOW (sem retreinar)')
    print('='*65)

    # Verifica quais modelos existem
    available = []
    for arch in ARCHITECTURES:
        path = os.path.join(MODELS_DIR, f'{arch}.keras')
        if os.path.exists(path):
            available.append(arch)
            print(f'  ✅ {arch}: {path} ({os.path.getsize(path)/1e6:.1f} MB)')
        else:
            print(f'  ❌ {arch}: não encontrado')

    if not available:
        print('\n  Nenhum modelo encontrado. Rode pipeline_mlflow.py primeiro.')
        exit(1)

    print(f'\n  Modelos disponíveis: {available}')
    test_ds = load_dataset('test')

    with mlflow.start_run(
        run_name=f"avaliacao_quantizacao_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags={
            "project": "pv-fault-detection",
            "pipeline_type": "evaluation_and_quantization",
            "description": "Avaliação e quantização de modelos já treinados",
        }
    ) as parent_run:

        eval_results  = {}
        quant_results = {}

        # ── Avaliação no conjunto de teste ────────────────────────
        print('\n' + '='*65)
        print('  ETAPA 1 — AVALIAÇÃO NO CONJUNTO DE TESTE')
        print('='*65)

        for arch in available:
            cfg = ARCH_CONFIGS[arch]
            with mlflow.start_run(
                run_name=f"{arch}_{cfg['version']}_eval",
                nested=True,
                tags={
                    "architecture": arch,
                    "version": cfg['version'],
                    "hypothesis": cfg['hypothesis'],
                    "phase": "evaluation_and_quantization",
                }
            ) as child_run:
                # Log hiperparâmetros
                mlflow.log_params({
                    "architecture": arch,
                    "version": cfg['version'],
                    "head_type": cfg['head'],
                    "dropout_1": cfg['dropout_1'],
                    "dropout_2": cfg['dropout_2'] or 0,
                    "lr_phase2": cfg['lr_phase2'],
                    "weight_decay": cfg['weight_decay'],
                    "augment_level": cfg['augment_level'],
                    "unfreeze_layers": cfg['unfreeze_layers'],
                })

                acc, report = evaluate_model(arch, test_ds)
                if acc is not None:
                    eval_results[arch] = {
                        'accuracy': acc,
                        'report': report,
                        'run_id': child_run.info.run_id,
                    }

        # ── Quantização ───────────────────────────────────────────
        print('\n' + '='*65)
        print('  ETAPA 2 — QUANTIZAÇÃO PARA TFLITE')
        print('='*65)

        for arch in available:
            if arch not in eval_results:
                continue
            with mlflow.start_run(
                run_id=eval_results[arch]['run_id'], nested=True
            ):
                pf16, pint8, sf, sf16, si8 = quantize_model(arch)
                if pf16:
                    quant_results[arch] = {
                        'path_f16': pf16, 'path_int8': pint8,
                        'size_full': sf, 'size_f16': sf16, 'size_int8': si8
                    }

        # ── Avaliação dos modelos quantizados ─────────────────────
        print('\n' + '='*65)
        print('  ETAPA 3 — AVALIAÇÃO DOS MODELOS QUANTIZADOS')
        print('='*65)

        quant_rows = []
        for arch in available:
            if arch not in quant_results:
                continue
            with mlflow.start_run(
                run_id=eval_results[arch]['run_id'], nested=True
            ):
                print(f'\n  {arch.upper()}')
                acc_full = eval_results[arch]['accuracy'] * 100
                acc_f16  = evaluate_tflite(quant_results[arch]['path_f16'], 'F16')
                acc_int8 = evaluate_tflite(quant_results[arch]['path_int8'], 'INT8')

                if acc_f16 is not None and acc_int8 is not None:
                    mlflow.log_metrics({
                        "accuracy_loss_f16": round(acc_full - acc_f16, 2),
                        "accuracy_loss_int8": round(acc_full - acc_int8, 2),
                    })
                elif acc_f16 is not None:
                    mlflow.log_metric("accuracy_loss_f16", round(acc_full - acc_f16, 2))

                qr = quant_results[arch]
                quant_rows.append({
                    'arquitetura': arch,
                    'tamanho_full_mb': round(qr['size_full'], 2),
                    'tamanho_f16_mb': round(qr['size_f16'], 2),
                    'tamanho_int8_mb': round(qr['size_int8'], 2),
                    'acuracia_full_pct': round(acc_full, 2),
                    'acuracia_f16_pct': round(acc_f16, 2) if acc_f16 else None,
                    'acuracia_int8_pct': round(acc_int8, 2) if acc_int8 else None,
                    'perda_f16_pct': round(acc_full - acc_f16, 2) if acc_f16 else None,
                    'perda_int8_pct': round(acc_full - acc_int8, 2) if acc_int8 else None,
                })

        # ── Resumo final ──────────────────────────────────────────
        if quant_rows:
            df_quant = pd.DataFrame(quant_rows)
            quant_csv = os.path.join(RESULTS_DIR, 'resumo_quantizacao.csv')
            df_quant.to_csv(quant_csv, index=False)
            mlflow.log_artifact(quant_csv, artifact_path="comparisons")

            print('\n' + '='*65)
            print('  RESUMO FINAL')
            print('='*65)
            print(df_quant.to_string(index=False))

        # Log melhor arquitetura no parent
        if eval_results:
            best_arch = max(eval_results, key=lambda x: eval_results[x]['accuracy'])
            mlflow.log_metric("best_test_accuracy", eval_results[best_arch]['accuracy'])
            mlflow.set_tag("best_architecture", best_arch)

        print(f'\n  ✅ Concluído!')
        print(f'  MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000')
        print(f'  Parent Run: {parent_run.info.run_id}')
