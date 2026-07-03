"""
predict_individual.py — Predição individual com comparação visual
Projeto: Classificação de Defeitos em Módulos Fotovoltaicos

Uso:
    # Testa uma imagem específica
    python predict_individual.py --image caminho/para/imagem.jpg --model models/mobilenetv2.keras

    # Testa múltiplas imagens e gera painel comparativo
    python predict_individual.py --batch --model models/mobilenetv2.keras

    # Usa modelo TFLite (quantizado)
    python predict_individual.py --image imagem.jpg --model models/mobilenetv2_f16.tflite --tflite
"""

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────────────────────────────────────

IMG_SIZE    = (224, 224)
DATA_DIR    = 'data_split/test'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# PRÉ-PROCESSAMENTO (mesmo pipeline do treino)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(image_path):
    """
    Lê e pré-processa uma imagem térmica para inferência.
    Retorna o batch pronto para o modelo e a imagem colorida para visualização.
    """
    img_raw   = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    img_resz  = cv2.resize(img_raw, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    img_norm  = cv2.normalize(img_resz, None, 0, 255, cv2.NORM_MINMAX)
    img_eq    = cv2.equalizeHist(img_norm)
    img_color = cv2.applyColorMap(img_eq, cv2.COLORMAP_INFERNO)
    img_rgb   = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_batch = np.expand_dims(img_rgb, axis=0)

    return img_batch, img_rgb.astype(np.uint8)

# ──────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DO MODELO
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_path, use_tflite=False):
    if use_tflite:
        try:
            from ai_edge_litert.interpreter import Interpreter
            interp = Interpreter(model_path=model_path)
        except ImportError:
            interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        return interp, True
    else:
        model = tf.keras.models.load_model(model_path)
        return model, False

# ──────────────────────────────────────────────────────────────────────────────
# INFERÊNCIA
# ──────────────────────────────────────────────────────────────────────────────

def predict(model, img_batch, is_tflite=False):
    """Retorna a probabilidade de ser defeito (0.0 a 1.0)."""
    if is_tflite:
        inp = model.get_input_details()
        out = model.get_output_details()
        model.set_tensor(inp[0]['index'], img_batch)
        model.invoke()
        prob = float(model.get_tensor(out[0]['index'])[0][0])
    else:
        prob = float(model.predict(img_batch, verbose=0)[0][0])
    return prob

# ──────────────────────────────────────────────────────────────────────────────
# PREDIÇÃO EM IMAGEM INDIVIDUAL — com visualização
# ──────────────────────────────────────────────────────────────────────────────

def predict_single(model_path, image_path, real_label=None,
                   threshold=0.5, use_tflite=False, save=True):
    """
    Faz predição em uma imagem e exibe:
        - Imagem com colormap
        - Rótulo predito vs rótulo real
        - Probabilidade e confiança
        - Indicação visual de acerto/erro
    """
    model, is_tflite = load_model(model_path, use_tflite)
    img_batch, img_vis = preprocess(image_path)

    prob  = predict(model, img_batch, is_tflite)
    pred  = 'DEFEITO' if prob > threshold else 'NORMAL'
    conf  = prob if prob > threshold else 1 - prob

    # Cores
    if real_label is not None:
        acerto = (pred.lower() == real_label.lower())
        cor_borda = '#2ECC71' if acerto else '#E74C3C'   # verde/vermelho
        status    = '✓ CORRETO' if acerto else '✗ ERRADO'
    else:
        cor_borda = '#2196F3'
        status    = ''

    cor_pred = '#E74C3C' if pred == 'DEFEITO' else '#27AE60'

    fig, ax = plt.subplots(figsize=(5, 5.5))

    # Imagem com borda colorida indicando acerto/erro
    ax.imshow(img_vis)
    for spine in ax.spines.values():
        spine.set_edgecolor(cor_borda)
        spine.set_linewidth(6)
    ax.set_xticks([]); ax.set_yticks([])

    # Título com predição
    ax.set_title(
        f'Predito: {pred}  ({conf:.1%})',
        fontsize=13, fontweight='bold', color=cor_pred, pad=10
    )

    # Rodapé com real e status
    if real_label is not None:
        cor_real = '#E74C3C' if real_label.upper() == 'DEFEITO' else '#27AE60'
        fig.text(0.5, 0.04,
                 f'Real: {real_label.upper()}   {status}',
                 ha='center', fontsize=11, fontweight='bold',
                 color=cor_real)
    else:
        fig.text(0.5, 0.04,
                 f'Prob. defeito: {prob:.4f}',
                 ha='center', fontsize=10, color='#555555')

    plt.tight_layout(rect=[0, 0.07, 1, 1])

    if save:
        fname = Path(image_path).stem
        out   = os.path.join(RESULTS_DIR, f'pred_{fname}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Salvo: {out}')

    plt.show()
    print(f'Predição:  {pred}  ({conf:.1%} de confiança)')
    print(f'Prob raw:  {prob:.4f}')
    if real_label:
        print(f'Real:      {real_label.upper()}')
        print(f'Resultado: {status}')

    return pred, prob

# ──────────────────────────────────────────────────────────────────────────────
# PAINEL COMPARATIVO — múltiplas imagens lado a lado
# ──────────────────────────────────────────────────────────────────────────────

def predict_batch_comparison(model_path, images_com_labels,
                              threshold=0.5, use_tflite=False,
                              titulo="Comparativo de Predições",
                              output_name="comparativo_predicoes.png"):
    """
    Gera um painel com N imagens, mostrando para cada uma:
        - Imagem pré-processada
        - Rótulo real vs predito
        - Borda verde (acerto) ou vermelha (erro)

    images_com_labels: lista de tuplas (caminho_imagem, label_real)
    Exemplo:
        [
            ('data_split/test/defect/img1.jpg', 'defeito'),
            ('data_split/test/normal/img2.jpg', 'normal'),
        ]
    """
    model, is_tflite = load_model(model_path, use_tflite)

    n    = len(images_com_labels)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 4.2))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    acertos = 0

    for idx, (img_path, real_label) in enumerate(images_com_labels):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        try:
            img_batch, img_vis = preprocess(img_path)
            prob = predict(model, img_batch, is_tflite)
            pred = 'DEFEITO' if prob > threshold else 'NORMAL'
            conf = prob if prob > threshold else 1 - prob

            acerto    = (pred.lower() == real_label.lower())
            cor_borda = '#2ECC71' if acerto else '#E74C3C'
            status    = '✓' if acerto else '✗'
            cor_pred  = '#E74C3C' if pred == 'DEFEITO' else '#27AE60'
            cor_real  = '#E74C3C' if real_label.upper() == 'DEFEITO' else '#27AE60'

            if acerto:
                acertos += 1

            ax.imshow(img_vis)
            for spine in ax.spines.values():
                spine.set_edgecolor(cor_borda)
                spine.set_linewidth(5)

            ax.set_title(
                f'{status} Pred: {pred}\n({conf:.0%})',
                fontsize=9, fontweight='bold', color=cor_pred
            )
            ax.set_xlabel(
                f'Real: {real_label.upper()}',
                fontsize=9, color=cor_real, fontweight='bold'
            )
            ax.set_xticks([]); ax.set_yticks([])

        except Exception as e:
            ax.text(0.5, 0.5, f'Erro:\n{e}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='red')
            ax.set_xticks([]); ax.set_yticks([])

    # Oculta eixos extras
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    acc = acertos / n * 100
    model_name = Path(model_path).stem
    fig.suptitle(
        f'{titulo}\nModelo: {model_name}  |  Acurácia nesta amostra: {acertos}/{n} ({acc:.0f}%)',
        fontsize=12, fontweight='bold', y=1.01
    )

    # Legenda
    patch_ok  = mpatches.Patch(color='#2ECC71', label='Acerto')
    patch_err = mpatches.Patch(color='#E74C3C', label='Erro')
    fig.legend(handles=[patch_ok, patch_err], loc='lower center',
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, output_name)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'\nPainel salvo em: {out}')
    print(f'Acurácia na amostra: {acertos}/{n} ({acc:.0f}%)')
    return acertos, n

# ──────────────────────────────────────────────────────────────────────────────
# SELEÇÃO AUTOMÁTICA DE IMAGENS PARA DEMONSTRAÇÃO
# ──────────────────────────────────────────────────────────────────────────────

def select_demo_images(n_per_class=4):
    """
    Seleciona automaticamente N imagens de cada classe do conjunto de teste
    para gerar um painel de demonstração balanceado.
    """
    import random
    random.seed(42)

    images = []
    for label in ['normal', 'defect']:
        folder = Path(DATA_DIR) / label
        if not folder.exists():
            print(f'⚠️  Pasta não encontrada: {folder}')
            continue
        files = list(folder.glob('*.*'))
        sample = random.sample(files, min(n_per_class, len(files)))
        display_label = 'defeito' if label == 'defect' else 'normal'
        images.extend([(str(f), display_label) for f in sample])

    # Embaralha para não mostrar tudo agrupado
    random.shuffle(images)
    return images

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predição individual com comparação visual')
    parser.add_argument('--model',  default='models/mobilenetv2.keras', help='Caminho do modelo (.keras ou .tflite)')
    parser.add_argument('--image',  default=None, help='Caminho de uma imagem OU pasta com imagens')
    parser.add_argument('--real',   default=None, help='Rótulo real: normal ou defeito')
    parser.add_argument('--batch',  action='store_true', help='Gera painel automático com imagens do conjunto de teste')
    parser.add_argument('--n',      type=int, default=4, help='Imagens por classe no painel automático (padrão: 4)')
    parser.add_argument('--tflite', action='store_true', help='Usa modelo .tflite quantizado')
    args = parser.parse_args()

    # ── Modo 1: pasta inteira ─────────────────────────────────────
    # Detecta automaticamente se --image aponta para uma pasta
    if args.image and Path(args.image).is_dir():
        folder      = Path(args.image)
        folder_name = folder.name                          # ex: no_anomaly ou anomaly
        model_name  = Path(args.model).stem               # ex: mobilenetv2
        output_name = f'comparativo_{folder_name}_modelo_{model_name}.png'

        # Inferência do rótulo real a partir do nome da pasta
        if args.real:
            real_label = args.real
        elif 'no_anomaly' in folder_name.lower() or 'normal' in folder_name.lower():
            real_label = 'normal'
        elif 'anomaly' in folder_name.lower() or 'defect' in folder_name.lower():
            real_label = 'defeito'
        else:
            real_label = None
            print(f'⚠️  Não foi possível inferir o rótulo da pasta "{folder_name}".')
            print('    Use --real normal  ou  --real defeito para definir manualmente.')

        # Coleta todas as imagens da pasta
        extensoes = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        arquivos  = []
        for ext in extensoes:
            arquivos.extend(folder.glob(ext))
        arquivos = sorted(arquivos)

        if not arquivos:
            print(f'Nenhuma imagem encontrada em: {folder}')
        else:
            print(f'\nPasta:   {folder}')
            print(f'Modelo:  {args.model}')
            print(f'Rótulo:  {real_label}')
            print(f'Imagens: {len(arquivos)}')
            print(f'Saída:   results/{output_name}\n')

            images_com_labels = [(str(f), real_label) for f in arquivos]

            predict_batch_comparison(
                model_path        = args.model,
                images_com_labels = images_com_labels,
                use_tflite        = args.tflite,
                titulo            = f'Comparativo — {folder_name}',
                output_name       = output_name
            )

    # ── Modo 2: imagem única ──────────────────────────────────────
    elif args.image:
        print(f'\nModelo:  {args.model}')
        print(f'Imagem:  {args.image}')
        if args.real:
            print(f'Real:    {args.real}')
        print()
        predict_single(
            model_path=args.model,
            image_path=args.image,
            real_label=args.real,
            use_tflite=args.tflite
        )

    # ── Modo 3: painel automático ─────────────────────────────────
    elif args.batch:
        print(f'\nGerando painel comparativo automático...')
        print(f'Modelo: {args.model}')
        print(f'Imagens por classe: {args.n}\n')

        images = select_demo_images(n_per_class=args.n)

        if not images:
            print('Nenhuma imagem encontrada. Verifique o DATA_DIR.')
        else:
            model_name = Path(args.model).stem
            predict_batch_comparison(
                model_path=args.model,
                images_com_labels=images,
                use_tflite=args.tflite,
                titulo="Demonstração — Classificação de Módulos Fotovoltaicos",
                output_name=f'comparativo_demo_modelo_{model_name}.png'
            )

    # ── Modo 4: sem argumentos — mostra exemplos ─────────────────
    else:
        print("Uso:")
        print()
        print("  # Processa uma pasta inteira (gera painel com todas as imagens):")
        print("  python predict_individual.py --image data_split/test/no_anomaly/images --model models/mobilenetv2.keras")
        print("  python predict_individual.py --image data_split/test/anomaly/images    --model models/mobilenetv2.keras")
        print()
        print("  # O rótulo real é inferido automaticamente pelo nome da pasta.")
        print("  # Para definir manualmente: adicione --real normal  ou  --real defeito")
        print()
        print("  # Imagem única:")
        print("  python predict_individual.py --image data_split/test/no_anomaly/images/10353.jpg --real normal --model models/mobilenetv2.keras")
        print()
        print("  # Com modelo quantizado:")
        print("  python predict_individual.py --image data_split/test/anomaly/images --model models/mobilenetv2_f16.tflite --tflite")
        print()
        print("  Saída gerada em: results/comparativo_{pasta}_modelo_{modelo}.png")
