"""
prepare_dataset.py — Organiza o dataset InfraredSolarModules em data/normal e data/defect

Lê o module_metadata.json e copia as imagens para:
    data/normal/   ← imagens com anomaly_class == "No-Anomaly"
    data/defect/   ← todas as outras classes (Cell, Hot-Spot, Shadowing, etc.)

Uso:
    python src/prepare_dataset.py
"""

import json
import shutil
from pathlib import Path

# Caminhos
METADATA_PATH = Path("InfraredSolarModules/2020-02-14_InfraredSolarModules/InfraredSolarModules/module_metadata.json")
IMAGES_DIR = Path("InfraredSolarModules/2020-02-14_InfraredSolarModules/InfraredSolarModules/images")
OUTPUT_DIR = Path("data")

def main():
    print("="*60)
    print("  PREPARAÇÃO DO DATASET")
    print("="*60)

    # Verifica se já existe
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        print(f"\n  Pasta '{OUTPUT_DIR}' já existe e não está vazia.")
        print("  Delete-a manualmente se quiser refazer.")
        return

    # Lê metadata
    print(f"\n  Lendo: {METADATA_PATH}")
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    print(f"  Total de imagens no metadata: {len(metadata)}")

    # Cria pastas de saída
    normal_dir = OUTPUT_DIR / "normal"
    defect_dir = OUTPUT_DIR / "defect"
    normal_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)

    # Classifica e copia
    counts = {"normal": 0, "defect": 0}
    anomaly_classes = {}

    for img_id, info in metadata.items():
        anomaly_class = info["anomaly_class"]
        img_filename = Path(info["image_filepath"]).name
        src_path = IMAGES_DIR / img_filename

        if not src_path.exists():
            continue

        # Contagem por classe original
        anomaly_classes[anomaly_class] = anomaly_classes.get(anomaly_class, 0) + 1

        # Classificação binária
        if anomaly_class == "No-Anomaly":
            dest = normal_dir / img_filename
            counts["normal"] += 1
        else:
            dest = defect_dir / img_filename
            counts["defect"] += 1

        shutil.copy2(src_path, dest)

    # Resumo
    print(f"\n  Classes originais no dataset:")
    for cls, count in sorted(anomaly_classes.items(), key=lambda x: -x[1]):
        label = "→ normal" if cls == "No-Anomaly" else "→ defect"
        print(f"    {cls:<20s} {count:5d} imagens  {label}")

    print(f"\n  Resultado (classificação binária):")
    print(f"    Normal:  {counts['normal']:5d} imagens → data/normal/")
    print(f"    Defeito: {counts['defect']:5d} imagens → data/defect/")
    print(f"    Total:   {counts['normal'] + counts['defect']:5d} imagens")

    print(f"\n  ✅ Dataset organizado em '{OUTPUT_DIR}/'")
    print("="*60)


if __name__ == "__main__":
    main()
