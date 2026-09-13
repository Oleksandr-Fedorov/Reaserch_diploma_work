"""Visually compare ELA artifacts with an SRM-style noise residual map.

Use this when checking whether a manipulation is clearer in ELA or in
high-pass residual noise. This explores forensic features; it does not evaluate
a neural model.
"""

import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# УКАЖИ ПУТЬ К ТВОЕМУ КОТУ
TEST_IMAGE_PATH = "D:/Diplom/ela_core/datasets/Some_data_sets/CASIA2/Tp/Tp_D_CRD_S_O_ani10111_ani10103_10635.jpg" 
ROOT_DIR = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT_DIR / "artifacts" / "figures"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def get_ela(img_path, quality=75):
    img = Image.open(img_path).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")
    
    ela_array = np.abs(np.array(img, dtype=float) - np.array(compressed, dtype=float)) * 10
    return np.clip(ela_array, 0, 255).astype(np.uint8)

def get_noise_residual(img_path):
    """
    Извлечение цифрового шума матрицы с помощью фильтра SRM 
    (Spatial Rich Model - стандарт цифровой криминалистики)
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    
    # SRM ядро (High-pass filter), который "убивает" картинку и оставляет только шум
    srm_kernel = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8, -12, 8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1]
    ]) / 12.0
    
    # Применяем фильтр к изображению
    noise_map = cv2.filter2D(img, -1, srm_kernel)
    
    # Усиливаем контраст шума для визуализации
    noise_map = np.abs(noise_map) * 5 
    return np.clip(noise_map, 0, 255).astype(np.uint8)

def run_visual_comparison():
    original = Image.open(TEST_IMAGE_PATH)
    ela_map = get_ela(TEST_IMAGE_PATH)
    noise_map = get_noise_residual(TEST_IMAGE_PATH)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Оригинальный фейк")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(ela_map)
    plt.title("ELA Карта (Слепая зона)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(noise_map, cmap='gray')
    plt.title("Noise Residual (Остаточный шум)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_comparison.png", dpi=300)
    print("✅ Картинка сохранена в feature_comparison.png")
    plt.show()

if __name__ == "__main__":
    run_visual_comparison()
