"""Measure how ELA statistics change under double JPEG quantization.

The script recompresses one source image at several primary JPEG qualities,
then computes ELA mean, standard deviation, and entropy for two ELA quality
settings.
"""

import io
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- НАСТРОЙКИ ---
# Возьми ЛЮБОЙ оригинальный файл из датасета (не фейк)
TEST_IMAGE_PATH = "D:/Diplom/ela_core/datasets/Some_data_sets/CASIA2/Au/Au_ani_00001.jpg"
TEST_QUALITIES = list(range(100, 45, -5)) # От 100 до 50 с шагом 5
ROOT_DIR = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT_DIR / "artifacts" / "images" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def calculate_entropy(img_array):
    """Вычисляет информационную энтропию Шеннона для ELA-карты"""
    # Считаем гистограмму яркости пикселей (от 0 до 255)
    hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
    hist = hist[hist > 0] # Убираем нули для логарифма
    probs = hist / np.sum(hist)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def get_ela_metrics(img: Image.Image, quality: int):
    """Генерирует ELA и возвращает три математические метрики"""
    img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")
    
    # Математика ELA
    ela_array = np.abs(np.array(img, dtype=float) - np.array(compressed, dtype=float)) * 10
    ela_array = np.clip(ela_array, 0, 255).astype(np.uint8)
    
    # Считаем метрики
    mean_val = np.mean(ela_array)
    std_val = np.std(ela_array)
    entropy_val = calculate_entropy(ela_array)
    
    return mean_val, std_val, entropy_val

def run_physics_experiment():
    print("🔬 ЗАПУСК: Анализ математического резонанса (Quantization Alignment)")
    original_img = Image.open(TEST_IMAGE_PATH).convert("RGB")
    
    # Хранилища результатов для Q_ela = 75
    means_75, stds_75, entropies_75 = [], [], []
    # Хранилища результатов для Q_ela = 90
    means_90, stds_90, entropies_90 = [], [], []
    
    for q_primary in TEST_QUALITIES:
        # 1. Симуляция первичного сжатия
        sim_buffer = io.BytesIO()
        original_img.save(sim_buffer, format="JPEG", quality=q_primary)
        sim_buffer.seek(0)
        simulated_img = Image.open(sim_buffer).convert("RGB")
        
        # 2. Снимаем метрики при анализе ELA=75
        m75, s75, e75 = get_ela_metrics(simulated_img, quality=75)
        means_75.append(m75)
        stds_75.append(s75)
        entropies_75.append(e75)
        
        # 3. Снимаем метрики при анализе ELA=90
        m90, s90, e90 = get_ela_metrics(simulated_img, quality=90)
        means_90.append(m90)
        stds_90.append(s90)
        entropies_90.append(e90)
        
        print(f"Обработан первичный JPEG Q={q_primary}")

    # --- ОТРИСОВКА ГРАФИКОВ ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Потеря дискриминирующей информации при эффекте двойного квантования (Quantization Alignment)", fontsize=16, fontweight='bold')
    
    # Функция для настройки каждого подграфика
    def setup_ax(ax, title, y_label, val_75, val_90):
        ax.plot(TEST_QUALITIES, val_75, marker='o', label='Анализ ELA (Q=75)', color='#e74c3c', linewidth=2)
        ax.plot(TEST_QUALITIES, val_90, marker='s', label='Анализ ELA (Q=90)', color='#3498db', linewidth=2)
        ax.invert_xaxis()
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Первичное качество JPEG (Q)", fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.axvline(75, color='#e74c3c', linestyle='--', alpha=0.3)
        ax.axvline(90, color='#3498db', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # График 1: Средняя яркость
    setup_ax(axes[0], "Средняя яркость шума (Mean)", "Интенсивность", means_75, means_90)
    
    # График 2: Дисперсия
    setup_ax(axes[1], "Структурность шума (Variance/Std)", "Стандартное отклонение", stds_75, stds_90)
    
    # График 3: Энтропия
    setup_ax(axes[2], "Информативность (Entropy)", "Энтропия Шеннона", entropies_75, entropies_90)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "thesis_quantization_alignment.png", dpi=300, bbox_inches='tight')
    print("\n✅ Графики успешно сохранены в 'thesis_quantization_alignment.png'")
    plt.show()

if __name__ == "__main__":
    run_physics_experiment()
