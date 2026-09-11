"""Compare two ELA models under JPEG recompression on CASIA1 test images.

The script samples REAL/FAKE images, recompresses each image at several JPEG
qualities, runs both models with their own thresholds, and plots accuracy plus
FAKE recall.
"""

import os
import io
import csv
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

# --- 0. ФИКСАЦИЯ RANDOM ДЛЯ НАУЧНОЙ ТОЧНОСТИ ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# --- 1. НАСТРОЙКИ ---
MODEL_V1_PATH = "models/model_resnet_ela_LATEST_.pth" 
MODEL_V2_PATH = "models/model_resnet_ela_LATEST_Kaggle.pth" 

# ПРАВИЛЬНЫЕ ПУТИ К ИСХОДНИКАМ CASIA1
DIR_FAKE = "D:/Diplom/ela_core/datasets/CASIA1/Sp" 
DIR_REAL = "D:/Diplom/ela_core/datasets/CASIA1/Au" 

# Путь к логу для извлечения честной тестовой выборки
CSV_LOG_PATH = "D:/Diplom/ela_core/datasets/CASIA_Unified_Randomized_v2/dataset_log.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_QUALITIES = list(range(100, 45, -5))
NUM_IMAGES_PER_CLASS = 100 
ROOT_DIR = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT_DIR / "artifacts" / "images" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ЧЕСТНЫЕ ПОРОГИ ДЛЯ КАЖДОЙ МОДЕЛИ
THRESH_V2 = 0.4327
THRESH_V1 = 0.5036

# --- 2. МАТЕМАТИКА ELA ---
def get_ela_image(img: Image.Image, quality: int = 75):
    img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")
    ela_array = np.abs(np.array(img, dtype=float) - np.array(compressed, dtype=float)) * 10
    ela_array = np.clip(ela_array, 0, 255).astype(np.uint8)
    return Image.fromarray(ela_array)

# --- 3. ЗАГРУЗКА ДВУХ НЕЙРОСЕТЕЙ ---
def load_model(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. ЧТЕНИЕ CSV ДЛЯ БЕЗОПАСНОЙ ВЫБОРКИ ---
def get_safe_test_files():
    safe_reals, safe_fakes = set(), set()
    if not os.path.exists(CSV_LOG_PATH):
        print(f"⚠️ CSV не найден! Тест будет нечестным (Data Leakage возможен).")
        return [], []
        
    with open(CSV_LOG_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'test' and row['source_dataset'] == 'CASIA1':
                if row['class'] == 'REAL':
                    safe_reals.add(row['original_filename'])
                elif row['class'] == 'FAKE':
                    safe_fakes.add(row['original_filename'])
    return list(safe_reals), list(safe_fakes)

# --- 5. ЯДРО ЭКСПЕРИМЕНТА ---
def run_comparison():
    print(f"🚀 Запуск Научного Сравнительного Теста на {DEVICE}...")
    
    model_v1 = load_model(MODEL_V1_PATH)
    model_v2 = load_model(MODEL_V2_PATH)
    
    safe_reals, safe_fakes = get_safe_test_files()
    
    # Собираем полные пути только для безопасных файлов
    all_fakes = [
        os.path.join(DIR_FAKE, f)
        for f in os.listdir(DIR_FAKE)
        if not safe_fakes or f in safe_fakes
    ]
    all_reals = [
        os.path.join(DIR_REAL, f)
        for f in os.listdir(DIR_REAL)
        if not safe_reals or f in safe_reals
    ]
    
    selected_fakes = random.sample(all_fakes, min(NUM_IMAGES_PER_CLASS, len(all_fakes)))
    selected_reals = random.sample(all_reals, min(NUM_IMAGES_PER_CLASS, len(all_reals)))
    
    results = {
        "v1": {q: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for q in TEST_QUALITIES},
        "v2": {q: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for q in TEST_QUALITIES}
    }
    
    total_fakes = len(selected_fakes)
    total_reals = len(selected_reals)
    total_images = total_fakes + total_reals
    print(f"Обработка {total_images} (F:{total_fakes}, R:{total_reals}) ЧИСТЫХ тестовых изображений...")
    
    with torch.no_grad():
        def process_files(file_list, true_label_idx):
            for i, img_path in enumerate(file_list):
                try:
                    original_img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue
                    
                for q in TEST_QUALITIES:
                    sim_buffer = io.BytesIO()
                    original_img.save(sim_buffer, format="JPEG", quality=q)
                    sim_buffer.seek(0)
                    simulated_img = Image.open(sim_buffer).convert("RGB")
                    
                    ela_img = get_ela_image(simulated_img, quality=75)
                    input_tensor = preprocess(ela_img).unsqueeze(0).to(DEVICE)
                    
                    # V1 (Используем её порог)
                    out_v1 = model_v1(input_tensor)
                    prob_v1 = torch.softmax(out_v1, dim=1)[0, 1].item()
                    pred_v1 = 1 if prob_v1 >= THRESH_V1 else 0
                    
                    if true_label_idx == 1:
                        if pred_v1 == 1: results["v1"][q]["TP"] += 1
                        else: results["v1"][q]["FN"] += 1
                    else:
                        if pred_v1 == 1: results["v1"][q]["FP"] += 1
                        else: results["v1"][q]["TN"] += 1

                    # V2 (Используем её порог)
                    out_v2 = model_v2(input_tensor)
                    prob_v2 = torch.softmax(out_v2, dim=1)[0, 1].item()
                    pred_v2 = 1 if prob_v2 >= THRESH_V2 else 0
                    
                    if true_label_idx == 1:
                        if pred_v2 == 1: results["v2"][q]["TP"] += 1
                        else: results["v2"][q]["FN"] += 1
                    else:
                        if pred_v2 == 1: results["v2"][q]["FP"] += 1
                        else: results["v2"][q]["TN"] += 1
                        
                if (i + 1) % 20 == 0:
                    print(f"  Обработано {i + 1}/{len(file_list)} файлов в текущем классе...")

        process_files(selected_fakes, 1)
        process_files(selected_reals, 0)

    # --- МАТЕМАТИКА МЕТРИК ---
    acc_v1, acc_v2 = [], []
    rec_v1, rec_v2 = [], []
    
    for q in TEST_QUALITIES:
        r_v1 = results["v1"][q]
        r_v2 = results["v2"][q]
        
        acc_v1.append(((r_v1["TP"] + r_v1["TN"]) / total_images) * 100)
        acc_v2.append(((r_v2["TP"] + r_v2["TN"]) / total_images) * 100)
        
        rec_v1.append((r_v1["TP"] / (r_v1["TP"] + r_v1["FN"])) * 100 if (r_v1["TP"] + r_v1["FN"]) > 0 else 0)
        rec_v2.append((r_v2["TP"] / (r_v2["TP"] + r_v2["FN"])) * 100 if (r_v2["TP"] + r_v2["FN"]) > 0 else 0)

    # --- ТЕКСТОВЫЙ ЛОГ (ТАБЛИЦА) ---
    print("\n" + "="*70)
    print("📊 СРАВНЕНИЕ УСТОЙЧИВОСТИ: ACCURACY (%)")
    print("="*70)
    print(f"{'Q_primary':<10} | {'V1 (x1)':<15} | {'V2 (x4)':<15} | {'Дельта (Δ)':<15}")
    print("-" * 65)
    for i, q in enumerate(TEST_QUALITIES):
        delta = acc_v2[i] - acc_v1[i]
        print(f"{q:<10} | {acc_v1[i]:<15.2f} | {acc_v2[i]:<15.2f} | {delta:<15.2f}")
        
    print("\n" + "="*70)
    print("🎯 СРАВНЕНИЕ УСТОЙЧИВОСТИ: FAKE RECALL (%)")
    print("="*70)
    print(f"{'Q_primary':<10} | {'V1 (x1)':<15} | {'V2 (x4)':<15} | {'Дельта (Δ)':<15}")
    print("-" * 65)
    for i, q in enumerate(TEST_QUALITIES):
        delta = rec_v2[i] - rec_v1[i]
        print(f"{q:<10} | {rec_v1[i]:<15.2f} | {rec_v2[i]:<15.2f} | {delta:<15.2f}")
    print("="*70 + "\n")

    # --- ОТРИСОВКА 2-Х ГРАФИКОВ ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Доказательство гипотезы: Повышение устойчивости к JPEG-сжатию (Модель V1 vs V2)", fontsize=16, fontweight='bold')
    
    # График 1: Accuracy
    ax1.plot(TEST_QUALITIES, acc_v1, marker='o', color='#7f8c8d', linestyle='--', linewidth=2, label=f"V1 (x1 Aug, Thr={THRESH_V1})")
    ax1.plot(TEST_QUALITIES, acc_v2, marker='D', color='#2ecc71', linewidth=3, label=f"V2 (x4 Aug, Thr={THRESH_V2})")
    ax1.set_title("Сравнение Общей Точности (Accuracy)", fontsize=12)
    ax1.set_xlabel("Первичное качество сжатия (Q)", fontsize=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.axvline(75, color='#f1c40f', linestyle=':', linewidth=2, alpha=0.8)
    ax1.axhline(50, color='red', linestyle='-', alpha=0.3, label="Случайное угадывание (50%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left')
    
    # Исправление бага с matplotlib
    ax1.set_xlim(ax1.get_xlim()[::-1])
    
    # График 2: FAKE Recall
    ax2.plot(TEST_QUALITIES, rec_v1, marker='o', color='#7f8c8d', linestyle='--', linewidth=2, label="V1 (x1 Aug)")
    ax2.plot(TEST_QUALITIES, rec_v2, marker='s', color='#e74c3c', linewidth=3, label="V2 (x4 Aug)")
    ax2.set_title("Сравнение Полноты Обнаружения Подделок (FAKE Recall)", fontsize=12)
    ax2.set_xlabel("Первичное качество сжатия (Q)", fontsize=10)
    ax2.set_ylabel("Recall FAKE (%)", fontsize=10)
    ax2.axvline(75, color='#f1c40f', linestyle=':', linewidth=2, alpha=0.8, label="Ось ELA (Q=75)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower left')
    
    ax2.set_xlim(ax2.get_xlim()[::-1])
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "thesis_final_comparison_sweep.png", dpi=300)
    print("✅ Графики сохранены: thesis_final_comparison_sweep.png")
    plt.show()

if __name__ == "__main__":
    run_comparison()
