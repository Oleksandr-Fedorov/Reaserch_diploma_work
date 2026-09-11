"""
Ultimate Test v4 (Forensic Edition): Влияние серверной настройки ELA.
- Архитектура: Модульная (Manifest -> Run -> Metrics -> Plot)
- Расширенные метрики: Accuracy, Recall, Precision, F1, Specificity, MCC, Mean Probability
- Оптимизация: O(1) поиск, tqdm прогресс-бары, проверки существования файлов
"""

import os
import io
import csv
import math
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm # Прогресс-бары

# --- 0. ФИКСАЦИЯ RANDOM ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# --- 1. НАСТРОЙКИ ---
MODEL_PATH = "models/model_resnet_ela_LATEST_Kaggle_5.pth"
CSV_LOG_PATH = "D:/Diplom/ela_core/datasets/CASIA_Unified_Randomized_v2/dataset_log.csv"

CASIA1_REAL = "D:/Diplom/ela_core/datasets/CASIA1/Au"
CASIA1_FAKE = "D:/Diplom/ela_core/datasets/CASIA1/Sp"
CASIA2_REAL = "D:/Diplom/ela_core/datasets/Some_data_sets/CASIA2/Au"
CASIA2_FAKE = "D:/Diplom/ela_core/datasets/Some_data_sets/CASIA2/Tp"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_QUALITIES = [100, 95, 90, 85, 80, 75, 70, 60, 50]
SERVER_INFERENCE_SETTINGS = [75, 85, 95, 100] 
NUM_IMAGES_PER_CLASS = 500 
OPTIMAL_THRESHOLD = 0.4327 

ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "images" / "plots"

MANIFEST_PATH = ARTIFACTS_DIR / "sweep_manifest_n1000.csv"
METRICS_CSV_PATH = ARTIFACTS_DIR / "sweep_metrics_results.csv"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# --- МОДУЛЬ 1: МАНИФЕСТ ---
def build_manifest():
    if MANIFEST_PATH.exists():
        print("📁 Чтение существующего манифеста...")
        fakes, reals = [], []
        with open(MANIFEST_PATH, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['class'] == 'FAKE': fakes.append(row['path'])
                else: reals.append(row['path'])
        return reals, fakes
    
    print(f"⚠️ Генерация манифеста (Max {NUM_IMAGES_PER_CLASS} на класс)...")
    # Оптимизация 1: Использование множеств (set) для O(1) поиска
    safe_reals_set, safe_fakes_set = set(), set()
    
    with open(CSV_LOG_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'test': 
                full_path = ""
                if row['source_dataset'] == 'CASIA1':
                    full_path = os.path.join(CASIA1_REAL if row['class'] == 'REAL' else CASIA1_FAKE, row['original_filename'])
                elif row['source_dataset'] == 'CASIA2':
                    full_path = os.path.join(CASIA2_REAL if row['class'] == 'REAL' else CASIA2_FAKE, row['original_filename'])
                
                # Оптимизация 2: Проверка физического существования файла
                if os.path.exists(full_path):
                    if row['class'] == 'REAL': safe_reals_set.add(full_path)
                    elif row['class'] == 'FAKE': safe_fakes_set.add(full_path)
    
    safe_fakes_paths = list(safe_fakes_set)
    safe_reals_paths = list(safe_reals_set)
    
    selected_fakes = random.sample(safe_fakes_paths, min(NUM_IMAGES_PER_CLASS, len(safe_fakes_paths)))
    selected_reals = random.sample(safe_reals_paths, min(NUM_IMAGES_PER_CLASS, len(safe_reals_paths)))
    
    with open(MANIFEST_PATH, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["class", "path"])
        for path in selected_fakes: writer.writerow(["FAKE", path])
        for path in selected_reals: writer.writerow(["REAL", path])
        
    print(f"✅ Манифест зафиксирован. FAKE: {len(selected_fakes)}, REAL: {len(selected_reals)}")
    return selected_reals, selected_fakes

# --- ФИЗИКА ELA ---
def get_ela_image(img: Image.Image, quality: int):
    img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")
    ela_array = np.abs(np.array(img, dtype=float) - np.array(compressed, dtype=float)) * 10
    return Image.fromarray(np.clip(ela_array, 0, 255).astype(np.uint8))

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- МОДУЛЬ 2: ЭКСПЕРИМЕНТ ---
def run_experiment(model, reals_paths, fakes_paths):
    # Добавлена новая структура для сбора средних вероятностей
    results = {
        server_q: {q: {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "probs": []} for q in TEST_QUALITIES} 
        for server_q in SERVER_INFERENCE_SETTINGS
    }
    
    with torch.no_grad():
        def process_files(file_list, true_label_idx, desc):
            # Оптимизация 3: Использование tqdm для красивого логгирования
            for img_path in tqdm(file_list, desc=desc):
                try:
                    original_img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue
                
                for q_primary in TEST_QUALITIES:
                    sim_buffer = io.BytesIO()
                    original_img.save(sim_buffer, format="JPEG", quality=q_primary)
                    sim_buffer.seek(0)
                    simulated_img = Image.open(sim_buffer).convert("RGB")
                    
                    for server_q in SERVER_INFERENCE_SETTINGS:
                        ela_img = get_ela_image(simulated_img, quality=server_q)
                        tensor = preprocess(ela_img).unsqueeze(0).to(DEVICE)
                        
                        out = model(tensor)
                        prob = torch.softmax(out, dim=1)[0, 1].item()
                        pred = 1 if prob >= OPTIMAL_THRESHOLD else 0
                        
                        # Сохраняем вероятность для анализа
                        results[server_q][q_primary]["probs"].append(prob)
                        
                        if true_label_idx == 1: 
                            if pred == 1: results[server_q][q_primary]["TP"] += 1
                            else: results[server_q][q_primary]["FN"] += 1
                        else: 
                            if pred == 1: results[server_q][q_primary]["FP"] += 1
                            else: results[server_q][q_primary]["TN"] += 1

        print("\n")
        process_files(fakes_paths, 1, "Анализ FAKE")
        process_files(reals_paths, 0, "Анализ REAL")
        
    return results

# --- МОДУЛЬ 3: МЕТРИКИ И CSV ---
def calculate_and_save_metrics(results, total_images):
    metrics = {server_q: {"acc": [], "rec": [], "prec": [], "f1": [], "spec": [], "mcc": [], "mean_prob": []} 
               for server_q in SERVER_INFERENCE_SETTINGS}
    
    print(f"\n💾 Сохранение forensic-метрик в {METRICS_CSV_PATH}...")
    with open(METRICS_CSV_PATH, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Server_Q", "Primary_Q", "TP", "FP", "TN", "FN", "Accuracy", "Recall", "Precision", "F1_Score", "Specificity", "MCC", "Mean_Prob_FAKE"])
        
        for server_q in SERVER_INFERENCE_SETTINGS:
            for q in TEST_QUALITIES:
                r = results[server_q][q]
                tp, fp, tn, fn = r["TP"], r["FP"], r["TN"], r["FN"]
                
                # Базовые метрики
                acc = ((tp + tn) / total_images) * 100 if total_images > 0 else 0
                rec = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
                prec = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
                
                # Новые Forensic метрики
                spec = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
                
                # Расчет MCC с защитой от деления на ноль
                denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                mcc = ((tp * tn) - (fp * fn)) / denominator if denominator > 0 else 0
                
                # Средняя вероятность
                mean_prob = (sum(r["probs"]) / len(r["probs"])) * 100 if len(r["probs"]) > 0 else 0
                
                metrics[server_q]["acc"].append(acc)
                metrics[server_q]["rec"].append(rec)
                metrics[server_q]["spec"].append(spec)
                metrics[server_q]["mcc"].append(mcc)
                
                writer.writerow([server_q, q, tp, fp, tn, fn, round(acc, 2), round(rec, 2), round(prec, 2), round(f1, 2), round(spec, 2), round(mcc, 3), round(mean_prob, 2)])
                
    return metrics

# --- МОДУЛЬ 4: ОТРИСОВКА ---
def draw_plots(metrics, total_images):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6)) # Теперь 3 графика
    fig.suptitle(f"Forensic Анализ: Влияние INFERENCE_ELA_QUALITY (N={total_images})", fontsize=16, fontweight='bold')
    
    colors = {75: '#3498db', 85: '#e67e22', 95: '#e74c3c'}
    markers = {75: 'o', 85: 's', 95: 'D'}
    
    for server_q in SERVER_INFERENCE_SETTINGS:
        axes[0].plot(TEST_QUALITIES, metrics[server_q]["acc"], marker=markers[server_q], color=colors[server_q], linewidth=2.5, label=f"Сервер Q={server_q}")
        axes[1].plot(TEST_QUALITIES, metrics[server_q]["rec"], marker=markers[server_q], color=colors[server_q], linewidth=2.5, label=f"Сервер Q={server_q}")
        axes[2].plot(TEST_QUALITIES, metrics[server_q]["mcc"], marker=markers[server_q], color=colors[server_q], linewidth=2.5, label=f"Сервер Q={server_q}")

    # Оформление графиков
    titles = ["Общая Точность (Accuracy)", "Обнаружение подделок (FAKE Recall)", "Метрика качества (MCC)"]
    ylabels = ["Accuracy (%)", "Recall (%)", "Matthews Correlation (-1 to 1)"]
    
    for i in range(3):
        axes[i].set_title(titles[i], fontsize=12)
        axes[i].set_xlabel("Первичное качество сжатия (Q)", fontsize=10)
        axes[i].set_ylabel(ylabels[i], fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='lower left')
        axes[i].set_xlim(axes[i].get_xlim()[::-1]) # Разворот оси X
        
        if i < 2: axes[i].axhline(50, color='gray', linestyle='--', alpha=0.5, label="Случайность")
        else: axes[i].axhline(0, color='gray', linestyle='--', alpha=0.5, label="Случайность")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "thesis_forensic_sweep_n1000.png", dpi=300)
    print(f"✅ Графики сохранены.")
    plt.show()

# --- ОСНОВНОЙ ВЫЗОВ ---
if __name__ == "__main__":
    print(f"🚀 Инициализация Forensic пайплайна на {DEVICE}...")
    
    # Загрузка модели
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    # Вызов модулей
    reals, fakes = build_manifest()
    total = len(reals) + len(fakes)
    
    raw_results = run_experiment(model, reals, fakes)
    final_metrics = calculate_and_save_metrics(raw_results, total)
    draw_plots(final_metrics, total)