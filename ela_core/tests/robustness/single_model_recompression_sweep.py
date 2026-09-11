"""Run one trained ELA model against selected images after JPEG recompression.

This is a robustness probe, not a full benchmark. It plots how the model's FAKE
probability changes when a source image is saved at different JPEG quality
levels before ELA is applied.
"""

import io
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --- 1. НАСТРОЙКИ ЭКСПЕРИМЕНТА ---
MODEL_PATH = "models/model_resnet_ela_LATEST_Kaggle_5.pth"
MODEL_NAME = "model_resnet_ela_LATEST_Kaggle_5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_QUALITIES = [100, 95, 90, 85, 80, 75, 70, 60, 50]
INFERENCE_ELA_QUALITY = 75
ROOT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = ROOT_DIR / "artifacts" / "images" / "examples"
PLOTS_DIR = ROOT_DIR / "artifacts" / "images" / "plots"
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ВАЖНО: Заполни этот список своими картинками!
# Укажи путь к файлу, реальный класс и понятное название для графика
TEST_IMAGES = [
    {
        "path": "D:/Diplom/ela_core/datasets/Some_data_sets/CASIA2/Tp/Tp_D_CRD_S_O_ani10111_ani10103_10635.jpg", 
        "true_label": "FAKE", 
        "name": "Фейк (Очевидная склейка - Кот)"
    },
    {
        "path": "D:/Diplom/ela_core/datasets/Some_data_sets/CASIA2/Tp/Tp_D_CNN_M_N_cha00026_cha00028_11784.jpg", # Замени на свой файл
        "true_label": "FAKE", 
        "name": "Фейк (Неплохая склейка - Человек)"
    },
    {
        "path": "D:/Diplom/ela_core/datasets/Some_data_sets/CASIA2/Au/Au_ani_00001.jpg", # Найди в датасете оригинальное фото (Au_...)
        "true_label": "REAL", 
        "name": "Оригинал (Контроль)"
    }
]

# --- 2. МАТЕМАТИКА (Твой родной ELA) ---
def run_ela_logic(img: Image.Image, quality: int = 75) -> Image.Image:
    img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")
    ela_array = np.abs(np.array(img, dtype=float) - np.array(compressed, dtype=float)) * 10
    ela_array = np.clip(ela_array, 0, 255).astype(np.uint8)
    return Image.fromarray(ela_array)

# --- 3. ЗАГРУЗКА МОДЕЛИ ---
def load_model():
    print("⚙️ Загрузка нейросети...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. ЯДРО ЭКСПЕРИМЕНТА ---
def run_experiment():
    model = load_model()
    
    plt.figure(figsize=(12, 7))
    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db'] # Цвета для линий
    
    print("\n" + "="*50)
    print("🔬 НАУЧНЫЙ ОТЧЕТ: УЯЗВИМОСТЬ К ПЕРЕЖАТИЮ (REQUANTIZATION)")
    print("="*50)

    for idx, item in enumerate(TEST_IMAGES):
        print(f"\n▶ Тестируем: {item['name']} ({item['path']})")
        original_img = Image.open(item["path"]).convert("RGB")
        
        results_prob = []
        
        with torch.no_grad():
            for q in TEST_QUALITIES:
                # Симуляция пережатия
                sim_buffer = io.BytesIO()
                original_img.save(sim_buffer, format="JPEG", quality=q)
                sim_buffer.seek(0)
                simulated_img = Image.open(sim_buffer).convert("RGB")
                
                # ELA и нейросеть
                ela_img = run_ela_logic(simulated_img, quality=INFERENCE_ELA_QUALITY)
                input_tensor = preprocess(ela_img).unsqueeze(0).to(DEVICE)
                outputs = model(input_tensor)
                
                # ВСЕГДА берем процент уверенности в классе FAKE (индекс 1)
                prob_fake = torch.softmax(outputs, dim=1)[0][1].item() * 100
                results_prob.append(prob_fake)
                print(f"   Q={q}: FAKE={prob_fake:.2f}%")
                if q == 50:
                    save_dir = EXAMPLES_DIR / f"{MODEL_NAME}" / f"{INFERENCE_ELA_QUALITY}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    ela_img.save(save_dir / f"anomaly_q{q}_{item['name']}.png")
        
        # --- АНАЛИТИКА (Метрика нестабильности) ---
        max_swing = max(results_prob) - min(results_prob)
        crosses_threshold = (max(results_prob) > 50) and (min(results_prob) < 50)
        
        print(f"   📉 Диапазон уверенности (FAKE %): от {min(results_prob):.1f}% до {max(results_prob):.1f}%")
        print(f"   ⚠️ Максимальный разброс (Instability Swing): {max_swing:.1f} пунктов")
        if crosses_threshold:
            print("   ❌ ВНИМАНИЕ: Произошла смена вердикта (Crossed 50% threshold)!")
            
        # Отрисовка линии на графике
        line_style = '-' if item["true_label"] == "FAKE" else '--'
        plt.plot(TEST_QUALITIES, results_prob, marker='o', linestyle=line_style, 
                 color=colors[idx % len(colors)], linewidth=2, 
                 label=f'{item["name"]} (Swing: {max_swing:.1f}%)')

    # --- 5. ОФОРМЛЕНИЕ ГРАФИКА ДЛЯ ДИПЛОМА ---
    plt.gca().invert_xaxis() # Ось X: Качество падает слева направо (100 -> 50)
    plt.title("Анализ генерализации ELA: Устойчивость к пережатию JPEG", fontsize=14, fontweight='bold')
    plt.xlabel("Качество симуляции пережатия (JPEG Quality %)", fontsize=12)
    plt.ylabel("Уверенность сети в манипуляции (Вероятность FAKE %)", fontsize=12)
    
    plt.axhspan(0, 50, facecolor='#2ecc71', alpha=0.1) # Зеленая зона (Считает оригиналом)
    plt.axhspan(50, 100, facecolor='#e74c3c', alpha=0.1) # Красная зона (Считает фейком)
    plt.axhline(50, color='gray', linestyle='--', linewidth=1.5)
    
    # Текстовые аннотации зон
    plt.text(98, 95, "Зона обнаружения (Фейк)", color='#c0392b', fontsize=10, fontweight='bold')
    plt.text(98, 5, "Зона пропуска (Оригинал)", color='#27ae60', fontsize=10, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Легенда сбоку
    plt.tight_layout()
    save_dir = PLOTS_DIR / f"{MODEL_NAME}" 
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"thesis_requantization_plot_{INFERENCE_ELA_QUALITY}.png", dpi=300, bbox_inches='tight')
    
    print("\n✅ Эксперимент завершен. График сохранен: thesis_requantization_plot.png")
    plt.show()

if __name__ == "__main__":
    run_experiment()
