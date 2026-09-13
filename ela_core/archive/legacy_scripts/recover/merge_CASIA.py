import os
import io
import random
import csv
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- 0. БЕЗОПАСНОСТЬ И ВОСПРОИЗВОДИМОСТЬ ---
random.seed(42)
np.random.seed(42)

# Отключаем лимит на размер картинки (ТОЛЬКО ДЛЯ ЛОКАЛЬНОГО СКРИПТА!)
# Некоторые оригиналы CASIA в формате TIFF очень большие.
Image.MAX_IMAGE_PIXELS = None 

# --- 1. НАСТРОЙКИ ---
CASIA1_REAL = "D:/Diplom/ela_core/datasets/CASIA1/Au" 
CASIA1_FAKE = "D:/Diplom/ela_core/datasets/CASIA1/Sp" 

CASIA2_REAL = "D:/Diplom/ela_core/datasets/Some_data_sets/CASIA2/Au"
CASIA2_FAKE = "D:/Diplom/ela_core/datasets/Some_data_sets/CASIA2/Tp"

OUTPUT_DIR = "D:/Diplom/ela_core/datasets/CASIA_Unified_Randomized"

# --- 2. ЛОГИКА АУГМЕНТАЦИИ ---
AUGMENTATIONS_PER_IMAGE = 1 

# Реалистичные шаги JPEG-сжатия
JPEG_PRESETS_PRIMARY = [65, 70, 75, 80, 85, 90, 95, 100]
JPEG_PRESETS_ELA = [65, 70, 75, 80, 85, 90, 95]

def create_dirs():
    for split in ['train', 'test']:
        for cls in ['REAL', 'FAKE']:
            Path(f"{OUTPUT_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

def generate_augmentations(img_path, output_dir, source_dataset, split, cls, original_filename, csv_writer):
    try:
        img = Image.open(img_path).convert("RGB")
        base_name = os.path.splitext(original_filename)[0]
        
        for aug_idx in range(1, AUGMENTATIONS_PER_IMAGE + 1):
            # Независимый выбор параметров
            q_primary = random.choice(JPEG_PRESETS_PRIMARY)
            q_ela = random.choice(JPEG_PRESETS_ELA)
            
            # 1. Симуляция первичного сжатия
            sim_buffer = io.BytesIO()
            img.save(sim_buffer, format="JPEG", quality=q_primary)
            sim_buffer.seek(0)
            simulated_img = Image.open(sim_buffer).convert("RGB")
            
            # 2. Качество алгоритма ELA
            ela_buffer = io.BytesIO()
            simulated_img.save(ela_buffer, format="JPEG", quality=q_ela)
            ela_buffer.seek(0)
            compressed_for_ela = Image.open(ela_buffer).convert("RGB")
            
            # 3. Математика ELA
            ela_array = np.abs(np.array(simulated_img, dtype=float) - np.array(compressed_for_ela, dtype=float)) * 10
            ela_array = np.clip(ela_array, 0, 255).astype(np.uint8)
            
            # 4. Формирование имени и сохранение в PNG (БЕЗ ПОТЕРЬ!)
            suffix = f"_aug{aug_idx}" if AUGMENTATIONS_PER_IMAGE > 1 else ""
            new_filename = f"{source_dataset}_{cls}_{base_name}{suffix}.png"
            output_path = os.path.join(output_dir, split, cls, new_filename)
            
            # Сохраняем математику как есть
            ela_img = Image.fromarray(ela_array)
            ela_img.save(output_path, format="PNG")
            
            # 5. Запись в лог CSV
            csv_writer.writerow([new_filename, source_dataset, original_filename, split, cls, q_primary, q_ela])
            
        return True
    except Exception as e:
        print(f"⚠️ Ошибка с файлом {img_path}: {e}")
        return False

def get_files(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ ОШИБКА: Папка не найдена -> {folder_path}")
        return []
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

def build_dataset():
    create_dirs()
    print("🛠 Сбор файлов и стратифицированное разделение...")
    
    c1_real = get_files(CASIA1_REAL)
    c1_fake = get_files(CASIA1_FAKE)
    c2_real = get_files(CASIA2_REAL)
    c2_fake = get_files(CASIA2_FAKE)
    
    c1_real_train, c1_real_test = train_test_split(c1_real, test_size=0.2, random_state=42)
    c1_fake_train, c1_fake_test = train_test_split(c1_fake, test_size=0.2, random_state=42)
    
    c2_real_train, c2_real_test = train_test_split(c2_real, test_size=0.2, random_state=42)
    c2_fake_train, c2_fake_test = train_test_split(c2_fake, test_size=0.2, random_state=42)
    
    tasks = [
        (c1_real_train, "CASIA1", "train", "REAL"),
        (c1_real_test,  "CASIA1", "test",  "REAL"),
        (c1_fake_train, "CASIA1", "train", "FAKE"),
        (c1_fake_test,  "CASIA1", "test",  "FAKE"),
        
        (c2_real_train, "CASIA2", "train", "REAL"),
        (c2_real_test,  "CASIA2", "test",  "REAL"),
        (c2_fake_train, "CASIA2", "train", "FAKE"),
        (c2_fake_test,  "CASIA2", "test",  "FAKE"),
    ]
    
    csv_path = os.path.join(OUTPUT_DIR, "dataset_log.csv")
    total_processed = 0
    total_generated = 0
    
    print("\n🚀 Начинаем генерацию Двойной Рандомизации (Формат: PNG)...")
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "source_dataset", "original_filename", "split", "class", "q_primary", "q_ela"])
        
        for file_list, source_dataset, split, cls in tasks:
            print(f"\n▶ Обработка {source_dataset} -> {split}/{cls} ({len(file_list)} исходников)...")
            
            for i, filepath in enumerate(file_list):
                filename = os.path.basename(filepath)
                
                if generate_augmentations(filepath, OUTPUT_DIR, source_dataset, split, cls, filename, writer):
                    total_processed += 1
                    total_generated += AUGMENTATIONS_PER_IMAGE
                    
                if (i + 1) % 500 == 0:
                    print(f"  Сгенерировано исходников: {i + 1}/{len(file_list)}")
                    
    print("\n" + "="*50)
    print("✅ ДАТАСЕТ УСПЕШНО СГЕНЕРИРОВАН!")
    print(f"Обработано оригиналов: {total_processed}")
    print(f"Создано ELA-карт (PNG): {total_generated}")
    print(f"Журнал параметров сохранен: {csv_path}")
    print("="*50)

if __name__ == "__main__":
    build_dataset()