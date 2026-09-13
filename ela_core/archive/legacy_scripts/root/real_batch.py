import os
from PIL import Image

# Укажи тот же путь к исходникам CASIA
CASIA_DIR = r"datasets/" 
au_dir = os.path.join(CASIA_DIR, "Au")
sp_dir = os.path.join(CASIA_DIR, "Sp")

# Новая папка для чистых картинок
OUTPUT_DIR = "datasets/raw_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def prepare_raw(folder_path, label):
    print(f"\nНачинаем копирование папки: {label}...")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.png'))]
    
    for filename in files:
        img_path = os.path.join(folder_path, filename)
        try:
            # Просто открываем картинку (без ELA)
            img = Image.open(img_path).convert('RGB')
            save_name = f"{label}_{filename}"
            
            # На случай, если в датасете есть .tif файлы, меняем им хвост на .jpg
            if not save_name.lower().endswith('.jpg'):
                save_name = save_name.rsplit('.', 1)[0] + '.jpg'
                
            # Сохраняем в максимальном качестве, чтобы не плодить новые артефакты
            img.save(os.path.join(OUTPUT_DIR, save_name), quality=100)
        except Exception as e:
            print(f"❌ Ошибка с файлом {filename}: {e}")

prepare_raw(au_dir, "REAL")
prepare_raw(sp_dir, "FAKE")

print(f"\n🎉 Готово! Проверь папку '{OUTPUT_DIR}'.")