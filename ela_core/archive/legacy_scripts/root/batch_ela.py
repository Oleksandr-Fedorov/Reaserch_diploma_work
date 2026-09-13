import os
from PIL import Image
from services.ela_logic import ela

# --- ВАЖНО: Вставь сюда полный путь к твоей скачанной папке CASIA1 ---
# Например: r"D:\Downloads\casia_dataset\CASIA1" (буква r перед кавычками обязательна!)
CASIA_DIR = r"datasets/" 

au_dir = os.path.join(CASIA_DIR, "Au")
sp_dir = os.path.join(CASIA_DIR, "Sp")

# Создаем папку для готовых рентгенов прямо в проекте
OUTPUT_DIR = "datasets/ela_diagrams"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_batch(folder_path, label):
    print(f"\nНачинаем обработку папки: {label}...")
    
    # Берем первые 10 картинок (игнорируем скрытые файлы и системный мусор)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.png'))]
    
    for filename in files:
        img_path = os.path.join(folder_path, filename)
        
        try:
            # 1. Прогоняем через твою математику
            ela_array = ela(img_path)
            
            # 2. Превращаем матрицу обратно в картинку и сохраняем
            result_img = Image.fromarray(ela_array)
            save_name = f"{label}_{filename}.jpg"
            result_img.save(os.path.join(OUTPUT_DIR, save_name))
            
            print(f"✅ Сохранено: {save_name}")
        except Exception as e:
            print(f"❌ Ошибка с файлом {filename}: {e}")

# Запускаем конвейер!
process_batch(au_dir, "REAL")
process_batch(sp_dir, "FAKE")

print(f"\n🎉 Готово! Проверь папку '{OUTPUT_DIR}' слева в VS Code.")