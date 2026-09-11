from PIL import Image
import numpy as np

def ela(image_path, quality=75):
    # Открываем оригинал
    original = Image.open(image_path).convert('RGB')
    # Сохраняем с заданным качеством (сжатие)
    original.save("temp.jpg", quality=quality)
    # Открываем сжатую копию
    compressed = Image.open("temp.jpg").convert('RGB')
    
    # Математика: вычисляем разницу пикселей и усиливаем ее (* 10)
    ela_image = np.abs(
        np.array(original, dtype=float) - 
        np.array(compressed, dtype=float)
    ) * 10
    return np.clip(ela_image, 0, 255).astype(np.uint8)