import io
import base64
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageChops, ImageEnhance
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. НАСТРОЙКА ПАРАМЕТРОВ ELA (Должны строго совпадать с твоим датасетом!)
JPEG_QUALITY = 75

# 2. ИНИЦИАЛИЗАЦИЯ И НАСТРОЙКА СЕРВЕРА
app = FastAPI(title="ELA Core Fake Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене поменяй на адрес своего Next.js (например, http://localhost:3000)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. ПОДГОТОВКА УСТРОЙСТВА И МОДЕЛИ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Вспомогательная функция для сборки архитектуры и загрузки весов
def load_resnet_model(weights_path: str):
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # Загружаем твои сохраненные веса fine-tuned модели
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"🧠 Модель успешно загружена из {weights_path} ({device})")
    except Exception as e:
        print(f"❌ Ошибка загрузки весов: {e}. Запущена пустая модель для теста.")
    
    model = model.to(device)
    model.eval()
    return model

# Загружаем модель при старте (укажи правильное имя твоего файла весов)
MODEL_PATH = "models/ELA_model/model_resnet_ela_sourceaware_v2.pth"
model = load_resnet_model(MODEL_PATH)

# Трансформации для ResNet (классический ImageNet)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОБРАБОТКИ
def run_ela_logic(img: Image.Image, quality: int = 75) -> Image.Image:
    """Генерация ELA строго по математике обучающего датасета (в оперативной памяти)"""
    img = img.convert("RGB")
    
    # 1. Сжатие в памяти (замена сохранения temp.jpg на диск)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")
    
    # 2. МАТЕМАТИКА 1 В 1 КАК В ТВОЕМ СКРИПТЕ ОБУЧЕНИЯ
    ela_array = np.abs(
        np.array(img, dtype=float) - 
        np.array(compressed, dtype=float)
    ) * 10
    
    # 3. Обрезка значений и конвертация обратно в картинку
    ela_array = np.clip(ela_array, 0, 255).astype(np.uint8)
    return Image.fromarray(ela_array)

def pil_to_base64(img: Image.Image) -> str:
    """Конвертация PIL Image в строку Base64 для передачи во фронтенд"""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# 5. ГЛАВНЫЙ ЭНДПОИНТ ДЛЯ АНАЛИЗА
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        # Читаем байты и создаем PIL Image оригинального изображения
        contents = await file.read()
        original_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Шаг 1: Генерируем ELA-карту
        ela_pil = run_ela_logic(original_pil, JPEG_QUALITY)
        
        # Шаг 2: Подготовка тензора для нейросети
        input_tensor = preprocess(ela_pil).unsqueeze(0).to(device)
        
        # Шаг 3: Инференс (Прямой проход)
        # Нам нужны градиенты для Grad-CAM, поэтому torch.no_grad() здесь НЕ используем
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(outputs, dim=1).item()
        
        confidence = float(probabilities[pred_class].item())
        verdict = "FAKE" if pred_class == 1 else "REAL"
        
        # Шаг 4: Генерация тепловой карты Grad-CAM
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(pred_class)]
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Подготовка фонового изображения для наложения Grad-CAM
        # Нам нужно денормализовать трансформированную ELA-картинку (как делали в Colab)
        img_for_cam = preprocess(ela_pil).cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_for_cam = std * img_for_cam + mean
        img_for_cam = np.clip(img_for_cam, 0, 1)
        
        # Накладываем тепловую карту на ELA картинку
        cam_image = show_cam_on_image(img_for_cam, grayscale_cam, use_rgb=True)
        cam_pil = Image.fromarray((cam_image * 255).astype(np.uint8))
        
        # Шаг 5: Кодируем результаты в Base64 для передачи на фронтенд
        ela_base64 = pil_to_base64(ela_pil.resize((400, 400)))
        cam_base64 = pil_to_base64(cam_pil.resize((400, 400)))
        
        return {
            "status": "success",
            "filename": file.filename,
            "verdict": verdict,
            "confidence": round(confidence * 100, 2),  # Переводим в проценты, например 94.25
            "ela_image": f"data:image/png;base64,{ela_base64}",
            "gradcam_image": f"data:image/png;base64,{cam_base64}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.get("/api/health")
async def health():
    return {"status": "alive", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)