"""Compute ROC-AUC and an optimal threshold for one baseline ELA model.

This common evaluation script reads a prepared REAL/FAKE test folder, reports
ROC-AUC, selects a threshold using Youden's J statistic, and saves the ROC plot.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report

# --- 1. НАСТРОЙКИ ---
MODEL_PATH = "models/23.pth" # Твои новые веса
# Путь к строгой ТЕСТОВОЙ выборке, которую мы честно отделили от обучения
TEST_DIR = "D:/Diplom/ela_core/datasets/CASIA_Unified_Randomized_v2/test" 
ROOT_DIR = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT_DIR / "artifacts" / "images" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. КЛАСС ДАТАСЕТА И ЗАГРУЗКА ---
class TestDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.transform = transform
        self.samples = []
        for label_name, label_idx in [("REAL", 0), ("FAKE", 1)]:
            class_dir = os.path.join(folder_path, label_name)
            if not os.path.exists(class_dir): continue
            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(class_dir, filename), label_idx))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("🛠 Загрузка тестовых данных...")
test_dataset = TestDataset(folder_path=TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- 3. ЗАГРУЗКА МОДЕЛИ ---
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# --- 4. ПРОГОН И СБОР ВЕРОЯТНОСТЕЙ ---
print(f"🚀 Запуск анализа ROC-AUC ({len(test_dataset)} изображений)...")
all_labels = []
all_probs_fake = [] # Нам нужна вероятность именно класса 1 (FAKE)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        
        all_labels.extend(labels.numpy())
        all_probs_fake.extend(probs[:, 1].cpu().numpy())

# --- 5. МАТЕМАТИКА И ОТРИСОВКА ROC-AUC ---
fpr, tpr, thresholds = roc_curve(all_labels, all_probs_fake)
roc_auc = auc(fpr, tpr)

# Ищем оптимальный порог (Youden's J statistic)
J = tpr - fpr
optimal_idx = np.argmax(J)
optimal_threshold = thresholds[optimal_idx]

print("\n" + "="*50)
print(f"✅ АНАЛИЗ ЗАВЕРШЕН!")
print(f"Итоговый ROC-AUC: {roc_auc:.4f}")
print(f"Оптимальный порог уверенности: {optimal_threshold:.4f}")
print("="*50)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайное угадывание (AUC = 0.50)')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', marker='o', s=100, label=f'Оптимальный порог ({optimal_threshold:.2f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ложноположительная частота (False Positive Rate)')
plt.ylabel('Истинно положительная частота (True Positive Rate)')
plt.title('ROC-кривая модели после двойной рандомизации')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.savefig(PLOTS_DIR / "thesis_roc_auc_curve.png", dpi=300)
plt.show()
