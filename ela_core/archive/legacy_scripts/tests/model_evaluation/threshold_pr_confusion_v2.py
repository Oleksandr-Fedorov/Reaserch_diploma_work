"""Evaluate model V2 with a validation-selected threshold.

The script derives a threshold from the train-derived validation subset, then
applies that fixed threshold to the isolated test split and reports PR-AUC plus
the confusion matrix.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, precision_recall_curve, average_precision_score

# --- 1. НАСТРОЙКИ ---
MODEL_PATH = "models/model_resnet_ela_LATEST_Kaggle_5.pth" # Твои текущие веса (где было 69%)
TRAIN_DIR = "D:/Diplom/ela_core/datasets/CASIA_Unified_Randomized_v2/train"
TEST_DIR = "D:/Diplom/ela_core/datasets/CASIA_Unified_Randomized_v2/test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT_DIR / "artifacts" / "images" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. КЛАСС ДАТАСЕТА ---
class ELADataset(Dataset):
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

# --- 3. ЧЕСТНОЕ РАЗБИЕНИЕ (ИЗОЛЯЦИЯ TEST) ---
full_train_dataset = ELADataset(folder_path=TRAIN_DIR, transform=transform)
test_dataset = ELADataset(folder_path=TEST_DIR, transform=transform)

val_size = int(len(full_train_dataset) * 0.1)
train_size = len(full_train_dataset) - val_size

# Строго тот же seed, чтобы получить ту же Val, что и при обучении!
_, val_subset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- 4. ЗАГРУЗКА МОДЕЛИ ---
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# --- 5. ФАЗА 1: ПОИСК ПОРОГА НА VAL ---
print("🔍 ФАЗА 1: Поиск оптимального порога на Validation выборке...")
val_labels, val_probs = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        probs = torch.softmax(model(images), dim=1)
        val_labels.extend(labels.numpy())
        val_probs.extend(probs[:, 1].cpu().numpy())

fpr_v, tpr_v, thresh_v = roc_curve(val_labels, val_probs)
optimal_idx = np.argmax(tpr_v - fpr_v)
optimal_threshold = thresh_v[optimal_idx]
print(f"✅ Оптимальный порог найден на Val: {optimal_threshold:.4f}")

# --- 6. ФАЗА 2: ЧЕСТНЫЙ ЭКЗАМЕН НА TEST ---
print(f"\n🚀 ФАЗА 2: Экзамен на Test выборке с порогом {optimal_threshold:.4f}...")
test_labels, test_probs = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        probs = torch.softmax(model(images), dim=1)
        test_labels.extend(labels.numpy())
        test_probs.extend(probs[:, 1].cpu().numpy())

# Применяем честный порог к тесту
test_preds_optimal = [1 if p >= optimal_threshold else 0 for p in test_probs]

# Считаем PR-AUC
pr_auc = average_precision_score(test_labels, test_probs)

print("\n" + "="*50)
print(f"📊 ФИНАЛЬНЫЙ ОТЧЕТ (Порог: {optimal_threshold:.4f})")
print(f"📉 PR-AUC: {pr_auc:.4f}")
print("="*50)
print(classification_report(test_labels, test_preds_optimal, target_names=['REAL', 'FAKE']))

# --- 7. ОТРИСОВКА МАТРИЦЫ ---
cm_optimal = confusion_matrix(test_labels, test_preds_optimal)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
plt.ylabel('Реальные классы')
plt.xlabel('Предсказанные классы')
plt.title(f'Матрица ошибок (Сдвиг порога до {optimal_threshold:.2f})')
plt.savefig(PLOTS_DIR / "thesis_confusion_matrix_optimal_new_model.png", dpi=300)
print("✅ График матрицы сохранен!")
plt.show()
