# ============================================================
# ELA MODEL EVALUATION SCRIPT (STANDALONE)
# ============================================================

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score

# 1. ЖЕЛЕЗОБЕТОННАЯ ФИКСАЦИЯ ВИПАДКОВОСТІ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 2. КЛАС ДАТАСЕТУ (Работает с предрассчитанными ELA-изображениями)
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
        return image, label, os.path.basename(img_path) 

# 3. ТРАНСФОРМАЦІЇ
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. ПУТИ (Настроено на ELA-датасет и веса)
INPUT_DIR = 'D:/Diplom/ela_core/datasets/CASIA_Unified_Randomized_v2'
train_dir = f'{INPUT_DIR}/train'
test_dir = f'{INPUT_DIR}/test'
csv_path = os.path.join(INPUT_DIR, "dataset_log_ela.csv")
MODEL_PATH = 'D:/Diplom/ela_core/models/model_resnet_ela_LATEST_Kaggle_5.pth' # Укажи актуальный лучший ELA-чекпоинт

# Создаем директории для артефактов ELA, если их нет
OUTPUT_PLOTS_DIR = 'D:/Diplom/ela_core/artifacts/images/plots/model_resnet_ela_LATEST'
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

print("⚙️ Загрузка датасетов и точное восстановление Source-Aware Val-сплита (ELA)...")
full_train_dataset = ELADataset(folder_path=train_dir, transform=transform)
test_dataset = ELADataset(folder_path=test_dir, transform=transform)

file_to_original = {}
with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        file_to_original[row["filename"]] = row["original_filename"]

unique_originals = sorted({file_to_original[os.path.basename(path)] for path, _ in full_train_dataset.samples})
rng = random.Random(42) 
rng.shuffle(unique_originals)

val_count = round(len(unique_originals) * 0.1)
val_originals = set(unique_originals[:val_count])

val_indices = []
for idx, (path, _) in enumerate(full_train_dataset.samples):
    filename = os.path.basename(path)
    if file_to_original[filename] in val_originals:
        val_indices.append(idx)

val_subset = Subset(full_train_dataset, val_indices)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"✅ Val-сплит восстановлен (Размер: {len(val_indices)}). Test-сплит загружен (Размер: {len(test_dataset)}).")

# 5. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ
print(f"⚙️ Загрузка весов из {MODEL_PATH}...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ==========================================
# ФАЗА 1: ПОИСК ПОРОГА НА VALIDATION
# ==========================================
print("\n🔍 ФАЗА 1: Анализ Validation выборки для поиска оптимального порога...")
val_labels, val_probs = [], []
with torch.no_grad():
    for images, labels, _ in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1] 
        val_labels.extend(labels.numpy())
        val_probs.extend(probs.cpu().numpy())

fpr_v, tpr_v, thresholds_v = roc_curve(val_labels, val_probs)

j_scores = tpr_v - fpr_v
optimal_idx_j = np.argmax(j_scores)
optimal_threshold_j = thresholds_v[optimal_idx_j]

f1_scores_list = [f1_score(val_labels, [1 if p >= t else 0 for p in val_probs]) for t in thresholds_v]
optimal_idx_f1 = np.argmax(f1_scores_list)
optimal_threshold_f1 = thresholds_v[optimal_idx_f1]

BEST_THRESHOLD = optimal_threshold_f1
print(f"  -> Порог по умолчанию: 0.5000")
print(f"  -> Оптимальный порог (Youden's J): {optimal_threshold_j:.4f}")
print(f"  -> Оптимальный порог (F1 Score): {optimal_threshold_f1:.4f}  <-- Выбран для Test")

# ==========================================
# ФАЗА 2: ЭКЗАМЕН НА TEST
# ==========================================
print(f"\n🚀 ФАЗА 2: Экзамен на изолированной Test выборке (ELA)...")
test_labels, test_probs, test_filenames = [], [], []

with torch.no_grad():
    for images, labels, filenames in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        
        test_labels.extend(labels.numpy())
        test_probs.extend(probs.cpu().numpy())
        test_filenames.extend(filenames)

preds_default = [1 if p >= 0.5 else 0 for p in test_probs]
preds_optimal = [1 if p >= BEST_THRESHOLD else 0 for p in test_probs]

print("\n" + "="*50)
print("📊 ОТЧЕТ СО СТАНДАРТНЫМ ПОРОГОМ (0.50):")
print("="*50)
print(classification_report(test_labels, preds_default, target_names=['REAL', 'FAKE']))

print("\n" + "="*50)
print(f"🎯 ОТЧЕТ С ОПТИМАЛЬНЫМ ПОРОГОМ ({BEST_THRESHOLD:.4f}):")
print("="*50)
print(classification_report(test_labels, preds_optimal, target_names=['REAL', 'FAKE']))

# Сохраняем новые, выверенные вероятности
predictions_csv_path = "D:/Diplom/ela_core/artifacts/ela_test_predictions_optimal.csv"
df_preds = pd.DataFrame({
    'filename': test_filenames,
    'true_label': test_labels,
    'prob_fake': test_probs,
    'pred_optimal': preds_optimal
})
df_preds.to_csv(predictions_csv_path, index=False)
print(f"\n📄 Оптимизированные вероятности сохранены: {predictions_csv_path}")

# ==========================================
# ФАЗА 3: ОТРИСОВКА ГРАФИКОВ (ДЛЯ ДИПЛОМА)
# ==========================================
print("\n🎨 Генерация графиков (ROC-AUC и Матрица ошибок)...")

# 1. ROC-AUC График
fpr_t, tpr_t, _ = roc_curve(test_labels, test_probs)
roc_auc_val = auc(fpr_t, tpr_t)

plt.figure(figsize=(8, 6))
plt.plot(fpr_t, tpr_t, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve (ELA Model Test)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'ela_roc_curve.png'), dpi=300)
plt.close()

# 2. Матрица ошибок (С оптимальным порогом)
cm = confusion_matrix(test_labels, preds_optimal)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
plt.ylabel('Реальные классы')
plt.xlabel('Предсказанные классы')
plt.title(f'Матрица ошибок — ELA Model (Порог: {BEST_THRESHOLD:.2f})')
plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'ela_confusion_matrix_optimal.png'), dpi=300)
plt.close()

print(f"✅ Графики сохранены в директорию: '{OUTPUT_PLOTS_DIR}'")