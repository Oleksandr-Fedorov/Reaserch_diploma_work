"""
LATE FEATURE FUSION (RGB + ELA) - CORRECTED VERSION
---------------------------------
Архитектура: 
1. Frozen RGB ResNet18 -> 512-d
2. Frozen ELA ResNet18 -> 512-d
3. Concatenation -> 1024-d
4. Trainable MLP Head (No BatchNorm) -> 2 classes
"""

import os
import io
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
from pathlib import Path

# ==========================================
# 1. ФИКСАЦИЯ СИДОВ (Для строгой научности)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print(f"🚀 Запуск Late Feature Fusion на {device}")

# ==========================================
# 2. НАСТРОЙКИ ПУТЕЙ
# ==========================================
ROOT_DIR = Path("D:/Diplom/ela_core")

DATASET_DIR = ROOT_DIR / "datasets" / "CASIA_Unified_Randomized_RGB_v1"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
CSV_LOG_PATH = DATASET_DIR / "dataset_log.csv"

RGB_MODEL_PATH = ROOT_DIR / "models" / "model_resnet_rgb_LATEST.pth" 
ELA_MODEL_PATH = ROOT_DIR / "models" / "model_resnet_ela_LATEST_Kaggle_5.pth" 

FUSION_OUT_DIR = ROOT_DIR / "artifacts" / "fusion"
FUSION_OUT_DIR.mkdir(parents=True, exist_ok=True)
FUSION_MODEL_SAVE = ROOT_DIR / "models" / "model_fusion_LATEST.pth"

# ==========================================
# 3. ФИЗИКА ELA И ДАТАСЕТ
# ==========================================
def get_ela_image(img: Image.Image, quality: int = 90):
    """Генерирует ELA на лету. Синхронизация 100%."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")
    ela = np.abs(np.array(img, dtype=float) - np.array(compressed, dtype=float)) * 10
    return Image.fromarray(np.clip(ela, 0, 255).astype(np.uint8))

class DualStreamDataset(Dataset):
    def __init__(self, folder_path, transform=None, ela_quality=90):
        self.transform = transform
        self.ela_quality = ela_quality
        self.samples = []
        for label_name, label_idx in [("REAL", 0), ("FAKE", 1)]:
            class_dir = os.path.join(folder_path, label_name)
            if not os.path.exists(class_dir): continue
            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(class_dir, filename), label_idx))
                    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_raw = Image.open(img_path).convert("RGB")
        
        img_ela = get_ela_image(img_raw, self.ela_quality)
        
        if self.transform:
            tensor_rgb = self.transform(img_raw)
            tensor_ela = self.transform(img_ela)
            
        return tensor_rgb, tensor_ela, label, os.path.basename(img_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 4. АРХИТЕКТУРА LATE FUSION
# ==========================================
class LateFusionModel(nn.Module):
    def __init__(self, rgb_weights, ela_weights):
        super(LateFusionModel, self).__init__()
        
        # RGB ветка
        self.rgb_branch = models.resnet18(weights=None)
        self.rgb_branch.fc = nn.Linear(self.rgb_branch.fc.in_features, 2)
        self.rgb_branch.load_state_dict(torch.load(rgb_weights, map_location=device))
        self.rgb_branch.fc = nn.Identity() # Выход: 512
        
        # ELA ветка
        self.ela_branch = models.resnet18(weights=None)
        self.ela_branch.fc = nn.Linear(self.ela_branch.fc.in_features, 2)
        self.ela_branch.load_state_dict(torch.load(ela_weights, map_location=device))
        self.ela_branch.fc = nn.Identity() # Выход: 512
        
        # ГЛУХАЯ ЗАМОРОЗКА БАЗОВЫХ СЕТЕЙ
        for param in self.rgb_branch.parameters(): param.requires_grad = False
        for param in self.ela_branch.parameters(): param.requires_grad = False
            
        # ОБУЧАЕМАЯ ГОЛОВА (Упрощена, без BatchNorm)
        self.mlp_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x_rgb, x_ela):
        with torch.no_grad():
            emb_rgb = self.rgb_branch(x_rgb)
            emb_ela = self.ela_branch(x_ela)
        
        fused_features = torch.cat((emb_rgb, emb_ela), dim=1)
        return self.mlp_head(fused_features)

# ==========================================
# 5. СТРОГИЙ SOURCE-AWARE СПЛИТ
# ==========================================
print("📁 Чтение логов и сборка Source-Aware датасетов...")
full_train_dataset = DualStreamDataset(folder_path=TRAIN_DIR, transform=transform)
test_dataset = DualStreamDataset(folder_path=TEST_DIR, transform=transform)

file_to_original = {}
with open(CSV_LOG_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        file_to_original[row["filename"]] = row["original_filename"]

unique_originals = sorted({file_to_original[os.path.basename(path)] for path, _ in full_train_dataset.samples})
rng = random.Random(42)
rng.shuffle(unique_originals)
    
val_count = round(len(unique_originals) * 0.1)
val_originals = set(unique_originals[:val_count])

train_indices, val_indices = [], []
# ИСПРАВЛЕНИЕ РАСПАКОВКИ: (path, _)
for idx, (path, _) in enumerate(full_train_dataset.samples):
    filename = os.path.basename(path)
    if file_to_original[filename] in val_originals:
        val_indices.append(idx)
    else:
        train_indices.append(idx)

train_subset = Subset(full_train_dataset, train_indices)
val_subset = Subset(full_train_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"✅ Split завершен. Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_dataset)}")

# ==========================================
# 6. ИНИЦИАЛИЗАЦИЯ И ОБУЧЕНИЕ
# ==========================================
print("\n⚙️ Инициализация Late Feature Fusion Модели...")
model = LateFusionModel(RGB_MODEL_PATH, ELA_MODEL_PATH).to(device)

# ИСПРАВЛЕНИЕ LR: Уменьшен до 3e-4
optimizer = optim.AdamW(model.mlp_head.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
criterion = nn.CrossEntropyLoss()

epochs = 15
best_val_loss = float("inf")
patience, no_improve = 4, 0

print("\n🚀 СТАРТ ОБУЧЕНИЯ (Обучается только MLP, бэкенды заморожены)...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for rgb_imgs, ela_imgs, labels, _ in train_loader:
        rgb_imgs, ela_imgs, labels = rgb_imgs.to(device), ela_imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(rgb_imgs, ela_imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    # Валидация
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for rgb_imgs, ela_imgs, labels, _ in val_loader:
            rgb_imgs, ela_imgs, labels = rgb_imgs.to(device), ela_imgs.to(device), labels.to(device)
            outputs = model(rgb_imgs, ela_imgs)
            val_loss += criterion(outputs, labels).item()
            
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    
    print(f"Эпоха {epoch+1:02d}/{epochs} | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), FUSION_MODEL_SAVE)
        print("  [+] Сохранена лучшая модель MLP-фьюжена")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"🛑 Early stopping на эпохе {epoch+1}")
            break

# ==========================================
# 7. ПОИСК ПОРОГА НА VALIDATION (ИСПРАВЛЕНО)
# ==========================================
print("\n🔍 ПОИСК ПОРОГА НА VALIDATION ВЫБОРКЕ...")
model.load_state_dict(torch.load(FUSION_MODEL_SAVE))
model.eval()

val_labels_roc, val_probs_roc = [], []
with torch.no_grad():
    for rgb_imgs, ela_imgs, labels, _ in val_loader:
        rgb_imgs, ela_imgs = rgb_imgs.to(device), ela_imgs.to(device)
        outputs = model(rgb_imgs, ela_imgs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        
        val_labels_roc.extend(labels.numpy())
        val_probs_roc.extend(probs.cpu().numpy())

fpr_v, tpr_v, thresholds_v = roc_curve(val_labels_roc, val_probs_roc)
f1_scores_v = [f1_score(val_labels_roc, [1 if p >= t else 0 for p in val_probs_roc]) for t in thresholds_v]
best_thresh = thresholds_v[np.argmax(f1_scores_v)]
print(f"✅ Оптимальный порог (F1) найден: {best_thresh:.4f}")

# ==========================================
# 8. ЧЕСТНЫЙ ЭКЗАМЕН И ЛОГИРОВАНИЕ
# ==========================================
print("\n🎯 ЭКЗАМЕН FUSION-МОДЕЛИ НА ИЗОЛИРОВАННОМ ТЕСТЕ...")

test_labels, test_probs, test_filenames = [], [], []

with torch.no_grad():
    for rgb_imgs, ela_imgs, labels, filenames in test_loader:
        rgb_imgs, ela_imgs = rgb_imgs.to(device), ela_imgs.to(device)
        outputs = model(rgb_imgs, ela_imgs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        
        test_labels.extend(labels.numpy())
        test_probs.extend(probs.cpu().numpy())
        test_filenames.extend(filenames)

# Применяем порог, найденный на Validation
test_preds = [1 if p >= best_thresh else 0 for p in test_probs]

print("\n" + "="*50)
print(f"📊 ФИНАЛЬНЫЙ ОТЧЕТ: FUSION (Порог: {best_thresh:.4f})")
print("="*50)
print(classification_report(test_labels, test_preds, target_names=['REAL', 'FAKE']))

# Экспорт предиктов
predictions_csv_path = FUSION_OUT_DIR / "fusion_test_predictions.csv"
df_preds = pd.DataFrame({
    'filename': test_filenames,
    'true_label': test_labels,
    'prob_fake': test_probs,
    'pred_label': test_preds
})
df_preds.to_csv(predictions_csv_path, index=False)
print(f"📄 Предикты сохранены: {predictions_csv_path}")

# ==========================================
# 9. ВИЗУАЛИЗАЦИЯ (ROC & Confusion Matrix)
# ==========================================
fpr_t, tpr_t, _ = roc_curve(test_labels, test_probs)
roc_auc_val = auc(fpr_t, tpr_t)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Late Feature Fusion Evaluation", fontsize=16, fontweight='bold')

# Матрица ошибок
cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'], ax=ax1)
ax1.set_ylabel('Реальные классы')
ax1.set_xlabel('Предсказанные классы')
ax1.set_title(f'Матрица ошибок (Порог: {best_thresh:.2f})')

# ROC-кривая
ax2.plot(fpr_t, tpr_t, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.3f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC-AUC Curve')
ax2.legend(loc="lower right")
ax2.grid(alpha=0.3)

plot_path = FUSION_OUT_DIR / "thesis_fusion_eval.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
print(f"✅ Графики сохранены: {plot_path}")