import os, csv, torch, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42); np.random.seed(42); torch.manual_seed(42)

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
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# --- пути (поправьте под себя) ---
ELA_DIR = 'D:/Diplom/ela_core/datasets/CASIA_Unified_Randomized_v2'
train_dir = f'{ELA_DIR}/train'
csv_path = f'{ELA_DIR}/dataset_log_ela.csv'
ELA_MODEL_PATH = 'D:/Diplom/ela_core/models/ELA_model/model_resnet_ela_sourceaware_v2.pth'
OUT_CSV = 'D:/Diplom/ela_core/models/ELA_model/ela_val_predictions.csv'

full_train_dataset = ELADataset(train_dir, transform)

# --- точно та же composite-key логика, что при обучении ELA ---
file_to_source = {}
with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        file_to_source[row["filename"]] = (row["source_dataset"], row["original_filename"])

unique_sources = sorted({file_to_source[os.path.basename(p)] for p,_ in full_train_dataset.samples})
rng = random.Random(42); rng.shuffle(unique_sources)
val_count = round(len(unique_sources) * 0.1)
val_sources = set(unique_sources[:val_count])

val_indices = [idx for idx,(p,_) in enumerate(full_train_dataset.samples)
               if file_to_source[os.path.basename(p)] in val_sources]
val_subset = Subset(full_train_dataset, val_indices)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)

# --- модель ---
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(ELA_MODEL_PATH, map_location=device))
model = model.to(device); model.eval()

val_filenames = [os.path.basename(full_train_dataset.samples[i][0]) for i in val_indices]
all_labels, all_probs = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

pd.DataFrame({
    'filename': val_filenames,
    'true_label': all_labels,
    'prob_fake': all_probs
}).to_csv(OUT_CSV, index=False)
print(f"Сохранено: {OUT_CSV}, строк: {len(val_filenames)}")