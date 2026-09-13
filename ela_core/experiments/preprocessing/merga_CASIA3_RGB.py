import os
import io
import csv
from PIL import Image
from pathlib import Path
from collections import defaultdict

Image.MAX_IMAGE_PIXELS = 100_000_000

# --- PATHS ---
CASIA1_REAL = "D:/Diplom/ela_core/datasets/CASIA1/Au"
CASIA1_FAKE = "D:/Diplom/ela_core/datasets/CASIA1/Sp"
CASIA2_REAL = "D:/Diplom/ela_core/datasets/Some_data_sets/CASIA2/Au"
CASIA2_FAKE = "D:/Diplom/ela_core/datasets/Some_data_sets/CASIA2/Tp"

ELA_CSV_LOG_PATH = "D:/Diplom/ela_core/datasets/CASIA_Unified_Randomized_v2/dataset_log_ela.csv"
# New folder name on purpose — keeps the broken v1 around for a before/after diff, avoids overwrite confusion
OUTPUT_DIR = "D:/Diplom/ela_core/datasets/CASIA_Unified_Randomized_RGB_v2"

SOURCE_FOLDERS = {
    ("CASIA1", "REAL"): CASIA1_REAL,
    ("CASIA1", "FAKE"): CASIA1_FAKE,
    ("CASIA2", "REAL"): CASIA2_REAL,
    ("CASIA2", "FAKE"): CASIA2_FAKE,
}

def create_dirs():
    for split in ['train', 'test']:
        for cls in ['REAL', 'FAKE']:
            Path(f"{OUTPUT_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

def load_ela_log(csv_path):
    """The ELA log is now the single source of truth — no re-splitting, no re-sampling."""
    with open(csv_path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def build_dataset():
    create_dirs()
    print("🛠 Reading ELA log as the single source of truth for RGB generation...")
    rows = load_ela_log(ELA_CSV_LOG_PATH)
    print(f"   {len(rows)} augmented instances found in ELA log.")

    # Group rows by original image so each source file is decoded once, not 4x
    by_original = defaultdict(list)
    for row in rows:
        by_original[(row['source_dataset'], row['class'], row['original_filename'])].append(row)

    csv_path = os.path.join(OUTPUT_DIR, "dataset_log.csv")
    total_originals = 0
    total_generated = 0
    missing_originals = 0

    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "source_dataset", "original_filename", "split", "class", "q_primary"])

        for (source_dataset, cls, original_filename), instance_rows in by_original.items():
            folder = SOURCE_FOLDERS[(source_dataset, cls)]
            src_path = os.path.join(folder, original_filename)

            if not os.path.exists(src_path):
                print(f"⚠️ Missing source file, skipping: {src_path}")
                missing_originals += 1
                continue

            try:
                img = Image.open(src_path).convert("RGB")
            except Exception as e:
                print(f"⚠️ Failed to open {src_path}: {e}")
                missing_originals += 1
                continue

            for row in instance_rows:
                q_primary = int(row['q_primary'])
                filename = row['filename']     # identical to ELA's filename -> guarantees pairing
                split = row['split']

                sim_buffer = io.BytesIO()
                img.save(sim_buffer, format="JPEG", quality=q_primary)
                sim_buffer.seek(0)
                simulated_img = Image.open(sim_buffer).convert("RGB")

                output_path = os.path.join(OUTPUT_DIR, split, cls, filename)
                simulated_img.save(output_path, format="PNG")

                writer.writerow([filename, source_dataset, original_filename, split, cls, q_primary])
                total_generated += 1

            total_originals += 1
            if total_originals % 500 == 0:
                print(f"  Processed {total_originals} source images...")

    assert total_generated == len(rows), \
        f"Mismatch: generated {total_generated} PNGs but ELA log has {len(rows)} rows — investigate before training on this."

    print("\n" + "="*50)
    print("✅ RGB dataset generated, row-for-row matched to the ELA log.")
    print(f"Source images processed:   {total_originals}")
    print(f"PNG files generated:       {total_generated}")
    print(f"Missing/skipped originals: {missing_originals}")
    print(f"Log saved to: {csv_path}")
    print("="*50)

if __name__ == "__main__":
    build_dataset()