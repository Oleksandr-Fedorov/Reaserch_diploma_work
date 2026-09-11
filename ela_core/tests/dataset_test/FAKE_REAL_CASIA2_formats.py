import os
from collections import Counter

# Поставь свои пути к локальным папкам CASIA2
CASIA2_REAL = "D:/Diplom/ela_core/datasets/Some_data_sets/CASIA2/Au"
CASIA2_FAKE = "D:/Diplom/ela_core/datasets/Some_data_sets/CASIA2/Tp"

def count_formats(folder):
    exts = Counter()
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        exts[ext] += 1
    return exts

print("CASIA2 Оригиналы (REAL):", count_formats(CASIA2_REAL))
print("CASIA2 Подделки (FAKE):", count_formats(CASIA2_FAKE))