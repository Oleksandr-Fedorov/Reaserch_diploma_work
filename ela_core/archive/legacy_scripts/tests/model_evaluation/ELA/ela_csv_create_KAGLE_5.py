import pandas as pd

# 1. Загружаем свежие предсказания ELA
preds = pd.read_csv('D:/Diplom/ela_core/artifacts/ela_test_predictions_optimal.csv')

# 2. Загружаем лог датасета, чтобы точно знать, из CASIA1 файл или из CASIA2
# (укажи правильный путь к твоему логу)
log = pd.read_csv('D:/Diplom/ela_core/datasets/CASIA_Unified_Randomized_v2/dataset_log_ela.csv')

# Оставляем только нужные колонки для сопоставления
log = log[['filename', 'source_dataset']].drop_duplicates()

# 3. Объединяем данные
merged = preds.merge(log, on='filename', how='left')
merged['correct'] = merged['true_label'] == merged['pred_optimal']

# 4. Считаем финальную статистику
summary = merged.groupby(['source_dataset', 'true_label']).agg(
    total=('filename', 'count'),
    correct=('correct', 'sum')
)
summary['accuracy (%)'] = (summary['correct'] / summary['total'] * 100).round(2)

print("\n" + "="*50)
print("📊 ТОЧНОСТЬ ELA-МОДЕЛИ ПО ДОМЕНАМ (CASIA1 vs CASIA2)")
print("="*50)
print(summary)
print("="*50)