import pandas as pd
from scipy.stats import binomtest

# 1. Загружаем твои предсказания (стандартный порог)
df = pd.read_csv('D:/Diplom/ela_core/artifacts/rgb_test_predictions.csv')

# 2. Вытаскиваем название датасета из имени файла
df['source_dataset'] = df['filename'].apply(lambda x: 'CASIA1' if 'CASIA1' in x else 'CASIA2')
df['correct'] = df['true_label'] == df['pred_label']

# 3. Фильтруем только CASIA1 REAL
c1_real = df[(df['source_dataset'] == 'CASIA1') & (df['true_label'] == 0)]

n = len(c1_real)
k = c1_real['correct'].sum()

# 4. Считаем доверительный интервал
ci = binomtest(k, n, 0.5).proportion_ci()

print(f"CASIA1 REAL acc: {k/n:.3f}, 95% CI: [{ci.low:.3f}, {ci.high:.3f}], n={n}")
print("\nРаспределение prob_fake в зависимости от правильности ответа (correct):")
print(c1_real.groupby('correct')['prob_fake'].describe())