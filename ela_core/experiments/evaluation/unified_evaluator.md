# Unified evaluator

Единая точка входа: `experiments/evaluation/unified_evaluator.py`.
Результаты четырёх вариантов считаются на одной выборке, после строгого выравнивания
по `filename` и сверки `true_label`. Дубликаты, неполные пары, несовпадение labels,
NaN/Inf, некорректные probabilities и несовместимые checkpoints вызывают ошибку.

## Быстрый запуск из D:\Diplom

```powershell
# Только сохранённые predictions; без PyTorch inference и чтения datasets.
& .\ela_core\.venv\Scripts\python.exe -B -X utf8 .\ela_core\experiments\evaluation\unified_evaluator.py cached

# Дополнительно проверить реальные checkpoints по сохранённым embeddings.
# Запускаются только классификационные heads на CPU, не image backbones.
& .\ela_core\.venv\Scripts\python.exe -B -X utf8 .\ela_core\experiments\evaluation\unified_evaluator.py cached --verify-checkpoints

# Новая calibration Decision Fusion только на общем held-out подмножестве.
& .\ela_core\.venv\Scripts\python.exe -B -X utf8 .\ela_core\experiments\evaluation\unified_evaluator.py cached --decision-policy common-validation
```

Без `--output-dir` результат — JSON в stdout. С `--output-dir <NEW_DIRECTORY>`
создаются `metrics.json` и единый long-format `predictions.csv` (model, filename,
true_label, prob_fake, pred_label). Существующий каталог запрещён, перезаписи нет.
`--root` позволяет явно указать другой корень `ela_core`; default определяется
от расположения evaluator, независимо от cwd. Зависимости используются из
существующего окружения; никаких downloads/pretrained weights нет.

## Реальные модели и входы

| Вариант | Checkpoint относительно ela_core | Архитектура / правило |
|---|---|---|
| RGB | `models/RGB_model/model_resnet_rgb_LATEST.pth` | ResNet18, fc 512→2 |
| ELA | `models/ELA_model/model_resnet_ela_sourceaware_v2.pth` | ResNet18, fc 512→2; не исторический Kaggle-5 |
| Decision Fusion | два checkpoints выше | `alpha_rgb*p_rgb + (1-alpha_rgb)*p_ela` |
| Feature Fusion V2 | `models/Fusion_v2/model_fusion_sourceaware_v2.pth` | frozen ResNet18 branches → [RGB512, ELA512] → Linear(1024,256), ReLU, Dropout(.5), Linear(256,2) |

Все state_dict загружаются через `weights_only=True`, CPU, `strict=True`.
Нет fallback к unrestricted pickle, удаления префиксов ключей или частичной загрузки.
Наличие finite tensors проверяется. Frozen branches Fusion V2 сравниваются с
соответствующими baseline tensors включая BatchNorm buffers; fc baseline не входит
в branch. Для inference обязательны `eval()` и `inference_mode()`.

Порядок классов из notebook: **REAL=0, FAKE=1**, `prob_fake=softmax(logits)[:,1]`.
Нельзя использовать алфавитный порядок ImageFolder (FAKE/REAL) или sigmoid от одного
logit. Threshold comparison: `>=`. Исторический argmax выбирает REAL при точной
ничьей .5; это различие фиксируется отдельно, если встретится.

Подготовка входов повторяет notebook cells 13, 20, 23 (нумерация с нуля):
PIL `.convert('RGB')`, bilinear `Resize((224,224))`, `ToTensor`, ImageNet mean
`[.485,.456,.406]`, std `[.229,.224,.225]`. Валидация не добавляет аугментаций.

RGB — подготовленные PNG после JPEG `q_primary`; ELA — парные offline PNG после
`abs(simulated_rgb - JPEG(q_ela))*10`, clip [0,255], uint8. Генератор использовал
четыре реализации на источник, q_primary из 50..100 с шагом 5 и q_ela из
65..100 с шагом 5. RGB generator использует тот же журнал и имена.
Evaluator **не строит ELA заново**, не усиливает готовые ELA maps и не заменяет их
online ELA или server preprocessing. Сам факт наличия двух PNG не доказывает
соблюдение параметров генерации — за это отвечает provenance входного manifest.

## Сохранённые результаты и проверка связи с weights

Используются `models/RGB_model/rgb_test_predictions.csv`,
`models/ELA_model/ela_sourceaware_test_predictions.csv`, соответствующие val CSV,
`artifacts/fusion_v2/fusion_v2_test_metadata.csv` и retained split manifest ELA.
Старые optimized CSV из других экспериментов не подставляются автоматически.

`--verify-checkpoints` использует четыре NPY из Fusion V2 bundle: проверяет
размерность/dtype/finite values, порядок строк и точную конкатенацию RGB+ELA.
Затем прогоняет baseline fc и Fusion MLP по готовым embeddings, сравнивая с
сохранёнными probabilities/logits. Это связывает текущие weights с сохранёнными
результатами при условии подлинности экспортированных embeddings, без повторного
прохода по 11 432 изображениям. Предел absolute error для probability — 2e-5,
для logits — 2e-4 (CPU/GPU float32). Исходные pixels этим способом не проверяются.
Report записывает SHA-256 источников, checkpoint-файлов при проверке и notebook.

## Threshold и ограничение split provenance

- RGB и ELA: исходный baseline threshold .5.
- Decision Fusion по умолчанию: заранее фиксированные alpha=.5, threshold=.5.
  Это equal-weight baseline, не ранее оптимизированный результат `comprasion.py`.
- `--decision-policy common-validation`: та же двухступенчатая процедура, что в
  `comprasion.py` (101 alpha при .5, затем ROC thresholds по macro-F1), но только
  на общем held-out подмножестве обоих baseline. Test не участвует в выборе.
  Это отдельная новая calibration; параметры и provenance сохраняются в report.
- Feature Fusion V2: .2893 из сохранённого validation output. Полная точность
  порога и validation probabilities не сохранены. Проверяется согласие с
  `pred_label` test CSV; порог не восстанавливается оптимизацией по test.

**Найденное противоречие:** notebook cell 20 формирует RGB validation по
`original_filename`; cells 13 и 23 формируют ELA/Fusion validation по
`(source_dataset, original_filename)`. Разный порядок перед shuffle(seed=42)
даёт разные splits даже без коллизий имён. Локальный `rgb_val.py` использует
уже composite split, поэтому его название не доказывает held-out для RGB.

Из retained ELA manifest восстанавливаются 11 423 источника. Среди 1 142
composite validation sources 487 (1 948 строк) относятся к RGB validation,
655 (2 620 строк) — к RGB train по сохранённому training code. Это **INFERENCE**,
а не встроенные metadata weights: raw state_dict не хранит split/seed/run ID.
Evaluator проверяет согласие composite split с manifest, coverage val CSV,
четыре augmentation на источник и отсутствие test sources в train/val manifest.

Нельзя называть настройку на всех 4 568 строках независимой от обучения RGB.
Это ограничение касается также выбора head/threshold Fusion V2: composite split
самого head не устраняет exposure frozen RGB backbone. Оно не отменяет факт
завершённого held-out test evaluation Fusion V2. Common-validation calibration
снижает конкретное пересечение для Decision Fusion, но сама основана на
восстановленной, а не записанной в checkpoint provenance. Не считать её новой
внешней, никогда прежде не использовавшейся validation-выборкой.

В validation CSV нет checkpoint hash и сохранённых embeddings. Их принадлежность
моделям поддерживается scripts и согласованием имён/labels, но независимо повторить
checkpoint-to-val-prediction проверку без image inference нельзя. Это ограничение
явно сохраняется в JSON, а не скрывается при выборе common-validation.

## Результат проверки 2026-09-29

Реальные checkpoints прошли strict loading. Все RGB/ELA backbone tensors внутри
Fusion V2 совпали с baseline. Head replay по сохранённым embeddings дал максимальную
absolute probability error: RGB `2.0622e-6`, ELA `1.0384e-6`, Fusion V2 `2.0347e-7`.
Это CPU/GPU float32 расхождения в пределах принятого допуска. 11 432 test filename
и labels согласованы между тремя CSV; сохранённые pred_label совпали с thresholds.

| Метод / политика | Accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|
| RGB, threshold .5 | 0.679059 | 0.666066 | 0.780047 |
| ELA Source-Aware V2, threshold .5 | 0.694542 | 0.680896 | 0.755913 |
| Decision Fusion fixed alpha=.5, threshold=.5 | 0.704689 | 0.691867 | 0.798626 |
| Feature Fusion V2, threshold .2893 | 0.737841 | 0.737674 | 0.802217 |
| Decision Fusion common-validation, alpha=.34, threshold=.2604767778 | 0.680546 | 0.676854 | 0.798159 |

Последняя строка — отдельная validation calibration с описанной выше условной
provenance. Она ухудшила test accuracy/macro-F1 относительно фиксированного baseline;
параметры не менялись после просмотра test. Фиксированная политика была выбрана
default до этого сравнения. Таблица не доказывает статистическую значимость различий.

## Явный новый inference

Он подготовлен, но при этой работе не запускался на project images/datasets.
CSV manifest должен содержать `filename,true_label,rgb_path,ela_path`. Пути к PNG
могут быть абсолютными или относительно manifest; basename обоих должен равняться
filename. Никакого обхода dataset directories или автоматического выбора split нет.
Class folder REAL/FAKE, если присутствует, дополнительно сверяется с true_label.

```powershell
& .\ela_core\.venv\Scripts\python.exe -B -X utf8 .\ela_core\experiments\evaluation\unified_evaluator.py infer --manifest <PAIRS.csv> --alpha-rgb 0.5 --decision-threshold 0.5 --output-dir <NEW_DIRECTORY>
```

Этот режим использует CPU и вычисляет каждый общий backbone один раз на batch.
Параметры Decision Fusion задаются заранее; обучение/поиск threshold на manifest
не выполняется. Для common-validation можно перенести alpha/threshold из cached
report явно, записав его как источник calibration. Несовпадение checkpoint branches
останавливает inference. Cross-dataset/raw-image protocol здесь не реализован.

## Проверки реализации

```powershell
& .\ela_core\.venv\Scripts\python.exe -B -X utf8 .\ela_core\experiments\evaluation\test_unified_evaluator.py
```

Regression checks используют синтетические tables и image, проверяют alignment,
ошибочные labels/probabilities, thresholds, исключение contaminated rows из
calibration и эквивалентность preprocessing. Старые scripts/notebook не запускаются.
