# Отчет по ЛР 01.1: Feature Importance and Selection

Оценивание выполняется по единому rubric: `../RUBRIC_TEMPLATE.md`.


## 1. Контекст
- ФИО автора работы: Мельник Наталья Олеговна
- Группа: 1.2
- Дата выполнения: 2026-06-01
- Используемая среда (OS, версия Python): Windows 11, Python 3.12.7

## 2. Данные и постановка
- Какой таргет предсказывается в медицинском датасете: Прогноз сердечно-сосудистого риска (бинарная классификация: 1 — риск есть, 0 — риска нет).
- Какой таргет предсказывается в финансовом датасете: Прогноз кредитного риска (бинарная классификация: 1 — дефолт, 0 — нет дефолта).
- Какие признаки оказались наиболее интуитивно важными до эксперимента (гипотеза):
  - Для medical: возраст, систолическое давление, холестерин, курение.
  - Для finance: кредитный рейтинг, отношение кредита к доходу, количество просрочек.

## 2.1 Глоссарий незнакомых терминов (обязательно)
- Ссылка на `study-notes/glossary.md`: `study-notes/glossary.md`
- Сколько новых терминов добавлено по ходу всей ЛР: 10
- Минимум 3 примера терминов и почему они были важны для ваших решений:
  1. **VarianceThreshold** — помог отфильтровать признаки с низкой дисперсией, которые не несут информации.
  2. **Mutual Information** — позволил уловить нелинейные зависимости, которые не видит корреляция.
  3. **Overlap** — использовался для оценки устойчивости shortlist при разных параметрах.

## 3. Сравнение методов значимости признаков

| Dataset | Метод | Топ-5 признаков | Краткий комментарий |
|---|---|---|---|
| medical | VarianceThreshold | age, cholesterol, systolic_bp, glucose, bmi | Удалил признаки с низкой дисперсией (почти константные). |
| medical | Correlation | age, cholesterol, systolic_bp, glucose, physical_activity_hours | Выделил признаки с сильной линейной связью с таргетом. |
| medical | Mutual Information | age, cholesterol, systolic_bp, glucose, stress_level | Показал более широкий набор, включая нелинейные связи. |
| medical | ANOVA F-test | age, cholesterol, systolic_bp, glucose, physical_activity_hours | Близок к корреляции, чувствителен к средним значениям. |
| medical | RFE | age, cholesterol, systolic_bp, glucose, physical_activity_hours | Постепенно убрал наименее важные признаки по мнению модели. |
| medical | SFS | age, cholesterol, systolic_bp, glucose, physical_activity_hours | Последовательно добавлял признаки, улучшающие метрику. |
| medical | L1 | age, cholesterol, systolic_bp, glucose, bmi | L1-регуляризация обнулила неважные признаки. |
| medical | RF Importance | age, cholesterol, systolic_bp, glucose, physical_activity_hours | Важности из Random Forest — стабильный топ. |
| medical | Permutation | age, cholesterol, systolic_bp, glucose, physical_activity_hours | Показал, какие признаки сильнее всего влияют на метрику при перемешивании. |
| finance | VarianceThreshold | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Удалил признаки с низкой дисперсией. |
| finance | Correlation | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Сильная линейная связь с таргетом. |
| finance | Mutual Information | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Подтвердил важность финансовых показателей. |
| finance | ANOVA F-test | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Близок к корреляции. |
| finance | RFE | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Стабильный топ. |
| finance | SFS | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Постепенное добавление признаков. |
| finance | L1 | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | L1-регуляризация подтвердила топ. |
| finance | RF Importance | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Важности из Random Forest. |
| finance | Permutation | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Перемешивание подтвердило важность этих признаков. |

### Что изучено по ходу выполнения (обязательно)
- Какие 2-3 идеи/свойства методов вы изучили в процессе выполнения этого раздела: 
  1. Filter-методы (VarianceThreshold, Correlation, Mutual Information, F-test) работают быстро и независимо от модели.
  2. Wrapper-методы (RFE, SFS) учитывают взаимодействие признаков, но требуют больше времени.
  3. Embedded-методы (L1, RF Importance) встраивают отбор в процесс обучения.
- Какие различия между методами вы увидели на своих данных: 
  - Все методы выделили схожие топы признаков, что говорит о стабильности данных.
  - Mutual Information показал чуть более широкий набор, включая `stress_level` для medical.
- Ссылки на источники (URL и/или `study-notes/*.md`): 
  - https://scikit-learn.org/stable/modules/feature_selection.html
  - `study-notes/glossary.md`
- Какие термины из `study-notes/glossary.md` использовали в этом разделе: 
  - VarianceThreshold, Mutual Information, ANOVA F-test, RFE, Permutation Importance.

## 4. Влияние отбора признаков на качество моделей

| Dataset | Feature set | Model | Accuracy | F1 | ROC-AUC | Fit time (sec) |
|---|---|---|---:|---:|---:|---:|
| medical | full | LogisticRegression | 0.672 | 0.468 | 0.762 | 0.016 |
| medical | shortlist (top 15) | LogisticRegression | 0.672 | 0.468 | 0.762 | 0.012 |
| finance | full | LogisticRegression | 0.673 | 0.571 | 0.723 | 0.014 |
| finance | shortlist (top 15) | LogisticRegression | 0.673 | 0.571 | 0.723 | 0.010 |

*Примечание: таблица заполнена на основе базового маршрута. В следующих ноутбуках будет более детальное сравнение.*

### Что изучено по ходу выполнения (обязательно)
- Что вы изучили о влиянии отбора признаков на метрики и время обучения: 
  - Отбор признаков не ухудшил метрики, но сократил время обучения.
  - На этих данных full-признаки и shortlist дали сопоставимые результаты.
- Какие сравнения моделей оказались наиболее показательными: 
  - Сравнение LogisticRegression на full и shortlist показало, что отбор признаков может быть эффективным без потери качества.
- Ссылки на источники (URL и/или `study-notes/*.md`): 
  - https://scikit-learn.org/stable/modules/feature_selection.html
- Какие термины из `study-notes/glossary.md` использовали в этом разделе: 
  - LogisticRegression, shortlist, baseline.

## 5. Интерпретация
- Какие признаки стабильно важны для обоих подходов (filter/wrapper/embedded)?
  - Для medical: age, cholesterol, systolic_bp, glucose, physical_activity_hours.
  - Для finance: loan_to_income, annual_income, credit_score, loan_amount, delinquency_count.
- Где отбор признаков дал прирост метрик, а где ухудшил результат?
  - На данных датасетах отбор признаков не дал значительного прироста, но и не ухудшил качество.
  - Основной выигрыш — в сокращении времени обучения и упрощении модели.
- Как изменилось время обучения после уменьшения числа признаков?
  - Время обучения сократилось на ~20-30% при переходе от full к shortlist.

### Что изучено по ходу выполнения (обязательно)
- Какие методические выводы вы сделали во время анализа, а не только в финале: 
  - Важно проверять устойчивость shortlist к изменению параметров.
  - Не всегда больше признаков = лучше; иногда можно сократить размерность без потери качества.
- Как ваши промежуточные наблюдения изменяли ход эксперимента: 
  - Я изменила `top_n` с 10 на 15, чтобы увидеть, как расширение shortlist влияет на метрики.
- Ссылки на источники (URL и/или `study-notes/*.md`): 
  - https://scikit-learn.org/stable/modules/feature_selection.html
- Какие термины из `study-notes/glossary.md` использовали в этом разделе: 
  - shortlist, baseline, overlap.

## 6. Практическая рекомендация
- Финальный рекомендуемый feature set для medical: 
  `['age', 'cholesterol', 'systolic_bp', 'glucose', 'physical_activity_hours', 'stress_level', 'diastolic_bp', 'resting_heart_rate', 'bmi', 'smoking_status_never', 'sex_female', 'sex_male', 'alcohol_units_weekly', 'smoking_status_former', 'smoking_status_current']` (top 15)
- Финальный рекомендуемый feature set для finance: 
  `['loan_to_income', 'annual_income', 'credit_score', 'loan_amount', 'delinquency_count', 'utilization_ratio', 'previous_default_yes', 'previous_default_no', 'age', 'employment_years', 'savings_balance', 'housing_status_rent', 'open_credit_lines', 'housing_status_mortgage', 'employment_type_salaried']` (top 15)
- Аргументация (метрики + интерпретируемость + скорость): 
  - Эти наборы дают метрики, сопоставимые с full-признаками.
  - Они включают наиболее интерпретируемые и стабильные признаки.
  - Время обучения сокращается, что важно для продакшн-систем.

## 7. Обязательные самостоятельные задания (без образца в solutions)

### 7.0 Методическое изучение по ходу самостоятельных заданий (обязательно)
- Что вы изучили по каждому самостоятельному блоку в процессе выполнения (короткий narrative): 
  - В задании 1 я исследовала устойчивость shortlist к изменению `variance_threshold` и `top_n`.
  - В задании 2 я рассчитала попарные метрики сходства (overlap и Jaccard) между конфигурациями.
  - В задании 3 я сохранила результаты и проанализировала сводку по устойчивости.
- Минимум одно сравнение подходов в каждом подпункте 7.1/7.2/7.3: 
  - Сравнила baseline с другими конфигурациями и увидела, что overlap уменьшается при увеличении `variance_threshold`.
- Ссылки на источники (URL и/или `study-notes/*.md`): 
  - https://scikit-learn.org/stable/modules/feature_selection.html
  - https://en.wikipedia.org/wiki/Jaccard_index
- Какие новые термины вы добавили в `study-notes/glossary.md` в ходе подпунктов 7.1/7.2/7.3: 
  - Overlap, Jaccard Index, Stability Grid, Pairwise Similarity.

### 7.1 Устойчивость filter-ранжирования
- Как менялись shortlist при разных `variance threshold` и `top_n`? 
  - При увеличении `variance_threshold` из shortlist выпадали признаки с низкой дисперсией.
  - При уменьшении `top_n` shortlist становился более строгим, включал только самые важные признаки.
- Какие конфигурации дали наибольший overlap/Jaccard? 
  - Наибольший overlap был при `variance_threshold=0.005` и `top_n=12` (близко к baseline).
- Файл: `outputs/filter_stability_grid.csv`.
- Минимальные колонки: `dataset`, `variance_threshold`, `top_n`, `shortlist_json`, `overlap_with_baseline`.
- Что изучено в этом подпункте (3-5 предложений) + источник(и): 
  - Я изучила, как изменение порога дисперсии и числа признаков влияет на состав shortlist. Оказалось, что увеличение порога удаляет больше признаков, но топ остаётся стабильным. Источник: scikit-learn документация.

### 7.2 Согласованность wrapper/embedded методов
- Этот раздел заполняется после выполнения ноутбуков 02 и 03.
- Какой уровень согласованности между `rfe`, `sfs_forward`, `l1_logreg`, `rf_importance`, `permutation`? 
  - (Будет заполнено после выполнения следующих ноутбуков)
- Какие признаки вошли в `set_D_robust` и почему? 
  - (Будет заполнено после выполнения следующих ноутбуков)
- Файлы: `outputs/method_agreement_long.csv`, `outputs/selection_stability.csv`.
- Что изучено в этом подпункте (3-5 предложений) + источник(и): 
  - (Будет заполнено после выполнения следующих ноутбуков)

### 7.3 Порог, CV и сегментный анализ ошибок
- Этот раздел заполняется после выполнения ноутбука 03.
- Что изменилось после тюнинга порога у лучшей пары `dataset+model`? 
  - (Будет заполнено после выполнения следующих ноутбуков)
- Насколько стабилен финальный feature set по CV? 
  - (Будет заполнено после выполнения следующих ноутбуков)
- В каких сегментах (например, `age`, `credit_score`) ошибок больше всего? 
  - (Будет заполнено после выполнения следующих ноутбуков)
- Файлы: `outputs/threshold_tuning_results.csv`, `outputs/cv_stability_results.csv`, `outputs/error_by_segment.csv`.
- Что изучено в этом подпункте (3-5 предложений) + источник(и): 
  - (Будет заполнено после выполнения следующих ноутбуков)

## 8. Проверка понимания
1. Почему важно делать отбор признаков только на train-части? 
   - Чтобы избежать утечки информации из тестовой выборки в процесс отбора. Если использовать всю выборку, метрики на тесте будут завышены, и модель не будет обобщаться на новые данные.

2. Почему разные методы значимости могут давать разные топы признаков? 
   - Потому что каждый метод использует разную математическую логику: корреляция — линейную, mutual information — любую, F-test — различия средних. Это нормально и полезно — комбинация методов даёт более устойчивый результат.

3. Когда `LinearSVC` может выигрывать у `RandomForest` на отобранных признаках? 
   - Когда данные линейно разделимы, а признаков немного. LinearSVC быстрее обучается и даёт интерпретируемые коэффициенты, в то время как RandomForest может переобучаться на малом числе признаков.

## 9. Что бы вы улучшили в следующей итерации
- Какие эксперименты вы добавили бы (или уже добавили) на расширенном треке: 
  - Добавить кросс-валидацию для выбора оптимального числа признаков.
  - Использовать другие модели (XGBoost, LightGBM) для сравнения.
  - Проверить устойчивость отбора на разных случайных разбиениях данных.
  - Визуализировать матрицу корреляции между признаками для обоих датасетов.


## 10. Скриншоты выполнения

### Скриншот 1. Содержимое папки outputs/
(вставь скриншот, показывающий файлы в папке `01-feature-importance-and-selection/outputs/`)

### Скриншот 2. Проверки пройдены успешно
(вставь скриншот с выводом `Проверки пройдены успешно`)

### Скриншот 3. График топ-8 признаков
(вставь скриншот с графиком)

### Скриншот 4. Таблица feature_ranking
(вставь скриншот с таблицей feature_ranking.head(20))

### Скриншот 5. Результат Задания 1 — filter_stability_grid
(вставь скриншот с таблицей)

### Скриншот 6. Результат Задания 2 — filter_pairwise_similarity
(вставь скриншот с таблицей)