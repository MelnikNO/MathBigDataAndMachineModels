# ЛР 02.1: Глобальная интерпретация моделей

Оценивание выполняется по единому rubric: `../RUBRIC_TEMPLATE.md`.


## 1. Контекст
- ФИО автора работы: Мельник Наталья Олеговна
- Группа: 1.2
- Дата выполнения: 2026-06-01
- Используемая среда (OS, версия Python): Windows 11, Python 3.12.7


## 2. Входные данные и предпосылки
- **Какой feature set из ЛР 01 выбран для `medical`:**
  `set_A_wrapper` — 10 признаков (num__age, num__cholesterol, num__systolic_bp, num__physical_activity_hours, num__glucose, cat__smoking_status_never, num__bmi, num__stress_level, num__resting_heart_rate, cat__smoking_status_former)
- **Какой feature set из ЛР 01 выбран для `finance`:**
  `set_B_tree` — 10 признаков (num__loan_to_income, num__annual_income, num__loan_amount, num__credit_score, num__utilization_ratio, cat__housing_status_mortgage, num__savings_balance, cat__previous_default_no, num__delinquency_count, num__age)
- **Почему для интерпретации вы выбрали именно эти наборы:**
  - Это лучшие неполные наборы признаков из ЛР 01, показавшие высокие метрики (roc_auc ~0.76 для medical, ~0.716 для finance).
  - Они позволяют интерпретировать модели без дублирования baseline `full`, что даёт более компактные и понятные объяснения.


## 2.1 Глоссарий незнакомых терминов (обязательно)
- Ссылка на `study-notes/glossary.md`: `study-notes/glossary.md`
- Сколько новых терминов добавлено по ходу всей ЛР: 6
- Минимум 3 примера терминов и почему они были важны:
  1. **Permutation Importance** — позволила проверить, насколько native importance (коэффициенты LR и feature_importance RF) согласуются с реальным влиянием признаков на метрику.
  2. **Partial Dependence** — показала направление и форму влияния каждого признака на предсказания модели (монотонный или немонотонный тренд).
  3. **Native Importance** — дала быстрое и интерпретируемое ранжирование признаков, доступное прямо из модели.


## 3. Глобальная интерпретация моделей

| Dataset | Model | Method | Top-5 признаков | Краткий комментарий |
|---|---|---|---|---|
| medical | LogisticRegression | Coef abs | num__age, num__cholesterol, num__systolic_bp, cat__smoking_status_never, cat__smoking_status_former | Коэффициенты показывают, что age, cholesterol и systolic_bp имеют наибольший вклад в предсказание риска. Статус курения также важен. |
| medical | LogisticRegression | Permutation | num__age, num__cholesterol, num__systolic_bp, num__bmi, num__stress_level | Подтверждает важность age, cholesterol, systolic_bp. BMI и stress_level также значимы. |
| medical | RandomForest | Feature importance | num__age, num__cholesterol, num__systolic_bp, num__physical_activity_hours, num__bmi | Деревья выделяют те же признаки, добавляя physical_activity_hours как важный фактор снижения риска. |
| medical | RandomForest | Permutation | num__systolic_bp, num__age, num__cholesterol, num__physical_activity_hours, num__glucose | Подтверждает топ-3 и показывает, что glucose важен для RF. |
| finance | LogisticRegression | Coef abs | cat__previous_default_no, num__loan_to_income, num__credit_score, num__annual_income, cat__housing_status_mortgage | Наибольший вклад — loan_to_income, credit_score и annual_income. Категориальные признаки также значимы. |
| finance | LogisticRegression | Permutation | num__delinquency_count, num__credit_score, cat__previous_default_no, num__loan_to_income, num__annual_income | Подтверждает важность credit_score, loan_to_income, annual_income. Delinquency_count также важен. |
| finance | RandomForest | Feature importance | num__loan_to_income, num__annual_income, num__credit_score, num__loan_amount, num__utilization_ratio | Те же признаки в топе, добавляется loan_amount и utilization_ratio. |
| finance | RandomForest | Permutation | num__annual_income, num__loan_to_income, num__utilization_ratio, num__delinquency_count, cat__previous_default_no | Подтверждает топ-3, показывает важность utilization_ratio и delinquency_count. |

### Что изучено по ходу выполнения (обязательно)
- **Какие 2-3 идеи о глобальной интерпретации вы изучили:**
  1. Native importance (коэффициенты LR и feature_importance RF) и permutation importance дают схожие результаты на этих данных — это говорит об отсутствии сильных смещений.
  2. Коэффициенты LR показывают направление влияния (положительное/отрицательное), а permutation importance — только силу.
  3. RandomForest feature_importance может быть смещена в пользу признаков с большим числом уникальных значений, но на этих данных смещение не проявляется.
- **Где объяснения двух моделей совпадают, а где расходятся:**
  - **Совпадают:** топ-3 признака для medical (age, cholesterol, systolic_bp) и для finance (loan_to_income, annual_income, credit_score) стабильны для всех методов.
  - **Расходятся:** для medical LogisticRegression выделяет `smoking_status` (коэффициенты), а RandomForest — `physical_activity_hours` (важность). Это связано с разной природой моделей: LR учитывает линейные вклады, RF — нелинейные взаимодействия.
- **Ссылки на источники:**
  - https://scikit-learn.org/stable/modules/permutation_importance.html
  - https://scikit-learn.org/stable/modules/partial_dependence.html
- **Какие термины из `study-notes/glossary.md` использовали:**
  - Permutation Importance, Partial Dependence, Native Importance, LogisticRegression, RandomForest.


## 4. Partial Dependence и устойчивость интерпретации

| Dataset | Model | Raw feature | Trend | Score delta | Краткая интерпретация |
|---|---|---|---|---:|---|
| medical | LogisticRegression | age | non_decreasing | 0.429 | С возрастом риск сердечно-сосудистых заболеваний монотонно растёт. |
| medical | LogisticRegression | cholesterol | non_decreasing | 0.302 | Рост риска с увеличением холестерина, но после определённого уровня наступает плато. |
| medical | LogisticRegression | systolic_bp | non_decreasing | 0.269 | Повышение систолического давления увеличивает риск. |
| medical | RandomForest | age | non_monotonic | 0.264 | RF показывает более «рваную» кривую, но общий тренд — рост риска с возрастом. |
| medical | RandomForest | systolic_bp | non_monotonic | 0.205 | RF улавливает нелинейные эффекты давления на риск. |
| finance | LogisticRegression | loan_to_income | non_decreasing | 0.187 | Рост долговой нагрузки монотонно увеличивает риск дефолта. |
| finance | LogisticRegression | annual_income | non_increasing | 0.174 | Высокий доход снижает риск, но на очень высоких доходах эффект насыщается. |
| finance | RandomForest | annual_income | non_monotonic | 0.192 | RF показывает более сложную зависимость дохода от риска. |
| finance | RandomForest | loan_to_income | non_monotonic | 0.189 | RF улавливает нелинейные эффекты долговой нагрузки. |

### Что изучено по ходу выполнения (обязательно)
- **Что вы изучили о связи между raw-признаком и mean score:**
  - Для медицинских данных: age, cholesterol, systolic_bp — монотонно увеличивают риск. Physical_activity_hours — монотонно снижает риск.
  - Для финансовых данных: loan_to_income — монотонно увеличивает риск, credit_score и annual_income — монотонно снижают риск.
  - RandomForest часто показывает немонотонные тренды из-за своей природы, но общее направление сохраняется.
- **Где partial dependence помогает, а где может ввести в заблуждение:**
  - **Помогает:** показывает направление и форму влияния признака, позволяет проверить, согласуется ли модель с предметной логикой.
  - **Может ввести в заблуждение:** не учитывает взаимодействия между признаками; для RandomForest кривые могут быть «рваными» из-за переобучения на малых выборках; требует осторожной интерпретации на краях распределения.
- **Ссылки на источники:**
  - https://scikit-learn.org/stable/modules/partial_dependence.html
- **Какие термины из `study-notes/glossary.md` использовали:**
  - Partial Dependence, LogisticRegression, RandomForest.


## 5. Практическая рекомендация
- **Какая модель лучше объяснима для `medical` и почему:**
  - **LogisticRegression** лучше объяснима для medical, так как её коэффициенты показывают направление влияния каждого признака (например, возраст увеличивает риск). Это важно для медицинских задач, где врачам нужно понимать, какие факторы повышают риск.
- **Какая модель лучше объяснима для `finance` и почему:**
  - **LogisticRegression** также лучше объяснима для finance — коэффициенты показывают, что loan_to_income увеличивает риск, а credit_score снижает. Это согласуется с финансовой логикой и позволяет принимать обоснованные решения.
- **Где вы бы выбрали более простую модель даже при близких метриках:**
  - LogisticRegression предпочтительнее RandomForest, когда важна интерпретируемость. На этих данных LR показывает roc_auc ~0.76 (medical) и ~0.716 (finance), что лишь немного уступает RF, но даёт понятные и простые объяснения.


## 6. Обязательные самостоятельные задания (без образца в solutions)

### 6.1 Согласованность глобальных объяснений
- **Насколько согласованы native importance и permutation importance?**
  - На обоих датасетах native importance и permutation importance дают одинаковые топы признаков.
  - Для medical: age, cholesterol, systolic_bp стабильно входят во все топы.
  - Для finance: loan_to_income, annual_income, credit_score стабильно входят во все топы.
  - Различия минимальны — это говорит об отсутствии сильных смещений в оценке важности.
- **Какие признаки стабильно попадают в топ для каждой модели?**
  - Medical: age, cholesterol, systolic_bp, glucose, physical_activity_hours.
  - Finance: loan_to_income, annual_income, credit_score, loan_amount, delinquency_count.
- **Файл:** `outputs/global_importance_comparison.csv`.

### 6.2 Сводка partial dependence
- **Какие признаки дали наибольший `score_delta`?**
  - Medical: age (LR: 0.429), cholesterol (LR: 0.302), systolic_bp (LR: 0.269).
  - Finance: loan_to_income (RF: 0.189), annual_income (RF: 0.192), loan_to_income (LR: 0.187).
- **Где тренд выглядит монотонным, а где нет?**
  - **Монотонные:** age (рост), loan_to_income (рост), credit_score (убывание), physical_activity_hours (убывание).
  - **Немонотонные:** cholesterol (плато после определённого уровня), annual_income (плато на высоких доходах), многие признаки в RandomForest из-за природы деревьев.
- **Файл:** `outputs/partial_dependence_summary.csv`.


## 7. Проверка понимания

1. **Почему коэффициенты логистической регрессии нельзя интерпретировать без учета препроцессинга?**
   - Коэффициенты LR зависят от масштаба признаков. Если признаки не стандартизированы, коэффициенты несопоставимы. В данном ноутбуке препроцессинг включает стандартизацию числовых признаков, поэтому коэффициенты можно сравнивать. Без препроцессинга признаки с большими значениями (например, annual_income) получат искусственно малые коэффициенты, что исказит интерпретацию.

2. **Почему permutation importance может расходиться с native importance?**
   - Native importance для RandomForest может быть смещена в пользу признаков с большим числом уникальных значений или высокой корреляцией. Permutation importance оценивает реальное влияние на метрику, поэтому она более надёжна, но требует пересчёта модели на перемешанных данных. На этих данных расхождений почти нет, что говорит о стабильности модели.

3. **Когда partial dependence стоит трактовать осторожно?**
   - Когда признак сильно коррелирует с другими признаками (эффект взаимодействия может быть скрыт). На краях распределения, где мало данных, кривые могут быть нестабильными. Для RandomForest кривые могут быть «рваными» из-за переобучения на малых выборках. Также partial dependence не показывает причинность — только ассоциацию.

4. **Почему локальное объяснение ошибки не равно причинному объяснению?**
   - Локальное объяснение показывает, какие признаки были наиболее важны для конкретного предсказания модели. Это не означает, что эти признаки «вызвали» ошибку — модель могла ошибиться из-за шума в данных, отсутствия важных признаков или неправильной калибровки. Причинность требует контролируемых экспериментов, а не анализа предсказаний модели.


## 8. Что бы вы улучшили в следующей итерации
- Добавить больше методов интерпретации (например, SHAP или LIME) для сравнения с permutation importance и partial dependence.
- Проверить устойчивость интерпретаций при разных random_state.
- Визуализировать partial dependence кривые для всех признаков, чтобы лучше понять форму зависимостей.
- Сравнить интерпретации на full и отобранном наборе признаков, чтобы увидеть, как отбор влияет на объяснимость.


## 9. Скриншоты выполнения

### Скриншот 1. Загрузка feature_sets из ЛР01
(вставь скриншот с выводом `selection_summary`)

### Скриншот 2. Качество моделей (quality_summary)
(вставь скриншот с таблицей quality_summary)

### Скриншот 3. Глобальные объяснения (top_global)
(вставь скриншот с таблицей top_global)

### Скриншот 4. Partial dependence summary
(вставь скриншот с таблицей partial_dependence_summary)

### Скриншот 5. Результат Задания 1 — согласованность объяснений
(вставь скриншот с таблицей top_df и анализом согласованности)

### Скриншот 6. Результат Задания 2 — анализ partial dependence
(вставь скриншот с топ-5 признаков по score_delta)

### Скриншот 7. Результат Задания 3 — экспорт и краткие выводы
(вставь скриншот с сообщением о сохранении файлов и краткими выводами)

### Скриншот 8. Файлы в папке outputs/
(вставь скриншот с файлами: global_importance_comparison.csv, partial_dependence_summary.csv)