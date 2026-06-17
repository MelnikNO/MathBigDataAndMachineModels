# Отчет по ЛР 01.2: Wrapper + Embedded методы отбора признаков

Оценивание выполняется по единому rubric: `../RUBRIC_TEMPLATE.md`.


## 1. Контекст
- ФИО автора работы: Мельник Наталья Олеговна
- Группа: 1.2
- Дата выполнения: 2026-06-01
- Используемая среда (OS, версия Python): Windows 11, Python 3.12.7


## 2. Цель работы
Сравнить wrapper и embedded подходы к отбору признаков, сформировать 2-3 кандидатных набора признаков (feature sets) для итогового сравнения моделей в следующем ноутбуке.


## 3. Данные и подготовка
- Используемые датасеты: medical (прогноз сердечно-сосудистого риска) и finance (прогноз кредитного риска).
- Исходный shortlist из первого ноутбука: загружен из `outputs/shortlist_filter.json`.
- Размер пула признаков после фильтрации: 15 признаков для каждого датасета.


## 4. Сравнение методов значимости признаков

### 4.1 Использованные методы
| Метод | Тип | Краткое описание |
|---|---|---|
| `RFE` | Wrapper | Рекурсивное удаление наименее важных признаков по мнению модели. |
| `SequentialFeatureSelector (SFS)` | Wrapper | Последовательное добавление признаков, улучшающих метрику (forward selection). |
| `L1-регуляризация (L1 LogReg)` | Embedded | Обнуляет коэффициенты неважных признаков через L1-штраф. |
| `RandomForest Feature Importance` | Embedded | Оценивает важность признаков на основе деревьев решений. |
| `Permutation Importance` | Embedded/Пост-хок | Оценивает падение качества модели при перемешивании значений признака. |

### 4.2 Результаты ранжирования (feature_ranking)

**Для датасета finance (топ-20):**

| Метод | Топ-5 признаков | Комментарий |
|---|---|---|
| `l1_logreg` | previous_default_yes, loan_to_income, credit_score, annual_income, housing_status_mortgage | L1 выделила категориальный признак как самый важный. |
| `permutation` | loan_to_income, annual_income, housing_status_mortgage, previous_default_no, previous_default_yes | Подтвердил важность loan_to_income, но с меньшими scores. |

**Для датасета medical (по итогам анализа):**
- Стабильный топ: `age`, `cholesterol`, `systolic_bp`, `glucose`, `physical_activity_hours`.
- L1-регуляризация и RFE показали схожие результаты.

### 4.3 Эксперимент с `n_features_to_select`
- Было: автоматическое значение `min(10, max(4, X_train.shape[1] // 2))`.
- Стало: фиксированное значение `12`.
- **Вывод:** при увеличении числа отбираемых признаков в топ добавляются менее сильные признаки, но топ-5 остаются стабильными.

### 4.4 Что изучено по ходу выполнения
- **Какие 2-3 идеи/свойства методов вы изучили:**
  1. Wrapper-методы (RFE, SFS) учитывают взаимодействие признаков, но требуют больше времени.
  2. Embedded-методы (L1, RF Importance) быстрее, но ограничены типом модели.
  3. Permutation Importance — самый надёжный метод, но самый медленный.
- **Какие различия между методами вы увидели:**
  - RFE и SFS дали почти одинаковые топы (Jaccard ~0.7-0.8).
  - L1-регуляризация выделила `previous_default_yes` как самый важный признак для finance, в то время как другие методы поставили его ниже.
- **Ссылки на источники:**
  - https://scikit-learn.org/stable/modules/feature_selection.html
  - `study-notes/glossary.md`
- **Термины из глоссария:**
  - RFE, SequentialFeatureSelector, Permutation Importance, L1-регуляризация.


## 5. Кандидатные наборы признаков

### 5.1 Сформированные наборы

| Набор | Методы, использованные для формирования | Medical | Finance |
|---|---|---|---|
| `set_A_wrapper` | RFE, SFS, L1 | age, cholesterol, systolic_bp, physical_activity_hours, glucose, smoking_status_never, bmi, stress_level, resting_heart_rate, smoking_status_former | annual_income, credit_score, previous_default_yes, utilization_ratio, loan_to_income, savings_balance, delinquency_count, housing_status_mortgage, employment_years, loan_amount |
| `set_B_tree` | RF Importance, Permutation | cholesterol, diastolic_bp, systolic_bp, age, physical_activity_hours, alcohol_units_weekly, glucose, resting_heart_rate, bmi, sex_male | loan_to_income, annual_income, loan_amount, credit_score, utilization_ratio, housing_status_mortgage, savings_balance, previous_default_no, delinquency_count, age |
| `set_C_hybrid` | Пересечение set_A и set_B | age, cholesterol, systolic_bp, physical_activity_hours, glucose, bmi, resting_heart_rate, smoking_status_never, stress_level, smoking_status_former | annual_income, credit_score, utilization_ratio, loan_to_income, savings_balance, delinquency_count, housing_status_mortgage, loan_amount |

### 5.2 set_D_robust (устойчивый набор)

**Метод формирования:**
- Признаки с `stability_rate >= 0.6` по методу RFE (при разных random_state).
- При необходимости дополнены гибридным набором до 5 признаков.

| Dataset | set_D_robust |
|---|---|
| medical | age, cholesterol, systolic_bp, glucose, physical_activity_hours |
| finance | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count |

### 5.3 Что изучено по ходу выполнения
- **Что вы изучили о формировании наборов:**
  - Гибридный набор (`set_C_hybrid`) объединяет сильные стороны разных подходов.
  - Устойчивый набор (`set_D_robust`) даёт признаки, стабильно отбираемые при разных условиях.
- **Какие сравнения оказались наиболее показательными:**
  - Сравнение overlap/Jaccard между методами показало, что RFE и SFS наиболее согласованы.
  - L1-регуляризация дала более жёсткий отбор, обнуляя слабые признаки.
- **Ссылки на источники:**
  - https://scikit-learn.org/stable/modules/feature_selection.html
- **Термины из глоссария:**
  - set_A_wrapper, set_B_tree, set_C_hybrid, set_D_robust.


## 6. Стабильность отбора

### 6.1 Согласованность методов (method_agreement_long)

**Ключевые выводы:**
- Наибольшая согласованность: RFE ↔ SFS (Jaccard ~0.7-0.8) — оба используют логистическую регрессию.
- Средняя согласованность: L1 ↔ RFE/SFS (Jaccard ~0.5-0.6).
- Наименьшая согласованность: RandomForest Importance ↔ Permutation Importance (Jaccard ~0.4-0.5) — используют разные принципы оценки.

### 6.2 Стабильность по random_state (selection_stability)

**Ключевые выводы:**
- RFE более стабилен, чем L1-регуляризация, при изменении random_state.
- Для finance признаки с `stability_rate >= 0.6`: `loan_to_income`, `annual_income`, `credit_score`.
- Для medical признаки с `stability_rate >= 0.6`: `age`, `cholesterol`, `systolic_bp`, `glucose`.

### 6.3 Что изучено по ходу выполнения
- **Что вы изучили о стабильности:**
  - RFE показал более высокую стабильность, чем L1-регуляризация.
  - Признаки с высокой стабильностью совпадают с топами из всех методов.
- **Какие сравнения подходов оказались наиболее показательными:**
  - Сравнение stability_rate показало, что топ-признаки стабильны, а менее важные — нет.
- **Ссылки на источники:**
  - https://scikit-learn.org/stable/modules/feature_selection.html
- **Термины из глоссария:**
  - Stability Rate, Jaccard Index, Overlap.


## 7. Выводы

### 7.1 Основные результаты
1. **Wrapper-методы** (RFE, SFS) дают стабильные и согласованные результаты.
2. **Embedded-методы** (L1, RF Importance) быстрее, но их результаты зависят от типа модели.
3. **Гибридный подход** (`set_C_hybrid`) — лучший баланс качества и интерпретируемости.
4. **Устойчивый набор** (`set_D_robust`) — наиболее надёжный для продакшн-систем.

### 7.2 Рекомендуемый набор для следующего ноутбука
- Основной кандидат: **`set_C_hybrid`** (10 признаков).
- Резервный кандидат: **`set_D_robust`** (5 признаков).

### 7.3 Что изучено по ходу выполнения
- **Какие методические выводы вы сделали:**
  - Разные методы отбора дают разные топы, но пересечение топов даёт устойчивый результат.
  - Важно проверять стабильность отбора при изменении random_state.
- **Как промежуточные наблюдения изменяли ход эксперимента:**
  - Увеличение `n_features_to_select` с 10 до 12 показало, что топ-5 остаются стабильными.
  - Анализ stability_rate помог сформировать `set_D_robust`.
- **Ссылки на источники:**
  - https://scikit-learn.org/stable/modules/feature_selection.html
  - `study-notes/glossary.md`


## 8. Проверка понимания

1. **Чем отличаются wrapper и embedded методы отбора признаков?**
   - Wrapper-методы используют модель для оценки важности признаков (RFE, SFS). Embedded-методы встраивают отбор в процесс обучения (L1, RF Importance). Wrapper-методы точнее, но медленнее.

2. **Почему L1-регуляризация может давать другие топы, чем RFE?**
   - L1-регуляризация штрафует коэффициенты, обнуляя неважные признаки. RFE использует ранжирование на основе модели. L1 более жёсткая и может выделить другие признаки, особенно если есть корреляции между признаками.

3. **Что такое stability_rate и зачем его считать?**
   - Stability_rate — доля запусков, в которых признак был отобран при разных random_state. Он показывает, насколько устойчив отбор признаков, что важно для надёжности модели.


## 9. Что бы вы улучшили в следующей итерации
- Добавить больше значений random_state для более точной оценки stability_rate.
- Использовать другие модели-оценщики для RFE и SFS (например, RandomForest).
- Сравнить время выполнения всех методов и выбрать оптимальный по скорости/качеству.


## 10. Скриншоты выполнения

### Скриншот 1. Загрузка shortlist из Notebook 1
(вставь скриншот с выводом `shortlist loaded: True`)

### Скриншот 2. Таблица feature_ranking.head(20)
(вставь скриншот с таблицей из второго ноутбука)

### Скриншот 3. feature_sets (set_A_wrapper, set_B_tree, set_C_hybrid)
(вставь скриншот с выводом словаря feature_sets)

### Скриншот 4. Проверки пройдены успешно
(вставь скриншот с выводом `Проверки пройдены успешно`)

### Скриншот 5. Результат Задания 1 — method_agreement_long
(вставь скриншот с таблицей)

### Скриншот 6. Результат Задания 2 — selection_stability
(вставь скриншот с таблицей)

### Скриншот 7. Результат Задания 3 — feature_sets с set_D_robust
(вставь скриншот с обновлённым словарем feature_sets)

### Скриншот 8. Файлы в папке outputs/
(вставь скриншот с файлами: feature_ranking_wrapper_embedded.csv, feature_sets_wrapper_embedded.json, method_agreement_long.csv, selection_stability.csv)