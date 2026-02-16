# Traffic Analysis

Проект анализа данных и предсказания зарплат на основе вакансий.

## Структура проекта

- **chain_pattern** — пайплайн подготовки данных из CSV (паттерн Chain of Responsibility)
- **regression** — обучение и предсказание зарплат регрессионной моделью
- **docs** — документация проекта

## Требования

- Python 3.10+
- numpy, pandas, scikit-learn, joblib

## Использование

### 1. Подготовка данных

```bash
python -m chain_pattern.main путь/к/файлу.csv
```

Создаёт в папке с CSV-файлом матрицы `x_data.npy` и `y_data.npy`.

### 2. Обучение модели

```bash
python -m regression.train путь/к/папке_с_x_data_и_y_data
```

Сохраняет модель в `regression/resources/salary_model.joblib`.

### 3. Предсказание зарплат

```bash
python -m regression.app путь/к/x_data.npy
```

Выводит предсказанные зарплаты (по одному значению на строку).

## Запуск из корня

```bash
python app.py chain_pattern/x_data.npy
```
