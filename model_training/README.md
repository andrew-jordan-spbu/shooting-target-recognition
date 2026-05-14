# 🎯 model_training — обучение YOLOv8 для распознавания пробоин

Подсистема обучения детектирующей модели на **объединённом** наборе публичных датасетов изображений спортивных мишеней. Работает независимо от Flask-приложения (`shooting_recognition/`).

## Два способа запуска

### 1. Google Colab (рекомендуется — GPU бесплатно)
1. Откройте [colab.research.google.com](https://colab.research.google.com)
2. `File → Upload notebook` → загрузите `notebooks/training_colab.ipynb`
3. `Runtime → Change runtime type → T4 GPU`
4. В ячейке `Step 1 — Configuration` вставьте свой Roboflow API-ключ
5. Запускайте ячейки по порядку

### 2. Локально (CUDA / Apple MPS / CPU)
```bash
cd model_training
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Все шаги за один запуск:
python pipeline.py --step all --api-key YOUR_ROBOFLOW_KEY

# Или по шагам (состояние сохраняется в .pipeline_state/):
python pipeline.py --step dataset  --api-key YOUR_ROBOFLOW_KEY  # скачать + объединить
python pipeline.py --step train                                  # обучение
python pipeline.py --step evaluate                               # 4 контрольные проверки
python pipeline.py --step report                                 # HTML-отчёт
```

## Структура

```
model_training/
├── pipeline.py             ← CLI: python pipeline.py --step all
├── config.yaml             ← все гиперпараметры (редактировать здесь)
├── requirements.txt
├── notebooks/
│   └── training_colab.ipynb   ← самодостаточный Colab-ноутбук
└── src/
    ├── merge.py            ← скачивание + объединение нескольких датасетов
    ├── dataset.py          ← legacy-утилиты для одиночного датасета
    ├── train.py            ← обучение YOLOv8 (CUDA / MPS / CPU)
    ├── evaluate.py         ← оценка на test + 4 контрольные проверки
    ├── report.py           ← HTML-отчёт с base64-графиками
    └── inference.py        ← пакетный инференс
```

## Объединение датасетов

В `config.yaml` (секция `datasets`) перечислены источники с per-source `remap`-таблицами. Каждая таблица сопоставляет имя класса источника либо строке `bulletholes`, либо `null` (drop). Все hole-классы сводятся к единственному классу `bulletholes` (id=0), нерелевантные (`Target`, `black_contour` и т. п.) — отбрасываются. Файлы префиксуются именем источника для защиты от коллизий имён.

При `version: latest` пайплайн автоматически выбирает максимальный номер версии из доступных в Roboflow. При недоступности одного источника он пропускается, merge продолжается с остальными (skip-on-error).

## Контрольные проверки качества (Quality Gates)

| Этап | Проверка | Действие при сбое |
|------|----------|-------------------|
| До обучения | Совпадение `remap` и реальных имён классов | CRITICAL WARNING со списком найденных классов |
| До обучения | Размер сплитов ≥ 200/50/50 | Warning о статистической нестабильности |
| До обучения | Совместимость модели и размера датасета | Warning + рекомендация `dropout: 0.1` |
| После обучения | Test ≥ 50 изображений | Warning: метрики ненадёжны |
| После обучения | Val − Test gap по mAP@50 ≤ 0.15 | OVERFITTING DETECTED + рекомендации |
| После обучения | Test Recall ≥ 0.70 | LOW RECALL + рекомендации |
| После обучения | Регрессия vs предыдущий best_metrics.json | REGRESSION DETECTED → baseline не обновляется |

## Целевые метрики

| Метрика | Цель | Приоритет |
|---|---|---|
| mAP@50 | > 0.80 | Основной |
| Recall | > 0.75 | Высокий — пропущенная пробоина = неверный счёт |
| mAP@50-95 | > 0.55 | Дополнительный |
| Precision | > 0.80 | Дополнительный |

## После обучения

Скопируйте `best.pt` в приложение:
```bash
cp runs/train/shooting_target/weights/best.pt \
   ../shooting_recognition/recognition/weights/best.pt
```

## Источник данных

Roboflow API-ключ можно бесплатно получить на [roboflow.com](https://roboflow.com) → Account settings → API keys.

Описание архитектуры и схема потока данных: [`../ARCHITECTURE.md`](../ARCHITECTURE.md), раздел 1.
