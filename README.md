# 🎯 Shooting Target Recognition

Система автоматического распознавания результатов пулевой стрельбы по бумажной мишени. Состоит из двух независимых подсистем: обучения модели YOLOv8 и Mac-приложения для инференса в реальном времени.

## Из чего состоит

| Подсистема | Назначение | Где запускается |
|------------|-----------|-----------------|
| [`model_training/`](./model_training/) | Обучение YOLOv8 на объединённом датасете публичных изображений мишеней | Google Colab (GPU) или локально (CUDA / Apple MPS / CPU) |
| [`shooting_recognition/`](./shooting_recognition/) | Flask web-приложение: захват видео → детекция → подсчёт очков → отчёты | macOS Apple Silicon (предпочтительно) или CPU/CUDA |

Связующее звено — файл весов модели `best.pt`, который переносится из `model_training/runs/train/shooting_target/weights/` в `shooting_recognition/recognition/weights/`. В этот репозиторий веса не входят, см. раздел [Веса модели](#веса-модели).

## Ключевые особенности

- **Объединение нескольких публичных датасетов** через per-source `remap`-таблицы (`model_training/src/merge.py`) — единый class `bulletholes` для всех источников, защита от коллизий имён файлов, skip-on-error для недоступных источников
- **Геометрическое определение очков** через автоматическую калибровку круговым преобразованием Хафа — модели достаточно находить пробоины, не классифицируя зоны
- **Двойной контроль движения** (пространственный + временной) — инференс запускается только когда нужно
- **7 контрольных проверок качества** на этапе обучения и оценки: совпадение `remap` и реальных классов, размер сплитов, val/test gap, recall threshold, регрессия vs baseline и др.
- **Многоуровневая отказоустойчивость**: модель отсутствует → mock-режим; Hough не нашёл круг → центр кадра; пробоина вне зон → score=0; дубликат → пропуск
- **Real-time UI**: MJPEG-стриминг видео + Server-Sent Events для счёта попаданий

## Быстрый старт

### 1. Клонирование репозитория
```bash
git clone https://github.com/USER/shooting-target-recognition.git
cd shooting-target-recognition
```

### 2. Обучение модели (или скачивание готовых весов)

**Вариант А — обучить самому:**
- Откройте `model_training/notebooks/training_colab.ipynb` в Google Colab, выполните по ячейкам. Подробности: [`model_training/README.md`](./model_training/README.md)
- Скачайте полученный `best.pt`

**Вариант Б — взять готовые веса** из [GitHub Releases](./releases) этого репозитория (когда они там появятся).

Положите файл сюда:
```bash
shooting_recognition/recognition/weights/best.pt
```

### 3. Запуск Mac-приложения
```bash
cd shooting_recognition

python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# PostgreSQL (один раз)
brew install postgresql@16
brew services start postgresql@16
createdb shooting_recognition

# Конфигурация
cp .env.example .env       # отредактируйте под себя

# Старт
python main.py             # http://localhost:5000
```

Подробная инструкция: [`shooting_recognition/README.md`](./shooting_recognition/README.md).

## Стек технологий

| Слой | Технология |
|------|------------|
| Детектор пробоин | YOLOv8 (Ultralytics 8.4+) |
| Ускоритель вычислений | Apple MPS на Mac, CUDA в Colab, CPU как fallback |
| Компьютерное зрение | OpenCV 4.8+ |
| Детектор мишени | Круговое преобразование Хафа (Hough Circles) |
| Веб-фреймворк | Flask 3.0 + SSE + MJPEG |
| СУБД | PostgreSQL 16 |
| Источник данных | Roboflow Universe (несколько датасетов, объединяемых пайплайном) |
| Отчёты | fpdf2 (PDF), pandas (CSV), base64-embedded HTML |

## Структура репозитория

```
shooting-target-recognition/
├── README.md                       ← этот файл
├── ARCHITECTURE.md                 ← подробное описание архитектуры (~1000 строк)
├── LICENSE                         ← MIT
├── .gitignore
├── model_training/                 ← обучение модели (Colab / локально)
│   ├── README.md
│   ├── pipeline.py                 ← CLI: python pipeline.py --step all
│   ├── config.yaml                 ← гиперпараметры
│   ├── notebooks/
│   │   └── training_colab.ipynb    ← самодостаточный ноутбук (21 ячейка)
│   └── src/                        ← merge, train, evaluate, report, inference
└── shooting_recognition/           ← Flask web-приложение
    ├── README.md
    ├── .env.example                ← шаблон конфигурации
    ├── main.py                     ← Flask entry point
    ├── test_photo.py               ← CLI-тест на одном фото
    ├── test_video.py               ← CLI-тест на видеопотоке
    ├── benchmark.py                ← замер производительности
    ├── config/ capture/ motion/    ← модули
    ├── recognition/ calibration/
    ├── database/ web/ reporting/
    ├── outputs/ uploads/           ← runtime артефакты (gitignored)
    └── recognition/weights/        ← здесь должен лежать best.pt (gitignored)
```

## Веса модели

Файлы весов (`*.pt`) **не хранятся в git** — слишком тяжёлые и регенерируемые. Распространяются через:
- **GitHub Releases** — прикрепляются к тегам версий, скачиваются через `wget`
- **Личное обучение** — запустите `model_training/` пайплайн

## Документация

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** — детальное описание системы: модули, потоки данных, контрольные точки качества, ограничения (~1000 строк)
- **[model_training/README.md](./model_training/README.md)** — как обучить модель
- **[shooting_recognition/README.md](./shooting_recognition/README.md)** — как запустить приложение

## Лицензия

[MIT](./LICENSE)
