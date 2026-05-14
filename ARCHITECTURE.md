# Архитектура системы автоматического распознавания результатов стрельбы

**Версия документа:** 2.0
**Дата:** 2026-05-14
**Язык реализации:** Python 3.12
**Целевая платформа:** macOS (Apple Silicon M4 Pro) + Google Colab (NVIDIA T4/A100)

---

## 0. Общее описание

Система решает задачу **автоматического подсчёта очков по фотографии/видеопотоку спортивной мишени** с пробоинами. Проект разделён на две независимые подсистемы:

| Подсистема | Путь | Среда исполнения | Назначение |
|------------|------|------------------|------------|
| **Training** | `model_training/` | Google Colab (GPU) или Mac (CPU/MPS) | Обучение нейросетевой модели YOLOv8 на объединённом наборе публичных датасетов |
| **Recognition** | `shooting_recognition/` | macOS (Apple Silicon MPS) | Реальное время: Flask web app, инференс, калибровка, PostgreSQL, отчёты |

Связующее звено — файл весов модели `best.pt`, который переносится вручную из `model_training/runs/train/shooting_target/weights/best.pt` в `shooting_recognition/recognition/weights/best.pt`. В git-репозиторий веса не входят (распространяются через GitHub Releases).

### Методологический стек

| Слой | Технология |
|------|------------|
| Детектор пробоин | YOLOv8 (Ultralytics 8.4+) |
| Ускоритель вычислений | Apple MPS (Metal Performance Shaders) на Mac, CUDA в Colab |
| Компьютерное зрение | OpenCV 4.8+ |
| Детектор мишени | Круговое преобразование Хафа (Hough Circles) |
| Веб-фреймворк | Flask 3.0 + SSE + MJPEG |
| СУБД | PostgreSQL 16 (с psycopg2 connection pooling) |
| Источник данных | Roboflow Universe — несколько датасетов, объединяемых через `src/merge.py` |
| Генерация отчётов | fpdf2 (PDF), pandas (CSV), base64-embedded HTML (отчёт обучения) |

---

## 1. Подсистема обучения (Google Colab / локально)

### 1.1. Структура каталога

```
model_training/
├── config.yaml                     # гиперпараметры (единый источник правды)
├── pipeline.py                     # CLI-оркестратор (5 шагов)
├── src/
│   ├── __init__.py                 # пустой
│   ├── merge.py                    # скачивание и объединение Roboflow-датасетов
│   ├── dataset.py                  # legacy-утилиты для одиночного датасета
│   ├── train.py                    # обучение YOLOv8 с pre-train проверками
│   ├── evaluate.py                 # оценка + 4 контрольные проверки качества
│   ├── report.py                   # HTML-отчёт с base64-графиками
│   └── inference.py                # пакетный инференс (картинки и видео)
├── notebooks/
│   └── training_colab.ipynb        # самодостаточный ноутбук (21 ячейка)
├── README.md                       # инструкция запуска
└── .pipeline_state/                # состояние между шагами CLI (автоген.)
```

### 1.2. Два режима запуска

Код обучения одинаково работает в двух сценариях:

#### Режим A — CLI (скрипт `pipeline.py`)

```bash
cd model_training/
export ROBOFLOW_API_KEY=xxxxxxxx
python pipeline.py --step all
```

Запускается **локально или в облаке**, вызывает по очереди:
1. `step_dataset()` — скачивает все источники из `cfg["datasets"]` и объединяет их в один датасет с классом `bulletholes`
2. `step_train()` — обучает модель
3. `step_evaluate()` — оценивает на test-сплите + выполняет 4 контроля
4. `step_report()` — генерирует HTML-отчёт
5. `step_inference()` — (опционально) пакетный инференс

Между шагами состояние (пути к `data.yaml`, `best.pt`) сохраняется в `.pipeline_state/` через функции `_save_state()` / `_load_state()`. Это позволяет перезапускать отдельные шаги без потери прогресса.

#### Режим B — Jupyter-ноутбук (`training_colab.ipynb`)

Ноутбук **самодостаточен** — не импортирует `src/`, а содержит весь код инлайн (включая логику merge). Это сделано для простоты запуска в Colab без клонирования репозитория. Структура из 21 ячейки:

| Ячейка | Тип | Назначение |
|--------|-----|------------|
| 0 | MD | Заголовок, таблица шагов |
| 1 | MD | Шаг 0 — Setup |
| 2 | Code | `pip install ultralytics roboflow opencv-python pandas pyyaml matplotlib seaborn` |
| 3 | Code | Проверка CUDA, вывод версий PyTorch и CUDA |
| 4 | MD | Шаг 1 — Конфигурация |
| 5 | Code | Задание `ROBOFLOW_API_KEY`, `DATASETS` (список источников с per-source `remap`-таблицами), `EPOCHS`, `BATCH`, `MODEL_SIZE`; авто-уменьшение batch при низкой VRAM; флаг `SAVE_TO_DRIVE` |
| 6 | Code | Монтирование Google Drive через `drive.mount('/content/drive')` |
| 7 | MD | Шаг 2 — Датасеты (download + merge) |
| 8 | Code | `resolve_version()` (определяет `latest` версию), `download_one()` (скачивание одного источника со skip-on-error), `ingest_source()` (переписывание лейблов с remap + sanity-check на пустые лейблы); итог — объединённый датасет в `data/merged/` |
| 9 | Code | Сводка по сплитам + разбивка по источникам (по префиксу имён файлов) + sanity-check минимальных размеров |
| 10 | MD | Шаг 3 — Обучение |
| 11 | Code | `model = YOLO(MODEL_SIZE)` → `model.train(...)` с гиперпараметрами |
| 12 | Code | Отображение 4 графиков: `results.png`, `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png` |
| 13 | MD | Шаг 4 — Оценка |
| 14 | Code | `YOLO(best.pt).val(split='test')` → сохранение метрик в `eval_metrics.json` |
| 15 | Code | Горизонтальная столбчатая диаграмма per-class mAP@50 |
| 16 | MD | Шаг 5 — Отчёт |
| 17 | Code | Сборка HTML-отчёта со встроенными base64-изображениями; отображение в ноутбуке через `IPython.display.HTML` |
| 18 | MD | Шаг 6 — Экспорт |
| 19 | Code | `shutil.copy(best.pt, /content/drive/...)` + `files.download(best.pt)` |
| 20 | MD | Следующие шаги: переместить `best.pt` в Mac-приложение |

**Важно:** несмотря на дублирование кода в ноутбуке и `src/*.py`, **оба варианта функционально эквивалентны** по результату. Ноутбук — для интерактивной работы с Colab, CLI — для воспроизводимости и автоматизации.

### 1.3. Модули `src/` — детальная спецификация

#### 1.3.1. `config.yaml`

Единый файл гиперпараметров. Загружается в `pipeline.py::_load_cfg()` через `yaml.safe_load()`. Структура:

```yaml
# Список Roboflow-датасетов для скачивания и объединения в один датасет
# с единственным классом "bulletholes" (class_id = 0).
datasets:
  - name: knsa
    workspace: knsashootingtargets
    project:   knsa-shooting-target
    version:   latest          # 'latest' → выбирается последняя версия автоматически
    remap:                     # source_class → 'bulletholes' (или null = drop)
      "0":  bulletholes
      "1":  bulletholes
      # … "2"…"10" — все зонные классы сводятся к "bulletholes"
      bulletholes: bulletholes
  - name: project_bat
    workspace: project-bat-bullet-hole-detection
    project:   bullet-hole-object-detection
    version:   latest
    remap:
      "0":  bulletholes
      # … "1"…"10"
      Target:        null      # не пробоина — drop
      black_contour: null

raw_data_dir:     ./data/raw      # сюда складываются исходные датасеты
merge_output_dir: ./data/merged   # сюда — объединённый датасет

train:
  model:         yolov8s.pt      # s/m/l/x — выбор зависит от размера датасета
  epochs:        150
  imgsz:         640
  batch:         16
  patience:      40              # early stopping
  optimizer:     AdamW
  lr0:           0.001
  lrf:           0.01
  warmup_epochs: 3
  dropout:       0.0             # 0.1 рекомендуется для m/l на малых датасетах
  weight_decay:  0.0005
  # аугментация (критична для инвариантности к углу):
  degrees:       30.0            # ±30° случайный поворот
  translate:     0.15
  scale:         0.6
  shear:         5.0
  perspective:   0.0005
  flipud:        0.10
  fliplr:        0.50
  mosaic:        1.0
  mixup:         0.15
  copy_paste:    0.10
  hsv_h:         0.015
  hsv_s:         0.70
  hsv_v:         0.40

evaluate:
  conf:  0.50
  iou:   0.50
  split: test

inference:
  conf:  0.50

output:
  project:  runs/train
  run_name: shooting_target
```

#### 1.3.2. `pipeline.py` — CLI-оркестратор

**Ключевые функции:**

```python
_load_cfg()                        → dict           # читает config.yaml
_save_state(key, value)            → None           # пишет в .pipeline_state/{key}
_load_state(key)                   → str            # читает или sys.exit(1)
step_dataset(cfg, api_key)         → None           # → merge_datasets() → _save_state('data_yaml', ...)
step_train(cfg)                    → None           # → _save_state('best_pt', ...)
step_evaluate(cfg)                 → None
step_report(cfg)                   → None
step_inference(cfg, input_path)    → None
main()                             → None           # argparse + dispatch
```

**Порядок исполнения при `--step all`:**

```
parse args → _load_cfg()
  → step_dataset(cfg, key)
      ├─ merge.merge_datasets()      ← скачивание + объединение всех источников
      └─ merge.verify_merged()       ← sanity-check итогового датасета
  → step_train(cfg)
      └─ train.run()                 (YOLO.train с гиперпараметрами + pre-train проверки)
  → step_evaluate(cfg)
      └─ evaluate.run()              (4 контрольные проверки)
  → step_report(cfg)
      └─ report.build()              (HTML + base64 PNG)
  → (step_inference не вызывается в 'all' — только явно)
```

#### 1.3.3. `src/merge.py` — объединение датасетов

**Назначение:** скачивает N датасетов Roboflow и сводит их к единой структуре с одним классом.

**Публичные функции:**

| Функция | Действие |
|---------|----------|
| `merge_datasets(sources_cfg, output_dir, api_key, raw_dir)` | Итерация по всем `datasets[*]`: скачивание каждого, применение `remap`, копирование изображений и переписывание `.txt`-лейблов в `output_dir/{train,valid,test}/{images,labels}/`. Файлы префиксуются именем источника (`knsa_*`, `project_bat_*`) для защиты от коллизий имён. Возвращает путь к `data.yaml`. |
| `verify_merged(merged_yaml)` | Контрольная проверка: nc=1, имя класса `bulletholes`, согласованность пар image↔label. |

**Внутренние:**

| Функция | Действие |
|---------|----------|
| `_resolve_version(proj, requested)` | Если `requested == 'latest'` или указанная версия отсутствует, выбирается **максимальный номер** среди `proj.versions()`. На несуществующих версиях печатает warning + список доступных. |
| `_download_one(api_key, src, raw_dir)` | Один источник: try/except оборачивает `_resolve_version()` + `proj.version(v).download()`. На ошибке возвращает `None` — каллер пропускает источник без падения пайплайна. |
| `_ingest_source(src_root, src_yaml, remap, prefix, output_dir)` | Строит `keep_ids: dict[int, int]` (исходный class_id → 0 для «оставить», отсутствие в map → drop). Для каждого изображения копирует файл с префиксом и переписывает соответствующий `.txt`. **Защита:** если `keep_ids` пуст (несовпадение имён классов источника и `remap`-таблицы) или результат содержит 0 аннотаций — печатает CRITICAL WARNING с фактическими именами классов источника. Возвращает `dict[split, count]`. |

#### 1.3.4. `src/dataset.py` (legacy)

Сохранена для обратной совместимости и одноразовых диагностических вызовов. Содержит `download()`, `inspect()`, `validate_classes()`, `patch_yaml_paths()` — функции одиночной загрузки датасета. В production-пайплайне **не используется** — её заменил `src/merge.py`.

#### 1.3.5. `src/train.py`

**Константа:** `_MODEL_SIZE_WARN = {"yolov8n": 0, "yolov8s": 0, "yolov8m": 300, "yolov8l": 500, "yolov8x": 800}` — минимальное рекомендуемое число обучающих изображений для данной архитектуры.

**Функции:**

| Функция | Действие |
|---------|----------|
| `get_device()` | Возвращает `"cuda"`, `"mps"` или `"cpu"` — проверкой в таком порядке |
| `_count_train_images(data_yaml)` | Открывает `data.yaml`, считает файлы в train-папке |
| `_pre_train_checks(t, data_yaml)` | Выводит предупреждение если выбрана крупная модель на малом датасете (например, `yolov8m` при <300 train-картинок); рекомендует `dropout: 0.1` |
| `run(cfg, data_yaml)` | Главная функция: инициализирует `YOLO(model)`, запускает `.train(...)` со всеми гиперпараметрами, возвращает путь к `best.pt` |

**Параметры, передаваемые в `YOLO.train()`:**

- `data`, `epochs`, `imgsz`, `batch`, `device`, `project`, `name`
- `patience` (early stopping; YOLO использует комбинированный fitness, в котором mAP@50-95 имеет вес 0.9)
- `save=True`, `save_period=10` (периодический чекпойнт)
- `plots=True` (сохранение кривых)
- Регуляризация: `dropout`, `weight_decay`
- Аугментация: `degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`, `fliplr`, `mosaic`, `mixup`, `copy_paste`, `hsv_h/s/v`
- Оптимизатор: `optimizer` (AdamW), `lr0`, `lrf`, `warmup_epochs`

#### 1.3.6. `src/evaluate.py` — 4 контрольные проверки

**Константы:**

```python
THRESHOLDS = {
    "mAP@50":    (0.80, 0.60),   # (GOOD, OK)
    "mAP@50-95": (0.55, 0.40),
    "Precision":  (0.80, 0.65),
    "Recall":     (0.75, 0.55),
}
_OVERFIT_GAP     = 0.15          # val_mAP − test_mAP > 0.15 ⇒ переобучение
_MIN_TEST_IMAGES = 50            # минимум для надёжной статистики
```

**Функция `run(cfg, weights, data_yaml)`** выполняет следующие шаги:

```
[CHECK 1/4] Dataset size check
  ├─ _count_test_images() → if < 50 → WARNING
[Evaluation: test split]
  └─ _eval_split(model, split='test') → test_m
[Evaluation: val split]   (только для сравнения, не влияет на метрику)
  └─ _eval_split(model, split='val')  → val_m
[CHECK 2/4] Val/Test gap
  ├─ gap = val_m['map50'] − test_m['map50']
  └─ if gap > 0.15 → OVERFITTING DETECTED + рекомендации
[CHECK 3/4] Recall gate
  └─ if test recall < 0.70 → LOW RECALL warning + рекомендации
[CHECK 4/4] Regression guard
  ├─ _load_best_metrics() (читает best_metrics.json)
  ├─ если новой модели хуже → REGRESSION DETECTED → baseline НЕ обновляется
  └─ если лучше/равной → _save_best_metrics() (обновление baseline)
Save: eval_metrics.json (полный отчёт с test + val + per_class)
```

Функция `_eval_split()` вызывает `YOLO.val(data, imgsz, split, device, conf, iou, plots, save_json, verbose)` и извлекает из результата `metrics.box.map50`, `metrics.box.map`, `metrics.box.mp`, `metrics.box.mr`, а также `metrics.box.maps[]` для per-class mAP.

#### 1.3.7. `src/report.py`

**Функции:**

| Функция | Действие |
|---------|----------|
| `_tag(v, g, o)` | Возвращает HTML-span с цветом (зелёный/оранжевый/красный) по градациям |
| `_embed(path)` | `base64.b64encode()` изображения → `<img src="data:image/png;base64,...">` |
| `_read_csv_metrics(run_dir)` | Парсит `results.csv` (выход YOLO): ищет последнюю непустую строку, извлекает финальные метрики |
| `build(run_dir, output)` | Собирает HTML: карточки с ключевыми метриками, таблица оценки, диаграмма per-class, 4 встроенных графика (results.png, confusion_matrix_normalized.png, PR_curve.png, F1_curve.png) |

Результирующий HTML-файл **самодостаточен** (все изображения встроены через base64) и может быть отправлен по email без дополнительных вложений.

#### 1.3.8. `src/inference.py`

**Функции:**

| Функция | Действие |
|---------|----------|
| `_device()` | Выбор устройства (CUDA > MPS > CPU) |
| `_process_image(model, path, out_dir, conf)` | Запуск `model(img)`, аннотация через `result.plot()`, сохранение как `ann_{name}.jpg`, возврат списка детекций |
| `_process_video(model, path, out_dir, conf)` | Покадровое чтение через `cv2.VideoCapture`, инференс каждого кадра, запись аннотированного MP4 через `cv2.VideoWriter`, накопление детекций с полем `frame` |
| `run(weights, input_path, output_dir, conf)` | Оркестратор: обходит входную папку, диспетчеризует на `_process_image` или `_process_video`, финально сохраняет `detections.json` и `detections.csv` |

### 1.4. Поток данных при обучении

```
┌─────────────────────────────────────────────────────────────────┐
│                      ROBOFLOW UNIVERSE                          │
│   knsa-shooting-target  +  bullet-hole-object-detection  + ...  │
└──────────────────────────────┬──────────────────────────────────┘
                               │  API-ключ + per-source remap-таблицы
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  merge.merge_datasets() → data/raw/{knsa,project_bat,...}/      │
│                         → data/merged/{train,valid,test}/       │
│                         → data/merged/data.yaml (nc=1)          │
│                                                                  │
│  Per-source remap: hole-классы → 'bulletholes' (id=0)           │
│                    Target, black_contour → null (drop)          │
│                                                                  │
│  Sanity-check:    keep_ids ≠ ∅, annotations_kept > 0            │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  train.run() [~30–240 мин на T4, зависит от epochs × imgsz]    │
│    ├─ pre_train_checks: модель vs размер датасета              │
│    ├─ Аугментация: mosaic, mixup, поворот ±30°, HSV-jitter     │
│    ├─ AdamW: lr=0.001, warmup 3 эпохи, cosine decay → 0.01×lr  │
│    └─ Early stopping (patience=40) по fitness ≈ 0.9·mAP@50-95   │
│                                                                  │
│  Выходные артефакты (runs/train/shooting_target/):              │
│    ├─ weights/{last.pt, best.pt}                                │
│    ├─ results.csv, results.png                                  │
│    ├─ confusion_matrix{,_normalized}.png                        │
│    ├─ PR_curve.png, F1_curve.png                                │
│    └─ args.yaml, events.out.tfevents.*                          │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  evaluate.run()                                                  │
│    ├─ [1] size check (test ≥ 50)                                │
│    ├─ [2] YOLO.val(split='test') → test metrics                 │
│    ├─ [2'] YOLO.val(split='val') → val metrics                  │
│    ├─ [3] gap check: val_mAP − test_mAP ≤ 0.15                  │
│    ├─ [4] recall ≥ 0.70                                         │
│    └─ [5] regression vs best_metrics.json                       │
│                                                                  │
│  Выход: eval_metrics.json (+ best_metrics.json если улучшение)  │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  report.build() → training_report.html (self-contained)         │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
                         best.pt (~22–50 MB)
                    ручной перенос в Mac-приложение
                  (или публикация через GitHub Releases)
```

---

## 2. Подсистема распознавания (Mac M4 Pro)

### 2.1. Структура каталога

```
shooting_recognition/
├── main.py                          # Flask entry point
├── test_photo.py                    # standalone CLI-тест на одном фото
├── test_video.py                    # standalone CLI-тест на видеопотоке
├── benchmark.py                     # замер производительности (mс/кадр, FPS)
├── requirements.txt
├── .env                             # секреты (не в git; есть .env.example)
├── config/
│   └── settings.py                  # единая конфигурация
├── capture/
│   ├── camera.py                    # USB-камера (CameraSource)
│   ├── ip_stream.py                 # IP-камера / RTSP (IPStreamSource)
│   └── file_source.py               # файлы (FileSource)
├── motion/
│   └── detector.py                  # детектор движения (frame differencing)
├── recognition/
│   ├── weights/                     # директория для best.pt (в .gitignore)
│   ├── model.py                     # TargetModel (обёртка YOLO)
│   └── scorer.py                    # ShotScorer (трекинг + scoring)
├── calibration/
│   ├── auto_detect.py               # Hough circles → 10 зон
│   ├── zone_store.py                # in-memory cache (JSON backed)
│   └── zones.json                   # текущие калибровочные зоны (gitignored)
├── database/
│   ├── db.py                        # connection pool PostgreSQL
│   ├── models.py                    # CRUD для sessions / shots
│   └── migrations/001_init.sql      # DDL схема
├── web/
│   ├── routes.py                    # Flask Blueprint + background pipeline
│   ├── templates/*.html             # index / calibrate / session / result
│   └── static/{css,js}/
├── reporting/
│   ├── csv_report.py                # export_csv(session_id)
│   └── pdf_report.py                # export_pdf(session_id)
├── outputs/
│   ├── reports/                     # CSV и PDF отчёты сессий
│   └── detection_frames/            # сохранённые кадры с пробоинами (per session)
└── uploads/                         # загруженные видеофайлы (source_type=file)
```

### 2.2. Точка входа и жизненный цикл

```bash
cd shooting_recognition/
source venv/bin/activate
python main.py                        # запуск сервера
```

**`main.py` выполняет:**
1. `init_db()` — проверяет подключение к PostgreSQL, прогоняет миграции
2. `create_app()` — создаёт Flask-приложение, регистрирует blueprint из `web/routes.py`
3. `app.run(host='0.0.0.0', port=5000, threaded=True, debug=DEBUG)` — threaded режим обязателен, т. к. MJPEG-стриминг и background-pipeline требуют отдельных потоков

Сервер открывается на `http://localhost:5000`.

### 2.3. HTTP-маршруты (`web/routes.py`)

| Маршрут | Метод | Описание |
|---------|-------|----------|
| `/` | GET | Главная страница (`index.html`) — форма выбора источника (USB/IP/файл) и имени игрока |
| `/start` | POST | Создаёт сессию в БД, запускает `_run_pipeline()` в daemon-потоке, перенаправляет на `/calibrate` |
| `/calibrate` | GET | Страница калибровки (`calibrate.html`) — canvas-редактор зон |
| `/calibrate/frame` | GET | Текущий кадр калибровки как JPEG (204 если нет) |
| `/calibrate/zones` | GET | JSON текущих зон |
| `/calibrate/zones` | POST | Сохранение отредактированных зон (в `ZoneStore` + БД + файл) |
| `/calibrate/confirm` | POST | Подтверждение без правки; `calibration_done = True`; редирект на `/session` |
| `/session` | GET | Страница сессии (`session.html`) — live stream + счётчик |
| `/stream` | GET | MJPEG-стрим (`multipart/x-mixed-replace; boundary=frame`, ~25 FPS) |
| `/events` | GET | Server-Sent Events: каждое попадание → `data: {"type": "shot", ...}` |
| `/status` | GET | JSON: `{running, calibration_done, shot_count, total_score}` |
| `/finish` | POST | `finish_session()`, `export_csv()`, `export_pdf()`, редирект на `/result/<id>` |
| `/result/<int:session_id>` | GET | Таблица попаданий + итоговый балл |
| `/download/<filename>` | GET | Скачивание CSV или PDF из `outputs/reports/` |

### 2.4. Глобальное состояние

В `routes.py` объявлен словарь `_state`, защищённый `threading.Lock`:

```python
_state = {
    "running": bool,                 # активна ли сессия
    "calibration_done": bool,        # подтвердил ли пользователь зоны
    "session_id": int,
    "player_name": str,
    "shot_count": int,
    "total_score": int,
    "events": list[dict],            # очередь событий для SSE
    "latest_frame": np.ndarray,      # последний кадр для /stream
    "calibration_frame": np.ndarray, # кадр для /calibrate/frame
    "calibration_zones": dict,
}
```

**Это однопользовательский прототип** — один `_state` на весь сервер. Для многопользовательского режима потребуется отдельный `_state` на `session_id` (см. §6.4).

### 2.5. Background pipeline: `_run_pipeline(source_tuple)`

Ядро системы распознавания. Запускается в **daemon-потоке** при вызове `/start`. Работает в цикле по кадрам из источника:

```
┌──────────────────────────────────────────────────────────────┐
│  _run_pipeline((src_type, src_val))                           │
│  ─────────────────────────────────                            │
│  Инициализация:                                               │
│    model  = TargetModel()                                     │
│    scorer = ShotScorer()                                      │
│    motion = MotionDetector()                                  │
│    zone_store = ZoneStore()                                   │
│                                                               │
│  Открытие источника (context manager):                        │
│    CameraSource | IPStreamSource | FileSource                 │
│                                                               │
│  Цикл по кадрам:                                              │
│  ═══════════════                                              │
│    for frame in ctx.get_frames():                             │
│       if not _state['running']: break                         │
│       frame_count += 1                                        │
│                                                               │
│  ▸ Фаза 1: КАЛИБРОВКА (первые 5 кадров)                       │
│    if frame_count <= CALIBRATION_FRAMES (=5):                 │
│        _state['calibration_frame'] = frame                    │
│        if frame_count == 5:                                   │
│            zones = detect_target(frame)   # Hough circles     │
│            zone_store.set(zones)                              │
│            _state['calibration_zones'] = zones                │
│        continue                                               │
│                                                               │
│  ▸ Фаза 2: ОЖИДАНИЕ ПОДТВЕРЖДЕНИЯ                             │
│    while not _state['calibration_done']:                      │
│        time.sleep(0.05); continue                             │
│                                                               │
│  ▸ Фаза 3: ДЕТЕКЦИЯ                                            │
│    display = _draw_zones(frame.copy(), zones)                 │
│    _state['latest_frame'] = display                           │
│                                                               │
│    if not motion.has_changed(frame):                          │
│        continue   # нет движения — нет инференса              │
│                                                               │
│    holes     = model.detect(frame)                            │
│    new_holes = scorer.find_new_holes(holes)                   │
│                                                               │
│    for hole in new_holes:                                     │
│        shot_num += 1                                          │
│        total_score += hole['score']                           │
│        # ── Persist annotated frame for analysis ──           │
│        frame_path = _save_shot_frame(                         │
│            frame, zones, hole, session_id, shot_num)          │
│        # outputs/detection_frames/session_{id}/               │
│        #     shot_{N}_score{S}_conf{C}.jpg                    │
│                                                               │
│        add_shot(session_id, shot_num, score,                  │
│                 confidence, bbox, frame_path)  # INSERT in DB │
│        _state['events'].append({              # → SSE         │
│            'type': 'shot', 'shot_number': shot_num,           │
│            'score': score, 'total': total_score               │
│        })                                                     │
│  finally:                                                     │
│    _state['running'] = False                                  │
└──────────────────────────────────────────────────────────────┘
```

### 2.6. Модульные компоненты (Mac)

#### 2.6.1. `config/settings.py`

Единая точка конфигурации. Загружает переменные окружения из `.env` (через `python-dotenv`), задаёт константы:

- `MODEL_PATH = BASE_DIR / "recognition/weights/best.pt"`
- `CALIBRATION_PATH = BASE_DIR / "calibration/zones.json"`
- `DB_*` — параметры PostgreSQL (все через `os.getenv` с дефолтами)
- `DEFAULT_CAMERA_INDEX = 0`, `RTSP_URL` (envar), `FRAME_WIDTH=1280`, `FRAME_HEIGHT=720`
- `MOTION_THRESHOLD = 5000` (пикселей), `MOTION_BLUR_KERNEL = 21`, `MOTION_DELAY_SEC = 1.5`
- `MODEL_CONFIDENCE = 0.50` (переопределяется через `.env`)
- `SCORE_CLASSES = {"10": 10, ..., "1": 1}` — классы модели с зональным scoring
- `HOLE_CLASSES = {"bulletholes", "bullet_hole", "hole", "Bulletholes"}` — общий класс (fallback)
- `CALIBRATION_FRAMES = 5`
- `DEVICE = _detect_device()` — порядок: `torch.backends.mps.is_available()` → `torch.cuda.is_available()` → `"cpu"`

#### 2.6.2. `recognition/model.py` — класс `TargetModel`

**Глобальные переменные** (для отложенной загрузки):

```python
_model: YOLO | None = None
_mock_mode: bool   = False
```

**Функция `_load_model()`** вызывается один раз при инициализации первого `TargetModel()`. Если файл весов не существует или его загрузка бросает исключение — устанавливается `_mock_mode = True` (для интеграционного тестирования UI без реальной модели).

**Метод `detect(frame) -> List[Dict]`:**

В реальном режиме (`_real_detect`):

```python
results = _model(frame, conf=MODEL_CONFIDENCE, device=DEVICE, verbose=False)
for box in results[0].boxes:
    cls_name = result.names[int(box.cls[0])]
    score = SCORE_CLASSES.get(cls_name, 0)           # 0 для "bulletholes"
    is_hole = score > 0 or cls_name in HOLE_CLASSES  # пропустить target, black_contour
    if not is_hole:
        continue
    holes.append({
        "score": score,                              # 0 = «пусть scorer решит»
        "class": cls_name,
        "confidence": float(box.conf[0]),
        "box": [x1, y1, x2, y2],
        "center": [(x1+x2)/2, (y1+y2)/2],
    })
return holes
```

**Почему score может быть 0:** модель обучена на объединённом датасете с единственным классом `"bulletholes"` (все исходные зонные классы 0–10 сводятся к одному через `remap` в `merge.py`). Определение очков делегируется `ShotScorer` через калибровочные зоны. Архитектура также поддерживает модели с 10 зонными классами — в таком случае очки приходят непосредственно из имени класса.

#### 2.6.3. `recognition/scorer.py` — класс `ShotScorer`

Решает две задачи: **(а) дедупликация пробоин между кадрами, (б) разрешение очков**.

**Константа:** `_IOU_THRESHOLD = 0.40` — пороговый IoU для определения «та же самая пробоина».

**Метод `find_new_holes(current_holes)`:**

```python
new_holes = []
for hole in current_holes:
    if not self._is_duplicate(hole, self._prev_holes):
        resolved = dict(hole)
        resolved["score"] = self._resolve_score(hole)
        new_holes.append(resolved)
self._prev_holes = current_holes    # обновление окна
return new_holes
```

**Приоритет определения очков (`_resolve_score`):**

```
1. Если есть калибровочные зоны → zone_store.score_for_point(cx, cy)
   │   ├─ Если точка в зоне → её score (1-10)
   │   └─ Иначе → 0 (промах за пределы)
2. Иначе (нет зон) → hole.get('score', 0)  # из модели напрямую
```

Это **ключевое архитектурное решение**: калибровка имеет приоритет над выводом модели, что позволяет использовать любую модель-детектор пробоин (даже с одним классом) и корректно определять очки через геометрию мишени.

#### 2.6.4. `calibration/auto_detect.py`

**`detect_target(frame) -> dict`:**

```python
gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 2)
circles = cv2.HoughCircles(
    blurred, cv2.HOUGH_GRADIENT,
    dp=1,
    minDist = min(h, w) // 2,
    param1=50,                 # верхний Canny threshold
    param2=30,                 # accumulator threshold
    minRadius = min(h, w) // 8,
    maxRadius = min(h, w) // 2,
)
```

Из всех обнаруженных окружностей выбирается наибольшая (предположительно — внешний контур мишени). При неудаче детекции применяется **fallback**: центр изображения + радиус = `min(h,w)/2 − 20`. Далее вызывается `_build_zones(cx, cy, outer_radius)`.

**`_build_zones(cx, cy, outer_radius) -> list`:**

Строит 10 концентрических зон:
```python
for score in range(10, 0, -1):
    zone_radius = int(outer_radius * (11 - score) / 10)
```

Формула: зона score=10 (центральная, «десятка») имеет радиус `outer_radius × 1/10`, зона score=1 (внешняя) — `outer_radius × 10/10 = outer_radius`. Зоны сохраняются в порядке от центра к краю.

**`get_score_from_zones(x, y, zones) -> int`:**

```python
for zone in zones:   # от центра к краю (10 → 1)
    dist = math.hypot(x - zone["cx"], y - zone["cy"])
    if dist <= zone["radius"]:
        return zone["score"]
return 0   # за пределами всех зон — промах
```

Итерация от центра гарантирует, что точке в зоне «10» возвращается значение 10, а не 1.

#### 2.6.5. `motion/detector.py` — класс `MotionDetector`

Обнаруживает **межкадровое изменение пикселей** через простое дифференцирование:

```python
def has_changed(frame):
    gray = cv2.cvtColor(frame, BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)     # MOTION_BLUR_KERNEL
    if self._prev_gray is None:
        self._prev_gray = gray
        return False
    diff    = cv2.absdiff(self._prev_gray, gray)
    _, thr  = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    changed = cv2.countNonZero(thr)
    self._prev_gray = gray

    now = time.monotonic()
    if changed > MOTION_THRESHOLD (5000) and (now - self._last_trigger) > MOTION_DELAY_SEC (1.5):
        self._last_trigger = now
        return True
    return False
```

Двойная защита: **пространственная** (число изменённых пикселей > 5000) и **временная** (≥1.5 с между срабатываниями). Без детектора движения модель запускалась бы на каждом кадре, что перегружало бы GPU и генерировало бы ложные многократные детекции одной и той же пробоины.

#### 2.6.6. `capture/*.py` — источники кадров

Все три класса реализуют **идентичный контракт** через context manager + генератор:

```python
class Source:
    def __enter__(self):  ...       # открытие ресурса
    def __exit__(self):  ...        # освобождение
    def get_frames(self):  ...      # генератор: yield frame while ok
```

- **`CameraSource(index=0)`** — `cv2.VideoCapture(index)` с `set(CAP_PROP_FRAME_WIDTH=1280)` и `set(CAP_PROP_FRAME_HEIGHT=720)`
- **`IPStreamSource(url)`** — `cv2.VideoCapture(rtsp_url)`
- **`FileSource(path)`** — если расширение в `ALLOWED_IMAGE_EXT` (`.jpg`, `.jpeg`, `.png`), yield один кадр через `cv2.imread`; если в `ALLOWED_VIDEO_EXT` (`.mp4`, `.avi`, `.mov`), покадровое чтение через `cv2.VideoCapture`

Единый интерфейс позволяет `_run_pipeline()` работать с любым источником без модификации кода.

#### 2.6.7. `database/` — PostgreSQL-слой

**`db.py` — пул соединений:**

```python
_pool = psycopg2.pool.SimpleConnectionPool(minconn=1, maxconn=10, ...)
get_connection() → conn              # из пула
release_connection(conn)             # возврат в пул
init_db() → executes migrations/001_init.sql
```

**`migrations/001_init.sql` — DDL:**

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    player_name TEXT NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    calibration_zones JSONB
);
CREATE TABLE IF NOT EXISTS shots (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    shot_number INTEGER NOT NULL,
    score INTEGER NOT NULL CHECK (score BETWEEN 0 AND 10),
    confidence REAL,
    bbox JSONB,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    frame_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_shots_session_id ON shots(session_id);
```

**`models.py` — CRUD:**

| Функция | SQL |
|---------|-----|
| `create_session(player_name, calibration_zones)` | `INSERT INTO sessions(player_name, calibration_zones) VALUES(%s, %s) RETURNING id` |
| `finish_session(session_id)` | `UPDATE sessions SET finished_at = NOW() WHERE id = %s` |
| `update_session_zones(session_id, zones)` | `UPDATE sessions SET calibration_zones = %s WHERE id = %s` |
| `add_shot(session_id, shot_number, score, confidence, bbox, frame_path=None)` | `INSERT INTO shots(...) VALUES(...) RETURNING id` |
| `get_session(session_id)` | `SELECT * FROM sessions WHERE id = %s` |
| `get_session_shots(session_id)` | `SELECT * FROM shots WHERE session_id = %s ORDER BY shot_number` |

#### 2.6.8. `reporting/`

**`csv_report.py::export_csv(session_id)`** — читает сессию и попадания, строит `pandas.DataFrame` со столбцами: `player`, `shot_number`, `score`, `cumulative`, `confidence`, `timestamp`; сохраняет как `outputs/reports/session_{id}_{safe_name}.csv`.

**`pdf_report.py::export_pdf(session_id, target_image_path=None)`** — использует `fpdf2`:

- Шапка с метаданными (игрок, время начала/конца)
- Опциональное изображение мишени
- Таблица попаданий (№, очки, кумулятивный счёт, время)
- Итоговая сумма и максимум (= `shot_count × 10`)

Оба отчёта вызываются в `/finish` маршруте.

### 2.7. Вспомогательные CLI-скрипты

#### 2.7.1. `test_photo.py` — одиночный кадр

**Изолированный тест** — не требует PostgreSQL и Flask. Используется для быстрой проверки модели на одном фото:

```bash
python test_photo.py /path/to/photo.jpg --conf 0.35
```

**Что делает:**
1. Импортирует только `YOLO`, `MODEL_PATH`, `DEVICE` из проекта
2. Загружает модель, выполняет инференс на указанном изображении
3. Печатает число найденных пробоин и их характеристики (класс, уверенность, центр)
4. Сохраняет аннотированное изображение как `result_{name}.jpg` в ту же папку
5. Принудительное завершение через `os._exit(0)` — предотвращает зависание torch/ultralytics background-потоков на macOS

#### 2.7.2. `test_video.py` — видеопоток

Standalone-аналог Flask-pipeline без БД и веб-интерфейса. Принимает USB-камеру, RTSP-поток или видеофайл:

```bash
python test_video.py --source 0                          # USB камера 0
python test_video.py --source rtsp://192.168.1.100/      # IP-камера
python test_video.py --source ~/Downloads/range.mp4      # видеофайл
python test_video.py --source 0 --save                   # с сохранением кадров
python test_video.py --source 0 --no-motion              # без motion-фильтра
```

**Что делает:**
1. Открывает указанный источник через `cv2.VideoCapture`
2. Использует тот же `MotionDetector`, что и Flask-pipeline
3. На каждый кадр с движением запускает YOLO и печатает `[14:32:01] frame #N → 2 holes (conf 0.71, 0.65)`
4. При `--save` сохраняет аннотированный кадр в `outputs/detection_frames/test_video_{timestamp}/`
5. По завершении выводит итоговую статистику: всего кадров, было движение, отправлено в YOLO, найдено пробоин

#### 2.7.3. `benchmark.py` — замер производительности

Измеряет время операций пайплайна для использования в научном отчёте:

```bash
python benchmark.py                       # синтетический кадр
python benchmark.py --image ~/test.jpg    # реальное изображение
python benchmark.py --iterations 200      # точнее перцентили
```

**Что измеряет:**
- Время загрузки модели (однократно)
- Инференс YOLO на разрешениях 640×640, 1280×720, 1920×1080 (n итераций, медиана/p95/p99/среднее)
- Детектор движения (`MotionDetector.has_changed()`)
- Полный цикл «motion check → inference → scoring»
- Throughput (FPS) в непрерывном 10-секундном цикле

**Два формата вывода:**
1. Компактная таблица в терминал (для быстрого инспектирования)
2. Готовый параграф для НИР с подставленными реальными цифрами (раздел 5.1)

---

## 3. Межмодульное взаимодействие и поток данных

### 3.1. Цепочка зависимостей (import graph)

```
main.py ─── web.routes (Blueprint) ─┬─ config.settings
                                     ├─ database.{db, models}
                                     ├─ capture.{camera, file_source, ip_stream}
                                     ├─ motion.detector
                                     ├─ recognition.{model, scorer}
                                     ├─ calibration.{auto_detect, zone_store}
                                     └─ reporting.{csv_report, pdf_report}

recognition.model ─── config.settings
recognition.scorer ─── calibration.zone_store
calibration.zone_store ─── calibration.auto_detect
```

### 3.2. Хронологический поток данных в одной сессии

```
User opens http://localhost:5000
     │
     ▼
/                   → форма выбора источника
     │
POST /start         → create_session(name) → session_id=N
     │                ↓ spawn daemon thread
     │              _run_pipeline(source)
     │                │
     │                ▼ [фрейм 1..5]
     │              frame_count++   state['calibration_frame']=frame
     │                ↓ [фрейм 5]
     │              detect_target() → Hough circles → 10 zones
     │              zone_store.set(zones)
     │              state['calibration_zones']=zones
     │                │
     ▼                │
GET /calibrate       → render calibrate.html
     │                │
GET /calibrate/frame  ◄── state['calibration_frame']   (JPEG)
GET /calibrate/zones  ◄── state['calibration_zones']   (JSON)
     │                │
     │   [user drags canvas in browser → calibrate.js sends PUT]
     │                │
POST /calibrate/zones  ── store.save(zones), update_session_zones()
  or /calibrate/confirm state['calibration_done']=True
     │                │
     ▼                ▼ [фрейм 6...]
GET /session         draw zones overlay on frame
                      motion.has_changed(frame)?
                         │ no → continue
                         │ yes
                         ▼
                      holes = model.detect(frame)
                      new = scorer.find_new_holes(holes)
                         │
                         ▼ for each new hole:
                      score = zone_store.score_for_point(cx, cy)
                      _save_shot_frame(frame, ...) → JPEG on disk
                      add_shot(session_id, n, score, conf, bbox, frame_path)
                      state['events'].append({type:'shot', ...})

GET /stream ─────────◄── state['latest_frame']   (MJPEG multipart)
GET /events ─────────◄── state['events']         (SSE text/event-stream)

     │
POST /finish         finish_session(id), export_csv(), export_pdf()
     │                state['running']=False → pipeline thread exits
     ▼
GET /result/<id>     render shot table + total score
GET /download/*      serve CSV/PDF from outputs/reports/
```

### 3.3. Концурентная модель

- **Главный Flask-поток** — обработка HTTP-запросов (threaded=True допускает параллельные запросы)
- **Pipeline daemon-поток** (1 шт. на сессию) — захват кадров + инференс
- **SSE-генератор** (1 шт. на открытую страницу /events) — опрашивает `_state["events"]` каждые 100 мс
- **MJPEG-генератор** (1 шт. на каждое `<img src="/stream">`) — читает `_state["latest_frame"]` каждые 40 мс

**Синхронизация:** все обращения к `_state` — под `threading.Lock()`. Крупные объекты (numpy frames) копируются через `.copy()` перед помещением в `_state`, чтобы избежать TOCTOU-гонок.

---

## 4. Сводная таблица: где что запускается

| Компонент | Где запускается | Триггер | Runtime |
|-----------|-----------------|---------|---------|
| `model_training/pipeline.py` | Colab или Mac | CLI: `python pipeline.py --step X` | Последовательные шаги |
| `model_training/notebooks/training_colab.ipynb` | **Только Colab** | Выполнение ячеек вручную | 21 ячейка последовательно |
| `model_training/src/merge.py::merge_datasets()` | Colab или Mac | `step_dataset()` или ячейка [8] | Скачивание + объединение (~5–15 мин) |
| `model_training/src/train.py::run()` | Colab (CUDA) / Mac (MPS) / CPU | `step_train()` | Одиночный прогон обучения (30 мин – 6 ч) |
| `model_training/src/evaluate.py::run()` | Аналогично | `step_evaluate()` | ~1 мин на test-сплите |
| `shooting_recognition/main.py` | **Только Mac** | `python main.py` | Flask dev-сервер (long-running) |
| `_run_pipeline()` в `routes.py` | Mac | POST `/start` | Daemon-поток (до `/finish`) |
| `TargetModel.detect()` | Mac (MPS) | В каждом кадре с движением | Зависит от imgsz; см. benchmark.py |
| `detect_target()` | Mac | Кадр №5 текущей сессии | Один вызов (Hough circles) |
| `MotionDetector.has_changed()` | Mac | В каждом кадре после калибровки | ~5 мс на кадр |
| `test_photo.py` | Mac | CLI: `python test_photo.py ...` | Одиночный запуск (~5 с) |
| `test_video.py` | Mac | CLI: `python test_video.py --source ...` | Long-running до Ctrl+C |
| `benchmark.py` | Mac | CLI: `python benchmark.py` | ~60–90 с (зависит от iterations) |

---

## 5. Контрольные точки качества (quality gates)

### 5.1. Pre-training (до обучения)

| Проверка | Где | Действие при несоответствии |
|----------|-----|-----------------------------|
| Roboflow API-ключ задан | `pipeline.py` / ячейка [5] | sys.exit(1) с инструкцией |
| Совпадение имён классов и `remap`-таблицы | `merge._ingest_source` | CRITICAL WARNING с фактическим списком классов источника |
| Количество сохранённых аннотаций > 0 | `merge._ingest_source` | ZERO ANNOTATIONS warning (предотвращает обучение «вхолостую») |
| Версия датасета существует | `merge._resolve_version` | Fallback на `latest` + warning |
| Падение скачивания одного источника | `merge._download_one` | Источник пропускается, merge продолжается |
| train ≥ 200, valid ≥ 50, test ≥ 50 | `pipeline.step_dataset` (после merge) | warning о риске статистической нестабильности |
| Model size vs dataset size | `train._pre_train_checks` | warning с рекомендациями (например, `dropout: 0.1`) |

### 5.2. Post-training (после обучения)

| Проверка | Порог | Если не выполнено |
|----------|-------|-------------------|
| Test size | ≥ 50 изображений | WARNING: метрики ненадёжны |
| Val − Test gap по mAP@50 | ≤ 0.15 | OVERFITTING DETECTED + рекомендации |
| Test Recall | ≥ 0.70 | LOW RECALL + рекомендации по conf/данным |
| Regression vs baseline | новые метрики ≥ старым | REGRESSION DETECTED, baseline не обновляется |

### 5.3. Runtime (Mac-приложение)

| Проверка | Действие |
|----------|----------|
| Model file отсутствует | `_mock_mode = True` (рандомные пробоины) |
| Загрузка модели упала с ошибкой | `_mock_mode = True` |
| Hough circles не нашёл круг | Fallback: центр кадра + радиус min(h,w)/2−20 |
| Пробоина за пределами всех зон | score = 0 (промах) |
| Дубликат пробоины (IoU > 0.4) | Пропуск без повторного учёта |
| Отсутствие движения | Инференс не запускается (экономия GPU) |

---

## 6. Рекомендации для научной публикации

### 6.1. Воспроизводимость
- Все гиперпараметры централизованы в `config.yaml`
- Версии источников фиксируются через `'latest'` с автоматическим fallback (документируется в выводе пайплайна)
- Seed для YOLO задаётся ultralytics по умолчанию (0)
- Артефакты обучения (`results.csv`, `eval_metrics.json`, `best_metrics.json`) сохраняются в одной директории и достаточны для полного восстановления эксперимента
- Объединённый датасет имеет каноническую структуру YOLOv8 — пригоден для повторного использования без дополнительной обработки

### 6.2. Метрики для отчёта
Система сохраняет в `eval_metrics.json`:
- `test.{map50, map5095, precision, recall, per_class}`
- `val.{map50, map5095, precision, recall, per_class}`
- `gap_val_test` (разница)

Это позволяет в статье продемонстрировать и абсолютные показатели, и оценку обобщающей способности модели. Для замеров производительности — отдельный скрипт `benchmark.py`, генерирующий готовый параграф для НИР.

### 6.3. Архитектурные особенности, достойные упоминания
1. **Объединение нескольких публичных датасетов** через per-source `remap`-таблицы — позволяет агрегировать разнородно размеченные источники в единую структуру с устойчивостью к схеме классов конкретного источника
2. **Делегирование scoring на геометрию мишени** (через `zone_store`) вместо обучения 10-классовой модели — снижает требования к разметке датасета и упрощает применение к мишеням другого формата
3. **Двойной контроль движения** (пространственный + временной) предотвращает многократный подсчёт одной пробоины
4. **Многоуровневая fault-tolerance**: отсутствует модель → mock; Hough не нашёл → центр кадра; одиночный источник упал → merge продолжается; remap не совпал → CRITICAL warning
5. **Регрессионная защита baseline** — автоматически предотвращает замену хорошей модели на ухудшенную
6. **Self-contained HTML-отчёт** с base64-вложениями — удобен для архивирования и email-рассылок

### 6.4. Ограничения реализации
1. Однопользовательская сессия (один `_state` на сервер) — прототип, не production
2. Нет аутентификации пользователей
3. Motion detection по порогу пикселей не различает пробоину и движение стрелка/мишени
4. Калибровка предполагает круговую мишень и близкую к фронтальной проекцию; для овальных (перспективных) искажений потребуется homography
5. `'latest'`-резолюция версий датасетов делает воспроизводимость зависимой от состояния Roboflow на момент запуска — для строгой воспроизводимости рекомендуется зафиксировать конкретные номера версий в `config.yaml` после первого успешного прогона

---

## 7. Ссылки на исходный код

Все утверждения этого документа верифицированы по актуальному состоянию репозитория. Ключевые исходники (пути относительные от корня репозитория):

- `model_training/pipeline.py`
- `model_training/src/merge.py`
- `model_training/src/{dataset,train,evaluate,report,inference}.py`
- `model_training/config.yaml`
- `model_training/notebooks/training_colab.ipynb`
- `shooting_recognition/main.py`
- `shooting_recognition/web/routes.py`
- `shooting_recognition/recognition/{model,scorer}.py`
- `shooting_recognition/calibration/{auto_detect,zone_store}.py`
- `shooting_recognition/motion/detector.py`
- `shooting_recognition/capture/{camera,ip_stream,file_source}.py`
- `shooting_recognition/database/{db,models}.py`
- `shooting_recognition/database/migrations/001_init.sql`
- `shooting_recognition/reporting/{csv_report,pdf_report}.py`
- `shooting_recognition/config/settings.py`
- `shooting_recognition/{test_photo,test_video,benchmark}.py`
