# 🎯 shooting_recognition — кроссплатформенное приложение для распознавания результатов стрельбы

Flask-приложение для автоматического подсчёта очков по видеопотоку спортивной мишени. Запускается **локально** на Windows, macOS и Linux. В качестве ускорителя поддерживается CUDA, Apple MPS и CPU.

## Что умеет

- Захват видео с USB-камеры, RTSP IP-потока или из файла
- Автоматическая калибровка зон мишени (Hough circles) с возможностью ручной правки в браузере
- Обнаружение пробоин нейросетью YOLOv8, дедупликация между кадрами по IoU
- Геометрическое определение очков по координатам центра пробоины
- Сохранение результатов в PostgreSQL, аннотированных кадров на диск
- Экспорт отчётов в CSV и PDF
- Real-time трансляция в браузер через MJPEG + Server-Sent Events

## Быстрый старт

### Зависимости
- Python 3.12+ / 3.11+ / 3.10+
- PostgreSQL 16
- Обученные веса модели `best.pt` (см. `../model_training/` или GitHub Releases)

### Установка

```powershell
cd shooting_recognition
python -m venv venv
.\venv\Scripts\Activate.ps1   # PowerShell
# либо в CMD:
# venv\Scripts\activate
pip install -r requirements.txt

# Если у вас NVIDIA GeForce RTX 50xx и вы видите предупреждение про sm_120,
# переустановите PyTorch с CUDA 13.1-совместимым колесом:
# pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu131
```

### PostgreSQL

На Windows установите PostgreSQL 16 через официальный инсталлятор или `winget`:

```powershell
winget install --id PostgreSQL.PostgreSQL.16
```

Если PostgreSQL уже установлен, создайте базу:

```powershell
createdb shooting_recognition
```

На macOS / Linux:

```bash
brew install postgresql@16
brew services start postgresql@16
createdb shooting_recognition
```

### Конфигурация

```bash
cp .env.example .env       # на Windows PowerShell: copy .env.example .env
```

### Модель

```bash
mkdir -p recognition/weights
# wget https://github.com/USER/REPO/releases/download/v1.0/best.pt \
#      -O recognition/weights/best.pt
```

### Запуск

```bash
python main.py
```

Откройте `http://localhost:5000` в браузере.

> Без файла `best.pt` приложение запустится в **mock-режиме** (генерирует случайные «пробоины») — пригодно для проверки UI.

Откройте `http://localhost:5000` в браузере.

> Без файла `best.pt` приложение запустится в **mock-режиме** (генерирует случайные «пробоины») — пригодно для проверки UI.

## CLI-инструменты (без Flask и БД)

Для быстрого тестирования модели и замера производительности есть три standalone-скрипта:

```bash
# Тест на одном фото
python test_photo.py /path/to/target.jpg --conf 0.35

# Тест на видеопотоке (USB / RTSP / файл)
python test_video.py --source 0                          # USB-камера 0
python test_video.py --source rtsp://192.168.1.100/      # IP-камера
python test_video.py --source ~/Downloads/range.mp4      # видеофайл
python test_video.py --source 0 --save                   # с сохранением кадров

# Замер производительности
python benchmark.py --iterations 100
```

`benchmark.py` выводит таблицу медиан/p95/p99 для разрешений 640/1280/1920 и готовый параграф для научного отчёта.

## Структура

```
shooting_recognition/
├── main.py                          ← Flask entry point
├── test_photo.py / test_video.py    ← standalone CLI-тесты
├── benchmark.py                     ← замер производительности
├── requirements.txt
├── .env.example                     ← шаблон для .env (.env в .gitignore)
├── config/        — единая конфигурация
├── capture/       — USB / RTSP / file источники
├── motion/        — детектор движения (frame differencing)
├── recognition/   — YOLOv8 wrapper + scoring
├── calibration/   — Hough circles → 10 зон
├── database/      — PostgreSQL: модели + миграции
├── web/           — routes + templates + static
├── reporting/     — CSV и PDF экспорт
├── outputs/       — генерируемые отчёты и кадры
└── uploads/       — загруженные пользователем видео
```

## Что в `.env`

Скопируйте `.env.example` → `.env` и поправьте. Ключевые поля:

| Переменная | Что |
|-----------|-----|
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | подключение к PostgreSQL |
| `SECRET_KEY` | секрет Flask (генерируйте уникальный для production) |
| `RTSP_URL` | URL IP-камеры (если будете использовать) |
| `MODEL_CONFIDENCE` | порог уверенности модели (0.30–0.45 рекомендуется) |
| `MOTION_THRESHOLD`, `MOTION_DELAY_SEC` | чувствительность детектора движения |
| `PORT` | порт веб-сервера (по умолчанию 5000) |

## Многоуровневая отказоустойчивость

| Условие | Поведение |
|---------|-----------|
| Нет файла `best.pt` | mock-режим (случайные пробоины для теста UI) |
| Hough circles не нашёл круг | fallback: центр кадра + радиус `min(h,w)/2 − 20` |
| Пробоина за пределами всех зон | score = 0 (промах) |
| Дубликат пробоины (IoU > 0.4) | пропуск без повторного учёта |
| Нет движения | инференс не запускается (экономия GPU) |

## Документация

Полное описание архитектуры, потоков данных и API-маршрутов: [`../ARCHITECTURE.md`](../ARCHITECTURE.md), раздел 2.
