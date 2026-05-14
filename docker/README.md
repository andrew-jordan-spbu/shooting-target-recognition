# 🐳 Docker-сборка `shooting_recognition`

Контейнеризованный вариант Flask-приложения для распознавания мишеней. Состоит из двух сервисов:

| Сервис | Контейнер | Назначение |
|--------|-----------|------------|
| **`app`** | `shooting-target-recognition:latest` (собирается из исходников) | Python 3.12 + Flask + PyTorch + Ultralytics YOLOv8 |
| **`db`** | `postgres:16-alpine` | PostgreSQL с persistent volume |

## ⚡ Быстрый старт

```bash
cd docker/

# 1. Положите best.pt в смонтированную папку
mkdir -p ../shooting_recognition/recognition/weights
# wget https://github.com/.../releases/download/v1.0/best.pt \
#      -O ../shooting_recognition/recognition/weights/best.pt

# 2. Скопируйте и отредактируйте .env
cp .env.example .env
# при необходимости поменяйте DB_PASSWORD, MODEL_CONFIDENCE и др.

# 3. Запуск с автодетекцией GPU
./scripts/start.sh up --build

# Альтернативный синтаксис (без авто-GPU):
# docker compose up --build
```

После запуска приложение доступно на **http://localhost:5000**.

## 🖥 Поддержка GPU

`./scripts/start.sh` сам определяет наличие NVIDIA GPU через `nvidia-smi` и подключает `docker-compose.gpu.yml` поверх базового конфига:

```text
✓ NVIDIA GPU detected — including GPU override (docker-compose.gpu.yml)
→ docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

или

```text
ℹ  No NVIDIA GPU detected — CPU inference mode
→ docker compose -f docker-compose.yml up --build
```

Внутри контейнера приложение ещё раз проверяет устройство (`torch.cuda.is_available()` / `torch.backends.mps`) и выбирает CUDA → MPS → CPU в порядке приоритета. На стартовых логах будет напечатано:

```text
[entrypoint] ✓ CUDA available: 1× Tesla T4
```

или:

```text
[entrypoint] ℹ  No GPU detected — using CPU (inference will be slower)
```

### ⚠️ Платформенные ограничения

| Платформа | USB-камера | NVIDIA GPU | Apple MPS | Веб-интерфейс |
|-----------|------------|------------|-----------|---------------|
| **macOS Docker Desktop** | ❌ нет проброса USB | ❌ нет на Mac | ❌ не виден в контейнере | ✅ работает |
| **Linux + NVIDIA + nvidia-container-toolkit** | ✅ с `--device` | ✅ через `--gpus all` | — | ✅ |
| **Linux без GPU** | ✅ | ❌ | — | ✅ |

На Mac контейнер поддерживает только RTSP-поток и видеофайлы. Для USB-камеры используйте локальный запуск без Docker (`python main.py`).

## 📦 Структура

```
docker/
├── Dockerfile                ← образ Flask-приложения
├── docker-compose.yml        ← базовый CPU-конфиг (works everywhere)
├── docker-compose.gpu.yml    ← override для NVIDIA GPU
├── .dockerignore             ← исключения из build context
├── .env.example              ← шаблон env-переменных compose
├── requirements.docker.txt   ← pin-list с opencv-python-headless
├── README.md                 ← вы здесь
└── scripts/
    ├── start.sh              ← wrapper с автодетекцией GPU
    └── entrypoint.sh         ← wait-for-postgres → start app
```

## 🗂 Тома (volumes)

| Имя volume | Точка монтирования | Назначение |
|-----------|---------------------|-----------|
| `../shooting_recognition/recognition/weights` (bind-mount, **ro**) | `/app/recognition/weights` | Веса модели — обновляются заменой файла на хосте, без пересборки образа |
| `str-app-uploads` | `/app/uploads` | Загруженные пользователем видеофайлы (persist) |
| `str-app-outputs` | `/app/outputs` | Сгенерированные отчёты CSV/PDF и кадры с пробоинами (persist) |
| `str-pg-data` | `/var/lib/postgresql/data` | Данные PostgreSQL (persist) |

## 🔄 Типовые команды

```bash
# Запуск в фоне
./scripts/start.sh up -d

# Логи приложения
./scripts/start.sh logs -f app

# Перезапуск только app (после обновления best.pt — не нужно, mount подхватит автоматически;
# нужно при изменении кода)
./scripts/start.sh up -d --no-deps --build app

# Подключиться к контейнеру для отладки
docker exec -it str-app bash

# Подключиться к PostgreSQL
docker exec -it str-db psql -U postgres -d shooting_recognition

# Полная остановка с удалением volume'ов (ВНИМАНИЕ — удалит БД!)
./scripts/start.sh down -v
```

## 🩺 Healthcheck и зависимости

- `db` имеет healthcheck `pg_isready` — `app` стартует только после готовности БД (`depends_on.condition: service_healthy`)
- `app` имеет healthcheck `curl /status` — Docker сообщит `(healthy)` в `docker ps` после успешного старта Flask

## 🛠 Troubleshooting

**Проблема:** при старте `app` пишет `MOCK mode`
**Причина:** `recognition/weights/best.pt` не существует на хосте
**Решение:** скачайте веса в `../shooting_recognition/recognition/weights/best.pt` и перезапустите `app`

**Проблема:** Docker build падает на `pip install torch`
**Причина:** мало места или таймаут (PyTorch ≈ 2 ГБ)
**Решение:** увеличьте лимиты Docker Desktop (Settings → Resources)

**Проблема:** контейнер падает с `connection refused` при попытке подключиться к БД
**Причина:** Postgres не успел стартовать — healthcheck должен это покрыть
**Решение:** проверьте `docker compose logs db` — если БД здорова, перезапустите `app`

**Проблема:** на Mac в браузере открывается, но видеофайл не воспроизводится
**Причина:** см. раздел «Платформенные ограничения» — USB-камера не работает; используйте upload файла

## 📚 Документация

Архитектура приложения и описание модулей: [`../ARCHITECTURE.md`](../ARCHITECTURE.md), раздел 2.

Инструкция по запуску без Docker: [`../shooting_recognition/README.md`](../shooting_recognition/README.md).
