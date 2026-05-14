"""
test_photo.py — быстрый тест модели на одном фото
──────────────────────────────────────────────────
Не требует PostgreSQL, Flask или калибровки.
Просто загружает модель и прогоняет инференс.

Использование:
    python test_photo.py путь/к/фото.jpg
    python test_photo.py путь/к/фото.jpg --conf 0.35
"""
import argparse
import sys
from pathlib import Path

# ── чтобы работали импорты из config/ ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np


def run(image_path: str, conf: float) -> None:
    from ultralytics import YOLO
    from config.settings import MODEL_PATH, DEVICE

    # ── Проверки ──────────────────────────────────────────────────
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[ERROR] Файл не найден: {img_path}")
        sys.exit(1)

    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Модель не найдена: {MODEL_PATH}")
        print("  Скопируйте best.pt в recognition/weights/best.pt")
        sys.exit(1)

    # ── Загрузка ──────────────────────────────────────────────────
    print(f"[INFO] Модель  : {MODEL_PATH}")
    print(f"[INFO] Устройство: {DEVICE}")
    print(f"[INFO] Порог   : conf={conf}")
    print(f"[INFO] Фото    : {img_path}")
    print("─" * 50)

    model = YOLO(str(MODEL_PATH))
    model.to(DEVICE)

    frame = cv2.imread(str(img_path))
    if frame is None:
        print("[ERROR] Не удалось прочитать изображение.")
        sys.exit(1)

    # ── Инференс ──────────────────────────────────────────────────
    results = model(frame, conf=conf, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id   = int(box.cls[0])
            cls_name = result.names[cls_id]
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            confidence = round(float(box.conf[0]), 3)
            detections.append({
                "class": cls_name,
                "confidence": confidence,
                "center": (cx, cy),
                "box": (x1, y1, x2, y2),
            })

    # ── Вывод результатов ─────────────────────────────────────────
    print(f"Найдено пробоин: {len(detections)}\n")
    for i, d in enumerate(detections, 1):
        print(f"  #{i}  класс={d['class']}  уверенность={d['confidence']}  центр={d['center']}")

    if not detections:
        print("  Пробоин не найдено.")
        print("  Попробуйте снизить --conf (например, --conf 0.25)")

    # ── Сохранение аннотированного фото ──────────────────────────
    annotated = results[0].plot()

    # Дорисовываем номера и центры
    for i, d in enumerate(detections, 1):
        cx, cy = d["center"]
        cv2.circle(annotated, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(
            annotated, f"#{i} {d['confidence']}",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
        )

    out_path = img_path.parent / f"result_{img_path.name}"
    cv2.imwrite(str(out_path), annotated)
    print(f"\n[OK] Результат сохранён: {out_path}")
    print(f"     Откройте файл: open \"{out_path}\"")


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description="Тест модели на одном фото")
    parser.add_argument("photo",          help="Путь к фото мишени")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Порог уверенности (default: 0.35)")
    args = parser.parse_args()

    run(args.photo, args.conf)
    os._exit(0)  # принудительно завершает все фоновые потоки torch/ultralytics
