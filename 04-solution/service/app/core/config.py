"""Константы сервиса: модель, размеры, порог отказа.

Всё, что влияет на численный результат, собрано здесь и снабжено происхождением,
чтобы ни одно "магическое число" не жило в коде молча.
"""
from __future__ import annotations

import os
from pathlib import Path

# Корень решения: каталог, содержащий app/ и model/ (работает и в контейнере
# /srv/..., и при запуске из репозитория).
SERVICE_ROOT = Path(__file__).resolve().parents[2]

# Модель: OSNet-AIN x1.0 (vehicle-reid-0001, Open Model Zoo 2022.1, лицензия MIT).
# Веса поставляются в составе решения (model/), интернет на запуске не нужен.
MODEL_PATH = Path(os.environ.get("MODEL_PATH", SERVICE_ROOT / "model" / "osnet_ain_x1_0_vehicle_reid.onnx"))
MODEL_NAME = "osnet_ain_x1_0_vehicle_reid (vehicle-reid-0001, OMZ 2022.1)"
MODEL_SHA256 = "4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f"

# Вход модели: RGB 208x208 без внешней нормализации (первый узел графа —
# InstanceNormalization, см. отчёт бейзлайна). Выход — 512-мерный вектор,
# сравнение по косинусу после L2-нормировки.
INPUT_SIZE = 208
EMBEDDING_DIM = 512

# Порог режима отказа. Значение полное, округлять нельзя — иначе candidates.csv
# разойдётся с проверенным прогоном.
#
# Выбран калибровкой (см. 04-solution/refusal/) по правилу, зафиксированному до
# просмотра результатов: максимум от min(TNR, F1 при долях отказных 0.10/0.25/0.40).
# Даёт F1 0.733 и TNR 0.698 против F1 0.867 и TNR 0.144 у прежнего порога,
# выбранного по максимуму F1. ТЗ (разд. 9) вводит TNR именно затем, чтобы решение
# не принимало всё подряд, поэтому максимум F1 здесь не годится.
#
# Прежний черновой порог argmax F1 — 0.34921352213815304; сохранён в истории для
# сверки со старыми артефактами бейзлайна.
DEFAULT_THRESHOLD = float(os.environ.get("REID_THRESHOLD", "0.5495953464415451"))

# Хранилище галереи (Qdrant). Пакетный прогон его НЕ использует.
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "gallery")

SERVICE_VERSION = "0.1.0"
