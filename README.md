# RU ANPR — распознавание государственных регистрационных знаков

Проект реализует полный цикл обработки видеопотока: детекция госномеров, трекинг, распознавание символов и экспорт результата в CSV. Решение ориентировано на российские прямоугольные номера гражданских автомобилей с флагом и удовлетворяет требованиям:

- Повторная фиксация одного и того же номера не чаще чем один раз в 200 мс (около 5 кадров при 25 FPS).
- Корректная обработка неполных распознаваний (непрочитанные символы помечаются `#`).
- Используются модели и данные, доступные для некоммерческого применения или созданные самостоятельно.

## 1. Содержимое репозитория

```
.
├── run.py                        # основной скрипт инференса
├── tools/
│   ├── train_plate.py            # дообучение детектора (YOLO11)
│   ├── train_crnn.py             # дообучение OCR (CRNN)
│   ├── prepare_nomeroff.py       # подготовка датасета autoriaNumberplateOcrRu
│   └── download_hf_datasets.py   # загрузка открытых датасетов HF
├── configs/                      # конфигурации для обучения
├── ocr/                          # модуль CRNN и утилиты
├── models/                       # веса детектора и OCR 
├── data/                         # рабочая папка для датасетов
├── README.md                     # этот документ
└── requirements.txt
```

## 2. Быстрый старт

### 2.1 Предварительные требования

- Python 3.11.
- Знание компьютерной грамотности

### 2.2 Создание виртуального окружения

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2.3 Установка зависимостей

- **CPU:**
  ```bash
  pip install -r requirements.txt
  ```
- **GPU (пример для CUDA 12.1):**
  ```bash
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  pip install -r requirements.txt
  ```

### 2.4 Запуск инференса

```bash
python run.py --input путь/до/video.mp4 --out out.csv \
  --ocr_engine crnn --crnn_checkpoint models/ocr_crnn_nomeroff_clean.pt \
  --ocr_every_n 1 --device auto
```
Однострочный пример:
```powershell
python run.py --input video.mp4 --out out.csv --device cuda --ocr_engine crnn --crnn_checkpoint models\ocr_crnn_nomeroff_clean.pt --ocr_every_n 1 --conf 0.5 --char_conf_thr 0.5 --char_conf_thr_letters 0.7 --char_conf_thr_digits 0.5
```

По умолчанию:
- используется детектор YOLO11 и CRNN для распознавания;
- события дедуплируются в окне 200 мс;
- символы с низкой уверенностью заменяются `#`;
- при запуске без `--show` результаты печатаются только в консоль и файл.

CSV имеет формат:
```
time;plate_num
00:41.44;P069XO73
01:06.80;T455CK73
```

### 2.5 Основные параметры запуска

| Параметр | Назначение |
|----------|------------|
| `--input` | путь к видеофайлу или каталогу изображений |
| `--out` | CSV-отчёт (по умолчанию `out.csv`) |
| `--device {auto,cuda,cpu}` | устройство инференса |
| `--target_fps` | целевая частота обработки (стандартно 15 FPS) |
| `--conf`, `--iou` | threshold'ы YOLO |
| `--match_iou`, `--max_misses` | параметры трекера |
| `--ocr_engine {easy,paddle,crnn}` | выбор OCR |
| `--ocr_every_n` | использовать OCR раз в N-й кадр |
| `--char_conf_thr*` | пороги уверенности символов |
| `--save_crops_dir` | папка для сохранения кропов |
| `--show` | включить визуализацию (не использовать на headless-сервере) |



## 3. Практические советы

- Для соблюдения лимита 100 мс используйте `--device cuda`; на CPU поднимайте `--ocr_every_n` и снижайте `--target_fps`.
- Ложные срабатывания уменьшите, повысив `--conf`/`--iou`.
- Дребезг треков устраняется увеличением `--match_iou` и уменьшением `--max_misses`.
- При запуске на сервере без дисплея уберите `--show` и включите `--save_crops_dir`.

## 4. Обучение моделей при необходимости

### 4.1 Детектор (YOLO11)

1. Скачайте открытые датасеты (лицензии допускают некоммерческое использование):
   ```bash
   python tools/download_hf_datasets.py
   ```
2. При необходимости добавьте собственные кропы:
   ```bash
   python tools/prepare_nomeroff.py --base autoriaNumberplateOcrRu-2021-09-01
   ```
3. Запустите дообучение:
   ```bash
   python tools/train_plate.py \
     --model yolo11n.pt \
     --data configs/hf_detection.yaml \
     --epochs 80 --imgsz 1280 --batch 16
   ```
4. Лучший чекпоинт (`runs/train/.../weights/best.pt`) скопируйте в `models/yolo11_plate.pt`.

### 4.2 OCR (CRNN)

1. Подготовьте набор:
   ```bash
   python tools/download_hf_datasets.py --skip-detection
   ```
2. Дообучите распознавание:
   ```bash
   python tools/train_crnn.py \
     --data_root autoriaNumberplateOcrRu-2021-09-01 \
     --train_dir train --val_dir val \
     --output models/ocr_crnn_nomeroff_clean.pt \
     --epochs 30 --batch_size 64 --device auto
   ```

### 4.3 Источники данных

- **AutoriaNumberplateOcrRu (2021-09-01)** — лицензия CC BY-NC.
- **HuggingFace/ru-license-plate-detection** и **HuggingFace/ru-license-plate-ocr** — открытые наборы с некоммерческой лицензией.

Все материалы соответствуют требованию использования бесплатных некоммерческих источников.

## 5. Архитектура

- **Детектор:** Ultralytics YOLO11 (вариант `n`).
- **Трекер:** простая IoU-ассоциация с накоплением голосов по символам.
- **Распознавание:** CRNN с CTC, интегрированный через модуль `ocr/`.
- **Постобработка:** суммирование вероятностей символов, нормализация к шаблону российского номера, окно дедупликации 200 мс.


