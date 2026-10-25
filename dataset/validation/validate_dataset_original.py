#!/usr/bin/env python3
"""Валидатор датасета нестандартных ГРЗ. Волга-IT 2026, полуфинал, номинация «ИИ и анализ данных».

Проверяет датасет участника на соответствие разделу 6 задания:
структура каталога, meta.csv, labels/*.txt, формат номеров, bbox/quad,
согласованность полей, лицензии, рекомендуемые объёмы.

Только стандартная библиотека Python 3.9+. Размеры изображений читаются
из заголовков JPEG/PNG без сторонних пакетов.

Запуск:
    python validate_dataset.py path/to/dataset
    python validate_dataset.py path/to/dataset --report report.txt --json report.json
    python validate_dataset.py path/to/dataset --forbidden-hashes debug_hashes.txt

Служебный режим для организаторов — список md5 отладочного набора,
чтобы проверять, что его изображения не попали в датасет:
    python validate_dataset.py --hash-dir path/to/debug_set > debug_hashes.txt

Код возврата: 0 — ошибок нет (предупреждения допускаются), 1 — есть ошибки,
2 — датасет не найден / не удалось прочитать.

Уровни сообщений:
    ERROR   — нарушение формата раздела 6; датасет к сдаче не допускается.
    WARNING — отклонение от рекомендаций (объёмы, лицензии, подозрительные данные);
              сдавать можно, но жюри увидит это в отчёте.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

VERSION = "1.0"

# ---------------------------------------------------------------------------
# Справочник формата (раздел 6.2 и «Справочная информация»)
# ---------------------------------------------------------------------------

PLATE_TYPES = ("type1", "type1a", "type1b", "other")
TARGET_TYPES = ("type1", "type1a", "type1b")

META_COLUMNS = (
    "image", "plate_num", "plate_type", "bbox", "quad",
    "is_vehicle", "is_synthetic", "source", "license", "conditions",
)

CONDITIONS = ("day", "night", "rain", "snow", "dirt", "glare", "motion_blur", "angle")

# Буквы, совпадающие по начертанию в кириллице и латинице.
LATIN_LETTERS = "ABEKMHOPCTYX"
CYR_LETTERS = "АВЕКМНОРСТУХ"
CYR_TO_LAT = str.maketrans(CYR_LETTERS + CYR_LETTERS.lower(), LATIN_LETTERS + LATIN_LETTERS)

_L = f"[{LATIN_LETTERS}#]"
_D = "[0-9#]"
_REGION = f"{_D}{{2,3}}"

# Маски номера по типам. Нераспознанная человеком позиция — «#».
#   type1, type1a : Л ЦЦЦ ЛЛ РР(Р)     — ГОСТ Р 50577-2018, типы 1 и 1А
#   type1b        : ЛЛ ЦЦЦ РР(Р)       — ГОСТ Р 50577-2018, тип 1Б (такси / пассажирские)
#   other         : произвольная строка из A-Z, 0-9, «#» (может быть пустой)
PLATE_MASKS = {
    "type1": re.compile(f"^{_L}{_D}{{3}}{_L}{{2}}{_REGION}$"),
    "type1a": re.compile(f"^{_L}{_D}{{3}}{_L}{{2}}{_REGION}$"),
    "type1b": re.compile(f"^{_L}{{2}}{_D}{{3}}{_REGION}$"),
    "other": re.compile(r"^[A-Z0-9#]{0,12}$"),
}
# Длина «тела» номера без кода региона — чтобы отделить регион.
BODY_LEN = {"type1": 6, "type1a": 6, "type1b": 5}

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Рекомендуемые минимальные объёмы (раздел 6.1).
MIN_REAL_IMAGES = {"type1a": 150, "type1b": 300, "other": 50}
MIN_REAL_UNIQUE_PLATES = {"type1a": 50, "type1b": 100}
MIN_SYNTHETIC_IMAGES = 5000

# Лицензии. Ключи — нормализованные подстроки в поле license.
LICENSES_OK = ("cc0", "cc-by-4", "cc by 4", "cc-by 4", "ccby4", "public domain", "pdm",
               "own", "own_photo", "собствен", "своя", "author")
LICENSES_SHARE_ALIKE = ("by-sa", "by sa", "bysa", "gpl", "odbl")
LICENSES_NC = ("-nc", " nc", "noncommercial", "non-commercial", "некоммерч")
LICENSES_ND = ("-nd", " nd", "noderiv", "no-deriv")

# Формат labels/*.txt: YOLO — «class cx cy w h» (нормированные 0..1), опционально
# ещё 8 чисел — quad x1 y1 ... x4 y4 (нормированные). Один знак — одна строка.
LABEL_CLASSES = {0: "type1", 1: "type1a", 2: "type1b", 3: "other"}


# ---------------------------------------------------------------------------
# Сбор сообщений
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    level: str          # ERROR | WARNING
    code: str           # короткий код проверки, для группировки
    where: str          # meta.csv:42 / images/real/x.jpg / <root>
    message: str

    def fmt(self) -> str:
        return f"[{self.level}] {self.code:<22} {self.where}: {self.message}"


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def error(self, code: str, where: str, message: str) -> None:
        self.issues.append(Issue("ERROR", code, where, message))

    def warn(self, code: str, where: str, message: str) -> None:
        self.issues.append(Issue("WARNING", code, where, message))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "ERROR"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "WARNING"]


# ---------------------------------------------------------------------------
# Чтение размеров изображений из заголовков (без Pillow)
# ---------------------------------------------------------------------------

def image_size(path: str) -> tuple[int, int] | None:
    """(width, height) для JPEG/PNG или None, если файл не распознан."""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
            if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
                w, h = struct.unpack(">II", head[16:24])
                return int(w), int(h)
            if head[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    marker = f.read(2)
                    if len(marker) < 2 or marker[0] != 0xFF:
                        return None
                    # Пропуск байтов-заполнителей 0xFF.
                    while marker[1] == 0xFF:
                        nxt = f.read(1)
                        if not nxt:
                            return None
                        marker = b"\xff" + nxt
                    code = marker[1]
                    if code in (0xD8, 0x01) or 0xD0 <= code <= 0xD7:
                        continue  # маркеры без длины
                    length_b = f.read(2)
                    if len(length_b) < 2:
                        return None
                    (length,) = struct.unpack(">H", length_b)
                    # SOF0..SOF15, кроме DHT (C4), JPG (C8), DAC (CC)
                    if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                        data = f.read(5)
                        if len(data) < 5:
                            return None
                        _prec, h, w = struct.unpack(">BHH", data)
                        return int(w), int(h)
                    if code == 0xD9 or code == 0xDA:
                        return None
                    f.seek(length - 2, os.SEEK_CUR)
    except OSError:
        return None
    return None


def md5_of_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Проверки отдельных полей
# ---------------------------------------------------------------------------

def check_plate_num(plate: str, plate_type: str) -> list[str]:
    """Список текстов ошибок для номера (пустой список — номер корректен)."""
    problems: list[str] = []
    if plate != plate.strip():
        problems.append("пробелы по краям plate_num")
        plate = plate.strip()
    if any(ch in CYR_LETTERS + CYR_LETTERS.lower() for ch in plate):
        problems.append(
            f"кириллица в plate_num «{plate}», нужна латиница: «{plate.translate(CYR_TO_LAT)}»"
        )
        return problems
    if plate != plate.upper():
        problems.append(f"plate_num «{plate}» должен быть в ВЕРХНЕМ регистре")
        return problems
    mask = PLATE_MASKS.get(plate_type)
    if mask is None:
        return problems  # тип проверяется отдельно
    if not mask.fullmatch(plate):
        expected = {
            "type1": "Л ЦЦЦ ЛЛ РР(Р), напр. A123BC777",
            "type1a": "Л ЦЦЦ ЛЛ РР(Р), напр. A123BC777",
            "type1b": "ЛЛ ЦЦЦ РР(Р), напр. AB123 77 → AB12377",
            "other": "A-Z, 0-9, «#», до 12 символов или пусто",
        }[plate_type]
        problems.append(f"plate_num «{plate}» не соответствует маске {plate_type}: {expected}")
        return problems
    if plate_type in BODY_LEN:
        region = plate[BODY_LEN[plate_type]:]
        # Первую цифру трёхзначного кода не проверяем: с 2024–2025 гг. выдаются
        # коды вида 323, 550, 252 — правило «начинается с 1, 2 или 7» устарело.
        if "#" not in region and set(region) == {"0"}:
            problems.append(f"код региона «{region}» не существует")
        if plate.count("#") == len(plate):
            problems.append("plate_num состоит только из «#» — такой знак лучше не включать")
    return problems


def parse_int_list(value: str, n: int) -> list[int] | None:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != n:
        return None
    try:
        return [int(float(p)) if re.fullmatch(r"-?\d+(\.0+)?", p) else int(p) for p in parts]
    except ValueError:
        return None


def polygon_area_signed(pts: list[tuple[float, float]]) -> float:
    """Знаковая площадь (формула шнурков). В экранных координатах (y вниз)
    положительное значение = обход по часовой стрелке."""
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def classify_license(value: str) -> str:
    v = value.strip().lower()
    if not v:
        return "empty"
    if any(k in v for k in LICENSES_NC):
        return "nc"
    if any(k in v for k in LICENSES_ND):
        return "nd"
    if any(k in v for k in LICENSES_SHARE_ALIKE):
        return "sa"
    if any(k in v for k in LICENSES_OK):
        return "ok"
    return "unknown"


# ---------------------------------------------------------------------------
# Основная проверка
# ---------------------------------------------------------------------------

def norm_rel(path: str) -> str:
    return path.replace("\\", "/").strip()


def list_files(root: str, subdir: str) -> dict[str, str]:
    """Относительные пути (с прямыми слэшами) → абсолютные, для файлов под root/subdir."""
    out: dict[str, str] = {}
    base = os.path.join(root, subdir)
    if not os.path.isdir(base):
        return out
    for dirpath, _dirs, files in os.walk(base):
        for name in files:
            abs_path = os.path.join(dirpath, name)
            rel = norm_rel(os.path.relpath(abs_path, root))
            out[rel] = abs_path
    return out


def validate(root: str, *, forbidden_hashes: set[str] | None = None,
             skip_hashes: bool = False, labels_format: str = "yolo") -> Report:
    rep = Report()
    root = os.path.abspath(root)

    # ---- 1. Структура каталога -------------------------------------------------
    required_dirs = ("images", "images/real", "images/synthetic", "labels", "generator")
    for d in required_dirs:
        if not os.path.isdir(os.path.join(root, d)):
            rep.error("structure", "<root>", f"нет каталога {d}/")
    for f_name in ("meta.csv", "README.md", "LICENSE"):
        if not os.path.isfile(os.path.join(root, f_name)):
            rep.error("structure", "<root>", f"нет файла {f_name}")

    gen_dir = os.path.join(root, "generator")
    if os.path.isdir(gen_dir):
        gen_files = [f for _d, _s, fs in os.walk(gen_dir) for f in fs]
        if not any(f.endswith(".py") for f in gen_files):
            rep.error("generator", "generator/", "нет ни одного .py — скрипт-генератор синтетики обязателен")
        if not any(f.lower().startswith("requirements") or f in ("pyproject.toml", "environment.yml")
                   for f in gen_files):
            rep.warn("generator", "generator/", "нет requirements.txt / pyproject.toml — зависимости генератора не зафиксированы")

    lic_path = os.path.join(root, "LICENSE")
    if os.path.isfile(lic_path):
        try:
            with open(lic_path, encoding="utf-8", errors="replace") as f:
                lic_text = f.read(4000).lower()
            if "cc by 4.0" not in lic_text and "cc-by-4.0" not in lic_text and \
               "attribution 4.0" not in lic_text and "creativecommons.org/licenses/by/4.0" not in lic_text:
                rep.warn("license-file", "LICENSE", "не похоже на CC BY 4.0 — датасет публикуется под CC BY 4.0 (п. 6.3.2)")
        except OSError:
            rep.error("license-file", "LICENSE", "не удалось прочитать")

    readme_path = os.path.join(root, "README.md")
    if os.path.isfile(readme_path):
        try:
            with open(readme_path, encoding="utf-8", errors="replace") as f:
                readme = f.read().lower()
            for kw, label in (("лиценз", "лицензии"), ("источник", "источники")):
                if kw not in readme and label not in readme:
                    rep.warn("readme", "README.md", f"в датащите не упомянуты {label}")
            if len(readme.strip()) < 300:
                rep.warn("readme", "README.md", "датащит слишком короткий (< 300 символов): нужны источники, лицензии, методика, статистика")
        except OSError:
            rep.error("readme", "README.md", "не удалось прочитать")

    # ---- 2. Индекс изображений и меток ----------------------------------------
    images = list_files(root, "images")
    labels = list_files(root, "labels")
    image_files = {p: a for p, a in images.items() if p.lower().endswith(IMAGE_EXTS)}
    for p in images:
        if p not in image_files:
            rep.warn("stray-file", p, "не изображение .jpg/.png — лишний файл в images/")
    for p in image_files:
        if not (p.startswith("images/real/") or p.startswith("images/synthetic/")):
            rep.error("structure", p, "изображение вне images/real/ и images/synthetic/")

    # ---- 3. meta.csv ----------------------------------------------------------
    meta_path = os.path.join(root, "meta.csv")
    rows: list[dict[str, str]] = []
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "rb") as fb:
                raw = fb.read()
            if raw.startswith(b"\xef\xbb\xbf"):
                rep.warn("meta-encoding", "meta.csv", "файл с BOM (UTF-8-SIG); допустимо, но лучше без BOM")
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                rep.error("meta-encoding", "meta.csv", "не UTF-8")
                text = raw.decode("utf-8", errors="replace")
            first_line = text.splitlines()[0] if text.strip() else ""
            if ";" not in first_line:
                rep.error("meta-format", "meta.csv:1", "разделитель должен быть «;»")
            reader = csv.DictReader(text.splitlines(), delimiter=";")
            header = tuple(reader.fieldnames or ())
            if header != META_COLUMNS:
                missing = [c for c in META_COLUMNS if c not in header]
                extra = [c for c in header if c not in META_COLUMNS]
                msg = []
                if missing:
                    msg.append(f"нет колонок {missing}")
                if extra:
                    msg.append(f"лишние колонки {extra}")
                if not missing and not extra:
                    msg.append(f"порядок колонок должен быть {list(META_COLUMNS)}")
                rep.error("meta-header", "meta.csv:1", "; ".join(msg))
            if not any(c in header for c in ("image", "plate_num")):
                reader = None  # заголовок безнадёжен — строки не разбираем
            if reader is not None:
                for i, row in enumerate(reader, start=2):
                    if row.get(None):
                        rep.error("meta-format", f"meta.csv:{i}", f"лишние поля в строке: {row[None]}")
                    row = {k: (v or "").strip() for k, v in row.items() if k is not None}
                    row["_line"] = str(i)
                    rows.append(row)
        except OSError as exc:
            rep.error("meta-format", "meta.csv", f"не удалось прочитать: {exc}")

    if os.path.isfile(meta_path) and not rows:
        rep.error("meta-empty", "meta.csv", "нет ни одной строки данных")

    # ---- 4. Построчная проверка meta.csv ---------------------------------------
    size_cache: dict[str, tuple[int, int] | None] = {}
    rows_by_image: dict[str, list[dict]] = defaultdict(list)
    real_images_by_type: dict[str, set[str]] = defaultdict(set)
    real_plates_by_type: dict[str, set[str]] = defaultdict(set)
    synthetic_images: set[str] = set()
    real_images: set[str] = set()
    type_counter: Counter = Counter()
    cond_counter: Counter = Counter()
    source_counter: Counter = Counter()
    license_counter: Counter = Counter()

    for row in rows:
        where = f"meta.csv:{row['_line']}"
        image = norm_rel(row.get("image", ""))
        plate_num = row.get("plate_num", "")
        plate_type = row.get("plate_type", "")
        row["_image"] = image

        # image
        if not image:
            rep.error("meta-image", where, "пустое поле image")
            continue
        if image.startswith("/") or re.match(r"^[A-Za-z]:", image) or ".." in image.split("/"):
            rep.error("meta-image", where, f"путь должен быть относительным к корню датасета: «{image}»")
        if "\\" in row.get("image", ""):
            rep.warn("meta-image", where, "в пути обратные слэши — используйте «/»")
        if image not in image_files:
            # Может, совпадает без учёта регистра (Windows) — это тоже ошибка для Linux.
            lower_map = {p.lower(): p for p in image_files}
            if image.lower() in lower_map:
                rep.error("meta-image", where, f"регистр имени не совпадает: в meta «{image}», на диске «{lower_map[image.lower()]}»")
            else:
                rep.error("meta-image", where, f"файл не найден: {image}")
            continue
        rows_by_image[image].append(row)

        # plate_type
        if plate_type not in PLATE_TYPES:
            rep.error("meta-type", where, f"plate_type «{plate_type}» не из {PLATE_TYPES}")
        else:
            type_counter[plate_type] += 1

        # plate_num
        for msg in check_plate_num(plate_num, plate_type if plate_type in PLATE_TYPES else "type1"):
            rep.error("meta-plate", where, msg)
        if plate_type == "other" and plate_num and PLATE_MASKS["type1"].fullmatch(plate_num) and "#" not in plate_num:
            rep.warn("meta-plate", where, f"plate_type=other, но номер «{plate_num}» выглядит как обычный знак типа 1 — проверьте тип")

        # is_vehicle / is_synthetic
        is_vehicle = row.get("is_vehicle", "")
        is_synth = row.get("is_synthetic", "")
        if is_vehicle not in ("0", "1"):
            rep.error("meta-flags", where, f"is_vehicle должен быть 0 или 1, получено «{is_vehicle}»")
        if is_synth not in ("0", "1"):
            rep.error("meta-flags", where, f"is_synthetic должен быть 0 или 1, получено «{is_synth}»")
        else:
            in_synth_dir = image.startswith("images/synthetic/")
            if is_synth == "1" and not in_synth_dir:
                rep.error("meta-flags", where, "is_synthetic=1, но файл не в images/synthetic/")
            if is_synth == "0" and in_synth_dir:
                rep.error("meta-flags", where, "is_synthetic=0, но файл в images/synthetic/")
            if is_synth == "1":
                synthetic_images.add(image)
            else:
                real_images.add(image)
                if plate_type in PLATE_TYPES:
                    real_images_by_type[plate_type].add(image)
                    if plate_num and "#" not in plate_num:
                        real_plates_by_type[plate_type].add(plate_num)

        # source / license
        source = row.get("source", "")
        license_v = row.get("license", "")
        if not source:
            rep.error("meta-source", where, "пустое поле source")
        else:
            source_counter[source if len(source) < 60 else source[:57] + "..."] += 1
        lic_class = classify_license(license_v)
        license_counter[license_v or "<пусто>"] += 1
        if lic_class == "empty":
            rep.error("meta-license", where, "пустое поле license")
        elif is_synth == "0":
            if lic_class == "nc":
                rep.warn("meta-license", where, f"лицензия «{license_v}» с NC: несовместима с публикацией датасета под CC BY 4.0 (п. 6.3.2) — жюри проверит")
            elif lic_class == "nd":
                rep.warn("meta-license", where, f"лицензия «{license_v}» с ND: запрещает производные (кропы, размытие) — жюри проверит")
            elif lic_class == "sa":
                rep.warn("meta-license", where, f"лицензия «{license_v}» ShareAlike: перелицензирование под CC BY 4.0 невозможно — жюри проверит")
            elif lic_class == "unknown":
                rep.warn("meta-license", where, f"лицензия «{license_v}» не распознана — укажите SPDX-идентификатор (CC0-1.0, CC-BY-4.0) или own_photo")

        # conditions
        conditions = row.get("conditions", "")
        if conditions:
            toks = [t.strip() for t in conditions.split(",")]
            bad = [t for t in toks if t not in CONDITIONS]
            if bad:
                rep.error("meta-conditions", where, f"неизвестные условия {bad}; допустимо: {', '.join(CONDITIONS)}")
            if "day" in toks and "night" in toks:
                rep.warn("meta-conditions", where, "одновременно day и night")
            for t in toks:
                cond_counter[t] += 1
        elif is_synth == "0":
            rep.warn("meta-conditions", where, "conditions пусто для реального изображения")

        # bbox / quad — нужны размеры изображения
        if image not in size_cache:
            size_cache[image] = image_size(image_files[image])
        size = size_cache[image]
        if size is None:
            rep.error("image-file", image, "не удалось прочитать заголовок JPEG/PNG — файл повреждён или не изображение")
            continue
        img_w, img_h = size

        bbox = parse_int_list(row.get("bbox", ""), 4)
        if bbox is None:
            rep.error("meta-bbox", where, f"bbox должен быть «x,y,w,h» целыми, получено «{row.get('bbox', '')}»")
        else:
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                rep.error("meta-bbox", where, f"bbox с неположительной шириной/высотой: {bbox}")
            elif x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                rep.error("meta-bbox", where, f"bbox {bbox} выходит за границы изображения {img_w}x{img_h}")
            elif w < 12 or h < 6:
                rep.warn("meta-bbox", where, f"очень маленький знак {w}x{h} px — проверьте разметку")
            elif w * h > 0.9 * img_w * img_h:
                rep.warn("meta-bbox", where, "bbox занимает почти весь кадр — это кроп знака, а не кадр с камеры?")
            if plate_type in ("type1", "type1b") and h > 0 and w / h < 2.0:
                rep.warn("meta-bbox", where, f"{plate_type}, но соотношение сторон bbox {w / h:.2f} — однострочный знак обычно ≥ 2.5 (перспектива?)")
            if plate_type == "type1a" and h > 0 and w / h > 3.0:
                rep.warn("meta-bbox", where, f"type1a, но соотношение сторон bbox {w / h:.2f} — квадратный знак обычно ≈ 1.7")

        quad = parse_int_list(row.get("quad", ""), 8)
        if quad is None:
            rep.error("meta-quad", where, f"quad должен быть 8 целых «x1,y1,...,x4,y4», получено «{row.get('quad', '')}»")
        else:
            pts = [(float(quad[i]), float(quad[i + 1])) for i in range(0, 8, 2)]
            if any(px < 0 or py < 0 or px > img_w or py > img_h for px, py in pts):
                rep.error("meta-quad", where, f"quad {quad} выходит за границы изображения {img_w}x{img_h}")
            area = polygon_area_signed(pts)
            if abs(area) < 1.0:
                rep.error("meta-quad", where, "вырожденный quad (нулевая площадь)")
            elif area < 0:
                rep.error("meta-quad", where, "quad обходится против часовой стрелки — нужно по часовой от левого верхнего угла")
            else:
                # Первая точка — левая верхняя: минимальная сумма x+y среди углов.
                first_idx = min(range(4), key=lambda k: pts[k][0] + pts[k][1])
                if first_idx != 0:
                    rep.error("meta-quad", where, f"первая точка quad должна быть левым верхним углом, а это точка №{first_idx + 1}")
            if bbox is not None:
                qx = [p[0] for p in pts]
                qy = [p[1] for p in pts]
                qbox = (min(qx), min(qy), max(qx) - min(qx), max(qy) - min(qy))
                if bbox_iou(tuple(map(float, bbox)), qbox) < 0.8:
                    rep.warn("meta-quad", where, f"bbox {bbox} не согласован с quad (IoU < 0.8) — bbox должен описывать quad")

    # ---- 5. Изображения без разметки, дубликаты строк -------------------------
    for p in image_files:
        if p not in rows_by_image:
            rep.warn("no-meta", p, "изображение не упомянуто в meta.csv (негативный кадр без знаков? тогда всё равно нужна строка с plate_type=other или удалите файл)")
    for image, rs in rows_by_image.items():
        seen = set()
        for r in rs:
            key = (r.get("plate_num"), r.get("bbox"))
            if key in seen:
                rep.warn("meta-duplicate", f"meta.csv:{r['_line']}", f"дубликат строки для {image} ({key[0]}, {key[1]})")
            seen.add(key)

    # ---- 6. labels/*.txt --------------------------------------------------------
    label_map: dict[str, str] = {}
    for rel, abs_path in labels.items():
        if not rel.endswith(".txt"):
            rep.warn("stray-file", rel, "не .txt — лишний файл в labels/")
            continue
        stem = rel[len("labels/"):-4]
        label_map[stem] = abs_path
    used_stems: set[str] = set()
    stem_collisions: Counter = Counter()
    for image in image_files:
        stem = os.path.splitext(os.path.basename(image))[0]
        stem_collisions[stem] += 1
    for stem, n in stem_collisions.items():
        if n > 1:
            rep.error("labels", f"labels/{stem}.txt", f"{n} изображения с одинаковым именем «{stem}» в разных каталогах — метка неоднозначна; переименуйте файлы")

    for image, abs_img in image_files.items():
        stem = os.path.splitext(os.path.basename(image))[0]
        used_stems.add(stem)
        if stem not in label_map:
            rep.error("labels", image, f"нет файла разметки labels/{stem}.txt")
            continue
        if labels_format == "none":
            continue
        meta_rows = rows_by_image.get(image, [])
        size = size_cache.get(image) or image_size(abs_img)
        try:
            with open(label_map[stem], encoding="utf-8", errors="replace") as f:
                lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        except OSError:
            rep.error("labels", f"labels/{stem}.txt", "не удалось прочитать")
            continue
        parsed: list[tuple[int, list[float]]] = []
        for ln_no, ln in enumerate(lines, start=1):
            toks = ln.split()
            where = f"labels/{stem}.txt:{ln_no}"
            if len(toks) not in (5, 13):
                rep.error("labels", where, f"ожидается 5 чисел (class cx cy w h) или 13 (+ quad), получено {len(toks)}")
                continue
            try:
                cls = int(toks[0])
                vals = [float(t) for t in toks[1:]]
            except ValueError:
                rep.error("labels", where, f"нечисловые значения: «{ln}»")
                continue
            if cls not in LABEL_CLASSES:
                rep.error("labels", where, f"класс {cls} не из {LABEL_CLASSES}")
            if any(v < 0 or v > 1 for v in vals):
                rep.error("labels", where, "координаты должны быть нормированы в [0, 1]")
            parsed.append((cls, vals))
        if meta_rows and len(parsed) != len(meta_rows):
            rep.warn("labels", f"labels/{stem}.txt", f"{len(parsed)} строк в разметке, но {len(meta_rows)} строк в meta.csv для {image}")
        # Сверка классов и bbox с meta.csv (по лучшему IoU).
        if size and parsed and meta_rows:
            img_w, img_h = size
            for cls, vals in parsed:
                cx, cy, w, h = vals[:4]
                lb = ((cx - w / 2) * img_w, (cy - h / 2) * img_h, w * img_w, h * img_h)
                best, best_row = 0.0, None
                for r in meta_rows:
                    bb = parse_int_list(r.get("bbox", ""), 4)
                    if bb is None:
                        continue
                    iou = bbox_iou(lb, tuple(map(float, bb)))
                    if iou > best:
                        best, best_row = iou, r
                if best < 0.5:
                    rep.warn("labels", f"labels/{stem}.txt", f"bbox класса {cls} не совпадает ни с одним bbox из meta.csv (лучший IoU {best:.2f})")
                elif best_row is not None and LABEL_CLASSES.get(cls) != best_row.get("plate_type"):
                    rep.warn("labels", f"labels/{stem}.txt", f"класс {cls} ({LABEL_CLASSES.get(cls)}) ≠ plate_type={best_row.get('plate_type')} в meta.csv:{best_row['_line']}")
    for stem in label_map:
        if stem not in used_stems:
            rep.warn("labels", f"labels/{stem}.txt", "разметка без изображения")

    # ---- 7. Хэши: дубликаты и отладочный набор ----------------------------------
    duplicates = 0
    forbidden_hits = 0
    if image_files and (not skip_hashes or forbidden_hashes):
        by_hash: dict[str, list[str]] = defaultdict(list)
        for rel, abs_path in image_files.items():
            try:
                by_hash[md5_of_file(abs_path)].append(rel)
            except OSError:
                rep.error("image-file", rel, "не удалось прочитать файл")
        for h, paths in by_hash.items():
            if len(paths) > 1:
                duplicates += len(paths) - 1
                rep.warn("duplicate-image", paths[0], f"побайтовые дубликаты: {', '.join(paths[1:])}")
            if forbidden_hashes and h in forbidden_hashes:
                forbidden_hits += 1
                for p in paths:
                    rep.error("debug-set", p, "изображение из отладочного набора организаторов (п. 6.3.5)")

    # ---- 8. Объёмы --------------------------------------------------------------
    for t, need in MIN_REAL_IMAGES.items():
        have = len(real_images_by_type[t])
        if have < need:
            rep.warn("volume", "<root>", f"реальных изображений {t}: {have}, рекомендуется не менее {need}")
    for t, need in MIN_REAL_UNIQUE_PLATES.items():
        have = len(real_plates_by_type[t])
        if have < need:
            rep.warn("volume", "<root>", f"уникальных полностью читаемых ГРЗ {t} на реальных фото: {have}, рекомендуется не менее {need}")
    if len(synthetic_images) < MIN_SYNTHETIC_IMAGES:
        rep.warn("volume", "<root>", f"синтетических изображений: {len(synthetic_images)}, рекомендуется не менее {MIN_SYNTHETIC_IMAGES}")

    # Синтетика должна воспроизводиться: чаще всего забывают seed.
    if os.path.isdir(gen_dir):
        seed_found = False
        for dirpath, _d, files in os.walk(gen_dir):
            for f_name in files:
                if f_name.endswith(".py"):
                    try:
                        with open(os.path.join(dirpath, f_name), encoding="utf-8", errors="replace") as f:
                            if re.search(r"seed", f.read(), re.IGNORECASE):
                                seed_found = True
                    except OSError:
                        pass
        if not seed_found:
            rep.warn("generator", "generator/", "в скриптах не найдено слово «seed» — набор должен воспроизводиться с фиксированным random seed (п. 6.1)")

    # ---- 9. Статистика ----------------------------------------------------------
    rep.stats = {
        "validator_version": VERSION,
        "dataset_root": root,
        "images_total": len(image_files),
        "images_real": len(real_images),
        "images_synthetic": len(synthetic_images),
        "images_without_meta": len([p for p in image_files if p not in rows_by_image]),
        "meta_rows": len(rows),
        "plates_by_type": dict(type_counter),
        "real_images_by_type": {t: len(s) for t, s in real_images_by_type.items()},
        "real_unique_plates_by_type": {t: len(s) for t, s in real_plates_by_type.items()},
        "conditions": dict(cond_counter),
        "sources_top": dict(source_counter.most_common(15)),
        "licenses": dict(license_counter.most_common(15)),
        "duplicate_images": duplicates,
        "debug_set_hits": forbidden_hits,
        "errors": len(rep.errors),
        "warnings": len(rep.warnings),
    }
    return rep


# ---------------------------------------------------------------------------
# Вывод отчёта
# ---------------------------------------------------------------------------

def render_report(rep: Report, *, max_per_code: int = 30) -> str:
    out: list[str] = []
    s = rep.stats
    out.append(f"validate_dataset.py v{VERSION} — Волга-IT 2026, полуфинал, номинация «ИИ и анализ данных»")
    out.append(f"Датасет: {s.get('dataset_root')}")
    out.append("")
    out.append("== Статистика ==")
    out.append(f"Изображений: {s.get('images_total')} (реальных {s.get('images_real')}, синтетических {s.get('images_synthetic')}, без разметки в meta {s.get('images_without_meta')})")
    out.append(f"Строк в meta.csv: {s.get('meta_rows')}")
    out.append(f"Знаков по типам: {s.get('plates_by_type')}")
    out.append(f"Реальных изображений по типам: {s.get('real_images_by_type')}  (рекомендуется {MIN_REAL_IMAGES})")
    out.append(f"Уникальных читаемых ГРЗ (реальные): {s.get('real_unique_plates_by_type')}  (рекомендуется {MIN_REAL_UNIQUE_PLATES})")
    out.append(f"Условия съёмки: {s.get('conditions')}")
    out.append(f"Лицензии: {s.get('licenses')}")
    out.append(f"Источники (топ): {s.get('sources_top')}")
    out.append(f"Побайтовых дубликатов: {s.get('duplicate_images')}; совпадений с отладочным набором: {s.get('debug_set_hits')}")
    out.append("")

    grouped: dict[tuple[str, str], list[Issue]] = defaultdict(list)
    for issue in rep.issues:
        grouped[(issue.level, issue.code)].append(issue)

    for level in ("ERROR", "WARNING"):
        items = [(k, v) for k, v in grouped.items() if k[0] == level]
        total = sum(len(v) for _k, v in items)
        out.append(f"== {level}: {total} ==")
        for (_lvl, code), issues in sorted(items, key=lambda kv: -len(kv[1])):
            out.append(f"-- {code} ({len(issues)})")
            for issue in issues[:max_per_code]:
                out.append("   " + issue.fmt())
            if len(issues) > max_per_code:
                out.append(f"   ... ещё {len(issues) - max_per_code} (см. JSON-отчёт)")
        out.append("")

    out.append("Напоминание (проверяется жюри вручную): лица людей размыты; нет кадров, где человек — смысловой центр;")
    out.append("материалы получены легально (п. 6.3); разметка не сфальсифицирована (п. 8).")
    out.append("")
    verdict = "НЕ ПРОЙДЕНА — исправьте ошибки" if rep.errors else "ПРОЙДЕНА" + (" с предупреждениями" if rep.warnings else "")
    out.append(f"ИТОГ: валидация {verdict}. Ошибок: {len(rep.errors)}, предупреждений: {len(rep.warnings)}.")
    return "\n".join(out)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Валидатор датасета ГРЗ (Волга-IT 2026)")
    parser.add_argument("dataset", nargs="?", help="корень датасета (каталог с meta.csv)")
    parser.add_argument("--report", default="validation_report.txt", help="текстовый отчёт (по умолчанию validation_report.txt рядом со скриптом; «-» — только stdout)")
    parser.add_argument("--json", default=None, help="JSON-отчёт с полным списком сообщений")
    parser.add_argument("--forbidden-hashes", default=None, help="файл с md5 изображений отладочного набора (по одному в строке)")
    parser.add_argument("--no-hash", action="store_true", help="не считать md5 (быстрее, без проверки дубликатов)")
    parser.add_argument("--labels-format", choices=("yolo", "none"), default="yolo", help="формат labels/*.txt (none — проверять только наличие файлов)")
    parser.add_argument("--max-per-code", type=int, default=30, help="сколько сообщений одного типа печатать")
    parser.add_argument("--hash-dir", default=None, help="служебный режим: вывести md5 всех изображений каталога и выйти")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Windows-консоль с cp866/cp1251 не переваривает часть символов отчёта.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if args.hash_dir:
        for dirpath, _d, files in os.walk(args.hash_dir):
            for name in sorted(files):
                if name.lower().endswith(IMAGE_EXTS):
                    p = os.path.join(dirpath, name)
                    print(f"{md5_of_file(p)}  {name}")
        return 0

    if not args.dataset:
        parser.error("укажите путь к датасету")
    if not os.path.isdir(args.dataset):
        print(f"Каталог не найден: {args.dataset}", file=sys.stderr)
        return 2

    forbidden: set[str] | None = None
    if args.forbidden_hashes:
        with open(args.forbidden_hashes, encoding="utf-8") as f:
            forbidden = {ln.split()[0].strip().lower() for ln in f if ln.strip()}

    rep = validate(args.dataset, forbidden_hashes=forbidden, skip_hashes=args.no_hash,
                   labels_format=args.labels_format)
    text = render_report(rep, max_per_code=args.max_per_code)
    print(text)
    if args.report and args.report != "-":
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nОтчёт сохранён: {os.path.abspath(args.report)}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"stats": rep.stats, "issues": [i.__dict__ for i in rep.issues]}, f, ensure_ascii=False, indent=2)
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
