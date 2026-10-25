# Генераторы: текущие сцены и историческая PHASE B

Для текущего сдаваемого формата используется `dataset/generator/scenes_current.py`: собственные полные сцены с автомобилями, физической геометрией и маской type1b `LLDDDRR/RRR`. Проверенный synthetic-only packet: `D:/volga/generated_data/organizer_synthetic_5000_v3_20261015`,5000 изображений/5706 объектов. Он содержит автономный `generator/reproduce.py`; полный цикл и ограничения описаны в [current_synthetic_review.md](../../docs/current_synthetic_review.md). Текущие real targets и internal/organizer форматы: [dataset README](../README.md).

Ниже сохранено историческое описание crop-генератора PHASE B. Его маска type1b и правила регионов относятся к `legacy_v2`; новый актуальный генератор выбирается явно. Старые пиксели/коллекции не мигрируются и не заменяются. Отдельная неизменённая копия: [legacy_synthetic_phase_b.md](../../docs/legacy_synthetic_phase_b.md).

---

# Synthetic generator V2 — PHASE B

Offline **plate crops** using phase-A metadata, pixel ground-truth checks and
plate-identity splits. Dependencies: NumPy/OpenCV/Pillow. No model weights,
system fonts, external images or downloads. No PHASE C stitching or training.

## Layouts

- type1: 520×112 canonical units, white, one row with a separate region field.
- type1a: 290×170, white, two actual rows: A123 / BC+region, never pre-stitched.
- type1b: 520×112, yellow, the user-selected **LDDDLLRR/RRR** mask. This differs
  from the organizer's newer LLDDDRR/RRR rule; the official scorer is unchanged.

Letters ABEKMHOPCTYX, digits 0–9; 2/3-digit regions, first digit of a 3-digit
region 1/2/7. # is not printed or used to hide rendering defects. No current
regional-code-list validity is claimed.

The authored glyphs in assets/ are an **approximation, not a certified GOST font**.
All semantic glyphs were visually checked and their frozen PNG hashes tested.
Auxiliary RUS text uses Pillow's bundled text font. Authored assets/data: CC BY4.0.
Yellow signs use a flagless historical appearance with the user-selected mask;
this is not a claim of conformity to the current road-sign standard.

## Generate a new version

Run from D:\volga\volga-anpr:

```powershell
.\.venv\Scripts\python.exe -X utf8 tools\generate_synthetic.py --seed 42 --count 600 --plate-type all --variants 2 --config configs\synthetic_v2.json --output D:\volga\generated_data\synthetic_new_version
```

count means images; variants groups views of the same plate. `all` cycles identities
across types; small arbitrary counts may be unequal. Region lengths alternate.
A repeated seed/config/runtime gives identical images/metadata/manifests. Existing
output is refused. `--profile clean` disables geometry randomness and degradation.

Output is a new canonical collection: images/synthetic/, labels/, meta.csv,
splits/, generator/reference/ clean raster proofs, generator/footprints/ masks,
local glyph assets/config and manifests. The main phase-A scaffold is preserved.
Versions are not automatically combined. All views of a plate share one split.

is_synthetic=1, annotation_kind=ocr_crop, **is_vehicle=unknown**. A neutral crop
background cannot establish vehicle context. Bbox/quad describe the projected plate
in output pixels, TL/TR/BR/BL; photometric effects do not change physical corners.
Provenance includes cell boxes, homography, exact effect parameters, hashes, seed,
config/source/runtime versions and rejected attempts. These are not full-scene
detector/context/E2E ground truth. meta.csv is authoritative; labels/ is reserved.

## Bounded domain randomization

All ranges are in configs/synthetic_v2.json and validated before writing. Default:
rotation ±6°, modest perspective/translation/scale and at most three effects.
Supported: brightness/contrast/gamma, motion/disk-defocus/Gaussian blur, sensor
noise, JPEG, shadow/glare, rain/snow/dirt, border-only partial occlusion, exposure
and color temperature. Texture/background are procedural.

Low resolution is explicitly **high resolution → downscale → JPEG at reduced
size → upscale**. Weather/occlusion avoid target ink with a margin. Occlusion is
localized to a physical plate edge. Full target occlusion/unreadable samples are
excluded instead of receiving complete labels. This is a conservative starter
corpus. clean_fraction is a probability, not an exact image count.

## Pixel ground-truth gate

```powershell
.\.venv\Scripts\python.exe -X utf8 tools\verify_synthetic.py --dataset D:\volga\generated_data\synthetic_new_version --report reports\phase_b\ground_truth_validation.json
.\.venv\Scripts\python.exe -X utf8 tools\validate_local.py --dataset D:\volga\generated_data\synthetic_new_version --report reports\phase_b\dataset_validation.json
.\.venv\Scripts\python.exe -X utf8 tools\preview_synthetic.py --dataset D:\volga\generated_data\synthetic_new_version --output reports\phase_b\preview.png --indices 0 1 2 3 4 5 --overlay
```

The verifier reads actual raster glyphs against frozen masters, with candidates
restricted only by mask position. It compares decoded text to CSV on both clean
proofs and inverse-warped final images; it does not return the supplied label.
It checks row/cell order, type color/dimensions, bbox/quad/homography/footprint and
hashes. Negative controls change CSV text, erase glyphs, blank final pixels,
mislabel a type and corrupt geometry. Persistent failure after up to 12 recorded
augmentation attempts publishes no collection. No ANPR model is used as an oracle.

This verifies renderer/label consistency for the authored font, **not real OCR
accuracy or photographic realism**. Visual review of actual output, all glyphs,
both region lengths, all types and representative degradations is still required.
The accepted candidate and evidence are documented in ../../docs/synthetic_phase_b.md
and reports/phase_b/. Do not consume earlier rejected versions for training.
