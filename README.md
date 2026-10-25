# Volga ANPR

Сборка FORMAT-CONSISTENCY-20261025 распознаёт российские номера на отдельных фотографиях:
белые однострочные type1, белые двухстрочные type1a и жёлтые type1b.

В репозиторий включены код, три checkpoint распознавания, датасет, генератор,
данные и материалы финального дообучения. Веса и массивы NPZ хранятся через Git LFS.
Распознавание работает без сети. Установка Python, библиотек и загрузка LFS
выполняются заранее с доступом к интернету.

## Получение файлов

Для весов и данных необходим Git LFS. После клонирования выполните:

    git lfs install
    git lfs pull

Веса должны быть настоящими бинарными файлами. Проверка целостности:

    py -3.12 verify_package.py

Все файлы сохраняются без преобразования переносов строк: это требуется для
контрольных сумм. Служебная папка .git и локальное окружение не входят в манифест.

## Установка на Windows

Проверено на Windows x64 с Python 3.12. Из корня репозитория:

    py -3.12 -m venv .venv

Выберите один вариант PyTorch.

CPU:

    .venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

NVIDIA CUDA 12.4:

    .venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

Затем:

    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe -m pip check

Точные ранее использованные наборы версий сохранены в dependency-locks.
Установка по сети на новом компьютере отдельно не повторялась; код этой сборки
проверен на CPU и CUDA в существующих закреплённых окружениях.

## Запуск из CMD

Пример для NVIDIA:

    .venv\Scripts\python.exe run.py --input "C:\photos" --output "output\result.csv" --device cuda --show

Для CPU замените устройство:

    .venv\Scripts\python.exe run.py --input "C:\photos" --output "output\result.csv" --device cpu --show

--show открывает каждое фото с рамками и итоговыми номерами. Любая клавиша
показывает следующее фото; Esc продолжает распознавание без окон.
Для показа нужны Tkinter и Pillow. Без --show графический интерфейс не используется.

Для трёх процедурных проверочных сцен выполните run_examples.cmd.
По умолчанию используется .venv и CPU. Можно задать VOLGA_PYTHON полным путём
к установленному Python, VOLGA_DEVICE значением cpu или cuda и добавить --show.

Вход: файл либо папка JPG/JPEG/PNG/BMP без обхода подпапок.
Выход: UTF-8 CSV с разделителем ; и полями
image;plate_num;plate_type;confidence. Символ # означает неизвестную позицию.
Confidence не является вероятностью правильности всего номера.
Рядом сохраняются trace, errors и manifest в JSON.

## Проверки и результаты

    .venv\Scripts\python.exe -m unittest discover -s solution/tests -p "test_*.py" -v
    .venv\Scripts\python.exe solution/tools/verify_candidate.py

32 проверки прошли. На public debug обычные номера улучшились с 6/15 до 9/15,
двухстрочные сохранили 78/100. На REAL VAL жёлтые сохранили 32/61.
Эти наборы ранее использовались в разработке и не являются независимым тестом.
Валидатор официального датасета: 0 ошибок и 113 предупреждений.

Короткий замер на GTX 1060 6GB дал среднее 98,10 мс, p95 121,00 мс без чтения,
декодирования, загрузки моделей и ожидания в окне.
Требование скорости на GTX 1050 Ti не подтверждено.

## Состав и документация

- solution: код, конфигурация и checkpoint распознавания.
- dataset: изображения, разметка, источники, генератор и отчёт валидатора.
- training: рецепты, входы и начальные веса финальной адаптации.
- docs/TECHNICAL_REPORT.md: устройство, показатели и ограничения.
- docs/explanatory_note.pdf: пояснительная записка.
- docs/REPOSITORY.md: ветка публикации и команды Git.
- evidence: сохранённые проверки и метрики.
- licenses: условия сторонних компонентов.

Документ о разрешении на 14 объектов Kaggle пока не приложен; прежнее заявление
пользователя и список материалов сохранены в dataset/permissions/status.json.
