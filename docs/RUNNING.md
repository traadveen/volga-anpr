# Установка и запуск из Git

Главная инструкция и точные команды находятся в ../README.md.

После clone загрузите веса командой git lfs pull. Создайте .venv с Python 3.12,
установите выбранный CPU/CUDA вариант PyTorch 2.6.0 и requirements.txt.
Сетевой доступ нужен при получении репозитория, LFS и библиотек.
Распознавание использует локальные веса и блокирует Python TCP/DNS/UDP.

Основная команда из корня:
    .venv\Scripts\python.exe run.py --input "C:\photos" --output "output\result.csv" --device cuda --show

Для CPU: --device cpu. Для запуска без окон уберите --show.
Для тестовых процедурных сцен: run_examples.cmd; устройство задаётся VOLGA_DEVICE.

Ошибка Candidate file changed означает несовпадение файлов с манифестом.
После clone сначала проверьте git lfs pull. Файлы не должны быть LFS-указателями.
.gitattributes отключает преобразование переносов строк.

Результаты защищены блокировкой и журналом восстановления.
После аварийного завершения, когда прежний процесс уже остановлен:
    .venv\Scripts\python.exe solution/tools/output_transaction.py output/result.csv --recover

Согласованность успешного результата:
    .venv\Scripts\python.exe solution/tools/output_transaction.py output/result.csv

Переносимость на другие ОС и задержка на GTX 1050 Ti отдельно не подтверждены.
