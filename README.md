# Автоматизация признаков рекомендательных систем

**Статус: Рабочий концепт.** Зафиксирован 20 октября 2026 года.

Исследовательский Python-проект для сравнения GBDT на ручных признаках с AutoML,
GraphSAGE, Sequential Two-Tower и Graph-Sequential Fusion. Поддерживаются
Retailrocket, Taobao, Yoochoose и MovieLens 20M. Признаки, графы и истории
строятся по данным строго раньше времени запроса; модели используют общие
evaluation pools.

Реализованы генератор Parquet-артефактов, пять моделей, два варианта дистилляции,
метрики и проверки воспроизводимости. Все модели прошли технические smoke-прогоны
на Retailrocket. Полная научная проверка преимущества Advanced не завершена.

## Полученные результаты

- Проверка публикации:176 fixture tests в чистой копии и4 real-data tests прошли;
  покрытие fixture-run с учётом ветвей —95,73%.
- Полностью подготовлены Retailrocket: 2 755 641 событие, и MovieLens:
  20 000 263 события. У каждого прошли все 35 quality gates.
- На общем smoke test pool Retailrocket из 12 групп Baseline и AutoML получили
  NDCG@10 = 1,0; GNN — 0,115571, Sequential — 0,056326, Fusion — 0,062178.
- Полная матрица моделей: 0/20 завершённых конфигураций. Большие запуски остановлены;
  фоновые вычисления при публикации не выполняются.

Указанные smoke-метрики относятся к одному seed и небольшому набору. Подробные
метрики, отрицательные результаты, стоимость и границы выводов приведены в
[итоговом отчёте](docs/experiment_results.md).

## Быстрый старт

Проверенная среда: Python 3.12, Windows, PyTorch/CUDA. Ядро требует Python >=3.10;
переносимость полного набора зависимостей на другие платформы отдельно не проверялась.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
uv pip install --python .venv/Scripts/python.exe -r requirements-neural.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
.venv/Scripts/python.exe -m pytest tests/ -m "not real_data"
```

Подробности установки — в [инструкции запуска](docs/usage.md).
Исходные датасеты скачиваются отдельно:
ожидаемые имена файлов и источники описаны в [справочнике данных](docs/raw_data_structure.md).

Ограниченный прогон генератора после размещения исходных файлов:

```powershell
.venv/Scripts/python.exe scripts/run_generator.py --dataset retailrocket --row-limit 20000
```

Подробный [smoke-сценарий](docs/usage.md) включает обучение, сбор метрик и notebook.
Полный исследовательский протокол требует существенно больше времени и ресурсов;
его команды запускаются только явно.

## Документация

- [Статус и границы рабочего концепта](docs/project_status.md).
- [Установка, запуск, тесты и воспроизведение](docs/usage.md).
- [Фактически полученные результаты](docs/experiment_results.md).
- [Point-in-time policy](docs/point_in_time_policy.md) — приоритетный контракт.
- [Архитектура генератора и схемы](docs/generator_architecture.md).
- [Архитектуры моделей](docs/recsys_architecture.md).
- [Конфигурация](docs/configuration.md) и [полный исследовательский протокол](docs/experiment_plan.md).
- [Источники и форматы данных](docs/raw_data_structure.md).

## Состав репозитория

`src/` — реализация; `scripts/` — CLI; `configs/` — переносимые YAML;
`tests/` — проверки; `notebooks/` — исходники анализа без сохранённых outputs;
`docs/` — пользовательская и техническая документация.

Датасеты, модельные веса, таблицы экспериментов, окружения, временные файлы и
журналы разработки не распространяются с исходниками. Локальные generated
артефакты сохраняются в `data/` и `experiments/`; `.gitignore` исключает их из Git.
Размеры и результаты в отчёте описывают выполненные локальные прогоны;
для независимой проверки требуется заново получить данные и запустить команды.
