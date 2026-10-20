# Установка и запуск рабочего концепта

## Среда

Проверено локально: Windows, Python3.12.14, Polars2.0, PyTorch2.11/CUDA12.8,
RTX3050 Laptop4GiB,15,41GiB RAM. Версии зафиксированы в `requirements*.txt`.
На других ОС может потребоваться установка подходящей сборки PyTorch.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
uv pip install --python .venv/Scripts/python.exe -r requirements-neural.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
```

## Проверка без датасетов

```powershell
.venv/Scripts/python.exe -m pytest tests/ -m "not real_data" --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80
```

Выборка `not real_data` содержит176 fixture tests;4 real-data tests требуют
исходных файлов. После их размещения доступен полный suite:

```powershell
.venv/Scripts/python.exe -m pytest tests/ --cov=src --cov-branch --cov-fail-under=80
```

## Подготовка данных

Источники и basenames — [справочник](raw_data_structure.md) и `configs/datasets/`.
Файлы располагаются в `data/<dataset>/raw/`, результаты — в `processed/`.
Raw не изменяется. Data и experiment files исключены из Git.

Для быстрого диска системного temp можно выбрать:

```powershell
$env:PIT_RECSYS_RUNTIME_STORAGE='system_temp'
```

По умолчанию временное хранилище — `.venv/runtime`. Native spill имеет время
жизни процесса, stage tables очищаются после этапа. После принудительной
остановки/выключения автоматическая очистка всех временных файлов не гарантируется.

## Воспроизведение технического smoke на Retailrocket

```powershell
$py = '.venv/Scripts/python.exe'
& $py scripts/run_generator.py --dataset retailrocket --row-limit 20000
$processed = 'data/retailrocket/processed/smoke/users_mod_1_rows_20000'
& $py scripts/run_baseline.py --dataset retailrocket --processed-root $processed --smoke
foreach ($model in @('adv1','adv2','adv3','adv4')) {
    & $py scripts/run_advanced.py --model $model --dataset retailrocket --processed-root $processed --smoke
}
foreach ($mode in @('soft_target_distillation','score_feature_stacking')) {
    & $py scripts/run_distillation.py --mode $mode --dataset retailrocket --processed-root $processed --smoke
}
& $py scripts/run_evaluation.py --dataset retailrocket --mode smoke --processed-root $processed --include-distillation
```

Smoke явно сокращает training protocol и использует один seed42. Его результаты
не заменяют полноценный benchmark. Перед чтением артефактов проверяются
`audit/generation_state.parquet` со статусом complete и manifest checksums.

## Отчёты и notebook

Для таблицы full, в которой незавершённые cells остаются пустыми:

```powershell
& $py scripts/run_evaluation.py --mode full --include-distillation --report docs/experiment_results.md --smoke-summary experiments/summary/smoke
```

Эта команда перезаписывает отчёт локальным срезом. Публикуемый итоговый Markdown
содержит зафиксированные результаты завершения концепта; перед пересчётом его
можно сохранить отдельно. `--strict` завершает collector с ошибкой при неполной матрице.

Для notebook02 сначала создайте smoke summary предыдущими командами и full summary
collector. Затем:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-notebooks.txt
& $py -m ipykernel install --prefix .venv --name pit-recsys --display-name 'PIT Recsys (.venv)'
& $py scripts/build_results_notebook.py
```

Notebook02 настроен на smoke-контракт Retailrocket20k/12 test groups и проверяет
его явно. Notebook01 читает ограниченные raw prefixes всех четырёх datasets.
После выполнения notebooks очищайте outputs перед публичным commit: они могут
содержать локальные paths и таблицы журналов.

## Полный исследовательский протокол

Он описан в [experiment_plan.md](experiment_plan.md) и не завершён в этой поставке.
`scripts/run_full_experiments.py` запускает большие задания последовательно;
`--generation-only` ограничивает его подготовкой данных, `--skip-generation`
разрешает обучение только на готовых datasets. По умолчанию —20 trials и3 seeds.
Такой запуск требует отдельного временного/ресурсного бюджета и не является
быстрым демонстрационным сценарием рабочего концепта.

Синтетические варианты: `scripts/run_synthetic_ablations.py`.
Objective/Fusion component variants: `scripts/run_model_ablations.py`.
`--collect-only` собирает статусы без запуска обучения. Generated configs,
checkpoints, journals и tables остаются в игнорируемом `experiments/`.
