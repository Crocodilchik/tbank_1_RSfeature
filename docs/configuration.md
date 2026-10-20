# Контракт конфигурации

Версия: 1.0
Статус: approved
Дата последнего обновления: 2026-10-20

Единственные нормативные PIT defaults — [политика §13](point_in_time_policy.md).
Этот документ определяет владение ключами YAML и правила валидации.
Исполняемые конфиги находятся в configs/, Pydantic loader — в src/config/loader.py.
Примеры ниже описывают реализованный контракт рабочего концепта.

## 1. Владение ключами

| Файл относительно project root | Верхний ключ файла | В собранном объекте |
|---|---|---|
| configs/main_config.yaml | project, runtime, point_in_time_policy | Одноименные; PIT здесь без evaluation/sequences/datasets |
| configs/splits_config.yaml | splits | splits |
| configs/evaluation_config.yaml | evaluation | point_in_time_policy.evaluation |
| configs/sequence_config.yaml | sequences | point_in_time_policy.sequences |
| configs/datasets/{name}.yaml | dataset | point_in_time_policy.datasets[name] |

main_config содержит остальные разделы PIT §13 ровно один раз (version,
default_mode, timestamp_handling, metadata, target_encoding, graph, synthetic).
Дублировать evaluation/sequences/datasets в нем нельзя. Dataset-файл расширяет
PIT dataset settings только путями/схемой источников и adapter settings;
primary/repeat задаются единожды в evaluation_config, не копируются в dataset.
models.yaml не требуется: модели и experiment budgets лежат в main_config.runtime.
Собранный объект включает все PIT defaults §13 без скрытых library defaults.

Loader последовательно читает main → splits → evaluation → sequences → выбранный
dataset, размещает по указанной карте и валидирует. Это сборка независимых
разделов, не recursive merge с правилом «последний победил». Duplicate YAML keys,
неизвестные поля и несовпадающий dataset name — ошибки. Параметры command line
только dataset/model/run_id и явно разрешенные ablation overrides;
каждый override сохраняется в resolved config. Противоречащий PIT override fatal.

## 2. Пути и runtime

```yaml
project:
  root: .
  data_dir: data
  experiments_dir: experiments
  schema_version: '1.0'
runtime:
  seed: 42
  training_seeds: [42, 43, 44]
  chunk_rows: 1000000
  device: auto
  deterministic: true
  training:
    hpo_trials: 20
    max_trial_hours: 2
    max_neural_epochs: 50
    max_booster_iterations: 2000
    neural_patience: 3
    booster_patience: 100
    neural_batch_size: 256
    gnn_fanout: [15, 10]
    inner_train_quantiles: [0.5, 0.7, 0.85]
    valid_hpo_fraction: 0.6
  models:
    baseline:
      backend: catboost
      windows_days: [1, 7, 14, 30]
    adv1:
      backend: catboost
      windows_days: [1, 3, 7, 14, 30]
      max_feature_depth: 2
      max_candidate_features: 500
      selected_features: 50
    adv2:
      backend: graphsage
      hidden_size: 64
      num_layers: 2
      loss: sampled_bce
    adv3:
      hidden_size: 64
      num_layers: 2
      attention_heads: 2
      loss: sampled_bce
    adv4:
      graph_feature_mode: temporal_oof
      hidden_size: 64
      lambda_graph: 0.1
    distillation:
      mode: soft_target_distillation
      alpha: 0.5
```

project.root='.' означает родителя configs, не process cwd. Все относительные пути
разрешаются от этого root через pathlib.Path. Абсолютный root допускается в
локальном override, не подшивается как E:/... в переносимый config.
Runtime device auto разрешается в конкретный CPU/CUDA с записью в manifest.
Hardware probe выбирает безопасный batch size до HPO; фиксируется всем сопоставимым
runs, произвольное молчаливое урезание данных не разрешено.

## 3. Splits: полные полуоткрытые UTC границы

```yaml
splits:
  timezone: UTC
  retailrocket:
    method: timestamp_lower_quantile
    train_quantile: 0.8
    valid_quantile: 0.9
  movielens:
    method: timestamp_lower_quantile
    train_quantile: 0.8
    valid_quantile: 0.9
  taobao:
    method: fixed_dates
    window_start: '2017-11-24T00:00:00Z'
    train_end: '2017-12-02T00:00:00Z'
    valid_end: '2017-12-03T00:00:00Z'
    window_end: '2017-12-04T00:00:00Z'
  yoochoose:
    method: fixed_dates
    window_start: '2014-04-01T00:00:00Z'
    train_end: '2014-09-01T00:00:00Z'
    valid_end: '2014-09-15T00:00:00Z'
    window_end: '2014-10-01T00:00:00Z'
```

Квантиль lower: sorted timestamps[floor(q*(n-1))]; train<T1,
valid `[T1,T2)`, test>=T2. Resolved seconds фиксируются в manifest. Taobao UTC
seconds: window_start1511481600, T1=1512172800, T2=1512259200,
window_end1512345600; проверено по полному timestamp-столбцу.

## 4. Dataset-файлы

Каждый dataset-файл имеет `dataset.name`, `raw_dir: data/{name}/raw`,
`processed_dir: data/{name}/processed`, `files` (mapping role→basename),
`adapter` и PIT availability settings из §13. Регистр директорий lowercase.

| Dataset | files roles → фактические basenames |
|---|---|
| retailrocket | events→events.csv; properties1→item_properties_part1.csv; properties2→item_properties_part2.csv; categories→category_tree.csv |
| taobao | events→UserBehavior.csv |
| yoochoose | clicks→yoochoose-clicks.dat; buys→yoochoose-buys.dat; official_test→yoochoose-test.dat; source_readme→dataset-README.txt |
| movielens | ratings→rating.csv; movies→movie.csv; links→link.csv; tags→tag.csv; genome_scores→genome_scores.csv; genome_tags→genome_tags.csv |

adapter.header=true для RetailRocket/MovieLens; false для Taobao/Yoochoose;
separator=',', encoding='utf-8'. Raw types/мэппинг в соответствующем
data/{name}/README.md. RetailRocket timestamp_unit=milliseconds,
Taobao=seconds, Yoochoose=iso8601_utc, MovieLens=datetime_utc_assumed.
Параметры фильтра/dedup/category conflict не скрыты в кодовых defaults:
adapter.invalid_required_field=error, adapter.exact_duplicates=keep_first,
adapter.metadata_conflict=error; специализированные допустимые случаи unknown
и Yoochoose context conflict явно определены generator_architecture.md §2.

Personas settings при ablation: k32, vector_hash_dim256, batch_size4096,
uncertainty_quantiles[0.5,0.85], min_prior_interactions5,
mixing_probabilities[0,0.25,0.5], snapshot_granularity=daily.
Co-purchase primary=false, max_partners20; synthetic_neighbors_per_node10.
Если persona-related negative strategy не имеет cluster — category/random
fallback с счетчиком, без обращения к future-fit cluster.

## 5. Cross-field checks Pydantic loader

- version/schema_version известны; unknown extra fields и duplicate keys запрещены.
- default_mode event_level_pit; unified_unit seconds, timezone UTC,
  strict_less_than=true, reliable_intra_timestamp_order=false.
- Даты упорядочены, q_train<q_valid, splits после очистки непустые.
- candidate_pool_size=negatives_per_group+1=100; target strategy совпадает
  с sequence eval strategy; train_strategy leave_one_out.
- Все четыре dataset keys primary/repeat/fallback заданы; predicate Yoochoose
  click; official supplementary только click и late window.
- max_history_length положительно и <=50 default; masks required; разрешено
  прошлое появление того же item по policy, не сам target event.
- synthetic used_for=train_only; strict future_conflict_policy только
  keep_as_noise_audit_only и sample_weight1; difficulty mix неотрицательна и sum1.
- genome_scores_enabled=true требует diagnostic mode; persona features имеют
  rolling provenance, не static-assumed behavioral fields.
- Paths raw/processed не совпадают; чтение inputs только raw, writes данных
  только processed; traversal вне project root запрещен, кроме явно разрешенных
  read-only source roots. Windows и Python path roundtrip проверяются тестом.
- Trial/batch/fanout/chunk ограничения положительны; hidden_size делится на heads.
  Автоматическое уменьшение объема dataset ради OOM меняет run mode на smoke.

## 6. Воспроизводимость

Resolved config сериализуется в canonical JSON: sorted keys, UTF-8, Path в
project-relative POSIX форме, UTC даты в ISO; SHA-256 всего объекта. Absolute
project root исключается из semantic hash и сохраняется отдельно в environment,
чтобы перенос каталога не менял смысл эксперимента. Code revision, package versions,
NumPy generator, raw SHA-256 и resolved split seconds фиксируются отдельно.
Processed manifest — Parquet; experiment run manifest — Parquet вне processed, вложенная environment/provenance
структура хранится канонической JSON-строкой.

Machine-specific cache backend выбирается через `PIT_RECSYS_RUNTIME_STORAGE`:
`project` по умолчанию либо `system_temp` для native SQLite/Polars scratch на
системном диске. Внешний произвольный path не задаётся; используется owned
project namespace в системном temp. Backend и temp root записываются в environment,
не меняя semantic YAML hash. Итоговые dataset paths и HPO budgets задаёт конфиг.

Runtime разделяет временные таблицы этапа и native spill. Таблицы SQLite/Parquet
удаляются при выходе из scope. Native Polars spill хранится в отдельном owned
каталоге до завершения процесса: Polars может кэшировать первый spill path между
последовательными этапами. Один процесс переиспользует native каталог внутри
выбранного runtime parent; стандартный TemporaryDirectory finalizer удаляет его
при штатном завершении. TMP/TEMP и Polars env восстанавливаются после scope.

Полная очередь при возобновлении сверяет незавершённые записи журнала с временем
последней загрузки ОС. Только попытки, начатые до этой загрузки, автоматически
получают статус interrupted под эксклюзивной блокировкой журнала. Их последние
wall/CPU/RSS сохраняются как неполные измерения; время завершения и exit code
остаются неизвестными. Незавершённый generator state с тем же config hash
помечается failed перед повторной генерацией. Завершённые артефакты и попытки
текущей загрузки не меняются; готовность данных по-прежнему проверяет manifest.

## История изменений

- 2026-10-19 — 1.0 draft: распределение YAML, UTC boundaries, dataset paths,
  строгая загрузка, defaults и cross-field validation для итерации1 Этапа3.

- 2026-10-19 — пользователь подтвердил план и переход к реализации; состояние выполнения отражено в project_status.md.
