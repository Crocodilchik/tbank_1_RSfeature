# Датасет: MovieLens 20M (локальное Kaggle-зеркало)

Версия: 1.0
Статус: справочник рабочего концепта
Дата последнего обновления: 2026-10-19

## Источник

[Kaggle / MovieLens20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset). Авторитетное описание версии: [GroupLens](https://files.grouplens.org/datasets/movielens/ml-20m-README.html). Локально singular basenames и datetime strings, отличающиеся от официального архива.

## Состав файлов (в подпапке raw/)

| Файл | Строк данных | Размер, байт | Формат |
|---|---:|---:|---|
| `genome_scores.csv` | 11 709 768 | 214322450 | CSV, UTF-8, запятая, с заголовком |
| `genome_tags.csv` | 1 128 | 20363 | CSV, UTF-8, запятая, с заголовком |
| `link.csv` | 27 278 | 539334 | CSV, UTF-8, запятая, с заголовком |
| `movie.csv` | 27 278 | 1493648 | CSV, UTF-8, запятая, с заголовком |
| `rating.csv` | 20 000 263 | 690353377 | CSV, UTF-8, запятая, с заголовком |
| `tag.csv` | 465 564 | 21725514 | CSV, UTF-8, запятая, с заголовком |

Размеры измерены по локальным файлам 2026-10-19. Counts соответствуют
локальному raw-профилю от2026-10-15 и проверке физических
строк 2026-10-19 (для CSV вычтен заголовок). Полный повторный CSV-profile здесь
не заявляется; quotes/multiline не пересчитывались отдельным parser.
Контрольные суммы исходников записываются при запуске в source_hashes манифеста генератора; локальные inventories не публикуются.

## Схема данных

### rating.csv

| Колонка | Тип | Описание |
|---|---|---|
| userId | int64 | Пользователь |
| movieId | int64 | Фильм |
| rating | float64 | Оценка0.5..5 с шагом0.5 |
| timestamp | string | YYYY-MM-DD HH:MM:SS, локальная копия |

### movie.csv

| Колонка | Тип | Описание |
|---|---|---|
| movieId | int64 | Фильм |
| title | string | Название, часто год в скобках |
| genres | string | Список жанров через вертикальную черту |

### link.csv

| Колонка | Тип | Описание |
|---|---|---|
| movieId | int64 | Фильм |
| imdbId | int64 | Внешний ID IMDb |
| tmdbId | int64 nullable | Внешний ID TMDb,252 пропуска |

### tag.csv

| Колонка | Тип | Описание |
|---|---|---|
| userId | int64 | Пользователь, назначивший tag |
| movieId | int64 | Фильм |
| tag | string | Пользовательский текстовый tag |
| timestamp | string | YYYY-MM-DD HH:MM:SS, время назначения |

### genome_scores.csv

| Колонка | Тип | Описание |
|---|---|---|
| movieId | int64 | Фильм |
| tagId | int64 | Genome tag |
| relevance | float64 | Movie-tag relevance score |

### genome_tags.csv

| Колонка | Тип | Описание |
|---|---|---|
| tagId | int64 | Genome tag |
| tag | string | Расшифровка tag ID |

## Ключевые поля (маппинг к единой схеме проекта)

| Роль | Исходная колонка | Единое поле проекта |
|---|---|---|
| Пользователь | userId | user_id |
| Товар | movieId | item_id |
| Время | timestamp datetime | timestamp seconds UTC assumed; timestamp_ms=null |
| Событие | rating | event_type=rating_<value>, label0/1/2 |
| Жанры | genres | metadata genres:list[string] |
| Теги пользователя | tag и timestamp | observed metadata as-of<T |
| Genome | relevance/tagId | Не strict primary, diagnostic only |

Служебные event_id/source_row_id/user_event_seq/split/target_cutoff_ts добавляет
генератор по [PIT §9](../../docs/point_in_time_policy.md), не исходные файлы.
Metadata хранится отдельно от events, в частности category не дублируется на
каждую строку большого журнала.

## Маппинг событий

rating>=4→label2; rating3/3.5→1; rating<=2.5→0. Primary rating>=4;
repeat_purchase_policy=unseen_only: target не оценен доT. Eval negatives исключают
все когда-либо оцененные user movies как evaluator oracle, не признаки.
Низкий rating — явное отрицательное предпочтение, но в источнике нет отрицательных
показов. Explicit low-rating classification — отдельная auxiliary task,
не скрытая замена sampled ranking. Fallback rating3–3.5 отдельно от primary.

## Известные аномалии и особенности

- Это20M, не25M. Файлы с quoted header, UTF-8 CRLF; заголовки подтверждены.
- Полных дубликатов ratings/tags0 по raw-отчету, movieId каталога уникальны.
  Future repeated user/movie target исключается unseen_only; факты истории сохранять.
- Rating0.5..5, битых datetime0 по исходному отчету;252 null tmdbId допустимы.
- Локальный datetime без timezone интерпретируется UTC как допущение конвертации
  зеркала; оно фиксируется в config. Официальный архив имеет unix timestamps.
- title/genres/year/links могут быть static_assumed; теги пользователя только<T.
- Genome рассчитан по пользовательским tags/ratings/reviews и не имеет времени fit.
  Поэтому strict primary его не использует; static_assumed genome — diagnostic
  ablation. Нельзя называть его гарантированно независимым историческим content.
- Genre list включает '(no genres listed)'→unknown; multi-genres не сводятся
  к одной категории в зависимости от target rating. Price/demographics отсутствуют.

## Статистика

По полному историческому rating.csv:

- Users138493, rated movies26744, catalog movies27278; ratings20000263.
- Среднее144.41 ratings/user (медиана68, минимум20) и747.84/movie (медиана18).
- Singleton users0, singleton rated movies3972 (14.85% именно rated catalog).
  Catalog movies без ratings534. Это не доказывает отсутствия temporal cold users.
- Историческое отношение ratings/(U×I)≈0.0054 (0.54%);
  точная unique-pair density не пересчитывалась.
- Tags465564 от7801 users; все tag users входят в rating users.
- Genome10381 movies×1128 tags=11709768 scores; охват≈38% полного каталога,
  не10%. Время fit genome не определено в отчете/файлах.
- Период ratings1995-01-09 11:46:44…2015-03-31 06:40:02,
  timestamp timezone локального зеркала не доказана, принят UTC assumption.

Temporal cold start по пользователям и товарам после принятых splits:
**не определено в отчете**; измеряется генератором относительно train cutoff.
Доли сущностей с одним событием — singleton shares, не cold-start rates.
Плотность events/(U×I) при повторяющихся парах не равна точно заполненности
матрицы уникальных user-item пар; исторические оценки трактуются как ориентир.

## Структура подпапок

- `raw/` — исходные файлы, только чтение, содержимое не модифицируется.
- `processed/` — выходы пайплайна проекта, только Parquet; готовые данные в публичный репозиторий не включены.
- `.gitkeep` в raw/processed сохраняет структуру в репозитории; data files
  игнорируются .gitignore, этот README.md версионируется.

## История изменений

- 2026-10-19 — 1.0 draft: локальный состав/размеры, raw-схемы, mapping, аномалии,
  статистика с provenance и dataset-specific PIT/evaluation правила.
