# Источники и форматы данных

Рабочий концепт использует четыре независимых dataset adapters. Файлы не входят
в репозиторий и размещаются пользователем в `data/<name>/raw/`. Нужная локальная
структура сохраняется `.gitkeep`; конкретные filenames задаются в
`configs/datasets/<name>.yaml`. Условия использования определяет источник данных.

| Dataset | Источник | Основные raw files | Primary target |
|---|---|---|---|
| Retailrocket | [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) | events.csv; category_tree.csv; item_properties_part1.csv; item_properties_part2.csv | transaction |
| Taobao | [UserBehavior](https://www.kaggle.com/datasets/marwa80/userbehavior) | UserBehavior.csv | buy |
| Yoochoose | [RecSys Challenge mirror](https://www.kaggle.com/datasets/phhasian0710/yoochoose) | yoochoose-clicks.dat; yoochoose-buys.dat; optional yoochoose-test.dat | click |
| MovieLens20M | [Kaggle mirror](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset), [GroupLens](https://files.grouplens.org/datasets/movielens/ml-20m-README.html) | rating.csv; movie.csv; link.csv; tag.csv; genome_scores.csv; genome_tags.csv | rating>=4 |

## Временные контракты

Retailrocket: event/property timestamps — Unix milliseconds; внутренняя граница
видимости округляется до seconds. Недельные свойства доступны только после своего
timestamp; дерево категорий — static_assumed.

Taobao: CSV без header, поля user_id,item_id,category_id,behavior,timestamp;
timestamp — UTC seconds. Основное окно [2017-11-24,2017-12-04), train до12-02,
valid до12-03. Category — observed_asof, поскольку одна static mapping опровергнута
полным локальным scan; simultaneous conflict даёт unknown.

Yoochoose: CSV без header, timestamp ISO8601 UTC. Session ID используется как
user entity; primary click, purchases сохраняются как контекст. Own temporal split:
train до2014-09-01, valid до09-15, test до10-01. Official test — отдельная
дополнительная population, отключённая в primary config.

MovieLens: используемая копия имеет singular basenames и timestamps вида
`YYYY-MM-DD HH:MM:SS`. Официальный архив может иметь другие filenames/time format;
адаптер рассчитан на описанную конфигурацию зеркала. Rating>=4 — positive,
rating<=2.5 — negative fact,3/3.5 — context. Genome по умолчанию выключен;
genres считаются static_assumed. Время видимости tags — observed_asof.

Подробные таблицы колонок и локальные наблюдения:
[Retailrocket](../data/retailrocket/README.md), [Taobao](../data/taobao/README.md),
[Yoochoose](../data/yoochoose/README.md), [MovieLens](../data/movielens/README.md).
Измеренные итоги генерации — в [отчёте](experiment_results.md).
Схемы выходных Parquet — в [архитектуре генератора](generator_architecture.md).
