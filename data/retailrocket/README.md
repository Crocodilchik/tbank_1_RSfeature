# Датасет: RetailRocket

Версия: 1.0
Статус: справочник рабочего концепта
Дата последнего обновления: 2026-10-19

## Источник

[Kaggle / RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset). Локальная копия описана в raw-отчете; демографии и описаний товаров нет.

## Состав файлов (в подпапке raw/)

| Файл | Строк данных | Размер, байт | Формат |
|---|---:|---:|---|
| `category_tree.csv` | 1 669 | 14454 | CSV, UTF-8, запятая, с заголовком |
| `events.csv` | 2 756 101 | 94237913 | CSV, UTF-8, запятая, с заголовком |
| `item_properties_part1.csv` | 10 999 999 | 484315749 | CSV, UTF-8, запятая, с заголовком |
| `item_properties_part2.csv` | 9 275 903 | 408929907 | CSV, UTF-8, запятая, с заголовком |

Размеры измерены по локальным файлам 2026-10-19. Counts соответствуют
локальному raw-профилю от2026-10-15 и проверке физических
строк 2026-10-19 (для CSV вычтен заголовок). Полный повторный CSV-profile здесь
не заявляется; quotes/multiline не пересчитывались отдельным parser.
Контрольные суммы исходников записываются при запуске в source_hashes манифеста генератора; локальные inventories не публикуются.

## Схема данных

### events.csv

| Колонка | Тип | Описание |
|---|---|---|
| timestamp | int64 | Unix milliseconds UTC |
| visitorid | int64 | Анонимный пользователь |
| event | string | view/addtocart/transaction |
| itemid | int64 | Товар |
| transactionid | string nullable | Идентификатор транзакции, иначе пусто |

### item_properties_part1.csv и item_properties_part2.csv

| Колонка | Тип | Описание |
|---|---|---|
| timestamp | int64 | Время недельного snapshot, Unix milliseconds |
| itemid | int64 | Товар |
| property | string | categoryid, available или анонимный код свойства |
| value | string | Значение либо список токенов через пробел |

### category_tree.csv

| Колонка | Тип | Описание |
|---|---|---|
| categoryid | int64 | Узел категории |
| parentid | int64 nullable | Родитель, null для корня |

## Ключевые поля (маппинг к единой схеме проекта)

| Роль | Исходная колонка | Единое поле проекта |
|---|---|---|
| Пользователь | visitorid | user_id |
| Товар | itemid | item_id |
| Время | timestamp (ms) | timestamp=floor(ms/1000), timestamp_ms=ms |
| Событие | event | event_type и label |
| Категория | property=categoryid → value, category_tree | metadata category_id/root/depth as-of |
| Транзакция | transactionid | Только lineage/закрытые baskets, не target feature |

Служебные event_id/source_row_id/user_event_seq/split/target_cutoff_ts добавляет
генератор по [PIT §9](../../docs/point_in_time_policy.md), не исходные файлы.
Metadata хранится отдельно от events, в частности category не дублируется на
каждую строку большого журнала.

## Маппинг событий

view→label0, addtocart→1, transaction→2. Primary event_type=transaction,
repeat_purchase_policy=allow_previous. Label0 здесь наблюдаемый view, не
доказанный отрицательный показ. Addtocart fallback только отдельной когортой.
Targets valid/test — last_event_only, одинаковые для всех моделей.

## Известные аномалии и особенности

- 460 полностью совпадающих event rows в историческом отчете; exact dedup
  до округления времени, сохранять первый source row. У properties полных дублей0.
- transactionid пуст в2733644 rows, это ожидаемо для view/addtocart.
- Недельные properties snapshots:18 дат, 03:00UTC; as-of строго snapshot<T,
  отсутствующий prefix→unknown, будущая category не присоединяется.
- В дереве1669 категорий,25 корней, max depth5. Из1196 category IDs properties
  24 не найдены в дереве; сохранить events, root/depth unknown, audit counter.
  Missing parent→orphan, cycle→error. Tree static, item-category связь dynamic.
- Numeric tokens вида n277.200 не имеют документированной семантики цены.
- 2219 transaction user-item pairs не имеют предыдущего view: не удалять как leak.
- Исходное время ms, целевое seconds UTC с сохранением ms для аудита;
  равные целевые секунды одновременны, source row не разрешает их упорядочить.

## Статистика

По историческому полному events.csv до очистки:

- Пользователей1407580, товаров235061; событий2756101.
- Среднее1.96 событий/user и11.73/item; медианы1 и3.
- Singleton users1001560 (71.15%), singleton items73609 (31.31%).
- Историческое отношение events/(U×I)≈8.33e-06; точная unique-pair density
  не определена в отчете.
- Properties содержит417053 items; пересечение с events185246,
  только events49815, только properties231807.
- Период2015-05-03 03:00:04 →2015-09-18 02:59:47UTC;139 дней.
- Transaction rows22457, уникальных transactionid17672; покупающих users11719.

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
