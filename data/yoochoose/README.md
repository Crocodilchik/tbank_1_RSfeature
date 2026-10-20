# Датасет: Yoochoose / RecSys Challenge 2015

Версия: 1.0
Статус: справочник рабочего концепта
Дата последнего обновления: 2026-10-19

## Источник

[Kaggle / Yoochoose](https://www.kaggle.com/datasets/phhasian0710/yoochoose); описание схемы также в локальном dataset-README.txt (`raw/dataset-README.txt`, скачивается с данными).

## Состав файлов (в подпапке raw/)

| Файл | Строк данных | Размер, байт | Формат |
|---|---:|---:|---|
| `dataset-README.txt` | не применимо (52 строки текста) | 3973 | UTF-8, описание источника |
| `yoochoose-buys.dat` | 1 150 753 | 55583744 | DAT: CSV без заголовка, UTF-8, запятая |
| `yoochoose-clicks.dat` | 33 003 944 | 1486798186 | DAT: CSV без заголовка, UTF-8, запятая |
| `yoochoose-test.dat` | 8 251 791 | 371725851 | DAT: CSV без заголовка, UTF-8, запятая |

Размеры измерены по локальным файлам 2026-10-19. Counts соответствуют
локальному raw-профилю от2026-10-15 и проверке физических
строк 2026-10-19 (для CSV вычтен заголовок). Полный повторный CSV-profile здесь
не заявляется; quotes/multiline не пересчитывались отдельным parser.
Контрольные суммы исходников записываются при запуске в source_hashes манифеста генератора; локальные inventories не публикуются.

## Схема данных

### yoochoose-clicks.dat и yoochoose-test.dat (без заголовка)

| № | Колонка | Тип | Описание |
|---|---|---|---|
| 1 | session_id | int64 | Анонимная session, не постоянный user |
| 2 | timestamp | string | YYYY-MM-DDThh:mm:ss.SSSZ, UTC |
| 3 | item_id | int64 | Товар |
| 4 | category | string | 0/S/category1..12/brand context |

### yoochoose-buys.dat (без заголовка)

| № | Колонка | Тип | Описание |
|---|---|---|---|
| 1 | session_id | int64 | Session, связанная с clicks |
| 2 | timestamp | string | ISO-8601 UTC с milliseconds |
| 3 | item_id | int64 | Купленный item |
| 4 | price | int64 | Обезличенная цена в событии покупки |
| 5 | quantity | int64 | Количество, включая аномальные0 |

### dataset-README.txt

Текстовое описание источника и колонок, не табличный набор; schema columns
не применимы. Точную семантику context определяет этот локальный источник.

## Ключевые поля (маппинг к единой схеме проекта)

| Роль | Исходная колонка | Единое поле проекта |
|---|---|---|
| Пользователь | session_id | user_id, user_type=session |
| Товар | item_id | item_id |
| Время | timestamp ISO | timestamp seconds UTC; timestamp_ms сохраняется |
| Событие | Имя файла clicks/buys | event_type=click/buy и label |
| Context | category | metadata category/brand из предыдущих clicks |
| Покупка | price/quantity | Только past-event aggregates, не current feature |

Служебные event_id/source_row_id/user_event_seq/split/target_cutoff_ts добавляет
генератор по [PIT §9](../../docs/point_in_time_policy.md), не исходные файлы.
Metadata хранится отдельно от events, в частности category не дублируется на
каждую строку большого журнала.

## Маппинг событий

click→label1, buy при quantity>0→2. Негативных показов в raw нет;
label0 только train sampled pairs, не реальный impression.
Primary target=click, repeat_purchase_policy=allow_within_session_only:
из NEGATIVES исключаются прошлые items этой session; повторный реальный target
click сам по себе не запрещен. Покупки — supplementary на own temporal split.

Основной own split: train до2014-09-01, valid до09-15, test до10-01UTC.
Supplementary test.dat click-only: отдельные sessions, targets только позднего
окна [2014-09-15,2014-10-01), иначе train weights могли бы видеть будущее апреля.
Ранние official clicks — только prefix своей session, не fit модели/графа.
Все модели используют last_event_only из одной registry; fallback last_click
не может создать target для session вообще без clicks в данном split.

## Известные аномалии и особенности

- В buys610030 rows (53%) quantity=0; исключить из purchase events,
  quantity<0 invalid. Price0 допустимо; источник не дает цену будущего каталога.
- В clicks69 лишних строк по(session,timestamp,item) по историческому отчету;
  exact duplicate с учетом context удаляется. Разные context одного click
  объединяются в core event с unknown context и audit, не произвольно выбираются.
- category читать string: '0' missing, 'S' offer,1..12 category, прочие числа brand.
  Это наблюденный context; current target context недоступен. Item category/brand
  — prefix mode, ties меньший ID, пустой prefix unknown.
- Во всех трех таблицах0 null по исходному отчету. ISO parsing строго UTC;
  сохранять ms, unified seconds conservative: same-second events одновременны.
- Official2312432 sessions не пересекаются с9249729 clicks sessions; buys для
  official нет. Проверять disjointness, не использовать official как purchase truth.
- 6078 clicks sessions занимают больше календарного дня; не резать их на
  выдуманных пользователей и не фильтровать по будущей длине.

## Статистика

Исторический отчет по исходным clicks до dedup:

- Sessions9249729, items52739; clicks33003944.
- Среднее3.57 clicks/session (медиана2, max200) и625.80/item (медиана22).
- Singleton sessions1259711 (13.62%), singleton items5657 (10.73%).
- Историческое отношение events/(sessions×items)≈6.77e-05;
  unique-pair density не определена в отчете.
- Buys1150753 rows,509696 sessions,19949 items;5.51% clicks sessions
  имеют buy row до quantity-фильтра. После фильтра доля не определена в отчете.
- Период2014-04-01…09-30,183 дня; последний день разрежен.
- Official test8251791 clicks и2312432 sessions, отдельная популяция.

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
