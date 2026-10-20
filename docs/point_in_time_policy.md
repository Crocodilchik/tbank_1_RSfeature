# Point-in-Time Policy: контракт защиты от Data Leak

Версия: 1.0
Статус: approved
Дата последнего обновления: 2026-10-19

Область: генератор, все пять моделей, дистилляция и оценка. Это приоритетный
контракт проекта. Редакция 1.0 интегрирует разрозненные правки 0.1 и разрешает
внутренние противоречия в пользу строгого прошлого; принципы §1–4 сохранены.
Статус поставки — рабочий концепт; выполненные проверки и ограничения описаны в project_status.md.

## 1. Философия политики

Для события T модель видит только W(T): реальные события timestamp < T, явно
объявленные static_assumed атрибуты и производные артефакты, доступные до T.
Будущее разрешено только оценщику для ground truth, исключений eval negatives
и диагностик. Оно не меняет признаки, обучающую выборку, sampling weights,
подбор гиперпараметров или доступность модельных артефактов.

## 2. Определения и часы

Target `(user_id,item_id,T,label)` — ответ. Модель получает user_id, candidate
item_id и cutoff T; target event_type, label и контекст покупки — не признаки.
Все timestamp/cutoff/snapshot/fit_cutoff — int64 UTC Unix seconds. timestamp_ms
сохраняется отдельно для аудита. Использование секунд консервативно: все события
одной секунды одновременны, даже если исходник содержит миллисекунды.

W(T) включает dynamic metadata с metadata_ts < T. Backward as-of join запрещает
exact matches. Статический атрибут разрешен только при явном допущении в конфиге;
сам факт появления товара в будущем не доказывает доступность каталога до T.

## 3. Инварианты

Для любого модельного признака f: `f(full,T)=f(prefix(timestamp<T),T)` при
неизменных разрешенных static inputs и запрошенной паре user/candidate.
Равные timestamps обрабатываются атомарной группой: сначала predict для всей
группы, затем update состояния. source_row_id/user_event_seq обеспечивают
детерминизм, не доказывают хронологию. reliable_intra_timestamp_order=false
для всех четырех наборов. Исключение допустимо только с документированной
гарантией источника, которой сейчас нет.

Label текущего события никогда не feature; предыдущие labels с timestamp<T
допустимы как история. Source file/row IDs — lineage, не predictive input.

## 4. Доступность классов данных

### 4.1. Взаимодействия и агрегаты

Только timestamp<T; окна `[T-window,T)`. Синтетические события не пополняют
реальную историю, popularity, граф взаимодействий или состояние конверсии.
Категория исторического события присоединяется as-of его времени; категория
candidate — as-of T. Это два разных join.

### 4.2. Static metadata

Явно разрешены: MovieLens title/genres/year/external IDs; RetailRocket дерево
категорий (но не связь item-category); Taobao category_id в режиме static_assumed
с проверкой постоянства mapping. Допущение о каталоге указывается в отчете.
Поведенческая популярность/кластеры/средние оценки не становятся статическими.

### 4.3. Изменяемые metadata

RetailRocket properties — последний snapshot `<T` для каждого `(item,property)`.
MovieLens tags — только tags.timestamp<T. Yoochoose category/brand — только из
прошлых кликов; target context, price и quantity текущей покупки запрещены.
При отсутствии доступной записи — unknown, никакого обратного заполнения.

### 4.4. Target encoding

`TE(c,T)=(sum_prior_y+alpha*global_prior(T))/(count_prior+alpha)`.
Обе суммы и global_prior вычисляются строго по прошлому; y — бинарный predicate
конкретной задачи, не смесь label 0/1/2. alpha=10, min_history_events=5;
при короткой истории используется прошлый global prior, без прошлого — 0.5.
Сглаживание уменьшает шум, временную защиту обеспечивает cutoff.

### 4.5. Fitted artifacts и supervised training

Для сохраненных embeddings, центроидов, scaler, vocab, feature selection и teacher
features сохраняются fit_cutoff и максимальное время fit-данных. Для применения
к T: max_fit_data_ts < fit_cutoff_ts <= T; время выпуска snapshot строго <T.
Исторические обучающие строки с derived features используют temporal out-of-fold
артефакты из предыдущего префикса. Артефакт, fitted на всем train, нельзя выдавать
за event-level признак ранней train-строки.

Обучение самой supervised модели на train labels разрешено; веса обучаемого
end-to-end predictor не являются заранее выданной таблицей признаков. In-sample
train predictions не считаются strict оценкой. Экспорт embeddings из этой модели
в другой learner вновь требует temporal out-of-fold. Это относится и к Fusion.

### 4.6. Графы

`G(T)` содержит только реально доступные до T ребра и metadata. Daily snapshot s
строится из `<s` и выбирается только s<T; ties не допускаются. Neighbor sampling
ограничен этим графом. Старые snapshot можно безопасно использовать с потерей
свежести. Для новых наблюдений разрешены инкременты с edge.timestamp<T.
Co-purchase доступен только после завершившейся транзакции; участники текущей
транзакции не являются контекстом ее собственного предсказания.

### 4.7. Последовательности

История — реальные события пользователя `<T`, последние N=50 в хронологическом
порядке. Нельзя включать сам target-event или другие события той же секунды.
Прежнее событие с таким же item не является утечкой само по себе. Causal mask и
padding mask обязательны; causal mask не защищает metadata, embeddings и fitting.

### 4.8. Негативы обучения

Только train. Catalog известен по префиксу; context item исключается.
При no_previous_interaction_negatives исключаются реальные user-item
взаимодействия с timestamp<=context_timestamp (одновременная supervision-группа
может участвовать в запрете негативных ответов, но не в features).
Будущие события с timestamp>context_timestamp не влияют на sampling или веса.
future_positive_conflict рассчитывается отдельно оценщиком после фиксации samples.

Из старых названий strict_no_future_conflict/drop_conflicts,
allow_weak_negatives/downweight_conflicts сохраняются только diagnostic варианты:
если они используют будущие сведения, run не является strict. Основная политика
keep_as_noise_audit_only: фиксированный вес 1.0, флаг только диагностика.

## 5. Режимы корректности

| Режим | Смысл | Применение |
|---|---|---|
| static_assumed | Явное допущение о конкретном поле | Отмечается в каждом результате |
| split_level_pit | Fit на train, transform valid/test | Диагностика; не гарантия внутри train |
| event_level_pit | Все inputs каждой строки из доступного префикса | Основные итоговые результаты |

Основной run: real_only (реальные события и реальные PIT metadata); при отсутствии
явных negatives обучающие ненаблюдаемые пары все равно нужны, но оформляются
контрастивным sampler, не выдуманными показами. Термин real_only означает отсутствие
синтетических personas/иерархий/инверсии labels, а не наличие реальных impression logs.

## 6. Правила датасетов

### 6.1. RetailRocket

view→0, addtocart→1, transaction→2; primary transaction. Мс сохранять в
timestamp_ms, timestamp=floor(ms/1000). transactionid — только аудит/группировка
завершенных корзин. item_properties — недельные observed snapshots, join `<T`;
category tree static_assumed. Повторы allow_previous. Дубли удаляются по полному
исходному событию, не по округленному времени. Полные правила очистки — в генераторе.

### 6.2. Taobao

pv→0, cart/fav→1, buy→2; primary buy. Окно UTC
`[2017-11-24 00:00:00,2017-12-04 00:00:00)`. Train до 2017-12-02,
valid 2017-12-02, test 2017-12-03. Сначала фильтрация окна, затем dedup;
отчетность не предполагает заранее точное число удаленных строк.
Полный scan 2026-10-19 опроверг постоянство item→category. Основной режим
category_id — observed_asof; static_assumed допустим только диагностически при
доказанном постоянстве. Противоречащие категории в одну секунду дают tombstone
category_id=-1 (зарезервированный unknown), доступный строго после этой секунды.
Он сбрасывает старую связь, не восстанавливает ее до следующего наблюдения.
При противоречащих категориях item глобальный static mapping не выбирается
по будущему большинству: strict run требует observed_asof либо согласованного
внешнего catalog snapshot. Повторы allow_previous.

### 6.3. Yoochoose

Session ID→user_id, user_type=session во всех архитектурах, никакого объединения
разных сессий в постоянных пользователей. click→1, buy при quantity>0→2.
quantity<=0 исключается из purchase facts с счетчиком. Primary только event_type=click,
а не label>=1. Покупки могут быть предыдущей историей, но не click ground truth.

category='0' unknown; 'S' special offer; 1..12 category; другие корректные числовые
токены brand. Это context предыдущих кликов, не static каталог. Item category/brand
в T — мода прошлых значений; tie → меньший raw ID. current context недоступен.

Основной own_temporal_split: train `<2014-09-01`, valid `[2014-09-01,2014-09-15)`,
test `[2014-09-15,2014-10-01)` UTC на clicks+buys; target click.
Official test.dat — отдельный supplementary click-only, сессии не пересекаются
с clicks и buys. Так как он покрывает тот же период, оценочные targets ограничены
`[2014-09-15,2014-10-01)`; ранние official clicks только собственная история `<T`.
Fit/scaler/vocab/global popularity/GNN fitting не используют official test.
Для inference разрешен только уже наблюденный prefix самой official session,
в том числе временные Session-Item edges в ее query context, без global fit. При желании оценить
весь апрель–сентябрь нужен отдельный rolling-origin протокол с cutoff<T.
Purchase на основном temporal split — третья, отдельная supplementary ablation.
Политика allow_within_session_only исключает из NEGATIVES прошлые items сессии,
но сама по себе не запрещает повторный ground-truth click. Это историческое имя
конфига, а не утверждение, что повторы кликов отсутствуют.

### 6.4. MovieLens

rating>=4→2; rating ∈ {3,3.5}→1; rating<=2.5→0. Primary rating>=4;
низкий рейтинг — явное отрицательное предпочтение, но не показ без клика.
Локальный timestamp — datetime string, трактуется UTC (допущение преобразования
зеркала), не unix number. unseen_only: target ранее не оценен до T; кандидаты-
негативы не имеют ни одной оценки этого пользователя во всей наблюдаемой выгрузке
(oracle только у оценщика). Genres/title/IDs могут быть static_assumed, tags `<T`.

Tag Genome вычислялся по пользовательскому контенту, включая ratings и tags;
это подтверждает [официальный README](https://files.grouplens.org/datasets/movielens/ml-20m-README.html).
Времени fit в локальных файлах нет. Поэтому genome выключен в strict primary.
Разрешение версии 0.1 объявлять его static_assumed сохраняется только как явно
помеченная diagnostic ablation; оно не доказывает event-level PIT.

## 7. Синтетическое обогащение

Personas/искусственные категории — только ablation, не реальные демографические
факты. Rolling daily snapshot s: fit на `<s`, назначение из `<s`, применение s<T.
Для ранних train targets без fitted prefix — cold_start, без первого target в fit.
fit до T1 годится для valid/test, не ранних train samples. Boundary noise
детерминирован по доступному состоянию и seed; инверсия реальных labels в primary
запрещена. Конкретные K, признаки, quantiles core/middle/boundary — в генераторе.

Синтетические отрицательные пары не подменяют реальные ground truth valid/test.
Их difficulty и веса определяются прошлым. Future-conflict annotation — evaluator
only (§4.8); future-dependent drop/downweight — diagnostic с явной маркировкой.

## 8. Evaluation sampler

### 8.1. Общая единица оценки

Для каждого пользователя/сессии и каждого split valid/test выбирается последний
реальный eligible primary target по `(timestamp,event_id)`. Это
last_event_only, один target на пользователя в сплите, одинаковый для всех моделей.
Train sequence strategy leave_one_out означает один sample для КАЖДОГО eligible
train target с предшествующим префиксом, а не случайное holdout-разбиение.
Нет primary — пользователь не входит в primary метрику; coverage учитывает это.
Fallback включается только в отдельной подписанной когорте, не меняет primary.

### 8.2. Исключения

`P(u,split)` — все primary-positive items пользователя в данном eval split,
включая одновременные и события после T; это обязательная evaluator-only защита.
`H(u,T)` — реальные прошлые items пользователя timestamp<T.

| Датасет | exclusion_policy | Базовое repeat-исключение R |
|---|---|---|
| RetailRocket | allow_previous | пусто |
| Taobao | allow_previous | пусто |
| Yoochoose | allow_within_session_only | H текущей сессии |
| MovieLens | unseen_only | все оцененные items пользователя во всей выгрузке |

Для negatives всегда `E={target_item} ∪ R ∪ P(u,split)`. Формулировка старой версии
«allow_previous исключает только target» относится к repeat-компоненте, не отменяет
обязательное P. MovieLens oracle обо всех оценках нужен только sampler, не features.
exclude_future_positives=false выключает ДОПОЛНИТЕЛЬНЫЙ oracle за пределами eval
split, не обязательное P. Если true — исключить все будущие primary positives
в наблюдаемой выгрузке, horizon зафиксировать в manifest. Названия и цели
fallback задаются §13; target eligibility всегда проверяется ДО отбора last_event.

### 8.3. Каталог и sampling

`C(T)` по умолчанию — товары, встречавшиеся в разрешенных реальных событиях или
observed metadata строго до T; будущий весь каталог не подключается автоматически.
Разрешенная static metadata обогащает известные candidate IDs, не сообщает о времени
появления товара. Ground truth добавляется отдельно, даже если item cold-start.

Из отсортированного `C(T)-E` выбрать 99 разных negatives без возвращения.
random — равномерно; popularity_biased — `(count_before_T+1)^0.75`;
hard — категория target известна до T, иначе random. При нехватке hard —
дополнение random из оставшихся. Если всего <99 — skip группы с причиной и
учетом coverage; silently shrinking/duplicates/infinite resampling запрещены.

Group seed = первые 8 байт SHA-256 от UTF-8 строки
`seed|dataset|mode|split|target_event_id|strategy` по big-endian, modulo (2^63-1).
Generator и его версия фиксируются (NumPy PCG64 на Этапе 3). Пул генерируется
один раз, сортируется по group/candidate и хранится по hash; модели читают тот же
артефакт. Порядок кандидатов не сообщает положение ground truth.

### 8.4. Cold-start slices

train-cold user/item — отсутствовали в train, независимо от singleton-доли всего
датасета. Event-cold — нет событий до T. Yoochoose short-session — длина доступной
истории <=2, не будущая длина сессии. Long tail — по popularity до T, пороги
fit только на train. Метрики и размер каждой когорты публикуются отдельно.

## 9. Требования к схеме данных

Физические типы — Arrow/Parquet int64, int8, string (UTF-8), bool, float64,
list соответствующего типа. Null разрешен только явно. Дополнительные поля
в model input запрещены без schema-version и feature registry.

### 9.1. Events table

```text
event_id: int64
user_id: int64
item_id: int64
timestamp: int64
timestamp_ms: int64 nullable
label: int8
event_type: string
dataset_source: string
user_type: string
is_synthetic: bool
source_row_id: int64
user_event_seq: int64
split: string
target_cutoff_ts: int64
```

Ключ `event_id`; все поля, кроме явно nullable, обязательны. В core только реальные
события (`is_synthetic=false`). `label` ∈ {0,1,2}, `user_type` ∈ {persistent,session},
`split` ∈ {train,valid,test}; в отдельном official-файле split=test, режим задан manifest.
`target_cutoff_ts=timestamp`. Все int64 времени — UTC Unix seconds, кроме timestamp_ms.
Если есть timestamp_ms, `timestamp=floor(timestamp_ms/1000)`; он служит аудиту,
не разрешает использовать события внутри той же секунды.

### 9.2. Item metadata table

```text
item_id: int64
dataset_source: string
field_name: string
field_value: string
valid_from: int64
valid_to: int64 nullable
availability_class: string
source: string
snapshot_timestamp: int64 nullable
```

Ключ `(dataset_source,item_id,field_name,valid_from)`. Long format.
`field_value` — канонический JSON в UTF-8 String (скаляр/список), физического union
в Parquet нет. Пропущенное значение — отсутствие записи; JSON null запрещен.
Логические типы полей задает реестр ниже. `availability_class` ∈
{static_assumed,observed_snapshot,derived_pit}; `source` ∈ {raw,synthetic,derived}.
Для static `valid_from=-9223372036854775808`, snapshot_timestamp=null.
Для observed `valid_from=snapshot_timestamp`; для derived — время выпуска
префиксного артефакта. `valid_to` — только audit, его нельзя подавать модели или
использовать для выбора записи с учетом будущего. As-of выбирает последнюю
из доступных записей с valid_from < T независимо от будущего valid_to.
`observed_asof` в конфиге означает способ join для observed_snapshot.

### 9.3. User metadata table

```text
user_id: int64
dataset_source: string
field_name: string
field_value: string
valid_from: int64
valid_to: int64 nullable
availability_class: string
feature_source: string
fit_cutoff_ts: int64 nullable
cluster_level: string nullable
seed: int64 nullable
```

Ключ `(dataset_source,user_id,field_name,valid_from)`. Кодирование field_value
и доступность как §9.2. `feature_source` ∈ {real,synthetic,derived};
`cluster_level` ∈ {core,middle,boundary,cold_start} или null для иных полей.
Для fitted fields обязательны fit_cutoff_ts и seed, с
`fit_data_max_ts < fit_cutoff_ts <= valid_from < T`. Cutoff — исключающая верхняя
граница, поэтому равенство fit_cutoff_ts и valid_from безопасно.

### 9.4. Graph edges table

```text
edge_id: int64
src_type: string
src_id: int64
dst_type: string
dst_id: int64
edge_type: string
timestamp: int64
valid_from: int64
valid_to: int64 nullable
weight: float64 nullable
dataset_source: string
split_snapshot_id: string
source: string
```

Ключ `(split_snapshot_id,edge_id)`. `src_type/dst_type` ∈ {User,Item,Category,Brand};
User означает Session для Yoochoose. `source` ∈ {real,synthetic}. Для реального
interaction timestamp=valid_from=время события. Статические ребра используют sentinel
§9.2. Для derived edges время равно времени доступности артефакта; fit cutoff
сохраняется в manifest. Изменяемая item-category связь выбирается as-of, а не
накапливается с будущими категориями. Никаких плотных adjacency matrices.

### 9.5. Sequence samples table

```text
sample_id: int64
user_id: int64
target_event_id: int64
target_item_id: int64
target_timestamp: int64
target_label: int8
cutoff_ts: int64
history_event_ids: list[int64]
history_item_ids: list[int64]
history_timestamps: list[int64]
history_labels: list[int8]
history_category_ids: list[int64]
history_time_deltas: list[int64]
max_history_timestamp: int64 nullable
dataset_source: string
split: string
```

Ключ sample_id=target_event_id. `cutoff_ts=target_timestamp`; все списки имеют
одинаковую длину ≤50, без null elements. Неизвестная категория кодируется 0 в
модельном словаре (raw categories перенумеровываются отдельно с 1).
Пустая история — шесть пустых списков и max_history_timestamp=null; ее не удаляют.
Для непустой max_history_timestamp < target_timestamp, дельты положительны.
Сам target_event_id отсутствует; прежнее событие с тем же item допустимо при
repeat-policy. Нельзя удалять прошлый item только из-за совпадения с ответом.

### 9.6. Negatives table

```text
negative_id: int64
user_id: int64
item_id: int64
context_event_id: int64
context_timestamp: int64
timestamp: int64
label: int8
negative_type: string
dataset_source: string
sampling_seed: int64
catalog_snapshot_ts: int64
future_positive_conflict: bool
conflict_first_future_ts: int64 nullable
```

Ключ negative_id; уникально `(context_event_id,item_id,negative_type)`.
`timestamp=context_timestamp`, label=0; negative_type ∈
{global,cluster_mixed,hard_boundary}. Только train. Последние два поля —
evaluator-only диагностические аннотации: хранить можно, читать моделью, feature
builder, sampler или вычислением training weights нельзя. Аннотация не меняет
состав или порядок обучающих строк. Без future conflict конфликтное время null;
иначе строго больше context_timestamp. Флаг true тогда и только тогда, когда
в наблюдаемой выгрузке есть реальное взаимодействие того же user/item после
context_timestamp; conflict_first_future_ts — минимум таких timestamps. Этот
широкий конфликт не ограничен primary-positive predicate, что фиксируется в
audit metadata. catalog_snapshot_ts < context_timestamp.

### 9.7. Evaluation candidates table

```text
eval_group_id: int64
user_id: int64
target_event_id: int64
target_timestamp: int64
ground_truth_item_id: int64
candidate_item_id: int64
candidate_label: int8
candidate_type: string
dataset_source: string
sampling_seed: int64
catalog_snapshot_ts: int64
exclusion_policy: string
```

Ключ `(eval_group_id,candidate_item_id)`; eval_group_id=target_event_id в пределах
протокола. Ровно 100 разных candidate_item_id, один candidate_label=1;
он совпадает с ground_truth_item_id и имеет candidate_type=ground_truth.
Остальные candidate_label=0, candidate_type ∈ {random,popularity,hard}.
`exclusion_policy` — строка имени repeat-policy; split, mode, primary_target,
exclusion_set_size, oracle horizon и hash пула — в отдельном group manifest,
не в model inputs. sampling_seed — детерминированный seed группы.

### 9.8. Идентификаторы, реестр и manifests

event_id = dataset_code*10^15 + file_code*10^12 + source_row_id.
Dataset codes: retailrocket=1, taobao=2, yoochoose=3, movielens=4;
file_code по порядку event files в генераторе, source_row_id — номер data-row с 1
до фильтров/dedup (<10^12). Duplicate сохраняет меньший исходный ID.
user_event_seq — 1-based порядок `(timestamp,event_id)`, не reliable order.
negative_id/edge_id — 63-bit SHA-256 key их источников, collision → error.

Реестр field_name→logical dtype: category_id/root_category_id/brand_id/item_cluster/
user_cluster/total_interactions/first_interaction_ts/last_interaction_ts/
unique_categories int64; category_depth/popularity_tier int8; available bool;
genres list[string]; title string; release_year/imdb_id/tmdb_id int64;
observed_price float64; property_<raw_id> list[string]; affinity_score float64.
Unknown — отсутствие значения; item/user cluster 0 reserved cold_start в модели.
Произвольные свойства не получают выдуманную семантику цены/бренда.

categories.parquet: category_id:int64, dataset_source:string,
parent_id:int64 nullable, level:int8, name:string nullable, full_path:string.
Ключ `(dataset_source,category_id)`; level начинается с 0; циклы запрещены.
MovieLens multi-genres сохраняются списком и отдельными Item-Category edges,
не выбираются по целевому рейтингу.

Manifest каждого артефакта хранится в processed/audit/artifact_manifest.parquet:
artifact_path:string, sha256:string, row_count:int64, schema_version:string,
policy_version:string, config_hash:string, code_revision:string,
created_at_utc:string, seed:int64, fit_cutoff_ts:int64 nullable,
max_input_ts:int64 nullable, snapshot_ts:int64 nullable, mode:string,
source_hashes:string (канонический JSON), environment:string (JSON),
parent_artifact_hashes:string (JSON). Ключ artifact_path. Файл manifest не
хеширует сам себя. Group manifest: eval_group_id:int64, split:string, mode:string,
primary_target:string, exclusion_set_size:int64, target_is_fallback:bool,
catalog_size:int64, oracle_horizon_end:int64 nullable, pool_hash:string.
Диагностики и evaluation labels физически отделены от whitelist model inputs.

## 10. Запрещенные паттерны и безопасные замены

Полные group mean/target encoding, CURRENT ROW, target labels/context, будущие
snapshot/граф/учитель и fit на train+valid+test запрещены. ROWS ... 1 PRECEDING
без обработки ties также небезопасен. Вместо этого: time-group streaming с
predict-before-update, backward as-of без exact, prior snapshot, temporal OOF.
Масштабирование/словарь/отбор обучаются на train, не на evaluation.
Для raw integer IDs допустим стабильный hashing в заранее фиксированный размер
с OOV, если он не fitted по будущей активности.

## 11. Quality gates и критерии прохождения

Любое нарушение strict run → error, метрики не публикуются как финальные.
Все структурные ограничения проверяются на всем артефакте. Prefix-тест:
все targets мини-fixtures, на реальных данных минимум 100 targets на split
(или все, если меньше), seed=42, плюс все граничные fixtures. Числа с плавающей
точкой сравнивать rtol=1e-6, atol=1e-8; keys/lists/IDs/timestamps — точно.

| Gate | Критерий |
|---|---|
| 11.1 Splits | Нулевое пересечение event_id; все времена в полуоткрытом split; непустые train/valid/test |
| 11.2 Prefix | Значения full и prefix равны; удаление/изменение будущих событий и target label не меняет inputs |
| 11.3 As-of | Для всех observed/derived inputs выбранный ts<T; exact/future отсутствуют, нет прошлого → unknown; cardinality событий не меняется |
| 11.4 TE | Равен эталонной формуле по prefix; изменение target и same-second labels не влияет; совпадение с full mean само по себе не ошибка |
| 11.5 Graph | Все dynamic edge times<T; snapshot<T; target-event edge отсутствует, fit provenance соблюден, neighbor sampling не выходит за snapshot |
| 11.6 Sequence | Равные длины списков, <=50, strictly past timestamps, положительные deltas, target_event_id отсутствует; empty/null контракт выполнен; causal+padding mutation test |
| 11.7 Negatives | Train-only, history/context item exclusions, уникальность, known catalog; future annotations не меняют training rows/features/weights |
| 11.8 Eval fairness | 100 уникальных candidates, один positive, negatives∩E=∅; одинаковые target/group/pool hashes всех моделей |
| 11.9 Artifact provenance | Schema/config/raw/code/environment hashes присутствуют; fit max_ts<cutoff; output воспроизводим по значениям |
| 11.10 Synthetic ablations | Опубликованы real_only, negatives, personas, full enrichment на реальном eval; эффект не объявлен заранее |

Обязательные adversarial fixtures: равные секунды, snapshot==T, первая запись,
новый user/item, повтор item в истории, пустая история, недостаточный каталог,
дубликат, orphan/cycle category, conflicting metadata, future conflict, ошибочный
fit cutoff, label mutation, official event до train cutoff. Отчет качества —
processed/audit/quality_gates.parquet, поля gate:string, passed:bool,
checked_rows:int64, violations:int64, details:string; ноль violations.

## 12. Контракты моделей

Все пять моделей получают одинаковые query groups, candidates, доступный
causal replay и static assumptions. Обученные weights frozen на train, tuning
и calibration только valid; test labels не участвуют в fit/selection. Ранее
произошедшие valid/test события можно добавлять в историю/агрегаты/граф строго
до T, одинаковым протоколом всем моделям; будущие test edges запрещены.
Official history используется только как собственный префикс session.

Baseline/Adv1: PIT агрегаты/TE и whitelist колонок; feature selection fit на
train folds. Adv1 принимает только типизированные DSL-примитивы с cutoff,
не произвольный SQL без проверки. Подозрительный train AUC — сигнал проверки,
не доказательство отсутствия/наличия leak.
Adv2: sparse heterogeneous GraphSAGE, bounded neighbor sampling. LightGCN
стандартной формы не обещает inductive cold-start и рассматривается отдельно.
Adv3: history `<T`, causal+padding masks, item metadata known before T.
Adv4: saved GNN features для train — temporal OOF; inference snapshots — только
из разрешенного прошлого; Fusion не наследует будущие embeddings.
Distillation: teacher scores historical samples — temporal OOF; при score-as-feature
teacher нужен на inference; student-only требует teacher как soft target,
не обязательную входную фичу. Нельзя учить student на outer valid и вновь
использовать этот же valid для честной оценки.

## 13. Каноническая конфигурация политики

Это единственный нормативный блок defaults. Разбиение по YAML-файлам и проверки
loader описаны в [configuration.md](configuration.md). Исполняемые YAML находятся в configs/. Дублировать один ключ в двух файлах нельзя.

```yaml
point_in_time_policy:
  version: '1.0'
  default_mode: event_level_pit
  timestamp_handling:
    unified_unit: seconds
    timezone: UTC
    store_milliseconds_when_available: true
    strict_less_than: true
    reliable_intra_timestamp_order: false
  metadata:
    default_availability: observed_asof
    allow_exact_match_for_static_assumed: true
  target_encoding:
    enabled: true
    smoothing_alpha: 10.0
    min_history_events: 5
    empty_prior: 0.5
    mode: expanding_point_in_time
  graph:
    snapshot_granularity: daily
    allow_exact_match: false
    neighbor_sampling_from_snapshot_only: true
  sequences:
    max_history_length: 50
    train_strategy: leave_one_out
    eval_strategy: last_event_only
    include_target_event_in_history: false
    allow_previous_same_item: dataset_policy
    causal_mask_required: true
    empty_history_policy: keep_with_cold_start
  synthetic:
    personas:
      enabled: false
      mode: rolling_snapshot
      fit_cutoff_source: previous_snapshot
    negatives:
      enabled: true
      used_for: train_only
      no_previous_interaction_negatives: true
      future_conflict_policy: keep_as_noise_audit_only
      sample_weight: 1.0
      negatives_per_positive: 4
      difficulty_mix: [0.5, 0.3, 0.2]
  evaluation:
    candidate_pool_size: 100
    negatives_per_group: 99
    seed: 42
    target_strategy: last_event_only
    exclude_user_eval_positives: true
    exclude_future_positives: false
    insufficient_candidates: skip_with_coverage
    fallback_mode: separate_cohort
    repeat_purchase_policy:
      retailrocket: allow_previous
      taobao: allow_previous
      yoochoose: allow_within_session_only
      movielens: unseen_only
    primary_targets:
      retailrocket: transaction
      taobao: buy
      yoochoose: click
      movielens: rating>=4.0
    fallback_targets:
      retailrocket: addtocart
      taobao: cart
      yoochoose: last_click
      movielens: rating_3.0-3.5
    negative_sampling:
      strategy: random
      hard_negative_category_depth: 1
    catalog_policy: observed_before_target
  datasets:
    retailrocket:
      item_properties_availability: observed_snapshot
      category_hierarchy_availability: static_assumed
      item_category_availability: observed_snapshot
    taobao:
      category_id_availability: observed_asof
      strict_ablation_category_id_availability: observed_asof
    yoochoose:
      user_entity_type: session
      category_availability: observed_asof
      primary_eval_mode: own_temporal_split
      include_official_test: false
      official_test_mode: supplementary_click_only
      official_test_window: same_as_primary_test
    movielens:
      genres_availability: static_assumed
      genome_scores_enabled: false
      genome_scores_availability: static_assumed
      genome_scores_usage: diagnostic_ablation_only
      user_tags_availability: observed_asof
```

## 14. Классификация нарушений

Critical leak: текущий answer/future input, будущий fit или teacher,
future-dependent training filtering/weights → experiment invalid, redo.
Diagnostic limitation: split-level personas, genome assumption без fit time,
oracle train negatives → отдельный diagnostic, не strict итог.
Evaluation oracle: future positives используются только в E/метриках/аннотации,
не передаются model input и одинаковы для всех моделей → допустим с manifest.

## 15. Проверка перед запуском

Нормализованные часы, непересекающиеся splits, schema validation, past-only
inputs/artifacts, whitelist колонок, locked seeds/config, raw fingerprints,
фиксированные eval pools, все gates §11 и отдельные cold-start slices обязательны.
Код и окружение воспроизводимы, smoke run явно отделен от full-data run.

## 16. Успешное прохождение

Одновременно выполняются: model input condition (только W(T)), evaluation fairness
(общие пулы/цели/протокол), auditability (source,cutoff,availability,seed,hash,version).
Положительные экспериментальные результаты не заменяют эти проверки.

## 17. Зафиксированные решения и вопросы перед реализацией

Primary targets/repeat policies — §13. Taobao observed_asof после полного consistency scan;
Yoochoose observed category; MovieLens strict без genome. Graph daily, personas
только ablation. Эти решения устраняют прежний список взаимоисключающих вариантов.

Пользователь подтвердил Этап3 и стартовый протокол experiment_plan.md.
Ресурсы и совместимость пакетов проверены: docs/project_status.md и project_status.md.
Измерения не являются обещанием времени полного обучения.

## 18. Итоговая норма

При противоречии с любым другим документом применяется этот контракт. Модель
знает прошлое; оценщик — ground truth будущего. Синтетика помогает обучению,
не заменяет реальную оценку. Изменения не могут молча ослаблять strict_less_than.

## История изменений

- 2026-10-19 — 1.0 draft: интегрированы разрозненные правки 0.1, уточнены часы,
  схемы, repeat/oracle, train conflicts, sequence strategies, snapshots, genome,
  official temporal window, distillation и измеримые quality gates (К1–К15).
- До 2026-10-19 — 0.1 draft: исходные PIT-инварианты; версия сохранена в Git.

- 2026-10-19 — пользователь подтвердил план и переход к реализации; состояние выполнения отражено в project_status.md.

- 2026-10-19 — уточнение отдельной fallback-когорты: ее P(u,split) строится по fallback predicate, primary не смешивается с fallback. Общие ablation targets передаются явной registry и проверяются на eligibility.

- 2026-10-19 — область gates уточнена: структурные/PIT/provenance gates выполняются
  генератором; gate11.10 synthetic ablations проверяется при итоговом сравнении
  экспериментов, поскольку метрик обученных моделей до генерации еще нет.
  Его отсутствие на фазе generator не означает успешного эксперимента.
