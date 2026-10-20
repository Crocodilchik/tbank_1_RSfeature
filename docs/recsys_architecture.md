# Архитектуры рекомендательных моделей

Версия: 1.0
Статус: approved
Дата последнего обновления: 2026-10-19

Обязательный контракт — [point_in_time_policy.md](point_in_time_policy.md) (PIT).
Входные схемы — [generator_architecture.md](generator_architecture.md), параметры
сравнения — [experiment_plan.md](experiment_plan.md). Качество подходов — проверяемая гипотеза. Выполненные smoke-прогоны
описаны в [отчёте](experiment_results.md); полное сравнение не завершено.

## 1. Общий интерфейс и задачи

Пять сравниваемых моделей: Baseline, Adv1 AutoML, Adv2 GNN, Adv3 Sequential
Two-Tower, Adv4 Graph-Sequential Fusion. Дистилляция — отдельное расширение,
не шестая обязательная строка основного сравнения. Нумерация везде одинаковая.

`fit(train_artifacts, inner_validation, config) -> model`; `score(queries,
candidate_items, available_state) -> scores`. На выходе одна конечная float64
оценка на `(eval_group_id,candidate_item_id)`, без пропусков или дубликатов;
больше — лучше. Перед метриками проверяется совпадение ключей с locked pool.

| Датасет | Primary predicate | label | Repeat policy |
|---|---|---|---|
| RetailRocket | event_type=transaction | 2 | allow_previous |
| Taobao | event_type=buy | 2 | allow_previous |
| Yoochoose | event_type=click | 1 | allow_within_session_only |
| MovieLens | rating>=4.0 | 2 | unseen_only |

Ordinal label описывает тип факта. Бинарная цель задачи вычисляется отдельно;
`label>=1` не заменяет click predicate Yoochoose. Positive anchors — реальные
primary train events; negatives — общий frozen prefix sampler генератора.
Просмотр/низкий рейтинг не тождествен отсутствующему impression. Для MovieLens
explicit low ratings используются отдельной auxiliary ablation, не смешиваются
молча с unobserved pairs. Cart/fav и другие прежние события могут входить в history.

Для sampled ranking все модели используют одинаковые train anchors и sampler,
valid/test last_event_only registry и 100 candidates. Per-event classification
на реальных событиях — отдельная задача/отчет, не основной sampled benchmark.

Whitelist inputs: user ID, candidate item ID, cutoff T, past aggregates,
разрешенные static/as-of metadata и history. Запрещены текущие label/event_type,
rating, transactionid, price/quantity покупки, source IDs как признаки,
ground_truth_item_id, candidate_label/type, exclusion size, future conflict,
split membership и oracle flags. Время T — контекст, не ответ.

Weights fit на train, frozen valid/test. Feature selection/scalers/vocabs —
только inner train folds; OOV defaults обязательны. История обновляется
предыдущими реальными valid/test events, predict-before-update по секундам,
одинаково для всех моделей. Outer valid — HPO/calibration, test — один раз
после фиксации решения. Fit provenance — PIT §4.5.

## 2. Baseline: настроенный GBDT

Основной backend CatBoostClassifier с LogLoss; LightGBM — конфигурируемая
альтернатива отдельного run. Ручной набор фиксируется до test:

- User/item/user-item counts за 1/7/14/30 дней по каждому event_type;
  время с последнего события, repeat counts, доли primary positives.
- User-category affinity, item-category и root-category counts; metadata as-of.
- Smoothed ratios прошлых событий: `(positive_count+alpha*prior)/(count+alpha)`;
  это отношение наблюдаемых фактов, не причинная вероятность конверсии показов.
- Expanding PIT target encoding категорий: alpha=10, min_history=5,
  prior из прошлого; empty prior=0.5. TE считает реальные прошлые primary и
  non-primary events, не synthetic impressions.
- MovieLens: прошлые mean/count ratings и genres; Yoochoose: session prefix
  length, recency, known category/brand. Неизвестные значения имеют defaults.

Агрегаты streaming/rolling по `[T-window,T)`, сначала scoring time batch,
потом update. Нулевые знаменатели не заменяются будущими averages.
Отбор: постоянные признаки и корреляции по train, importance через temporal inner
validation; HPO depth/learning_rate/l2/iterations по общему бюджету.
Baseline проходит такую же настройку, как Advanced.

Отбор признаков для Baseline/Adv1 использует только прошлый inner prefix
до lower-quantile .85 train, максимум50000 строк с полными query groups,
выбранными стабильным hash(seed,query_group_id). Это ресурсный cap отбора,
фиксируемый до полного HPO; финальный booster обучается на всех train rows.
Для Adv1 после отбора читаются top50 колонок, а не вся матрица500 кандидатов.
Число строк отбора сохраняется в run_manifest. Полное обучение при недостатке
RAM завершается явным resource error; уменьшение train скрытно не разрешено.

Реализация PIT агрегатов объединяет реальные события и уникальные query markers,
сворачивает целую секунду, затем применяет rolling closed=left. Marker имеет
нулевой счетчик и не становится наблюдением. Expanding counts вычитают всю
текущую секунду из cumulative sum. Запросы делятся на hash shards по entity;
каждая entity всегда целиком в одном shard, временная история не обрезается.
Статические списки genres дают стабильный ID первого genre для affinity и
отдельную длину списка; neural Item Tower использует все доступные genres.

## 3. Adv1: AutoML Feature Factory

Типизированная DSL генерирует ограниченную сетку: entity ∈ {user,item,user-item,
user-category}, window ∈ {1,3,7,14,30} дней, aggregation ∈ {count,sum,mean,min,max},
event filter ∈ доступных event_types. Дополнительно ratios/differences с безопасным
делением и фиксированным smoothing. Глубина выражений <=2, максимум 500 кандидатов
до отбора, top50 после отбора. Недоступные metadata не превращаются в признаки.

Каждый primitive принимает cutoff T и имеет provenance. Validator запрещает
CURRENT ROW, full-group без cutoff, target fields, будущие joins, затем запускает
prefix/mutation gate. Подозрительный train-vs-valid AUC инициирует проверку,
но не доказывает PIT. LLM/Featuretools — optional генераторы поверх той же DSL;
произвольный LLM Python/SQL не требуется. Selection fit только на inner train,
HPO — valid NDCG@10. Booster/sampling сопоставимы с Baseline. Логируются числа
созданных/отклоненных/выбранных признаков и ручные правки.

## 4. Adv2: гетерогенный GraphSAGE

Sparse heterogeneous graph: User (Session для Yoochoose), Item, Category,
Brand при наличии. Interaction edges по event_type, item-category/brand as-of,
category-parent из разрешенного дерева. Не создавать выдуманные brand nodes.
Reverse edges имеют тот же cutoff. Neighbor sampling fanout [15,10], 2 слоя,
hidden size по HPO. Target edge отсутствует.

Для RTX3050/4GiB Item ID представлен суммой двух независимых фиксированных
hash embeddings (65536 slots; модули65535 и65533), category/brand — одним
hash embedding. User ID используется только для поиска прошлого neighborhood;
его начальный vector — общий learned cold token, без отдельной случайной записи
для нового user. Hash mappings не обучаются по будущему catalog; collision и
parameter sharing — явный ресурсный tradeoff, размеры записываются в model manifest.
Доступные metadata и GraphSAGE neighborhood дают content/prefix контекст.

Индекс графа — удаляемый SQLite runtime. Семплер берет последние past interaction
neighbors с fanout15/10 и доступные category/brand/parent relations. Это
детерминированный recent-neighbor sampler, не равномерная выборка всех edges.
Устаревшие item-category связи отфильтровываются по последнему snapshot<T;
unknown tombstone сбрасывает старую связь. Для запросов с разными T строятся
раздельные subgraphs, чтобы один запрос batch не получил более поздние edges
другого. Рабочий neural batch ограничен64 query groups до HPO, без удаления rows.

GraphSAGE учит функцию агрегации соседних признаков; cold item получает
content/category features или OOV default. Cold user — prefix encoder/default,
не случайно обученная запись неизвестного ID. Metadata/node mappings не
подсматривают будущую активность. Loss основной BCE на общем sampler;
BPR — отдельная objective ablation. Score dot product плюс bias.

Стандартный LightGCN — отдельный transductive comparator, а не гарантированная
замена индуктивному GraphSAGE: он распространяет обучаемые user/item embeddings.
См. [GraphSAGE](https://arxiv.org/abs/1706.02216) и
[LightGCN](https://arxiv.org/abs/2002.02126). Cold-start преимущества проверяются.

Обучающий graph для каждого target — prefix/snapshot<T. Frozen train weights
могут обрабатывать ранее произошедшие test edges по общему replay; test labels
не обновляют weights. Daily base+delta snapshots ограничивают disk duplication.
Сохраненные embeddings для другого learner требуют temporal OOF.

## 5. Adv3: Sequential Two-Tower

User Tower: последние 50 past events, item embedding + event-type embedding
прошлых labels + position + log-bucket time-delta embedding. Два transformer
блока, causal и padding masks; hidden size64/128, heads2/4. Последний непустой
history state проецируется в d-dimensional user vector. Empty history получает
learned cold context token внутри collator; искусственного события в Parquet нет.

Item Tower: ID/OOV и доступные genre/category/brand embeddings с mean pooling
multivalue metadata, MLP64/128→d. MovieLens title/year optional; genome выключен
strict primary. Dot product дает score. Loss sampled BCE; BPR/contrastive —
отдельная ablation с теми же anchors/negatives. In-batch negatives — только
с prefix exclusion-проверкой и отдельным протоколом, не default.

Train leave_one_out = sample на каждом eligible train event с past prefix.
Valid/test last_event_only берутся из того же registry, что у Baseline/GNN.
Повторение target item в прошлом допустимо, сам target event недопустим.
Маска ограничивает attention; полную PIT-корректность проверяют также cutoff,
metadata joins, embedding provenance и quality gates.

Sequence collator читает готовые generator samples по sample_id из удаляемого
SQLite runtime index; candidate rows не размножают history в Parquet. Метаданные
исторического item берутся строго до timestamp самого history event; candidate
metadata — строго до query T. Mean pooling сохраняет все известные genres.
Хеш-таблицы ID/category/brand совпадают по ресурсному правилу с Adv2. Cold token
всегда первый и доступен attention; padding справа, последний реальный state
выбирается по длине history. Это устраняет all-masked строки для пустой истории.

## 6. Adv4: Graph-Sequential Fusion

History token: projection(concat(ID, metadata, GNN-prefix-embedding)).
Candidate Item Tower использует доступный GNN+metadata vector.
User Tower — causal transformer Adv3. Базовый loss BCE; optional
`L=BCE_ranking + lambda_graph*BCE_graph`, lambda_graph ∈ {0,0.1,1} по valid.

Двухэтапный default: GNN из Adv2, temporal out-of-fold embeddings для train
Fusion; final GNN fitted train для future inference. Для раннего train fold
нужен snapshot из предыдущего префикса, cold warmup без fitted GNN.
Graph representation не вычисляется моделью, уже видевшей target или более поздние
train события. Joint fusion — отдельный run и проверка. Граф/sequence имеют
независимые cutoff/fit metadata. Приемка: ablations graph-off, sequence-off,
metadata-off. Преимущество Fusion над Adv2/Adv3 не гарантировано.

Для Fusion upstream GraphSAGE имеет заранее заданные config width/layers и
dropout0.1,lr0.001,weight_decay1e-5. Это фиксированный prefix encoder, не
выбор teacher по outer valid/test. На train quantiles .5/.7/.85 и final train
обучаются отдельные checkpoints по5 полных epochs (smoke2), суммарный лимит
2 часа на seed; невыполненный checkpoint делает cell failed. Shared encoder
для HPO имеет seed42; финальные Fusion seeds43/44 получают собственные banks.
Весь upstream wall time и hashes включаются в manifest. Upstream seed42
вычитается из общего HPO wall budget (число trials ×2 часа); дополнительно
каждый trial ограничен2 часами. Final per-seed retraining учитывается отдельно.

Bank хранит max_fit_data_ts<fit_cutoff<=snapshot. Выбор строго snapshot<T;
на самой границе используется предыдущий checkpoint. Граф каждого запроса
все равно ограничен T, включая history-item representations: это текущие
на query T представления прошлых items. Исторические raw metadata сохраняют
собственный event cutoff. Нет плотной выгрузки всех history embeddings: они
считаются frozen encoder на batch, CPU cache ограничен128 запросами. Warmup
сохраняет все train anchors с нулевым graph vector. Auxiliary graph BCE обучает
проекции graph vectors в Fusion; frozen teacher weights не обновляются.

## 7. Дистилляция и stacking

Teacher для исторических OOF features выбирается по inner train folds. Outer
valid выбирает готовый student pipeline, но не участвует в teacher fit.
Внутри train — expanding temporal folds:
teacher fit предыдущего префикса, score следующего. Warmup без истории не
получает in-sample teacher score. Outer valid не превращается в student train.

`score_feature_stacking`: student GBDT получает PIT teacher score как feature;
teacher или causal score service нужен на inference. Замеряется суммарная latency.

`soft_target_distillation`: teacher score — soft target, не input student.
`L=alpha*BCE(y,p_student)+(1-alpha)*BCE(p_teacher,p_student)`, alpha выбирается
по inner train validation. Student-only inference возможен, качество измеряется.
Исторические teacher scores обязательны OOT в обоих режимах (PIT §12).

Калибровка нейросетей читает поздние valid scores из временного Parquet: статистики
и L-BFGS loss/gradient считаются проходами по chunks. Целиком загружать candidate
pools в RAM для двух параметров Platt не требуется.

Distillation использует тот же фиксированный temporal OOF GraphSAGE teacher
(config Adv2,5 epochs/full или2/smoke на prefix checkpoint), без выбора teacher
по outer valid/test. Два режима запускаются в разных experiment directories.
Teacher artifact хранит score/probability, availability, snapshot и fit_end
для каждого исходного query key; checksums связывают его с source и bank manifest.
Warmup сохраняется с score0,probability0.5,availability=false.

Для soft targets alpha∈{0.25,0.5,0.75,1} выбирается по binary LogLoss последнего
inner train блока (.85 boundary), с50-tree booster (smoke30), fit только на предыдущем префиксе.
Потом teacher mixture alpha*y+(1-alpha)*p_teacher обучает student на всех train
rows. Teacher score/availability не входят в input soft student. Stacking
обязательно включает оба teacher input fields. HPO student использует обычные
20 trials/early valid NDCG, calibration — позднюю valid. Upstream, alpha search
и final student costs учитываются вместе.

CatBoost RAM quota учитывает уже загруженный процесс: текущий RSS + min(512MiB,
65% доступной RAM) для дополнительных CatBoost allocations. Прежний абсолютный
512MiB порог неприменим к процессу с импортированным PyTorch (>1GiB сам по себе).
Предварительная проверка плотного fit остается обязательной; RAM quota backend
сама по себе не является жестким ограничителем всей памяти процесса.

## 8. Ресурсы, инференс и приемка

Нельзя материализовать user×item scores/dense adjacency. Embedding memory
O((U+I)*d), sparse graph O(E), bounded sequence attention O(batch*50²*d),
neighbor memory ограничена batch/fanout/depth. Квадрат по фиксированным 50 токенам
не означает квадрат по всему N. Sparse transductive graph сам по себе также
не обязан иметь O(N²) память.

Основной benchmark точно ранжирует locked 100 candidates; ANN не требуется.
Faiss retrieval — optional production experiment; latency/recall зависят от
размера/типа индекса, O(1) не обещается. Item embeddings пересчитываются при
изменении weights/доступного snapshot, а не навсегда.

Готовность: meaningful smoke fit/predict, конечные scores всех общих queries,
PIT gates, reproducing seeds, train-only preprocessing, unit/integration tests,
coverage>=80% ключевых модулей, config/code/environment и pool hashes.
Полное сравнение — 20 full-data cells и отчет; критерии experiment_plan.md.

## История изменений

- 2026-10-19 — 1.0 draft: единая нумерация/цели; удалены придуманные результаты;
  добавлены алгоритмы моделей, temporal OOF, зависимости distillation и приемка.

- 2026-10-19 — пользователь подтвердил план и переход к реализации; состояние выполнения отражено в project_status.md.
