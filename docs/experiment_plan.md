# План экспериментов

Версия: 1.0
Статус: approved
Дата последнего обновления: 2026-10-19

Контракт: [PIT](point_in_time_policy.md); модели:
[recsys_architecture.md](recsys_architecture.md). Пороги ниже — предлагаемый
исследовательский протокол, не измеренные результаты. Полная матрица еще
не выполнена; реальные smoke-прогоны отражены в project_status.md.

## 1. Основная матрица: четыре датасета × пять моделей

| Датасет / основной target | Baseline GBDT | Adv1 AutoML | Adv2 GNN | Adv3 Seq | Adv4 Fusion |
|---|---|---|---|---|---|
| RetailRocket / transaction | обязательно | обязательно | обязательно | обязательно | обязательно |
| Taobao / buy | обязательно | обязательно | обязательно | обязательно | обязательно |
| Yoochoose / click, own temporal split | обязательно | обязательно | обязательно | обязательно | обязательно |
| MovieLens 20M / rating>=4 | обязательно | обязательно | обязательно | обязательно | обязательно |

Это 20 конфигураций на полных данных; итог каждой — три training seeds
{42,43,44}. Eval pool seed=42 одинаковый для моделей и training seeds. Датасеты
нельзя ранжировать между собой по абсолютному NDCG: цели и политики различны.
Дистилляция и ablations — сверх матрицы, с отдельными run_id/mode.

## 2. Разбиения и единица сравнения

Splits фиксирует генератор: lower timestamp quantiles .8/.9 для RetailRocket и
MovieLens; Taobao UTC 8+1+1 день; Yoochoose train<2014-09-01, valid до09-15,
test до10-01. Одинаковые split/config/raw fingerprints для всех пяти моделей.
Main test содержит последний eligible primary event на user/session в сплите.
Каждая группа — один ground truth и 99 уникальных sampled negatives по PIT §8.
Обязательный oracle исключает primary positives данного eval-сплита; MovieLens
также исключает все оцененные фильмы. Это часть определения экзамена, не feature.

Модели оценивают все locked groups, включая cold users/items и empty histories.
Нельзя удалять неудобные группы одной модели. Failure одной модели — failure cell,
не улучшение среднего за счет удаления строк. Если sampler не набрал 99 negatives,
skip одинаков для всех моделей с причиной/coverage. Публикуются число активных
user/session, eligible targets, dropped groups и доля cold cohort.

Основной режим real_only без personas, искусственных иерархий, label flips и
genome. Ненаблюдаемые train pairs нужны всем моделям и имеют одинаковый prefix
sampler; они не выдаются за реальные отрицательные показы. Train anchor registry
фиксирован для всех моделей. История replay общая, frozen train weights;
прошлые valid/test события можно использовать как контекст строго до T.

## 3. Tuning и защита test

Inner expanding train folds: границы lower quantiles .5/.7/.85 train timestamps;
fold fit на предыдущем префиксе, следующие блоки для feature selection/OOF.
Warmup до первой fit-границы не получает embeddings/teacher из будущего;
используется cold default либо документированная общая OOF train subset.
Если subset нужен для Fusion/teacher, его сравнивают с Baseline, переобученным
на том же subset; основной end-to-end run не заменяется этим сравнением молча.

Outer valid делится по времени: первые 60% упорядоченных valid timestamps
(lower quantile, без разрезания равных секунд) для HPO, оставшиеся — calibration.
Tuning 20 trials на модель/датасет, максимум 2 часа на trial; одинаковый host,
доступный CPU/GPU и лимит потоков. Эта квота предварительна: hardware probe в
итерации 1 может изменить ее ОДИНАКОВО для всех моделей до начала HPO; изменение
фиксируется config hash и историей, а не подстраивается после test.
Time for feature search и GNN pretraining входит в бюджет Adv1/Fusion.
Одинаковый trial cap не доказывает одинаковую вычислительную стоимость;
публикуются фактические wall time и расходы каждого компонента.

Early stopping: patience3 neural epochs /100 booster iterations; caps50 epochs /
2000 trees. Objective выбора — macro NDCG@10 на HPO-valid; при равенстве (1e-8)
предпочесть меньше fit time, затем меньший trial index. Feature generation,
scaling и selection не читают outer valid labels до применения модели.
После HPO фиксируются гиперпараметры, модель переобучается только train на трех
seeds. Test не используется для выбора epochs, features, teacher или ablations.

Параметры начального поиска: GBDT depth{4,6,8,10}, lr log-uniform[.01,.2],
l2 log-uniform[1,20]; neural d{64,128}, lr log-uniform[1e-4,3e-3],
dropout{0,.1,.2}, weight_decay log-uniform[1e-6,1e-3], layers{1,2}.
Graph fanout фиксирован [15,10], sequence max50; AutoML top{25,50},
Fusion lambda_graph{0,.1,1}. Search-space edits до test отражаются в run manifest.

## 4. Метрики и детерминизм

K∈{3,5,10,20}; primary NDCG@10. Sort scores descending, tie candidate_item_id
ascending. На группу с одним relevant item его rank r начиная с1:

- NDCG@K = 1/log2(r+1), если r<=K, иначе0; IDCG=1.
- Recall@K = HitRate@K = 1[r<=K]. В single-positive sampled pool они совпадают;
  это ожидаемо, не две независимые проверки.
- MRR = 1/r без усечения.
- AUC — macro per-group ROC AUC (один positive, 99 negatives), tied pair credit0.5.
- LogLoss — среднее бинарного loss по всем 100 candidates всех групп, p clipped
  в [1e-7,1-1e-7]. Эти AUC/LogLoss условны на sampler и prevalence1/100,
  не означают calibrated purchase/click probability реальных impressions.

Для score-only/BPR моделей вероятность получается Platt calibration
`sigmoid(a*score+b)`, a>=0, fit только на позднем calibration-valid; primary ranking
считается по исходному score. Для BCE/GBDT также применяется тот же calibration
протокол для сравнения LogLoss; raw LogLoss можно показать отдельно.
Empty/one-class calibration — явный undefined/error, не подставлять выдуманную AUC.
Test calibration model обучен раньше любого test target. Calibrated метрики на
самом calibration-valid только diagnostic; они не являются отложенной оценкой.

User/session macro averaging по группам, затем mean/std по training seeds.
Prediction schema: eval_group_id:int64, candidate_item_id:int64, score:float64,
probability:float64, model:string, training_seed:int64. Ключ включает model/seed.
Metric records содержат dataset/mode/split/cohort/metric/K/value/n_groups,
training_seed, config_hash, pool_hash. NaN/inf scores — fatal; отсутствующая
метрика (например, пустая cohort) — null с причиной и n_groups=0.

## 5. Статистическая проверка и критерии успеха

Парное сравнение на одних группах; cluster bootstrap по user/session, 2000
resamples, seed42, 95% percentile CI для разности NDCG@10. При нескольких seeds
сначала усреднить score/метрику каждой группы по seeds, затем ресемплировать
user clusters; std training seeds показывать отдельно. Holm correction alpha.05
для четырех Advanced-vs-Baseline проверок внутри датасета. Report содержит
absolute delta и relative delta, при Baseline=0 relative=null.

Предлагаются два способа подтвердить исходную гипотезу:

1. Качество: хотя бы один Advanced дает >=2% относительного прироста NDCG@10
   и положительную нижнюю границу CI с корректировкой значимости хотя бы на двух
   датасетах. Другие датасеты и регрессии публикуются, не скрываются.
2. Сопоставимое качество при меньшем ручном труде: нижняя граница relative delta
   не ниже −1% хотя бы на двух датасетах и измеренные ручные затраты <=50% Baseline.
   Human time измеряется журналом работ, а не wall time обучения; без журнала
   экономия труда не считается доказанной.

Нулевое преимущество или проигрыш — допустимый научный вывод, но не доказательство
улучшения. Техническая завершенность (все модели и отчеты) не означает успех
исследовательской гипотезы. PIT/fairness gates обязательны для обоих способов.

Журнал затрат: время общих инфраструктурных работ отдельно; на модель — минуты
ручного создания/правок признаков, число ручных primitives, dataset-specific
веток и настроек, время tuning/fit/inference, peak RAM/VRAM, disk artifacts,
p50/p95 latency при одном batch/query protocol. Нельзя приписывать всю общую
инфраструктуру только Baseline, а Advanced считать бесплатным.

## 6. Обязательные ablations

| Фактор | Контроль / варианты | Правило сравнения |
|---|---|---|
| Синтетика | real_only; extra train negatives; personas; full enrichment | Одинаковые реальные primary eval pools, anchors и seeds |
| Повторы | dataset default; unseen_only; allow_previous | Общие targets, eligible при обеих политиках; новые pool hashes, coverage отдельно |
| Evaluation negatives | random; popularity_biased; hard | Все модели одной ветки читают один новый pool |
| Oracle horizon | eval-split positives; вся будущая выгрузка | Только evaluator, не train inputs |
| Fallback | primary; отдельные fallback cohorts | Не объединять primary и fallback метрики |
| Taobao category | observed_asof primary; metadata-off ablation | Static mapping опровергнут полным scan; unknown rate отдельно |
| Yoochoose источник | основной own split click; official late-window click | Разные session populations, результаты отдельно |
| Yoochoose target | основной click; own-split purchase supplementary | Никакого смешанного purchase/click ground truth |
| MovieLens genome | strict off; static_assumed diagnostic | Второй результат не объявлять доказанным strict PIT |
| Fusion components | graph-off; sequence-off; metadata-off | Общие temporal OOF artifacts/queries |
| Objective | BCE; BPR/contrastive | Общие anchors/negatives и calibration rules |

Сначала проводятся четыре synthetic ветки для пяти моделей на каждом датасете;
реалистичный объем и лимиты фиксируются после resource probe, до test. Если
какая-либо ветка не выполнена, она помечается incomplete, не заменяется примером.
Другие ablations допускают preregistered сокращение числа моделей по valid;
правило выбора записать до просмотра test, явно ограничить область выводов.
Дополнительные train negatives используют тот же механизм sampled pairs, большее
число (8 вместо4); это отличает ветку от обязательного contrastive sampler.

Runner `run_synthetic_ablations.py` переиспользует primary real_only control;
три дополнительные ветки имеют отдельные configs, processed roots и experiments
roots. Config snapshots сохраняются по версии исходных YAML и переносятся вместе
с проектом. Перед обучением проверяются exact train-anchor hash и оба pool hashes.
Несовпадение блокирует сравнение. В summary сохраняются все80 ожидаемых cells;
табличный delta к real_only той же модели пока описательный, без вывода значимости.

2026-10-20, до появления full model scores: objective сравнивается для всех
трёх neural архитектур (Adv2/3/4), BCE control против BPR и contrastive. Fusion
дополнительно имеет graph-off, sequence-off и metadata-tower-off. Последний
отключает category/brand/genre в Item Tower; graph encoder остаётся прежним,
поэтому это не полная Taobao metadata-off ablation. Полное отключение metadata
во всех компонентах остаётся отдельной незавершённой задачей.
Выбор архитектур фиксирован заранее и не зависит от test quality.

Эти девять контрастов на dataset образуют одну Holm family (отсутствующий
контраст имеет p=1). Paired user-cluster bootstrap использует2000 resamples,
seed42, сначала усредняет метрики группы по трём seeds. Canonical config,
train anchors и primary eval pools сохраняются; варианты получают отдельные
model IDs и artifact roots. Fusion варианты обязаны читать те же OOF bank
manifests/weights, что завершённый primary Adv4, с проверкой config/parent/
seed/checksums. Отсутствующий либо изменённый donor запрещает запуск варианта.
Собственное время варианта и ранее затраченный shared upstream fit указываются
раздельно. Эти48 cells (включая12 primary controls) не подменяют другие ablations.

## 7. Отчет и воспроизводимость

Для каждого датасета — таблица всех пяти моделей, NDCG/Recall/HR/AUC/LogLoss/MRR,
seed mean/std, CI/delta к Baseline, cold user/item/short-session/long-tail slices,
coverage/skip причины и ресурсные затраты. Official/fallback/purchase/diagnostic
отчеты физически и визуально отделены от primary. Test неизменяем, hashes проверены.

Результаты primary: experiments/{model}/{dataset}/{full|smoke}/seed_{seed}/
test_metrics.parquet,test_predictions.parquet; run_manifest.parquet в корне run; сводный docs/experiment_results.md
создается после фактических прогонов. Manifest: code revision, package versions,
Python/OS/device, raw/config/pool/feature hashes, seeds, clocks/cutoffs, policy/schema
versions, search trials и elapsed time. Неудачные/прерванные cells тоже учитываются.

## 8. Исполнение и учет незавершенных cells

Evaluation collector проверяет parent dataset manifest,config и candidate keys
каждого seed, пересчитывает метрики по predictions. Smoke собирается отдельным
режимом; ожидается seed42, основной full требует42/43/44. Missing/failed/running
cells остаются строками матрицы с null scores. Метрики разных pools не усредняются.

Long-tail threshold:80-й lower quantile числа реальных train interactions/item;
cohort на query T имеет положительную past popularity<=этого порога. Cold items
с нулевым past count учитываются в event-cold, отдельно от long tail. Train-cold
сравнивает ID с train inventory; event-cold использует strict prefix counts.
Yoochoose short-session: past history<=2. Пустая cohort: n_groups=0 и null metric.
Парные95% CI считаются по средним per-group метрикам seeds; двухсторонний
bootstrap sign p-value с поправкой (count+1)/(B+1), затем Holm для4 contrasts.
CI публикуется вместе с adjusted p; гипотеза требует и CI_low>0, и Holm p<.05.

Журнал subprocess запусков сохраняет wall,peak RSS,exit code и diagnostic log
для каждого generator/model cell; Windows GPU memory может быть unavailable
и тогда публикуется null с причиной. Ненаблюдавшиеся human minutes/latency
не заменяются wall time и отмечаются unmeasured.

Cold-start counts вычисляются merge двух упорядоченных потоков (events и queries)
с условием event timestamp<query timestamp. В памяти остаются query counters и
ограниченный batch, а не cumulative column на каждый исходный event. Native
sort может spill в run-owned runtime directory; после расчёта временные файлы удаляются.

## История изменений

- 2026-10-19 — 1.0 draft: добавлены матрица4×5, метрики/калибровка, HPO/test lock,
  ablations, paired uncertainty, измеримые критерии качества и ручных затрат.

- 2026-10-19 — пользователь подтвердил план и переход к реализации; состояние выполнения отражено в project_status.md.
