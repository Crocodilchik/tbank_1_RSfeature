"""Build and execute the results notebook exclusively from verified saved tables."""
from __future__ import annotations
import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for name, relative in (("JUPYTER_RUNTIME_DIR", ".venv/runtime/notebooks"),
                           ("IPYTHONDIR", ".venv/runtime/ipython"),
                           ("MPLCONFIGDIR", ".venv/runtime/matplotlib")):
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    os.environ["JUPYTER_PATH"] = str(root / ".venv/share/jupyter")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["POLARS_MAX_THREADS"] = "4"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    import nbformat as nbf
    from nbclient import NotebookClient
    notebook = nbf.v4.new_notebook(metadata={"kernelspec": {
        "display_name": "PIT Recsys (.venv)", "language": "python", "name": "pit-recsys"},
        "language_info": {"name": "python", "version": "3.12"}})
    cells = [
        ("markdown", """# Проверяемые результаты экспериментов

Notebook читает сохранённые Parquet-таблицы collector. Full и smoke разделены.
Незавершённые ячейки не получают оценок из smoke. Этот документ обновляет снимок
состояния при каждом исполнении; полная приёмка включает отдельные ablations.
"""),
        ("code", """from pathlib import Path
from datetime import datetime, timezone
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from IPython.display import display, Markdown
root = Path.cwd()
assert (root / 'src').is_dir(), 'Run with project root as working directory'
summary = root / 'experiments/summary'
figures = root / 'docs/figures'
figures.mkdir(parents=True, exist_ok=True)
full = pl.read_parquet(summary / 'full/cells.parquet')
smoke = pl.read_parquet(summary / 'smoke/cells.parquet')
display(Markdown('Снимок UTC: ' + datetime.now(timezone.utc).isoformat()))
display(full.select('dataset', 'model', 'status', 'n_seeds', 'n_groups', 'reason').to_pandas())
"""),
        ("code", """datasets = ['retailrocket', 'taobao', 'yoochoose', 'movielens']
models = ['baseline', 'adv1', 'adv2', 'adv3', 'adv4']
grid = np.zeros((len(datasets), len(models)), dtype=int)
labels = {}
for row in full.iter_rows(named=True):
    i, j = datasets.index(row['dataset']), models.index(row['model'])
    status = row['status']
    grid[i, j] = 3 if status == 'complete' else (1 if 'running' in status else (2 if status in ['failed', 'invalid_run'] else 0))
    labels[i, j] = 'data running' if status == 'running_dataset' else status.replace('_', '\\n')
fig, ax = plt.subplots(figsize=(11, 4.5), layout='constrained')
colors = ListedColormap(['#e2e8f0', '#fde68a', '#fecaca', '#bbf7d0'])
ax.imshow(grid, cmap=colors, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4), aspect='auto')
ax.set_xticks(range(len(models)), models)
ax.set_yticks(range(len(datasets)), datasets)
for (i, j), text in labels.items():
    ax.text(j, i, text, ha='center', va='center', fontsize=9)
completed = full.filter(pl.col('status') == 'complete').height
ax.set_title(f'Full matrix: {completed}/20 verified cells; 3 seeds required')
fig.savefig(figures / '02_experiment_completion.png', dpi=160)
display(fig)
plt.close(fig)
"""),
        ("markdown", """## Ограниченный smoke

Пять архитектур проверены на одном малом RR prefix и одинаковых кандидатах.
Один seed и 12 test groups не позволяют делать вывод о преимуществе подхода.
Показатели Baseline/AutoML=1 отражают лёгкость этого prefix.
"""),
        ("code", """rr = smoke.filter((pl.col('dataset') == 'retailrocket') & (pl.col('status') == 'complete'))
assert rr.height == 5 and rr['pool_hash'].n_unique() == 1
assert rr['n_groups'].unique().to_list() == [12]
fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
bars = ax.bar(rr['model'].to_list(), rr['NDCG@10_mean'].to_numpy(), color=['#475569', '#0f766e', '#2563eb', '#7c3aed', '#db2777'])
ax.bar_label(bars, fmt='%.4f', padding=3)
ax.set_ylim(0, 1.15)
ax.set_ylabel('NDCG@10')
ax.set_title('RR smoke only: 12 groups, seed 42; no full benchmark claims')
fig.savefig(figures / '02_smoke_ndcg.png', dpi=160)
display(fig)
plt.close(fig)
display(rr.select('model', 'NDCG@10_mean', 'AUC_mean', 'LogLoss_mean', 'wall_seconds').to_pandas())
"""),
        ("code", """cohorts = pl.read_parquet(summary / 'smoke/metrics.parquet').filter((pl.col('metric') == 'NDCG@10') & (pl.col('cohort') != 'all'))
display(Markdown('### Cold-start smoke: пустой cohort даёт undefined, а не нулевой score'))
display(cohorts.select('dataset', 'model', 'cohort', 'n_groups', 'value', 'reason').to_pandas())
contrasts = pl.read_parquet(summary / 'full/comparisons.parquet')
if contrasts.height:
    display(contrasts.to_pandas())
else:
    display(Markdown('Full contrasts, confidence intervals и Holm correction пока недоступны: нужны завершённые общие full pools и seeds.'))
"""),
        ("code", """students_path = summary / 'distillation_smoke/cells.parquet'
if students_path.exists():
    students = pl.read_parquet(students_path)
    display(students.select('dataset', 'model', 'status', 'NDCG@10_mean', 'AUC_mean', 'n_groups').to_pandas())
    for model in students['model'].unique():
        contracts = list((root / 'experiments/distillation' / model / 'retailrocket/smoke').glob('seed_*/inference_contract.parquet'))
        for path in contracts:
            display(pl.read_parquet(path).to_pandas())
display(Markdown('Soft-target alpha=1 означает отказ от teacher mixture на inner-train probe. Stacking требует teacher на inference. Эти smoke-прогоны не доказывают пользу дистилляции.'))
"""),
        ("code", """journal_path = root / 'experiments/execution/journal.parquet'
if journal_path.exists():
    journal = pl.read_parquet(journal_path)
    display(journal.select('dataset', 'stage', 'model', 'attempt', 'status', 'wall_seconds', 'peak_rss_bytes', 'cpu_seconds', 'reason').to_pandas())
display(Markdown('RAM — максимум суммы RSS дерева процесса с дискретным опросом; это наблюдавшийся пик. Running — незавершённое измерение. Null означает отсутствие измерения. Human minutes не измерены; вывод об экономии ручного труда пока не подтверждён.'))
"""),
        ("markdown", """## Что требуется для завершения

Полная матрица 4 datasets × 5 моделей × 3 seeds, обязательные synthetic/repeat/
negative/objective/component ablations и дополнительные диагностические ветки.
Гипотеза о преимуществе Advanced пока не проверена. Сохраняются отрицательные и
нулевые результаты; test не используется для изменения пространств HPO.
"""),
    ]
    for index, (kind, source) in enumerate(cells):
        create = nbf.v4.new_code_cell if kind == "code" else nbf.v4.new_markdown_cell
        cell = create(source=source.strip())
        cell["id"] = f"results-{index:02d}"
        notebook.cells.append(cell)
    destination = args.output or root / "notebooks/02_experiments_results.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(notebook, kernel_name="pit-recsys", timeout=180,
                            resources={"metadata": {"path": str(root)}})
    try:
        client.execute()
    finally:
        nbf.write(notebook, destination)
    print(destination)


if __name__ == "__main__":
    main()
