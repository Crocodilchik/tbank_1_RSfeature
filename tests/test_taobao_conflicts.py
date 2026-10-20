"""Observed Taobao category state must reset on simultaneous ambiguity."""

from pathlib import Path
import polars as pl

from src.generator.metadata import MetadataIndex
from src.generator.exporters.graph import graph_at, iter_metadata_edges
from src.generator.schemas import CATEGORIES_SCHEMA
from tests.test_adapters import fixtures


def test_conflicting_second_tombstone_and_recovery(tmp_path: Path) -> None:
    adapter = fixtures(tmp_path, "taobao")
    path = adapter.settings.raw_dir / adapter.settings.files["events"]
    path.write_text("1,10,2,pv,1511481600\n2,10,3,pv,1511481602\n3,10,4,pv,1511481602\n1,10,5,pv,1511481604\n1,10,5,pv,1511481605\n")
    metadata = adapter.metadata_lazy().collect()
    states = metadata.filter(pl.col("field_name") == "category_id").sort("valid_from")
    assert states["field_value"].to_list() == ["2", "-1", "5"]
    assert metadata.filter(pl.col("field_name") == "category_conflict").height == 1
    index = MetadataIndex(metadata.lazy(), tmp_path / "lookup.sqlite")
    assert index.category(10, 1511481602) == 2
    assert index.category(10, 1511481603) is None
    assert index.category(10, 1511481605) == 5
    index.close()
    edges = pl.concat(list(iter_metadata_edges(metadata.lazy(), pl.DataFrame(schema=CATEGORIES_SCHEMA), "taobao")))
    assert graph_at(edges.lazy(), 1511481603).collect().is_empty()
    assert graph_at(edges.lazy(), 1511481605).collect()["dst_id"].to_list() == [5]
    adapter.metadata_item_ids = pl.DataFrame({"item_id": [999]}).lazy()
    assert adapter.metadata_lazy().collect().is_empty()
