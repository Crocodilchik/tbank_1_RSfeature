"""Adapters on tiny adversarial inputs and bounded real raw samples."""

import json
from pathlib import Path

import polars as pl
import pytest

from src.config.loader import load_config
from src.generator.adapters.base import DataValidationError
from src.generator.adapters.registry import make_adapter
from src.generator.schemas import EVENTS_SCHEMA, ITEMS_SCHEMA


def fixtures(tmp_path: Path, dataset: str):
    """Create isolated source files preserving the true raw format."""
    original = load_config(dataset=dataset).dataset(dataset)
    raw = tmp_path / dataset / "raw"
    raw.mkdir(parents=True)
    settings = original.model_copy(update={"raw_dir": raw, "processed_dir": raw.parent / "processed"})
    data = {
        "retailrocket": {
            "events": "timestamp,visitorid,event,itemid,transactionid\n1420070401001,1,view,10,\n1420070401001,1,view,10,\n1420070401999,1,addtocart,11,\n1420070402000,1,transaction,10,5\n",
            "properties1": "timestamp,itemid,property,value\n1420070401000,10,categoryid,2\n1420070401000,10,available,1\n1420070401000,10,888,n3.5 abc\n",
            "properties2": "timestamp,itemid,property,value\n1420070402000,10,categoryid,3\n",
            "categories": "categoryid,parentid\n1,\n2,1\n3,99\n",
        },
        "taobao": {"events": "1,10,2,pv,1511481600\n1,10,2,pv,1511481600\n1,11,3,cart,1511481601\n1,10,2,buy,1511481602\n2,20,4,pv,-1\n2,20,4,pv,1512345600\n"},
        "yoochoose": {
            "clicks": "1,2014-09-01T00:00:00.001Z,10,1\n1,2014-09-01T00:00:00.001Z,10,1\n1,2014-09-01T00:00:00.999Z,11,S\n1,2014-09-01T00:00:01.000Z,10,2\n2,2014-09-01T00:00:01.000Z,12,10000000\n",
            "buys": "1,2014-09-01T00:00:02.000Z,10,100,1\n1,2014-09-01T00:00:03.000Z,10,100,0\n",
            "official_test": "99,2014-09-16T00:00:00.000Z,10,0\n",
            "source_readme": "fixture description",
        },
        "movielens": {
            "ratings": '"userId","movieId","rating","timestamp"\n1,10,2.5,2005-01-01 00:00:01\n1,11,3.0,2005-01-01 00:00:01\n1,12,4.0,2005-01-01 00:00:02\n1,12,4.0,2005-01-01 00:00:02\n',
            "movies": 'movieId,title,genres\n10,Film (2000),Drama|Comedy\n11,Other,(no genres listed)\n12,Third (2001),Action\n',
            "links": "movieId,imdbId,tmdbId\n10,100,\n11,101,201\n12,102,202\n",
            "tags": "userId,movieId,tag,timestamp\n1,10,tag,2005-01-01 00:00:03\n",
            "genome_scores": "movieId,tagId,relevance\n10,1,0.9\n",
            "genome_tags": "tagId,tag\n1,tag\n",
        },
    }
    for role, value in data[dataset].items():
        (raw / original.files[role]).write_text(value, encoding="utf-8")
    return make_adapter(settings, row_limit=1000)


@pytest.mark.parametrize("dataset", ["retailrocket", "taobao", "yoochoose", "movielens"])
def test_normalization(tmp_path: Path, dataset: str) -> None:
    adapter = fixtures(tmp_path, dataset)
    assert adapter.validate() == []
    result = adapter.normalize()
    assert result.schema == EVENTS_SCHEMA
    assert result["event_id"].n_unique() == result.height
    assert result["source_row_id"].min() >= 1
    assert result["target_cutoff_ts"].to_list() == result["timestamp"].to_list()
    assert result["is_synthetic"].sum() == 0
    assert result["split"].unique().to_list() == ["unassigned"]
    assert result["user_type"].unique().to_list() == ["session" if dataset == "yoochoose" else "persistent"]
    assert result.height == (5 if dataset == "yoochoose" else 3)
    for group in result.partition_by("user_id"):
        assert group["user_event_seq"].to_list() == list(range(1, group.height + 1))
    assert pl.concat(list(adapter.iter_normalized(2))).equals(result)
    with pytest.raises(DataValidationError, match="max_rows"):
        adapter.normalize(max_rows=1)
    with pytest.raises(ValueError):
        list(adapter.iter_normalized(0))
    assert adapter.normalize(partition=adapter.normalize_lazy().limit(1)).height == 1
    metadata = adapter.metadata_lazy().collect()
    assert metadata.schema == ITEMS_SCHEMA
    assert metadata.unique(subset=["item_id", "field_name", "valid_from"]).height == metadata.height
    for row in metadata.iter_rows(named=True):
        assert json.loads(row["field_value"]) is not None


def test_retail_temporal_properties_and_tree(tmp_path: Path) -> None:
    adapter = fixtures(tmp_path, "retailrocket")
    metadata = adapter.metadata_lazy().collect().filter(pl.col("field_name") == "category_id")
    assert metadata.sort("valid_from")["field_value"].to_list() == ["2", "3"]
    tree = adapter.categories()
    assert tree.filter(pl.col("category_id") == 2)["full_path"].item() == "1/2"
    assert tree.filter(pl.col("category_id") == 3)["parent_id"].item() is None
    path = adapter.settings.raw_dir / adapter.settings.files["categories"]
    path.write_text("categoryid,parentid\n1,2\n2,1\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="cycle"):
        adapter.categories()
    path.write_text("categoryid,parentid\n1,\n1,\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="duplicate"):
        adapter.categories()


def test_retail_partitioned_metadata_equivalence_and_errors(tmp_path: Path) -> None:
    adapter = fixtures(tmp_path, "retailrocket")
    adapter.row_limit = None
    path = adapter.settings.raw_dir / adapter.settings.files["properties2"]
    path.write_text("timestamp,itemid,property,value\n1420070401000,10,categoryid,2\n1420070402000,10,categoryid,3\n1420070402999,10,categoryid,7\n1420070401000,11,888,abc def\n", encoding="utf-8")
    keys = ["item_id", "field_name", "valid_from"]
    expected = adapter.metadata_lazy().collect().sort(keys)
    actual = pl.concat(list(adapter.iter_metadata(1, shards=2))).sort(keys)
    assert actual.equals(expected)
    assert actual.filter((pl.col("field_name") == "category_id") & (pl.col("valid_from") == 1420070402))["field_value"].item() == "7"
    for options in ({"chunk_rows": 0}, {"shards": 0}):
        with pytest.raises(ValueError):
            list(adapter.iter_metadata(**options))
    path.write_text("timestamp,itemid,property,value\n1420070401000,10,categoryid,99\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="conflicting"):
        list(adapter.iter_metadata(1, shards=2))
    path.write_text("timestamp,itemid,property,value\n1420070401000,-1,available,1\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="required"):
        list(adapter.iter_metadata(1, shards=2))


def test_retail_conflicts_and_invalid_year(tmp_path: Path) -> None:
    adapter = fixtures(tmp_path, "retailrocket")
    p = adapter.settings.raw_dir / adapter.settings.files["properties2"]
    p.write_text("timestamp,itemid,property,value\n1420070401000,10,categoryid,99\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="conflicting"):
        adapter.metadata_lazy()
    p.write_text("timestamp,itemid,property,value\n1420070402000,10,available,2\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="invalid"):
        adapter.metadata_lazy()
    p = adapter.settings.raw_dir / adapter.settings.files["events"]
    p.write_text("timestamp,visitorid,event,itemid,transactionid\n0,1,view,10,\n", encoding="utf-8")
    assert "2015" in adapter.validate()[0]


def test_taobao_modes_and_missing_timestamp(tmp_path: Path) -> None:
    adapter = fixtures(tmp_path, "taobao")
    path = adapter.settings.raw_dir / adapter.settings.files["events"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("1,10,99,pv,1511481603\n")
    adapter.settings = adapter.settings.model_copy(update={"category_id_availability": "static_assumed"})
    with pytest.raises(DataValidationError, match="not static"):
        adapter.metadata_lazy()
    adapter.settings = adapter.settings.model_copy(update={"category_id_availability": "observed_asof"})
    assert adapter.metadata_lazy().collect()["availability_class"].unique().to_list() == ["observed_snapshot"]
    path.write_text("1,10,2,pv,\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="null"):
        adapter.normalize()


@pytest.mark.parametrize("dataset", ["retailrocket", "taobao", "yoochoose", "movielens"])
def test_early_user_sampling_keeps_complete_histories_and_source_ids(tmp_path: Path, dataset: str) -> None:
    adapter = fixtures(tmp_path, dataset)
    full = adapter.normalize()
    adapter.observation_user_modulus = 2
    selected = adapter.normalize()
    assert selected.equals(full.filter(pl.col("user_id") % 2 == 0))
    if dataset == "yoochoose":
        assert selected["source_row_id"].to_list() == [5]


def test_yoochoose_prefix_modes_and_official_isolation(tmp_path: Path) -> None:
    adapter = fixtures(tmp_path, "yoochoose")
    primary = adapter.normalize()
    assert 99 not in primary["user_id"]
    with pytest.raises(ValueError, match="disabled"):
        adapter.official_observations_lazy()
    adapter.settings = adapter.settings.model_copy(update={"include_official_test": True})
    assert adapter.official_observations_lazy().collect()["user_id"].to_list() == [99]
    modes = pl.concat(list(adapter.iter_metadata(1)))
    item = modes.filter((pl.col("item_id") == 10) & (pl.col("field_name") == "category_id")).sort("valid_from")
    assert item["field_value"].to_list() == ["1", "1"]  # equal counts choose smaller category
    adapter.row_limit = None
    with pytest.raises(ValueError, match="iter_metadata"):
        adapter.metadata_lazy()
    with pytest.raises(ValueError):
        list(adapter.iter_metadata(0))


def test_yoochoose_conflicting_context_and_invalid_purchase(tmp_path: Path) -> None:
    adapter = fixtures(tmp_path, "yoochoose")
    path = adapter.settings.raw_dir / adapter.settings.files["clicks"]
    path.write_text("1,2014-09-01T00:00:00.000Z,10,1\n1,2014-09-01T00:00:00.000Z,10,2\n", encoding="utf-8")
    assert adapter.clicks().collect()["category"].to_list() == ["0"]
    assert adapter.metadata_lazy().collect().is_empty()
    path = adapter.settings.raw_dir / adapter.settings.files["buys"]
    path.write_text("1,2014-09-01T00:00:00.000Z,10,-1,1\n", encoding="utf-8")
    assert "invalid" in adapter.validate()[0]


def test_movielens_rating_boundaries_and_tags(tmp_path: Path) -> None:
    adapter = fixtures(tmp_path, "movielens")
    assert adapter.normalize().sort("item_id")["label"].to_list() == [0, 1, 2]
    assert adapter.tags_lazy().collect()["timestamp"].item() == 1104537603
    path = adapter.settings.raw_dir / adapter.settings.files["ratings"]
    path.write_text("userId,movieId,rating,timestamp\n1,10,2.7,2005-01-01 00:00:00\n", encoding="utf-8")
    assert "half-star" in adapter.validate()[0]


def test_bad_events_and_missing_sources(tmp_path: Path) -> None:
    adapter = fixtures(tmp_path, "taobao")
    path = adapter.settings.raw_dir / adapter.settings.files["events"]
    path.write_text("1,10,2,unsupported,1511481600\n", encoding="utf-8")
    assert adapter.validate()
    path.write_text("-1,10,2,pv,1511481600\n", encoding="utf-8")
    assert "invalid required" in adapter.validate()[0]
    path.unlink()
    with pytest.raises(FileNotFoundError):
        adapter.load_raw()
    with pytest.raises(ValueError):
        make_adapter(adapter.settings, row_limit=0)
    assert adapter.categories().is_empty()


@pytest.mark.real_data
@pytest.mark.parametrize("dataset", ["retailrocket", "taobao", "yoochoose", "movielens"])
def test_real_bounded_sample(dataset: str) -> None:
    adapter = make_adapter(load_config(dataset=dataset).dataset(dataset), row_limit=1000)
    result = adapter.normalize()
    assert result.height > 0
    assert result.schema == EVENTS_SCHEMA
    assert result["event_id"].n_unique() == result.height
