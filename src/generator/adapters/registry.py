"""Closed adapter registry matching the four approved dataset contracts."""

from src.config.loader import DatasetSettings
from src.generator.adapters.base import BaseAdapter
from src.generator.adapters.movielens import MovieLensAdapter
from src.generator.adapters.retailrocket import RetailRocketAdapter
from src.generator.adapters.taobao import TaobaoAdapter
from src.generator.adapters.yoochoose import YoochooseAdapter


def make_adapter(settings: DatasetSettings, row_limit: int | None = None) -> BaseAdapter:
    """Construct the explicit dataset adapter without dynamic code loading."""
    classes = {"retailrocket": RetailRocketAdapter, "taobao": TaobaoAdapter,
               "yoochoose": YoochooseAdapter, "movielens": MovieLensAdapter}
    return classes[settings.name](settings, row_limit=row_limit)
