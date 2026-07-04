"""Market data: schema, quality control, providers, synthetic generation, storage."""

from titan.data.providers import DataProvider, build_provider
from titan.data.quality import DataQualityReport, assess_quality
from titan.data.schema import normalize_ohlcv
from titan.data.store import MarketDataset, MarketDataStore
from titan.data.synthetic import SyntheticMarket

__all__ = [
    "DataProvider",
    "DataQualityReport",
    "MarketDataStore",
    "MarketDataset",
    "SyntheticMarket",
    "assess_quality",
    "build_provider",
    "normalize_ohlcv",
]
