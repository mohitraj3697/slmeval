"""Golden datasets: the cases every candidate is measured against."""

from __future__ import annotations

from slmeval.dataset.loader import Dataset, dataset_hash, load_dataset
from slmeval.dataset.models import Assertions, Case

__all__ = ["Assertions", "Case", "Dataset", "dataset_hash", "load_dataset"]
