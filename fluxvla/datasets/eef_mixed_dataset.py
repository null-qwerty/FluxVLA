# Copyright 2026 Limx Dynamics
"""Interleave several already-processed VLA datasets with fixed weights.

Why not DistributedBalancedRepeatingDataset: that wrapper assumes Arrow-backed
ParquetDataset sources (it indexes parquet row groups). The EgoDex EEF data is a
JSONL + tar-pack dataset, so mixed training needs a plain map-style dataset that
can front any source implementing len/__getitem__.

Each source builds its own transform chain, so every sample carries its own
statistics under "stats" and its own per-dimension action mask; the loss already
supports per-dimension masks, which is what lets the RoboCasa (all dims) and
EgoDex (12 arm dims) samples share one batch.
"""
from typing import Dict, List, Optional

import numpy as np

from fluxvla.engines.utils import DATASETS


@DATASETS.register_module()
class EEFMixedDataset:
    """Deterministic weighted interleaving of processed VLA datasets.

    Args:
        datasets: Dataset configs (each a dict with a "type").
        weights: Sampling weight per source; normalised to sum to one.
        epoch_length: Samples per epoch. Defaults to the largest source so one
            epoch is one pass over the biggest stream while smaller sources are
            wrapped as often as their weight requires.
    """

    def __init__(self,
                 datasets: List[Dict],
                 weights: Optional[List[float]] = None,
                 epoch_length: Optional[int] = None,
                 statistics: Optional[List[Optional[Dict]]] = None) -> None:
        import inspect

        from fluxvla.engines.utils import build_dataset_from_cfg

        if not datasets:
            raise ValueError('EEFMixedDataset needs at least one source.')
        self.datasets = [
            build_dataset_from_cfg(cfg) if isinstance(cfg, dict) else cfg
            for cfg in datasets
        ]
        self.source_lengths = [len(dataset) for dataset in self.datasets]
        if any(length <= 0 for length in self.source_lengths):
            raise ValueError('Every EEFMixedDataset source must be non-empty.')
        weights = ([1.0] * len(self.datasets)
                   if weights is None else [float(w) for w in weights])
        if len(weights) != len(self.datasets):
            raise ValueError('weights must match the number of datasets.')
        weights = np.asarray(weights, dtype=np.float64)
        if (weights <= 0).any():
            raise ValueError('weights must be positive.')
        self.weights = weights / weights.sum()

        total = int(epoch_length) if epoch_length else max(self.source_lengths)
        self.total_len = max(total, 1)

        # Evenly spaced source assignment: sample i -> source where the
        # cumulative weight interval contains (i + 0.5) / total.
        boundaries = np.cumsum(self.weights)
        positions = (np.arange(self.total_len) + 0.5) / self.total_len
        source = np.searchsorted(boundaries, positions, side='right')
        source = np.clip(source, 0, len(self.datasets) - 1)

        # Within a source, walk its samples in order and wrap around, so a
        # smaller stream is repeated exactly as often as its weight implies.
        local = np.zeros(self.total_len, dtype=np.int64)
        for index, dataset in enumerate(self.datasets):
            slots = np.flatnonzero(source == index)
            if slots.size:
                local[slots] = np.arange(slots.size) % self.source_lengths[
                    index]
        self._source = source.astype(np.int64)
        self._local = local

        # ParquetDataset.__getitem__ takes the statistics blob as a second
        # argument (normally supplied by the balanced wrapper); other sources
        # carry their statistics themselves. Detect instead of hard-coding.
        if statistics is None:
            statistics = [None] * len(self.datasets)
        if len(statistics) != len(self.datasets):
            raise ValueError('statistics must match the number of datasets.')
        self._statistics = statistics
        self._wants_statistics = [
            'dataset_statistics' in inspect.signature(
                dataset.__getitem__).parameters for dataset in self.datasets
        ]

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, index: int):
        if index < 0:
            index += self.total_len
        if not 0 <= index < self.total_len:
            raise IndexError(index)
        source = int(self._source[index])
        local = int(self._local[index])
        dataset = self.datasets[source]
        if self._wants_statistics[source]:
            return dataset.__getitem__(
                local, dataset_statistics=self._statistics[source])
        return dataset[local]
