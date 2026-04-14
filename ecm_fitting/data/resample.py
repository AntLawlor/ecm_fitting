from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from ecm_fitting.data.segment import DataSegment

class ResampleStrategy(ABC):
    """Abstract base for DataSegment resampling strategies."""

    @abstractmethod
    def resample_time(self, segment: DataSegment) -> np.ndarray:
        """
        Return a new time array for ``segment``.

        The returned array must be sorted, unique, and within
        ``[segment.time[0], segment.time[-1]]``.
        """
