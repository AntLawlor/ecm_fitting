
from collections.abc import Callable

from pathlib import Path
from numpy.typing import NDArray

import numpy as np

_CACHE_DIR = Path(__file__).resolve().parent / ".ocv_cache"
_SOC_INTERP = np.linspace(0, 100, 1001, endpoint=True)

def _load_or_compute_ocv(cell_code: str) -> NDArray:
    cache_file = _CACHE_DIR / f"{cell_code}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    else:
        raise ValueError("Where's the cache?!")

def get_ocv(
    cell_code: str = "APR18650M1B",
) -> Callable[[float | NDArray], float | NDArray]:
    """
    Return an interpolator that maps SOC (0-100) to OCV in volts.

    The processed OCV curve is cached to disk after the first call, so subsequent
    calls for the same cell_code skip the raw file read entirely.
    """
    ocv = _load_or_compute_ocv(cell_code)
    return lambda soc: np.interp(soc, _SOC_INTERP, ocv)


def get_soc_from_ocv(
    cell_code: str = "APR18650M1B",
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Return an interpolator that maps OCV (in volts) to SOC in percent (0-100)."""
    ocv = _load_or_compute_ocv(cell_code)

    sort_idx = np.argsort(ocv)
    ocv_sorted = ocv[sort_idx]
    soc_sorted = _SOC_INTERP[sort_idx]

    return lambda voltage: np.interp(voltage, ocv_sorted, soc_sorted)
