from pathlib import Path
from numpy.typing import NDArray

import pickle


_CACHE_DIR = Path(__file__).resolve().parent / ".hppc_cache"

def dummy_data() -> dict[str, NDArray]:
    """Dummy HPPC signal for a LGP/Gr 18650 cell."""
    f_data = _CACHE_DIR / "dummy_data.pkl"
    if not f_data.exists():
        raise FileNotFoundError("Couldn't find cached HPPC data.")
    
    with f_data.open("rb") as f:
        data = pickle.load(f)

    return data
    
    


