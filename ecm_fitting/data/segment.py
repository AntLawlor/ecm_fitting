from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from ecm_fitting.config import CurrentConvention, get_convention

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import NDArray

    from ecm_fitting.data.device import Battery, BatteryType
    from ecm_fitting.data.resample import ResampleStrategy


class Extrapolation(Enum):
    """How to handle ParameterSurface extrapolation."""

    NONE = "none"
    """Return NaN for all values beyond segment bounds."""

    CLAMP = "clamp"
    """Clamp extrapolated points to bound values (default)."""

    LINEAR = "linear"
    """Extends segment bounds via linear extrapolation."""


@dataclass
class DataSegment:
    """One continuous time-series block with its own initiallization conditions."""

    time: NDArray[np.float32]
    """Monotonically increasing time (in seconds)."""

    voltage: NDArray[np.float32]
    """Voltage measurements for each time point (in volts)."""

    current: NDArray[np.float32]
    """Current measurements for each time point (in amps). Positive = discharge."""

    soc_init: float
    """State-of-charge (SOC) at `time=0`. Must be between 0 and 1."""

    ocv_func: Callable[[float], float]
    """Open-circuit voltage as function of SOC."""

    capacity_ah: float
    """Capacity of the device under test (in Amp-hours)."""

    conditioning: dict[str, float | NDArray[np.float32]] = field(default_factory=dict)
    """
    External conditioning variables matching shape of `time`.
    E.g., `{'temperature': [...], 'soh': 0.95}`
    """

    x_init: NDArray | None = None
    """
    Optional initial ECM state override.
    If None, component-defined defaults are used.
    """

    battery: Battery | None = None
    """
    Optional battery instance this segment was recorded from.
    Segments sharing the same :class:`~ecm_fitting.data.device.Battery` are jointly fit.
    """

    metadata: dict = field(default_factory=dict)
    """Additional metadata to track along with each DataSegment."""

    def __post_init__(self) -> None:
        # Reserved conditioning keys
        if "soc" in self.conditioning:
            msg = "'soc' is a reserved variable. It cannot be passed to `conditioning`."
            raise ValueError(msg)

        # Check time is monotonic
        if not np.all(np.diff(self.time) > 0):
            msg = "`time` must be monotonically increasing."
            raise ValueError(msg)

        # Check shapes
        n = len(self.time)
        for name, arr in (("voltage", self.voltage), ("current", self.current)):
            if len(arr) != n:
                msg = f"`{name}` has length {len(arr)}, expected {n} (same as `time`)."
                raise ValueError(msg)
        for key, val in self.conditioning.items():
            if isinstance(val, (int, float)):
                continue
            if hasattr(val, "__len__") and len(val) != n:
                msg = (
                    f"conditioning['{key}'] has length {len(val)}, "
                    f"expected {n} (same as `time`) or a scalar float."
                )
                raise ValueError(msg)

        # Warn if current polarity appears inconsistent with the configured convention.
        # Strategy:
        # - find continuous runs where current is:
        #   - consistently above +threshold (candidate discharge), or
        #   - below -threshold (candidate charge)
        # - Then check whether the net voltage change over each run has the expected sign
        #   - For POSITIVE_IS_DISCHARGE:
        #       - discharge run should result in dV < 0
        #       - charge run should result in dV > 0
        #   - For POSITIVE_IS_CHARGE:
        #       - discharge run should result in dV > 0
        #       - charge run should result in dV < 0
        threshold = self.capacity_ah / 50.0
        i_arr = self.current
        n_pts = len(i_arr)

        # Label each sample: +1 (above threshold), -1 (below -threshold), 0 (rest)
        label = np.zeros(n_pts, dtype=np.int8)
        label[i_arr > threshold] = 1
        label[i_arr < -threshold] = -1

        # Find continuous-run boundaries
        boundaries = np.where(np.diff(label) != 0)[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [n_pts]])

        run_dot = 0.0
        n_active_runs = 0
        for s, e in zip(starts, ends, strict=True):
            run_label = int(label[s])
            if run_label == 0:
                continue
            # Net voltage change over the run (start-to-end)
            dv_run = float(self.voltage[e - 1] - self.voltage[s])
            run_dot += run_label * dv_run
            n_active_runs += 1

        if n_active_runs > 0:
            convention = get_convention()
            expected_negative = convention is CurrentConvention.POSITIVE_IS_DISCHARGE
            if (expected_negative and run_dot > 0) or (
                not expected_negative and run_dot < 0
            ):
                warnings.warn(
                    f"DataSegment current polarity may not match the configured "
                    f"convention ({convention.value!r}). The data suggests that "
                    f"positive current corresponds to "
                    f"{'charging' if expected_negative else 'discharge'}, "
                    f"but the convention expects positive = "
                    f"{'discharge' if expected_negative else 'charging'}. "
                    "Use ecm_fitting.set_convention() to change the setting, or "
                    "negate your current array.",
                    UserWarning,
                    stacklevel=2,
                )

    def resample(self, strategy: ResampleStrategy) -> DataSegment:
        """
        Return a new ``DataSegment`` resampled onto a new time grid.

        ``strategy.resample_time(self)`` determines the new time array.
        All array-valued fields (``voltage``, ``current``, and array entries in
        ``conditioning``) are linearly interpolated onto the new grid. Scalar
        conditioning values and all other fields (``soc_init``, ``capacity_ah``,
        ``ocv_func``, ``battery``, ``x_init``, ``metadata``) are copied unchanged.

        Arguments:
            strategy (ResampleStrategy):
                A :class:`~ecm_fitting.data.resample.ResampleStrategy` instance
                that defines the new time array.

        Returns:
            DataSegment: A new segment on the resampled time grid.

        """
        new_time = strategy.resample_time(self).astype(np.float32)
        new_voltage = np.interp(new_time, self.time, self.voltage).astype(np.float32)
        new_current = np.interp(new_time, self.time, self.current).astype(np.float32)
        new_conditioning: dict[str, float | NDArray[np.float32]] = {
            key: (
                np.interp(new_time, self.time, val).astype(np.float32)
                if isinstance(val, np.ndarray)
                else val
            )
            for key, val in self.conditioning.items()
        }
        return DataSegment(
            time=new_time,
            voltage=new_voltage,
            current=new_current,
            soc_init=self.soc_init,
            ocv_func=self.ocv_func,
            capacity_ah=self.capacity_ah,
            conditioning=new_conditioning,
            x_init=self.x_init,
            battery=self.battery,
            metadata=self.metadata,
        )

    @property
    def duration(self) -> float:
        """Duration of this data segment (in seconds)."""
        return float(self.time[-1] - self.time[0])

    @property
    def n_samples(self) -> int:
        """Number of data points in this segment."""
        return len(self.time)


@dataclass
class DataCollection:
    """
    A group of DataSegments, optionally spanning multiple battery instances.

    Segments are grouped by their :attr:`~DataSegment.battery` instance prior to
    fitting. A warning is raised at construction time if segments carry ``battery``
    values whose chemistry (cathode/anode) differs, since population fitting across
    different chemistries may be physically inappropriate.
    """

    segments: list[DataSegment]

    def __post_init__(self) -> None:
        battery_types = {
            seg.battery.battery_type for seg in self.segments if seg.battery is not None
        }
        if len(battery_types) > 1:
            chemistries = {(bt.cathode, bt.anode) for bt in battery_types}
            if len(chemistries) > 1:
                labels = ", ".join(bt.label for bt in battery_types)
                warnings.warn(
                    f"Segments have different chemistries: {labels}. "
                    "Population fitting across different chemistries may be physically "
                    "inappropriate.",
                    stacklevel=2,
                )
        self._battery_type: BatteryType | None = (
            next(iter(battery_types)) if len(battery_types) == 1 else None
        )

    @property
    def battery_type(self) -> BatteryType | None:
        """The common battery type for all segments, or ``None`` if ambiguous or unset."""
        return self._battery_type

    @property
    def data_by_battery(self) -> dict[str | None, list[DataSegment]]:
        """
        Group segments by unique battery cell IDs.

        Returns:
            dict[str | None, list[DataSegment]]: Key is the ``cell_id`` string,
            or ``None`` if no battery was set on any segment.

        """
        groups: dict[str | None, list[DataSegment]] = {}
        for seg in self.segments:
            key = seg.battery.cell_id if seg.battery is not None else None
            groups.setdefault(key, []).append(seg)
        return groups

    @property
    def n_devices(self) -> int:
        return len(self.data_by_battery)

    def filter_to_meta(self, meta_key: str, meta_value: Any) -> list[DataSegment]:
        """Return segments where `DataSegment.metadata[meta_key] == meta_value`."""
        return [
            seg for seg in self.segments if seg.metadata.get(meta_key) == meta_value
        ]

    def leave_one_out_splits(
        self,
    ) -> Iterator[tuple[DataCollection, list[DataSegment]]]:
        """
        Yield ``(train_collection, held_out_segments)`` for each battery cell.

        Each fold holds out all segments from one cell and trains on the rest.
        Requires segments to carry :attr:`~DataSegment.battery` instances.
        Folds where only one cell exists are valid — the train collection will
        be empty but still a valid :class:`DataCollection`.
        """
        for held_id, held_segs in self.data_by_battery.items():
            train = [
                s
                for s in self.segments
                if s.battery is None or s.battery.cell_id != held_id
            ]
            yield DataCollection(train), held_segs
