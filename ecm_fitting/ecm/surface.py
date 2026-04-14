from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ecm_fitting.data.segment import Extrapolation

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ecm_fitting.ecm.model import ECM


class ConditioningGrid:
    """
    Base class for arrays defined on a regular multi-dimensional grid.

    Stores a values array at discrete grid points and provides multilinear
    interpolation at arbitrary query points in the conditioning space Z::

        G_grid = z_1^(1),...,z_1^(m_1) x ... x z_nc^(1),...,z_nc^(m_nc)

    `'soc'` is always a required axis and must not appear in
    DataSegment.conditioning.

    Attributes:
        axes (dict[str, NDArray]):
            Breakpoints per conditioning axis k, keyed
            by axis name (e.g. ``{'soc': np.linspace(0.1, 0.9, 9)}``).
        extrapolation (Extrapolation):
            Out-of-range handling policy.
        values (NDArray | None):
            Values at grid points. Shape depends on subclass;
            trailing dimensions always match grid_shape.
        grid_shape (tuple[int, ...]):
            Grid dimensions (m_1, ..., m_nc);
            total grid size M = m_1 * ... * m_nc.

    """

    def __init__(
        self,
        axes: dict[str, NDArray],
        extrapolation: Extrapolation = Extrapolation.CLAMP,
    ) -> None:
        """
        Initialize a ConditioningGrid.

        Args:
            axes (dict[str, NDArray]):
                Breakpoints per conditioning axis. Must contain 'soc'.
            extrapolation (Extrapolation):
                Out-of-range handling policy.
                Defaults to Extrapolation.CLAMP.

        """
        if "soc" not in axes:
            raise ValueError("'soc' is a required axis in ConditioningGrid.")
        self.axes = axes
        self.extrapolation = extrapolation
        self.values: NDArray | None = None

    def set_values(self, values: NDArray) -> None:
        """
        Set the grid values array.

        Args:
            values (NDArray):
                Array of values at grid points.
                Trailing dimensions must match grid_shape.

        """
        self.values = values

    # ================================================
    # Interpolation
    # ================================================
    def at(self, soc: float, **cond: float) -> NDArray:
        """
        Evaluate the grid values at a single conditioning point.

        Applies multilinear interpolation over the enclosing hypercell and
        returns a 1D vector of interpolated values, one per leading dimension
        of values (e.g. one per ECM parameter). Out-of-range inputs are
        handled according to ``self.extrapolation``.

        Args:
            soc (float):
                State-of-charge query value.
            **cond (float):
                Additional conditioning axis values, keyed by axis name.
                Must cover all non-SOC axes declared in self.axes.

        Returns:
            NDArray: Interpolated values with shape (values.shape[0],).

        """
        if self.values is None:
            raise RuntimeError("`values` has not been set yet.")
        result = np.zeros(self.values.shape[0])
        for multi_idx, w in self.interpolation_weights(soc=soc, **cond):
            result += w * self.values[(slice(None), *multi_idx)]
        return result

    def interpolation_weights_batch(
        self,
        soc: NDArray,
        **cond: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Compute multilinear interpolation weights for a batch of query points.

        Vectorized form of ``interpolation_weights``. Returns the 2**N_axes
        hypercell corner indices and weights for all n_samples query points
        simultaneously.

        Args:
            soc (NDArray):
                SOC query values with shape (n_samples,).
            **cond (NDArray):
                Additional conditioning arrays, each with shape (n_samples,).
                Keys must match non-SOC axes declared in ``self.axes``.

        Returns:
            tuple[NDArray, NDArray]: A pair (corner_indices, corner_weights).
                corner_indices has shape (n_samples, 2**N_axes, N_axes)
                containing grid corner multi-indices for each sample.
                corner_weights has shape (n_samples, 2**N_axes) containing
                interpolation weights that sum to 1 per sample.

        """
        point_arrays = {"soc": soc, **cond}
        n = len(soc)
        n_axes = len(self.axes)

        lo = np.empty((n, n_axes), dtype=int)
        hi = np.empty((n, n_axes), dtype=int)
        fracs = np.empty((n, n_axes))

        for ax_idx, (axis_name, bp) in enumerate(self.axes.items()):
            v = np.clip(point_arrays[axis_name], float(bp[0]), float(bp[-1]))
            idx = np.clip(np.searchsorted(bp, v, side="right") - 1, 0, len(bp) - 2)
            lo[:, ax_idx] = idx
            hi[:, ax_idx] = idx + 1
            span = bp[idx + 1] - bp[idx]
            fracs[:, ax_idx] = np.where(span > 0, (v - bp[idx]) / span, 0.0)

        n_corners = 2**n_axes
        corner_indices = np.empty((n, n_corners, n_axes), dtype=int)
        corner_weights = np.empty((n, n_corners))

        for c in range(n_corners):
            w = np.ones(n)
            for ax in range(n_axes):
                if c & (1 << ax):
                    corner_indices[:, c, ax] = hi[:, ax]
                    w *= fracs[:, ax]
                else:
                    corner_indices[:, c, ax] = lo[:, ax]
                    w *= 1.0 - fracs[:, ax]
            corner_weights[:, c] = w

        return corner_indices, corner_weights

    def interpolation_weights(
        self,
        soc: float,
        **cond: float,
    ) -> list[tuple[tuple[int, ...], float]]:
        """
        Compute multilinear interpolation weights for a single query point.

        Returns the 2**N_axes hypercell corner multi-indices and their weights.
        The weighted sum of values at these corners reconstructs the
        interpolated value at the query point. Weights sum to 1.

        Args:
            soc (float):
                SOC query value.
            **cond (float):
                Additional conditioning axis values, keyed by axis name.
                Must cover all non-SOC axes declared in self.axes.

        Returns:
            list[tuple[tuple[int, ...], float]]: List of (multi_index, weight)
                pairs, one per hypercell corner (2**N_axes total).

        """
        point = {"soc": soc, **cond}
        n_axes = len(self.axes)

        lo_indices: list[int] = []
        hi_indices: list[int] = []
        fracs: list[float] = []

        for axis_name, breakpoints in self.axes.items():
            val = point[axis_name]
            val = self._apply_extrapolation(val, breakpoints)
            idx = int(np.searchsorted(breakpoints, val, side="right")) - 1
            idx = int(np.clip(idx, 0, len(breakpoints) - 2))

            lo = idx
            hi = idx + 1
            span = float(breakpoints[hi] - breakpoints[lo])
            frac = float(val - breakpoints[lo]) / span if span > 0.0 else 0.0

            lo_indices.append(lo)
            hi_indices.append(hi)
            fracs.append(frac)

        weights: list[tuple[tuple[int, ...], float]] = []
        for corner in range(2**n_axes):
            multi_idx: list[int] = []
            weight = 1.0
            for ax in range(n_axes):
                if corner & (1 << ax):
                    multi_idx.append(hi_indices[ax])
                    weight *= fracs[ax]
                else:
                    multi_idx.append(lo_indices[ax])
                    weight *= 1.0 - fracs[ax]
            weights.append((tuple(multi_idx), weight))

        return weights

    # ================================================
    # Extrapolation
    # ================================================
    def _apply_extrapolation(
        self,
        val: float,
        breakpoints: NDArray,
    ) -> float:
        """
        Adjust a query value according to the configured extrapolation policy.

        In-range values are returned unchanged. Out-of-range values are
        clamped, linearly extrapolated, or replaced with NaN depending on
        ``self.extrapolation``.

        Args:
            val (float):
                Query value along one conditioning axis.
            breakpoints (NDArray):
                Sorted 1D breakpoint array for that axis.

        Returns:
            float: The adjusted query value after applying the extrapolation policy.

        """
        lo, hi = float(breakpoints[0]), float(breakpoints[-1])
        if lo <= val <= hi:
            return val
        if self.extrapolation is Extrapolation.NONE:
            return float("nan")
        if self.extrapolation is Extrapolation.CLAMP:
            return float(np.clip(val, lo, hi))
        if self.extrapolation is Extrapolation.LINEAR:
            if val < lo:
                slope = float(breakpoints[1] - breakpoints[0])
                return lo + (val - lo) / slope * slope
            slope = float(breakpoints[-1] - breakpoints[-2])
            return hi + (val - hi) / slope * slope
        return val

    # ================================================
    # Properties
    # ================================================
    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Grid dimensions (m_1, ..., m_nc); total size M = prod(grid_shape)."""
        return tuple(len(v) for v in self.axes.values())

    @property
    def soc_breakpoints(self) -> NDArray:
        """Breakpoints for the SOC axis."""
        return self.axes["soc"]

    def get_breakpoints(self, axis: str) -> NDArray:
        """
        Return the breakpoints for a named conditioning axis.

        Args:
            axis (str): Axis name; must be a key in ``self.axes``.

        Returns:
            NDArray: Sorted 1D breakpoint array for the requested axis.

        Raises:
            KeyError: If axis is not a declared conditioning axis.

        """
        if axis not in self.axes:
            msg = f"'{axis}' is not a valid axis. Available: {self.axes.keys()}"
            raise KeyError(msg)
        return self.axes[axis]


class ParameterSurface(ConditioningGrid):
    """
    Discrete representation of a cell's ECM parameter surface.

    Maps a conditioning space Z to a vector of ECM parameters theta_i via
    multilinear interpolation over a regular grid::

        theta_i : Z -> R^n_p,   theta_i(z) = sum_g alpha_g(z) * theta_i^g

    where alpha_g(z) are the multilinear interpolation weights and theta_i^g
    is the parameter vector stored at grid point g. `'soc'` is always required
    as an axis. All other axes (temperature, soh, etc.) are provided via
    segment conditioning at evaluation time.

    Attributes:
        axes (dict[str, NDArray]):
            Conditioning grid breakpoints per axis k.
        model (ECM):
            The ECM model defining parameter names and bounds.
        values (NDArray | None):
            Parameter values at each grid point, shape `[n_p, *grid_shape]`.
            `values[j, g] = theta_i^j` at grid point `g`.
        grid_shape (tuple[int, ...]):
            Grid dimensions (m_1, ..., m_nc).
        n_free (int):
            Total scalar degrees of freedom: `n_p * M`.

    Examples:
    ```python
        ParameterSurface({"soc": np.arange(0.1, 0.95, 0.1)}, model)

        ParameterSurface(
            {"soc": np.arange(0.1, 0.95, 0.1), "temperature": np.array([0, 25, 40])},
            model,
        )
    ```

    """

    def __init__(
        self,
        axes: dict[str, NDArray],
        model: ECM,
        extrapolation: Extrapolation = Extrapolation.CLAMP,
    ) -> None:
        """
        Initialize a ParameterSurface.

        Args:
            axes (dict[str, NDArray]):
                Conditioning grid breakpoints per axis. Must contain `'soc'`.
            model (ECM):
                Assembled ECM model instance. Determines `n_p` and parameter names.
            extrapolation (Extrapolation):
                Out-of-range handling policy. Defaults to Extrapolation.CLAMP.

        """
        super().__init__(axes, extrapolation)
        self.model = model

    # ================================================
    # Flat vector interface (used by optimizer)
    # ================================================
    def to_flat(self) -> NDArray:
        """
        Return the parameter values as a 1D array.

        Flattens values in C-order: parameter index varies slowest, grid
        index varies fastest. Inverse of ``from_flat``.

        Returns:
            NDArray: 1D array of length n_p * M containing all parameter
                values at all grid points.

        """
        if self.values is None:
            raise RuntimeError("`ParameterSurface.values` has not been set yet.")
        return self.values.ravel()

    def from_flat(self, flat: NDArray) -> None:
        """
        Restore parameter values from a 1D array.

        Inverse of ``to_flat``. Reshapes the flat vector back to
        [n_p, *grid_shape] and stores via ``set_values``.

        Args:
            flat (NDArray): 1D array of length `n_p * M`, as returned by
                ``to_flat`` or produced by an optimizer.

        """
        self.set_values(flat.reshape(self.model.n_params, *self.grid_shape))

    def copy(self) -> ParameterSurface:
        """
        Return a deep copy with the same axes, model, and current values.

        Returns:
            ParameterSurface: New instance with copied values. Values are
                None if not yet set on the original.

        """
        new = ParameterSurface(self.axes, self.model, self.extrapolation)
        if self.values is not None:
            new.set_values(self.values.copy())
        return new

    # ================================================
    # Properties
    # ================================================
    @property
    def n_free(self) -> int:
        """Total scalar degrees of freedom: `n_p * M`."""
        return self.model.n_params * int(np.prod(self.grid_shape))
