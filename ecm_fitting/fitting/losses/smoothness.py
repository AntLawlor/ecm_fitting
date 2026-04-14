from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from ecm_fitting.fitting.losses.base import LossResult, _GradientFreeTerm

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ecm_fitting.data.segment import DataSegment
    from ecm_fitting.ecm.simulator import ECMSimulator, SimulationResult
    from ecm_fitting.ecm.surface import ParameterSurface


class SmoothnessLoss(_GradientFreeTerm):
    """
    Curvature penalty on the ``ParameterSurface`` along each conditioning axis.

    Fits a ``CubicSpline`` to the parameter values along each axis and penalizes
    the integral of the squared second derivative evaluated at the breakpoints.
    Curvature is normalized by each parameter's physical range so that the penalty
    is scale-independent and consistent across different grid densities.

    Loss (per parameter ``j``, per axis, per perpendicular slice)::

        0.5 * lambda_smooth * sum_i(d2_spline[i] / scale_j) ^ 2

    where ``d2_spline[i]`` is the spline second derivative at breakpoint ``i``
    and ``scale_j = bounds_j[1] - bounds_j[0]``.

    Residuals::

        r[i] = sqrt(lambda_smooth) * d2_spline[i] / scale_j

    so that ``0.5 * ||r||^2`` equals the scalar loss above.
    Gradient is computed analytically (no simulation needed).
    """

    name = "smoothness"

    def __init__(self, lambda_smooth: float) -> None:
        self.lambda_smooth = lambda_smooth

    @staticmethod
    def _spline_d2_matrix(breakpoints: NDArray) -> NDArray:
        """
        Constructs matrix of parameter smoothness penalties.

        Build the ``(n, n)`` linear operator such that
        ``mat @ y == CubicSpline(breakpoints, y).derivative(2)(breakpoints)``
        for any value array ``y``.
        Constructed column-by-column by evaluating the spline second derivative
        for each standard basis vector.
        """
        n = len(breakpoints)
        mat = np.zeros((n, n))
        for col in range(n):
            e = np.zeros(n)
            e[col] = 1.0
            mat[:, col] = CubicSpline(breakpoints, e).derivative(2)(breakpoints)
        return mat

    def evaluate(
        self,
        surface: ParameterSurface,
        segments: list[DataSegment],
        results: list[SimulationResult] | None,
        simulator: ECMSimulator | None = None,
    ) -> LossResult:
        """
        Evaluate the curvature loss.

        Arguments:
            surface (ParameterSurface):
                Current parameter surface.
            segments (list[DataSegment]):
                Unused; present for interface consistency.
            results (list[SimulationResult] | None):
                Unused; present for interface consistency.
            simulator (ECMSimulator | None):
                Unused; present for interface consistency.

        """
        if surface.values is None:
            raise RuntimeError("`ParameterSurface.values` not set.")

        vals = surface.values  # (n_params, *grid_shape)
        full_shape = vals.shape
        n_params = vals.shape[0]
        n_free = vals.size

        # Per-parameter normalisation: physical range from bounds
        param_scale = np.array(
            [
                max(spec.bounds[1] - spec.bounds[0], 1e-10)
                for spec in surface.model.param_specs
            ],
        )  # (n_params,)

        sqrt_lam = float(np.sqrt(self.lambda_smooth))
        grad = np.zeros_like(vals)
        all_r: list[NDArray] = []
        all_jac_rows: list[NDArray] = []

        for ax_pos, (_, ax_bp) in enumerate(surface.axes.items()):
            # axis index in vals: vals[0] = n_params, then grid axes in order
            vals_ax = ax_pos + 1
            n_ax = len(ax_bp)
            if n_ax < 2:
                continue

            # Spline second-derivative operator for this axis: (n_ax, n_ax)
            d2_mat = self._spline_d2_matrix(ax_bp)

            # Stride of vals_ax in the flat layout (C-order)
            stride = int(np.prod(full_shape[vals_ax + 1 :]))

            # Other grid axes (everything except the param axis 0 and the current vals_ax)
            other_axes = [d for d in range(vals.ndim) if d not in {0, vals_ax}]
            other_shape = tuple(vals.shape[d] for d in other_axes)
            other_ax_map = {d: oi for oi, d in enumerate(other_axes)}

            for j in range(n_params):
                r_scale = sqrt_lam / param_scale[j]
                lam_scale2 = self.lambda_smooth / param_scale[j] ** 2

                for other_idx in np.ndindex(*other_shape) if other_shape else [()]:
                    # Build full index for this 1-D slice (int at every dim except vals_ax)
                    full_idx_base: list[int] = [0] * vals.ndim
                    full_idx_base[0] = j
                    full_idx_base[vals_ax] = 0  # placeholder; stride handles the rest
                    for d in other_axes:
                        full_idx_base[d] = other_idx[other_ax_map[d]]

                    # Flat index of the first element of this slice
                    base = int(np.ravel_multi_index(full_idx_base, full_shape))
                    flat_indices = base + np.arange(n_ax) * stride  # (n_ax,)

                    # Extract 1-D slice and apply spline second-derivative operator
                    y = vals.ravel()[flat_indices]  # (n_ax,)
                    d2 = d2_mat @ y  # (n_ax,)
                    r_block = r_scale * d2  # (n_ax,)

                    all_r.append(r_block)

                    # Gradient: d(0.5*||r||^2)/dy = lambda/scale^2 * d2_mat^T @ d2
                    grad.ravel()[flat_indices] += lam_scale2 * (d2_mat.T @ d2)

                    # Jacobian rows: d(r_block)/d(flat) = r_scale * d2_mat at flat_indices
                    jac_block = np.zeros((n_ax, n_free))
                    jac_block[:, flat_indices] = r_scale * d2_mat
                    all_jac_rows.append(jac_block)

        residuals = np.concatenate(all_r) if all_r else np.zeros(0)
        total = float(0.5 * np.dot(residuals, residuals))
        residual_jac = (
            np.vstack(all_jac_rows) if all_jac_rows else np.zeros((0, n_free))
        )

        return LossResult(
            total=total,
            components={self.name: total},
            gradient=grad.ravel(),
            residuals=residuals,
            residual_jac=residual_jac,
        )
