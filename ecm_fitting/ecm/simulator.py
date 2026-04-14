from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from ecm_fitting.config import _current_sign

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ecm_fitting.data.segment import DataSegment
    from ecm_fitting.ecm.model import ECM
    from ecm_fitting.ecm.surface import ParameterSurface


@dataclass
class SimulationResult:
    """Output of simulating one DataSegment."""

    v_sim: NDArray[np.float32]
    """Simulated terminal voltage with shape ``(n_samples,)``."""

    state_trajectory: NDArray[np.float32]
    """State trajectory with shape ``(n_samples, n_x)``."""

    soc_trajectory: NDArray[np.float32]
    """Coulomb-counted SOC with shape ``(n_samples,)``."""

    residuals: NDArray[np.float32]
    """``v_sim - voltage_measured`` with shape ``(n_samples,)``."""

    state_jac_trajectory: NDArray[np.float32] | None
    """State Jacobian trajectory with shape ``(n_samples, n_x, n_r)``."""

    voltage_jac: NDArray[np.float32] | None
    """``d(v_sim)/d(params)`` at each step with shape ``(n_samples, n_r)``."""

    voltage_components: dict[str, NDArray[np.float32]] | None
    """Per-component voltage contributions at each step."""


class ECMSimulator:
    """
    Forward integrator for an ECM over a DataSegment.

    Advances the state-space model using forward Euler with SOC-dependent
    parameters interpolated from a ParameterSurface at each timestep::

        soc(t) = soc_init + sign * integral(current, 0, t) / (capacity_ah * 3600)

    At each timestep k:

    1. Coulomb-count soc(k) from soc(k-1) and current(k-1)
    2. Read conditioning from segment.conditioning at time k
    3. Interpolate theta_i(z(k)) from the parameter surface
    4. Evaluate terminal voltage y(k) = h(x(k), u(k), theta_i(z(k)))
    5. Optionally evaluate voltage Jacobian d(y)/d(theta_i)
    6. Advance state: x(k+1) = x(k) + dt * f(x(k), u(k), theta_i(z(k)))

    """

    def __init__(self, model: ECM) -> None:
        """
        Initialize an ECMSimulator.

        Args:
            model (ECM): Assembled ECM model to simulate.

        """
        self.model = model

    # ================================================
    # Simulate DataSegment(s)
    # ================================================
    def simulate(
        self,
        segment: DataSegment,
        surface: ParameterSurface,
        *,
        compute_jacobian: bool = True,
        compute_components: bool = False,
    ) -> SimulationResult:
        """
        Simulate one segment end-to-end.

        Initial ECM state comes from segment.x_init when provided, otherwise
        from model.initial_state() (component defaults, typically zeros).

        Args:
            segment (DataSegment):
                The data segment to simulate.
            surface (ParameterSurface):
                Parameter surface used to interpolate ECM parameters theta_i(z(t))
                at each timestep.
            compute_jacobian (bool):
                When True, compute d(v_sim)/d(params) at each timestep via the
                variational equations. Required for gradient-based fitting.
                Defaults to True.
            compute_components (bool):
                When True, populate `SimulationResult.voltage_components` with
                per-component voltage contributions at every timestep.
                Defaults to False to avoid overhead during fitting.

        Returns:
            SimulationResult: Simulated voltage, state trajectory, SOC
                trajectory, residuals, and optionally Jacobians and voltage
                components.

        """
        model = self.model
        n = segment.n_samples

        # Initialize result buffers
        v_sim = np.empty(n)
        state_traj = np.empty((n, model.n_states))
        soc_traj = np.empty(n)
        state_jac_traj = (
            np.zeros((n, model.n_states, model.n_params)) if compute_jacobian else None
        )
        voltage_jac = np.empty((n, model.n_params)) if compute_jacobian else None
        # Per-component buffers: keyed by label, filled lazily on first timestep
        component_bufs: dict[str, NDArray] | None = {} if compute_components else None

        # Get initial state
        state = model.initial_state(segment.x_init).copy()
        state_jac = np.zeros((model.n_states, model.n_params))
        soc = segment.soc_init

        # Simulate full segment:
        # dt[0] = 0 so the first step does no integration
        dt_arr = np.diff(segment.time, prepend=segment.time[0])
        sign = _current_sign()
        for k in range(n):
            # 1. Coulomb-count SOC
            if k > 0:
                soc += (
                    sign
                    * float(segment.current[k - 1])
                    * dt_arr[k]
                    / 3600.0
                    / segment.capacity_ah
                )
            soc_traj[k] = soc

            # 2. Get surface conditions at this timestep
            cond = {
                key: (float(val[k]) if isinstance(val, np.ndarray) else float(val))
                for key, val in segment.conditioning.items()
            }

            # 3. Interpolate parameter surface
            params = surface.at(soc=soc, **cond)
            current = sign * float(segment.current[k])
            v_oc = float(segment.ocv_func(soc))

            # 4. Terminal voltage and state snapshot
            v_sim[k] = model.voltage(state, params, current, v_oc)
            state_traj[k] = state

            # 4b. Per-component breakdown (optional)
            if component_bufs is not None:
                breakdown = model.voltage_breakdown(state, params, current, v_oc)
                for label, val in breakdown.items():
                    if label not in component_bufs:
                        component_bufs[label] = np.empty(n)
                    component_bufs[label][k] = val

            # 5. Jacobians
            if (
                compute_jacobian
                and voltage_jac is not None
                and state_jac_traj is not None
            ):
                voltage_jac[k] = model.voltage_jacobian(
                    state,
                    state_jac,
                    params,
                    current,
                )
                state_jac_traj[k] = state_jac

            # 6. Advance ODEs (forward Euler)
            if k < n - 1:
                dt = float(dt_arr[k + 1])
                dx = model.ode(state, params, current)
                if compute_jacobian:
                    dj = model.jacobian_ode(state, state_jac, params, current)
                    state_jac = state_jac + dt * dj
                state = state + dt * dx

        residuals = v_sim - segment.voltage
        return SimulationResult(
            v_sim=v_sim,
            state_trajectory=state_traj,
            soc_trajectory=soc_traj,
            residuals=residuals,
            state_jac_trajectory=state_jac_traj,
            voltage_jac=voltage_jac,
            voltage_components=component_bufs,
        )

    def simulate_segments(
        self,
        segments: list[DataSegment],
        surface: ParameterSurface,
        *,
        compute_jacobian: bool = True,
        compute_components: bool = False,
    ) -> list[SimulationResult]:
        """
        Simulate a list of segments independently.

        No state carries over between segments. Returns one SimulationResult
        per segment in the same order as the input list.

        Args:
            segments (list[DataSegment]):
                The data segments to simulate.
            surface (ParameterSurface):
                Parameter surface shared across all segments.
            compute_jacobian (bool):
                When True, compute voltage Jacobians in each result.
                Required for gradient-based fitting. Defaults to True.
            compute_components (bool):
                When True, populate voltage_components in each result.
                Defaults to False.

        Returns:
            list[SimulationResult]: One result per segment, in input order.

        """
        return [
            self.simulate(
                seg,
                surface,
                compute_jacobian=compute_jacobian,
                compute_components=compute_components,
            )
            for seg in segments
        ]

    # ================================================
    # Residual Jacobian (for least_squares)
    # ================================================
    def build_residual_jacobian(
        self,
        segments: list[DataSegment],
        results: list[SimulationResult],
        surface: ParameterSurface,
    ) -> tuple[NDArray, csr_matrix]:
        """
        Build the full residual vector and sparse Jacobian w.r.t. flat surface params.

        Uses vectorized batch interpolation weights to eliminate per-sample
        Python loops. Each Jacobian row has at most n_r * 2**n_axes
        non-zero entries (the corners of the enclosing interpolation
        hypercell)::

            J[t, flat_idx(j, grid_idx)] = voltage_jac[t, j] * weight(grid_idx, soc_t)

        Args:
            segments (list[DataSegment]):
                The data segments corresponding to each result.
            results (list[SimulationResult]):
                Simulation results with voltage_jac populated (compute_jacobian=True).
            surface (ParameterSurface):
                Parameter surface used to compute interpolation weights.

        Returns:
            tuple[NDArray, csr_matrix]: A pair (residuals, jac) where
                residuals has shape (n_total_samples,) and jac is a sparse
                csr_matrix of shape (n_total_samples, n_free_params).

        """
        n_free = surface.n_free
        grid_shape = surface.grid_shape
        n_r = self.model.n_params
        n_grid = int(np.prod(grid_shape))
        n_axes = len(surface.axes)
        n_corners = 2**n_axes

        # Collect all residuals
        all_residuals = np.concatenate([res.residuals for res in results])
        n_total = len(all_residuals)

        # Concatenate soc trajectories and voltage Jacobians across all segments
        soc_all = np.concatenate([res.soc_trajectory for res in results])
        vj_all = np.concatenate(
            [res.voltage_jac for res in results if res.voltage_jac is not None],
        )  # (n_total, n_r)

        # Concatenate non-SOC conditioning arrays across all segments
        cond_all: dict[str, NDArray] = {}
        for ax_name in surface.axes:
            if ax_name == "soc":
                continue
            arrays = []
            for seg in segments:
                val = seg.conditioning.get(ax_name)
                if isinstance(val, np.ndarray):
                    arrays.append(val)
                else:
                    arrays.append(np.full(seg.n_samples, float(val)))
            cond_all[ax_name] = np.concatenate(arrays)

        # Batch interpolation:
        # - corner_indices (n_total, n_corners, n_axes)
        # - corner_weights (n_total, n_corners)
        corner_indices, corner_weights = surface.interpolation_weights_batch(
            soc_all,
            **cond_all,
        )

        # Build COO sparse matrix: n_r * n_corners numpy operations (no sample loop)
        t_idx = np.arange(n_total)
        rows_parts: list[NDArray] = []
        cols_parts: list[NDArray] = []
        vals_parts: list[NDArray] = []

        for j in range(n_r):
            for c in range(n_corners):
                # Flat grid index for this corner (C-order, consistent with values.ravel())
                grid_flat = np.ravel_multi_index(
                    tuple(corner_indices[:, c, ax] for ax in range(n_axes)),
                    grid_shape,
                )  # (n_total,)
                col = j * n_grid + grid_flat  # (n_total,)
                val = corner_weights[:, c] * vj_all[:, j]  # (n_total,)
                rows_parts.append(t_idx)
                cols_parts.append(col)
                vals_parts.append(val)

        jac = csr_matrix(
            (
                np.concatenate(vals_parts),
                (np.concatenate(rows_parts), np.concatenate(cols_parts)),
            ),
            shape=(n_total, n_free),
        )
        return all_residuals, jac

    # ================================================
    # Gradient Accumulation
    # ================================================
    def accumulate_surface_gradient(
        self,
        segments: list[DataSegment],
        results: list[SimulationResult],
        surface: ParameterSurface,
    ) -> np.ndarray:
        """
        Accumulate per-segment voltage Jacobians into a loss gradient over the surface.

        Computes dL/d(surface.values) by summing contributions from all
        timesteps across all segments::

            dL/d(values[j, grid_idx]) =
                sum_t residual(t) * voltage_jac[t, j] * weight(grid_idx, soc(t))

        Args:
            segments (list[DataSegment]):
                The data segments corresponding to each result.
            results (list[SimulationResult]):
                Simulation results with voltage_jac and residuals populated.
            surface (ParameterSurface):
                Parameter surface used to compute interpolation weights.

        Returns:
            np.ndarray: Flat gradient array with the same layout as
                surface.to_flat(), shape (n_free,).

        """
        n_r = self.model.n_params
        grid_shape = surface.grid_shape
        grad = np.zeros((n_r, *grid_shape))

        # Aggregate results over each segment and time step
        # This is how we are to perform joint-fitting over multiple distinct segments
        for seg, res in zip(segments, results, strict=True):
            if res.voltage_jac is None:
                continue

            n = seg.n_samples
            # scaled_dv[t, j] = residual(t) * voltage_jac[t, j]
            scaled_dv = res.residuals[:, np.newaxis] * res.voltage_jac  # [T, n_r]

            for k in range(n):
                soc_k = float(res.soc_trajectory[k])
                cond_k = {
                    key: (float(val[k]) if isinstance(val, np.ndarray) else float(val))
                    for key, val in seg.conditioning.items()
                }
                for multi_idx, w in surface.interpolation_weights(soc=soc_k, **cond_k):
                    grad[(slice(None), *multi_idx)] += w * scaled_dv[k]

        return grad.ravel()


# TODO: BatteryType @device.py now includes a operating_voltage_v - the operating range of the battery
# If the OCV term is ever outside of this range, the simulation should fail and provide a message as to why.
