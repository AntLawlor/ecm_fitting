from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import lmfit
import numpy as np
from scipy import ndimage

from ecm_fitting.config import _current_sign
from ecm_fitting.data.segment import Extrapolation
from ecm_fitting.ecm.simulator import ECMSimulator
from ecm_fitting.ecm.surface import ParameterSurface
from ecm_fitting.fitting.losses.base import CompositeLoss, LossTerm
from ecm_fitting.fitting.losses.smoothness import SmoothnessLoss
from ecm_fitting.fitting.losses.voltage_mse import VoltageMSELoss

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ecm_fitting.data.resample import ResampleStrategy
    from ecm_fitting.data.segment import DataSegment
    from ecm_fitting.ecm.model import ECM


@dataclass
class FitResult:
    """Outcome of fitting one cell (one ``cell_id``)."""

    surface: ParameterSurface
    """The fitted parameter surface."""

    loss_history: list[float]
    """Scalar loss value at each optimizer iteration."""

    converged: bool
    """Whether the optimizer reported successful convergence."""

    device_id: str
    """Identifier for the fitted battery cell (from ``Battery.cell_id``, or ``'__single__'``)."""

    model_name: str
    """Dash-joined parameter names, e.g. ``'R0-R1-Tau1'``."""


class CellFitter:
    """
    Fits a ParameterSurface for a single device from its DataSegment list.

    The loss passed to fit() determines the regularisation regime:

    - Standalone (default): ``[VoltageMSELoss, SmoothnessLoss]`` — spline
      smoothness regularisation, suitable when no population prior exists.
    - Within HierarchicalFitter E-step: ``[VoltageMSELoss, GPShrinkageLoss]``
      is passed explicitly; SmoothnessLoss is omitted because the GP prior
      covariance K_z already encodes smoothness over the conditioning space.

    """

    def __init__(
        self,
        model: ECM,
        surface_axes: dict[str, NDArray],
        extrapolation: Extrapolation = Extrapolation.CLAMP,
        lambda_smooth: float = 1e-6,
    ) -> None:
        """
        Initialize a CellFitter.

        Args:
            model (ECM):
                Assembled ECM topology. Components determine which parameters exist
                and their physical bounds.
            surface_axes (dict[str, NDArray]):
                Grid breakpoints for each conditioning axis. Must include `'soc'`.
            extrapolation (Extrapolation):
                How the ParameterSurface handles out-of-range queries
                Defaults to Extrapolation.CLAMP.
            lambda_smooth (float):
                Weight of the SmoothnessLoss term in the default loss.
                Ignored when a custom loss is passed to fit().
                Defaults to 1e-6.

        """
        self.model = model
        self.surface_axes = surface_axes
        self.extrapolation = extrapolation
        self.lambda_smooth = lambda_smooth
        self.simulator = ECMSimulator(model)

    def fit(
        self,
        segments: list[DataSegment],
        loss: list[LossTerm] | LossTerm | None = None,
        init: ParameterSurface | None = None,
        device_id: str = "__single__",
        *,
        resample: ResampleStrategy | None = None,
        multi_start: bool = False,
        de_kwargs: dict | None = None,
    ) -> FitResult:
        """
        Fit all segments from one battery cell simultaneously.

        All segments must belong to the same cell (or all have
        battery=None). Mixing segments from different cells raises
        ValueError.

        Args:
            segments (list[DataSegment]):
                All segments belonging to one battery cell.
            loss (list[LossTerm] | LossTerm | None):
                One or more loss terms wrapped in a CompositeLoss internally.
                Defaults to ``[VoltageMSELoss(), SmoothnessLoss(self.lambda_smooth)]``.
            init (ParameterSurface | None):
                Starting surface. Defaults to ``bootstrap_init(segments)``.
            device_id (str):
                Identifier stored in the returned FitResult.
                Defaults to '__single__'.
            resample (ResampleStrategy | None):
                Optional resampling strategy applied to each segment before fitting.
                Original segments are used for the final posterior variance calculation.
                Defaults to None (no resampling).
            multi_start (bool):
                When True, run differential evolution to find a global starting point
                before polishing with least_squares. Much slower but more robust to
                local minima. Defaults to False.
            de_kwargs (dict | None):
                Extra keyword arguments forwarded to scipy.optimize.differential_evolution
                when multi_start=True. Defaults to None.

        Returns:
            FitResult: Fitted surface, loss history, convergence flag, and
                Laplace posterior variance.

        """
        # Validate all segments belong to the same battery cell
        cell_ids = {
            seg.battery.cell_id if seg.battery is not None else None for seg in segments
        }
        if len(cell_ids) > 1:
            msg = (
                "All segments passed to fit() must belong to the same battery cell. "
                f"Found cell_ids: {cell_ids}"
            )
            raise ValueError(msg)

        # Optionally resample segments for the optimizer (original kept for posterior)
        fit_segments = (
            [seg.resample(resample) for seg in segments]
            if resample is not None
            else segments
        )

        # Build starting surface: copy from init or estimate
        surface = self._make_surface()
        if init is not None:
            surface.from_flat(init.to_flat().copy())
        else:
            surface.from_flat(self.bootstrap_init(segments).to_flat().copy())

        # Normalize loss to CompositeLoss
        if loss is None:
            composite = CompositeLoss(
                [VoltageMSELoss(), SmoothnessLoss(self.lambda_smooth)],
                self.simulator,
            )
        elif isinstance(loss, LossTerm):
            composite = CompositeLoss([loss], self.simulator)
        elif isinstance(loss, list):
            composite = CompositeLoss(loss, self.simulator)
        else:
            composite = loss

        # Build per-parameter physical bounds tiled across all grid cells
        n_grid = int(np.prod([len(v) for v in self.surface_axes.values()]))
        lb = np.concatenate([[b[0]] * n_grid for b in self.model.bounds])
        ub = np.concatenate([[b[1]] * n_grid for b in self.model.bounds])

        n_free = len(lb)
        x0 = surface.to_flat()

        # Cache last (key, r, J) to avoid double simulation when lmfit calls the
        # residual function and the jac callable in the same optimizer step
        _cache: dict[str, object] = {}

        def _compute(flat: NDArray) -> tuple[NDArray, object]:
            key = flat.tobytes()
            if _cache.get("key") != key:
                r, j = composite.residuals_and_jacobian(flat, surface, fit_segments)
                _cache["key"] = key
                _cache["r"] = r
                _cache["j"] = j
            return _cache["r"], _cache["j"]

        loss_history: list[float] = []

        def lmfit_residual(lm_params: lmfit.Parameters) -> NDArray:
            flat = np.fromiter(
                (p.value for p in lm_params.values()),
                dtype=float,
                count=n_free,
            )
            r, _ = _compute(flat)
            loss_history.append(float(0.5 * np.sum(r**2)))
            return r

        def jacobian_fnc(lm_params: lmfit.Parameters) -> object:
            flat = np.fromiter(
                (p.value for p in lm_params.values()),
                dtype=float,
                count=n_free,
            )
            _, j = _compute(flat)
            return j

        # Build lmfit Parameters
        params = lmfit.Parameters()
        for i, (lo_i, hi_i, x0_i) in enumerate(zip(lb, ub, x0, strict=True)):
            params.add(f"p_{i}", value=float(x0_i), min=float(lo_i), max=float(hi_i))

        # Optional multi-start: differential evolution via lmfit
        if multi_start:
            result_de = lmfit.minimize(
                lmfit_residual,
                params,
                method="differential_evolution",
                max_nfev=50,
                tol=1e-4,
                seed=0,
                **(de_kwargs or {}),
            )
            params = result_de.params

        # Main solver: least squares via lmfit
        result = lmfit.minimize(
            lmfit_residual,
            params,
            method="least_squares",
            jac=jacobian_fnc,
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-8,
            max_nfev=2000,
        )

        x_final = np.fromiter(
            (p.value for p in result.params.values()),
            dtype=float,
            count=n_free,
        )
        surface.from_flat(x_final)

        return FitResult(
            surface=surface,
            loss_history=loss_history,
            converged=bool(result.success),
            device_id=device_id,
            model_name="-".join(self.model.param_names),
        )

    def bootstrap_init(self, segments: list[DataSegment]) -> ParameterSurface:
        """
        Produce a physically motivated starting ParameterSurface.

        Estimates are derived analytically where possible:

        - R_0: |dV/dI| at current transitions where |dI| > current_threshold,
          weighted by 1/dt (smaller timesteps give better estimates).
        - Tau1: exponential fit V(t) = V_ocv - A * exp(-t / tau) to
          active-to-rest voltage relaxations. Rest windows must be at least
          10 s long, weighted by rest duration.
        - current_threshold: C/20 from segment.battery if available, else 0.1 A.

        All other parameters default to the registered guess (if available)
        or the midpoint of their bounds.

        Args:
            segments (list[DataSegment]): Segments used to estimate initial
                parameter values. All must belong to the same cell.

        Returns:
            ParameterSurface: Initial surface with analytically estimated
                values where available and mid-range defaults elsewhere.

        """
        # Create empty surface
        surface = self._make_surface()

        # Set init vals to per-parameter guess or midpoint of bounds
        init_vals = np.array(
            [
                spec.guess
                if spec.guess is not None
                else 0.5 * (spec.bounds[0] + spec.bounds[1])
                for spec in self.model.param_specs
            ],
        )
        flat = np.repeat(init_vals, int(np.prod(surface.grid_shape)))
        surface.from_flat(flat)

        # Given parameter index of R0 term (if defined)
        r0_idx = next(
            (i for i, n in enumerate(self.model.param_names) if n == "R0"),
            None,
        )
        if r0_idx is not None:
            # (soc, r0_estimate, weight) --> weight = 1/dt
            # smaller dt captures a better R0 estimate
            r0_estimates: list[tuple[float, float, float]] = []

            for seg in segments:
                current_threshold = 0.1
                if seg.battery is not None:
                    current_threshold = (
                        seg.battery.battery_type.nominal_capacity_ah / 20
                    )

                # Coulomb-count SOC at each sample (same convention as simulator)
                dt_arr = np.diff(seg.time, prepend=seg.time[0])
                soc_traj = np.empty(len(seg.time))
                soc = seg.soc_init
                _sign = _current_sign()
                for k in range(len(seg.time)):
                    if k > 0:
                        soc += (
                            _sign
                            * float(seg.current[k - 1])
                            * dt_arr[k]
                            / 3600.0
                            / seg.capacity_ah
                        )
                    soc_traj[k] = soc

                # Compute time-weighted R0 estimate
                di = np.diff(seg.current)
                steps = np.where(np.abs(di) > current_threshold)[0]
                for k in steps:
                    d_i = float(di[k])
                    d_v = float(seg.voltage[k + 1] - seg.voltage[k])
                    if abs(d_i) > 0:
                        r0_est = float(
                            np.clip(abs(d_v / d_i), *self.model.bounds[r0_idx]),
                        )
                        dt_k = float(seg.time[k + 1] - seg.time[k])
                        weight = 1.0 / dt_k if dt_k > 0 else 1.0
                        r0_estimates.append((float(soc_traj[k]), r0_est, weight))

            # Compute avg R0 (weighted by time-delta)
            if r0_estimates and surface.values is not None:
                soc_bp = self.surface_axes["soc"]
                weighted_sum = np.zeros(len(soc_bp))
                weight_total = np.zeros(len(soc_bp))
                for soc_val, r0_val, w in r0_estimates:
                    idx = int(np.argmin(np.abs(soc_bp - soc_val)))
                    weighted_sum[idx] += w * r0_val
                    weight_total[idx] += w
                mask = weight_total > 0
                surface.values[r0_idx, mask] = weighted_sum[mask] / weight_total[mask]

        # Tau1 estimate: exponential relaxation fit to active->rest transitions
        # Longer rest periods give better asymptotic estimates (weight = rest_duration)
        tau1_idx = next(
            (i for i, n in enumerate(self.model.param_names) if n == "Tau1"),
            None,
        )
        if tau1_idx is not None:
            tau_lo, tau_hi = self.model.bounds[tau1_idx]
            # (soc, tau_estimate, weight) tuples
            tau1_estimates: list[tuple[float, float, float]] = []

            for seg in segments:
                current_threshold = 0.1
                if seg.battery is not None:
                    current_threshold = (
                        seg.battery.battery_type.nominal_capacity_ah / 20
                    )
                rest_tolerance = current_threshold * 0.1

                # Coulomb-count SOC at each sample (same convention as simulator)
                dt_arr = np.diff(seg.time, prepend=seg.time[0])
                soc_traj = np.empty(len(seg.time))
                soc = seg.soc_init
                _sign = _current_sign()
                for k in range(len(seg.time)):
                    if k > 0:
                        soc += (
                            _sign
                            * float(seg.current[k - 1])
                            * dt_arr[k]
                            / 3600.0
                            / seg.capacity_ah
                        )
                    soc_traj[k] = soc

                # Scan for active->rest transitions
                n = len(seg.time)
                k = 0
                while k < n - 1:
                    was_active = abs(float(seg.current[k])) > current_threshold
                    now_rest = abs(float(seg.current[k + 1])) < rest_tolerance
                    if not (was_active and now_rest):
                        k += 1
                        continue

                    # Find end of rest window
                    rest_start = k + 1
                    rest_end = rest_start
                    while (
                        rest_end < n
                        and abs(float(seg.current[rest_end])) < rest_tolerance
                    ):
                        rest_end += 1

                    rest_duration = float(seg.time[rest_end - 1] - seg.time[rest_start])
                    k = rest_end  # advance past rest regardless of fit outcome

                    if rest_duration < 10.0 or rest_end - rest_start < 4:
                        continue

                    # Smooth voltage to suppress noise before log transform
                    # Kernel: ~5% of window length, minimum 3, forced odd
                    t_rest = seg.time[rest_start:rest_end] - seg.time[rest_start]
                    v_raw = seg.voltage[rest_start:rest_end]
                    smooth_k = max(3, int(len(v_raw) * 0.05) | 1)
                    v_smooth = ndimage.uniform_filter1d(v_raw, size=smooth_k)

                    # Log-linear fit: ln(V_ocv - V(t)) = ln(A) - t/tau
                    # Uses decay rate directly: more robust for short windows
                    v_ocv_guess = float(v_smooth[-1])  # smoothed end ~ equilibrium
                    residual = v_ocv_guess - v_smooth  # should be positive and decaying
                    valid = residual > 1e-6
                    if valid.sum() < 4:
                        continue  # too little relaxation to fit

                    slope, _ = np.polyfit(t_rest[valid], np.log(residual[valid]), 1)
                    if slope >= 0:
                        continue  # non-decaying segment, skip

                    tau1_est = float(np.clip(-1.0 / slope, tau_lo, tau_hi))
                    # SOC from last active sample before rest began
                    tau1_estimates.append(
                        (float(soc_traj[rest_start - 1]), tau1_est, rest_duration),
                    )

            # Compute weighted avg Tau1 per grid cell
            if tau1_estimates and surface.values is not None:
                soc_bp = self.surface_axes["soc"]
                weighted_sum = np.zeros(len(soc_bp))
                weight_total = np.zeros(len(soc_bp))
                for soc_val, tau1_val, w in tau1_estimates:
                    idx = int(np.argmin(np.abs(soc_bp - soc_val)))
                    weighted_sum[idx] += w * tau1_val
                    weight_total[idx] += w
                mask = weight_total > 0
                surface.values[tau1_idx, mask] = weighted_sum[mask] / weight_total[mask]

        return surface

    def _make_surface(self) -> ParameterSurface:
        """
        Create a fresh, uninitialized ParameterSurface for this fitter.

        Returns:
            ParameterSurface: New surface with the fitter's axes, model, and
                extrapolation policy. Values are not set.

        """
        return ParameterSurface(
            axes=self.surface_axes,
            model=self.model,
            extrapolation=self.extrapolation,
        )
