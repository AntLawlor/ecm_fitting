from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import lmfit
import numpy as np
from scipy import ndimage

from ecm_fitting.ecm.simulator import ECMSimulator

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ecm_fitting.data.segment import DataSegment
    from ecm_fitting.ecm.model import ECM


class ParameterSet:
    """
    Scalar ECM parameters for a single operating point.

    Unlike ``ParameterSurface``, which maps a conditioning space to parameter
    vectors via grid interpolation, ``ParameterSet`` holds one fixed value per
    ECM parameter. It is used by ``LocalFitter`` when SOC-dependence is not
    modeled and parameters are fitted as global scalars over a single segment.

    Attributes:
        model (ECM): The ECM model defining parameter names and bounds.

    Examples:
    ```python
        ps = ParameterSet(model)
        ps.from_flat(np.array([0.01, 0.005, 30.0]))
        ps["R0"]          # 0.01
        ps.to_dict()      # {"R0": 0.01, "R1": 0.005, "Tau1": 30.0}
    ```

    """

    def __init__(self, model: ECM) -> None:
        """
        Initialize a ParameterSet.

        Args:
            model (ECM): Assembled ECM model. Determines parameter names and count.

        """
        self.model = model
        self._values: NDArray | None = None

    def at(self, soc: float, **cond: float) -> NDArray:
        """Ignore. Only for internally simulator compatability."""
        return self._values

    # ================================================
    # Flat vector interface (used by optimizer)
    # ================================================
    def from_flat(self, flat: NDArray) -> None:
        """
        Set parameter values from a 1D array ordered by model parameter index.

        Args:
            flat (NDArray): 1D array of length ``model.n_params``.

        """
        self._values = np.asarray(flat, dtype=float).copy()

    def to_flat(self) -> NDArray:
        """
        Return parameter values as a 1D array ordered by model parameter index.

        Returns:
            NDArray: 1D array of length ``model.n_params``.

        """
        if self._values is None:
            raise RuntimeError("ParameterSet values have not been set yet.")
        return self._values.copy()

    # ================================================
    # Dict-like access
    # ================================================
    def __getitem__(self, name: str) -> float:
        """
        Return the scalar value for a named parameter.

        Args:
            name (str): ECM parameter name (e.g. ``'R0'``, ``'Tau1'``).

        Returns:
            float: Current value of the parameter.

        Raises:
            ValueError: If ``name`` is not in the model's parameter list.
            RuntimeError: If values have not been set yet.

        """
        if self._values is None:
            raise RuntimeError("ParameterSet values have not been set yet.")
        try:
            idx = self.model.param_names.index(name)
        except ValueError:
            msg = (
                f"'{name}' is not a parameter of this model. "
                f"Available: {self.model.param_names}"
            )
            raise ValueError(msg) from None
        return float(self._values[idx])

    def to_dict(self) -> dict[str, float]:
        """
        Return a copy of all parameter values as a ``{name: value}`` dict.

        Returns:
            dict[str, float]: Mapping from parameter name to scalar value.

        """
        if self._values is None:
            raise RuntimeError("ParameterSet values have not been set yet.")
        return {name: float(self._values[i]) for i, name in enumerate(self.model.param_names)}

    def copy(self) -> ParameterSet:
        """Return a deep copy of this ParameterSet."""
        new = ParameterSet(self.model)
        if self._values is not None:
            new._values = self._values.copy()
        return new

    def __repr__(self) -> str:
        if self._values is None:
            return f"ParameterSet(model={self.model!r}, values=<unset>)"
        pairs = ", ".join(f"{n}={v:.4g}" for n, v in self.to_dict().items())
        return f"ParameterSet({pairs})"


@dataclass
class LocalFitResult:
    """Outcome of fitting an ECM to a single ``DataSegment``."""

    parameters: ParameterSet
    """The fitted scalar parameter values."""

    loss_history: list[float]
    """Scalar loss (0.5 * ||r||^2) at each optimizer iteration."""

    converged: bool
    """Whether the optimizer reported successful convergence."""

    device_id: str
    """Identifier stored from the ``device_id`` argument to ``fit()``."""

    model_name: str
    """Dash-joined parameter names, e.g. ``'R0-R1-Tau1'``."""


class LocalFitter:
    """
    Fits an ECM to a single DataSegment using scalar (SOC-independent) parameters.

    Unlike ``CellFitter``, which fits a ``ParameterSurface`` over a conditioning
    grid, ``LocalFitter`` treats each ECM parameter as a single scalar value
    constant across the whole segment. This is appropriate when SOC-dependence
    is not of interest or when data is too limited to support a surface fit.

    The loss is always voltage MSE. The Jacobian is taken directly from the
    simulator's ``voltage_jac`` output (shape ``n_samples × n_params``), which
    is already the correct ``d(v_sim) / d(params)`` for scalar parameters —
    no grid interpolation weighting is needed.

    """

    def __init__(self, model: ECM) -> None:
        """
        Initialize a LocalFitter.

        Args:
            model (ECM):
                Assembled ECM topology. Components determine which parameters
                exist and their physical bounds.

        """
        self.model = model
        self.simulator = ECMSimulator(model)

    def fit(
        self,
        segment: DataSegment,
        init: ParameterSet | None = None,
        device_id: str = "__single__",
        *,
        multi_start: bool = False,
        de_kwargs: dict | None = None,
    ) -> LocalFitResult:
        """
        Fit ECM parameters to a single DataSegment.

        Args:
            segment (DataSegment):
                The segment to fit.
            init (ParameterSet | None):
                Starting parameter values. Defaults to ``bootstrap_init(segment)``.
            device_id (str):
                Identifier stored in the returned ``LocalFitResult``.
                Defaults to ``'__single__'``.
            multi_start (bool):
                When True, run differential evolution to find a global starting
                point before polishing with least_squares. Much slower but more
                robust to local minima. Defaults to False.
            de_kwargs (dict | None):
                Extra keyword arguments forwarded to
                ``scipy.optimize.differential_evolution`` when
                ``multi_start=True``. Defaults to None.

        Returns:
            LocalFitResult: Fitted parameters, loss history, and convergence flag.

        """
        # Build starting ParameterSet
        param_set = ParameterSet(self.model)
        if init is not None:
            param_set.from_flat(init.to_flat())
        else:
            param_set.from_flat(self.bootstrap_init(segment).to_flat())

        # Physical bounds
        lb = np.array([b[0] for b in self.model.bounds])
        ub = np.array([b[1] for b in self.model.bounds])
        n_free = len(lb)
        x0 = param_set.to_flat()

        # Cache last (key, residuals, voltage_jac) to avoid double simulation
        # when lmfit calls the residual and jac callables in the same step
        _cache: dict[str, object] = {}

        def _compute(flat: NDArray) -> tuple[NDArray, NDArray]:
            key = flat.tobytes()
            if _cache.get("key") != key:
                param_set.from_flat(flat)
                result = self.simulator.simulate(segment, param_set, compute_jacobian=True)
                _cache["key"] = key
                _cache["r"] = result.residuals
                _cache["j"] = result.voltage_jac  # shape (n_samples, n_params)
            return _cache["r"], _cache["j"]  # type: ignore[return-value]

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

        def jacobian_fnc(lm_params: lmfit.Parameters) -> NDArray:
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
        param_set.from_flat(x_final)

        return LocalFitResult(
            parameters=param_set,
            loss_history=loss_history,
            converged=bool(result.success),
            device_id=device_id,
            model_name="-".join(self.model.param_names),
        )

    def bootstrap_init(self, segment: DataSegment) -> ParameterSet:
        """
        Produce a physically motivated starting ParameterSet for a segment.

        Estimates are derived analytically where possible:

        - ``R0``: weighted mean of ``|dV/dI|`` at current transitions where
          ``|dI| > current_threshold``, weighted by ``1/dt`` (smaller
          timesteps capture faster ohmic steps more accurately).
        - ``Tau1``: exponential fit ``V(t) = V_ocv - A * exp(-t / tau)`` to
          active-to-rest voltage relaxations, weighted by rest duration.
        - ``current_threshold``: C/20 from ``segment.battery`` if available,
          else 0.1 A.

        All other parameters default to their registered guess (if set) or
        the midpoint of their physical bounds.

        Args:
            segment (DataSegment):
                The segment used to estimate initial parameter values.

        Returns:
            ParameterSet: Initial parameters with analytically estimated
                values where available and mid-range defaults elsewhere.

        """
        param_set = ParameterSet(self.model)

        # Default: per-parameter guess or midpoint of bounds
        init_vals = np.array(
            [
                spec.guess
                if spec.guess is not None
                else 0.5 * (spec.bounds[0] + spec.bounds[1])
                for spec in self.model.param_specs
            ],
        )
        param_set.from_flat(init_vals)

        current_threshold = 0.1
        if segment.battery is not None:
            current_threshold = segment.battery.battery_type.nominal_capacity_ah / 20

        # ------------------------------------------------
        # R0 estimate: weighted mean |dV/dI| at current transitions
        # ------------------------------------------------
        r0_idx = next(
            (i for i, n in enumerate(self.model.param_names) if n == "R0"),
            None,
        )
        if r0_idx is not None:
            di = np.diff(segment.current)
            steps = np.where(np.abs(di) > current_threshold)[0]
            r0_sum = 0.0
            w_sum = 0.0
            for k in steps:
                d_i = float(di[k])
                d_v = float(segment.voltage[k + 1] - segment.voltage[k])
                if abs(d_i) > 0:
                    r0_est = float(
                        np.clip(abs(d_v / d_i), *self.model.bounds[r0_idx]),
                    )
                    dt_k = float(segment.time[k + 1] - segment.time[k])
                    weight = 1.0 / dt_k if dt_k > 0 else 1.0
                    r0_sum += weight * r0_est
                    w_sum += weight
            if w_sum > 0:
                init_vals[r0_idx] = r0_sum / w_sum
                param_set.from_flat(init_vals)

        # ------------------------------------------------
        # Tau1 estimate: exponential relaxation fit to active->rest transitions
        # ------------------------------------------------
        tau1_idx = next(
            (i for i, n in enumerate(self.model.param_names) if n == "Tau1"),
            None,
        )
        if tau1_idx is not None:
            tau_lo, tau_hi = self.model.bounds[tau1_idx]
            rest_tolerance = current_threshold * 0.1

            tau1_sum = 0.0
            w_sum = 0.0
            n = len(segment.time)
            k = 0
            while k < n - 1:
                was_active = abs(float(segment.current[k])) > current_threshold
                now_rest = abs(float(segment.current[k + 1])) < rest_tolerance
                if not (was_active and now_rest):
                    k += 1
                    continue

                # Find end of rest window
                rest_start = k + 1
                rest_end = rest_start
                while (
                    rest_end < n
                    and abs(float(segment.current[rest_end])) < rest_tolerance
                ):
                    rest_end += 1

                rest_duration = float(segment.time[rest_end - 1] - segment.time[rest_start])
                k = rest_end  # advance past rest regardless of fit outcome

                if rest_duration < 10.0 or rest_end - rest_start < 4:
                    continue

                # Smooth voltage to suppress noise before log transform
                # Kernel: ~5% of window length, minimum 3, forced odd
                t_rest = segment.time[rest_start:rest_end] - segment.time[rest_start]
                v_raw = segment.voltage[rest_start:rest_end]
                smooth_k = max(3, int(len(v_raw) * 0.05) | 1)
                v_smooth = ndimage.uniform_filter1d(v_raw, size=smooth_k)

                # Log-linear fit: ln(V_ocv - V(t)) = ln(A) - t/tau
                v_ocv_guess = float(v_smooth[-1])
                residual = v_ocv_guess - v_smooth
                valid = residual > 1e-6
                if valid.sum() < 4:
                    continue

                slope, _ = np.polyfit(t_rest[valid], np.log(residual[valid]), 1)
                if slope >= 0:
                    continue

                tau1_est = float(np.clip(-1.0 / slope, tau_lo, tau_hi))
                tau1_sum += rest_duration * tau1_est
                w_sum += rest_duration

            if w_sum > 0:
                init_vals[tau1_idx] = tau1_sum / w_sum
                param_set.from_flat(init_vals)

        return param_set
