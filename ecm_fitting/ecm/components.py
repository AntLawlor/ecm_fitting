from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

T_BOUNDS: TypeAlias = tuple[float, float] | tuple[float, float, float]
"""A 2-tuple ``(lo, hi)`` or 3-tuple ``(lo, guess, hi)`` of floats."""


def _parse_bounds(
    v: T_BOUNDS,
) -> tuple[tuple[float, float], float | None]:
    """
    Parse a bounds tuple into a (bounds, guess) pair.

    A 2-tuple ``(lo, hi)`` yields ``((lo, hi), None)``.
    A 3-tuple ``(lo, guess, hi)`` yields ``((lo, hi), guess)``.

    Args:
        v (T_BOUNDS): A 2-tuple ``(lo, hi)`` or 3-tuple ``(lo, guess, hi)``.

    Returns:
        tuple[tuple[float, float], float | None]: Physical bounds and an
            optional initial guess extracted from the middle value.

    """
    if len(v) == 3:
        return (v[0], v[2]), float(v[1])
    return (float(v[0]), float(v[1])), None


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    """The parameter name."""

    bounds: tuple[float, float]
    """Physical bounds of this parameter: (lower, upper)."""

    guess: float | None = None
    """
    Optional initial guess for this parameter.
    When set (e.g. from a registry 3-tuple), :meth:`CellFitter.bootstrap_init`
    uses this value instead of the midpoint of ``bounds``.
    """


class ECMComponent(ABC):
    """
    One additive, self-contained contribution to the ECM terminal voltage.

    Each component owns a slice of the full state vector x(t) and parameter
    vector theta_i(z(t)), and is responsible for four operations:

    - State ODE (ode): how its own state x_k(t) evolves with current u(t)
    - Voltage (voltage): its instantaneous additive contribution to y(t)
    - State Jacobian ODE (jacobian_ode): propagates J_k(t) = dx_k/dtheta
      forward in time alongside the state ODE
    - Voltage Jacobian (voltage_jacobian): converts accumulated J_k(t)
      into dy/dtheta at each timestep

    """

    # ================================================
    # Abstract properties
    # ================================================
    @property
    @abstractmethod
    def param_specs(self) -> list[ParameterSpec]:
        """Ordered parameter specifications for this component."""

    @property
    @abstractmethod
    def state_names(self) -> list[str]:
        """Ordered names of this component's hidden state variables."""

    @property
    @abstractmethod
    def voltage_label(self) -> str:
        """Short label identifying this component's voltage contribution."""

    # ================================================
    # Abstract methods
    # ================================================
    @abstractmethod
    def ode(
        self,
        state: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
    ) -> NDArray[np.float32]:
        """
        Compute the rate of change of this component's state.

        Evaluates the component's contribution to f(x(t), u(t), theta_i(z(t))),
        returning dx_k/dt for this component's state slice only.

        Args:
            state (NDArray[np.float32]):
                Component state variables with shape (n_x,).
            params (NDArray[np.float32]):
                Component parameter values with shape (n_r,).
            current (float):
                Applied current in amps. Positive = discharge.

        Returns:
            NDArray[np.float32]: State derivative dx_k/dt with shape
                (n_x,).

        """

    @abstractmethod
    def voltage(
        self,
        state: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
    ) -> float:
        """
        Compute this component's instantaneous additive voltage contribution.

        Returns the component's contribution to h(x(t), u(t), theta_i(z(t))).

        Args:
            state (NDArray[np.float32]):
                Component state variables with shape (n_x,).
            params (NDArray[np.float32]):
                Component parameter values with shape (n_r,).
            current (float):
                Applied current in amps. Positive = discharge.

        Returns:
            float: Voltage contribution in volts.

        """

    @abstractmethod
    def jacobian_ode(
        self,
        state: NDArray[np.float32],
        state_jac: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
    ) -> NDArray[np.float32]:
        """
        Compute the rate of change of this component's state Jacobian.

        Implements the variational equation for J_k(t) = dx_k/dtheta
        integrated alongside ode::

            dJ_k/dt = (df_k/dx_k) * J_k + df_k/dtheta,   J_k(t=0) = 0

        where state_jac[i, j] = d(state[i]) / d(params[j]).

        Args:
            state (NDArray[np.float32]):
                Component state variables with shape (n_x,).
            state_jac (NDArray[np.float32]):
                Current state Jacobian with shape (n_x, n_r).
            params (NDArray[np.float32]):
                Component parameter values with shape (n_r,).
            current (float):
                Applied current in amps. Positive = discharge.

        Returns:
            NDArray[np.float32]:
                State Jacobian derivative dJ_k/dt with shape (n_x, n_r).

        """

    @abstractmethod
    def voltage_jacobian(
        self,
        state: NDArray[np.float32],
        state_jac: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
    ) -> NDArray[np.float32]:
        """
        Compute the gradient of this component's voltage w.r.t. its parameters.

        Applies the chain rule at each timestep::

            d(voltage)/d(params[j]) = (d(voltage)/d(state)) * state_jac[:, j]
                                      + d(voltage)/d(params[j])

        The first term (state path) uses state_jac accumulated by jacobian_ode.
        The second term captures any direct dependence of voltage on params
        that does not flow through state.

        Args:
            state (NDArray[np.float32]):
                Component state variables with shape (n_x,).
            state_jac (NDArray[np.float32]):
                Accumulated state Jacobian with shape (n_x, n_r).
            params (NDArray[np.float32]):
                Component parameter values with shape (n_r,).
            current (float):
                Applied current in amps. Positive = discharge.

        Returns:
            NDArray[np.float32]: d(voltage)/d(params) with shape (n_r,).

        """

    # ================================================
    # Helpers / Convenience Methods
    # ================================================
    def initial_state(self) -> np.ndarray:
        """
        Return the default initial state for this component.

        Returns a zero vector. Override when a non-zero default is physically
        meaningful. DataSegment.x_init always takes precedence when supplied.

        Returns:
            np.ndarray: Zero vector with shape (n_x,).

        """
        return np.zeros(len(self.state_names))

    @property
    def param_names(self) -> list[str]:
        """Ordered parameter names for this component."""
        return [p.name for p in self.param_specs]

    @property
    def n_params(self) -> int:
        """Number of parameters owned by this component: n_r."""
        return len(self.param_specs)

    @property
    def n_states(self) -> int:
        """Number of state variables owned by this component: n_x."""
        return len(self.state_names)


class OhmicComponent(ECMComponent):
    """
    Instantaneous ohmic voltage drop with no hidden state.

    Contributes R_0(z) * u(t) to the terminal voltage, where R_0(z) is the
    series resistance indexed j=0 in the parameter vector theta_i(z). Because
    there are no dynamics, n_x = 0 and jacobian_ode returns an empty
    array::

        voltage = current * R_0
        d(voltage)/d(R_0) = current

    """

    def __init__(
        self,
        r0_bounds: T_BOUNDS = (1e-5, 0.5),
    ) -> None:
        """
        Initialize an OhmicComponent.

        Args:
            r0_bounds (T_BOUNDS):
                Physical bounds for R_0 in ohms.
                Pass a 2-tuple ``(lo, hi)`` or a 3-tuple ``(lo, guess, hi)`` where
                the middle value is stored as the initial guess used by
                CellFitter.bootstrap_init. Defaults to ``(1e-5, 0.5)``.

        """
        bounds, guess = _parse_bounds(r0_bounds)
        self._r0_bounds = bounds
        self._r0_guess = guess

    @property
    def param_specs(self) -> list[ParameterSpec]:
        """Parameter specification for R_0."""
        return [ParameterSpec("R0", self._r0_bounds, guess=self._r0_guess)]

    @property
    def voltage_label(self) -> str:
        """Voltage contribution label: 'V_R0'."""
        return "V_R0"

    @property
    def state_names(self) -> list[str]:
        """Empty list; OhmicComponent has no state variables."""
        return []

    def ode(
        self,
        state: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
    ) -> NDArray[np.float32]:
        """
        Return an empty state derivative array.

        OhmicComponent has no dynamics; R_0 produces an instantaneous voltage
        drop with no memory of prior current.

        Args:
            state (NDArray[np.float32]): Unused; empty state vector.
            params (NDArray[np.float32]): Component parameters ``[R0]``.
            current (float): Applied current in amps. Positive = discharge.

        Returns:
            NDArray[np.float32]: Empty array with shape (0,).

        """
        return np.array([])

    def voltage(
        self,
        state: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
    ) -> float:
        """
        Compute the ohmic voltage contribution.

        Returns current * R_0 where params[0] is R_0.

        Args:
            state (NDArray[np.float32]):
                Unused; empty state vector.
            params (NDArray[np.float32]):
                Component parameters ``[R0]``.
            current (float):
                Applied current in amps. Positive = discharge.

        Returns:
            float: Ohmic voltage drop in volts.

        """
        return float(current * params[0])

    def jacobian_ode(
        self,
        state: NDArray[np.float32],
        state_jac: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
    ) -> NDArray[np.float32]:
        """
        Return an empty state Jacobian derivative array.

        OhmicComponent has no state, so the variational equation is empty.

        Args:
            state (NDArray[np.float32]):
                Unused; empty state vector.
            state_jac (NDArray[np.float32]):
                Unused; empty Jacobian.
            params (NDArray[np.float32]):
                Component parameters ``[R0]``.
            current (float):
                Applied current in amps. Positive = discharge.

        Returns:
            NDArray[np.float32]: Zero array with shape (0, 1).

        """
        return np.zeros((0, 1))

    def voltage_jacobian(
        self,
        state: NDArray[np.float32],
        state_jac: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
    ) -> NDArray[np.float32]:
        """
        Compute the gradient of voltage with respect to R_0.

        d(voltage)/d(R_0) = current (purely direct-path; no state dependence).

        Args:
            state (NDArray[np.float32]):
                Unused; empty state vector.
            state_jac (NDArray[np.float32]):
                Unused; empty Jacobian.
            params (NDArray[np.float32]):
                Component parameters ``[R0]``.
            current (float):
                Applied current in amps. Positive = discharge.

        Returns:
            NDArray[np.float32]: Gradient array ``[current]`` with shape (1,).

        """
        return np.array([current])


class RCComponent(ECMComponent):
    """
    One RC pair contributing polarization voltage to the terminal voltage.

    Implements the j-th RC pair with parameters R_j(z) (resistance, ohms)
    and tau_j(z) = R_j(z) * C_j(z) (time constant, seconds). State v_j(t)
    is the voltage across the j-th capacitor::

        d(v_j)/dt = -v_j / tau_j + u(t) * R_j / tau_j

    State Jacobian ODE (state_jac shape (1, 2))::

        d(state_jac[0, 0])/dt = -state_jac[0, 0] / tau + current / tau
        d(state_jac[0, 1])/dt = -state_jac[0, 1] / tau + (v_rc - current * r) / tau**2

    Voltage contribution (state path only)::

        d(voltage)/d(r)   = state_jac[0, 0]
        d(voltage)/d(tau) = state_jac[0, 1]

    """

    def __init__(
        self,
        index: int,
        r_bounds: T_BOUNDS = (1e-5, 0.5),
        tau_bounds: T_BOUNDS = (0.5, 600.0),
    ) -> None:
        """
        Initialize an RCComponent.

        Args:
            index (int):
                1-based index used to name the parameters (e.g. ``R1``, ``Tau1``).
                Two RCComponent instances in the same model must have different indices.
            r_bounds (T_BOUNDS):
                Physical bounds for R_j in ohms.
                Pass a 2-tuple ``(lo, hi)`` or a 3-tuple ``(lo, guess, hi)``.
                Defaults to ``(1e-5, 0.5)``.
            tau_bounds (T_BOUNDS):
                Physical bounds for tau_j in seconds.
                Pass a 2-tuple ``(lo, hi)`` or a 3-tuple ``(lo, guess, hi)``.
                Defaults to ``(0.5, 600.0)``.

        """
        self._k = index
        self._r_bounds, self._r_guess = _parse_bounds(r_bounds)
        self._tau_bounds, self._tau_guess = _parse_bounds(tau_bounds)

    @property
    def param_specs(self) -> list[ParameterSpec]:
        """Parameter specifications for R_j and tau_j."""
        k = self._k
        return [
            ParameterSpec(f"R{k}", self._r_bounds, guess=self._r_guess),
            ParameterSpec(f"Tau{k}", self._tau_bounds, guess=self._tau_guess),
        ]

    @property
    def voltage_label(self) -> str:
        """Voltage contribution label: 'V_RCk'."""
        return f"V_RC{self._k}"

    @property
    def state_names(self) -> list[str]:
        """State variable name: 'V_RCk'."""
        return [f"V_RC{self._k}"]

    def ode(
        self,
        state: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
    ) -> NDArray[np.float32]:
        """
        Compute the RC state derivative.

        The RC pair relaxes toward the steady-state polarization voltage
        current * R_j with time constant tau_j::

            d(v_j)/dt = -v_j / tau_j + current * R_j / tau_j

        state[0] is v_j; params = [R_j, tau_j].

        Args:
            state (NDArray[np.float32]):
                Component state ``[v_j]`` with shape (1,).
            params (NDArray[np.float32]):
                Component parameters ``[R_j, tau_j]`` with shape (2,).
            current (float):
                Applied current in amps. Positive = discharge.

        Returns:
            NDArray[np.float32]: State derivative d(v_j)/dt with shape (1,).

        """
        r, tau = params
        v_rc = state[0]
        return np.array([-v_rc / tau + current * r / tau])

    def voltage(
        self,
        state: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
    ) -> float:
        """
        Return the RC voltage contribution.

        The RC pair voltage equals its state variable v_j(t), which
        accumulates the polarization history over time. state[0] is v_j.

        Args:
            state (NDArray[np.float32]):
                Component state ``[v_j]`` with shape (1,).
            params (NDArray[np.float32]):
                Component parameters ``[R_j, tau_j]`` with shape (2,). Unused.
            current (float):
                Applied current in amps. Unused.

        Returns:
            float: Polarization voltage v_j(t) in volts.

        """
        return float(state[0])

    def jacobian_ode(
        self,
        state: NDArray[np.float32],
        state_jac: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
    ) -> NDArray[np.float32]:
        """
        Compute the variational equation for the RC state Jacobian.

        Differentiating the state ODE w.r.t. R_j and tau_j gives::

            d(state_jac[0, 0])/dt = -state_jac[0, 0] / tau + current / tau
            d(state_jac[0, 1])/dt = -state_jac[0, 1] / tau + (v_rc - current * r) / tau**2

        state_jac[0, 0] = d(v_j)/d(R_j), state_jac[0, 1] = d(v_j)/d(tau_j).

        Args:
            state (NDArray[np.float32]):
                Component state ``[v_j]`` with shape (1,).
            state_jac (NDArray[np.float32]):
                Current state Jacobian with shape (1, 2).
            params (NDArray[np.float32]):
                Component parameters ``[R_j, tau_j]`` with shape (2,).
            current (float):
                Applied current in amps. Positive = discharge.

        Returns:
            NDArray[np.float32]: State Jacobian derivative dJ/dt with shape (1, 2).

        """
        r, tau = params
        v_rc = state[0]
        dj = np.zeros_like(state_jac)
        dj[0, 0] = -state_jac[0, 0] / tau + current / tau
        dj[0, 1] = -state_jac[0, 1] / tau + (v_rc - current * r) / tau**2
        return dj

    def voltage_jacobian(
        self,
        state: NDArray[np.float32],
        state_jac: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
    ) -> NDArray[np.float32]:
        """
        Compute the gradient of the RC voltage w.r.t. R_j and tau_j.

        Since voltage = v_j = state[0], the gradient is purely a state path
        with no direct dependence of voltage on params::

            d(voltage)/d(R_j)   = d(v_j)/d(R_j)   = state_jac[0, 0]
            d(voltage)/d(tau_j) = d(v_j)/d(tau_j) = state_jac[0, 1]

        Args:
            state (NDArray[np.float32]):
                Component state ``[v_j]`` with shape (1,). Unused.
            state_jac (NDArray[np.float32]):
                Accumulated state Jacobian with shape (1, 2).
            params (NDArray[np.float32]):
                Component parameters ``[R_j, tau_j]`` with shape (2,). Unused.
            current (float):
                Applied current in amps. Unused.

        Returns:
            NDArray[np.float32]: Voltage gradient state_jac[0, :] with shape (2,).

        """
        return state_jac[0, :]
