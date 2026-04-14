from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ecm_fitting.ecm.components import ECMComponent, ParameterSpec


class ECM:
    """
    An assembled equivalent circuit model (ECM).

    Implements the state-space model (f, h) used for voltage prediction::

        dx(t)/dt = f(x(t), u(t), theta_i(z(t)))
        y(t)     = h(x(t), u(t), theta_i(z(t)))

    where x(t) is the internal state (dimension n_x), u(t) is the applied
    current, y(t) is the predicted terminal voltage, and theta_i(z(t)) is
    the cell parameter vector (dimension n_r) evaluated at conditioning
    point z(t). Individual ECMComponents are concatenated into full state
    and parameter vectors. SOC-dependence is external: v_oc is passed in
    at each call site.

    Attributes:
        n_params (int): n_r -- number of parameters per conditioning point.
        n_states (int): n_x -- dimension of the internal state x(t).

    """

    def __init__(self, components: list[ECMComponent]) -> None:
        """
        Initialize an ECM from a list of components.

        Args:
            components (list[ECMComponent]):
                Ordered list of ECM components.
                Determines the structure of the state and parameter vectors.

        """
        self.components = components
        self._s_slices, self._p_slices = self._build_slices()

    def _build_slices(self) -> tuple[list[slice], list[slice]]:
        """
        Build index slices for accessing per-component state and parameter blocks.

        Returns:
            tuple[list[slice], list[slice]]:
                A pair (state_slices, param_slices) where each list contains one
                slice per component into the concatenated state and parameter
                vectors respectively.

        """
        s_idx, p_idx = 0, 0
        ss: list[slice] = []
        ps: list[slice] = []
        for c in self.components:
            ss.append(slice(s_idx, s_idx + c.n_states))
            s_idx += c.n_states
            ps.append(slice(p_idx, p_idx + c.n_params))
            p_idx += c.n_params
        return ss, ps

    def ode(
        self,
        state: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
    ) -> NDArray[np.float32]:
        """
        Compute the state derivative for all ECM components.

        Evaluates f(x(t), u(t), theta_i(z(t))) by concatenating each
        component's state ODE output.

        Args:
            state (NDArray[np.float32]):
                Full state vector with shape (n_x,).
            params (NDArray[np.float32]):
                Full parameter vector with shape (n_r,).
            current (float):
                Applied current in amps. Positive = discharge.

        Returns:
            NDArray[np.float32]: State derivative dx/dt with shape (n_x,).

        """
        return np.concatenate(
            [c.ode(state[s], params[p], current) for c, s, p in self._iter()],
        )

    def voltage(
        self,
        state: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
        v_oc: float,
    ) -> float:
        """
        Compute the predicted terminal voltage at a single timestep.

        Evaluates h(x(t), u(t), theta_i(z(t))) = v_oc + sum of component
        voltage contributions.

        Args:
            state (NDArray[np.float32]):
                Full state vector with shape (n_x,).
            params (NDArray[np.float32]):
                Full parameter vector with shape (n_r,).
            current (float):
                Applied current in amps. Positive = discharge.
            v_oc (float):
                Open-circuit voltage in volts.

        Returns:
            float: Predicted terminal voltage in volts.

        """
        return v_oc + sum(
            c.voltage(state[s], params[p], current) for c, s, p in self._iter()
        )

    def voltage_breakdown(
        self,
        state: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
        v_oc: float,
    ) -> dict[str, float]:
        """
        Return the per-component voltage contributions at a single timestep.

        Returns a dict mapping each component's voltage_label to its
        additive voltage contribution, plus 'V_oc' for the open-circuit
        voltage.

        Args:
            state (NDArray[np.float32]):
                Full state vector with shape (n_x,).
            params (NDArray[np.float32]):
                Full parameter vector with shape (n_r,).
            current (float):
                Applied current in amps. Positive = discharge.
            v_oc (float):
                Open-circuit voltage in volts.

        Returns:
            dict[str, float]: Mapping from component label (or 'V_oc') to
                voltage contribution in volts.

        """
        breakdown: dict[str, float] = {"V_oc": v_oc}
        for c, s, p in self._iter():
            breakdown[c.voltage_label] = c.voltage(state[s], params[p], current)
        return breakdown

    def jacobian_ode(
        self,
        state: NDArray[np.float32],
        state_jac: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
    ) -> NDArray[np.float32]:
        """
        Compute the state Jacobian ODE for all components.

        Evaluates dJ/dt where J = d(state)/d(params), assembling the
        block-diagonal result from each component's jacobian_ode. Cross-
        component entries are zero.

        Args:
            state (NDArray[np.float32]):
                Full state vector with shape (n_x,).
            state_jac (NDArray[np.float32]):
                Current state Jacobian with shape (n_x, n_r).
                state_jac[i, j] = d(state[i])/d(params[j]).
            params (NDArray[np.float32]):
                Full parameter vector with shape (n_r,).
            current (float):
                Applied current in amps. Positive = discharge.

        Returns:
            NDArray[np.float32]:
                State Jacobian derivative dJ/dt with shape (n_x, n_r).

        """
        d_state_jac = np.zeros_like(state_jac)
        for c, s, p in self._iter():
            d_state_jac[s, p] = c.jacobian_ode(
                state[s],
                state_jac[s, p],
                params[p],
                current,
            )
        return d_state_jac

    def voltage_jacobian(
        self,
        state: NDArray[np.float32],
        state_jac: NDArray[np.float32],
        params: NDArray[np.float32],
        current: float,
    ) -> NDArray[np.float32]:
        """
        Compute the gradient of terminal voltage with respect to all parameters.

        Assembles d(voltage)/d(params) by writing each component's
        contribution into its own parameter slice. v_oc has no dependence
        on params and contributes nothing.

        Args:
            state (NDArray[np.float32]):
                Full state vector with shape (n_x,).
            state_jac (NDArray[np.float32]):
                Accumulated state Jacobian with shape (n_x, n_r).
            params (NDArray[np.float32]):
                Full parameter vector with shape (n_r,).
            current (float):
                Applied current in amps. Positive = discharge.

        Returns:
            NDArray[np.float32]:
                Voltage gradient d(voltage)/d(params) with shape (n_r,).

        """
        d_voltage = np.zeros(self.n_params)
        for c, s, p in self._iter():
            d_voltage[p] = c.voltage_jacobian(
                state[s],
                state_jac[s, p],
                params[p],
                current,
            )
        return d_voltage

    def initial_state(
        self,
        state_override: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """
        Return the initial state vector for the ECM.

        Concatenates the default initial states from each component.
        state_override (from ``DataSegment.x_init``) takes precedence when
        provided.

        Args:
            state_override (NDArray[np.float32] | None):
                Optional full state vector to use instead of component defaults.
                Defaults to None.

        Returns:
            NDArray[np.float32]: Initial state vector with shape (n_x,).

        """
        default = np.concatenate([c.initial_state() for c in self.components])
        return state_override if state_override is not None else default

    @property
    def param_specs(self) -> list[ParameterSpec]:
        """Ordered parameter specifications across all components."""
        return [p for c in self.components for p in c.param_specs]

    @property
    def state_names(self) -> list[str]:
        """Ordered state variable names across all components."""
        return [s for c in self.components for s in c.state_names]

    @property
    def n_params(self) -> int:
        """Number of ECM parameters per conditioning point: n_r."""
        return sum(c.n_params for c in self.components)

    @property
    def n_states(self) -> int:
        """Dimension of the internal state x(t): n_x."""
        return sum(c.n_states for c in self.components)

    @property
    def param_names(self) -> list[str]:
        """Ordered parameter names across all components."""
        return [p.name for p in self.param_specs]

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Physical (lower, upper) bounds for each parameter."""
        return [s.bounds for s in self.param_specs]

    def param_index(self, name: str) -> int:
        """
        Return the index of a named parameter in the flat parameter vector.

        Args:
            name (str):
                Parameter name as declared in the component's ParameterSpec.

        Returns:
            int: Zero-based index into the parameter vector.

        """
        return self.param_names.index(name)

    def _iter(self):
        """Yield (component, state_slice, param_slice) tuples for all components."""
        yield from zip(self.components, self._s_slices, self._p_slices, strict=True)
