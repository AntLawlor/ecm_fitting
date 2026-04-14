from __future__ import annotations

from ecm_fitting.ecm.components import T_BOUNDS, OhmicComponent, RCComponent
from ecm_fitting.ecm.model import ECM

# Global default parameter bounds
_DFLT: dict[str, T_BOUNDS] = {
    "r0_bounds": (1e-5, 0.5),
    "r1_bounds": (1e-5, 0.5),
    "tau1_bounds": (0.5, 600.0),
    "r2_bounds": (1e-5, 0.5),
    "tau2_bounds": (0.5, 600.0),
}


def ohmic_ecm(
    r0_bounds: T_BOUNDS = _DFLT["r0_bounds"],
    **kwargs,  # noqa: ARG001
) -> ECM:
    """
    Build a purely-ohmic ECM: one ``OhmicComponent`` with no RC dynamics.

    Arguments:
        r0_bounds (T_BOUNDS):
            Bounds for ``R0`` in ohms. Either ``(lo, hi)`` or ``(lo, guess, hi)``.
            Defaults to ``(1e-5, 0.5)``.
        **kwargs:
            Extra keyword arguments are silently ignored, so registry dicts from
            :func:`~ecm_fitting.data.registry.get_init_bounds_for_device` can be
            passed directly without filtering.

    Returns:
        ECM: Assembled ohmic ECM with ``param_names == ["R0"]``.

    """
    return ECM([OhmicComponent(r0_bounds)])


def first_order_ecm(
    r0_bounds: T_BOUNDS = _DFLT["r0_bounds"],
    r1_bounds: T_BOUNDS = _DFLT["r1_bounds"],
    tau1_bounds: T_BOUNDS = _DFLT["tau1_bounds"],
    **kwargs,  # noqa: ARG001
) -> ECM:
    """
    Build a first-order (1RC) ECM: ``OhmicComponent`` + one ``RCComponent``.

    Arguments:
        r0_bounds (T_BOUNDS):
            Bounds for ``R0`` in ohms. Either ``(lo, hi)`` or ``(lo, guess, hi)``.
            Defaults to ``(1e-5, 0.5)``.
        r1_bounds (T_BOUNDS):
            Bounds for ``R1`` in ohms. Defaults to ``(1e-5, 0.5)``.
        tau1_bounds (T_BOUNDS):
            Bounds for ``Tau1`` in seconds. Defaults to ``(0.5, 600.0)``.
        **kwargs:
            Silently ignored — allows passing a full registry dict (e.g. one that
            also contains ``r2_bounds``, ``tau2_bounds``) without error.

    Returns:
        ECM: Assembled 1RC ECM with ``param_names == ["R0", "R1", "Tau1"]``.

    Example::

        from ecm_fitting.data.registry import get_init_bounds_for_device
        from ecm_fitting.ecm.common import first_order_ecm

        model = first_order_ecm(**get_init_bounds_for_device("APR18650M1B"))

    """
    return ECM(
        [
            OhmicComponent(r0_bounds),
            RCComponent(1, r1_bounds, tau1_bounds),
        ],
    )


def second_order_ecm(
    r0_bounds: T_BOUNDS = _DFLT["r0_bounds"],
    r1_bounds: T_BOUNDS = _DFLT["r1_bounds"],
    tau1_bounds: T_BOUNDS = _DFLT["tau1_bounds"],
    r2_bounds: T_BOUNDS = _DFLT["r2_bounds"],
    tau2_bounds: T_BOUNDS = _DFLT["tau2_bounds"],
    **kwargs,  # noqa: ARG001
) -> ECM:
    """
    Build a second-order (2RC) ECM: ``OhmicComponent`` + two ``RCComponent`` s.

    Arguments:
        r0_bounds (T_BOUNDS):
            Bounds for ``R0`` in ohms. Defaults to ``(1e-5, 0.5)``.
        r1_bounds (T_BOUNDS):
            Bounds for ``R1`` in ohms. Defaults to ``(1e-5, 0.5)``.
        tau1_bounds (T_BOUNDS):
            Bounds for ``Tau1`` in seconds. Defaults to ``(0.5, 600.0)``.
        r2_bounds (T_BOUNDS):
            Bounds for ``R2`` in ohms. Defaults to ``(1e-5, 0.5)``.
        tau2_bounds (T_BOUNDS):
            Bounds for ``Tau2`` in seconds. Defaults to ``(0.5, 600.0)``.
        **kwargs:
            Silently ignored for registry-dict compatibility.

    Returns:
        ECM: Assembled 2RC ECM with ``param_names == ["R0", "R1", "Tau1", "R2", "Tau2"]``.

    """
    return ECM(
        [
            OhmicComponent(r0_bounds),
            RCComponent(1, r1_bounds, tau1_bounds),
            RCComponent(2, r2_bounds, tau2_bounds),
        ],
    )
