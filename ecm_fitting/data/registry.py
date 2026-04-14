from __future__ import annotations

from ecm_fitting.data.device import BatteryType

# ================================================
# Battery type registry
# ================================================
BATTERY_REGISTRY: dict[str, BatteryType] = {
    "APR18650M1B": BatteryType(
        nominal_capacity_ah=1.25,
        brand="LithiumWerks",
        model_number="APR18650M1B",
        form_factor="18650",
        cathode="LFP",
        anode="Gr",
        nominal_voltage_v=3.2,
        operating_voltage_v=(2.0, 3.6),
    ),
}

# ================================================
# Initial parameter bounds registry
#
# Values are stored as (min, guess, max) 3-tuples. Factory functions in
# ecm_fitting.ecm.common accept these directly and extract (min, max) as bounds
# while storing the guess in ParameterSpec for use by CellFitter.bootstrap_init.
# ================================================
_INIT_BOUNDS: dict[str, dict[str, tuple[float, float, float]]] = {
    "APR18650M1B": {
        "r0_bounds": (0.010, 0.020, 0.060),
        "r1_bounds": (0.001, 0.060, 0.100),
        "tau1_bounds": (1, 20.0, 150.0),
        "r2_bounds": (0.005, 0.100, 0.300),
        "tau2_bounds": (0.5, 40.0, 900.0),
    },
}


def get_battery_type(model_number: str) -> BatteryType:
    """
    Return the :class:`~ecm_fitting.data.device.BatteryType` for a registered model.

    Arguments:
        model_number (str):
            Manufacturer part number, e.g. ``"APR18650M1B"``.

    Returns:
        BatteryType: The registered battery type.

    Raises:
        KeyError: If ``model_number`` is not in the registry.

    """
    if model_number not in BATTERY_REGISTRY:
        available = ", ".join(sorted(BATTERY_REGISTRY))
        msg = (
            f"No battery type registered for model '{model_number}'. "
            f"Available models: {available}"
        )
        raise KeyError(msg)
    return BATTERY_REGISTRY[model_number]


def get_init_bounds_for_device(
    model_number: str,
) -> dict[str, tuple[float, float, float]]:
    """
    Return initial parameter bounds for a registered battery model.

    Bounds are stored as ``(min, guess, max)`` 3-tuples and can be passed
    directly to ECM factory functions::

        from ecm_fitting.data.registry import get_init_bounds_for_device
        from ecm_fitting.ecm.common import first_order_ecm

        x0 = get_init_bounds_for_device("APR18650M1B")
        model = first_order_ecm(**x0)  # extra keys silently ignored

    Arguments:
        model_number (str):
            Manufacturer part number, e.g. ``"APR18650M1B"``.

    Returns:
        dict[str, tuple[float, float, float]]: A copy of the bounds dict.

    Raises:
        KeyError: If ``model_number`` is not in the bounds registry.

    """
    if model_number not in _INIT_BOUNDS:
        available = ", ".join(sorted(_INIT_BOUNDS))
        msg = (
            f"No parameter bounds registered for model '{model_number}'. "
            f"Available models: {available}"
        )
        raise KeyError(msg)
    return dict(_INIT_BOUNDS[model_number])
