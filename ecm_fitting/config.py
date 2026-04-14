from __future__ import annotations

from enum import Enum


class CurrentConvention(Enum):
    """
    Sign convention for the current stored in :class:`~ecm_fitting.data.DataSegment`.

    The ECM model equations use the **electrochemical** convention internally
    (positive current = charging). This setting tells the simulator how to
    convert the segment current before passing it to the model.
    """

    POSITIVE_IS_DISCHARGE = "positive_is_discharge"
    """
    Positive current indicates the cell is discharging (SOC decreasing).

    This matches common BMS data formats.
    """

    POSITIVE_IS_CHARGE = "positive_is_charge"
    """
    Positive current indicates the cell is charging (SOC increasing).

    This is the default and matches the electrochemical convention.
    """


_convention: CurrentConvention = CurrentConvention.POSITIVE_IS_CHARGE


def get_convention() -> CurrentConvention:
    """Return the currently configured :class:`CurrentConvention`."""
    return _convention


def set_convention(convention: CurrentConvention) -> None:
    """
    Set the package-wide current sign convention.

    Call this once at the start of a script or notebook before constructing
    any :class:`~ecm_fitting.data.DataSegment` objects.

    Arguments:
        convention (CurrentConvention): The desired sign convention.

    """
    global _convention  # noqa: PLW0603
    _convention = convention


def _current_sign() -> float:
    """
    Factor that converts segment current to the electrochemical convention.

    The ECM model expects positive = charging internally. Multiply the raw
    segment current by this value before passing it to model methods.

    Returns:
        -1.0 for :attr:`CurrentConvention.POSITIVE_IS_DISCHARGE`,
        +1.0 for :attr:`CurrentConvention.POSITIVE_IS_CHARGE`.

    """
    if _convention is CurrentConvention.POSITIVE_IS_DISCHARGE:
        return -1.0
    return 1.0
