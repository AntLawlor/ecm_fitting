from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryType:
    """Physical and chemical specification of a class of batteries under test."""

    nominal_capacity_ah: float
    """Nominal rated capacity (in Amp-hours). Required."""

    brand: str | None = None
    """Manufacturer brand name. E.g. ``"LithiumWerks"``, ``"Samsung"``, ``"Panasonic"``."""

    model_number: str | None = None
    """Manufacturer model/part number. E.g. ``"APR18650M1B"``, ``"INR21700"``."""

    form_factor: str | None = None
    """Physical form factor. E.g. ``"18650"``, ``"21700"``, ``"prismatic"``, ``"pouch"``."""

    cathode: str | None = None
    """Cathode active material. E.g. ``"LFP"``, ``"NMC811"``, ``"NCA"``, ``"LCO"``."""

    anode: str | None = None
    """Anode active material. E.g. ``"Gr"``, ``"LTO"``, ``"Si-Gr"``."""

    nominal_voltage_v: float | None = None
    """Nominal operating voltage (in volts)."""

    operating_voltage_v: tuple[float, float] | None = None
    """Nominal operating voltage range (in volts)."""

    @property
    def label(self) -> str:
        """Human-readable label built from all non-None fields."""
        parts: list[str] = []
        if self.brand is not None:
            parts.append(self.brand)
        if self.model_number is not None:
            parts.append(self.model_number)
        if self.form_factor is not None:
            parts.append(self.form_factor)
        if self.cathode is not None and self.anode is not None:
            parts.append(f"{self.cathode}/{self.anode}")
        elif self.cathode is not None:
            parts.append(self.cathode)
        elif self.anode is not None:
            parts.append(self.anode)
        return " ".join(parts) if parts else f"{self.nominal_capacity_ah}Ah"


@dataclass(frozen=True)
class Battery:
    """A specific physical instance of a :class:`BatteryType`."""

    battery_type: BatteryType
    """The class of battery this instance belongs to."""

    cell_id: str
    """Unique identifier for this physical cell or pack."""

    @property
    def label(self) -> str:
        """Human-readable label: ``"<battery_type.label>:<cell_id>"``."""
        return f"{self.battery_type.label}:{self.cell_id}"
