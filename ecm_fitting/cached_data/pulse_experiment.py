"""
pulse_experiment.py
====================
Directory layout expected on disk
----------------------------------
experiment_root/
    pulsing_cell_3_220_ohm/          ← protocol "pulse sequence sweep"
        experiment_settings.txt      ← JSON with "Experiment" key
        8_pulsing_cell_3_220_ohm_<ts>.csv
        9_pulsing_cell_3_220_ohm_<ts>.csv
        ...
    pulsing3_cell_3_220_ohm/         ← protocol "pulse sequence sweep 2"
        experiment_settings.txt
        14_pulsing3_cell_3_220_ohm_<ts>.csv
        ...
    pulsing_cell_3_baseline/
        ...
    pulsing3_cell_3_baseline/
        ...

File naming convention
-----------------------
{step_number}_{folder_name}_{timestamp}.csv

Each CSV contains one pulse group:
    brief OCP  → Constant Current (charge)  → OCP (rest)
              → Constant Current (discharge) → OCP (rest) → 5-min OCP

Hierarchy
----------
Experiment
  └── Cell  (one physical cell)
        ├── CellCondition "baseline"
        │     ├── Protocol "pulse sequence sweep"
        │     │     └── [StepGroup, StepGroup, ...]
        │     └── Protocol "pulse sequence sweep 2"
        │           └── [StepGroup, StepGroup, ...]
        └── CellCondition "shorted"  (e.g. 220_ohm)
              ├── Protocol "pulse sequence sweep"
              └── Protocol "pulse sequence sweep 2"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps the folder-name fragment → canonical protocol name
PROTOCOL_NAME_MAP = {
    "pulse sequence sweep 2": "pulse sequence sweep 2",
    "pulse sequence sweep": "pulse sequence sweep",
}

# Step-name strings used by the Squidstat software
_OCP = "Open Circuit Potential"
_CC = "Constant Current"

# Regex to extract step number from a CSV filename
_STEP_RE = re.compile(r"^(\d+)_")

# Regex to extract cell_id and condition from a folder name, e.g.:
#   pulsing_cell_3_220_ohm   → cell_id=3, condition=220_ohm
#   pulsing3_cell_11_baseline → cell_id=11, condition=baseline
_FOLDER_RE = re.compile(r"pulsing\d*_cell_(\d+)_(.*)")


# ---------------------------------------------------------------------------
# StepGroup  (the atomic unit — one charge + one discharge pulse + rests)
# ---------------------------------------------------------------------------

@dataclass
class StepGroup:
    """
    One charge/discharge pulse pair from a single CSV file.

    Attributes
    ----------
    first_step : int
        Step number of the leading brief OCP (the number in the filename).
    df : pd.DataFrame
        Raw data for all steps in this group, with an added ``t_rel`` column
        (seconds relative to the start of this group).
    protocol_name : str
        Canonical protocol name ("pulse sequence sweep" or "pulse sequence sweep 2").
    source_file : Path
        Path to the CSV this group was loaded from.
    """

    first_step: int
    df: pd.DataFrame
    protocol_name: str
    source_file: Path

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def charge_pulse(self) -> pd.DataFrame:
        """Rows belonging to the first (charge) CC step."""
        cc_steps = self.df[self.df["Step name"] == _CC]["Step number"].unique()
        if len(cc_steps) == 0:
            return pd.DataFrame()
        return self.df[self.df["Step number"] == cc_steps[0]]

    @property
    def discharge_pulse(self) -> pd.DataFrame:
        """Rows belonging to the second (discharge) CC step."""
        cc_steps = self.df[self.df["Step name"] == _CC]["Step number"].unique()
        if len(cc_steps) < 2:
            return pd.DataFrame()
        return self.df[self.df["Step number"] == cc_steps[1]]

    @property
    def nominal_current_mA(self) -> float:
        """Nominal charge current magnitude in mA (from the CC step)."""
        cp = self.charge_pulse
        if cp.empty:
            return float("nan")
        return round(cp["Current (A)"].mean() * 1000, 1)

    @property
    def v0_charge(self) -> float:
        """Voltage just before the charge pulse (last OCP sample before CC)."""
        ocp_steps = self.df[self.df["Step name"] == _OCP]["Step number"].unique()
        if len(ocp_steps) == 0:
            return float("nan")
        pre_ocp = self.df[self.df["Step number"] == ocp_steps[0]]
        return float(pre_ocp["Working Electrode (V)"].iloc[-1])

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def plot(self, ax_v=None, ax_i=None, **plot_kwargs):
        """
        Plot voltage and current vs. relative time for this step group.

        Parameters
        ----------
        ax_v, ax_i : matplotlib Axes, optional
            If provided, plots onto existing axes.  Otherwise creates a new
            figure with two stacked subplots.
        **plot_kwargs
            Passed directly to ``ax.plot()``.  Useful for ``color``, ``label``,
            ``linestyle``, etc.
        """
        if ax_v is None or ax_i is None:
            fig, (ax_v, ax_i) = plt.subplots(2, 1, sharex=True)
        ax_v.plot(self.df["t_rel"], self.df["Working Electrode (V)"], **plot_kwargs)
        ax_i.plot(self.df["t_rel"], self.df["Current (A)"] * 1000, **plot_kwargs)
        ax_v.set_ylabel("Voltage (V)")
        ax_i.set_ylabel("Current (mA)")
        ax_i.set_xlabel("Time (s)")
        return ax_v, ax_i

    def __repr__(self):
        return (
            f"StepGroup(first_step={self.first_step}, "
            f"I≈{self.nominal_current_mA:.0f} mA, "
            f"protocol='{self.protocol_name}')"
        )


# ---------------------------------------------------------------------------
# Protocol  (collection of StepGroups from one protocol run on one condition)
# ---------------------------------------------------------------------------

class Protocol:
    """
    All pulse step-groups from a single protocol directory.

    Parameters
    ----------
    protocol_name : str
        Canonical protocol name.
    step_groups : list of StepGroup
        Ordered by ``first_step`` (i.e., ascending step number).
    """

    def __init__(self, protocol_name: str, step_groups: list[StepGroup]):
        self.protocol_name = protocol_name
        # Sort by first_step so iteration order is always chronological
        self._groups: list[StepGroup] = sorted(step_groups, key=lambda g: g.first_step)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_by_step(self, first_step: int) -> StepGroup:
        """Return the StepGroup whose CSV filename starts with *first_step*."""
        for g in self._groups:
            if g.first_step == first_step:
                return g
        raise KeyError(f"No StepGroup with first_step={first_step} in {self}")

    def get_by_index(self, idx: int) -> StepGroup:
        """Return the idx-th pulse group (0-based, chronological order)."""
        return self._groups[idx]

    def get_by_current(self, nominal_mA: float, tol_mA: float = 20.0) -> list[StepGroup]:
        """
        Return all step groups whose charge current is within *tol_mA* of
        *nominal_mA*.
        """
        return [
            g for g in self._groups
            if abs(abs(g.nominal_current_mA) - abs(nominal_mA)) <= tol_mA
        ]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def step_groups(self) -> list[StepGroup]:
        return list(self._groups)

    def __len__(self):
        return len(self._groups)

    def __iter__(self):
        return iter(self._groups)

    def __repr__(self):
        return (
            f"Protocol(name='{self.protocol_name}', "
            f"n_groups={len(self._groups)})"
        )


# ---------------------------------------------------------------------------
# CellCondition  (one physical condition: baseline or a specific short)
# ---------------------------------------------------------------------------

class CellCondition:
    """
    One cell under one condition (baseline or shorted).

    Parameters
    ----------
    condition : str
        ``'baseline'`` or a resistance string such as ``'220_ohm'``.
    protocols : dict mapping canonical protocol name → Protocol
    """

    def __init__(self, condition: str, protocols: dict[str, Protocol]):
        self.condition = condition
        self._protocols = protocols

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_protocol(self, protocol_name: str) -> Protocol:
        """
        Retrieve a Protocol by name.  Accepts the full canonical name
        (``'pulse sequence sweep'`` / ``'pulse sequence sweep 2'``) or the
        shorthand ``'sweep1'`` / ``'sweep2'``.
        """
        aliases = {
            "sweep1": "pulse sequence sweep",
            "sweep2": "pulse sequence sweep 2",
            "1": "pulse sequence sweep",
            "2": "pulse sequence sweep 2",
        }
        key = aliases.get(protocol_name, protocol_name)
        if key not in self._protocols:
            raise KeyError(
                f"Protocol '{protocol_name}' not found.  "
                f"Available: {list(self._protocols)}"
            )
        return self._protocols[key]

    def get_step_group(self, protocol_name: str, first_step: int) -> StepGroup:
        return self.get_protocol(protocol_name).get_by_step(first_step)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def protocol_names(self) -> list[str]:
        return list(self._protocols)

    def __repr__(self):
        return (
            f"CellCondition(condition='{self.condition}', "
            f"protocols={self.protocol_names})"
        )


# ---------------------------------------------------------------------------
# Cell  (one physical cell — owns both baseline and shorted CellConditions)
# ---------------------------------------------------------------------------

class Cell:
    """
    One physical LFP cell, containing both baseline and shorted conditions.

    Parameters
    ----------
    cell_id : str
        e.g. ``'3'``, ``'11'``, ``'12'``
    baseline : CellCondition
    shorted : CellCondition
    """

    def __init__(
        self,
        cell_id: str,
        baseline: CellCondition,
        shorted: CellCondition,
    ):
        self.cell_id = cell_id
        self.baseline = baseline
        self.shorted = shorted

    # ------------------------------------------------------------------
    # Core comparison method
    # ------------------------------------------------------------------

    def compare_step(
        self,
        protocol_name: str,
        first_step: int,
        fig=None,
        show: bool = True,
    ) -> tuple[plt.Figure, tuple]:
        """
        Overlay baseline vs. shorted voltage/current for one step group.

        Parameters
        ----------
        protocol_name : str
            Full or alias protocol name (e.g. ``'sweep1'``, ``'sweep2'``,
            ``'pulse sequence sweep 2'``).
        first_step : int
            First-step number identifying the group within the protocol.
        fig : matplotlib Figure, optional
            Reuse an existing figure.
        show : bool
            Call ``plt.show()`` at the end if True.

        Returns
        -------
        fig, (ax_v, ax_i)
        """
        sg_base = self.baseline.get_step_group(protocol_name, first_step)
        sg_short = self.shorted.get_step_group(protocol_name, first_step)

        if fig is None:
            fig, (ax_v, ax_i) = plt.subplots(
                2, 1, sharex=True, figsize=(9, 5),
                gridspec_kw={"hspace": 0.05},
            )
        else:
            ax_v, ax_i = fig.axes[0], fig.axes[1]

        sg_base.plot(ax_v, ax_i, label="baseline", color="steelblue")
        sg_short.plot(ax_v, ax_i,
                      label=f"shorted ({self.shorted.condition})",
                      color="tomato", linestyle="--")

        proto_short = self.shorted.get_protocol(protocol_name)
        fig.suptitle(
            f"Cell {self.cell_id} | {proto_short.protocol_name} | "
            f"step {first_step} | "
            f"I≈{sg_base.nominal_current_mA:.0f} mA"
        )
        ax_v.legend(fontsize=8)
        ax_v.set_xlabel("")

        if show:
            plt.show()

        return fig, (ax_v, ax_i)

    def compare_all_steps(
        self,
        protocol_name: str,
        show: bool = True,
    ) -> list[tuple[plt.Figure, tuple]]:
        """
        Call ``compare_step`` for every step group present in *both*
        baseline and shorted for the given protocol.

        Returns a list of (fig, axes) tuples in step order.
        """
        proto_base = self.baseline.get_protocol(protocol_name)
        proto_short = self.shorted.get_protocol(protocol_name)

        base_steps = {g.first_step for g in proto_base}
        short_steps = {g.first_step for g in proto_short}
        common_steps = sorted(base_steps & short_steps)

        if not common_steps:
            raise ValueError(
                f"No common first_steps between baseline and "
                f"{self.shorted.condition} for protocol '{protocol_name}'."
            )

        results = []
        for step in common_steps:
            fig_axes = self.compare_step(protocol_name, step, show=show)
            results.append(fig_axes)

        return results

    # ------------------------------------------------------------------
    # Factory / loader
    # ------------------------------------------------------------------

    @classmethod
    def from_directories(
        cls,
        baseline_dirs: list[Path | str],
        shorted_dirs: list[Path | str],
        cell_id: Optional[str] = None,
    ) -> "Cell":
        """
        Build a Cell from lists of directory paths.

        Parameters
        ----------
        baseline_dirs : list of Path
            One or more directories for the baseline condition
            (one per protocol).
        shorted_dirs : list of Path
            One or more directories for the shorted condition
            (one per protocol).  All dirs must share the same condition
            string (e.g. all ``220_ohm``).
        cell_id : str, optional
            If not given, inferred from the folder names.

        Returns
        -------
        Cell

        Examples
        --------
        >>> cell = Cell.from_directories(
        ...     baseline_dirs=["data/pulsing_cell_3_baseline",
        ...                    "data/pulsing3_cell_3_baseline"],
        ...     shorted_dirs=["data/pulsing_cell_3_220_ohm",
        ...                   "data/pulsing3_cell_3_220_ohm"],
        ... )
        """
        baseline_dirs = [Path(d) for d in baseline_dirs]
        shorted_dirs = [Path(d) for d in shorted_dirs]

        # Infer cell_id from first directory name if not supplied
        if cell_id is None:
            m = _FOLDER_RE.search(baseline_dirs[0].name)
            if m:
                cell_id = m.group(1)
            else:
                cell_id = "unknown"

        baseline_condition = cls._build_condition(baseline_dirs)
        shorted_condition = cls._build_condition(shorted_dirs)

        return cls(
            cell_id=cell_id,
            baseline=baseline_condition,
            shorted=shorted_condition,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_condition(dirs: list[Path]) -> CellCondition:
        """
        Parse one or more protocol directories into a CellCondition.

        Each directory must contain:
          - A .txt settings file with a JSON ``"Experiment"`` field.
          - One or more CSV files named ``{step}_{folder_name}_{timestamp}.csv``.
        """
        protocols: dict[str, Protocol] = {}
        condition: Optional[str] = None

        for directory in dirs:
            directory = Path(directory)
            if not directory.is_dir():
                raise FileNotFoundError(f"Directory not found: {directory}")

            # ---- 1. Identify condition from folder name ----
            m = _FOLDER_RE.search(directory.name)
            if m:
                dir_condition = m.group(2)  # e.g. "baseline" or "220_ohm"
                if condition is None:
                    condition = dir_condition
                elif condition != dir_condition:
                    raise ValueError(
                        f"Condition mismatch across dirs: "
                        f"'{condition}' vs '{dir_condition}'"
                    )
            else:
                raise ValueError(
                    f"Cannot parse cell_id/condition from folder name: "
                    f"'{directory.name}'.  "
                    f"Expected pattern: pulsing[N]_cell_<id>_<condition>"
                )

            # ---- 2. Read protocol name from the settings .txt ----
            txt_files = list(directory.glob("*.txt"))
            if not txt_files:
                raise FileNotFoundError(
                    f"No .txt settings file found in {directory}"
                )
            protocol_name = Cell._parse_protocol_name(txt_files[0])

            # ---- 3. Load all CSV files in this directory ----
            csv_files = sorted(directory.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {directory}")

            step_groups = [
                Cell._load_step_group(csv_path, protocol_name)
                for csv_path in csv_files
            ]

            protocols[protocol_name] = Protocol(protocol_name, step_groups)

        if condition is None:
            condition = "unknown"

        return CellCondition(condition=condition, protocols=protocols)

    @staticmethod
    def _parse_protocol_name(txt_path: Path) -> str:
        """
        Extract the canonical protocol name from the experiment settings .txt.

        The file is JSON; we read the ``"Experiment"`` field from
        ``Application Metadata``.
        """
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(text)
            raw_name = data["Application Metadata"]["Experiment"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(
                f"Could not parse protocol name from {txt_path}: {exc}"
            ) from exc

        # Normalise
        raw_lower = raw_name.strip().lower()
        for key, canonical in PROTOCOL_NAME_MAP.items():
            if raw_lower == key:
                return canonical

        # Unknown protocol — store as-is with a warning
        import warnings
        warnings.warn(
            f"Unrecognised protocol name '{raw_name}' in {txt_path}. "
            "Storing as-is."
        )
        return raw_name.strip()

    @staticmethod
    def _load_step_group(csv_path: Path, protocol_name: str) -> StepGroup:
        """
        Load one CSV into a StepGroup.

        Adds a ``t_rel`` column (seconds since the first row of this file).
        """
        m = _STEP_RE.match(csv_path.name)
        if not m:
            raise ValueError(
                f"Cannot infer step number from filename: '{csv_path.name}'.  "
                f"Expected name to start with an integer step number."
            )
        first_step = int(m.group(1))

        df = pd.read_csv(csv_path)

        # Validate expected columns
        required = {"Step number", "Step name", "Elapsed Time (s)",
                    "Working Electrode (V)", "Current (A)"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV {csv_path.name} is missing columns: {missing}"
            )

        # Relative time within this step group
        df = df.copy()
        df["t_rel"] = df["Elapsed Time (s)"] - df["Elapsed Time (s)"].iloc[0]

        return StepGroup(
            first_step=first_step,
            df=df,
            protocol_name=protocol_name,
            source_file=csv_path,
        )

    def __repr__(self):
        return (
            f"Cell(id='{self.cell_id}', "
            f"baseline={self.baseline}, "
            f"shorted={self.shorted})"
        )


# ---------------------------------------------------------------------------
# Experiment  (top-level container for all cells)
# ---------------------------------------------------------------------------

class Experiment:
    """
    All cells in the study.

    Parameters
    ----------
    cells : list of Cell
    """

    def __init__(self, cells: list[Cell]):
        self._cells: dict[str, Cell] = {c.cell_id: c for c in cells}

    def get_cell(self, cell_id: str) -> Cell:
        if cell_id not in self._cells:
            raise KeyError(
                f"Cell '{cell_id}' not found.  "
                f"Available: {list(self._cells)}"
            )
        return self._cells[cell_id]

    @property
    def cells(self) -> list[Cell]:
        return list(self._cells.values())

    def compare_all(
        self,
        protocol_name: str,
        first_step: int,
        show: bool = True,
    ):
        """Compare one step across every cell."""
        for cell in self.cells:
            cell.compare_step(protocol_name, first_step, show=show)

    def __repr__(self):
        return f"Experiment(cells={list(self._cells)})"