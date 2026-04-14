from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ecm_fitting.fitting.losses.base import LossResult, LossTerm

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ecm_fitting.data.segment import DataSegment
    from ecm_fitting.ecm.simulator import ECMSimulator, SimulationResult
    from ecm_fitting.ecm.surface import ParameterSurface


class VoltageMSELoss(LossTerm):
    """
    Mean squared voltage residual across all segments and timesteps.

    Loss::

        sum_{segments} sum_t residual(t)^2 / T_total

    """

    name = "voltage_mse"

    def evaluate(
        self,
        surface: ParameterSurface,
        segments: list[DataSegment],
        results: list[SimulationResult] | None,
        simulator: ECMSimulator | None = None,
    ) -> LossResult:
        if results is None:
            raise ValueError("VoltageMSELoss requires simulation results.")

        t_total = sum(r.residuals.size for r in results)
        mse = sum(float(np.sum(r.residuals**2)) for r in results) / t_total

        grad: NDArray[np.float32] | None = None
        residuals: NDArray[np.float32] | None = None
        residual_jac: NDArray[np.float32] | None = None

        if simulator is not None and any(r.voltage_jac is not None for r in results):
            raw = simulator.accumulate_surface_gradient(segments, results, surface)
            grad = (2.0 / t_total) * raw
            residuals, residual_jac = simulator.build_residual_jacobian(
                segments,
                results,
                surface,
            )

        return LossResult(
            total=mse,
            components={self.name: mse},
            gradient=grad,
            residuals=residuals,
            residual_jac=residual_jac,
        )
