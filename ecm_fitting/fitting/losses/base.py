from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse import vstack as sp_vstack

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ecm_fitting.data.segment import DataSegment
    from ecm_fitting.ecm.simulator import ECMSimulator, SimulationResult
    from ecm_fitting.ecm.surface import ParameterSurface


@dataclass
class LossResult:
    """Scalar loss value, named components, optional flat gradient, and optional residuals."""

    total: float
    """Scalar loss value."""

    components: dict[str, float]
    """Named loss components and associated scalar loss value."""

    gradient: NDArray[np.float32] | None
    """Optional gradient associated with this loss value."""

    residuals: NDArray[np.float32] | None = None
    """Residual vector with shape ``(n_r,)`` for use with ``scipy.optimize.least_squares``."""

    residual_jac: NDArray[np.float32] | csr_matrix | None = None
    """Residual Jacobian with shape ``(n_r, n_free_params)`` for use with least_squares.
    May be a dense ``NDArray`` or a ``scipy.sparse.csr_matrix``."""


class LossTerm(ABC):
    """Abstract base for a single additive loss contribution."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier used in ``LossResult.components``."""

    @abstractmethod
    def evaluate(
        self,
        surface: ParameterSurface,
        segments: list[DataSegment],
        results: list[SimulationResult] | None,
        simulator: ECMSimulator | None = None,
    ) -> LossResult:
        """
        Compute this term's scalar loss and flat gradient.

        Arguments:
            surface (ParameterSurface):
                Current parameter surface (already restored from flat params).
            segments (list[DataSegment]):
                The segments being fitted.
            results (list[SimulationResult] | None):
                Simulation results for each segment. ``None`` when simulation
                was skipped (e.g. gradient-free terms in a no-Jacobian pass).
            simulator (ECMSimulator | None):
                The simulator instance, passed by ``CompositeLoss``. Terms that
                need ``accumulate_surface_gradient`` (i.e. ``VoltageMSELoss``)
                use this; all other terms ignore it.

        Returns:
            LossResult: Scalar total, named breakdown, and flat gradient
            (or ``None`` if this term contributes no gradient).

        """


class _GradientFreeTerm(LossTerm):
    """
    Mixin for terms whose gradient is computed analytically, not via simulation.

    Subclasses (``SmoothnessLoss``, ``GPShrinkageLoss``) never need the
    Jacobian simulation pass. ``CompositeLoss`` uses this marker to skip
    the Jacobian when only gradient-free terms are present.
    """


class CompositeLoss:
    """
    Summed ``LossTerm`` objects.

    The simulator is run once per call; all terms receive the same results.
    The Jacobian simulation pass is skipped entirely when every term is a
    ``_GradientFreeTerm``.
    """

    def __init__(
        self,
        terms: LossTerm | list[LossTerm],
        simulator: ECMSimulator,
    ) -> None:
        self.terms = terms if isinstance(terms, list) else [terms]
        self.simulator = simulator
        self._last_components: dict[str, float] = {}

    def _evaluate_terms(
        self,
        flat_params: NDArray[np.float32],
        surface: ParameterSurface,
        segments: list[DataSegment],
    ) -> list[LossResult]:
        """Run simulation once and evaluate all terms."""
        surface.from_flat(flat_params)
        compute_jac = any(not isinstance(t, _GradientFreeTerm) for t in self.terms)
        results = self.simulator.simulate_segments(
            segments,
            surface,
            compute_jacobian=compute_jac,
        )
        term_results = [
            term.evaluate(surface, segments, results, self.simulator)
            for term in self.terms
        ]
        self._last_components = {
            term.name: lr.total
            for term, lr in zip(self.terms, term_results, strict=True)
        }
        return term_results

    def residuals_and_jacobian(
        self,
        flat_params: NDArray[np.float32],
        surface: ParameterSurface,
        segments: list[DataSegment],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Return stacked residuals and Jacobian for ``scipy.optimize.least_squares``.

        Steps:
            1. Restore ``surface`` from ``flat_params``
            2. Simulate all segments (with Jacobian if any loss term needs it)
            3. Collect ``residuals`` and ``residual_jac`` from each term

        Returns:
            tuple of residuals and jacobian
                - ``r_full``: ``(n_total_residuals,)``
                - ``J_full``: ``(n_total_residuals, n_free_params)``

        """
        term_results = self._evaluate_terms(flat_params, surface, segments)

        all_r: list[NDArray[np.float32]] = []
        all_j: list = []
        has_sparse = False
        for lr in term_results:
            if lr.residuals is not None and lr.residual_jac is not None:
                all_r.append(lr.residuals)
                all_j.append(lr.residual_jac)
                if issparse(lr.residual_jac):
                    has_sparse = True

        r_full = np.concatenate(all_r) if all_r else np.zeros(0, dtype=np.float32)
        if not all_j:
            j_full = np.zeros((0, len(flat_params)), dtype=np.float32)
        elif has_sparse:
            blocks = [j if issparse(j) else csr_matrix(j) for j in all_j]
            j_full = sp_vstack(blocks, format="csr")
        else:
            j_full = np.vstack(all_j)
        return r_full, j_full

    def __call__(
        self,
        flat_params: NDArray[np.float32],
        surface: ParameterSurface,
        segments: list[DataSegment],
    ) -> tuple[float, NDArray[np.float32]]:
        """
        Evaluate total scalar loss and flat gradient (derived from residuals).

        Returns:
            tuple[float, NDArray[np.float32]]: ``(total_loss, flat_gradient)``.

        """
        r_full, j_full = self.residuals_and_jacobian(flat_params, surface, segments)
        total_loss = float(0.5 * np.sum(r_full**2))
        if j_full.shape[0] > 0:
            grad = np.asarray(j_full.T @ r_full).ravel()
        else:
            grad = np.zeros_like(flat_params)
        return total_loss, grad.astype(np.float32)

    def component_losses(self) -> dict[str, float]:
        """Named loss components from the most recent evaluation."""
        return dict(self._last_components)
