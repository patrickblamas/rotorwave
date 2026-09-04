"""Isotropic materials used by the finite element models.

All quantities are in SI units (Pa, kg/m^3).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Material", "STEEL"]


@dataclass(frozen=True)
class Material:
    """An isotropic, homogeneous material.

    Parameters
    ----------
    young_modulus:
        Young's modulus :math:`E` [Pa].
    density:
        Mass density :math:`\\rho` [kg/m^3].
    poisson_ratio:
        Poisson's ratio :math:`\\nu` [-]. Only used by shear-flexible
        formulations; the Euler-Bernoulli element ignores it.
    name:
        Human readable label, used in ``__str__`` and in report headers.
    """

    young_modulus: float
    density: float
    poisson_ratio: float = 0.3
    name: str = "material"

    def __post_init__(self) -> None:
        if self.young_modulus <= 0.0:
            raise ValueError("young_modulus must be positive")
        if self.density <= 0.0:
            raise ValueError("density must be positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must lie in (-1, 0.5)")

    @property
    def shear_modulus(self) -> float:
        """Shear modulus :math:`G = E / [2(1+\\nu)]` [Pa]."""
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.name} (E = {self.young_modulus:.3e} Pa, "
            f"rho = {self.density:.1f} kg/m^3)"
        )


#: Structural steel used in Lamas & Nicoletti (2024).
STEEL = Material(young_modulus=2.1e11, density=7.85e3, poisson_ratio=0.3, name="steel")
