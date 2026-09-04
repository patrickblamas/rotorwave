"""Finite elements for rotating shafts.

The shaft element is the classical Euler-Bernoulli rotating beam element of
Nelson & McVaugh (1976), with translational inertia, rotary inertia and the
gyroscopic coupling between the two lateral planes.  Disks are treated as
rigid bodies attached to a node.

Degree of freedom ordering (per node, and throughout the whole package)::

    [x, y, beta, gamma]

with ``x``/``y`` the lateral translations along the X/Y axes and
``beta``/``gamma`` the rotations about the X/Y axes.  The shaft spins about
the Z axis, so a positive spin speed corresponds to a rotation from X to Y.

References
----------
H.D. Nelson, J.M. McVaugh, "The dynamics of rotor-bearing systems using
finite elements", J. Eng. Ind. 98 (2) (1976) 593-600.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .materials import Material

__all__ = ["DOF_PER_NODE", "DOF_NAMES", "ShaftElement", "RigidDisk"]

#: Number of degrees of freedom of every node of the models.
DOF_PER_NODE = 4

#: Name of each nodal degree of freedom, in assembly order.
DOF_NAMES = ("x", "y", "beta", "gamma")


@dataclass(frozen=True)
class ShaftElement:
    """A two-node rotating Euler-Bernoulli shaft element.

    Parameters
    ----------
    length:
        Element length [m].
    outer_diameter:
        Outer diameter of the cross section [m].
    material:
        Element material.
    inner_diameter:
        Inner diameter [m], for hollow shafts.  Defaults to a solid section.

    Notes
    -----
    The element matrices reproduce exactly the ones used in the reference
    MATLAB implementation for a solid section; the hollow section is the
    natural generalisation obtained by keeping :math:`\\rho A` and
    :math:`\\rho I` as the inertia coefficients instead of hard-coding
    :math:`I = A r^2 / 4`.
    """

    length: float
    outer_diameter: float
    material: Material
    inner_diameter: float = 0.0

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError("element length must be positive")
        if self.outer_diameter <= 0.0:
            raise ValueError("outer_diameter must be positive")
        if not 0.0 <= self.inner_diameter < self.outer_diameter:
            raise ValueError("inner_diameter must lie in [0, outer_diameter)")

    # ------------------------------------------------------------------
    # Section properties
    # ------------------------------------------------------------------
    @property
    def outer_radius(self) -> float:
        """Outer radius of the shaft cross-section [m]."""
        return 0.5 * self.outer_diameter

    @property
    def inner_radius(self) -> float:
        """Inner radius [m]; zero for a solid shaft."""
        return 0.5 * self.inner_diameter

    @property
    def area(self) -> float:
        """Cross-sectional area :math:`A` [m^2]."""
        return math.pi * (self.outer_radius**2 - self.inner_radius**2)

    @property
    def second_moment(self) -> float:
        """Diametral second moment of area :math:`I` [m^4]."""
        return 0.25 * math.pi * (self.outer_radius**4 - self.inner_radius**4)

    @property
    def mass(self) -> float:
        """Element mass [kg]."""
        return self.material.density * self.area * self.length

    # ------------------------------------------------------------------
    # Element matrices
    # ------------------------------------------------------------------
    def mass_matrix(self) -> np.ndarray:
        """Consistent inertia matrix (translational + rotary), shape ``(8, 8)``."""
        le = self.length
        mu = self.material.density * self.area
        c_translation = mu * le / 420.0
        c_rotary = self.material.density * self.second_moment / (30.0 * le)
        return c_translation * _MT(le) + c_rotary * _MR(le)

    def gyroscopic_matrix(self) -> np.ndarray:
        """Skew-symmetric gyroscopic matrix, shape ``(8, 8)``.

        The matrix multiplies the spin speed :math:`\\Omega` in the equation of
        motion :math:`M\\ddot q + (C - \\Omega G)\\dot q + K q = f`.
        """
        le = self.length
        c_gyro = 2.0 * self.material.density * self.second_moment / (30.0 * le)
        return c_gyro * _G(le)

    def stiffness_matrix(self) -> np.ndarray:
        """Bending stiffness matrix, shape ``(8, 8)``."""
        le = self.length
        c_stiff = self.material.young_modulus * self.second_moment / le**3
        return c_stiff * _K(le)


@dataclass(frozen=True)
class RigidDisk:
    """A rigid disk lumped at a node.

    Parameters
    ----------
    mass:
        Disk mass [kg].
    polar_inertia:
        Polar moment of inertia :math:`I_p` [kg m^2] (about the spin axis).
    diametral_inertia:
        Diametral moment of inertia :math:`I_t` [kg m^2].
    """

    mass: float
    polar_inertia: float
    diametral_inertia: float

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("disk mass must be positive")
        if self.polar_inertia <= 0.0 or self.diametral_inertia <= 0.0:
            raise ValueError("disk inertias must be positive")

    @classmethod
    def from_geometry(
        cls,
        diameter: float,
        thickness: float,
        material: Material,
        inner_diameter: float = 0.0,
    ) -> RigidDisk:
        """Build a disk from its geometry, assuming a uniform annular plate."""
        r_o = 0.5 * diameter
        r_i = 0.5 * inner_diameter
        if not 0.0 <= r_i < r_o:
            raise ValueError("inner_diameter must lie in [0, diameter)")
        mass = material.density * math.pi * (r_o**2 - r_i**2) * thickness
        polar = 0.5 * mass * (r_o**2 + r_i**2)
        diametral = mass * (3.0 * (r_o**2 + r_i**2) + thickness**2) / 12.0
        return cls(mass=mass, polar_inertia=polar, diametral_inertia=diametral)

    def mass_matrix(self) -> np.ndarray:
        """Nodal inertia matrix, shape ``(4, 4)``."""
        return np.diag(
            [self.mass, self.mass, self.diametral_inertia, self.diametral_inertia]
        )

    def gyroscopic_matrix(self) -> np.ndarray:
        """Nodal gyroscopic matrix, shape ``(4, 4)``."""
        g = np.zeros((4, 4))
        g[2, 3] = -self.polar_inertia
        g[3, 2] = self.polar_inertia
        return g


# ----------------------------------------------------------------------
# Shape-function matrices (Nelson & McVaugh, 1976)
# ----------------------------------------------------------------------
def _MT(le: float) -> np.ndarray:
    """Translational inertia pattern (to be scaled by ``rho*A*le/420``)."""
    le2 = le * le
    return np.array(
        [
            [156, 0, 0, 22 * le, 54, 0, 0, -13 * le],
            [0, 156, -22 * le, 0, 0, 54, 13 * le, 0],
            [0, -22 * le, 4 * le2, 0, 0, -13 * le, -3 * le2, 0],
            [22 * le, 0, 0, 4 * le2, 13 * le, 0, 0, -3 * le2],
            [54, 0, 0, 13 * le, 156, 0, 0, -22 * le],
            [0, 54, -13 * le, 0, 0, 156, 22 * le, 0],
            [0, 13 * le, -3 * le2, 0, 0, 22 * le, 4 * le2, 0],
            [-13 * le, 0, 0, -3 * le2, -22 * le, 0, 0, 4 * le2],
        ],
        dtype=float,
    )


def _MR(le: float) -> np.ndarray:
    """Rotary inertia pattern (to be scaled by ``rho*I/(30*le)``)."""
    le2 = le * le
    return np.array(
        [
            [36, 0, 0, 3 * le, -36, 0, 0, 3 * le],
            [0, 36, -3 * le, 0, 0, -36, -3 * le, 0],
            [0, -3 * le, 4 * le2, 0, 0, 3 * le, -le2, 0],
            [3 * le, 0, 0, 4 * le2, -3 * le, 0, 0, -le2],
            [-36, 0, 0, -3 * le, 36, 0, 0, -3 * le],
            [0, -36, 3 * le, 0, 0, 36, 3 * le, 0],
            [0, -3 * le, -le2, 0, 0, 3 * le, 4 * le2, 0],
            [3 * le, 0, 0, -le2, -3 * le, 0, 0, 4 * le2],
        ],
        dtype=float,
    )


def _G(le: float) -> np.ndarray:
    """Gyroscopic pattern (to be scaled by ``2*rho*I/(30*le)``)."""
    le2 = le * le
    return np.array(
        [
            [0, -36, 3 * le, 0, 0, 36, 3 * le, 0],
            [36, 0, 0, 3 * le, -36, 0, 0, 3 * le],
            [-3 * le, 0, 0, -4 * le2, 3 * le, 0, 0, le2],
            [0, -3 * le, 4 * le2, 0, 0, 3 * le, -le2, 0],
            [0, 36, -3 * le, 0, 0, -36, -3 * le, 0],
            [-36, 0, 0, -3 * le, 36, 0, 0, -3 * le],
            [-3 * le, 0, 0, le2, 3 * le, 0, 0, -4 * le2],
            [0, -3 * le, -le2, 0, 0, 3 * le, 4 * le2, 0],
        ],
        dtype=float,
    )


def _K(le: float) -> np.ndarray:
    """Bending stiffness pattern (to be scaled by ``E*I/le**3``)."""
    le2 = le * le
    return np.array(
        [
            [12, 0, 0, 6 * le, -12, 0, 0, 6 * le],
            [0, 12, -6 * le, 0, 0, -12, -6 * le, 0],
            [0, -6 * le, 4 * le2, 0, 0, 6 * le, 2 * le2, 0],
            [6 * le, 0, 0, 4 * le2, -6 * le, 0, 0, 2 * le2],
            [-12, 0, 0, -6 * le, 12, 0, 0, -6 * le],
            [0, -12, 6 * le, 0, 0, 12, 6 * le, 0],
            [0, -6 * le, 2 * le2, 0, 0, 6 * le, 4 * le2, 0],
            [6 * le, 0, 0, 2 * le2, -6 * le, 0, 0, 4 * le2],
        ],
        dtype=float,
    )
