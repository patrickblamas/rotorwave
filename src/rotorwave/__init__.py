"""rotorwave — wave propagation analysis of rotors with longitudinal periodicity.

An open source implementation of the methodology of

    P.B. Lamas, R. Nicoletti, "Wave analysis of rotors with longitudinal
    periodicity", Journal of Sound and Vibration 571 (2024) 118095.
    https://doi.org/10.1016/j.jsv.2023.118095

The package combines a rotordynamic finite element model (Nelson & McVaugh
elements plus rigid disks), the Bloch-Floquet periodicity conditions and
directional response functions, so that the forward and backward whirl waves
of a periodic rotor can be analysed separately.

Typical use::

    import numpy as np
    from rotorwave import OmegaKSolver, reference_rotor

    rotor = reference_rotor(n_disks=11)
    dispersion = OmegaKSolver(rotor.unit_cell()).solve(spin_rpm=6000)
    print(dispersion.summary("forward"))
"""

from __future__ import annotations

from .analysis import (
    BandGapMap,
    attenuation_report,
    band_gap_map,
    mesh_convergence,
)
from .dispersion import BandGap, KOmegaDispersion, OmegaKDispersion
from .elements import DOF_NAMES, DOF_PER_NODE, RigidDisk, ShaftElement
from .frf import FRFResult, ReceptanceSolver
from .materials import STEEL, Material
from .model import PeriodicRotor, RotorFEModel, UnitCell
from .reference import reference_rotor, test_rig_rotor
from .units import hz_to_rad, rad_to_hz, rad_to_rpm, rpm_to_rad
from .waves import KOmegaSolver, OmegaKSolver
from .whirl import whirl_index, whirl_label

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # materials and elements
    "Material",
    "STEEL",
    "ShaftElement",
    "RigidDisk",
    "DOF_NAMES",
    "DOF_PER_NODE",
    # models
    "RotorFEModel",
    "UnitCell",
    "PeriodicRotor",
    "reference_rotor",
    "test_rig_rotor",
    # solvers
    "OmegaKSolver",
    "KOmegaSolver",
    "ReceptanceSolver",
    # results
    "OmegaKDispersion",
    "KOmegaDispersion",
    "BandGap",
    "FRFResult",
    # analyses
    "band_gap_map",
    "BandGapMap",
    "attenuation_report",
    "mesh_convergence",
    # helpers
    "whirl_index",
    "whirl_label",
    "rpm_to_rad",
    "rad_to_rpm",
    "hz_to_rad",
    "rad_to_hz",
]
