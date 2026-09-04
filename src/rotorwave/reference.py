"""Ready-made rotors from the literature, so the examples are one line long."""

from __future__ import annotations

from .materials import STEEL
from .model import PeriodicRotor

__all__ = ["reference_rotor", "test_rig_rotor"]


def reference_rotor(n_disks: int = 11, elements_per_cell: int = 40) -> PeriodicRotor:
    """The numerical case study of Lamas & Nicoletti (2024), Section 3.

    A 1500 mm steel shaft of 100 mm diameter carrying evenly spaced disks of
    380 mm diameter and 22 mm thickness — properties close to those of an
    industrial gas compressor.  With 11 disks the rotor shows a band gap
    between roughly 2000 Hz and 3500 Hz.

    Parameters
    ----------
    n_disks:
        Number of disks, i.e. number of unit cells.
    elements_per_cell:
        Mesh density of one cell.
    """
    return PeriodicRotor(
        shaft_length=1.5,
        shaft_diameter=100e-3,
        n_cells=n_disks,
        disk_diameter=380e-3,
        disk_thickness=22e-3,
        material=STEEL,
        elements_per_cell=elements_per_cell,
        disk_position=0.5,
    )


def test_rig_rotor(n_cells: int = 4, elements_per_cell: int = 40) -> PeriodicRotor:
    """Nominal geometry of the laboratory rotor of Section 4.

    A slender steel shaft of 20 mm diameter with 960 mm between bearings,
    carrying disks of 200 mm diameter and 12 mm thickness spaced every 240 mm —
    so the periodic cell is 240 mm long and the span between bearings covers
    four cells.

    Notes
    -----
    The band gap predicted from this *nominal* geometry sits near
    210-285 Hz, whereas the paper reports 190-260 Hz.  The difference is
    expected: the experimental section uses a model updated against the
    measured FRFs, in which the conical bushings that clamp the disks add mass
    that the nominal geometry does not account for (each disk weighs 3 kg with
    its bushing against 2.96 kg for the bare disk).  The correlated model is
    not reproduced here.

    The physical rig has three disks over four spans; ``full_rotor()`` builds
    the ideal rotor with one disk per cell, so use it for the wave analysis and
    build the finite model explicitly if you need the exact rig layout.
    """
    return PeriodicRotor(
        shaft_length=0.24 * n_cells,
        shaft_diameter=20e-3,
        n_cells=n_cells,
        disk_diameter=200e-3,
        disk_thickness=12e-3,
        material=STEEL,
        elements_per_cell=elements_per_cell,
        disk_position=0.5,
    )
