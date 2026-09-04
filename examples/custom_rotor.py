"""Template: run the whole analysis on a rotor of your own.

Change the block marked PARAMETERS and run the file.  Everything below it is
generic: the same code produces the band gaps, the dispersion diagram and the
receptance for any periodic rotor.

The bottom of the file shows two less common cases: a custom material and a
unit cell assembled element by element, for rotors that are not uniform.

"""

from __future__ import annotations

import _path_setup  # noqa: F401  (lets the file run without installing)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rotorwave import STEEL, OmegaKSolver, PeriodicRotor, ReceptanceSolver
from rotorwave.plotting import GAP_COLOR, plot_dispersion, set_style

OUTPUT = Path(__file__).parent / "figures"

# ----------------------------------------------------------------------
# PARAMETERS — everything here is yours to change
# ----------------------------------------------------------------------
ROTOR = PeriodicRotor(
    shaft_length=1.5,           # total shaft length [m]
    shaft_diameter=100e-3,      # shaft outer diameter [m]
    n_cells=11,                 # number of disks = number of unit cells
    disk_diameter=380e-3,       # disk outer diameter [m]
    disk_thickness=22e-3,       # disk thickness [m]
    material=STEEL,             # see "custom material" at the bottom
    elements_per_cell=40,       # mesh density of one cell
    disk_position=0.5,          # disk position inside the cell, 0 to 1
    shaft_inner_diameter=0.0,   # > 0 for a hollow shaft [m]
)

SPIN_RPM = 6000.0               # rotating speed of the analysis [rpm]
F_MAX = 7000.0                  # upper limit of the plots [Hz]
N_BRANCHES = 8                  # dispersion branches to compute
# ----------------------------------------------------------------------


def main() -> None:
    set_style()
    OUTPUT.mkdir(exist_ok=True)
    print(ROTOR.summary())

    cell = ROTOR.unit_cell()
    dispersion = OmegaKSolver(cell).solve(
        spin_rpm=SPIN_RPM, n_points=61, n_branches=N_BRANCHES
    )
    print(dispersion.summary("forward"))
    print(dispersion.summary("backward"))

    frequencies = np.arange(5.0, F_MAX + 1.0, 5.0)
    receptance = ReceptanceSolver(ROTOR.full_rotor(elements_per_cell=20)).compute(
        frequencies, spin_rpm=SPIN_RPM
    )

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.5, 5.0), sharex=True, constrained_layout=True
    )
    plot_dispersion(dispersion, ax=top, f_max=F_MAX)
    top.set_title(f"{SPIN_RPM:.0f} rpm", loc="left")
    top.set_xlabel("")

    for gap in dispersion.band_gaps("both", f_max=F_MAX):
        bottom.axvspan(gap.start_hz, gap.stop_hz, color=GAP_COLOR, alpha=0.18, lw=0)
    bottom.plot(frequencies, np.log10(np.abs(receptance.direct)), color="0.15", lw=1.0)
    bottom.set_xlabel("Frequency (Hz)")
    bottom.set_ylabel(r"$\log_{10}|H|$")
    bottom.set_xlim(0.0, F_MAX)

    path = OUTPUT / "custom_rotor.png"
    fig.savefig(path)
    print(f"saved {path}")


# ----------------------------------------------------------------------
# Less common cases
# ----------------------------------------------------------------------
def custom_material_example() -> PeriodicRotor:
    """A rotor made of a material other than the built-in steel."""
    from rotorwave import Material

    aluminium = Material(
        young_modulus=70e9,      # [Pa]
        density=2700.0,          # [kg/m^3]
        poisson_ratio=0.33,
        name="aluminium",
    )
    return PeriodicRotor(
        shaft_length=1.5,
        shaft_diameter=100e-3,
        n_cells=11,
        disk_diameter=380e-3,
        disk_thickness=22e-3,
        material=aluminium,
        elements_per_cell=20,
    )


def hand_built_cell_example():
    """A unit cell assembled element by element.

    Use this when the cell is not uniform — a stepped shaft, a hollow section
    over part of the length, or a disk whose inertias were measured rather than
    computed from its geometry.  The elements are given in order, the first and
    last nodes are the periodic boundaries, and the disks are placed by node
    index.
    """
    from rotorwave import RigidDisk, ShaftElement, UnitCell

    cell_length = 1.5 / 11
    n_elements = 20
    le = cell_length / n_elements

    # Thicker section in the middle of the cell, thinner towards the boundaries.
    elements = [
        ShaftElement(
            length=le,
            outer_diameter=100e-3 if 5 <= i < 15 else 80e-3,
            material=STEEL,
        )
        for i in range(n_elements)
    ]

    # Inertias entered directly, instead of RigidDisk.from_geometry(...).
    disk = RigidDisk(mass=19.6, polar_inertia=0.354, diametral_inertia=0.178)

    return UnitCell(elements=elements, disks={10: disk})


if __name__ == "__main__":
    main()
