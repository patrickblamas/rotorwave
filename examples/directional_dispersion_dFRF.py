"""Directional dispersion diagrams and dFRFs.

Forward waves are drawn on the positive frequency axis and backward waves on
the negative one, together with the directional frequency response functions
of the finite rotor.  Read together, the two halves show that the forward and
the backward band gaps of a rotating machine are not the same band.

"""

from __future__ import annotations

import _path_setup  # noqa: F401  (lets the file run without installing)

from pathlib import Path

import numpy as np

from rotorwave import STEEL, KOmegaSolver, PeriodicRotor, ReceptanceSolver
from rotorwave.plotting import plot_directional_dispersion, set_style

OUTPUT = Path(__file__).parent / "figures"

SPIN_RPM = 6000.0
F_MAX = 7000.0


# ----------------------------------------------------------------------
# Rotor geometry and material — change these to analyse a different rotor
# ----------------------------------------------------------------------
ROTOR = PeriodicRotor(
    shaft_length=1.5,          # total shaft length [m]
    shaft_diameter=100e-3,     # shaft outer diameter [m]
    n_cells=11,                # number of disks = number of unit cells
    disk_diameter=380e-3,      # disk outer diameter [m]
    disk_thickness=22e-3,      # disk thickness [m]
    material=STEEL,            # Material(young_modulus, density, poisson_ratio)
    elements_per_cell=40,      # mesh density of one cell
)


def main() -> None:
    set_style()
    OUTPUT.mkdir(exist_ok=True)

    frequencies = np.arange(10.0, F_MAX + 1.0, 10.0)

    dispersion = KOmegaSolver(ROTOR.unit_cell()).solve(frequencies, spin_rpm=SPIN_RPM)
    receptance = ReceptanceSolver(ROTOR.full_rotor(elements_per_cell=10)).compute(
        frequencies, spin_rpm=SPIN_RPM
    )

    fig, _ = plot_directional_dispersion(dispersion, receptance, f_max=F_MAX)
    path = OUTPUT / "directional_disp_dFRF.png"
    fig.savefig(path)

    forward = receptance.resonances("forward")
    backward = receptance.resonances("backward")
    print(
        "last forward resonance below the gap : "
        f"{forward[forward < 3000].max():.0f} Hz"
    )
    print(
        "last backward resonance below the gap: "
        f"{backward[backward < 3000].max():.0f} Hz"
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
