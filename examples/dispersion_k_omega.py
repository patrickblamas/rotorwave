"""Dispersion from the k(omega) problem, with the rotor receptance.

The k(omega) problem returns complex wavenumbers, so it also shows the
evanescent waves that the omega(k) problem cannot represent: the top panel is
propagation, the middle one attenuation, and the bottom one the receptance of
the finite 11-disk rotor, whose silent regions coincide with the stop bands.

"""

from __future__ import annotations

import _path_setup  # noqa: F401  (lets the file run without installing)

from pathlib import Path

import numpy as np

from rotorwave import STEEL, KOmegaSolver, PeriodicRotor, ReceptanceSolver
from rotorwave.plotting import plot_komega_dispersion, set_style

OUTPUT = Path(__file__).parent / "figures"

SPIN_RPM = 6000.0
F_MAX = 13000.0


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
    receptance = ReceptanceSolver(ROTOR.full_rotor(elements_per_cell=20)).compute(
        frequencies, spin_rpm=SPIN_RPM
    )

    fig, _ = plot_komega_dispersion(dispersion, receptance, f_max=F_MAX)
    path = OUTPUT / "dispersion_k_omega.png"
    fig.savefig(path)

    print(dispersion.summary("forward"))
    print(dispersion.summary("backward"))
    print("quiet bands of the receptance:")
    for start, stop in receptance.quiet_bands():
        print(f"  {start:8.0f} - {stop:8.0f} Hz")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
