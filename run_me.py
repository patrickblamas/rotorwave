"""START HERE — runs without installing anything.

Open this file in Spyder (or any IDE shipped with Anaconda) and press F5.  It
puts the package on the Python path by itself, runs a complete analysis of the
11-disk rotor of the paper and shows the figures.

What it does, in order:

1. builds the rotor and prints its geometry;
2. solves the omega(k) problem — the dispersion diagram of the unit cell, from
   which the band gaps are read;
3. solves the k(omega) problem and computes the receptance of the finite rotor,
   so that the predicted stop band can be checked against the frequency
   response an experiment would measure.

To install the package properly (recommended if you want to use it from your
own scripts), see INSTALACAO.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- Make the package importable without installing it ----------------------
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rotorwave import (  # noqa: E402
    STEEL,
    KOmegaSolver,
    OmegaKSolver,
    ReceptanceSolver,
    PeriodicRotor,
)
from rotorwave.plotting import (  # noqa: E402
    plot_dispersion,
    plot_komega_dispersion,
    set_style,
)


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

SPIN_RPM = 6000.0              # rotating speed of the analysis [rpm]
F_MAX = 5000.0                 # upper limit of the figures [Hz]


def main() -> None:
    set_style()

    # 1) The rotor of the paper: a 1500 mm shaft carrying 11 disks of 380 mm.
    print(ROTOR.summary())
    print()

    # 2) omega(k) problem.  A real wavenumber is prescribed and the
    #    frequencies that propagate with it are computed; the frequency ranges
    #    no branch reaches are the band gaps.
    cell = ROTOR.unit_cell()
    dispersion = OmegaKSolver(cell).solve(
        spin_rpm=SPIN_RPM, n_points=61, n_branches=8
    )
    print(dispersion.summary("forward"))
    print(dispersion.summary("backward"))
    print()

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    plot_dispersion(dispersion, ax=ax, f_max=F_MAX)
    ax.set_title(f"Dispersion of the periodic rotor at {SPIN_RPM:.0f} rpm", loc="left")

    # 3) k(omega) problem and frequency response of the finite rotor.  The real
    #    part of the wavenumber describes the propagation and the imaginary
    #    part the attenuation, so the stop bands must line up with the quiet
    #    regions of the receptance — that is the physical check.
    frequencies = np.arange(10.0, F_MAX + 1.0, 10.0)
    complex_k = KOmegaSolver(cell).solve(frequencies, spin_rpm=SPIN_RPM)
    receptance = ReceptanceSolver(ROTOR.full_rotor(elements_per_cell=10)).compute(
        frequencies, spin_rpm=SPIN_RPM
    )
    plot_komega_dispersion(complex_k, receptance, f_max=F_MAX)

    plt.show()
    print("Done. The other examples are in the examples/ folder.")


if __name__ == "__main__":
    main()
