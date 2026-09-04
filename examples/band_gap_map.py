"""How the stop bands move with the rotating speed.

The wave-Campbell diagram shows the drift of the band gaps as a colour map.
This example measures it: the band gap edges are detected automatically for
every speed and reported as numbers, so the shift and the (almost constant)
width of each stop band can be quoted directly.

"""

from __future__ import annotations

import _path_setup  # noqa: F401  (lets the file run without installing)

from pathlib import Path

import numpy as np

from rotorwave import STEEL, PeriodicRotor, band_gap_map
from rotorwave.plotting import plot_band_gap_map, set_style

OUTPUT = Path(__file__).parent / "figures"
F_MAX = 5000.0


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
    elements_per_cell=20,      # mesh density of one cell
)


def main() -> None:
    set_style()
    OUTPUT.mkdir(exist_ok=True)

    speeds = np.arange(0.0, 12001.0, 1000.0)

    gap_map = band_gap_map(
        ROTOR.unit_cell(), speeds, n_points=31, n_branches=8, f_max=F_MAX
    )
    print(gap_map.summary())

    ax = plot_band_gap_map(gap_map, f_max=F_MAX)
    ax.set_title("Band gaps of the 11-disk rotor versus rotating speed", loc="left")
    path = OUTPUT / "band_gap_map.png"
    ax.figure.savefig(path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
