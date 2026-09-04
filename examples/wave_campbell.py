"""Wave-Campbell diagram of the 11-disk rotor.

Natural frequencies against rotating speed, forward whirl above the axis and
backward whirl below it, coloured by the normalised wavenumber.  The blank
strips are the band gaps: they drift apart with speed while keeping almost the
same width.

"""

from __future__ import annotations

import _path_setup  # noqa: F401  (lets the file run without installing)

from pathlib import Path

import numpy as np

from rotorwave import STEEL, OmegaKSolver, PeriodicRotor
from rotorwave.plotting import plot_wave_campbell, set_style

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

    solver = OmegaKSolver(ROTOR.unit_cell())
    speeds = np.arange(0.0, 6001.0, 60.0)

    dispersions = solver.wave_campbell(speeds, n_points=31, n_branches=8)
    ax, _ = plot_wave_campbell(dispersions, f_max=F_MAX)
    ax.figure.savefig(OUTPUT / "wave_campbell.png")

    # How the first stop band of each family moves between the two ends of
    # the sweep.  Both lists may be empty for a rotor without a band gap in
    # the analysed range, hence the check.
    first, last = dispersions[0], dispersions[-1]
    for kind in ("forward", "backward"):
        gaps_rest = first.band_gaps(kind, f_max=F_MAX)
        gaps_spin = last.band_gaps(kind, f_max=F_MAX)
        if not gaps_rest or not gaps_spin:
            print(f"{kind:<8s}: no band gap below {F_MAX:.0f} Hz")
            continue
        gap_rest, gap_spin = gaps_rest[0], gaps_spin[0]
        print(
            f"{kind:<8s}: {gap_rest.start_hz:7.0f}-{gap_rest.stop_hz:7.0f} Hz at rest -> "
            f"{gap_spin.start_hz:7.0f}-{gap_spin.stop_hz:7.0f} Hz at "
            f"{last.spin_rpm:.0f} rpm "
            f"(width {gap_rest.width_hz:.0f} -> {gap_spin.width_hz:.0f} Hz)"
        )
    print(f"saved {OUTPUT / 'wave_campbell.png'}")


if __name__ == "__main__":
    main()
