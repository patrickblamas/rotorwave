"""Dispersion diagrams from the omega(k) problem, with the rotor receptance.

Reproduces the dispersion of the 11-disk reference rotor at rest and at
6000 rpm, each one above the frequency response function of the finite rotor at
the same speed, and prints the band gaps that the diagrams show.  The empty
strips of the dispersion diagram are the frequency ranges where the receptance
has no resonance and collapses by several orders of magnitude.

One figure is written per rotating speed.

"""

from __future__ import annotations

import _path_setup  # noqa: F401  (lets the file run without installing)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rotorwave import STEEL, OmegaKSolver, PeriodicRotor, ReceptanceSolver
from rotorwave.plotting import GAP_COLOR, plot_dispersion, set_style

OUTPUT = Path(__file__).parent / "figures"
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

    solver = OmegaKSolver(ROTOR.unit_cell())
    receptance_solver = ReceptanceSolver(ROTOR.full_rotor(elements_per_cell=20))
    frequencies = np.arange(5.0, F_MAX + 1.0, 5.0)

    for spin_rpm, title in zip((0.0, 6000.0), ("0 rpm", "6000 rpm")):
        dispersion = solver.solve(spin_rpm=spin_rpm, n_points=61, n_branches=8)
        gaps = dispersion.band_gaps("both", f_max=F_MAX)
        receptance = receptance_solver.compute(frequencies, spin_rpm=spin_rpm)

        fig, (top, bottom) = plt.subplots(
            2,
            1,
            figsize=(6.5, 5.0),
            sharex=True,
            constrained_layout=True,
        )

        # Top panel: dispersion. The legend is only needed where the two
        # precession directions are distinguishable.
        plot_dispersion(dispersion, ax=top, f_max=F_MAX, label_kinds=spin_rpm > 0.0)
        top.set_title(title, loc="left")
        top.set_xlabel("")

        # Bottom panel: receptance of the finite rotor at the same speed.
        for gap in gaps:
            bottom.axvspan(gap.start_hz, gap.stop_hz, color=GAP_COLOR, alpha=0.18, lw=0)
        bottom.plot(
            frequencies, np.log10(np.abs(receptance.direct)), color="0.15", lw=1.0
        )
        bottom.set_xlabel("Frequency (Hz)")
        bottom.set_ylabel(r"$\log_{10}|H|$")
        bottom.set_ylim(-24.0, -6.0)
        bottom.set_yticks([-24.0, -15.0, -6.0])
        bottom.set_xlim(0.0, F_MAX)

        path = OUTPUT / f"dispersion_omega_k_{spin_rpm:.0f}rpm.png"
        fig.savefig(path)

        print(dispersion.summary("forward"))
        print(dispersion.summary("backward"))
        magnitude = np.log10(np.abs(receptance.direct))
        for gap in gaps:
            inside = (frequencies > gap.start_hz) & (frequencies < gap.stop_hz)
            # Compare with the pass band immediately below the gap, not with
            # everything below it: the range below also contains the previous
            # gaps, whose low response would flatter the result.
            previous = [g.stop_hz for g in gaps if g.stop_hz <= gap.start_hz]
            lower_edge = max(previous) if previous else 0.0
            pass_band = (frequencies > lower_edge) & (frequencies < gap.start_hz)
            if not pass_band.any():
                continue
            drop = 20.0 * (magnitude[pass_band].mean() - magnitude[inside].mean())
            print(
                f"  receptance inside the {gap.start_hz:.0f}-{gap.stop_hz:.0f} Hz "
                f"gap: {drop:.0f} dB below the adjacent pass band"
            )
        print(f"saved {path}\n")


if __name__ == "__main__":
    main()
