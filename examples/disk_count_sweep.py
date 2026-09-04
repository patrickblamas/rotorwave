"""Modal and wave analysis as a function of the number of disks.

Sweeps the number of working elements mounted on the same shaft and produces
two maps, one per figure:

* ``disk_sweep_frf.png`` — natural frequencies of the finite rotor (dots) over
  the amplitude of its receptance (colours).  The dark horizontal bands are the
  frequency ranges the rotor does not respond in.
* ``disk_sweep_wavenumber.png`` — the same natural frequencies over the
  normalised wavenumber of the unit cell (colours).  Blank areas are the stop
  bands: no wave propagates there, which is why the response collapses in the
  first figure.

Increasing the number of disks shortens the unit cell, which pushes the Bragg
limit up and opens the band gaps.

"""

from __future__ import annotations

import _path_setup  # noqa: F401  (lets the file run without installing)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rotorwave import STEEL, KOmegaSolver, PeriodicRotor, ReceptanceSolver
from rotorwave.plotting import set_style

OUTPUT = Path(__file__).parent / "figures"

# ----------------------------------------------------------------------
# The swept variable: how many disks are mounted on the shaft
# ----------------------------------------------------------------------
DISK_COUNTS = range(1, 16)     # e.g. range(1, 21) or (3, 5, 11, 15)


# ----------------------------------------------------------------------
# Rotor geometry and material — change these to analyse a different rotor.
# The number of disks is not set here: it comes from DISK_COUNTS above.
# ----------------------------------------------------------------------
def build_rotor(n_disks: int) -> PeriodicRotor:
    """The rotor of the sweep, with ``n_disks`` disks on the same shaft."""
    return PeriodicRotor(
        shaft_length=1.5,          # total shaft length [m]
        shaft_diameter=100e-3,     # shaft outer diameter [m]
        n_cells=n_disks,           # the swept variable
        disk_diameter=380e-3,      # disk outer diameter [m]
        disk_thickness=22e-3,      # disk thickness [m]
        material=STEEL,            # Material(young_modulus, density, poisson_ratio)
        elements_per_cell=40,      # mesh density of one cell, for the wave analysis
    )


# ----------------------------------------------------------------------
# Analysis settings
# ----------------------------------------------------------------------
SPIN_RPM = 1000.0              # rotating speed of the analysis [rpm]
F_MAX = 7000.0                 # upper limit of the maps [Hz]
FREQUENCY_STEP = 10.0          # frequency resolution of the maps [Hz]
ELEMENTS_PER_CELL_FRF = 10     # mesh of the finite rotor, per cell
COLUMN_WIDTH = 0.84            # width of each column of the maps, in disks
# ----------------------------------------------------------------------


def run_sweep():
    """Compute both maps and the natural frequencies for every disk count."""
    counts = list(DISK_COUNTS)
    frequencies = np.arange(FREQUENCY_STEP, F_MAX + FREQUENCY_STEP, FREQUENCY_STEP)

    receptance_map = np.full((frequencies.size, len(counts)), np.nan)
    wavenumber_map = np.full_like(receptance_map, np.nan)
    natural_frequencies: list[np.ndarray] = []

    for column, n_disks in enumerate(counts):
        rotor = build_rotor(n_disks)

        # Finite rotor: receptance and natural frequencies.
        model = rotor.full_rotor(elements_per_cell=ELEMENTS_PER_CELL_FRF)
        receptance = ReceptanceSolver(model).compute(frequencies, spin_rpm=SPIN_RPM)
        receptance_map[:, column] = np.log(np.abs(receptance.direct))
        natural_frequencies.append(
            model.natural_frequencies_below(F_MAX, spin_rpm=SPIN_RPM)
        )

        # Infinite periodic medium: wavenumber of the propagating waves.
        waves = KOmegaSolver(rotor.unit_cell()).solve(frequencies, spin_rpm=SPIN_RPM)
        propagating = waves.propagating(tol_db=1e-3)
        count = propagating.sum(axis=0)
        total = np.where(propagating, waves.normalized_real, 0.0).sum(axis=0)
        wavenumber_map[:, column] = np.where(
            count > 0, total / np.maximum(count, 1), np.nan
        )

        # Report the *first* stop band, not the widest one: the sweep stops at
        # F_MAX, so a higher gap may be reported open-ended and would look
        # wider than it is.  The first gap is fully inside the swept range and
        # is the one the paper follows as disks are added.
        gaps = waves.band_gaps("both", min_width_hz=5 * FREQUENCY_STEP)
        first = gaps[0] if gaps else None
        print(
            f"{n_disks:3d} disks | cell {rotor.cell_length * 1e3:6.1f} mm | "
            f"{natural_frequencies[-1].size:3d} modes below {F_MAX:.0f} Hz | "
            + (
                f"first gap {first.start_hz:6.0f} - {first.stop_hz:6.0f} Hz"
                if first
                else "no band gap"
            )
        )

    return counts, frequencies, receptance_map, wavenumber_map, natural_frequencies


def draw_map(values, counts, frequencies, cmap, label, vmin=None, vmax=None):
    """One column of colour per disk count, on a frequency axis."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.grid(False)

    edges = np.concatenate(
        [frequencies - 0.5 * FREQUENCY_STEP, [frequencies[-1] + 0.5 * FREQUENCY_STEP]]
    )
    if vmin is None or vmax is None:
        finite = values[np.isfinite(values)]
        vmin, vmax = np.percentile(finite, [5.0, 99.0])

    mesh = None
    half = 0.5 * COLUMN_WIDTH
    for column, n_disks in enumerate(counts):
        mesh = ax.pcolormesh(
            [n_disks - half, n_disks + half],
            edges,
            values[:, column : column + 1],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
            rasterized=True,
        )

    ax.set_xlabel("Number of disks")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xticks(counts[:: max(1, len(counts) // 15)])
    ax.set_xlim(counts[0] - 0.6, counts[-1] + 0.6)
    ax.set_ylim(0.0, F_MAX)
    bar = fig.colorbar(mesh, ax=ax, pad=0.02)
    bar.set_label(label)
    bar.outline.set_visible(False)
    return fig, ax


def scatter_natural_frequencies(ax, counts, natural_frequencies):
    """Overlay the natural frequencies of the finite rotor."""
    x = np.concatenate(
        [np.full(f.size, n) for n, f in zip(counts, natural_frequencies)]
    )
    y = np.concatenate(natural_frequencies)
    ax.scatter(x, y, s=7.0, c="black", linewidths=0.4, edgecolors="white", zorder=3)


def main() -> None:
    set_style()
    OUTPUT.mkdir(exist_ok=True)
    print(build_rotor(max(DISK_COUNTS)).summary())
    print(f"sweep at {SPIN_RPM:.0f} rpm\n")

    counts, frequencies, receptance, wavenumber, naturals = run_sweep()

    # ---- Figure 1: response of the finite rotor ----------------------
    fig, ax = draw_map(receptance, counts, frequencies, "magma", r"$\ln|H(\omega)|$")
    scatter_natural_frequencies(ax, counts, naturals)
    ax.set_title(
        "Natural frequencies (dots) over the receptance of the finite rotor",
        loc="left", fontsize=10,
    )
    path = OUTPUT / "disk_sweep_frf.png"
    fig.savefig(path)
    print(f"\nsaved {path}")

    # ---- Figure 2: wavenumber of the unit cell -----------------------
    fig, ax = draw_map(
        wavenumber, counts, frequencies, "viridis",
        r"$\Re(k\Delta/\pi)$", vmin=0.0, vmax=1.0,
    )
    scatter_natural_frequencies(ax, counts, naturals)
    ax.set_title(
        "Natural frequencies (dots) over the wavenumber of the unit cell\n"
        "blank areas: stop bands, where no wave propagates",
        loc="left", fontsize=10,
    )
    path = OUTPUT / "disk_sweep_wavenumber.png"
    fig.savefig(path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
