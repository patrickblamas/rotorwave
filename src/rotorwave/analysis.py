"""Higher level analyses built on top of the wave solvers.

These go beyond reproducing the original figures: they turn the dispersion
diagrams into the quantities an engineer actually asks for — where the stop
bands are, how they move with the rotating speed, how fast energy travels in
the pass bands and how strongly the rotor filters a given frequency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dispersion import BandGap, KOmegaDispersion, OmegaKDispersion, WhirlKind
from .model import UnitCell
from .waves import KOmegaSolver, OmegaKSolver

__all__ = [
    "BandGapMap",
    "band_gap_map",
    "attenuation_report",
    "mesh_convergence",
]


# ----------------------------------------------------------------------
# Band gap map: how the stop bands move with the rotating speed
# ----------------------------------------------------------------------
@dataclass
class BandGapMap:
    """Band gaps of a periodic rotor as a function of the spin speed.

    Attributes
    ----------
    spin_speeds_rpm:
        Spin speeds of the sweep [rpm].
    gaps:
        ``{'forward': [[BandGap, ...], ...], 'backward': [...]}`` — one list of
        band gaps per spin speed and per precession kind.
    dispersions:
        The underlying :class:`~rotorwave.dispersion.OmegaKDispersion` objects.
    """

    spin_speeds_rpm: np.ndarray
    gaps: dict[str, list[list[BandGap]]]
    dispersions: list[OmegaKDispersion] = field(default_factory=list, repr=False)

    def tracks(self, kind: WhirlKind = "forward", tol_hz: float | None = None):
        """Group the band gaps into continuous tracks across the speed sweep.

        A track follows one stop band as the speed changes.  It is extended
        only by a gap found at the *immediately following* speed of the sweep:
        a stop band that closes at some speed ends its track there, and a stop
        band that reopens later starts a new one.  Without that rule
        ``fill_between`` would draw a straight line across the speeds where the
        gap does not exist, showing a band gap that was never computed.

        Parameters
        ----------
        kind:
            ``'forward'`` or ``'backward'``.
        tol_hz:
            How far the centre of a gap may move between two consecutive
            speeds and still count as the same stop band.  Defaults to half
            the gap width plus 50 Hz.

        Returns
        -------
        list of dict
            Each track has ``speeds_rpm``, ``start_hz``, ``stop_hz``,
            ``center_hz`` and ``width_hz`` arrays, ready for ``fill_between``.
        """
        if kind not in self.gaps:
            raise KeyError(
                f"no band gaps stored for kind={kind!r}; "
                f"available: {sorted(self.gaps)}"
            )
        per_speed = self.gaps[kind]
        tracks: list[dict] = []
        for step, (speed, gaps) in enumerate(zip(self.spin_speeds_rpm, per_speed)):
            claimed: set[int] = set()
            for gap in sorted(gaps, key=lambda g: g.center_hz):
                tolerance = tol_hz if tol_hz is not None else 0.5 * gap.width_hz + 50.0
                # Only tracks that reached the previous speed of the sweep can
                # be continued, and among those the nearest one wins.
                best, best_distance = None, np.inf
                for index, track in enumerate(tracks):
                    if index in claimed or track["_last_step"] != step - 1:
                        continue
                    distance = abs(track["center_hz"][-1] - gap.center_hz)
                    if distance <= tolerance and distance < best_distance:
                        best, best_distance = index, distance
                if best is None:
                    tracks.append(
                        {
                            "speeds_rpm": [], "start_hz": [], "stop_hz": [],
                            "center_hz": [], "width_hz": [], "_last_step": step,
                        }
                    )
                    best = len(tracks) - 1
                track = tracks[best]
                claimed.add(best)
                track["_last_step"] = step
                track["speeds_rpm"].append(float(speed))
                track["start_hz"].append(gap.start_hz)
                track["stop_hz"].append(gap.stop_hz)
                track["center_hz"].append(gap.center_hz)
                track["width_hz"].append(gap.width_hz)
        return [
            {
                key: np.array(value)
                for key, value in track.items()
                if not key.startswith("_")
            }
            for track in tracks
        ]

    def summary(self) -> str:
        """One line per stop band: where it starts and how it drifts."""
        lines = ["Band gap map"]
        for kind in ("forward", "backward"):
            lines.append(f"  {kind}:")
            tracks = self.tracks(kind)
            if not tracks:
                lines.append("    no band gap over the swept speeds")
            for track in tracks:
                first, last = track["speeds_rpm"][0], track["speeds_rpm"][-1]
                drift = track["center_hz"][-1] - track["center_hz"][0]
                widening = track["width_hz"][-1] - track["width_hz"][0]
                lines.append(
                    f"    {track['center_hz'][0]:7.0f} Hz gap "
                    f"({track['start_hz'][0]:.0f}-{track['stop_hz'][0]:.0f} Hz "
                    f"at {first:.0f} rpm): "
                    f"centre shifts {drift:+.0f} Hz, width changes {widening:+.0f} Hz "
                    f"up to {last:.0f} rpm"
                )
        return "\n".join(lines)


def band_gap_map(
    cell: UnitCell,
    spin_speeds_rpm: np.ndarray,
    n_points: int = 41,
    n_branches: int = 8,
    f_max: float | None = None,
    min_width_hz: float = 20.0,
) -> BandGapMap:
    """Track the forward and backward band gaps over a range of spin speeds.

    This is the quantitative counterpart of the wave-Campbell diagram: instead
    of a colour map, it returns the *edges* of every stop band, so the shift
    and the widening caused by the gyroscopic effect can be measured.

    Parameters
    ----------
    cell:
        The unit cell of the periodic rotor.
    spin_speeds_rpm:
        Speeds of the sweep [rpm].  They should be ordered and evenly enough
        spaced for a gap to be recognisable from one speed to the next, since
        :meth:`BandGapMap.tracks` links gaps between *consecutive* speeds.
    n_points:
        Wavenumbers per dispersion, over the irreducible Brillouin zone.
    n_branches:
        Branches computed at each speed.  This sets how high in frequency the
        map reaches.
    f_max:
        Upper limit of the reported range [Hz]; ``None`` uses the full span of
        the computed branches.
    min_width_hz:
        Gaps narrower than this are ignored, so that a stop band barely
        opening does not start a track of its own.
    """
    solver = OmegaKSolver(cell)
    speeds = np.atleast_1d(np.asarray(spin_speeds_rpm, dtype=float))
    gaps: dict[str, list[list[BandGap]]] = {"forward": [], "backward": []}
    dispersions: list[OmegaKDispersion] = []

    for speed in speeds:
        dispersion = solver.solve(
            spin_rpm=float(speed), n_points=n_points, n_branches=n_branches
        )
        dispersions.append(dispersion)
        for kind in ("forward", "backward"):
            gaps[kind].append(
                dispersion.band_gaps(kind=kind, f_max=f_max, min_width_hz=min_width_hz)
            )

    return BandGapMap(spin_speeds_rpm=speeds, gaps=gaps, dispersions=dispersions)


# ----------------------------------------------------------------------
# Attenuation
# ----------------------------------------------------------------------
def attenuation_report(
    cell: UnitCell,
    frequencies_hz: np.ndarray,
    spin_rpm: float = 0.0,
    n_cells: int | None = None,
) -> dict[str, np.ndarray]:
    """Attenuation of the least attenuated wave, per cell and over the rotor.

    Parameters
    ----------
    cell:
        The unit cell.
    frequencies_hz:
        Frequency grid [Hz].
    spin_rpm:
        Spin speed [rpm].
    n_cells:
        If given, the attenuation is also reported for the whole rotor, i.e.
        multiplied by the number of cells the wave has to cross.
    """
    dispersion: KOmegaDispersion = KOmegaSolver(cell).solve(
        frequencies_hz, spin_rpm=spin_rpm
    )
    report = {
        "frequencies_hz": dispersion.frequencies_hz,
        "forward_db_per_cell": dispersion.attenuation_envelope("forward"),
        "backward_db_per_cell": dispersion.attenuation_envelope("backward"),
        "any_db_per_cell": dispersion.attenuation_envelope("both"),
    }
    if n_cells is not None:
        for key in ("forward", "backward", "any"):
            report[f"{key}_db_total"] = report[f"{key}_db_per_cell"] * n_cells
    return report


# ----------------------------------------------------------------------
# Verification helpers
# ----------------------------------------------------------------------
def mesh_convergence(
    build_cell,
    elements_per_cell: list[int],
    wavenumber_fraction: float = 0.5,
    spin_rpm: float = 0.0,
    n_branches: int = 4,
) -> dict[str, np.ndarray]:
    """Convergence of the first branches with the number of elements per cell.

    Parameters
    ----------
    build_cell:
        Callable ``n_elements -> UnitCell``.
    elements_per_cell:
        Mesh densities to test.
    wavenumber_fraction:
        Position inside the Brillouin zone where the frequencies are compared,
        as a fraction of :math:`\\pi/\\Delta`.

    Notes
    -----
    The reference used for the relative error is the *finest* mesh of the
    list, whatever order the list is given in.
    """
    meshes = [int(n) for n in elements_per_cell]
    if not meshes:
        raise ValueError("elements_per_cell must contain at least one mesh density")
    frequencies = []
    for n_elements in meshes:
        cell = build_cell(n_elements)
        k = wavenumber_fraction * np.pi / cell.cell_length
        dispersion = OmegaKSolver(cell).solve(
            wavenumbers=np.array([k]), spin_rpm=spin_rpm, n_branches=n_branches
        )
        frequencies.append(dispersion.frequencies_hz[:, 0])
    table = np.array(frequencies)
    reference = table[int(np.argmax(meshes))]
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.abs(table - reference) / reference
    return {
        "elements_per_cell": np.asarray(meshes),
        "frequencies_hz": table,
        "relative_error": error,
    }
