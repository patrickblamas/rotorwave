"""Result containers for the wave analyses, band gap detection and post-processing."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = [
    "BandGap",
    "OmegaKDispersion",
    "KOmegaDispersion",
    "merge_intervals",
    "complement_intervals",
]

WhirlKind = Literal["forward", "backward", "both"]

#: Whirl indices whose *magnitude* is below this threshold are considered
#: non-precessing (mixed), and belong to both the forward and the backward
#: family.  Forward and backward waves have a whirl index near +1 and -1.
WHIRL_TOL = 1e-2

_NEPER_TO_DB = 20.0 / np.log(10.0)


# ----------------------------------------------------------------------
# Interval algebra used by the band gap detection
# ----------------------------------------------------------------------
def merge_intervals(
    intervals: Iterable[tuple[float, float]], tol: float = 0.0
) -> list[tuple[float, float]]:
    """Merge overlapping ``(start, stop)`` intervals, sorted by start."""
    ordered = sorted((min(a, b), max(a, b)) for a, b in intervals)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, stop in ordered[1:]:
        last_start, last_stop = merged[-1]
        if start <= last_stop + tol:
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def complement_intervals(
    intervals: Sequence[tuple[float, float]], lower: float, upper: float
) -> list[tuple[float, float]]:
    """Complement of ``intervals`` inside ``[lower, upper]``."""
    gaps: list[tuple[float, float]] = []
    cursor = lower
    for start, stop in merge_intervals(intervals):
        if start > cursor:
            gaps.append((cursor, min(start, upper)))
        cursor = max(cursor, stop)
        if cursor >= upper:
            break
    if cursor < upper:
        gaps.append((cursor, upper))
    return [(a, b) for a, b in gaps if b > a]


# ----------------------------------------------------------------------
# Band gaps
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BandGap:
    """A frequency band in which no wave of a given kind propagates."""

    start_hz: float
    stop_hz: float
    kind: str = "both"

    @property
    def width_hz(self) -> float:
        """Width of the gap [Hz]."""
        return self.stop_hz - self.start_hz

    @property
    def center_hz(self) -> float:
        """Mid-gap frequency [Hz]."""
        return 0.5 * (self.start_hz + self.stop_hz)

    @property
    def relative_width(self) -> float:
        """Gap width normalised by its centre frequency (gap-midgap ratio)."""
        return self.width_hz / self.center_hz if self.center_hz > 0 else np.inf

    def contains(self, frequency_hz: float) -> bool:
        """Whether a frequency falls inside the gap, edges included."""
        return self.start_hz <= frequency_hz <= self.stop_hz

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.kind:<8s} band gap: {self.start_hz:8.1f} - {self.stop_hz:8.1f} Hz "
            f"(width {self.width_hz:7.1f} Hz, {100 * self.relative_width:5.1f} %)"
        )


# ----------------------------------------------------------------------
# omega(k) results
# ----------------------------------------------------------------------
@dataclass
class OmegaKDispersion:
    """Dispersion branches obtained from the :math:`\\omega(k)` problem.

    Attributes
    ----------
    wavenumbers:
        Real wavenumbers of the analysis [rad/m], shape ``(n_k,)``.
    frequencies_hz:
        Branch frequencies [Hz], shape ``(n_branches, n_k)``.
    whirl:
        Whirl index of every solution, same shape as ``frequencies_hz``.
    cell_length:
        Unit cell length :math:`\\Delta` [m].
    spin_rpm:
        Spin speed of the analysis [rpm].
    mode_shapes:
        Optional complex mode shapes, shape ``(n_dofs, n_branches, n_k)``.
    """

    wavenumbers: np.ndarray
    frequencies_hz: np.ndarray
    whirl: np.ndarray
    cell_length: float
    spin_rpm: float = 0.0
    mode_shapes: np.ndarray | None = field(default=None, repr=False)

    # -- geometry -------------------------------------------------------
    @property
    def n_branches(self) -> int:
        """Number of dispersion branches that were tracked."""
        return self.frequencies_hz.shape[0]

    @property
    def normalized_wavenumber(self) -> np.ndarray:
        """:math:`k\\Delta/\\pi`, equal to 1 at the Bragg limit."""
        return self.wavenumbers * self.cell_length / np.pi

    # -- whirl ----------------------------------------------------------
    @property
    def branch_whirl(self) -> np.ndarray:
        """Mean whirl index of every branch, shape ``(n_branches,)``.

        A branch that changes precession direction along the Brillouin zone
        averages towards zero and is reported as ``'mixed'``, which is the
        conservative answer: mixed branches belong to both families.
        """
        with np.errstate(invalid="ignore"):
            return np.nanmean(self.whirl, axis=1)

    def branch_kinds(self, tol: float = WHIRL_TOL) -> list[str]:
        """``'forward'``, ``'backward'`` or ``'mixed'`` for each branch."""
        kinds = []
        for value in self.branch_whirl:
            if value > tol:
                kinds.append("forward")
            elif value < -tol:
                kinds.append("backward")
            else:
                kinds.append("mixed")
        return kinds

    def branch_indices(self, kind: WhirlKind, tol: float = WHIRL_TOL) -> np.ndarray:
        """Indices of the branches of a given precession kind.

        Branches that are not clearly precessing (``'mixed'``, which is what
        happens at zero spin speed where forward and backward are degenerate)
        belong to both families.
        """
        kinds = np.array(self.branch_kinds(tol=tol))
        if kind == "both":
            return np.arange(self.n_branches)
        return np.flatnonzero((kinds == kind) | (kinds == "mixed"))

    # -- derived quantities ---------------------------------------------
    def phase_velocity(self) -> np.ndarray:
        """Phase velocity :math:`c_p = \\omega/k` [m/s] (``inf`` at ``k = 0``)."""
        omega = 2.0 * np.pi * self.frequencies_hz
        with np.errstate(divide="ignore", invalid="ignore"):
            return omega / self.wavenumbers[None, :]

    def band_gaps(
        self,
        kind: WhirlKind = "both",
        f_max: float | None = None,
        f_min: float = 0.0,
        min_width_hz: float = 1.0,
        tol: float = WHIRL_TOL,
        include_open: bool = False,
    ) -> list[BandGap]:
        """Frequency bands with no propagating wave of the requested kind.

        The pass bands are the frequency ranges spanned by each branch over
        the irreducible Brillouin zone; the band gaps are their complement.

        Parameters
        ----------
        kind:
            ``'forward'``, ``'backward'`` or ``'both'``.
        f_max:
            Upper limit of the *reported* range [Hz], a display window.  Gaps
            are always detected against the full set of computed branches and
            only then clipped to this window, so a genuine gap wider than the
            window is still reported (clipped), instead of vanishing.
        f_min:
            Lower limit of the search [Hz].
        min_width_hz:
            Gaps narrower than this are discarded as numerical noise.  The
            default of 1 Hz only removes zero-width artefacts; raise it to the
            width you would consider physically meaningful (``band_gap_map``
            uses 20 Hz, so that a gap opening by a hair as the speed changes
            does not start a track of its own).
        include_open:
            Whether to report the region above the highest computed branch.
            It is discarded by default: its upper edge is where the
            computation stopped, not a physical band edge, so quoting it as a
            band gap would be misleading.  This has nothing to do with
            ``f_max``.
        """
        indices = self.branch_indices(kind, tol=tol)
        if indices.size == 0:
            return []
        branches = self.frequencies_hz[indices]
        finite = branches[np.isfinite(branches)]
        if finite.size == 0:
            return []
        pass_bands = [
            (float(np.nanmin(row)), float(np.nanmax(row)))
            for row in branches
            if np.isfinite(row).any()
        ]
        # Detect against the branches themselves ...
        top_branch = float(finite.max())
        gaps = complement_intervals(pass_bands, f_min, top_branch)
        if not include_open:
            gaps = [(a, b) for a, b in gaps if b < top_branch - 1e-9]
        # ... then clip to the requested display window.
        upper = top_branch if f_max is None else float(f_max)
        clipped = [(a, min(b, upper)) for a, b in gaps if a < upper]
        return [
            BandGap(start, stop, kind)
            for start, stop in clipped
            if stop - start >= min_width_hz
        ]

    # -- export ---------------------------------------------------------
    def summary(self, kind: WhirlKind = "both") -> str:
        """Human readable summary of the dispersion and its band gaps."""
        lines = [
            f"omega(k) dispersion at {self.spin_rpm:.0f} rpm — "
            f"{self.n_branches} branches, {self.wavenumbers.size} wavenumbers, "
            f"cell length {self.cell_length * 1e3:.2f} mm",
        ]
        for gap in self.band_gaps(kind=kind):
            lines.append("  " + str(gap))
        if len(lines) == 1:
            lines.append("  no band gap detected in the computed range")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, np.ndarray | float]:
        """Plain arrays, ready for ``numpy.savez`` or a DataFrame."""
        return {
            "wavenumbers": self.wavenumbers,
            "normalized_wavenumber": self.normalized_wavenumber,
            "frequencies_hz": self.frequencies_hz,
            "whirl": self.whirl,
            "cell_length": self.cell_length,
            "spin_rpm": self.spin_rpm,
        }


# ----------------------------------------------------------------------
# k(omega) results
# ----------------------------------------------------------------------
@dataclass
class KOmegaDispersion:
    """Complex wavenumbers obtained from the :math:`k(\\omega)` problem.

    Attributes
    ----------
    frequencies_hz:
        Frequencies of the analysis [Hz], shape ``(n_f,)``.
    wavenumbers:
        Complex wavenumbers [rad/m], shape ``(n_waves, n_f)``.
    whirl:
        Whirl index of every wave, same shape as ``wavenumbers``.
    multipliers:
        Bloch multipliers :math:`\\lambda`, same shape.
    cell_length:
        Unit cell length :math:`\\Delta` [m].
    spin_rpm:
        Spin speed of the analysis [rpm].
    """

    frequencies_hz: np.ndarray
    wavenumbers: np.ndarray
    whirl: np.ndarray
    multipliers: np.ndarray
    cell_length: float
    spin_rpm: float = 0.0

    @property
    def n_waves(self) -> int:
        """Number of waves returned per frequency."""
        return self.wavenumbers.shape[0]

    @property
    def normalized_real(self) -> np.ndarray:
        """:math:`|\\Re(k)|\\Delta/\\pi`, in ``[0, 1]`` inside the first zone."""
        return np.abs(self.wavenumbers.real) * self.cell_length / np.pi

    @property
    def normalized_imag(self) -> np.ndarray:
        """:math:`|\\Im(k)|\\Delta/\\pi`, the decay exponent per cell over
        :math:`\\pi`.

        Normalised like :attr:`normalized_real` so the two can share an axis.
        It is **not** the attenuation per cell: that is
        :math:`|\\Im k|\\Delta` nepers, without the :math:`\\pi` — see
        :attr:`attenuation_db_per_cell`, which is the quantity to quote.
        """
        return np.abs(self.wavenumbers.imag) * self.cell_length / np.pi

    @property
    def attenuation_db_per_cell(self) -> np.ndarray:
        """Amplitude decay across one cell [dB].

        :math:`20\\log_{10} e^{|\\Im k|\\Delta}`, i.e. how much a wave is
        attenuated every time it crosses one period of the rotor.
        """
        return _NEPER_TO_DB * np.abs(self.wavenumbers.imag) * self.cell_length

    @property
    def valid(self) -> np.ndarray:
        """Boolean mask of the entries that hold an actual wave.

        A solver may return fewer waves than the array can hold at some
        frequency; those entries stay NaN and must never be mistaken for a
        propagating wave.
        """
        return np.isfinite(self.wavenumbers.real) & np.isfinite(self.wavenumbers.imag)

    def propagating(self, tol_db: float = 1e-6) -> np.ndarray:
        """Boolean mask of the waves that propagate without attenuation.

        Entries that hold no wave (see :attr:`valid`) are never propagating.

        ``tol_db`` is how much attenuation per cell still counts as "none".
        The default, 1e-6 dB, is essentially exact and is meant for inspecting
        individual waves.  :meth:`band_gaps` deliberately relaxes it to
        1e-3 dB/cell, which is far below anything measurable yet comfortably
        above the eigensolver noise, so a stop band is not split in two by a
        single marginal point.
        """
        with np.errstate(invalid="ignore"):
            return self.valid & (self.attenuation_db_per_cell <= tol_db)

    def whirl_mask(self, kind: WhirlKind, tol: float = WHIRL_TOL) -> np.ndarray:
        """Boolean mask selecting waves of a given precession kind.

        Waves that are not clearly precessing (``abs(whirl) <= tol``, which
        is what a general eigensolver returns when forward and backward whirl
        are degenerate) belong to *both* families, exactly as in
        :meth:`OmegaKDispersion.branch_indices`.
        """
        if kind == "both":
            return self.valid
        with np.errstate(invalid="ignore"):
            mixed = np.abs(self.whirl) <= tol
            if kind == "forward":
                return self.valid & ((self.whirl > tol) | mixed)
            return self.valid & ((self.whirl < -tol) | mixed)

    def band_gaps(
        self,
        kind: WhirlKind = "both",
        tol_db: float = 1e-3,
        min_width_hz: float = 1.0,
    ) -> list[BandGap]:
        """Frequency bands where every wave of the requested kind is evanescent.

        Unlike the :math:`\\omega(k)` version, this detection also flags the
        frequency regions where waves are strongly attenuated but not strictly
        evanescent, which is precisely the distinction reported in the paper
        for the second band gap of the reference rotor.
        """
        propagating = self.propagating(tol_db=tol_db) & self.whirl_mask(kind)
        any_propagating = propagating.any(axis=0)
        return _mask_to_gaps(
            self.frequencies_hz, ~any_propagating, kind, min_width_hz
        )

    def attenuation_envelope(self, kind: WhirlKind = "both") -> np.ndarray:
        """Smallest attenuation among the waves of a given kind [dB/cell].

        This is the quantity that governs how much a periodic rotor filters a
        given frequency: the least attenuated wave dominates the response far
        from the excitation.
        """
        mask = self.whirl_mask(kind)
        attenuation = np.where(mask, self.attenuation_db_per_cell, np.inf)
        envelope = attenuation.min(axis=0)
        # ``inf`` means "no wave of this kind was found at that frequency",
        # which is a missing result, not an infinitely attenuated wave.
        return np.where(np.isinf(envelope), np.nan, envelope)

    def summary(self, kind: WhirlKind = "both") -> str:
        """Human readable summary of the analysis and its band gaps."""
        lines = [
            f"k(omega) dispersion at {self.spin_rpm:.0f} rpm — "
            f"{self.n_waves} waves, {self.frequencies_hz.size} frequencies",
        ]
        for gap in self.band_gaps(kind=kind):
            lines.append("  " + str(gap))
        if len(lines) == 1:
            lines.append("  no band gap detected in the computed range")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, np.ndarray | float]:
        """Plain arrays, ready for ``numpy.savez`` or a DataFrame."""
        return {
            "frequencies_hz": self.frequencies_hz,
            "wavenumbers": self.wavenumbers,
            "normalized_real": self.normalized_real,
            "normalized_imag": self.normalized_imag,
            "attenuation_db_per_cell": self.attenuation_db_per_cell,
            "whirl": self.whirl,
            "cell_length": self.cell_length,
            "spin_rpm": self.spin_rpm,
        }


def _mask_to_gaps(
    frequencies: np.ndarray, mask: np.ndarray, kind: str, min_width_hz: float
) -> list[BandGap]:
    """Convert a boolean mask over a frequency grid into a list of band gaps.

    A gap opens at the first flagged sample and closes at the first unflagged
    one, so both edges are the grid point just *after* the true edge: they are
    resolved to one frequency step and are biased high by up to one step.  The
    width is therefore accurate to within one step, in either direction.
    Refine the grid, or use the :math:`\\omega(k)` formulation, whose band
    edges are branch extrema and therefore grid independent, when the edge
    frequencies themselves are the answer.
    """
    gaps: list[BandGap] = []
    start: float | None = None
    for frequency, flagged in zip(frequencies, mask):
        if flagged and start is None:
            start = float(frequency)
        elif not flagged and start is not None:
            if frequency - start >= min_width_hz:
                gaps.append(BandGap(start, float(frequency), kind))
            start = None
    if start is not None and frequencies[-1] - start >= min_width_hz:
        gaps.append(BandGap(start, float(frequencies[-1]), kind))
    return gaps
