"""Publication-quality figures for the wave analysis of periodic rotors.

Every function takes result objects and an optional ``ax``/``fig``, returns the
matplotlib objects it created, and never calls ``show`` — so the same code
works in a script, in a notebook and in a test.

Colour conventions
------------------
Forward and backward whirl keep the blue/red convention of the rotordynamics
literature, but with the colour-blind-safe Okabe-Ito blue and vermillion and a
different line style for each family, so the two are never distinguished by
colour alone.  Branches whose precession direction is not defined — at zero
spin speed, where forward and backward are degenerate — are drawn in a neutral
grey with a third line style, so they are not mistaken for forward ones.
Wavenumber maps use a perceptually uniform sequential colormap by default;
pass ``cmap='cool'`` to match the original figures exactly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .analysis import BandGapMap
from .dispersion import WHIRL_TOL, KOmegaDispersion, OmegaKDispersion
from .frf import FRFResult

__all__ = [
    "FORWARD_COLOR",
    "BACKWARD_COLOR",
    "GAP_COLOR",
    "set_style",
    "plot_dispersion",
    "plot_dispersion_two_sided",
    "plot_komega_dispersion",
    "plot_directional_dispersion",
    "plot_wave_campbell",
    "plot_band_gap_map",
    "plot_attenuation",
]

#: Okabe-Ito blue, used for every forward whirl quantity.
FORWARD_COLOR = "#0072B2"
#: Okabe-Ito vermillion, used for every backward whirl quantity.
BACKWARD_COLOR = "#D55E00"
#: Neutral ink for band gap shading.
GAP_COLOR = "#9AA0A6"

_FORWARD_STYLE = {"color": FORWARD_COLOR, "linestyle": "-", "linewidth": 2.0}
_BACKWARD_STYLE = {"color": BACKWARD_COLOR, "linestyle": "--", "linewidth": 2.0}
#: Branches with no defined precession direction (degenerate at zero spin).
MIXED_COLOR = "#5A5A5A"
_MIXED_STYLE = {"color": MIXED_COLOR, "linestyle": ":", "linewidth": 1.8}


def set_style(font_size: float = 11.0) -> None:
    """Apply a compact, journal-friendly matplotlib style."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": font_size - 1,
            "lines.linewidth": 2.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


def _axes(ax=None, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(**kwargs)
    return ax


def _styles_for(kinds: Sequence[str]) -> list[dict]:
    """One style dict per branch kind: forward, backward or mixed."""
    styles = {
        "forward": _FORWARD_STYLE,
        "backward": _BACKWARD_STYLE,
        "mixed": _MIXED_STYLE,
    }
    return [styles.get(kind, _MIXED_STYLE) for kind in kinds]


# ----------------------------------------------------------------------
# omega(k) dispersion
# ----------------------------------------------------------------------
def plot_dispersion(
    dispersion: OmegaKDispersion,
    ax=None,
    f_max: float | None = None,
    show_gaps: bool = True,
    label_kinds: bool = True,
):
    """Dispersion diagram: frequency on the abscissa, :math:`k\\Delta/\\pi` on the ordinate.

    This is the layout of Fig. 4 of Lamas & Nicoletti (2024): the pass bands
    appear as curves climbing from 0 to 1 and the band gaps as the empty
    vertical strips between them.
    """
    ax = _axes(ax)
    normalized = dispersion.normalized_wavenumber
    kinds = dispersion.branch_kinds()
    seen: set[str] = set()

    if show_gaps:
        for gap in dispersion.band_gaps(kind="both", f_max=f_max):
            ax.axvspan(gap.start_hz, gap.stop_hz, color=GAP_COLOR, alpha=0.18, lw=0)

    # Forward first, backward (dashed) on top: at zero spin speed the two
    # families are degenerate and would otherwise hide each other.
    for wanted in ("forward", "mixed", "backward"):
        for row, kind, style in zip(
            dispersion.frequencies_hz, kinds, _styles_for(kinds)
        ):
            if kind != wanted:
                continue
            label = None
            if label_kinds and kind not in seen:
                label = f"{kind} whirl"
                seen.add(kind)
            ax.plot(row, normalized, label=label, **style)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$\Re(k\Delta/\pi)$")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    if f_max is not None:
        ax.set_xlim(0.0, f_max)
    if label_kinds and seen:
        ax.legend(loc="best")
    return ax


def plot_dispersion_two_sided(
    dispersion: OmegaKDispersion, ax=None, f_max: float | None = None
):
    """Dispersion with backward branches mirrored onto negative frequencies.

    Putting the two precession directions on opposite halves of the frequency
    axis is what makes the forward and backward band gaps directly comparable.
    """
    ax = _axes(ax)
    normalized = dispersion.normalized_wavenumber
    kinds = dispersion.branch_kinds()
    seen: set[str] = set()

    for row, kind, style in zip(dispersion.frequencies_hz, kinds, _styles_for(kinds)):
        sign = -1.0 if kind == "backward" else 1.0
        label = None
        if kind not in seen:
            label = f"{kind} whirl"
            seen.add(kind)
        ax.plot(sign * row, normalized, label=label, **style)

    ax.axvline(0.0, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$\Re(k\Delta/\pi)$")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    if f_max is not None:
        ax.set_xlim(-f_max, f_max)
    ax.legend(loc="upper center", ncols=2)
    return ax


# ----------------------------------------------------------------------
# k(omega) dispersion, with and without the receptance
# ----------------------------------------------------------------------
def plot_komega_dispersion(
    dispersion: KOmegaDispersion,
    frf: FRFResult | None = None,
    fig=None,
    f_max: float | None = None,
):
    """Real part, imaginary part and receptance, stacked on a shared axis.

    Reproduces the layout of Fig. 5 of the reference: propagation on top,
    attenuation in the middle and the response of the finite rotor at the
    bottom, so a band gap can be read as "``Re`` saturated at 1, ``Im`` lifting
    off zero, and the FRF collapsing" in a single glance.
    """
    import matplotlib.pyplot as plt

    n_rows = 3 if frf is not None else 2
    if fig is None:
        fig, axes = plt.subplots(
            n_rows, 1, sharex=True, figsize=(7.0, 2.1 * n_rows), constrained_layout=True
        )
    else:
        axes = fig.subplots(n_rows, 1, sharex=True)

    # Same threshold as ``KOmegaDispersion.whirl_mask``: a wave is drawn as
    # forward only if its whirl index is clearly positive, and slots that
    # hold no wave (NaN) are drawn as neither.
    forward = dispersion.whirl > WHIRL_TOL
    backward = dispersion.whirl < -WHIRL_TOL
    for row, (values, label) in enumerate(
        ((dispersion.normalized_real, r"$\Re(k\Delta/\pi)$"),
         (dispersion.normalized_imag, r"$\Im(k\Delta/\pi)$"))
    ):
        ax = axes[row]
        for wave in range(dispersion.n_waves):
            for mask, style in (
                (forward[wave], _FORWARD_STYLE),
                (backward[wave], _BACKWARD_STYLE),
            ):
                ax.plot(
                    np.where(mask, dispersion.frequencies_hz, np.nan),
                    np.where(mask, values[wave], np.nan),
                    **style,
                )
        ax.set_ylabel(label)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].plot([], [], label="forward whirl", **_FORWARD_STYLE)
    axes[0].plot([], [], label="backward whirl", **_BACKWARD_STYLE)
    axes[0].legend(loc="lower right", ncols=2)

    if frf is not None:
        ax = axes[-1]
        ax.plot(frf.frequencies_hz, np.log10(np.abs(frf.direct)), color="0.15", lw=1.2)
        ax.set_ylabel(r"$\log_{10}|H|$")
    axes[-1].set_xlabel("Frequency (Hz)")
    if f_max is not None:
        axes[-1].set_xlim(0.0, f_max)
    return fig, axes


def plot_directional_dispersion(
    dispersion: KOmegaDispersion,
    frf: FRFResult | None = None,
    fig=None,
    f_max: float | None = None,
):
    """Two-sided version of :func:`plot_komega_dispersion` (Fig. 6 layout).

    Forward waves are drawn at positive frequencies and backward waves at
    negative ones, next to the directional FRFs of the finite rotor — the
    representation that makes the two band gaps of a rotating machine
    immediately readable.
    """
    import matplotlib.pyplot as plt

    n_rows = 3 if frf is not None else 2
    if fig is None:
        fig, axes = plt.subplots(
            n_rows, 1, sharex=True, figsize=(7.5, 2.1 * n_rows), constrained_layout=True
        )
    else:
        axes = fig.subplots(n_rows, 1, sharex=True)

    # Same threshold as ``KOmegaDispersion.whirl_mask``: a wave is drawn as
    # forward only if its whirl index is clearly positive, and slots that
    # hold no wave (NaN) are drawn as neither.
    forward = dispersion.whirl > WHIRL_TOL
    backward = dispersion.whirl < -WHIRL_TOL
    frequencies = dispersion.frequencies_hz
    for row, (values, label) in enumerate(
        ((dispersion.normalized_real, r"$\Re(k\Delta/\pi)$"),
         (dispersion.normalized_imag, r"$\Im(k\Delta/\pi)$"))
    ):
        ax = axes[row]
        for wave in range(dispersion.n_waves):
            for mask, axis, style in (
                (forward[wave], frequencies, _FORWARD_STYLE),
                (backward[wave], -frequencies, _BACKWARD_STYLE),
            ):
                ax.plot(
                    np.where(mask, axis, np.nan),
                    np.where(mask, values[wave], np.nan),
                    **style,
                )
        ax.axvline(0.0, color="0.4", lw=0.8, ls=":")
        ax.set_ylabel(label)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].plot([], [], label="forward whirl", **_FORWARD_STYLE)
    axes[0].plot([], [], label="backward whirl", **_BACKWARD_STYLE)
    axes[0].legend(loc="lower center", ncols=2)

    if frf is not None:
        ax = axes[-1]
        ax.plot(
            frf.frequencies_hz,
            np.log10(np.abs(frf.forward)),
            color=FORWARD_COLOR,
            lw=1.2,
            label="dFRF forward",
        )
        ax.plot(
            -frf.frequencies_hz,
            np.log10(np.abs(frf.backward)),
            color=BACKWARD_COLOR,
            lw=1.2,
            ls="--",
            label="dFRF backward",
        )
        ax.axvline(0.0, color="0.4", lw=0.8, ls=":")
        ax.set_ylabel(r"$\log_{10}|H|$")
        ax.legend(loc="lower center", ncols=2)

    axes[-1].set_xlabel("Frequency (Hz)")
    if f_max is not None:
        axes[-1].set_xlim(-f_max, f_max)
    return fig, axes


# ----------------------------------------------------------------------
# Wave-Campbell diagram
# ----------------------------------------------------------------------
def plot_wave_campbell(
    dispersions: Iterable[OmegaKDispersion],
    ax=None,
    f_max: float | None = None,
    cmap: str = "viridis",
    marker_size: float = 14.0,
):
    """Wave-Campbell diagram (Fig. 7 layout).

    Natural frequencies against rotating speed, forward whirl on the positive
    half and backward whirl on the negative half, coloured by the normalised
    wavenumber.  The blank horizontal strips are the band gaps, and their drift
    with speed is the gyroscopic effect made visible.
    """
    ax = _axes(ax, figsize=(7.0, 5.0))
    speeds, frequencies, colors = [], [], []

    for dispersion in dispersions:
        normalized = dispersion.normalized_wavenumber
        kinds = dispersion.branch_kinds()
        for row, kind in zip(dispersion.frequencies_hz, kinds):
            sign = -1.0 if kind == "backward" else 1.0
            speeds.append(np.full(row.size, dispersion.spin_rpm))
            frequencies.append(sign * row)
            colors.append(normalized)

    scatter = ax.scatter(
        np.concatenate(speeds),
        np.concatenate(frequencies),
        c=np.concatenate(colors),
        cmap=cmap,
        s=marker_size,
        marker="s",
        linewidths=0.0,
        vmin=0.0,
        vmax=1.0,
    )
    ax.axhline(0.0, color="0.3", lw=0.8, ls=":")
    ax.set_xlabel("Rotating speed (rpm)")
    ax.set_ylabel("Frequency (Hz)")
    if f_max is not None:
        ax.set_ylim(-f_max, f_max)
    bar = ax.figure.colorbar(scatter, ax=ax, pad=0.02)
    bar.set_label(r"$\Re(k\Delta/\pi)$")
    ax.text(
        0.01, 0.98, "forward whirl", transform=ax.transAxes, va="top", ha="left",
        color=FORWARD_COLOR, fontsize=9,
    )
    ax.text(
        0.01, 0.02, "backward whirl", transform=ax.transAxes, va="bottom", ha="left",
        color=BACKWARD_COLOR, fontsize=9,
    )
    return ax, scatter


# ----------------------------------------------------------------------
# New analyses
# ----------------------------------------------------------------------
def plot_band_gap_map(gap_map: BandGapMap, ax=None, f_max: float | None = None):
    """Band gap edges as a function of the rotating speed.

    The quantitative counterpart of the wave-Campbell diagram: each shaded band
    is one stop band, so its shift and the change of its width with speed can
    be read off the axes instead of estimated from a colour map.
    """
    ax = _axes(ax, figsize=(7.0, 4.5))
    for kind, color, hatch in (
        ("forward", FORWARD_COLOR, None),
        ("backward", BACKWARD_COLOR, "//"),
    ):
        for index, track in enumerate(gap_map.tracks(kind)):
            ax.fill_between(
                track["speeds_rpm"],
                track["start_hz"],
                track["stop_hz"],
                color=color,
                alpha=0.25,
                hatch=hatch,
                edgecolor=color,
                linewidth=0.0,
                label=f"{kind} whirl" if index == 0 else None,
            )
            ax.plot(track["speeds_rpm"], track["start_hz"], color=color, lw=1.4)
            ax.plot(track["speeds_rpm"], track["stop_hz"], color=color, lw=1.4)

    ax.set_xlabel("Rotating speed (rpm)")
    ax.set_ylabel("Frequency (Hz)")
    if f_max is not None:
        ax.set_ylim(0.0, f_max)
    ax.legend(loc="upper left", ncols=2)
    return ax


def plot_attenuation(
    report: dict[str, np.ndarray],
    ax=None,
    f_max: float | None = None,
    total: bool = False,
):
    """Attenuation of the least attenuated wave, in dB per cell.

    Inside a band gap every wave is evanescent, so the curve lifts off zero;
    the height of the plateau says how many dB one period of the rotor removes
    from a disturbance at that frequency.  Set ``total`` to plot the
    attenuation over the whole rotor instead, when the report was built with
    ``n_cells``.
    """
    ax = _axes(ax, figsize=(7.0, 4.0))
    frequencies = report["frequencies_hz"]
    per_cell = not (total and "forward_db_total" in report)
    suffix = "_db_per_cell" if per_cell else "_db_total"
    ax.plot(
        frequencies, report["forward" + suffix], label="forward whirl", **_FORWARD_STYLE
    )
    ax.plot(
        frequencies,
        report["backward" + suffix],
        label="backward whirl",
        **_BACKWARD_STYLE,
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Attenuation (dB per cell)" if per_cell else "Attenuation (dB)")
    if f_max is not None:
        ax.set_xlim(0.0, f_max)
    ax.legend(loc="upper left")
    return ax
