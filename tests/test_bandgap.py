"""Band gap detection, and the physical claims of Lamas & Nicoletti (2024)."""

from __future__ import annotations

import numpy as np
import pytest

from rotorwave import (
    KOmegaSolver,
    OmegaKSolver,
    band_gap_map,
    mesh_convergence,
    reference_rotor,
)
from rotorwave.dispersion import complement_intervals, merge_intervals


@pytest.fixture(scope="module")
def cell():
    return reference_rotor(n_disks=11, elements_per_cell=20).unit_cell()


# ----------------------------------------------------------------------
# Interval algebra
# ----------------------------------------------------------------------
def test_merge_and_complement_intervals() -> None:
    merged = merge_intervals([(0.0, 2.0), (1.0, 3.0), (5.0, 6.0)])
    assert merged == [(0.0, 3.0), (5.0, 6.0)]
    assert complement_intervals(merged, 0.0, 6.0) == [(3.0, 5.0)]


# ----------------------------------------------------------------------
# The reference case study
# ----------------------------------------------------------------------
def test_first_band_gap_of_the_reference_rotor(cell) -> None:
    """The paper reports a band gap between roughly 2000 Hz and 3500 Hz."""
    dispersion = OmegaKSolver(cell).solve(spin_rpm=0.0, n_points=41, n_branches=6)
    gaps = dispersion.band_gaps("both")
    assert gaps, "no band gap detected"
    first = gaps[0]
    assert first.start_hz == pytest.approx(2037.0, abs=25.0)
    assert first.stop_hz == pytest.approx(3579.0, abs=25.0)
    assert 0.5 < first.relative_width < 0.6


def test_band_gap_shifts_but_keeps_its_width_with_speed(cell) -> None:
    """Forward gap up, backward gap down, width essentially unchanged."""
    solver = OmegaKSolver(cell)
    at_rest = solver.solve(spin_rpm=0.0, n_points=41, n_branches=6)
    spinning = solver.solve(spin_rpm=6000.0, n_points=41, n_branches=6)

    rest_gap = at_rest.band_gaps("both")[0]
    forward = spinning.band_gaps("forward")[0]
    backward = spinning.band_gaps("backward")[0]

    assert forward.center_hz > rest_gap.center_hz > backward.center_hz
    for gap in (forward, backward):
        assert gap.width_hz == pytest.approx(rest_gap.width_hz, rel=0.01)


def test_the_two_formulations_report_the_same_gap(cell) -> None:
    spin = 6000.0
    from_omega_k = OmegaKSolver(cell).solve(
        spin_rpm=spin, n_points=41, n_branches=6
    ).band_gaps("forward")[0]

    frequencies = np.arange(1500.0, 4200.0, 5.0)
    from_k_omega = KOmegaSolver(cell).solve(frequencies, spin_rpm=spin).band_gaps(
        "forward", min_width_hz=100.0
    )[0]

    assert from_k_omega.start_hz == pytest.approx(from_omega_k.start_hz, abs=10.0)
    assert from_k_omega.stop_hz == pytest.approx(from_omega_k.stop_hz, abs=10.0)


def test_band_gap_map_tracks_both_families(cell) -> None:
    speeds = np.array([0.0, 3000.0, 6000.0])
    gap_map = band_gap_map(cell, speeds, n_points=31, n_branches=6, f_max=5000.0)

    forward_tracks = gap_map.tracks("forward")
    backward_tracks = gap_map.tracks("backward")
    assert len(forward_tracks) == len(backward_tracks) >= 1

    # The first (lowest) stop band is the one the paper reports: it drifts up
    # with the forward whirl and down with the backward whirl.
    forward = min(forward_tracks, key=lambda t: t["center_hz"][0])
    backward = min(backward_tracks, key=lambda t: t["center_hz"][0])
    assert forward["center_hz"][-1] > forward["center_hz"][0]
    assert backward["center_hz"][-1] < backward["center_hz"][0]

    # Every track spans consecutive speeds of the sweep: a track must never
    # bridge a speed at which the gap did not exist.
    for track in forward_tracks + backward_tracks:
        positions = [int(np.flatnonzero(speeds == s)[0]) for s in track["speeds_rpm"]]
        assert positions == list(range(positions[0], positions[0] + len(positions)))

    assert "Band gap map" in gap_map.summary()
    with pytest.raises(KeyError):
        gap_map.tracks("both")


def test_attenuation_is_zero_in_the_pass_band_and_large_in_the_gap(cell) -> None:
    frequencies = np.array([1000.0, 2800.0])
    dispersion = KOmegaSolver(cell).solve(frequencies, spin_rpm=0.0)
    envelope = dispersion.attenuation_envelope("forward")
    assert envelope[0] < 1e-6  # propagates freely
    assert envelope[1] > 1.0  # evanescent inside the gap


# ----------------------------------------------------------------------
# Discretisation
# ----------------------------------------------------------------------
def test_mesh_convergence_of_the_first_branches() -> None:
    rotor = reference_rotor(n_disks=11)

    def build(n_elements):
        return reference_rotor(n_disks=11, elements_per_cell=n_elements).unit_cell()

    report = mesh_convergence(build, [5, 10, 20, 40], n_branches=4)
    # Ten elements per cell is already within 0.5% of the finest mesh.
    assert report["relative_error"][1].max() < 5e-3
    assert report["relative_error"][2].max() < 1e-3
    assert rotor.cell_length == pytest.approx(1.5 / 11.0)


# ----------------------------------------------------------------------
# Regressions: bugs found in the pre-publication review
# ----------------------------------------------------------------------
def test_unfilled_waves_are_never_counted_as_propagating(cell) -> None:
    """A wave slot with no wave in it must not read as a propagating wave.

    ``np.full(shape, np.nan, dtype=complex)`` yields ``nan+0j``, whose
    imaginary part is exactly zero.  If such a slot reached the band gap
    detection it would look like a wave with zero attenuation and would erase
    every stop band.
    """
    frequencies = np.arange(500.0, 4000.0, 50.0)
    dispersion = KOmegaSolver(cell).solve(frequencies, spin_rpm=0.0)

    dispersion.wavenumbers[-1, :] = np.nan  # simulate a missing wave
    dispersion.whirl[-1, :] = np.nan
    assert not dispersion.valid[-1].any()
    assert not dispersion.propagating()[-1].any()
    assert not dispersion.whirl_mask("both")[-1].any()

    gaps = dispersion.band_gaps("both", min_width_hz=100.0)
    assert gaps, "the stop band must survive a missing wave"
    assert gaps[0].start_hz == pytest.approx(2050.0, abs=60.0)


def test_the_coupled_solver_finds_the_same_gap_as_the_decoupled_one(cell) -> None:
    """The general path must agree with the whirl-decoupled one.

    At zero spin speed forward and backward whirl are degenerate, so the whirl
    index of the general eigensolver is ~0 for every wave.  Selecting the
    positive-going waves per whirl family must not discard those waves.
    """
    frequencies = np.arange(500.0, 5000.0, 50.0)
    decoupled = KOmegaSolver(cell).solve(frequencies, spin_rpm=0.0)
    coupled = KOmegaSolver(cell, decouple_whirl=False).solve(frequencies, spin_rpm=0.0)

    assert coupled.valid.all(), "no wave slot may be left empty"
    gap_decoupled = decoupled.band_gaps("both", min_width_hz=200.0)[0]
    gap_coupled = coupled.band_gaps("both", min_width_hz=200.0)[0]
    assert gap_coupled.start_hz == pytest.approx(gap_decoupled.start_hz, abs=50.0)
    assert gap_coupled.stop_hz == pytest.approx(gap_decoupled.stop_hz, abs=50.0)


def test_f_max_clips_a_band_gap_instead_of_discarding_it(cell) -> None:
    """``f_max`` is a display window, not part of the physics.

    The second stop band of the reference rotor is very wide; asking for the
    gaps below 7 kHz must still report it (clipped at 7 kHz), not drop it.
    """
    dispersion = OmegaKSolver(cell).solve(spin_rpm=0.0, n_points=41, n_branches=8)
    unlimited = dispersion.band_gaps("both")
    windowed = dispersion.band_gaps("both", f_max=7000.0)

    assert len(unlimited) >= 2
    assert len(windowed) == 2
    assert windowed[0].stop_hz == pytest.approx(unlimited[0].stop_hz)
    assert windowed[1].start_hz == pytest.approx(unlimited[1].start_hz)
    assert windowed[1].stop_hz == pytest.approx(7000.0)

    # The region above the highest computed branch is still not a band gap.
    assert all(gap.stop_hz < dispersion.frequencies_hz.max() for gap in unlimited)


def test_constraining_a_unit_cell_returns_a_unit_cell(cell) -> None:
    constrained = cell.constrain([0])
    assert type(constrained) is type(cell)
    assert constrained.cell_length == pytest.approx(cell.cell_length)


def test_mesh_convergence_reference_is_the_finest_mesh() -> None:
    def build(n_elements):
        return reference_rotor(n_disks=11, elements_per_cell=n_elements).unit_cell()

    ascending = mesh_convergence(build, [5, 10, 40], n_branches=4)
    descending = mesh_convergence(build, [40, 10, 5], n_branches=4)
    # The finest mesh is the reference in both orders, so its error is zero.
    assert ascending["relative_error"][-1].max() == pytest.approx(0.0)
    assert descending["relative_error"][0].max() == pytest.approx(0.0)
