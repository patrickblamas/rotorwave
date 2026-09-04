"""Receptance of the finite rotor, and its agreement with the wave analysis."""

from __future__ import annotations

import numpy as np
import pytest

from rotorwave import OmegaKSolver, ReceptanceSolver, reference_rotor

SPIN = 6000.0


@pytest.fixture(scope="module")
def rotor():
    return reference_rotor(n_disks=11, elements_per_cell=20)


@pytest.fixture(scope="module")
def receptance(rotor):
    model = rotor.full_rotor(elements_per_cell=10)
    solver = ReceptanceSolver(model)
    return solver.compute(np.arange(10.0, 5001.0, 10.0), spin_rpm=SPIN)


@pytest.fixture(scope="module")
def gaps(rotor):
    dispersion = OmegaKSolver(rotor.unit_cell()).solve(
        spin_rpm=SPIN, n_points=41, n_branches=6
    )
    return {
        kind: dispersion.band_gaps(kind)[0] for kind in ("forward", "backward")
    }


def test_excitation_and_response_are_next_to_the_bearings(rotor) -> None:
    model = rotor.full_rotor(elements_per_cell=10)
    solver = ReceptanceSolver(model)
    assert solver.excitation_node == 1
    assert solver.response_node == model.n_nodes - 2


def test_no_resonance_inside_the_band_gap(receptance, gaps) -> None:
    """A stop band must be free of resonances of the finite rotor."""
    margin = 25.0  # the frequency grid is 10 Hz, band edges are not exact
    common_start = max(gaps["forward"].start_hz, gaps["backward"].start_hz) + margin
    common_stop = min(gaps["forward"].stop_hz, gaps["backward"].stop_hz) - margin
    resonances = receptance.resonances()
    inside = resonances[(resonances > common_start) & (resonances < common_stop)]
    assert inside.size == 0


def test_response_collapses_inside_the_band_gap(receptance, gaps) -> None:
    frequencies = receptance.frequencies_hz
    magnitude = receptance.magnitude_db()
    pass_band = magnitude[frequencies < gaps["backward"].start_hz]
    stop_band = magnitude[
        (frequencies > gaps["forward"].start_hz)
        & (frequencies < gaps["backward"].stop_hz)
    ]
    assert stop_band.mean() < pass_band.mean() - 40.0


def test_directional_frf_separates_the_precession_directions(receptance, gaps) -> None:
    """The dFRF convention is locked to the (unambiguous) wave analysis.

    The last resonance below the gap must be a forward one for the forward
    dFRF and a backward one for the backward dFRF, and the forward gap starts
    higher than the backward gap.
    """
    forward = receptance.resonances("forward")
    backward = receptance.resonances("backward")

    last_forward = forward[forward < gaps["forward"].start_hz].max()
    last_backward = backward[backward < gaps["backward"].start_hz].max()

    assert last_forward > last_backward
    assert last_forward < gaps["forward"].start_hz
    assert last_backward < gaps["backward"].start_hz
    # ... and each one sits close below its own band edge.
    assert gaps["forward"].start_hz - last_forward < 120.0
    assert gaps["backward"].start_hz - last_backward < 120.0


def test_direct_and_directional_frfs_are_consistent(receptance) -> None:
    """The two dFRFs average back to the mean of the two direct receptances.

    This is an identity of the definition — the cross terms cancel — and holds
    for any rotor, isotropic or not.  Note that it is ``(H_xx + H_yy)/2`` and
    not ``receptance.direct``, which is ``H_xx`` alone; the two coincide only
    for an isotropic rotor, which is checked separately below.
    """
    average = 0.5 * (receptance.forward + receptance.backward)
    expected = 0.5 * (receptance.h_xx + receptance.h_yy)
    assert np.allclose(average, expected)


def test_isotropic_rotor_has_symmetric_receptances(receptance) -> None:
    assert np.allclose(receptance.h_xx, receptance.h_yy)
    assert np.allclose(receptance.h_xy, -receptance.h_yx)


def test_deepest_response_falls_inside_the_predicted_gap(receptance, gaps) -> None:
    """The quietest frequency below the second pass band is inside the gap."""
    frequencies = receptance.frequencies_hz
    window = frequencies < gaps["forward"].stop_hz
    quietest = frequencies[window][np.argmin(receptance.magnitude_db()[window])]
    assert gaps["forward"].start_hz < quietest < gaps["backward"].stop_hz


def test_quiet_bands_are_reported(receptance) -> None:
    bands = receptance.quiet_bands()
    assert bands, "no quiet band found"
    assert all(stop > start for start, stop in bands)
