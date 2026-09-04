"""Wave solver checks, including the cross-validation of the two formulations."""

from __future__ import annotations

import numpy as np
import pytest

from rotorwave import KOmegaSolver, OmegaKSolver, reference_rotor
from rotorwave.whirl import whirl_generator

SPIN = 6000.0


@pytest.fixture(scope="module")
def cell():
    return reference_rotor(n_disks=11, elements_per_cell=10).unit_cell()


def test_model_is_isotropic(cell) -> None:
    """The whole method assumes isotropy; check it instead of trusting it.

    The generator of rotations about the spin axis must commute with the
    inertia, gyroscopic and stiffness matrices — that commutation is what makes
    the forward/backward separation exact.
    """
    generator = whirl_generator(cell.n_dofs)
    for matrix in (cell.M, cell.G, cell.K):
        commutator = generator @ matrix - matrix @ generator
        assert np.abs(commutator).max() <= 1e-9 * np.abs(matrix).max()


def test_zero_spin_makes_precession_modes_degenerate(cell) -> None:
    dispersion = OmegaKSolver(cell).solve(spin_rpm=0.0, n_points=9, n_branches=4)
    forward = dispersion.frequencies_hz[dispersion.branch_indices("forward")]
    backward = dispersion.frequencies_hz[dispersion.branch_indices("backward")]
    assert forward.shape == backward.shape
    assert np.allclose(np.sort(forward, axis=0), np.sort(backward, axis=0), rtol=1e-8)


def test_gyroscopic_effect_splits_the_branches(cell) -> None:
    dispersion = OmegaKSolver(cell).solve(spin_rpm=SPIN, n_points=9, n_branches=4)
    kinds = np.array(dispersion.branch_kinds())
    assert set(kinds) == {"forward", "backward"}
    # Whirl is essentially circular for an isotropic rotor.
    assert np.abs(np.abs(dispersion.branch_whirl) - 1.0).max() < 1e-6

    forward = dispersion.frequencies_hz[kinds == "forward"]
    backward = dispersion.frequencies_hz[kinds == "backward"]
    # Away from k = 0 the forward branches are stiffened by the gyroscopic
    # effect and the backward ones softened.
    assert np.all(np.sort(forward, axis=0)[:, 1:] > np.sort(backward, axis=0)[:, 1:])


def test_rigid_body_modes_at_zero_wavenumber(cell) -> None:
    dispersion = OmegaKSolver(cell).solve(
        wavenumbers=np.array([0.0]), spin_rpm=0.0, n_branches=4
    )
    frequencies = dispersion.frequencies_hz[:, 0]
    # Two rigid translations; the rotations are not compatible with the
    # periodicity condition, so they carry elastic energy.
    assert np.sum(frequencies < 1e-6 * frequencies.max()) == 2


def test_dispersion_is_periodic_in_the_wavenumber(cell) -> None:
    delta = cell.cell_length
    k = 0.3 * np.pi / delta
    solver = OmegaKSolver(cell)
    first = solver.solve(wavenumbers=np.array([k]), spin_rpm=SPIN, n_branches=4)
    shifted = solver.solve(
        wavenumbers=np.array([k + 2.0 * np.pi / delta]), spin_rpm=SPIN, n_branches=4
    )
    assert first.frequencies_hz == pytest.approx(shifted.frequencies_hz, rel=1e-8)


def test_omega_k_and_k_omega_agree(cell) -> None:
    """The two formulations must return the same point of the same curve.

    Taking a frequency from an omega(k) branch and feeding it to the k(omega)
    solver has to give back the wavenumber that branch was evaluated at, with
    the same whirl direction.
    """
    delta = cell.cell_length
    target = 0.4 * np.pi / delta
    dispersion = OmegaKSolver(cell).solve(
        wavenumbers=np.array([target]), spin_rpm=SPIN, n_branches=4
    )
    solver = KOmegaSolver(cell)

    for frequency, whirl in zip(
        dispersion.frequencies_hz[:, 0], dispersion.whirl[:, 0]
    ):
        result = solver.solve(np.array([frequency]), spin_rpm=SPIN)
        same_whirl = np.sign(result.whirl[:, 0]) == np.sign(whirl)
        propagating = result.propagating(tol_db=1e-3)[:, 0]
        candidates = np.abs(result.wavenumbers[:, 0].real)[same_whirl & propagating]
        assert np.min(np.abs(candidates - target)) < 1e-6 * target


def test_propagating_waves_have_unit_modulus(cell) -> None:
    """Bloch multipliers of propagating waves must sit on the unit circle.

    This is the numerical quality check of the k(omega) solver: the exact
    determinant expansion keeps the error below 1e-6 even at the stiff,
    ill-conditioned low frequency end.
    """
    result = KOmegaSolver(cell).solve(np.array([10.0, 100.0, 1000.0]), spin_rpm=SPIN)
    propagating = result.propagating(tol_db=1e-3)
    moduli = np.abs(result.multipliers)[propagating]
    assert moduli.size > 0
    assert np.abs(moduli - 1.0).max() < 1e-6


def test_branch_tracking_keeps_branches_continuous(cell) -> None:
    dispersion = OmegaKSolver(cell).solve(
        spin_rpm=SPIN, n_points=41, n_branches=4, track=True
    )
    steps = np.abs(np.diff(dispersion.frequencies_hz, axis=1))
    span = dispersion.frequencies_hz.max() - dispersion.frequencies_hz.min()
    # No branch may jump by a sizeable fraction of the whole spectrum.
    assert steps.max() < 0.2 * span
