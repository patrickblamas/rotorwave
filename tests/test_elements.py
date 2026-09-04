"""Element level checks: structure of the matrices and analytical beam results."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rotorwave import STEEL, RigidDisk, RotorFEModel, ShaftElement


@pytest.fixture
def element() -> ShaftElement:
    return ShaftElement(length=0.05, outer_diameter=0.1, material=STEEL)


def test_mass_matrix_is_symmetric_and_positive_definite(element: ShaftElement) -> None:
    mass = element.mass_matrix()
    assert np.allclose(mass, mass.T)
    assert np.all(np.linalg.eigvalsh(mass) > 0.0)


def test_gyroscopic_matrix_is_skew_symmetric(element: ShaftElement) -> None:
    gyroscopic = element.gyroscopic_matrix()
    assert np.allclose(gyroscopic, -gyroscopic.T)


def test_stiffness_matrix_has_four_rigid_body_modes(element: ShaftElement) -> None:
    stiffness = element.stiffness_matrix()
    assert np.allclose(stiffness, stiffness.T)
    eigenvalues = np.linalg.eigvalsh(stiffness)
    scale = eigenvalues.max()
    # Two rigid translations and two rigid rotations, one per bending plane.
    assert np.sum(eigenvalues < 1e-9 * scale) == 4


def test_section_properties_of_a_hollow_shaft() -> None:
    hollow = ShaftElement(
        length=0.1, outer_diameter=0.1, inner_diameter=0.06, material=STEEL
    )
    assert hollow.area == pytest.approx(math.pi * (0.05**2 - 0.03**2))
    assert hollow.second_moment == pytest.approx(0.25 * math.pi * (0.05**4 - 0.03**4))
    assert hollow.mass < ShaftElement(0.1, 0.1, STEEL).mass


def test_disk_inertias_follow_the_uniform_plate_formulas() -> None:
    disk = RigidDisk.from_geometry(diameter=0.38, thickness=0.022, material=STEEL)
    radius = 0.19
    mass = STEEL.density * math.pi * radius**2 * 0.022
    assert disk.mass == pytest.approx(mass)
    assert disk.polar_inertia == pytest.approx(0.5 * mass * radius**2)
    assert disk.diametral_inertia == pytest.approx(
        mass * (3.0 * radius**2 + 0.022**2) / 12.0
    )
    assert disk.polar_inertia > disk.diametral_inertia  # a thin disk


def test_simply_supported_beam_matches_euler_bernoulli_theory() -> None:
    """The classical closed-form solution is the reference for the element.

    For a uniform simply supported beam the n-th natural frequency is
    ``f_n = (n^2 pi / 2) sqrt(E I / (rho A L^4))``.
    """
    length, diameter, n_elements = 1.0, 0.02, 40
    elements = [
        ShaftElement(length / n_elements, diameter, STEEL) for _ in range(n_elements)
    ]
    model = RotorFEModel(elements=elements)
    model = model.constrain(
        [
            model.dof(0, "x"),
            model.dof(0, "y"),
            model.dof(-1, "x"),
            model.dof(-1, "y"),
        ]
    )

    area = elements[0].area
    inertia = elements[0].second_moment
    analytical = [
        (n**2 * math.pi / 2.0)
        * math.sqrt(STEEL.young_modulus * inertia / (STEEL.density * area * length**4))
        for n in (1, 2, 3)
    ]

    computed = model.natural_frequencies(spin_rpm=0.0)
    # Each mode appears twice (one per bending plane) in an isotropic rotor.
    unique = computed[::2][:3]
    assert unique == pytest.approx(analytical, rel=2e-3)
