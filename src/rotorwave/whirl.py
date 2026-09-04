"""Classification of whirl direction (forward / backward precession).

A rotor mode is an elliptical orbit that can always be decomposed into two
counter-rotating circular components.  For a nodal motion

.. math::

    x(t) = \\Re\\{X e^{i\\omega t}\\}, \\qquad y(t) = \\Re\\{Y e^{i\\omega t}\\},

the complex position :math:`z = x + iy` reads

.. math::

    z(t) = \\underbrace{\\tfrac{1}{2}(X + iY)}_{\\text{forward}} e^{i\\omega t}
         + \\underbrace{\\tfrac{1}{2}\\overline{(X - iY)}}_{\\text{backward}}
           e^{-i\\omega t} ,

so a purely forward orbit satisfies :math:`Y = -iX`.

That condition is an eigenvalue problem in disguise.  The generator of
rotations about the spin axis,

.. math::
    J = \\mathrm{diag}_\\text{nodes}
        \\begin{bmatrix} 0 & -1 \\\\ 1 & 0 \\end{bmatrix}
        \\oplus \\begin{bmatrix} 0 & -1 \\\\ 1 & 0 \\end{bmatrix},

satisfies :math:`Jv = +i v` for forward motion and :math:`Jv = -i v` for
backward motion, and it *commutes* with the inertia, gyroscopic and stiffness
matrices of an isotropic rotor.  The whirl index used here is therefore the
Rayleigh quotient of the Hermitian operator :math:`-iJ`,

.. math:: W = \\frac{v^H M (-iJ) v}{v^H M v} \\in [-1, 1],

which equals :math:`+1` for a circular forward orbit, :math:`-1` for a
circular backward one and :math:`0` for a straight line.  Being a Rayleigh
quotient it is insensitive to the arbitrary scaling and phase of a mode shape
and is numerically stable, unlike heuristics based on the ordering of the
eigenvalues.

The same commutation property is used to clean up near-degenerate solutions:
when two waves share (almost) the same eigenvalue, any combination of them is
also a solution, and the physically meaningful pair is recovered by
diagonalising :math:`J` inside that subspace.
"""

from __future__ import annotations

import numpy as np

from .dispersion import WHIRL_TOL
from .elements import DOF_PER_NODE

__all__ = [
    "whirl_generator",
    "whirl_basis",
    "whirl_index",
    "whirl_components",
    "whirl_label",
    "separate_degenerate_whirl",
]


def whirl_generator(n_dofs: int) -> np.ndarray:
    """The rotation generator :math:`J` for ``n_dofs`` degrees of freedom."""
    if n_dofs % DOF_PER_NODE:
        raise ValueError("n_dofs must be a multiple of the nodal dofs")
    block = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    generator = np.zeros((n_dofs, n_dofs))
    for node in range(n_dofs // DOF_PER_NODE):
        sl = slice(node * DOF_PER_NODE, (node + 1) * DOF_PER_NODE)
        generator[sl, sl] = block
    return generator


def whirl_basis(n_dofs: int) -> tuple[np.ndarray, int]:
    """Unitary basis that splits a model into forward and backward subspaces.

    The columns are the eigenvectors of :math:`J`: the first half spans the
    forward subspace (:math:`Jv = +iv`, i.e. :math:`y = -ix`), the second half
    the backward one.  Because :math:`J` commutes with the matrices of an
    isotropic rotor, this basis block-diagonalises the whole problem into two
    independent halves — the wave counterpart of the directional frequency
    response functions used to separate precession modes experimentally.

    Returns
    -------
    basis:
        Unitary matrix ``T`` of shape ``(n_dofs, n_dofs)``.
    n_forward:
        Number of forward columns, i.e. ``n_dofs // 2``.
    """
    if n_dofs % DOF_PER_NODE:
        raise ValueError("n_dofs must be a multiple of the nodal dofs")
    n_nodes = n_dofs // DOF_PER_NODE
    root = 1.0 / np.sqrt(2.0)

    forward = np.zeros((n_dofs, 2 * n_nodes), dtype=complex)
    backward = np.zeros_like(forward)
    for node in range(n_nodes):
        base = node * DOF_PER_NODE
        for pair in (0, 1):  # translations (x, y) and rotations (beta, gamma)
            column = 2 * node + pair
            forward[base + 2 * pair, column] = root
            forward[base + 2 * pair + 1, column] = -1j * root
            backward[base + 2 * pair, column] = root
            backward[base + 2 * pair + 1, column] = 1j * root
    return np.hstack([forward, backward]), 2 * n_nodes


def whirl_index(
    mode_shape: np.ndarray, metric: np.ndarray | None = None
) -> float | np.ndarray:
    """Whirl index of one or several mode shapes.

    Parameters
    ----------
    mode_shape:
        Complex mode shape(s) with the package dof ordering
        ``[x, y, beta, gamma]`` per node.  Shape ``(n_dofs,)`` for a single
        mode or ``(n_dofs, n_modes)`` for several of them.
    metric:
        Optional Hermitian weighting matrix, normally the inertia matrix, so
        that the index is weighted by the kinetic energy of each node.

    Returns
    -------
    float or ndarray
        Whirl index in ``[-1, 1]``; positive means forward precession.
    """
    shape = np.atleast_2d(np.asarray(mode_shape, dtype=complex).T).T
    single = np.asarray(mode_shape).ndim == 1
    generator = whirl_generator(shape.shape[0])

    weighted = shape if metric is None else metric @ shape
    numerator = np.einsum("ij,ij->j", weighted.conj(), -1j * (generator @ shape)).real
    denominator = np.einsum("ij,ij->j", weighted.conj(), shape).real

    index = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=np.abs(denominator) > 0.0,
    )
    index = np.clip(index, -1.0, 1.0)
    return float(index[0]) if single else index


def whirl_components(mode_shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward and backward circular amplitudes of the orbit of every node.

    Returns
    -------
    forward, backward:
        Arrays with one entry per node (per mode, if several are given), equal
        to :math:`|X + iY|/2` and :math:`|X - iY|/2`.
    """
    shape = np.atleast_2d(np.asarray(mode_shape, dtype=complex).T).T
    x = shape[0::DOF_PER_NODE]
    y = shape[1::DOF_PER_NODE]
    return 0.5 * np.abs(x + 1j * y), 0.5 * np.abs(x - 1j * y)


def whirl_label(index: float, tol: float = WHIRL_TOL) -> str:
    """Map a whirl index to ``'forward'``, ``'backward'`` or ``'mixed'``."""
    if index > tol:
        return "forward"
    if index < -tol:
        return "backward"
    return "mixed"


def separate_degenerate_whirl(
    values: np.ndarray, vectors: np.ndarray, rtol: float = 1e-6
) -> np.ndarray:
    """Rotate near-degenerate eigenvectors into pure whirl directions.

    When two eigenvalues coincide (which happens at low frequency, at zero
    spin speed, and generally wherever the forward and backward waves are
    close), any linear combination of their eigenvectors is also a solution
    and a general purpose eigensolver returns an arbitrary mixture.  Because
    the rotation generator commutes with an isotropic rotor model, the pure
    forward and backward pair can be recovered by diagonalising :math:`J`
    inside the degenerate subspace.

    Parameters
    ----------
    values:
        Eigenvalues, shape ``(n,)``.
    vectors:
        Eigenvectors, shape ``(n_dofs, n)``.
    rtol:
        Relative tolerance used to decide that two eigenvalues coincide.

    Returns
    -------
    ndarray
        Eigenvectors with the same shape, cleaned inside each degenerate group.
    """
    vectors = np.array(vectors, dtype=complex, copy=True)
    if vectors.shape[1] < 2:
        return vectors

    generator = -1j * whirl_generator(vectors.shape[0])
    scale = np.max(np.abs(values)) or 1.0
    order = np.argsort(np.abs(values))

    group: list[int] = []
    for index in order:
        if not group:
            group = [index]
            continue
        if abs(values[index] - values[group[-1]]) <= rtol * scale:
            group.append(index)
        else:
            _diagonalise_group(vectors, group, generator)
            group = [index]
    _diagonalise_group(vectors, group, generator)
    return vectors


def _diagonalise_group(
    vectors: np.ndarray, group: list[int], generator: np.ndarray
) -> None:
    """Diagonalise ``generator`` inside the subspace spanned by ``group``."""
    if len(group) < 2:
        return
    basis = vectors[:, group]
    # Orthonormalise first, so the projected operator stays Hermitian.
    q, _ = np.linalg.qr(basis)
    projected = q.conj().T @ (generator @ q)
    projected = 0.5 * (projected + projected.conj().T)
    _, rotation = np.linalg.eigh(projected)
    vectors[:, group] = q @ rotation
