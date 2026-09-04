"""Wave finite element (WFE) solvers for longitudinally periodic rotors.

Two complementary eigenvalue problems are implemented:

``OmegaKSolver`` — the :math:`\\omega(k)` problem
    The wavenumber :math:`k` is real and prescribed; the Bloch multiplier
    :math:`\\lambda = e^{-ik\\Delta}` is therefore known and the reduced cell
    matrices give a quadratic eigenvalue problem in :math:`\\omega` whose
    solutions are the propagating frequencies.  This is the problem that
    produces the dispersion diagrams and the wave-Campbell diagram.

``KOmegaSolver`` — the :math:`k(\\omega)` problem
    The frequency is real and prescribed; the internal degrees of freedom of
    the cell are condensed out and the periodicity conditions give a
    *quadratic* eigenvalue problem in :math:`\\lambda`, whose roots may be
    complex.  The real part of :math:`k` describes propagation, the imaginary
    part describes attenuation, so this problem also captures evanescent and
    strongly attenuated waves that the :math:`\\omega(k)` problem cannot see.

Both solvers classify every wave as forward or backward whirl from its own
eigenvector (see :mod:`rotorwave.whirl`) instead of relying on the ordering of
the eigenvalues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cholesky, eig, eigh, lu_factor, lu_solve
from scipy.optimize import linear_sum_assignment

from .dispersion import KOmegaDispersion, OmegaKDispersion
from .model import UnitCell
from .units import hz_to_rad, rpm_to_rad
from .whirl import separate_degenerate_whirl, whirl_basis, whirl_index

__all__ = ["OmegaKSolver", "KOmegaSolver"]


# ----------------------------------------------------------------------
# Core eigenvalue routines
# ----------------------------------------------------------------------
def _hermitian_frequencies(
    M: np.ndarray,
    G: np.ndarray,
    K: np.ndarray,
    spin: float,
    n_modes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Natural frequencies of a gyroscopic system, exploiting its symmetry.

    The quadratic eigenvalue problem of an undamped rotor,

    .. math:: (K + \\omega\\,\\Omega H - \\omega^2 M)\\,Q = 0,
              \\qquad H = -i G = H^H,

    has Hermitian coefficients, so it admits the symmetric linearisation

    .. math::
        \\underbrace{\\begin{bmatrix}\\Omega H & -M\\\\ -M & 0\\end{bmatrix}}_{B} v
        = \\mu \\underbrace{\\begin{bmatrix}K & 0\\\\ 0 & M\\end{bmatrix}}_{A} v,
        \\qquad \\mu = -1/\\omega, \\quad v = [Q,\\ \\omega Q]^T .

    ``A`` is Hermitian positive definite whenever ``K`` is, so the problem can
    be solved with a Hermitian solver instead of a general one.  Two things
    follow: the eigenvalues are **real by construction** (no spurious
    imaginary parts to clean up), and because the lowest positive frequencies
    map to the algebraically smallest :math:`\\mu`, only the few modes that are
    actually needed have to be computed.  This is roughly an order of
    magnitude faster than the state-space formulation for the same accuracy.

    Falls back to :func:`_state_space_frequencies` when that hypothesis does
    not hold.  The important case is :math:`k = 0`, where the Bloch-reduced
    cell has rigid-body modes and ``K`` is singular; there the smallest
    eigenvalue of ``K`` sits at the rounding level of the largest one (a
    condition number around ``1e17`` for this rotor), so its computed *sign*
    is numerical noise and a Cholesky factorisation may well succeed.  Testing
    the factorisation alone would therefore take the fast path or not by luck.
    The guard used here is the factorisation **and** an a posteriori check
    that the modes actually returned satisfy the original quadratic problem;
    that check is decisive whatever the rounding does.
    """
    n = M.shape[0]
    M = 0.5 * (M + M.conj().T)
    K = 0.5 * (K + K.conj().T)
    H = -1j * G
    H = 0.5 * (H + H.conj().T)

    zeros = np.zeros((n, n), dtype=complex)
    A = np.block([[K.astype(complex), zeros], [zeros, M.astype(complex)]])
    B = np.block([[spin * H, -M.astype(complex)], [-M.astype(complex), zeros]])

    try:
        cholesky(A, lower=False)
        subset = None if n_modes is None else (0, min(n_modes, 2 * n) - 1)
        mu, vectors = eigh(B, A, subset_by_index=subset)
    except (LinAlgError, np.linalg.LinAlgError):
        return _state_space_frequencies(M, G, K, spin)

    with np.errstate(divide="ignore"):
        omega = np.where(mu != 0.0, -1.0 / mu, np.inf)
    keep = np.isfinite(omega) & (omega > 0.0)
    omega, vectors = omega[keep], vectors[:, keep]

    order = np.argsort(omega)
    omega, vectors = omega[order], vectors[:, order]
    modes = vectors[:n, :]

    if not _satisfies_quadratic(M, H, K, spin, omega, modes):
        return _state_space_frequencies(M, G, K, spin)
    return omega / (2.0 * np.pi), modes


def _satisfies_quadratic(
    M: np.ndarray,
    H: np.ndarray,
    K: np.ndarray,
    spin: float,
    omega: np.ndarray,
    modes: np.ndarray,
    rtol: float = 1e-6,
) -> bool:
    """Check that ``(K + omega*spin*H - omega^2 M) Q = 0`` for every mode.

    The residual is normalised by the size of the individual terms, so the
    test is scale free.  It is what makes the fast path safe: a factorisation
    that succeeded on a numerically singular ``K`` produces modes that fail
    here, and the caller then uses the general solver instead.
    """
    if omega.size == 0:
        return False
    norms = np.linalg.norm(modes, axis=0)
    if np.any(norms == 0.0):
        return False
    q = modes / norms                       # one unit mode per column
    terms = (K @ q, spin * (H @ q) * omega, (M @ q) * omega**2)
    residual = np.linalg.norm(terms[0] + terms[1] - terms[2], axis=0)
    scale = np.max([np.linalg.norm(t, axis=0) for t in terms], axis=0)
    checked = scale > 0.0
    return bool(np.all(residual[checked] <= rtol * scale[checked]))


def _state_space_frequencies(
    M: np.ndarray, G: np.ndarray, K: np.ndarray, spin: float
) -> tuple[np.ndarray, np.ndarray]:
    """Natural frequencies and mode shapes of ``M q'' + (-spin G) q' + K q = 0``.

    Parameters
    ----------
    M, G, K:
        Inertia, gyroscopic and stiffness matrices (possibly complex
        Hermitian, as produced by the Bloch reduction).
    spin:
        Spin speed :math:`\\Omega` [rad/s].

    Returns
    -------
    frequencies:
        ``n`` non-negative natural frequencies [Hz], in ascending order.
    modes:
        ``(n_dofs, n)`` complex displacement mode shapes, columns matching the
        frequencies.

    Notes
    -----
    The problem is linearised in state-space form with
    :math:`u = [\\dot q, q]^T`, giving :math:`\\dot u = A u` with

    .. math::
        A = \\begin{bmatrix} \\Omega M^{-1}G & -M^{-1}K \\\\ I & 0\\end{bmatrix}.

    For real ``M``, ``G`` and ``K`` — the case in which this fallback is
    actually used, namely the Bloch reduction at :math:`k = 0`, where
    :math:`\\lambda = 1` — the ``2n`` eigenvalues come in conjugate pairs
    :math:`\\pm i\\omega`, and the ``n`` values with the largest imaginary part
    are kept.  That rule is stable even when rigid-body modes make
    :math:`\\omega = 0` a repeated root, which is why it is preferred here over
    selecting the roots with a positive imaginary part.

    The system is conservative, so every physical eigenvalue is purely
    imaginary.  A root whose real part is *not* negligible against its
    imaginary part is therefore not an oscillation but the numerical debris of
    a zero root — at :math:`k = 0` the reduced ``K`` is singular and the two
    rigid-body translations come back as, say,
    :math:`-1.9\\times10^{-2} + 9.8\\times10^{-4}i` instead of exactly zero.
    Those are reported as zero frequency rather than as a spurious 0.001 Hz
    mode; genuine modes miss the test by ten orders of magnitude, so the
    threshold is not delicate.
    """
    n = M.shape[0]
    lu = lu_factor(M)
    minv_g = lu_solve(lu, G)
    minv_k = lu_solve(lu, K)

    A = np.zeros((2 * n, 2 * n), dtype=complex)
    A[:n, :n] = spin * minv_g
    A[:n, n:] = -minv_k
    A[n:, :n] = np.eye(n)

    values, vectors = eig(A)
    order = np.argsort(values.imag)[n:]  # the n largest imaginary parts
    values, vectors = values[order], vectors[:, order]

    # Discard the debris of the zero roots (see Notes above): a physical mode
    # of a conservative system has a negligible real part.
    oscillatory = np.abs(values.real) < 1e-3 * np.abs(values.imag)
    frequencies = np.where(oscillatory, np.maximum(values.imag, 0.0), 0.0)
    frequencies = frequencies / (2.0 * np.pi)
    modes = vectors[n:, :]  # displacement block of the state vector

    order = np.argsort(frequencies)
    return frequencies[order], modes[:, order]


def _matrix_polynomial_determinant(entries: list[list[np.ndarray]]) -> np.ndarray:
    """Determinant of a matrix whose entries are polynomials.

    ``entries[i][j]`` holds the coefficients of the ``(i, j)`` entry, highest
    degree first.  The expansion is exact (Laplace cofactors with polynomial
    arithmetic), which is affordable for the small boundary blocks handled
    here and much better conditioned than a linearisation.
    """
    size = len(entries)
    if size == 1:
        return entries[0][0]
    total = np.zeros(1, dtype=complex)
    for column in range(size):
        minor = [
            [entries[row][other] for other in range(size) if other != column]
            for row in range(1, size)
        ]
        term = np.convolve(entries[0][column], _matrix_polynomial_determinant(minor))
        if column % 2:
            term = -term
        total = np.polyadd(total, term)
    return total


def _quadratic_eigenvalues(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, exact_determinant: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the quadratic eigenvalue problem ``(A l^2 + B l + C) x = 0``.

    Two algorithms are available.

    * ``exact_determinant`` (default for small blocks): the characteristic
      polynomial :math:`\\det(A\\lambda^2 + B\\lambda + C)` is expanded exactly
      and its roots are computed with a balanced companion matrix, then the
      eigenvectors are obtained from the null space of the pencil evaluated at
      each root.  For the ill-conditioned low-frequency end of a stiff rotor
      cell this is dramatically more accurate.  Measured on the reference
      rotor between 10 Hz and 1 kHz, the propagating waves come out with
      :math:`||\\lambda| - 1| \\le 6\\times10^{-9}`, against
      :math:`2\\times10^{-3}` for the linearisation — the difference between a
      clean dispersion diagram and one peppered with spurious narrow stop
      bands.  Restricted to blocks of order four or less, above which
      expanding the determinant costs more than it is worth and the
      linearisation is used instead.
    * otherwise, the first companion linearisation solved with the QZ
      algorithm, which tolerates a singular leading matrix and any size.
    """
    n = A.shape[0]
    if exact_determinant and n <= 4:
        entries = [
            [np.array([A[i, j], B[i, j], C[i, j]], dtype=complex) for j in range(n)]
            for i in range(n)
        ]
        polynomial = _matrix_polynomial_determinant(entries)
        polynomial = np.trim_zeros(polynomial, "f")
        if polynomial.size > 1:
            scale = np.abs(polynomial).max()
            values = np.roots(polynomial / scale)
            vectors = np.column_stack(
                [_null_vector(A * lam**2 + B * lam + C) for lam in values]
            )
            return values, vectors

    eye = np.eye(n)
    zeros = np.zeros((n, n))
    lhs = np.block([[-B, -C], [eye, zeros]])
    rhs = np.block([[A, zeros], [zeros, eye]])
    values, vectors = eig(lhs, rhs)
    return values, vectors[n:, :]  # lower block is x itself


def _null_vector(matrix: np.ndarray) -> np.ndarray:
    """Right null vector of a (numerically) singular matrix."""
    _, _, vh = np.linalg.svd(matrix)
    return vh[-1].conj()


def _mass_normalised(modes: np.ndarray, metric: np.ndarray | None) -> np.ndarray:
    """Normalise mode shapes with respect to a metric (the inertia matrix)."""
    if metric is None:
        norms = np.linalg.norm(modes, axis=0)
    else:
        norms = np.sqrt(np.abs(np.einsum("ij,ik->k", modes.conj(), metric @ modes)))
    norms = np.where(norms > 0.0, norms, 1.0)
    return modes / norms


def _track_branches(
    frequencies: np.ndarray,
    modes: list[np.ndarray],
    metrics: list[np.ndarray] | None = None,
    frequency_weight: float = 1.0,
) -> np.ndarray:
    """Reorder eigen-solutions so that each row follows a continuous branch.

    Successive parameter steps are matched with the Hungarian algorithm on a
    cost that combines two continuity criteria:

    * a **mass-weighted MAC**, :math:`|v_a^H M v_b|^2 /
      [(v_a^H M v_a)(v_b^H M v_b)]`.  The plain
      Euclidean MAC is not reliable here because the modes of a gyroscopic
      system are orthogonal with respect to the inertia matrix, not to the
      identity, so unrelated branches can show a correlation of 0.4 or more;
    * a **frequency prediction**, obtained by linear extrapolation of the two
      previous steps, which resolves the ambiguity inside the degenerate
      forward/backward pairs that appear at zero spin speed.

    Sorting purely by frequency, as the original implementation does, swaps
    branches wherever two curves veer or cross — the group velocity and the
    colour maps computed from such a table are meaningless.

    Parameters
    ----------
    frequencies:
        ``(n_modes, n_steps)`` array of frequencies [Hz].
    modes:
        ``n_steps`` arrays of shape ``(n_dofs, n_modes)``.
    metrics:
        Optional inertia matrices used to weight the MAC, one per step.
    frequency_weight:
        Relative weight of the frequency-continuity term.  ``0`` reproduces a
        pure MAC tracking.

    Returns
    -------
    ndarray
        ``(n_modes, n_steps)`` integer array of permutations to apply.
    """
    n_modes, n_steps = frequencies.shape
    permutation = np.zeros((n_modes, n_steps), dtype=int)
    permutation[:, 0] = np.arange(n_modes)
    if n_steps == 1:
        return permutation

    previous = _mass_normalised(modes[0], metrics[0] if metrics else None)
    tracked = frequencies[:, 0].copy()
    older: np.ndarray | None = None

    for step in range(1, n_steps):
        metric = metrics[step] if metrics else None
        current = _mass_normalised(modes[step], metric)
        overlap = (
            previous.conj().T @ (metric @ current)
            if metric is not None
            else previous.conj().T @ current
        )
        mac = np.abs(overlap) ** 2

        predicted = tracked if older is None else 2.0 * tracked - older
        span = float(np.ptp(frequencies[:, step]))
        # Frequency distances are made dimensionless by the typical spacing
        # between modes, so that the two halves of the cost are comparable:
        # a full MAC mismatch (cost 1) then weighs about as much as being one
        # mode-spacing away in frequency.  The 1 Hz floor guards the
        # degenerate case of a spectrum with no spread.
        scale = max(span / max(n_modes, 1), 1.0)
        distance = np.abs(frequencies[None, :, step] - predicted[:, None]) / scale

        # Distances beyond ten mode-spacings are clipped: past that point the
        # candidate is simply "far", and letting the distance grow without
        # bound would let one very distant pair dominate the assignment.
        cost = -mac + frequency_weight * np.minimum(distance, 10.0)
        rows, cols = linear_sum_assignment(cost)
        assignment = np.empty(n_modes, dtype=int)
        assignment[rows] = cols

        permutation[:, step] = assignment
        previous = current[:, assignment]
        older, tracked = tracked, frequencies[assignment, step]
    return permutation


# ----------------------------------------------------------------------
# omega(k) problem
# ----------------------------------------------------------------------
@dataclass
class OmegaKSolver:
    """Solve the :math:`\\omega(k)` problem of a unit cell.

    Parameters
    ----------
    cell:
        The unit cell of the periodic rotor.
    """

    cell: UnitCell

    def solve(
        self,
        wavenumbers: np.ndarray | None = None,
        spin_rpm: float = 0.0,
        n_branches: int | None = 8,
        n_points: int = 41,
        track: bool = True,
    ) -> OmegaKDispersion:
        """Compute the dispersion branches over the irreducible Brillouin zone.

        Parameters
        ----------
        wavenumbers:
            Real wavenumbers [rad/m].  Defaults to ``n_points`` values evenly
            spaced in :math:`[0, \\pi/\\Delta]`, the irreducible Brillouin zone.
        spin_rpm:
            Spin speed of the rotor [rpm].
        n_branches:
            Number of dispersion branches to keep (lowest frequencies).
            ``None`` keeps every branch of the reduced model.
        n_points:
            Number of wavenumbers, used only when ``wavenumbers`` is ``None``.
        track:
            Follow each branch with a MAC-based assignment (recommended).

        Returns
        -------
        OmegaKDispersion
        """
        delta = self.cell.cell_length
        if wavenumbers is None:
            wavenumbers = np.linspace(0.0, np.pi / delta, n_points)
        wavenumbers = np.atleast_1d(np.asarray(wavenumbers, dtype=float))
        spin = rpm_to_rad(spin_rpm)

        # A few extra modes are computed so that branch tracking is not
        # disturbed by modes entering and leaving the requested window.
        # A few extra branches are computed and discarded afterwards: the
        # tracker needs neighbours above the last branch of interest to avoid
        # mis-assigning it where it veers against the one just above.
        n_computed = None if n_branches is None else n_branches + 4

        frequencies: list[np.ndarray] = []
        modes: list[np.ndarray] = []
        metrics: list[np.ndarray] = []
        for k in wavenumbers:
            lam = np.exp(-1j * k * delta)
            m_bar, g_bar, k_bar = self.cell.reduced_bloch_matrices(lam)
            freq, shape = _hermitian_frequencies(
                m_bar, g_bar, k_bar, spin, n_modes=n_computed
            )
            if n_computed is not None:
                freq, shape = freq[:n_computed], shape[:, :n_computed]
            shape = separate_degenerate_whirl(freq, shape)
            frequencies.append(freq)
            modes.append(shape)
            metrics.append(m_bar)

        n_kept = min(freq.size for freq in frequencies)
        frequencies = [freq[:n_kept] for freq in frequencies]
        modes = [shape[:, :n_kept] for shape in modes]

        freq_matrix = np.column_stack(frequencies)
        if track:
            permutation = _track_branches(freq_matrix, modes, metrics)
            freq_matrix = np.take_along_axis(freq_matrix, permutation, axis=0)
            modes = [modes[j][:, permutation[:, j]] for j in range(len(modes))]

        whirl = np.column_stack(
            [
                whirl_index(shape, metric=metric)
                for shape, metric in zip(modes, metrics)
            ]
        )

        # Keep the lowest branches, ranked by their mean frequency.
        if n_branches is not None and n_branches < freq_matrix.shape[0]:
            keep = np.argsort(freq_matrix.mean(axis=1))[:n_branches]
            keep = keep[np.argsort(freq_matrix[keep].mean(axis=1))]
            freq_matrix = freq_matrix[keep]
            whirl = whirl[keep]
            modes = [shape[:, keep] for shape in modes]

        return OmegaKDispersion(
            wavenumbers=wavenumbers,
            frequencies_hz=freq_matrix,
            whirl=whirl,
            cell_length=delta,
            spin_rpm=float(spin_rpm),
            mode_shapes=np.stack(modes, axis=-1),
        )

    def wave_campbell(
        self,
        spin_speeds_rpm: np.ndarray,
        wavenumbers: np.ndarray | None = None,
        n_branches: int | None = 8,
        n_points: int = 41,
    ) -> list[OmegaKDispersion]:
        """Dispersion diagrams for a range of spin speeds (wave-Campbell data)."""
        return [
            self.solve(
                wavenumbers=wavenumbers,
                spin_rpm=float(speed),
                n_branches=n_branches,
                n_points=n_points,
            )
            for speed in np.atleast_1d(spin_speeds_rpm)
        ]


# ----------------------------------------------------------------------
# k(omega) problem
# ----------------------------------------------------------------------

@dataclass
class KOmegaSolver:
    """Solve the :math:`k(\\omega)` problem of a unit cell.

    Parameters
    ----------
    cell:
        The unit cell of the periodic rotor.
    decouple_whirl:
        When ``True`` (default) the boundary degrees of freedom are rotated
        into the forward/backward whirl basis, which block-diagonalises the
        problem for an isotropic rotor.  Each wave is then labelled by
        construction instead of by post-processing an eigenvector, the two
        halves are solved independently, and near-degenerate forward/backward
        pairs — which a general eigensolver returns hopelessly mixed — are
        resolved exactly.  The solver checks that the decoupling is legitimate
        and falls back to the coupled problem otherwise.
    assume_no_boundary_coupling:
        When ``True`` (default) the direct coupling between the left and right
        boundary degrees of freedom is neglected, which is exact for any cell
        meshed with more than one element.  Set it to ``False`` to keep the
        ``D_LR`` and ``D_RL`` blocks, so that single-element cells also work.
    """

    cell: UnitCell
    decouple_whirl: bool = True
    assume_no_boundary_coupling: bool = True

    def solve(
        self,
        frequencies_hz: np.ndarray,
        spin_rpm: float = 0.0,
        selection: str = "positive-going",
    ) -> KOmegaDispersion:
        """Compute complex wavenumbers for each prescribed frequency.

        Parameters
        ----------
        frequencies_hz:
            Frequencies of the analysis [Hz].
        spin_rpm:
            Spin speed [rpm].
        selection:
            ``'positive-going'`` keeps the waves that carry energy (or decay)
            towards positive z, which is the physically meaningful half of the
            spectrum; ``'all'`` returns both halves.  In either case the null
            and infinite roots of the pencil have already been discarded, and
            above ``2 * n_boundary`` roots only those closest to the unit
            circle are kept.

        Returns
        -------
        KOmegaDispersion
        """
        frequencies_hz = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
        delta = self.cell.cell_length
        n_boundary = self.cell.n_boundary_dofs

        n_waves = n_boundary if selection == "positive-going" else 2 * n_boundary
        # NOTE: the fill value must have a NaN *imaginary* part as well.
        # ``np.full(..., np.nan, dtype=complex)`` produces ``nan+0j``, whose
        # imaginary part is exactly zero, so any row left unfilled would be
        # read downstream as a perfectly propagating wave and would silently
        # erase the band gaps.
        nan_complex = complex(np.nan, np.nan)
        wavenumbers = np.full((n_waves, frequencies_hz.size), nan_complex)
        multipliers = np.full_like(wavenumbers, nan_complex)
        whirl = np.full((n_waves, frequencies_hz.size), np.nan)

        previous: tuple[np.ndarray, np.ndarray] | None = None
        for j, frequency in enumerate(frequencies_hz):
            lam, shapes, forces, families = self._solve_single(frequency, spin_rpm)
            if selection == "positive-going":
                keep = _select_positive_going(lam, shapes, forces, frequency, families)
            else:
                keep = np.argsort(-np.abs(lam))
            lam, shapes = lam[keep], shapes[:, keep]
            indices = whirl_index(shapes)
            indices = np.atleast_1d(indices)

            order = _order_waves(lam, indices, previous)
            lam, shapes, indices = lam[order], shapes[:, order], indices[order]
            previous = (lam, indices)

            count = min(n_waves, lam.size)
            multipliers[:count, j] = lam[:count]
            with np.errstate(divide="ignore", invalid="ignore"):
                wavenumbers[:count, j] = 1j * np.log(lam[:count]) / delta
            whirl[:count, j] = indices[:count]

        return KOmegaDispersion(
            frequencies_hz=frequencies_hz,
            wavenumbers=wavenumbers,
            whirl=whirl,
            multipliers=multipliers,
            cell_length=delta,
            spin_rpm=float(spin_rpm),
        )

    # ------------------------------------------------------------------
    def condensed_blocks(
        self, frequency_hz: float, spin_rpm: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Dynamic stiffness of the cell with the internal dofs condensed out.

        Returns the blocks :math:`\\hat D_{LL}, \\hat D_{LR}, \\hat D_{RL},
        \\hat D_{RR}` relating the boundary displacements to the boundary
        forces, obtained from the Schur complement of the internal block.
        """
        blocks = self.cell.partition_dynamic_stiffness(frequency_hz, spin_rpm)
        d_ll, d_rr = blocks["LL"], blocks["RR"]
        d_li, d_ir = blocks["LI"], blocks["IR"]
        d_ri, d_il = blocks["RI"], blocks["IL"]
        d_ii = blocks["II"]

        if d_ii.size:
            lu = lu_factor(d_ii)
            hat_ll = d_ll - d_li @ lu_solve(lu, d_il)
            hat_lr = -d_li @ lu_solve(lu, d_ir)
            hat_rl = -d_ri @ lu_solve(lu, d_il)
            hat_rr = d_rr - d_ri @ lu_solve(lu, d_ir)
        else:  # single-element cell: there is nothing to condense
            zeros = np.zeros_like(d_ll)
            hat_ll, hat_lr, hat_rl, hat_rr = d_ll, zeros, zeros.copy(), d_rr

        if not self.assume_no_boundary_coupling:
            hat_lr = hat_lr + blocks["LR"]
            hat_rl = hat_rl + blocks["RL"]
        return hat_ll, hat_lr, hat_rl, hat_rr

    def _solve_single(
        self, frequency_hz: float, spin_rpm: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bloch multipliers, boundary displacements, forces and whirl labels.

        Imposing :math:`q_R = \\lambda q_L` and the equilibrium
        :math:`f_L + \\lambda^{-1} f_R = 0` on the condensed cell gives the
        quadratic eigenvalue problem

        .. math::
            [\\lambda^2 \\hat D_{LR}
             + \\lambda(\\hat D_{LL} + \\hat D_{RR})
             + \\hat D_{RL}]\\, q_L = 0 .
        """
        hat_ll, hat_lr, hat_rl, hat_rr = self.condensed_blocks(frequency_hz, spin_rpm)
        quadratic, linear, constant = hat_lr, hat_ll + hat_rr, hat_rl

        values, vectors, families = self._eigen_pencil(quadratic, linear, constant)

        forces = np.column_stack(
            [
                (hat_ll + lam * hat_lr) @ vectors[:, index]
                for index, lam in enumerate(values)
            ]
        )
        return values, vectors, forces, families

    def _eigen_pencil(
        self, quadratic: np.ndarray, linear: np.ndarray, constant: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the quadratic pencil, decoupling the whirl directions if possible."""
        n_boundary = quadratic.shape[0]
        n_keep = 2 * n_boundary

        if self.decouple_whirl:
            basis, n_forward = whirl_basis(n_boundary)
            blocks = [
                basis.conj().T @ matrix @ basis
                for matrix in (quadratic, linear, constant)
            ]
            if _is_block_diagonal(blocks, n_forward):
                values: list[np.ndarray] = []
                vectors: list[np.ndarray] = []
                families: list[np.ndarray] = []
                halves = (
                    (1.0, slice(0, n_forward)),
                    (-1.0, slice(n_forward, None)),
                )
                for label, sl in halves:
                    sub = [matrix[sl, sl] for matrix in blocks]
                    sub_values, sub_vectors = _quadratic_eigenvalues(*sub)
                    sub_values, sub_vectors = _keep_physical_roots(
                        sub_values, sub_vectors, 2 * sub[0].shape[0]
                    )
                    values.append(sub_values)
                    vectors.append(basis[:, sl] @ sub_vectors)
                    families.append(np.full(sub_values.size, label))
                return (
                    np.concatenate(values),
                    np.hstack(vectors),
                    np.concatenate(families),
                )

        values, vectors = _quadratic_eigenvalues(quadratic, linear, constant)
        values, vectors = _keep_physical_roots(values, vectors, n_keep)
        # Waves whose Bloch multipliers nearly coincide come out of a general
        # eigensolver as an arbitrary mixture of forward and backward whirl;
        # rotate them back into pure precession directions.
        vectors = separate_degenerate_whirl(values, vectors)
        # No family labels here.  Only the whirl basis above guarantees a
        # balanced forward/backward split; the sign of the whirl index of the
        # general solution is unreliable where the two directions are
        # degenerate (at rest, for instance), and splitting the selection on
        # it would silently drop waves.  Select globally instead.
        return values, vectors, None


def _order_waves(
    multipliers: np.ndarray,
    whirl: np.ndarray,
    previous: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """Order the waves of one frequency so that each row is a continuous group.

    Without this the rows of the result are whatever order the eigensolver
    produced, and a dispersion curve jumps between wave groups from one
    frequency to the next — the vertical strokes that clutter hand-made
    versions of these diagrams.  Waves are matched to the previous frequency by
    the distance between their Bloch multipliers, with a large penalty on
    pairing waves of different precession direction.  The penalty is a cost,
    not a constraint: where the whirl labels are themselves unreliable — the
    general solver at zero spin returns indices near zero for every wave — the
    penalty does not apply and the matching falls back to multiplier distance
    alone, which is the best available criterion there.
    """
    if previous is None:
        # Forward waves first, propagating ones before evanescent ones.
        return np.lexsort((-np.abs(multipliers), -whirl))

    previous_multipliers, previous_whirl = previous
    if previous_multipliers.size != multipliers.size:
        return np.lexsort((-np.abs(multipliers), -whirl))

    distance = np.abs(previous_multipliers[:, None] - multipliers[None, :])
    mismatch = np.abs(previous_whirl[:, None] - whirl[None, :]) > 1.0
    # The penalty is far larger than any multiplier distance (which is at most
    # a few units), so a whirl mismatch is chosen only when no same-whirl
    # pairing exists at all.
    cost = distance + np.where(mismatch, 1e3, 0.0)
    rows, cols = linear_sum_assignment(cost)
    order = np.empty(multipliers.size, dtype=int)
    order[rows] = cols
    return order


def _is_block_diagonal(
    blocks: list[np.ndarray], n_forward: int, rtol: float = 1e-8
) -> bool:
    """Check that the whirl basis really decouples the problem."""
    for matrix in blocks:
        scale = np.abs(matrix).max()
        if scale == 0.0:
            continue
        off = max(
            np.abs(matrix[:n_forward, n_forward:]).max(),
            np.abs(matrix[n_forward:, :n_forward]).max(),
        )
        if off > rtol * scale:
            return False
    return True


def _keep_physical_roots(
    values: np.ndarray, vectors: np.ndarray, n_keep: int
) -> tuple[np.ndarray, np.ndarray]:
    """Drop null and infinite roots, keeping those closest to the unit circle."""
    finite = np.isfinite(values) & (np.abs(values) > 0.0)
    values, vectors = values[finite], vectors[:, finite]
    if values.size > n_keep:
        order = np.argsort(np.abs(np.log(np.abs(values))))[:n_keep]
        values, vectors = values[order], vectors[:, order]
    return values, vectors


def _select_positive_going(
    multipliers: np.ndarray,
    displacements: np.ndarray,
    forces: np.ndarray,
    frequency_hz: float,
    families: np.ndarray | None = None,
    tol: float = 1e-8,
) -> np.ndarray:
    """Indices of the waves travelling (or decaying) towards positive z.

    Evanescent waves are selected by their decay (:math:`|\\lambda| < 1`);
    propagating waves, for which :math:`|\\lambda| = 1`, are selected by the
    sign of the time-averaged power they carry into the cell,

    .. math:: P = -\\tfrac{1}{2}\\,\\omega\\,\\Im\\{f_L^H q_L\\} .

    When the whirl directions have been decoupled the selection is applied
    inside each family, which guarantees the same number of forward and
    backward waves at every frequency.
    """
    if families is not None:
        # Split by the labels that are actually present.  Using the fixed pair
        # ``(> 0, < 0)`` would silently discard every wave labelled exactly 0,
        # which is what the general (non-decoupled) path returns whenever
        # forward and backward whirl are degenerate — at zero spin speed, for
        # instance.  The waves so dropped would leave NaN rows in the result.
        labels = np.unique(families)
        masks = [families == label for label in labels]
        indices = np.concatenate(
            [
                np.flatnonzero(mask)[
                    _select_positive_going(
                        multipliers[mask],
                        displacements[:, mask],
                        forces[:, mask],
                        frequency_hz,
                        None,
                        tol,
                    )
                ]
                for mask in masks
            ]
        )
        return indices

    omega = hz_to_rad(frequency_hz)
    magnitude = np.abs(multipliers)
    decaying = magnitude < 1.0 - tol
    propagating = np.abs(magnitude - 1.0) <= tol

    power = -0.5 * omega * np.imag(np.sum(forces.conj() * displacements, axis=0))
    positive = decaying | (propagating & (power > 0.0))

    n_expected = multipliers.size // 2
    indices = np.flatnonzero(positive)
    if indices.size == n_expected:
        return indices
    # Fall back to the smallest magnitudes when the criterion is ambiguous
    # (numerically degenerate roots, e.g. exactly at a band edge).  The roots
    # of the pencil come in pairs (lambda, 1/lambda) — a wave and its mirror
    # image travelling the other way — so the half with |lambda| <= 1 is the
    # positive-going half whenever the power criterion cannot decide.  At a
    # band edge the two members of a pair coincide on the unit circle and the
    # choice is immaterial.
    return np.argsort(magnitude)[:n_expected]
