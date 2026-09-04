"""Receptance and directional frequency response functions of a finite rotor.

The wave analysis works on an infinite periodic medium; the receptance of the
*finite* rotor is what an experiment actually measures, and comparing the two
is how a predicted band gap is confirmed — a stop band must show up as a wide
frequency range with no resonance peaks and a very low response level.

Directional FRFs (Lee, 1991) split that response into its forward and backward
whirl contributions,

.. math::

    H_{pg}     = \\tfrac{1}{2}[H_{xx} + H_{yy} + i(H_{yx} - H_{xy})], \\qquad
    H_{p^*g^*} = \\tfrac{1}{2}[H_{xx} + H_{yy} - i(H_{yx} - H_{xy})],

which is exactly the separation the wave analysis performs on the dispersion
branches, so the two can be overlaid on the same axes.

The sign of the cross term is tied to the time convention: these expressions
are the ones consistent with :math:`e^{+i\\omega t}`, used throughout the
package.  Under the conjugate convention :math:`e^{-i\\omega t}`, common in
part of the literature, the two signs swap and the labels forward/backward
would be exchanged.  The convention used here is not assumed: the test suite
checks that the peaks of :math:`H_{pg}` fall on the forward branches of the
wave analysis, which are labelled independently by the whirl index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .model import RotorFEModel
from .units import hz_to_rad, rpm_to_rad

__all__ = ["ReceptanceSolver", "FRFResult"]


@dataclass
class FRFResult:
    """Receptance matrix of a rotor between two nodes, over a frequency grid.

    Attributes
    ----------
    frequencies_hz:
        Frequency grid [Hz].
    h_xx, h_xy, h_yx, h_yy:
        Complex receptances [m/N]; the first index is the response direction
        and the second the excitation direction.
    spin_rpm:
        Spin speed of the analysis [rpm].
    """

    frequencies_hz: np.ndarray
    h_xx: np.ndarray
    h_xy: np.ndarray
    h_yx: np.ndarray
    h_yy: np.ndarray
    spin_rpm: float = 0.0

    @property
    def forward(self) -> np.ndarray:
        """Forward whirl dFRF :math:`H_{pg}`.

        Obtained from the complex coordinates :math:`p = x + iy` and
        :math:`g = f_x + i f_y`.  The sign of the cross term follows the
        :math:`e^{+i\\omega t}` convention used throughout the package (the
        opposite sign found in part of the literature corresponds to the
        conjugate convention); it is verified in the test suite against the
        forward branches of the wave analysis, which are unambiguous.
        """
        return 0.5 * (self.h_xx + self.h_yy + 1j * (self.h_yx - self.h_xy))

    @property
    def backward(self) -> np.ndarray:
        """Backward whirl dFRF :math:`H_{p^*g^*}`."""
        return 0.5 * (self.h_xx + self.h_yy - 1j * (self.h_yx - self.h_xy))

    @property
    def direct(self) -> np.ndarray:
        """Direct receptance :math:`H_{xx}`, the classical FRF."""
        return self.h_xx

    def magnitude_db(self, which: str = "direct", reference: float = 1.0) -> np.ndarray:
        """Magnitude in dB of one of the response functions."""
        data = {
            "direct": self.direct,
            "forward": self.forward,
            "backward": self.backward,
        }[which]
        return 20.0 * np.log10(np.abs(data) / reference)

    def resonances(self, which: str = "direct", min_prominence_db: float = 6.0):
        """Frequencies of the response peaks [Hz], a cheap modal identification."""
        from scipy.signal import find_peaks

        magnitude = self.magnitude_db(which)
        peaks, _ = find_peaks(magnitude, prominence=min_prominence_db)
        return self.frequencies_hz[peaks]

    def quiet_bands(
        self, which: str = "direct", threshold_db: float | None = None
    ) -> list[tuple[float, float]]:
        """Frequency ranges where the response stays below a threshold.

        With the default threshold (20 dB below the median response) these
        ranges are the experimental signature of a band gap.
        """
        magnitude = self.magnitude_db(which)
        if threshold_db is None:
            threshold_db = float(np.median(magnitude)) - 20.0
        quiet = magnitude < threshold_db
        bands: list[tuple[float, float]] = []
        start = None
        for frequency, flag in zip(self.frequencies_hz, quiet):
            if flag and start is None:
                start = float(frequency)
            elif not flag and start is not None:
                bands.append((start, float(frequency)))
                start = None
        if start is not None:
            bands.append((start, float(self.frequencies_hz[-1])))
        return bands


@dataclass
class ReceptanceSolver:
    """Frequency response of a finite rotor model.

    Parameters
    ----------
    model:
        The (constrained) finite element model of the complete rotor.
    excitation_node:
        Node where the unit force is applied.  ``None`` selects the first
        unconstrained node, as in the reference implementation.
    response_node:
        Node where the response is measured.  ``None`` selects the last
        unconstrained node, i.e. the opposite side of the shaft.
    alpha, beta:
        Optional proportional damping, :math:`C = \\alpha M + \\beta K`.

    Notes
    -----
    The dynamic stiffness of a beam model is banded, so each frequency is
    solved with a banded LU factorisation instead of a dense one.  The cost
    per frequency drops from :math:`O(n^3)` to :math:`O(n b^2)`, which is what
    makes a fine frequency sweep on a fine mesh practical.
    """

    model: RotorFEModel
    excitation_node: int | None = None
    response_node: int | None = None
    alpha: float = 0.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        free = self.model.free_dofs
        self._free = free
        self._position = -np.ones(self.model.n_dofs, dtype=int)
        self._position[free] = np.arange(free.size)

        # Excite and measure at the first/last nodes whose lateral translations
        # are free, i.e. right next to the bearings, as in the reference work.
        lateral_free = [
            node
            for node in range(self.model.n_nodes)
            if self._position[self.model.dof(node, "x")] >= 0
            and self._position[self.model.dof(node, "y")] >= 0
        ]
        if not lateral_free:
            raise ValueError("every lateral degree of freedom of the model is constrained")
        if self.excitation_node is None:
            self.excitation_node = lateral_free[0]
        if self.response_node is None:
            self.response_node = lateral_free[-1]

        m, g, k = self.model.reduced_matrices()
        # One bandwidth for all three matrices.  It is taken from K, which for
        # a chain of shaft elements with nodal disks is the widest of the
        # three: M and G couple the same node pairs as K, never more.  Taking
        # the maximum keeps that true for any future element whose mass matrix
        # reaches further.
        self._bandwidth = max(
            (_bandwidth(matrix) for matrix in (k, m, g) if matrix.size), default=0
        )
        self._m_band = _to_banded(m, self._bandwidth)
        self._g_band = _to_banded(g, self._bandwidth)
        self._k_band = _to_banded(k, self._bandwidth)

    # ------------------------------------------------------------------
    def _reduced_index(self, node: int, component: str) -> int:
        index = self._position[self.model.dof(node, component)]
        if index < 0:
            raise ValueError(
                f"dof {component!r} of node {node} is constrained; "
                "choose an unconstrained node for the FRF"
            )
        return int(index)

    def compute(
        self, frequencies_hz: np.ndarray, spin_rpm: float = 0.0
    ) -> FRFResult:
        """Receptances between the excitation and response nodes."""
        frequencies_hz = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
        spin = rpm_to_rad(spin_rpm)
        bw = self._bandwidth
        n = self._free.size

        exc = [self._reduced_index(self.excitation_node, c) for c in ("x", "y")]
        res = [self._reduced_index(self.response_node, c) for c in ("x", "y")]

        h = np.zeros((2, 2, frequencies_hz.size), dtype=complex)
        rhs = np.zeros((n, 2), dtype=complex)
        rhs[exc[0], 0] = 1.0
        rhs[exc[1], 1] = 1.0

        damping_band = self.alpha * self._m_band + self.beta * self._k_band
        for j, frequency in enumerate(frequencies_hz):
            omega = hz_to_rad(frequency)
            d_band = (
                self._k_band
                - omega**2 * self._m_band
                - 1j * omega * spin * self._g_band
                + 1j * omega * damping_band
            )
            solution = solve_banded((bw, bw), d_band, rhs)
            for a in range(2):
                for b in range(2):
                    h[a, b, j] = solution[res[a], b]

        return FRFResult(
            frequencies_hz=frequencies_hz,
            h_xx=h[0, 0],
            h_xy=h[0, 1],
            h_yx=h[1, 0],
            h_yy=h[1, 1],
            spin_rpm=float(spin_rpm),
        )


# ----------------------------------------------------------------------
def _bandwidth(matrix: np.ndarray) -> int:
    """Half bandwidth of a symmetric sparse-in-structure matrix."""
    rows, cols = np.nonzero(matrix)
    return int(np.max(np.abs(rows - cols))) if rows.size else 0


def _to_banded(matrix: np.ndarray, bandwidth: int) -> np.ndarray:
    """Convert a dense banded matrix to LAPACK banded storage."""
    n = matrix.shape[0]
    banded = np.zeros((2 * bandwidth + 1, n), dtype=complex)
    for offset in range(-bandwidth, bandwidth + 1):
        diagonal = np.diagonal(matrix, offset=offset)
        row = bandwidth - offset
        if offset >= 0:
            banded[row, offset : offset + diagonal.size] = diagonal
        else:
            banded[row, : diagonal.size] = diagonal
    return banded
