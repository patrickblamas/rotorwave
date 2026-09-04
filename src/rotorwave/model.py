"""Finite element models of shafts, unit cells and full periodic rotors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .elements import DOF_PER_NODE, RigidDisk, ShaftElement
from .materials import Material
from .units import hz_to_rad, rpm_to_rad

__all__ = ["RotorFEModel", "UnitCell", "PeriodicRotor"]


@dataclass
class RotorFEModel:
    """Assembled finite element model of a rotating shaft.

    The equation of motion follows the usual rotordynamic convention

    .. math::

        M\\,\\ddot q + (C - \\Omega G)\\,\\dot q + K\\,q = f ,

    so that the dynamic stiffness matrix reads
    :math:`D(\\omega, \\Omega) = K + i\\omega C - i\\omega\\Omega G - \\omega^2 M`.

    Parameters
    ----------
    elements:
        Shaft elements, ordered from the first to the last node.
    disks:
        Mapping ``node index -> RigidDisk``.  Node indices are zero based.
    bearings:
        Mapping ``node index -> (kxx, kyy)`` of isotropic/orthotropic support
        stiffnesses [N/m].  Use :meth:`constrain` for rigid supports instead.
    """

    elements: Sequence[ShaftElement]
    disks: Mapping[int, RigidDisk] = field(default_factory=dict)
    bearings: Mapping[int, tuple[float, float]] = field(default_factory=dict)
    constrained_dofs: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.elements) == 0:
            raise ValueError("the model needs at least one shaft element")
        for node in self.disks:
            if not 0 <= node < self.n_nodes:
                raise IndexError(f"disk node {node} outside the mesh")
        for node in self.bearings:
            if not 0 <= node < self.n_nodes:
                raise IndexError(f"bearing node {node} outside the mesh")

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------
    @property
    def n_elements(self) -> int:
        """Number of shaft elements in the model."""
        return len(self.elements)

    @property
    def n_nodes(self) -> int:
        """Number of nodes: the elements are in a chain, so ``n_elements + 1``."""
        return len(self.elements) + 1

    @property
    def n_dofs(self) -> int:
        """Total number of degrees of freedom, four per node."""
        return self.n_nodes * DOF_PER_NODE

    @cached_property
    def node_positions(self) -> np.ndarray:
        """Axial coordinate of every node [m]."""
        lengths = np.array([el.length for el in self.elements])
        return np.concatenate([[0.0], np.cumsum(lengths)])

    @property
    def length(self) -> float:
        """Total axial length of the model [m]."""
        return float(self.node_positions[-1])

    def dof(self, node: int, component: int | str) -> int:
        """Global index of a nodal degree of freedom.

        ``component`` may be an integer in ``0..3`` or one of
        ``'x'``, ``'y'``, ``'beta'``, ``'gamma'``.
        """
        from .elements import DOF_NAMES

        if isinstance(component, str):
            component = DOF_NAMES.index(component)
        if not 0 <= component < DOF_PER_NODE:
            raise IndexError("component out of range")
        node = range(self.n_nodes)[node]  # supports negative indices
        return node * DOF_PER_NODE + component

    # ------------------------------------------------------------------
    # Global matrices
    # ------------------------------------------------------------------
    @cached_property
    def _assembled(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.n_dofs
        M = np.zeros((n, n))
        G = np.zeros((n, n))
        K = np.zeros((n, n))
        for i, element in enumerate(self.elements):
            sl = slice(i * DOF_PER_NODE, i * DOF_PER_NODE + 2 * DOF_PER_NODE)
            M[sl, sl] += element.mass_matrix()
            G[sl, sl] += element.gyroscopic_matrix()
            K[sl, sl] += element.stiffness_matrix()
        for node, disk in self.disks.items():
            sl = slice(node * DOF_PER_NODE, (node + 1) * DOF_PER_NODE)
            M[sl, sl] += disk.mass_matrix()
            G[sl, sl] += disk.gyroscopic_matrix()
        for node, (kxx, kyy) in self.bearings.items():
            K[self.dof(node, "x"), self.dof(node, "x")] += kxx
            K[self.dof(node, "y"), self.dof(node, "y")] += kyy
        return M, G, K

    @property
    def M(self) -> np.ndarray:
        """Assembled inertia matrix, ``(n_dofs, n_dofs)`` [kg, kg.m^2].

        The **unconstrained** matrix, with the dof ordering
        ``[x, y, beta, gamma]`` node by node.  Use :meth:`reduced_matrices`
        for the version with the constrained dofs removed.
        """
        return self._assembled[0]

    @property
    def G(self) -> np.ndarray:
        """Assembled gyroscopic matrix, ``(n_dofs, n_dofs)``.

        Skew-symmetric, and multiplied by the spin speed in the equation of
        motion; unconstrained, like :attr:`M`.
        """
        return self._assembled[1]

    @property
    def K(self) -> np.ndarray:
        """Assembled stiffness matrix, ``(n_dofs, n_dofs)`` [N/m, N.m/rad].

        Unconstrained, like :attr:`M`.
        """
        return self._assembled[2]

    @cached_property
    def free_dofs(self) -> np.ndarray:
        """Indices of the degrees of freedom that are not constrained."""
        mask = np.ones(self.n_dofs, dtype=bool)
        mask[list(self.constrained_dofs)] = False
        return np.flatnonzero(mask)

    def constrain(self, dofs: Iterable[int]) -> RotorFEModel:
        """Return a copy of the model with additional constrained dofs.

        The copy keeps the concrete type of the original, so constraining a
        :class:`UnitCell` returns a :class:`UnitCell` and not a plain model.
        """
        merged = tuple(sorted(set(self.constrained_dofs) | set(int(d) for d in dofs)))
        return type(self)(
            elements=self.elements,
            disks=dict(self.disks),
            bearings=dict(self.bearings),
            constrained_dofs=merged,
        )

    def reduced_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(M, G, K)`` with the constrained degrees of freedom removed."""
        idx = self.free_dofs
        M, G, K = self._assembled
        return M[np.ix_(idx, idx)], G[np.ix_(idx, idx)], K[np.ix_(idx, idx)]

    def dynamic_stiffness(
        self,
        frequency_hz: float,
        spin_rpm: float = 0.0,
        damping: np.ndarray | None = None,
        reduced: bool = True,
    ) -> np.ndarray:
        """Dynamic stiffness :math:`D(\\omega,\\Omega)` at one frequency."""
        omega = hz_to_rad(frequency_hz)
        spin = rpm_to_rad(spin_rpm)
        M, G, K = self.reduced_matrices() if reduced else self._assembled
        D = K.astype(complex) - omega**2 * M - 1j * omega * spin * G
        if damping is not None:
            D = D + 1j * omega * damping
        return D

    # ------------------------------------------------------------------
    # Modal analysis
    # ------------------------------------------------------------------
    def natural_frequencies(
        self, spin_rpm: float = 0.0, n_modes: int | None = None
    ) -> np.ndarray:
        """Natural frequencies [Hz] of the (constrained) model.

        Returned in ascending order.  Each whirl mode appears once, so a
        forward/backward pair shows up as two entries.

        Parameters
        ----------
        spin_rpm:
            Spin speed [rpm]; the gyroscopic effect splits each pair.
        n_modes:
            Number of modes to compute, lowest first.  Asking only for the
            modes that are needed is much cheaper on a fine mesh, because the
            Hermitian solver can then work on a subset of the spectrum.
        """
        from .waves import _hermitian_frequencies

        M, G, K = self.reduced_matrices()
        spin = rpm_to_rad(spin_rpm)
        freqs, _ = _hermitian_frequencies(M, G, K, spin, n_modes=n_modes)
        return freqs if n_modes is None else freqs[:n_modes]

    def natural_frequencies_below(
        self, frequency_hz: float, spin_rpm: float = 0.0, chunk: int = 32
    ) -> np.ndarray:
        """Every natural frequency below a limit, computed a chunk at a time.

        Convenient when the number of modes in a frequency band is not known in
        advance, as in a sweep over the number of working elements.
        """
        n_modes = chunk
        while True:
            frequencies = self.natural_frequencies(spin_rpm, n_modes=n_modes)
            if frequencies.size < n_modes or frequencies[-1] > frequency_hz:
                return frequencies[frequencies <= frequency_hz]
            n_modes *= 2


@dataclass
class UnitCell(RotorFEModel):
    """A unit cell of a longitudinally periodic rotor.

    The cell is a standard finite element model whose *first* node is the left
    boundary (L), whose *last* node is the right boundary (R), and whose
    remaining nodes are internal (I).  Periodicity is imposed through the
    Bloch-Floquet condition :math:`q_R = \\lambda\\, q_L`, with
    :math:`\\lambda = e^{-i k \\Delta}` and :math:`\\Delta` the cell length.
    """

    @property
    def cell_length(self) -> float:
        """Cell length :math:`\\Delta` [m]."""
        return self.length

    @property
    def n_boundary_dofs(self) -> int:
        return DOF_PER_NODE

    @property
    def n_internal_dofs(self) -> int:
        return self.n_dofs - 2 * DOF_PER_NODE

    @property
    def left_dofs(self) -> np.ndarray:
        return np.arange(DOF_PER_NODE)

    @property
    def internal_dofs(self) -> np.ndarray:
        return np.arange(DOF_PER_NODE, self.n_dofs - DOF_PER_NODE)

    @property
    def right_dofs(self) -> np.ndarray:
        return np.arange(self.n_dofs - DOF_PER_NODE, self.n_dofs)

    # ------------------------------------------------------------------
    # Bloch-Floquet reduction
    # ------------------------------------------------------------------
    def bloch_matrices(self, lam: complex) -> tuple[np.ndarray, np.ndarray]:
        """Return the projection matrices :math:`\\Lambda_L, \\Lambda_R`.

        With ``q_c = Lambda_R @ [q_L; q_I]`` (kinematic periodicity) and
        ``Lambda_L @ f_c = 0`` (force equilibrium at the shared boundary),
        the reduced cell matrices are ``Lambda_L @ X @ Lambda_R``.

        For a real wavenumber :math:`|\\lambda| = 1` and
        :math:`\\Lambda_L = \\Lambda_R^H`, so the reduced matrices keep the
        Hermitian/skew-Hermitian structure of the original ones.
        """
        nb, ni = self.n_boundary_dofs, self.n_internal_dofs
        eye_b, eye_i = np.eye(nb), np.eye(ni)
        zeros_bi = np.zeros((nb, ni))

        lambda_r = np.block(
            [
                [eye_b, zeros_bi],
                [zeros_bi.T, eye_i],
                [lam * eye_b, zeros_bi],
            ]
        )
        lambda_l = np.block(
            [
                [eye_b, zeros_bi, eye_b / lam],
                [zeros_bi.T, eye_i, np.zeros((ni, nb))],
            ]
        )
        return lambda_l, lambda_r

    def reduced_bloch_matrices(
        self, lam: complex
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(M_bar, G_bar, K_bar)`` of the cell for a Bloch multiplier."""
        lambda_l, lambda_r = self.bloch_matrices(lam)
        M, G, K = self._assembled
        return (
            lambda_l @ M @ lambda_r,
            lambda_l @ G @ lambda_r,
            lambda_l @ K @ lambda_r,
        )

    def partition_dynamic_stiffness(
        self, frequency_hz: float, spin_rpm: float = 0.0
    ) -> dict[str, np.ndarray]:
        """Blocks of :math:`D` partitioned into L/I/R degrees of freedom."""
        D = self.dynamic_stiffness(frequency_hz, spin_rpm, reduced=False)
        left, internal, right = self.left_dofs, self.internal_dofs, self.right_dofs
        return {
            "LL": D[np.ix_(left, left)],
            "LI": D[np.ix_(left, internal)],
            "LR": D[np.ix_(left, right)],
            "IL": D[np.ix_(internal, left)],
            "II": D[np.ix_(internal, internal)],
            "IR": D[np.ix_(internal, right)],
            "RL": D[np.ix_(right, left)],
            "RI": D[np.ix_(right, internal)],
            "RR": D[np.ix_(right, right)],
        }


@dataclass(frozen=True)
class PeriodicRotor:
    """Specification of a rotor with longitudinal periodicity.

    A rotor made of ``n_cells`` identical cells, each one carrying a rigid
    disk at a relative position ``disk_position`` along the cell.  The object
    builds both the *unit cell* used by the wave analysis and the *full rotor*
    used by the receptance/dFRF analysis, guaranteeing that both share exactly
    the same discretisation.

    Parameters
    ----------
    shaft_length:
        Total shaft length :math:`L` [m].
    shaft_diameter:
        Shaft outer diameter [m].
    n_cells:
        Number of unit cells, i.e. number of disks.
    disk_diameter, disk_thickness:
        Disk geometry [m].
    material:
        Shaft and disk material.
    elements_per_cell:
        Number of finite elements used to mesh one cell.
    disk_position:
        Relative position of the disk inside the cell, in ``[0, 1]``.
        ``0.5`` puts the disk at the centre of the cell.
    shaft_inner_diameter:
        Inner diameter for hollow shafts [m].
    """

    shaft_length: float
    shaft_diameter: float
    n_cells: int
    disk_diameter: float
    disk_thickness: float
    material: Material
    elements_per_cell: int = 40
    disk_position: float = 0.5
    shaft_inner_diameter: float = 0.0

    def __post_init__(self) -> None:
        if self.n_cells < 1:
            raise ValueError("n_cells must be at least 1")
        if self.elements_per_cell < 2:
            raise ValueError("elements_per_cell must be at least 2")
        if not 0.0 <= self.disk_position <= 1.0:
            raise ValueError("disk_position must lie in [0, 1]")

    @property
    def cell_length(self) -> float:
        """Unit cell length :math:`\\Delta = L / N` [m]."""
        return self.shaft_length / self.n_cells

    @property
    def disk(self) -> RigidDisk:
        """The rigid disk mounted on every cell."""
        return RigidDisk.from_geometry(
            diameter=self.disk_diameter,
            thickness=self.disk_thickness,
            material=self.material,
        )

    @property
    def disk_node_in_cell(self) -> int:
        """Index of the cell node that carries the disk."""
        return int(round(self.disk_position * self.elements_per_cell))

    def _shaft_elements(self, n_elements: int, element_length: float) -> list[ShaftElement]:
        return [
            ShaftElement(
                length=element_length,
                outer_diameter=self.shaft_diameter,
                material=self.material,
                inner_diameter=self.shaft_inner_diameter,
            )
            for _ in range(n_elements)
        ]

    def unit_cell(self) -> UnitCell:
        """Finite element model of a single cell (no boundary conditions)."""
        le = self.cell_length / self.elements_per_cell
        return UnitCell(
            elements=self._shaft_elements(self.elements_per_cell, le),
            disks={self.disk_node_in_cell: self.disk},
        )

    def full_rotor(
        self,
        elements_per_cell: int | None = None,
        supports: str = "simply-supported",
        bearing_stiffness: float | None = None,
    ) -> RotorFEModel:
        """Finite element model of the complete rotor with ``n_cells`` cells.

        Parameters
        ----------
        elements_per_cell:
            Overrides the cell mesh density (the full rotor may be coarser
            than the unit cell used for the wave analysis).
        supports:
            ``'simply-supported'`` constrains the lateral translations of the
            first and last nodes, matching the reference implementation;
            ``'free'`` leaves the rotor unconstrained; ``'elastic'`` adds
            isotropic bearings of stiffness ``bearing_stiffness``.
        bearing_stiffness:
            Support stiffness [N/m], only used when ``supports='elastic'``.
        """
        npc = elements_per_cell or self.elements_per_cell
        n_elements = npc * self.n_cells
        le = self.cell_length / npc
        disk_offset = int(round(self.disk_position * npc))
        # One disk per cell, at the same relative position inside each.  Note
        # that ``disk_position`` is measured from the left end of the cell, so
        # 0.0 places every disk on a cell boundary and 0.5 at mid-cell; the
        # node indices stay distinct either way, one per cell.
        disks = {
            cell * npc + disk_offset: self.disk for cell in range(self.n_cells)
        }
        model = RotorFEModel(
            elements=self._shaft_elements(n_elements, le),
            disks=disks,
        )
        if supports == "simply-supported":
            constrained = [
                model.dof(0, "x"),
                model.dof(0, "y"),
                model.dof(-1, "x"),
                model.dof(-1, "y"),
            ]
            return model.constrain(constrained)
        if supports == "elastic":
            if bearing_stiffness is None:
                raise ValueError("bearing_stiffness is required for elastic supports")
            return RotorFEModel(
                elements=model.elements,
                disks=disks,
                bearings={
                    0: (bearing_stiffness, bearing_stiffness),
                    model.n_nodes - 1: (bearing_stiffness, bearing_stiffness),
                },
            )
        if supports == "free":
            return model
        raise ValueError(f"unknown support type: {supports!r}")

    # Convenience -------------------------------------------------------
    def summary(self) -> str:
        """One-paragraph description of the rotor, handy for logs and papers."""
        disk = self.disk
        return (
            f"Periodic rotor: L = {self.shaft_length * 1e3:.0f} mm, "
            f"d_shaft = {self.shaft_diameter * 1e3:.0f} mm, "
            f"{self.n_cells} cells of {self.cell_length * 1e3:.1f} mm, "
            f"disk {self.disk_diameter * 1e3:.0f} x {self.disk_thickness * 1e3:.0f} mm "
            f"(m = {disk.mass:.2f} kg, Ip = {disk.polar_inertia:.4f} kg.m^2), "
            f"{self.elements_per_cell} elements/cell."
        )
