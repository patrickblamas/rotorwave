# From the paper's equations to the code

Equation numbers refer to

> P. B. Lamas, R. Nicoletti, *Wave analysis of rotors with longitudinal periodicity*,
> Journal of Sound and Vibration 571 (2024) 118095.

The sign convention used throughout the package is `M q̈ + (C − Ω G) q̇ + K q = f`
with an `e^{+iωt}` time dependence, and the nodal degrees of freedom are ordered
`[x, y, β, γ]`.

| Paper | Meaning | Code |
|---|---|---|
| Eq. (1) | equation of motion of the rotor | `RotorFEModel` (`model.py`) |
| Eq. (2) | dof vector | `DOF_NAMES`, `RotorFEModel.dof` |
| Eq. (3)–(4) | state-space eigenvalue problem | `waves._state_space_frequencies` |
| Eq. (5) | receptance matrix | `ReceptanceSolver.compute` |
| Eq. (6)–(7) | directional FRFs | `FRFResult.forward`, `FRFResult.backward` |
| Eq. (8) | dynamic stiffness of the cell | `RotorFEModel.dynamic_stiffness` |
| Eq. (9)–(11) | L/I/R partition | `UnitCell.partition_dynamic_stiffness` |
| Eq. (12)–(13) | Bloch–Floquet condition, `Λ_R` | `UnitCell.bloch_matrices` |
| Eq. (14)–(16) | force equilibrium, `Λ_L` | `UnitCell.bloch_matrices` |
| Eq. (17)–(18) | reduced dynamic stiffness | `UnitCell.reduced_bloch_matrices` |
| Eq. (19)–(23) | the k(ω) eigenvalue problem | `KOmegaSolver._solve_single` |
| Eq. (24) | wavenumber from the Bloch multiplier | `KOmegaSolver.solve` |
| Sec. 3 | numerical case study (11 disks) | `reference.reference_rotor` |
| Sec. 4 | laboratory rotor (3 disks) | `reference.test_rig_rotor` |

## Two differences worth knowing about

**The k(ω) problem is condensed before it is solved.** The paper keeps the internal
degrees of freedom in the eigenvector and obtains a pencil of size `n_L + n_I` that is
*linear* in λ (Eqs. 21–23). The code eliminates them first with a Schur complement, which
gives the smaller pencil

```
[λ² D̂_LR + λ (D̂_LL + D̂_RR) + D̂_RL] q_L = 0 ,     D̂ = condensed blocks
```

*quadratic* in λ and of size `n_L` only. The two formulations have the same non-trivial
roots; the condensed one avoids the zero and infinite eigenvalues that the larger pencil
carries, and is the standard wave-finite-element route (Mead; Mace *et al.*).

**The characteristic polynomial is expanded exactly.** Instead of linearising the
quadratic pencil into a companion form, `det(A λ² + B λ + C)` is expanded with
polynomial arithmetic and its roots are found from a balanced companion matrix
(`waves._matrix_polynomial_determinant`). At low frequency, where the cell is
stiffness-dominated and the pencil is badly conditioned, this is the difference between
`||λ| − 1| ≈ 10⁻³` and `≈ 10⁻⁸` for a propagating wave — and therefore between a
dispersion diagram with spurious narrow stop bands and a clean one.

## Whirl separation

For an isotropic rotor the generator of rotations about the spin axis,

```
J = diag_nodes( [[0,−1],[1,0]] ⊕ [[0,−1],[1,0]] ) ,
```

commutes with `M`, `G` and `K` (checked in `tests/test_waves.py`). Two consequences are
used throughout:

* the whirl index of a mode is the Rayleigh quotient of the Hermitian operator `−iJ`,
  equal to `+1` for a circular forward orbit and `−1` for a backward one
  (`whirl.whirl_index`);
* the eigenvectors of `J` split the model into a forward and a backward half
  (`whirl.whirl_basis`), so the k(ω) problem can be solved separately for each
  precession direction. This is the wave-analysis counterpart of the directional FRFs of
  Eqs. (6)–(7).

