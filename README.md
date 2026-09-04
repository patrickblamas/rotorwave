# rotorwave

[![tests](https://github.com/patrickblamas/rotorwave/actions/workflows/tests.yml/badge.svg)](https://github.com/patrickblamas/rotorwave/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
<!-- After archiving on Zenodo, paste the DOI badge here:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

**Wave propagation analysis of rotors with longitudinal periodicity, in Python.**

A rotor whose disks are equally spaced along the shaft is a periodic structure,
and like any periodic structure it has **band gaps**: frequency ranges in which
no wave can travel along it. `rotorwave` computes where those band gaps are, how
they move with the rotating speed, and how strongly the rotor rejects the
frequencies inside them.

It is the open-source implementation of

> P. B. Lamas, R. Nicoletti, **Wave analysis of rotors with longitudinal periodicity**,
> *Journal of Sound and Vibration* **571** (2024) 118095.
> [doi:10.1016/j.jsv.2023.118095](https://doi.org/10.1016/j.jsv.2023.118095)

and reproduces every figure of that paper from the scripts in `examples/`.

---

## Quick start

**No installation needed to try it.** Open `run_me.py` in Spyder (or any IDE
that comes with Anaconda) and press F5; or open `rotorwave_demo.ipynb` in
Jupyter. Both put `src/` on the Python path themselves, as do the scripts in
`examples/`.

To install the package so you can `import rotorwave` from your own scripts:

```bash
pip install -e ".[plot,dev]"     # from a clone of this repository
```

Requires Python ≥ 3.10, NumPy and SciPy; matplotlib is needed only for the
figures.

Windows/Anaconda users: step-by-step instructions in Portuguese, including the
usual error messages and what they mean, are in [INSTALACAO.md](INSTALACAO.md);
double-clicking `instalar_windows.bat` runs the installation and checks it.

Six lines are enough to get the band gap of the rotor of the paper:

```python
from rotorwave import OmegaKSolver, reference_rotor

rotor = reference_rotor(n_disks=11)                     # the paper's case study
dispersion = OmegaKSolver(rotor.unit_cell()).solve(spin_rpm=6000)

print(dispersion.summary("forward"))
# omega(k) dispersion at 6000 rpm — 8 branches, 41 wavenumbers, cell length 136.36 mm
#   forward  band gap:   2136.1 -   3679.8 Hz (width  1543.7 Hz,  53.1 %)
```

---

## The idea in one page

**The unit cell.** A rotor with equally spaced disks repeats one piece of
itself — a length of shaft plus one disk — over and over. That piece is the
*unit cell*. Bloch–Floquet theory says that everything about wave propagation
in the infinite repetition of a cell can be obtained from that single cell, by
imposing `q_right = λ q_left` with `λ = e^{−ikΔ}`, where `Δ` is the cell length.
So the whole analysis costs one small eigenvalue problem, not a model of the
entire rotor.

**Two ways to ask the question.** They are complementary and the package solves
both:

| | Prescribe | Obtain | Answers |
|---|---|---|---|
| **ω(k)** | a real wavenumber | real frequencies | *where* the band gaps are |
| **k(ω)** | a real frequency | complex wavenumbers | *how strongly* each frequency is attenuated |

The ω(k) problem gives clean band edges (they are branch extrema, so they do
not depend on any grid). The k(ω) problem gives the attenuation per cell, which
ω(k) cannot say, and it also flags frequencies that are strongly attenuated
without being strictly evanescent.

**Forward and backward whirl.** A rotor is not an ordinary periodic structure:
the gyroscopic effect splits every wave into a forward-whirling and a
backward-whirling one, and it does so differently at every speed. `rotorwave`
labels each wave by its own eigenvector (the *whirl index*, a Rayleigh quotient
of the rotation generator), so the labels remain correct at any speed and
across branch crossings. For an isotropic rotor it goes further and
*block-diagonalises* the problem into a forward half and a backward half, which
makes the separation exact rather than a post-processing step.

**The check that matters.** The dispersion describes an *infinite* periodic
medium. The `ReceptanceSolver` computes the frequency response of the *finite*
rotor, which is what an experiment measures. A predicted stop band is only
believable if the measured response collapses inside it — which is exactly what
Figure 4 of the paper shows, and what `examples/dispersion_omega_k.py`
reproduces.

More detail, with the equations, is in [docs/theory.md](docs/theory.md).

---

## What it computes

| Problem | Class | Given | Returns |
|---|---|---|---|
| ω(k) | `OmegaKSolver` | real wavenumber | propagating frequencies, mode shapes, whirl |
| k(ω) | `KOmegaSolver` | real frequency | complex wavenumbers (propagation **and** attenuation), whirl |
| receptance | `ReceptanceSolver` | frequency grid | FRFs and directional FRFs of the finite rotor |
| band gaps | `.band_gaps()`, `band_gap_map` | either dispersion | stop bands per precession direction, and their drift with speed |

The finite element model underneath is a Nelson–McVaugh rotating
Euler–Bernoulli shaft with rigid disks, four degrees of freedom per node
(`x, y, β, γ`), assembled as `M q̈ + (C − ΩG) q̇ + K q = f`.

---

## The examples

Each script has a parameter block at the top — geometry, material, mesh, speed —
so you can point it at a different rotor without reading the rest of the file.
Figures are written to `examples/figures/`.

**Reproducing the paper**

```bash
python examples/dispersion_omega_k.py          # Fig. 4: dispersion + receptance, 0 and 6000 rpm
python examples/dispersion_k_omega.py          # complex wavenumbers + receptance
python examples/directional_dispersion_dFRF.py # two-sided diagram + directional FRFs
python examples/wave_campbell.py               # wave-Campbell diagram
python examples/disk_count_sweep.py            # Fig. 3: modal and wave maps vs number of disks
```

**Going beyond it**

```bash
python examples/band_gap_map.py                # band gap edges vs rotating speed
```

**Your own rotor**

```bash
python examples/custom_rotor.py
```

Start from this one: the top of the file is the geometry, the material, the
mesh density and the speed; the bottom shows how to build a non-uniform cell
element by element, for a rotor that is not a shaft with identical disks.

---

## Validation

Reproduced by the test suite, for the 1500 mm × 100 mm steel shaft carrying 11
disks of 380 mm × 22 mm:

| Quantity | `rotorwave` | Paper |
|---|---|---|
| First band gap at rest | 2037 – 3579 Hz | "between 2000 and 3500 Hz" |
| Forward gap at 6000 rpm | 2136 – 3680 Hz | shifts up |
| Backward gap at 6000 rpm | 1943 – 3482 Hz | shifts down |
| Gap width, 0 → 6000 rpm | 1542 → 1544 Hz (fw), 1539 Hz (bw) | width preserved |
| Peak attenuation across the rotor | ≈ 166 dB | — |

The gap **moves** with speed but keeps its width — the quantitative version of
a statement the paper makes qualitatively, now measured by `band_gap_map`.

The suite also checks the model against things that are known independently:
the analytical natural frequencies of a simply supported beam, the exact
integration of the element matrices, the isotropy of the assembled rotor, mesh
convergence, the agreement between the ω(k) and k(ω) formulations, and the
agreement between the whirl-decoupled solver and the general one.

```bash
pytest              # 34 tests, about a minute
```

---

## What changed with respect to the reference MATLAB scripts

The physics is the same; the numerics and the interfaces are not.

**Correctness**

* **Whirl classification from the eigenvectors.** Forward and backward branches
  were identified by sorting frequencies and picking every other one, which
  breaks at zero spin speed, near crossings and whenever the assumed ordering
  does not hold. Here the whirl direction is the Rayleigh quotient of the
  rotation generator `J` (`rotorwave.whirl`), so it is a property of the mode,
  valid at any speed.
* **Exact forward/backward decoupling.** `J` commutes with the matrices of an
  isotropic rotor, so the k(ω) problem is block-diagonalised into a forward and
  a backward half. Every wave is labelled by construction, and near-degenerate
  pairs — which a general eigensolver returns hopelessly mixed — come out
  clean. The commutation is *checked*, not assumed, with a fallback for the
  general case; the two paths are tested against each other.
* **Branch tracking.** Sorting by frequency swaps branches at every veering.
  Branches are followed with a mass-weighted MAC plus frequency extrapolation,
  matched with the Hungarian algorithm; the k(ω) waves are ordered by
  continuity of their Bloch multipliers, which is what removes the spurious
  vertical strokes from the diagrams.
* **Accurate Bloch multipliers.** The quadratic eigenvalue problem is solved
  from the exactly expanded characteristic polynomial instead of a companion
  linearisation. Measured on the reference rotor between 10 Hz and 1 kHz — the
  stiff, ill-conditioned end — the propagating waves satisfy `||λ|−1| ≤ 6e-9`
  against `2e-3` for the linearisation: the difference between a clean
  dispersion diagram and one peppered with spurious 10 Hz stop bands.
* **Guarded fast paths.** The Hermitian eigensolver is used only when the
  modes it returns are verified to satisfy the original quadratic problem; at
  `k = 0` the reduced stiffness is numerically singular (condition number
  around 1e17), so a factorisation test alone would pass or fail by rounding
  luck. The rigid-body modes then come out as exactly zero at every mesh
  density, instead of a spurious few thousandths of a hertz.
* **A documented dFRF sign convention**, verified in the test suite against the
  forward branches of the wave analysis rather than assumed.

**Speed**

* The ω(k) problem is posed as a *Hermitian* eigenvalue problem
  (`K + ωΩH − ω²M` with `H = −iG`), which gives real eigenvalues by
  construction and lets only the requested low modes be computed — roughly an
  order of magnitude faster than the state-space form, with a state-space
  fallback where `K` is singular (k = 0).
* The receptance uses a banded factorisation: `O(n b²)` per frequency instead
  of `O(n³)`, so a 1300-point sweep on a 1764-dof rotor takes well under a
  second.

**Usability**

* Object model — `Material`, `ShaftElement`, `RigidDisk`, `RotorFEModel`,
  `UnitCell`, `PeriodicRotor` — instead of matrices passed positionally between
  scripts. Hollow shafts, arbitrary disk positions, elastic or rigid supports
  are parameters now.
* One `PeriodicRotor` builds both the unit cell and the full rotor, so the wave
  analysis and the receptance can never drift apart.
* Results are objects with methods (`band_gaps`, `summary`,
  `attenuation_db_per_cell`, `to_dict`) rather than loose arrays.
* Units are explicit in every signature (`spin_rpm`, `frequencies_hz`); SI
  internally.

---

## Layout

```
run_me.py                                      # runnable demo, no installation needed
rotorwave_demo.ipynb                           # the same, as a notebook
INSTALACAO.md                                  # setup guide (Windows/Anaconda, in Portuguese)
docs/theory.md                                 # the equations behind the code
src/rotorwave/
    materials.py   elements.py   model.py      # finite element model
    waves.py       dispersion.py whirl.py      # wave analysis and results
    frf.py         analysis.py   plotting.py   # receptance, higher level studies, figures
examples/                                      # the figures of the paper, and two new ones
tests/                                         # pytest suite
```

---

## Limitations

* The Bloch–Floquet condition describes an **infinite** periodic medium built
  from one cell, so the dispersion results assume isotropic supports and carry
  no information about the boundaries — the same limitation stated in the
  paper. Finite-rotor effects (boundary conditions, support anisotropy,
  localized modes) have to be studied with the `ReceptanceSolver` on the full
  model, which the examples do side by side with the dispersion.
* The shaft element is Euler–Bernoulli. Shear deformation matters for short,
  thick shafts and for the high branches of a fine cell; check with
  `mesh_convergence` before trusting a branch near the top of the computed
  range.
* Band edges taken from the k(ω) problem are resolved only to the frequency
  step of the grid. Use the ω(k) problem when the edge frequency itself is the
  answer.
* Damping is not part of the wave analysis: the band gaps computed here are the
  undamped ones. `ReceptanceSolver` accepts a damping matrix for the response.

---

## Citing

If this code contributes to your work, please cite the original paper (above)
and this software:

```bibtex
@software{rotorwave,
  author  = {Lamas, P. B. and Nicoletti, R.},
  title   = {rotorwave: wave propagation analysis of rotors with longitudinal periodicity},
  year    = {2026},
  url     = {https://github.com/patrickblamas/rotorwave}
}
```

`CITATION.cff` in the repository root keeps this up to date; GitHub turns it
into a ready-made citation under *"Cite this repository"*.

## License

MIT — see [LICENSE](LICENSE). You may use, modify and redistribute the code,
including commercially, provided the copyright notice is kept.

## Contributing

Bug reports, questions and pull requests are welcome through the
[issue tracker](https://github.com/patrickblamas/rotorwave/issues). If you find
a case where the results disagree with an independent calculation, please open
an issue with the geometry — that is the most useful contribution of all.
