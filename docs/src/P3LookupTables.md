# P3 lookup tables

The `P3Scheme` module precomputes lookup tables for the ice-phase rate quantities that are otherwise evaluated by quadrature every grid cell.
This page describes the four clean tabulations, the normalized inputs they use, the grid and range choices, the interpolation scheme, and the build-at-init and parameter-invalidation contract.
The quadrature implementations remain the single source of truth; the tables are filled from them at a high order and are validated against them in `test/p3_lookup_table_tests.jl`.

## Tabulated quantities

Phase 1 tabulates the four ice-phase quantities that are exactly separable, that is, whose integrals depend on the state only through a small set of shape coordinates, with the remaining dependence a closed-form analytic prefactor.

| Quantity | Prefactor | Shape coordinates |
|:---------|:----------|:------------------|
| `ice_self_collection` `dNdt` | ``N_\mathrm{ice}^2`` | ``\log\lambda,\ F_\mathrm{rim},\ \rho_\mathrm{rim},\ \rho_\mathrm{air}`` |
| `ice_terminal_velocity_number_weighted` | ``1`` | ``\log\lambda,\ F_\mathrm{rim},\ \rho_\mathrm{rim},\ \rho_\mathrm{air}`` |
| `ice_terminal_velocity_mass_weighted` | ``1`` | ``\log\lambda,\ F_\mathrm{rim},\ \rho_\mathrm{rim},\ \rho_\mathrm{air}`` |
| `ice_melt` `dLdt` | ``N_\mathrm{ice}\, \tfrac{4 K_\mathrm{therm}}{L_f}(T_a - T_\mathrm{freeze})`` | ``\log\lambda,\ F_\mathrm{rim},\ \rho_\mathrm{rim},\ \rho_\mathrm{air}`` |

The ice size distribution is ``N'(D) = N_0 D^\mu e^{-\lambda D}`` with ``(\lambda, \mu)`` fixed by ``(\log\lambda, F_\mathrm{rim}, \rho_\mathrm{rim})`` and ``N_0 \propto N_\mathrm{ice}``.
The self-collection rate is a double integral of the collision kernel against ``N' N'``, so it scales as ``N_\mathrm{ice}^2`` times a shape function.
The terminal velocities are ratios of size-distribution moments, so the number scale cancels.
The melt rate factors the entire temperature dependence into the closed-form scalar ``\tfrac{4 K_\mathrm{therm}}{L_f(T_a)}(T_a - T_\mathrm{freeze})``, leaving a ventilation integral proportional to ``N_\mathrm{ice}``.

The ventilation factor is ``F_v(D) = a_v + b_v\, \mathrm{Sc}^{1/3}\, \mathrm{Re}(D)^{1/2}`` with ``\mathrm{Re}(D) = D\, v(D) / \nu_\mathrm{air}``.
The melt integral is stored as two moments,

```math
M_a = \int \frac{\partial m}{\partial D}\, \frac{N'}{N_\mathrm{ice}}\, \frac{\mathrm{d}D}{D},
\qquad
M_b = \int \frac{\partial m}{\partial D}\, \sqrt{D\, v(D)}\, \frac{N'}{N_\mathrm{ice}}\, \frac{\mathrm{d}D}{D},
```

so the melt rate is reconstructed at the use site as

```math
\frac{\mathrm{d}L}{\mathrm{d}t} = \frac{4 K_\mathrm{therm}}{L_f}(T_a - T_\mathrm{freeze})\, N_\mathrm{ice}
\left( a_v M_a + \frac{b_v\, \mathrm{Sc}^{1/3}}{\sqrt{\nu_\mathrm{air}}}\, M_b \right).
```

Splitting the integral into the ``a`` and ``b`` moments keeps the air-property factors ``(a_v, b_v, \mathrm{Sc}, \nu_\mathrm{air})`` outside the table, so a later move to temperature-dependent air properties, and the Phase-2 bulk wet-growth limit, reuse the same two moments.

## Why the table is indexed by ``\log\lambda``

Each tabulated integral is a function of ``\log\lambda`` and ``\mu(\log\lambda)`` and the geometry set by ``(F_\mathrm{rim}, \rho_\mathrm{rim})`` and ``\rho_\mathrm{air}``, independent of the absolute mass and number.
The slope ``\log\lambda`` is already solved once per cell and passed into every rate call, so the rate table is gridded on ``(\log\lambda, F_\mathrm{rim}, \rho_\mathrm{rim}, \log\rho_\mathrm{air})`` and looked up with that ``\log\lambda``.

Indexing on ``\log\lambda`` rather than on ``x_\mathrm{ice} = L_\mathrm{ice}/N_\mathrm{ice}`` avoids a discontinuity.
For the [`CMP.SlopePowerLaw`](@ref) shape law, the map ``x_\mathrm{ice} \mapsto \log\lambda`` is multi-valued over a narrow range, and the single-valued solver jumps between branches there, so a rate indexed by ``x_\mathrm{ice}`` inherits a jump that no uniform interpolant can represent.
The map ``\log\lambda \mapsto x_\mathrm{ice} = \exp(\mathrm{logLdivN})`` is single-valued, so each ``\log\lambda`` node maps to one synthetic state; the build uses that state and reproduces the node ``\log\lambda``.

The slope solver output is tabulated separately as ``\log\lambda = \Lambda(\log x_\mathrm{ice}, F_\mathrm{rim}, \rho_\mathrm{rim})``.
This table carries the branch jump above and is intended as a warm start for, or a coarse replacement of, the iterative solver; near the fold it is accurate only to the interpolation error of a discontinuous function.

## Grid, ranges, and clamping

The axes are uniform in a transformed coordinate, with a closed-form fractional index and edge clamping.
`LinAxis` is uniform in the coordinate; `LogAxis` is uniform in ``\log_{10}`` of the coordinate.
A coordinate outside the axis range clamps to the nearest edge node, that is, the interpolation weight saturates at ``0`` or ``1`` and the table extrapolates flat.

The default ranges are:

- ``\log\lambda \in [2, 17]``, the shape-solver bracket, so no realistic state clamps.
- ``F_\mathrm{rim} \in [0, 1 - \varepsilon]``, including ``F_\mathrm{rim} = 0`` exactly; the upper bound is the regularization used by [`state_from_prognostic`](@ref).
- ``\rho_\mathrm{rim} \in [100, 0.8\,\rho_l]``, the physical rime-density range; the upper bound is the regularization limit.
  Unrimed states carry ``\rho_\mathrm{rim} = 0`` and clamp to the lower node, which is exact because the rates do not depend on ``\rho_\mathrm{rim}`` at ``F_\mathrm{rim} = 0``.
- ``\rho_\mathrm{air} \in [0.05, 1.5]`` kg m``^{-3}``, covering the column.

The lower ``\rho_\mathrm{rim}`` bound of ``100`` kg m``^{-3}`` excludes the region where a rimed particle would be less dense than that; there the graupel density approaches the unrimed limit, the mass-regime thresholds spread by orders of magnitude, and the rates vary too rapidly for a uniform grid.
Such states are outside the domain where the P3 thresholds are well-posed.

## Interpolation

`lookup` performs exact multilinear interpolation over the ``2^N`` cell corners.
The backing array has the quantity index as its first, fastest-varying dimension, so one corner read fetches every quantity of a table contiguously.
The interpolation is statically unrolled, allocation-free, and differentiable in the coordinates.

The self-collection kernel and the two melt moments are positive and span several decades across the ``\log\lambda`` axis, so they are stored as ``\log`` and exponentiated at the use site; the interpolant then acts on a near-linear function.
The terminal velocities are order one and stored directly.

Interpolation accuracy is limited to first order near the kinks of the shape law, where ``\mu(\log\lambda)`` reaches its bounds, and near the ``F_\mathrm{rim} \to 1`` regime, where the partially-rimed size range vanishes.
At the default grid the relative error against a GL(128) reference over the error-study harness and a physically realizable sweep has a 95th percentile of a few times ``10^{-2}`` and a maximum below ``1.5 \times 10^{-1}`` per quantity; `test/p3_lookup_error_study.jl` reports the per-axis resolution dependence.
These errors are within the parameterization uncertainty of the tabulated processes; for example, the self-collection rate carries an assumed collision efficiency of one.

## Build at init and parameter invalidation

[`build_p3_lookup_tables`](@ref) fills the tables from the P3 quadrature at a high order, threaded over the grid nodes, with no global state.
At the default resolution the build takes on the order of tens of seconds on a multithreaded node.
Node exactness is exact by construction: at a grid node the table reconstructs the direct quadrature evaluation at the build order to machine precision.

The tables depend on every parameter that enters the tabulated integrals: the mass and area power laws, the slope law, the ventilation factor, the terminal-velocity closure, the air properties, and ``\rho_i``, ``\rho_l``, and ``T_\mathrm{freeze}``.
A change to any of these requires rebuilding the tables.

## The container is additive

`RateTable` stores a tuple of named quantities.
Phase 1 stores rates only; additional quantities, such as build-time derivatives for the Jacobian, are a purely additive extension that appends names and array columns without changing the axes or the lookup.

## API

```@docs
P3Scheme.build_p3_lookup_tables
P3Scheme.P3LookupTables
P3Scheme.P3TableGrid
P3Scheme.RateTable
P3Scheme.lookup
P3Scheme.LinAxis
P3Scheme.LogAxis
P3Scheme.fractional_index
```
