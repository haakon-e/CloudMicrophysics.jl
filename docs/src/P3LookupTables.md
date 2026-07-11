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
At the default grid the relative error against a GL(128) reference over the error-study harness and a physically realizable sweep has a 95th percentile of a few times ``10^{-2}`` per quantity.
The maximum stays below ``1.5 \times 10^{-1}`` for self-collection, both velocities except the mass-weighted one, and melting; the mass-weighted velocity reaches about ``4 \times 10^{-1}`` in the ``F_\mathrm{rim} \to 1`` corner.
The accepted 95th-percentile target of a few times ``10^{-2}`` is looser than an initial aspiration of ``3 \times 10^{-3}`` (95th percentile) and ``3 \times 10^{-2}`` (maximum).
Reaching the tighter values requires roughly a hundred megabytes of table and a build of several minutes, and the accepted values remain within the parameterization uncertainty of the tabulated processes, for example the self-collection rate carries an assumed collision efficiency of one.
`test/p3_lookup_error_study.jl` reports the per-axis resolution dependence behind these values.

The ``\log\lambda`` shape table reproduces the iterative solver to a few times ``10^{-2}`` at the 95th percentile over in-grid states, with a maximum near the ``x_\mathrm{ice} \to \log\lambda`` fold of about ``10^{-1}``, so it serves as a warm start or coarse replacement rather than a drop-in for the solver near the fold.

## Build at init and parameter invalidation

[`build_p3_lookup_tables`](@ref) fills the tables from the P3 quadrature at a high order, threaded over the grid nodes, with no global state.
At the default resolution the build takes on the order of tens of seconds on a multithreaded node.
Node exactness is exact by construction: at a grid node the table reconstructs the direct quadrature evaluation at the build order to machine precision.

The tables depend on every parameter that enters the tabulated integrals: the mass and area power laws, the slope law, the ventilation factor, the terminal-velocity closure, the air properties, and ``\rho_i``, ``\rho_l``, and ``T_\mathrm{freeze}``.
A change to any of these requires rebuilding the tables.

## The container is additive

`RateTable` stores a tuple of named quantities.
Phase 1 stores rates only; additional quantities, such as build-time derivatives for the Jacobian, are a purely additive extension that appends names and array columns without changing the axes or the lookup.

## Liquid-ice collisions

The liquid-ice collision source is the dominant quadrature cost of the ice-phase tendency.
Its integrand is separable into a temperature-free, bilinear part that is tabulated and a temperature-dependent freeze/shed partition that is applied at the use site.

### The two collision tables

The unpartitioned collision moments factor into an analytic prefactor and a shape function of the same shape coordinates as the Phase-1 rates plus one liquid-distribution coordinate.
The cloud droplet distribution scales as ``N_c`` times a shape set by the mean droplet mass ``x_c = L_c / N_c``.
The rain distribution scales as ``N_{0r}`` times a shape set by the mean drop diameter ``D_{r,\mathrm{mean}}``, where ``(N_{0r}, D_{r,\mathrm{mean}})`` are the closed-form parameters of the limited Seifert-Beheng rain PDF.

| Table | Prefactor | Coordinates | Quantities |
|:------|:----------|:------------|:-----------|
| cloud collision | ``N_\mathrm{ice} N_c`` | ``\log\lambda,\ F_\mathrm{rim},\ \rho_\mathrm{rim},\ \log\rho_\mathrm{air},\ \log x_c`` | ``G_{NC}, G_{MC}`` |
| rain collision | ``N_\mathrm{ice} N_{0r}`` | ``\log\lambda,\ F_\mathrm{rim},\ \rho_\mathrm{rim},\ \log\rho_\mathrm{air},\ \log D_{r,\mathrm{mean}}`` | ``G_{NR}, G_{MR}`` |

The tabulated quantities are the number and mass collision moments normalized by the prefactor,

```math
G_{NC} = \frac{\mathrm{NCCOL}}{N_\mathrm{ice} N_c}, \quad
G_{MC} = \frac{M_C}{N_\mathrm{ice} N_c}, \quad
G_{NR} = \frac{\mathrm{NRCOL}}{N_\mathrm{ice} N_{0r}}, \quad
G_{MR} = \frac{M_R}{N_\mathrm{ice} N_{0r}},
```

stored as ``\log``.
The rain channel is normalized by ``N_{0r}`` rather than by ``N_r`` because the ``N_{0,\max}`` clamp of the limited rain PDF binds at realistic loadings, so the rain shape depends on ``L_r`` alone through ``D_{r,\mathrm{mean}}``.
The ``D_{r,\mathrm{mean}}`` axis spans the interval into which the rain PDF slope is clamped, ``[10^{-4}, 10^{-3}]`` m, so no realizable rain state leaves the grid.
The build fills the moments without the wet-growth partition (no onset location, no Musil limit, no freeze/shed split), which is both cheaper than the runtime path and exactly the temperature-free part.

### The Musil freeze capacity

The bulk wet-growth limit reuses the same ``a`` and ``b`` ventilation split as the melt rate, but with the ice diameter moment rather than the mass-derivative moment.
The maximum freezing rate factors as ``\partial_t M_\mathrm{max}(D_i) = A(T_a, \rho_\mathrm{air})\, D_i\, F_v(D_i)`` with the scalar

```math
A = 2\pi \frac{K_\mathrm{therm}\, \Delta T + L_v D_\mathrm{vapor}\, \Delta\rho_{v,\mathrm{sat}}}{L_f - c_{p,l}\, \Delta T}, \qquad \Delta T = T_\mathrm{freeze} - T_a,
```

so the bulk freeze capacity is

```math
\int M_\mathrm{max} = A\, N_\mathrm{ice} \left( a_v V_a + \frac{b_v\, \mathrm{Sc}^{1/3}}{\sqrt{\nu_\mathrm{air}}}\, V_b \right),
\quad V_a = \int \hat{n}(D) D\, \mathrm{d}D, \quad V_b = \int \hat{n}(D) D^{3/2} \sqrt{v_i(D)}\, \mathrm{d}D,
```

where ``\hat{n}`` is the unit-number ice distribution.
The moment ``V_a`` is air-density independent and tabulated in 3D on ``(\log\lambda, F_\mathrm{rim}, \rho_\mathrm{rim})``; ``V_b`` carries the ice fall speed and is tabulated in 4D with the added ``\log\rho_\mathrm{air}`` axis.
At ``T_a \lesssim 220`` K the denominator turns non-positive and every colliding droplet freezes; the scalar returns `floatmax` there, so a bulk ``f_\mathrm{frz} = \min(1, \int M_\mathrm{max} / \int M_\mathrm{col})`` saturates to one.

### The bulk-partition assembly

The runtime assembly reads the four collision moments and the freeze capacity from the tables, forms one bulk freeze fraction ``f_\mathrm{frz} = \min(1, \int M_\mathrm{max} / \int M_\mathrm{col})`` with ``\int M_\mathrm{col} = M_C + M_R``, and splits each channel proportionally:

```math
\mathrm{QCFRZ} = f_\mathrm{frz} M_C, \quad
\mathrm{QCSHD} = (1 - f_\mathrm{frz}) M_C, \quad
\mathrm{QRFRZ} = f_\mathrm{frz} M_R, \quad
\mathrm{QRSHD} = (1 - f_\mathrm{frz}) M_R.
```

Shed rain number uses the mass of a fixed shed drop, ``\mathrm{NRSHD} = \mathrm{QRSHD} / m_\mathrm{liq}(D_\mathrm{shd})``.
The rime-volume sources use a representative-size Cober-List density, evaluated with the mass-weighted mean ice velocity from the Phase-1 velocity table, the cloud mass-mean diameter ``\bar{D}_c = M_4/M_3``, and the rain representative diameter ``\bar{D}_r = 4 D_{r,\mathrm{mean}}``.
The wet fraction ``f_\mathrm{wet} = 1 - f_\mathrm{frz}`` feeds the densification of rime mass and volume.

### Accuracy and the bulk-partition delta

The cloud mass and number sources are partition-free: ``\partial_t q_c = -M_C / \rho_\mathrm{air}`` and ``\partial_t N_c = -\mathrm{NCCOL}`` reproduce the quadrature values to the table interpolation error at every temperature.
At temperatures cold enough that the Musil limit does not bind, ``f_\mathrm{frz} = 1`` and every one of the seven outputs reduces to a table interpolation of the quadrature value, again to the interpolation error.
Over the error-study harness and a physically realizable sweep, the interpolation error of the four collision moments has a 95th percentile of a few times ``10^{-2}``, with the maximum in the ``F_\mathrm{rim} \to 1`` corner, matching the Phase-1 rate tables.

The bulk freeze/shed partition is a physics change relative to the per-diameter partition of the quadrature scheme, not only an interpolation.
The two partitions agree to below one percent in the cold band and diverge in the warm band ``T \in [258, 272]`` K, where the direct scheme resolves a per-size wet-growth window that the single bulk fraction cannot represent.
The five partition-dependent outputs (``\partial_t q_r``, ``\partial_t N_r``, ``\partial_t L_\mathrm{rim}``, ``\partial_t L_\mathrm{ice}``, ``\partial_t B_\mathrm{rim}``) then differ by 10 to 34 percent at the warmest states, with the largest error on the rain sources.
This delta is a property of the bulk closure and is bounded below by no grid refinement; it is reported here so that its acceptability is a modeling decision.
The two softest parts of the closure are the wet fraction, which the bulk mean reads near zero while the per-size fraction reaches 0.1 to 0.5, and the representative-size rime density near ``0`` °C where the Cober-List index leaves its floor; both are quantified in the evaluation study.

## Variant C: inner-component tables and the exact outer path

Variant A tabulates the fully integrated collision moments and replaces the per-diameter freeze/shed partition with one bulk fraction, which introduces the warm-band bias above.
Variant C instead tabulates only the inner collision moments and keeps the outer integral over ice size, so the per-diameter partition is retained exactly.

The inner cloud and rain collision integrals depend on the ice state only through the ice fall speed ``v_i(D_i)`` and the effective radius ``r_i(D_i) = \sqrt{a_i(D_i)/\pi}`` at the outer ice diameter.
The collision cross-section is ``\pi (r_i + D_l/2)^2`` and the collision rate carries ``|v_i - v_l|``, so the shape coordinates ``(\log\lambda, F_\mathrm{rim}, \rho_\mathrm{rim})`` of the variant-A tables collapse to the two coordinates ``(v_i, r_i)``.
The collapse is exact: the closed-form rain inner and the cloud inner quadrature take no other state-dependent argument, so the leakage is at machine precision.

| Table | Prefactor | Coordinates | Quantities |
|:------|:----------|:------------|:-----------|
| cloud inner | ``N_c`` | ``v_i,\ r_i,\ \log\rho_\mathrm{air},\ \log x_c`` | ``H_{NC}, H_{MC}`` |
| rain inner | ``N_{0r}`` | ``v_i,\ r_i,\ \log\rho_\mathrm{air},\ \log D_{r,\mathrm{mean}}`` | ``H_{NR}, H_{MR}`` |

The tabulated quantities are the inner number and mass collision moments per unit liquid-number prefactor, ``H_{NC} = \partial_t N_{c,\mathrm{col}} / N_c`` and so on, stored as ``\log``.

The variant-C method of [`bulk_liquid_ice_collision_sources`](@ref) builds the ice size distribution, the collision rate, and the Musil freeze limit at runtime, then integrates over ice size with a low-order rule.
At each outer node it reads the inner cloud and rain moments from the tables at ``(v_i, r_i, \rho_\mathrm{air}, x_c)`` and ``(v_i, r_i, \rho_\mathrm{air}, D_{r,\mathrm{mean}})``, applies the per-diameter freeze/shed partition, and accumulates the seven sources.
The wet-growth onset scan uses the same tabulated inner masses, so the onset location is cheap.
The rime-volume sources use the representative-density Cober-List closure shared with variant A.

Because the outer integral and the partition are exact, the variant-C error is the table interpolation error at every temperature, with no warm-band bias.
Over the error-study harness and sweep, the seven-output error against a GL(128) reference has a 95th percentile of a few times ``10^{-2}`` from the cold band to one degree below freezing, against about ``1.5 \times 10^{-1}`` and rising for variant A.
The inner tables are small, on the order of a few megabytes, because the ``(v_i, r_i)`` collapse removes three shape axes, and the build is a few seconds.

## The hybrid switch

The hybrid method forms the bulk freeze ratio ``\int M_\mathrm{max} / \int M_\mathrm{col}`` from the variant-A tables and compares it to a threshold ``\theta``.
Where the ratio is at least ``\theta`` the bulk partition does not bind, variant A is exact up to its interpolation error, and the cheap variant-A assembly is used; elsewhere the exact variant-C path is used.
The natural value is ``\theta = 1``, the freeze-everything threshold; raising ``\theta`` toward infinity uses variant C everywhere, and lowering it toward zero uses variant A everywhere.
``\theta`` is a tunable parameter of the method, not a hardcoded constant.

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
P3Scheme.build_p3_collision_tables
P3Scheme.P3CollisionTables
P3Scheme.P3CollisionGrid
P3Scheme.bulk_max_freeze_rate
P3Scheme.build_p3_collision_inner_tables
P3Scheme.P3CollisionInnerTables
P3Scheme.P3CollisionInnerGrid
```
