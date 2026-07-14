# P3 Multiple Ice Categories

The multiple-ice-category extension of the P3 scheme lets several free ice populations coexist at the same point, following [MilbrandtMorrison2016](@cite) (Part III).
A single P3 category forces all ice into one gamma population with one set of bulk properties, so adding freshly nucleated crystals or splinters to an existing population of large graupel dilutes that population's properties and corrupts its subsequent growth.
Running `nCat` free categories reduces this dilution: physically distinct modes (for example, small high-number pristine ice alongside large low-density rimed ice) evolve independently.

Each category carries the same prognostic set as single-category P3 (total ice mass, number, rime mass, and rime volume; plus reflectivity under three-moment ice, and liquid-on-ice mass under predicted liquid fraction).
The categories couple only through gravitational collection, and the number of active categories is kept in check by routing newly formed ice to a destination category and by merging categories that have converged in size and density.

This page documents the pure-physics layer: the inter-category collection kernel, the destination-selection algorithm, and the merge algorithm.
The parameters live in [`InterCategoryParams`](@ref CloudMicrophysics.Parameters.InterCategoryParams), attached to `P3IceParams` (`nothing` for a single category).

## Inter-category collection

For an ordered pair of categories, collection transfers mass from the collectee `j` to the collector `i`.
The bulk rate for the prognostic quantity `X` is the gravitational-collection double integral over the two size distributions ([MilbrandtMorrison2016](@cite), Eq. 1):

```math
\frac{\partial X_{j \to i}}{\partial t}
= \int \int E \; \sigma(D_i, D_j) \; \left( v_i(D_i) - v_j(D_j) \right)_+ \; Y_X(D_j) \; N_i(D_i) \, N_j(D_j) \; \mathrm{d}D_i \, \mathrm{d}D_j .
```

The kernel is the classic geometric sweep-out form.
``\sigma(D_i, D_j) = \pi \left( r_i(D_i) + r_j(D_j) \right)^2`` is the ice-ice collision cross-section built from the two particles' effective radii ``r = \sqrt{A/\pi}`` (see [`collision_cross_section_ice_ice`](@ref CloudMicrophysics.P3Scheme.collision_cross_section_ice_ice)).
The one-sided differential fall speed ``\left( v_i - v_j \right)_+`` selects only pairs in which the collector falls faster than the collectee, so mass moves from the slower-falling to the faster-falling category.
Its kink at the fall-speed crossover ``v_j(D^\star) = v_i(D_i)`` is placed on an inner subinterval boundary (see [`crossover_diameter`](@ref CloudMicrophysics.P3Scheme.crossover_diameter)) so the Gauss-Legendre rule resolves it, exactly as in the ice-liquid collision path.
The outer integrand remains ``C^1`` where the crossover enters or leaves the inner domain, because the ``(\cdot)_+`` integrand vanishes at ``D^\star``; only the collector's own mass-regime and velocity breakpoints bound the outer integral.

### Transfer factors

The transfer factor ``Y_X`` is the amount of quantity `X` carried per collected collectee particle of mass ``m_j(D_j)``.
Using the collectee's bulk rime fraction ``F_{rim,j}`` and rime density ``\rho_{rim,j}`` (assumed uniform across its size distribution):

| Quantity `X`        | Transfer factor ``Y_X``            |
|:--------------------|:-----------------------------------|
| number              | ``1``                              |
| ice-core mass       | ``m_j(D_j)``                       |
| rime mass           | ``m_j(D_j) \, F_{rim,j}``          |
| rime volume         | ``m_j(D_j) \, F_{rim,j} / \rho_{rim,j}`` |

The collector's number is unchanged (collection merges particles rather than creating them), so the collectee number rate is a pure sink on `j`.
The rime volume factor divides by the collectee rime density, so the transferred rime arrives at the collectee's rime density: ``\Delta q_{rim} / \Delta b_{rim} = \rho_{rim,j}`` on the collector side.
Whatever ice mass, rime mass, and rime volume leave the collectee arrive at the collector unchanged; [`inter_category_collection`](@ref CloudMicrophysics.P3Scheme.inter_category_collection) returns the collectee-side rates, and the caller applies them as the paired collector source and collectee sink.
Collection is directional: [`ordered_category_pairs`](@ref CloudMicrophysics.P3Scheme.ordered_category_pairs) enumerates both `(i, j)` and `(j, i)` for every pair, and the caller evaluates each.

### Collection efficiency and its rime-fraction shutoff

The efficiency is ``E = E_{ii} \cdot f(F_{rim,i})``, with a base efficiency ``E_{ii} = 0.1`` and a shutoff factor ``f`` that depends on the collector rime fraction ``F_{rim,i}``.
Heavily rimed ice (graupel and hail) aggregates little, so ``f`` ramps from one to zero as ``F_{rim,i}`` increases from `0.6` to `0.9`.
The reference Fortran uses a linear ramp ``f = 1 - (F_{rim} - 0.6)/0.3``, which is only ``C^0`` (its slope jumps at both bounds).
This implementation uses the ``C^1`` Hermite smoothstep ``f = 1 - t^2(3 - 2t)`` with ``t = (F_{rim} - 0.6)/0.3``, matching the linear ramp's endpoint values while removing the slope discontinuities.
See [`rime_collection_shutoff`](@ref CloudMicrophysics.P3Scheme.rime_collection_shutoff).

### Air-density normalization

The Fortran lookup table for inter-category collection integrates size distributions normalized by number and assumes an air density of ``1 \; \mathrm{kg\,m^{-3}}``, then multiplies the process rate by the air density and by both categories' number mixing ratios; the table generator flags this as a defect to be fixed.
This implementation integrates true number concentrations ``N_i(D)`` and ``N_j(D)`` directly, so the rate is already in concentration units ``[\mathrm{m^{-3}\,s^{-1}}]`` and carries the air-density dependence cleanly through the terminal-velocity closure, without the ``\times \rho`` correction factor.

### Reflectivity transfer (constant-μ)

Inter-category reflectivity (sixth-moment) transfer is new physics with no reference-Fortran counterpart.
Under the constant-μ closure, the reflectivity of a gamma population is ``Z = c(\mu) \, L^2 / N`` with ``c(\mu)`` fixed while ``\lambda`` responds to the changing mass and number, so the collectee reflectivity sink is

```math
\partial_t Z_j = Z_j \left( 2 \, \frac{\partial_t L_j}{L_j} - \frac{\partial_t N_j}{N_j} \right) ,
```

evaluated on the collection mass and number rates.
Collection is size-selective (``\partial_t L_j / L_j \neq \partial_t N_j / N_j`` in general), and this form tracks it: when large collectee particles are preferentially swept up, the sink exceeds the number-proportional value ``(Z_j/N_j) \, \partial_t N_j``, and in the monodisperse limit the two coincide.
This term is zero under two-moment ice (no reflectivity is carried).

Reflectivity is not conserved pairwise, by design: the collectee loses ``Z`` by the closure above, while the collector's reflectivity is recomputed from its updated mass and number by the three-moment shape solve in the tendency-entry wiring phase.
P3 treats ``Z`` as a shape-diagnostic moment rather than a conserved tracer across category transfers.

## Destination category for new ice

Newly formed ice (from nucleation, freezing, or ice multiplication) is routed to a single destination category by [`icecat_destination`](@ref CloudMicrophysics.P3Scheme.icecat_destination), following [MilbrandtMorrison2016](@cite), section 2b(1).
The selection metric is the mean-mass diameter ``D_m`` (see [`D_m`](@ref CloudMicrophysics.P3Scheme.D_m)).
Given the new-ice mean diameter ``D_{new}``:

- if all categories are empty, use category 1;
- if all categories are populated, use the category whose ``|D_m - D_{new}|`` is smallest;
- otherwise, use the closest populated category when its ``|D_m - D_{new}|`` is below the initiation threshold ``\Delta D_{init}``, and start a new (first empty) category when it is not.

``\Delta D_{init}`` is not a physical parameter and does not enter any process rate; it only controls how ice is distributed among the categories.
Its optimal value decreases with the number of categories, so it is selected per configuration at parameter-construction time from five values (``500, 400, 235, 175, 150 \; \mu\mathrm{m}`` for two through six categories).
A category is populated when its ice mass content exceeds the shared species-presence threshold (``10^{-10} \; \mathrm{kg\,m^{-3}}``); residual sub-threshold categories count as empty in both the destination selection and the merge sweep.
``D_{new}`` must use the same mass-weighted mean-diameter definition as ``D_m``; the reference Fortran computes its ``D_{new}`` as the equivalent mean-mass sphere diameter, a different moment, and mixing the two definitions degrades the selection metric.

## Merging similar categories

After sedimentation, [`merge_categories`](@ref CloudMicrophysics.P3Scheme.merge_categories) collapses adjacent categories that have converged, freeing category slots for new initiation.
Following the two-condition criterion of [MilbrandtMorrison2016](@cite), section 2b(3), two adjacent populated categories are merged when both:

```math
|D_m(n_1) - D_m(n_2)| < 150 \; \mu\mathrm{m}
\quad \text{and} \quad
|\rho_i(n_1) - \rho_i(n_2)| < 100 \; \mathrm{kg\,m^{-3}} ,
```

where ``\rho_i`` is the mean bulk density (ice mass over equivalent-sphere volume; see [`mean_ice_density`](@ref CloudMicrophysics.P3Scheme.mean_ice_density)).
Merging sums every prognostic quantity into the lower-index category and zeros the higher-index one; reflectivity sums additively ([Milbrandt2021](@cite)).

The reference Fortran diverges from the paper here: it reuses ``\Delta D_{init}`` as the merge diameter threshold and applies only the diameter condition, omitting the density condition.
This implementation follows the two-condition paper form with the paper's fixed ``150 \; \mu\mathrm{m}`` and ``100 \; \mathrm{kg\,m^{-3}}`` thresholds.

## Host contract for the multi-category tendency entry

The packed 2M+P3 tendency entry accepts one to four ice categories: the parameter set is built with `n_categories`
(`Microphysics2MParams(FT; with_ice = true, n_categories = N)`, carried as the leading type parameter of `P3IceParams{N}`),
and the entry receives `N`-tuples of per-category prognostic inputs and frozen shapes.
The host carries the per-category prognostic scalars with `_1.._N` suffixes (`ρq_ice_1`, ..., following the Fortran wrapper convention),
builds each category's state with `state_from_prognostic` from its own `(ρq_ice, ρn_ice, ρq_rim, ρb_rim[, ρq_liq_on_ice][, ρz_ice])`,
and solves each category's shape independently.
For `N > 1` the returned tendency fields carry the category index before the `_dt` suffix (`dq_ice_1_dt`, ..., `dz_ice_2_dt`);
a single category keeps the unsuffixed names byte-for-byte.

Process semantics inside the entry follow the reference's per-category loop:

- Every category's warm-coupled processes (collection of cloud and rain, melting, shedding) consume the same
  step-initial cloud, rain, and vapor state; the rates are evaluated simultaneously and the warm-block sinks
  accumulate across categories.
  The host applies the returned rates over its step, as in the reference, where all process rates are computed
  from the state at the beginning of the step and applied together.
- The constant-timescale vapor exchange sees the full ice condensate in its vapor and heat budgets and is
  partitioned among the categories by core-mass share (deposition/sublimation) and liquid-mass share
  (shell condensation/evaporation).
  This partition stands in for the reference's per-category ventilation-factor weighting, which requires a
  PSD-resolved exchange; a single category keeps its unconditional relaxation.
- Newly formed ice routes to the destination category ([`icecat_destination`](@ref CloudMicrophysics.P3Scheme.icecat_destination)):
  deposition nucleation at the nascent-crystal diameter, and drop freezing at the mean-mass sphere diameter of the
  frozen drops at solid-ice density.
  The frozen-drop population is treated as monodisperse, for which that diameter coincides with the mass-weighted
  mean diameter required by the selection metric.
- Inter-category collection applies the source-sink double entry of
  [`inter_category_collection`](@ref CloudMicrophysics.P3Scheme.inter_category_collection).
  Both sides' reflectivity budgets are carried by the per-category constant-μ growth pass over the net rates:
  by linearity, the collectee side realizes the documented ``\partial_t Z_j`` sink with the entry's frozen
  (mean-mass-band-clamped) coefficients, and the collector side gains through its own growth term from the gained
  mass at unchanged number, so ``Z`` is not conserved pairwise.

Outside the entry, the host is responsible for:

- Sedimentation per category, with each category's own number-, mass- (and reflectivity-) weighted terminal
  velocities evaluated from its own state and shape; there is no cross-category coupling in sedimentation.
- Calling [`merge_categories`](@ref CloudMicrophysics.P3Scheme.merge_categories) after sedimentation to collapse
  converged categories; merging is not part of the tendency entry, which only routes sources.
- Positivity per category: the host floors every per-category prognostic independently, exactly as for a single
  category.

The Rosenbrock substep driver supports `rosenbrock_exact()` for any category count; `rosenbrock_manual()` and
`Verbose` throw for more than one category.
The substep loop and the instantaneous entry are allocation-free for every supported layout: above the
StaticArrays public-solve cutoff (state lengths over 14) the substep system is solved through the
size-independent static LU kernel, which keeps the larger layouts (for example two three-moment
liquid-fraction categories, 16 states) runnable inside GPU kernels, where the public heap-fallback solve
cannot compile.

## API

```@docs
CloudMicrophysics.Parameters.InterCategoryParams
CloudMicrophysics.P3Scheme.inter_category_collection
CloudMicrophysics.P3Scheme.ordered_category_pairs
CloudMicrophysics.P3Scheme.rime_collection_shutoff
CloudMicrophysics.P3Scheme.icecat_destination
CloudMicrophysics.P3Scheme.merge_categories
CloudMicrophysics.P3Scheme.mean_ice_density
```
