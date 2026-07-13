# [P3 Predicted Liquid Fraction](@id P3-liquid-fraction)

The predicted-liquid-fraction extension of the P3 ice scheme tracks the bulk liquid mass held on melting or wet-growing ice, following Cholette et al. (2019) [Choletteetal2019](@cite).
It is selected through the [`CMP.PredictedLiquidFraction`](@ref) treatment on [`CMP.ParametersP3`](@ref); the default [`CMP.NoLiquidFraction`](@ref) treatment is unchanged and reproduces the dry scheme exactly.

## State representation

The ice category gains an independent prognostic liquid mass on ice, ``ρq_{liq}`` `[kg m⁻³]`, alongside the frozen-core mass ``ρq_{ice}``.
The existing ``ρq_{ice}`` keeps its meaning as the *frozen* core (deposition plus rime); the total mixed-phase mass is the sum

```math
ρq_{tot} = ρq_{ice} + ρq_{liq}.
```

Two mass fractions are defined with different normalisations:

```math
F_{liq} = \frac{ρq_{liq}}{ρq_{ice} + ρq_{liq}}, \qquad
F_{rim} = \frac{ρq_{rim}}{ρq_{ice}}.
```

``F_{liq}`` is the liquid mass over the total; ``F_{rim}`` is the rime mass over the frozen core only, so it is untouched by the liquid and reduces exactly to the dry definition when ``ρq_{liq} = 0``.
Both are regularised ratios: ``F_{liq}`` uses a physical volumetric presence scale ``q_{present}`` (`P3_liquid_presence_mass_concentration`, `1e-10 kg m⁻³`) rather than `eps`, so its derivative ``∂F_{liq}/∂ρq_{liq} = 1/(ρq_{ice} + q_{present})`` is finite as ``ρq_{liq} → 0``.
See [`UT.liquid_mass_fraction`](@ref).

The [`P3State`](@ref) constructor clamps ``F_{liq}`` into ``[0, F_{melt}]`` before any use.
The regime thresholds ``D_{th}, D_{gr}, D_{cr}`` and the rime density depend only on ``(F_{rim}, ρ_{rim})`` and are independent of ``F_{liq}``.
[`state_from_prognostic`](@ref) gains a five-argument method (dispatching on the predicted treatment) that takes the trailing ``ρq_{liq}`` and recovers ``ρq_{tot} = ρq_{ice}/(1 - F_{liq})`` via [`total_mass_concentration`](@ref).
The state stores the regularised, clamped ``F_{liq}`` rather than ``ρq_{liq}`` itself, so the recovered total equals the prognostic sum only where the regularisation and the ``F_{melt}`` clamp are inert; past the clamp the true total is not recoverable from the state.
The recovered total feeds only the diagnostic whole-particle shape solve; conserved budgets are assembled from the prognostic variables.
Storing ``ρq_{liq}`` on the state, making the total a literal sum, is deferred to a state-layout revision.

## Particle-property blends

At a given maximum dimension ``D`` the whole (mixed-phase) particle mass, projected area, and terminal velocity are linear blends between the ice-core relation and the pure-drop relation (C19 Eqs. 10, 12, 13):

```math
\begin{aligned}
m_t(D) &= (1 - F_{liq})\, m_{core}(D) + F_{liq}\, \tfrac{π}{6} ρ_l D^3, \\
A_t(D) &= (1 - F_{liq})\, A_{core}(D) + F_{liq}\, \tfrac{π}{4} D^2, \\
V_t(D) &= (1 - F_{liq})\, V_{ice}(D) + F_{liq}\, V_{drop}(D).
\end{aligned}
```

These are [`mixed_mass`](@ref), [`mixed_area`](@ref), and [`mixed_particle_terminal_velocity`](@ref), built on the shared [`liquid_blend`](@ref) helper; ``V_{ice}`` already includes the aspect-ratio factor.
The existing ice-core closures ([`ice_mass`](@ref), [`ice_area`](@ref), [`ϕᵢ`](@ref)) keep their names and meaning; the whole-particle entry points are distinct.
Every pure-drop branch is finite over the ice quadrature support, so the blend reduces to the ice-core value at ``F_{liq} = 0`` with no ``0 \cdot \infty`` hazard.

## Two size distributions and one shared shape parameter

Two gamma size distributions share the same number ``ρn_{ice}`` and the same shape parameter ``μ`` but have different slopes, because they are normalised to different masses (C19 Eqs. 3, 11):

- the **whole-particle** PSD, slope ``λ``, solves ``\int m_t(D)\, n(D)\, \mathrm{d}D = ρq_{tot}``;
- the **ice-core** PSD, slope ``λ_{core}``, solves ``\int m_{core}(D)\, n(D)\, \mathrm{d}D = ρq_{ice}``.

The stored slope is the whole-particle slope.
``μ`` is diagnosed from the whole-particle slope and then held fixed while the ice-core slope is solved (a second fixed-iteration Brent solve at that ``μ``), following the shared-``μ`` convention of Cholette et al. (2023).
The [`P3Shape`](@ref) holds `logλ` (whole), `μ`, and `logλ_core`; at ``F_{liq} = 0`` under the predicted treatment ``λ_{core} → λ`` as intended physics (the dry scheme is reproduced exactly only under the `NoLiquidFraction` dispatch).

Each process integrates over the PSD appropriate to the phase it acts on:

| Process | PSD | slope |
|---|---|---|
| melting, deposition/sublimation | ice core | `logλ_core` |
| refreezing, shedding, condensation/evaporation | whole particle | `logλ` |
| collection, self-collection, sedimentation | whole particle | `logλ` |

### Whole-particle mass moment

The shape target for the whole-particle solve needs ``\log \int m_t(D)\, D^n\, G(D)\, \mathrm{d}D``, where ``G(D) = D^μ e^{-λD}``.
Splitting the blend keeps the piecewise ice-core moment ``\mathrm{core}`` and adds one closed drop moment ``\mathrm{liq0}`` (the full ``D^3`` moment scaled by ``π ρ_l/6``).
The convex combination is evaluated with ``F_{liq}`` outside every logarithm,

```math
\log M = \mathrm{core} + \log\!\big(1 + F_{liq}\,(e^{\mathrm{liq0} - \mathrm{core}} - 1)\big),
```

so the value equals ``\mathrm{core}`` exactly at ``F_{liq} = 0`` and the derivative with respect to ``F_{liq}`` is finite there (``e^{\mathrm{liq0}-\mathrm{core}} - 1``), avoiding the singular ``\log F_{liq}`` intermediate.
Across the physical states and the shape-solve bracket, ``\mathrm{liq0} - \mathrm{core}`` stays well within the exponential range, verified in the test suite.
See [`log_mixed_mass_moment`](@ref).

The whole and core slope solves reuse the fixed-iteration branchless Brent solver.
A calibration study across ``F_{liq} \in (0, 1)`` sets the iteration budget to 12 (Float32) / 14 (Float64); outside a handful of ill-conditioned corner states the budget reproduces a high-iteration reference to well below the test gate (slope error ``< 0.05``, with at most a few states above ``10^{-2}``).
The remaining corners (extreme mean-mass, low-number states) are flat and ill-conditioned regardless of the budget; the dry solve is equally ill-conditioned there, so the budget is not the limiting factor.

## Processes

### Melting

Under the predicted treatment the ice-core melt (see the [dry melting derivation](@ref P3-assumed-particle-size-relationships)) integrates over the ice-core PSD and splits at the small-spherical threshold ``D_{th}`` (C19 Eqs. A1, A2):

```math
Q_{rain} = \int_{0}^{D_{th}} \dot m\, n_{core}\, \mathrm{d}D, \qquad
Q_{liq}  = \int_{D_{th}}^{\infty} \dot m\, n_{core}\, \mathrm{d}D.
```

The ventilation factor is evaluated at the blended whole-particle fall speed [`mixed_particle_terminal_velocity`](@ref), per C19 Eq. A3 and consistent with refreezing; at ``F_{liq} = 0`` it reduces to the ice-core fall speed.
Particles smaller than ``D_{th}`` melt completely to rain; larger particles melt into liquid retained on the ice at constant number.
The complete-melt number transfer converts the mass rate through the mean ice particle mass, bounded into the band `[P3_ice_mean_mass_min, P3_ice_mean_mass_max]`, so the number rate stays bounded and vanishes with the mass rate as the core empties.
The rime drains with the total core-mass loss ``Q_{rain} + Q_{liq}`` to hold ``F_{rim}`` constant.
[`ice_melt`](@ref) returns the double entry explicitly:

| Field | Tendency |
|---|---|
| `dLdt_ice`  | frozen-core mass, ``-(Q_{rain} + Q_{liq})`` |
| `dLdt_rain` | rain mass gain, ``Q_{rain}`` |
| `dNdt_rain` | ice-to-rain number transfer |
| `dLdt_liq`  | liquid-on-ice gain, ``Q_{liq}`` |
| `dLdt_rim`  | rime mass drain, ``-(Q_{rain} + Q_{liq})\,F_{rim}`` |
| `dBdt_rim`  | rime volume drain |

Gains are positive and the core loss is explicit, symmetric with [`ice_refreeze`](@ref).
Both melt destinations are ice-to-liquid phase changes with the fusion enthalpy ``L_f``, while liquid already present undergoes no phase change.

### Refreezing

Below the freezing temperature the liquid on ice refreezes back into rime (C19 Eqs. A4, A5).
[`ice_refreeze`](@ref) integrates the ventilated capacitance-surrogate form of the melt rate over the whole-particle PSD, weighted by ``F_{liq}`` and driven by ``(T_{freeze} - T)``, so it is continuous to zero at ``T_{freeze}`` and vanishes for dry ice.
The frozen-core mass grows and the liquid shrinks by the same rate, conserving the total mixed-phase mass and the ice number; the refrozen mass joins the rime at the solid-ice density ``ρ_i``.
This is a deliberate deviation from the reference 900 kg m⁻³ refreeze density, reusing the existing solid-ice density rather than adding a parameter.

### Shedding

Liquid on particles larger than the shedding onset diameter ``D_{shd}`` (`P3_shedding_onset_diameter`, `9e-3 m`) is shed to rain, scaled by the frozen-core rime fraction (unrimed wet snow retains its liquid; Rasmussen and Heymsfield 1987).
The shed-able liquid mass concentration is the closed-form whole-particle liquid moment above the onset,

```math
L_{shd} = F_{rim}\, F_{liq}\, \frac{π ρ_l}{6} \int_{D_{shd}}^{\infty} D^3\, n(D)\, \mathrm{d}D ,
```

evaluated through the log-space upper incomplete gamma ([`log_upper_incomplete_gamma`](@ref)).
At ``λ D_{shd} \sim 90`` a linear-space tail is at the edge of the Float32 range; the regularised ratio ``Q`` remains representable well past that point and the log-space form stays accurate there, underflowing cleanly to zero only at extreme ``λ D_{shd}``, where shedding is negligible.
An `eps`-floored linear-space moment would instead overestimate the tail.
Shedding is instantaneous in the reference; [`ice_shed`](@ref) returns the shed-able mass and the corresponding rain number as drops of mean diameter ``D_{shd,drop}`` (`P3_shedding_drop_diameter`, `1e-3 m`).
The tendency entry converts the shed-able mass to a rate over `P3_shedding_timescale` (1 s): the reference applies the shed-able mass as a per-step rate over one implicit second, so the default reproduces its rate; the value is the reference's numerical convention, not a physically derived relaxation.
There is no in-scheme per-step availability limit on the shed rate; the rate self-limits as the liquid depletes, and the Rosenbrock substep and host floors bound the applied amount.

### Vapor path

The vapor exchange switches between deposition/sublimation on the dry core and condensation/evaporation on the wet shell at a small liquid fraction (C19 §3d).
The reference discontinuous switch at ``F_{dry}`` is replaced by a ``C^1`` Hermite ramp [`vapor_path_weight`](@ref) on the band ``[F_{dry}, F_{dry} + ΔF]``, with ``F_{dry}`` from `P3_liquid_fraction_dry_threshold` (0.01) and the width ``ΔF`` from `P3_liquid_fraction_switch_width` (0.02).
The ramp satisfies ``w(0) = 0`` and ``w'(0) = 0``, so the dry-ice deposition/sublimation baseline is untouched and the manual/exact Jacobian sees no kink in the state variable ``F_{liq}``.
This is a modeling deviation justified by the tuning latitude of ``F_{dry}`` in the reference (0.01 up to 0.2) and by Jacobian smoothness; the band is asserted strictly inside ``(0, F_{melt})`` at construction.

## Host wiring contract

The predicted-liquid-fraction scheme is selected with
`Microphysics2MParams(FT; with_ice = true, liquid = :predicted)` and enters the bulk-tendency interface through the packed 2M+P3 entry; there is no positional form (the `logλ`-based positional wrappers remain dry-only and throw under the predicted treatment).
Per grid cell and substep, the host:

 1. builds the state from the prognostic variables including the liquid on ice,
    `state = state_from_prognostic(params, ρq_ice, ρn_ice, ρq_rim, ρb_rim, ρq_liq_on_ice)`,
    and solves the shape once, `shape = get_distribution_shape(state)` (the whole-particle slope, the shared ``μ``, and the frozen-core slope `logλ_core`, all frozen for the call);
 2. calls the packed entry with the per-category input
    `ice = ((; q_ice, n_ice, q_rim, b_rim, q_liq_on_ice),)` (specific quantities, kg/kg) and `shapes = (shape,)`.

The returned tendency `NamedTuple` gains a `dq_liq_on_ice_dt` field [kg/kg/s] after the rime-volume field, in the instantaneous entry and in the `RosenbrockAverage{ExactJacobian}` mode (`rosenbrock_exact`).
`rosenbrock_manual` and the `Verbose` diagnostic path do not support the predicted treatment and throw; the manual-Jacobian tiers for the liquid row are deferred, and `ExactJacobian` (the validation path) differentiates the entry with the three-slot shape frozen on the substep context.
Box A/B campaigns run `rosenbrock_exact`: a non-finite Jacobian at a corner state routes that substep to the built-in forward-Euler fallback, and the manual tiers are not a prerequisite.

Every liquid source and sink has a matching entry against rain, rime, or the frozen core: melting (core to rain and to retained liquid, with the rime drain), refreezing (liquid to rime and core), shedding (liquid to rain over `P3_shedding_timescale`), and collection (collected cloud and rain to retained liquid or rime).
The vapor exchange is the only non-transfer term: the ramp splits it between core deposition/sublimation and shell condensation/evaporation, both bulk relaxations toward ice and liquid saturation on the `SubDep2M` timescale (not PSD-resolved integrals, consistent with the dry treatment), with the liquid on ice included in the vapor and heat budgets (the reference translates every total-ice reference as core plus liquid).

The sedimentation velocities for the mixed particle are
[`ice_terminal_velocity_number_weighted`](@ref) and
[`ice_terminal_velocity_mass_weighted`](@ref) evaluated on the predicted-liquid state: both dispatch on the liquid treatment and use the blended whole-particle fall speed over the whole-particle PSD (C19 Eqs. A8, A9), with the mass weighting on [`mixed_mass`](@ref) normalised by the total mixed-phase mass.
The prognostic sedimentation wrappers ([`ice_terminal_velocity_number_weighted_from_prognostic`](@ref), [`ice_terminal_velocity_mass_weighted_from_prognostic`](@ref)) gain forms with a trailing `ρq_liq` argument before `logλ`.
`ρq_liq_on_ice` sediments with the mass-weighted velocity of the whole particle, as does the frozen core.

Positivity is layered: the host (ClimaAtmos) floors `ρq_liq_on_ice` alongside the other cold-moment species; the state construction clamps the regularised `F_liq` into `[0, F_melt]`; and the Rosenbrock substep floors the species vector at zero.
For the condensate rescale, the saturation adjustment, and the energy partition, `ρq_liq_on_ice` is liquid-phase condensate (it enters with the vaporization enthalpy ``L_v``; the frozen core with ``L_s``), even though it is floored and advected with the cold-moment species.
Residual liquid on an emptied core (frozen core below the entry's ice-presence threshold) drains to rain as ``D_{shd,drop}`` drops over `P3_shedding_timescale`; the drain ramps linearly to zero at the presence threshold, so it is continuous in the core mass and identically zero wherever the liquid transfer processes are active.
The instantaneous complete-melt and tiny-film cleanups of the reference are represented continuously by this drain, the shedding rate, and the refreezing integral (which acts on any liquid fraction below freezing).
