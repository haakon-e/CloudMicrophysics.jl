"""
    het_ice_nucleation(aerosol, tps, q_lcl, N_lcl, RH, T, ρₐ)

Calculate the ice nucleation rate from heterogeneous freezing due to some `aerosol`

# Arguments
  - `aerosol`: aerosol parameters (supported types: desert dust, illite, kaolinite)
  - `tps`: thermodynamics parameters
  - `q_lcl`: cloud liquid water specific content
  - `N_lcl`: cloud droplet number concentration
  - `RH`: relative humidity
  - `T`: temperature
  - `ρₐ`: air density

# Returns
- A `NamedTuple` with the fields:
  - `dNdt`: ice number concentration change rate [m⁻³ s⁻¹]
  - `dLdt`: ice content change rate [kg m⁻³ s⁻¹]
"""
function het_ice_nucleation(
    aerosol::Union{CMP.DesertDust, CMP.Illite, CMP.Kaolinite},
    tps::TDI.PS,
    q_lcl, N_lcl, RH, T, ρₐ,
)
    FT = eltype(tps)
    #TODO - Also consider rain freezing

    # Immersion freezing nucleation rate coefficient
    J = CM_HetIce.ABIFM_J(aerosol, RH - CO.a_w_ice(tps, T))

    # Assumed erosol surface area
    # TODO - Make it a parameter of ABIFM scheme
    # We could consider making it a function of the droplet size distribution
    A_aer = FT(1e-10)

    # NaN guard: if ABIFM returns a non-finite J for a degenerate input,
    # treat non-finite J as "no nucleation" (0 rate), consistent with
    # the physical limit of an unresolvable state.
    JA_aer = ifelse(isfinite(J), J * A_aer, zero(J))

    dNdt = max(0, JA_aer * N_lcl)
    dLdt = max(0, JA_aer * q_lcl * ρₐ)

    return (; dNdt, dLdt)
end

"""
    ice_melt(velocity_params, aps, tps, Tₐ, ρₐ, state, shape; ∫kwargs...)

# Arguments
 - `velocity_params`: [`CMP.Chen2022VelType`](@ref)
 - `aps`: [`CMP.AirProperties`](@ref)
 - `tps`: thermodynamics parameters
 - `Tₐ`: temperature (K)
 - `ρₐ`: air density
 - `state`: a [`P3State`](@ref) object
 - `shape`: the diagnosed [`P3Shape`](@ref)

# Keyword arguments
 - `quad`: quadrature rule (a `Quadrature.QuadratureRule`)

Returns the melting rate of ice (QIMLT in Morrison and Mildbrandt (2015)).

Dispatches on the liquid-fraction treatment. Under
[`CMP.NoLiquidFraction`](@ref) the whole particle melts to rain and the return
is `(; dNdt, dLdt)`. Under [`CMP.PredictedLiquidFraction`](@ref) the frozen-core
melt splits at `D_th`: particles with `D ≤ D_th` melt completely to rain, larger
particles melt into retained liquid on ice; the return is
`(; dLdt_ice, dLdt_rain, dNdt_rain, dLdt_liq, dLdt_rim, dBdt_rim)` with
`dLdt_ice = -(dLdt_rain + dLdt_liq)` the frozen-core mass tendency, symmetric
with [`ice_refreeze`](@ref) (gains positive, the core loss explicit). See the
[P3 liquid-fraction documentation](@ref P3-liquid-fraction).
"""
@inline ice_melt(
    velocity_params, aps::CMP.AirProperties, tps::TDI.PS,
    Tₐ, ρₐ, state::P3State, shape::P3Shape; quad,
) = _ice_melt(state.params.liquid, velocity_params, aps, tps, Tₐ, ρₐ, state, shape; quad)

@inline function _ice_melt(
    ::CMP.NoLiquidFraction, velocity_params, aps::CMP.AirProperties, tps::TDI.PS,
    Tₐ, ρₐ, state::P3State, shape::P3Shape; quad,
)
    (; K_therm) = aps
    L_f = TDI.Lf(tps, Tₐ)

    (; ρq_ice, ρn_ice) = state
    (; T_freeze, vent) = state.params

    v_term = ice_particle_terminal_velocity(velocity_params, ρₐ, state)
    F_v = CO.ventilation_factor(vent, aps, v_term)
    N′ = size_distribution(state, shape)

    # Integrate; the ventilation factor carries the terminal-velocity regime
    # break, so the velocity cutoff is a subinterval boundary
    fac = 4 * K_therm / L_f * (Tₐ - T_freeze)
    bnds = velocity_integral_bounds(state, shape, v_term; p = 1e-6)
    melt_integrand = D -> ∂ice_mass_∂D(state, D) * F_v(D) * N′(D) / D
    dLdt_unclamped = fac * integrate(melt_integrand, bnds, quad)

    # only consider melting (not fusion)
    dLdt = max(0, dLdt_unclamped)
    # Remove number in proportion to mass through the mean particle mass,
    # floored (not ceilinged) so the rate stays finite when `ρq_ice`
    # underflows to zero while `ρn_ice` and the melt integral remain positive,
    # and vanishes as `ρn_ice` does rather than saturating at a fixed rate.
    FT = eltype(state)
    m̄ = max(ρq_ice / max(ρn_ice, floatmin(FT)), ice_mean_particle_mass_min(FT))
    dNdt = dLdt / m̄

    return (; dNdt, dLdt)
end

@inline function _ice_melt(
    ::CMP.PredictedLiquidFraction, velocity_params, aps::CMP.AirProperties, tps::TDI.PS,
    Tₐ, ρₐ, state::P3State, shape::P3Shape; quad,
)
    (; K_therm) = aps
    L_f = TDI.Lf(tps, Tₐ)
    (; ρq_ice, ρn_ice, F_rim, ρ_rim) = state
    (; T_freeze, vent) = state.params

    # Melting acts on the frozen core: integrate over the ice-core PSD
    # (`logλ_core`) with the ice-core mass gradient. The ventilation factor uses
    # the blended whole-particle fall speed (C19 Eq A3), as in `ice_refreeze`.
    core = P3Shape(; logλ = shape.logλ_core, μ = shape.μ)
    v_term = mixed_particle_terminal_velocity(velocity_params, ρₐ, state)
    F_v = CO.ventilation_factor(vent, aps, v_term)
    N′ = size_distribution(state, core)
    fac = 4 * K_therm / L_f * (Tₐ - T_freeze)
    bnds = velocity_integral_bounds(state, core, v_term; p = 1e-6)
    melt_integrand = D -> ∂ice_mass_∂D(state, D) * F_v(D) * N′(D) / D

    # Split at the small-spherical threshold D_th (already a subinterval
    # boundary): D ≤ D_th melts completely to rain, D > D_th is retained as
    # liquid on the ice (C19 Eqs A1, A2).
    D_split = clamp(state.D_th, first(bnds), last(bnds))
    bnds_rain = map(D -> min(D, D_split), bnds)
    bnds_liq = map(D -> max(D, D_split), bnds)
    Q_rain = max(zero(fac), fac * integrate(melt_integrand, bnds_rain, quad))
    Q_liq = max(zero(fac), fac * integrate(melt_integrand, bnds_liq, quad))

    # Complete-melt part reduces number through the mean-mass band, so the number
    # rate is bounded and vanishes with the mass rate as the core empties. Rime
    # drains with the total core-mass loss at fixed F_rim.
    (; mean_mass_min, mean_mass_max) = state.params
    m̄ = clamp(ρq_ice / max(ρn_ice, floatmin(eltype(state))), mean_mass_min, mean_mass_max)
    dNdt_rain = Q_rain / m̄
    ΔL_core = Q_rain + Q_liq
    dLdt_rim = -ΔL_core * F_rim
    dBdt_rim = ifelse(ρ_rim > 0, -ΔL_core * F_rim / ρ_rim, zero(fac))
    return (; dLdt_ice = -ΔL_core, dLdt_rain = Q_rain, dNdt_rain, dLdt_liq = Q_liq, dLdt_rim, dBdt_rim)
end

"""
    ice_refreeze(velocity_params, aps, tps, Tₐ, ρₐ, state, shape; quad)

Refreeze the liquid on ice back into rime below the freezing
temperature (C19 Eq A4, A5). The frozen-core mass grows and the liquid shrinks
by the same rate, so the total mixed-phase mass and the ice number are
conserved. Nonzero only for `Tₐ < T_freeze` and `F_liq > 0`.

Integrated over the whole-particle PSD (`shape.logλ`) with the mixed-particle
terminal velocity for ventilation, mirroring the ventilated capacitance form of
[`ice_melt`](@ref); the refrozen mass joins the rime at the solid-ice density
`params.ρ_i` (a deviation from the reference 900 kg/m³; see the
[P3 liquid-fraction documentation](@ref P3-liquid-fraction)).

# Arguments
 - `velocity_params`: [`CMP.Chen2022VelType`](@ref)
 - `aps`: [`CMP.AirProperties`](@ref)
 - `tps`: thermodynamics parameters
 - `Tₐ`: temperature [K]
 - `ρₐ`: air density [kg/m³]
 - `state`: a [`P3State`](@ref)
 - `shape`: the diagnosed [`P3Shape`](@ref)

# Returns
A `NamedTuple` `(; dLdt_liq, dLdt_ice, dLdt_rim, dBdt_rim)` of volumetric
tendencies [kg/m³/s], [m³/m³/s]: `dLdt_liq = -dLdt_ice` (liquid → frozen core),
`dLdt_rim` (rime mass gain), `dBdt_rim` (rime volume gain).
"""
@inline function ice_refreeze(
    velocity_params, aps::CMP.AirProperties, tps::TDI.PS,
    Tₐ, ρₐ, state::P3State, shape::P3Shape; quad,
)
    (; K_therm) = aps
    L_f = TDI.Lf(tps, Tₐ)
    (; F_liq) = state
    (; T_freeze, vent, ρ_i) = state.params

    v_term = mixed_particle_terminal_velocity(velocity_params, ρₐ, state)
    F_v = CO.ventilation_factor(vent, aps, v_term)
    N′ = size_distribution(state, shape)
    # (T_freeze - Tₐ) driver ⇒ positive below freezing, continuous to zero at T_freeze
    fac = 4 * K_therm / L_f * (T_freeze - Tₐ)
    bnds = velocity_integral_bounds(state, shape, v_term; p = 1e-6)
    frz_integrand = D -> ∂ice_mass_∂D(state, D) * F_v(D) * N′(D) / D
    Q_frz = max(zero(fac), F_liq * fac * integrate(frz_integrand, bnds, quad))

    dBdt_rim = Q_frz / ρ_i
    return (; dLdt_liq = -Q_frz, dLdt_ice = Q_frz, dLdt_rim = Q_frz, dBdt_rim)
end

"""
    ice_shed(state, shape)

Shed the liquid on ice from particles larger than the shedding onset diameter into
rain (C19; Rasmussen and Heymsfield 1987). Only particles with maximum
dimension above `params.liquid.D_shd_onset` shed, and shedding scales with the
frozen-core rime fraction `F_rim` (unrimed wet snow retains its liquid). Active
only under [`CMP.PredictedLiquidFraction`](@ref).

The shed-able liquid mass concentration is the closed-form whole-particle liquid
moment above the onset diameter,

```math
L_\\mathrm{shd} = F_\\mathrm{rim}\\, F_\\mathrm{liq}\\,
    \\frac{\\pi \\rho_l}{6} \\int_{D_\\mathrm{shd}}^{\\infty} D^3\\, n(D)\\, \\mathrm{d}D ,
```

evaluated through the log-space upper incomplete gamma
([`log_upper_incomplete_gamma`](@ref)), which stays accurate over the physical
`λ D_shd` range in Float32 and underflows cleanly only in the extreme tail,
where shedding is negligible. It is shed as rain drops of mean diameter
`params.liquid.D_shd_drop`.

# Returns
A `NamedTuple` `(; L_shd, N_shd)`: the shed liquid mass concentration [kg/m³]
and the corresponding rain number concentration [1/m³]. Shedding is
instantaneous in the reference; the host applies these amounts limited to the
available liquid `ρq_liq`.
"""
@inline ice_shed(state::P3State, shape::P3Shape) = _ice_shed(state.params.liquid, state, shape)

@inline function _ice_shed(::CMP.NoLiquidFraction, state::P3State, shape::P3Shape)
    z = zero(eltype(state))
    return (; L_shd = z, N_shd = z)
end

@inline function _ice_shed(liquid::CMP.PredictedLiquidFraction, state::P3State, shape::P3Shape)
    (; ρn_ice, F_rim, F_liq) = state
    (; ρ_l) = state.params
    (; D_shd_onset, D_shd_drop) = liquid
    (; μ, logλ) = shape
    x₀ = exp(logλ) * D_shd_onset
    # log ∫_{D_shd_onset}^∞ D³ n(D) dD / ρn_ice via the log-space upper incomplete
    # gamma; ρn_ice stays outside the exponential.
    log_tail = -3 * logλ - SF.loggamma(μ + 1) + log_upper_incomplete_gamma(μ + 4, x₀)
    M₃ = ρn_ice * exp(log_tail)
    L_shd = F_rim * F_liq * (π * ρ_l / 6) * M₃
    N_shd = L_shd / (ρ_l * CO.volume_sphere_D(D_shd_drop))
    return (; L_shd, N_shd)
end

"""
    vapor_path_weight(liquid::PredictedLiquidFraction, F_liq)

C1 ramp weight selecting the vapor-exchange path from the liquid mass fraction:
`0` for a dry core (deposition/sublimation), `1` for a wet shell
(condensation/evaporation), with a smooth Hermite ramp on the band
`[F_dry, F_dry + ΔF_switch]`. Satisfies `w(0) = 0` and `w'(0) = 0`, so the
dry-ice vapor baseline is untouched (C19 §3d; the band is asserted inside
`(0, F_melt)` at construction).
"""
@inline vapor_path_weight((; F_dry, ΔF_switch)::CMP.PredictedLiquidFraction, F_liq) =
    _smoothstep(F_liq, F_dry, F_dry + ΔF_switch)

# C1 Hermite smoothstep: 0 below `lo`, 1 above `hi`, 3t² - 2t³ in between, with
# zero slope at both ends.
@inline function _smoothstep(x, lo, hi)
    t = clamp((x - lo) / (hi - lo), zero(x), one(x))
    return t * t * (3 - 2 * t)
end

"""
    ice_mean_particle_mass_min(FT)
    ice_mean_particle_mass_max(FT)

Bounds of the physical mean ice particle mass range [kg], matching the ice
number-adjustment bounds of the 2-moment scheme.
"""
@inline ice_mean_particle_mass_min(::Type{FT}) where {FT} = FT(1e-12)
@inline ice_mean_particle_mass_max(::Type{FT}) where {FT} = FT(1e-5)

"""
    ice_deposition_timescale(velocity_params, aps, tps, Tₐ, ρₐ, state, shape; quad)

Compute the vapor deposition relaxation timescale of the ice population from
its capacitance integral,

```math
τ_{dep} = \\frac{ρₐ q_{v,si}}{2π G_i ∫ D F_v(D) N'(D) dD},
```

with spherical capacitance `C = D/2`, following Morrison and Milbrandt (2015).
The timescale diverges as the population vanishes and shrinks as the
integrated particle surface grows.

# Arguments
 - `velocity_params`: [`CMP.Chen2022VelType`](@ref)
 - `aps`: [`CMP.AirProperties`](@ref)
 - `tps`: thermodynamics parameters
 - `Tₐ`: temperature (K)
 - `ρₐ`: air density
 - `state`: a [`P3State`](@ref) object
 - `shape`: The diagnosed [`P3Shape`](@ref), or a real `logλ` from which the
   shape is diagnosed via [`get_distribution_shape`](@ref)

# Keyword arguments
 - `quad`: quadrature rule (a `Quadrature.QuadratureRule`)

# Returns
- Deposition timescale [s], bounded above at `1e10` to stay finite.
"""
@inline function ice_deposition_timescale(
    velocity_params, aps::CMP.AirProperties, tps::TDI.PS,
    Tₐ, ρₐ, state::P3State, shape::P3Shape;
    quad,
)
    FT = eltype(state)
    (; vent) = state.params

    G = CO.G_func_ice(aps, tps, Tₐ)
    qᵥ_sat_ice = TDI.saturation_vapor_specific_content_over_ice(tps, Tₐ, ρₐ)

    v_term = ice_particle_terminal_velocity(velocity_params, ρₐ, state)
    F_v = CO.ventilation_factor(vent, aps, v_term)
    N′ = size_distribution(state, shape)

    bnds = velocity_integral_bounds(state, shape, v_term; p = 1e-6)
    dep_integrand = D -> D * F_v(D) * N′(D)
    ∫DFvN = integrate(dep_integrand, bnds, quad)

    denom = 2 * FT(π) * G * ∫DFvN
    return min(ρₐ * qᵥ_sat_ice / max(denom, floatmin(FT)), FT(1e10))
end
@inline ice_deposition_timescale(
    velocity_params, aps::CMP.AirProperties, tps::TDI.PS,
    Tₐ, ρₐ, state::P3State, logλ::Real;
    kw...,
) = ice_deposition_timescale(
    velocity_params, aps, tps, Tₐ, ρₐ, state, get_distribution_shape(state, logλ); kw...,
)

"""
    collision_cross_section_ice_liquid_coeffs(rᵢ)
    collision_cross_section_ice_liquid_coeffs(state, Dᵢ)

Monomial coefficients `(k₀, k₁, k₂)` of the ice-liquid collision cross-section as
a polynomial in the liquid diameter `Dₗ`,

```math
σ(Dᵢ, Dₗ) = π (rᵢ + Dₗ/2)² = k₀ + k₁ Dₗ + k₂ Dₗ²,
```

with `k₀ = π rᵢ²`, `k₁ = π rᵢ`, `k₂ = π/4`, where the ice effective radius
is `rᵢ = √(ice_area(state, Dᵢ)/π)`; see [`ice_area`](@ref).

Used in [`collision_cross_section_ice_liquid`](@ref)
"""
@inline collision_cross_section_ice_liquid_coeffs(rᵢ::FT) where {FT} =
    (π * rᵢ^2, π * rᵢ, FT(π / 4))
@inline collision_cross_section_ice_liquid_coeffs(state, Dᵢ) =
    collision_cross_section_ice_liquid_coeffs(√(ice_area(state, Dᵢ) / π))

"""
    collision_cross_section_ice_liquid(state, Dᵢ, Dₗ)

Ice-liquid collision cross-section [m²], `π (rᵢ(Dᵢ) + Dₗ/2)²`, evaluated by
Horner from the shared [`collision_cross_section_ice_liquid_coeffs`](@ref).
"""
collision_cross_section_ice_liquid(state, Dᵢ, Dₗ) =
    evalpoly(Dₗ, collision_cross_section_ice_liquid_coeffs(state, Dᵢ))

"""
    VolumetricCollisionRate(state, v_i, v_l)

Volumetric collision rate for ice-liquid collisions [m³/s], evaluated as
`(D_ice, D_liq) -> E * K * |vᵢ - vₗ|` where:
- `D_ice` and `D_liq` are the (maximum) diameters of the ice and liquid particles
- `E` is the collision efficiency
- `K` is the collision cross section
- `vᵢ` and `vₗ` are the terminal velocities of ice and liquid particles

Note that `E`, `K`, `vᵢ` and `vₗ` are all, in general, functions of `D_ice` and `D_liq`.

This is a component of integrals like

```math
∫ ∫ E * K * |vᵢ - vₗ| * N'_i * N'_l dD_i dD_l
```

The terminal-velocity closures `v_i` and `v_l` are fields, so consumers of the
collision rate can query the velocities and their structure directly: the
fall-speed crossing (see [`crossing_integral_bounds`](@ref)), the
[`velocity_breakpoints`](@ref), and the coefficients of a
[`CO.Chen2022VelocityCurve`](@ref) used by the closed-form rain integrals.
"""
struct VolumetricCollisionRate{S, VI, VL} <: Function
    state::S
    v_i::VI
    v_l::VL
end
@inline function (∂ₜV::VolumetricCollisionRate)(D_ice::FT, D_liq::FT) where {FT}
    E = FT(1)  # TODO - Make collision efficiency a function of Dᵢ and Dₗ
    K = collision_cross_section_ice_liquid(∂ₜV.state, D_ice, D_liq)
    return E * K * abs(∂ₜV.v_i(D_ice) - ∂ₜV.v_l(D_liq))
end

"""
    volumetric_collision_rate_integrand(velocity_params, ρₐ, state)

Construct the [`VolumetricCollisionRate`](@ref) for the Chen 2022 ice and
rain terminal velocities at air density `ρₐ`.

!!! note
    We use the same terminal velocity parametrization for cloud and rain water.
"""
volumetric_collision_rate_integrand(velocity_params, ρₐ, state) = VolumetricCollisionRate(
    state,
    ice_particle_terminal_velocity(velocity_params, ρₐ, state),
    CO.particle_terminal_velocity(velocity_params.rain, ρₐ),
)

"""
    compute_max_freeze_rate(aps, tps, velocity_params, ρₐ, Tₐ, state)

Returns a function `max_freeze_rate(Dᵢ)` that returns the maximum possible freezing rate [kg/s]
    for an ice particle of diameter `Dᵢ` [m]. Evaluates to `0` if `T ≥ T_freeze`.

# Arguments
- `aps`: [`CMP.AirProperties`](@ref)
- `tps`: `TDP.ThermodynamicsParameters`
- `velocity_params`: velocity parameterization, e.g. [`CMP.Chen2022VelType`](@ref)
- `ρₐ`: air density [kg/m³]
- `Tₐ`: air temperature [K]
- `state`: [`P3State`](@ref)

This rate represents the thermodynamic upper limit to collisional freezing,
which occurs when the heat transfer from the ice particle to the environment is
balanced by the latent heat of fusion.

From Eq (A7) in Musil (1970), [Musil1970](@cite).
"""
function compute_max_freeze_rate(aps, tps, velocity_params, ρₐ, Tₐ, state)
    (; D_vapor, K_therm) = aps
    cp_l = TDI.cp_l(tps)
    T_frz = TDI.T_freeze(tps)
    Lᵥ = TDI.Lᵥ(tps, Tₐ)
    L_f = TDI.Lf(tps, Tₐ)
    Tₛ = T_frz  # the surface of the ice particle is assumed to be at the freezing temperature
    ΔT = Tₛ - Tₐ  # temperature difference between the surface of the ice particle and the air
    Δρᵥ_sat =
        ρₐ * (  # saturation vapor density difference between the surface of the ice particle and the air
            TDI.p2q(tps, Tₛ, ρₐ, TDI.saturation_vapor_pressure_over_ice(tps, Tₛ)) -
            TDI.p2q(tps, Tₐ, ρₐ, TDI.saturation_vapor_pressure_over_ice(tps, Tₐ))
        )
    v_term = ice_particle_terminal_velocity(velocity_params, ρₐ, state)
    F_v = CO.ventilation_factor(state.params.vent, aps, v_term)
    # Musil (1970) dry-growth formula: the denominator `(L_f - cp_l·ΔT)`
    # represents the *net* latent heat per unit mass available to freeze a
    # colliding droplet. At Tₐ ≲ 220 K (ΔT ≳ L_f/cp_l ≈ 53 K with
    # T-dependent L_f, see Eq. A7 in Musil 1970), the denominator flips
    # sign, making `max_freeze_rate < 0` — which is unphysical. Cold air
    # is *further from* the dry/wet-growth transition, not closer to it:
    # the physical answer is `f_frz → 1` (every colliding droplet
    # freezes). We enforce that by returning `floatmax(FT)` when the
    # denominator is non-positive, so `min(∂ₜM_col, ∂ₜM_max) = ∂ₜM_col` and
    # `f_frz = 1`.
    denom = L_f - cp_l * ΔT
    function max_freeze_rate(Dᵢ)
        # fallback values typed by the promotion of the node and the captured state
        # (mixed plain/Dual under differentiation)
        FT = UT.promote_typeof(Dᵢ, ΔT, Δρᵥ_sat, denom)
        # `rate` is non-finite when `denom ≤ 0`; the selection below discards it
        rate = 2 * (π * Dᵢ) * F_v(Dᵢ) * (K_therm * ΔT + Lᵥ * D_vapor * Δρᵥ_sat) / denom
        # zero above the freezing temperature; floatmax when denom ≤ 0 (see above)
        return ifelse(Tₐ ≥ T_frz, zero(FT), ifelse(denom > 0, FT(rate), floatmax(FT)))
    end
    return max_freeze_rate
end

"""
    compute_local_rime_density(velocity_params, ρₐ, T, state)

Provides a function `ρ′_rim(Dᵢ, Dₗ)` that computes the local rime density [kg/m³]
    for a given ice particle diameter `Dᵢ` [m] and liquid particle diameter `Dₗ` [m].

# Arguments
- `velocity_params`: velocity parameterization, e.g. [`CMP.Chen2022VelType`](@ref)
- `ρₐ`: air density [kg/m³]
- `T`: temperature [K]
- `state`: [`P3State`](@ref)

# Returns
A function that computes the local rime density [kg/m³] using the equation:

```math
ρ'_{rim} = a + b R_i + c R_i^2
```
where
```math
R_i = -\\frac{ 10^6 ⋅ D_{liq} ⋅ |v_{liq} - v_{ice}| }{ 2 T_{sfc} }
```
and ``T_{sfc} < 0`` is the sub-zero surface temperature [°C], ``D_{liq}`` is the liquid
particle diameter [m], ``v_{liq/ice}`` is the particle terminal velocity [m/s].
So the units of ``R_i`` are [m² s⁻¹ °C⁻¹]. The units of ``ρ'_{rim}`` are [kg/m³].

We assume for simplicity that ``T_{sfc}`` equals ``T``, the ambient air temperature.
For real graupel, ``T_{sfc}`` is slightly higher than ``T`` due to latent heat release
of freezing liquid particles onto the ice particle. Morrison & Milbrandt (2013)
found little sensitivity to "realistic" increases in ``T_{sfc}``.

See also [`LocalRimeDensity`](@ref CloudMicrophysics.Parameters.LocalRimeDensity).

# Extended help

 Implementation follows Cober and List (1993), Eq. 16 and 17, and the P3 fortran code,
 `microphy_p3.f90`. The leading minus sign in ``R_i`` matches both: with ``T_{sfc} < 0``
 the minus makes ``R_i`` positive, and the fortran carries the explicit minus in
 `Ri = -(0.5e6 D_c) V_impact iTc` (its `0.5e6 D_c` equals ``10^6 D_{liq} / 2``).

 See also the P3 fortran code, Line 3315-3323, which extends the range of the calculation
 to ``R_i ≤ 12``, the upper limit of which then equals the solid bulk ice density,
 ``ρ_ice = 916.7 kg/m^3``.
 ``T_{sfc}`` is additionally bounded strictly below 0°C, a numerical guard against the
 ``R_i → ∞`` limit as ``T_{sfc} → 0``, carried over from the fortran code, Line 3380
 (`iTc = 1/min(-0.001, Tc)`); this bound is not part of the paper's continuous form.

 Note that Morrison & Milbrandt (2015) [MorrisonMilbrandt2015](@cite) only uses this
 parameterization for collisions with cloud droplets.
 For rain drops, they use a value near the solid bulk ice density, ``ρ^* = 900 kg/m^3``.
 We do not consider this distinction, and use this parameterization for all liquid particles.
"""
struct RimeDensityRate{R, FT, VI, VL} <: Function
    ρ_rim_local::R
    T°C::FT
    v_ice::VI
    v_liq::VL
end
@inline function (ρ′::RimeDensityRate)(Dᵢ, Dₗ)
    v_term = abs(ρ′.v_ice(Dᵢ) - ρ′.v_liq(Dₗ))
    return rime_density_at(ρ′, v_term, Dₗ)
end

"""
    rime_density_at(ρ′::RimeDensityRate, v_term, Dₗ)

Local rime density [kg/m³] at a precomputed sedimentation velocity difference
`v_term = |v_ice(Dᵢ) - v_liq(Dₗ)|`, shared between the collision-rate and rime-density
evaluations at the same liquid diameter node. See [`compute_local_rime_density`](@ref).
"""
@inline function rime_density_at(ρ′::RimeDensityRate, v_term, Dₗ)
    μm = 1_000_000  # m to μm factor, c.f. units of rₘ in Eq. 16 in Cober and List (1993)
    # Leading minus: Cober and List (1993), Eq. 16, and fortran `microphy_p3.f90`
    # (`Ri = -(0.5e6*D_c)*V_impact*iTc`); T°C < 0 then makes Rᵢ positive.
    Rᵢ = -(Dₗ * μm * v_term) / (2 * ρ′.T°C)
    return ρ′.ρ_rim_local(Rᵢ)
end

function compute_local_rime_density(velocity_params, ρₐ, T, state)
    (; T_freeze, ρ_rim_local) = state.params
    # Sub-zero surface temperature [°C], bounded strictly below 0°C. The bound is a
    # numerical guard against R_i → ∞ as T°C → 0, carried over from the fortran code
    # `microphy_p3.f90` Line 3380 (`iTc = 1/min(-0.001, Tc)`).
    T°C = min(T - T_freeze, -oftype(T_freeze, 1e-3))
    v_ice = ice_particle_terminal_velocity(velocity_params, ρₐ, state)
    v_liq = CO.particle_terminal_velocity(velocity_params.rain, ρₐ)
    return RimeDensityRate(ρ_rim_local, T°C, v_ice, v_liq)
end

"""
    get_liquid_integrals(n, ∂ₜV, m_liq, ρ′_rim, liq_bounds; quad)

Return a function `liquid_integrals(Dᵢ)` that computes the liquid particle integrals
    for a given ice particle diameter `Dᵢ`.

# Arguments
- `n`: liquid particle size distribution function `n(D)`
- `∂ₜV`: the [`VolumetricCollisionRate`](@ref) `∂ₜV(Dᵢ, D)`
- `m_liq`: liquid particle mass function `m_liq(D)`
- `ρ′_rim`: local rime density function `ρ′_rim(Dᵢ, D)`
- `liq_bounds`: integration bounds for liquid particles; the fall-speed
    crossing `∂ₜV.v_l(D) = ∂ₜV.v_i(Dᵢ)` is inserted as a subinterval boundary,
    see [`crossing_integral_bounds`](@ref)

# Keyword arguments
- `quad`: quadrature rule (a `Quadrature.QuadratureRule`)

# Notes
The function `liquid_integrals(Dᵢ)` returns a tuple `(∂ₜN_col, ∂ₜM_col, ∂ₜB_col)`
    of collision rates at `Dᵢ`, where:
- `∂ₜN_col`: number collision rate [1/s]
- `∂ₜM_col`: mass collision rate [kg/s]
- `∂ₜB_col`: rime volume collision rate [m³/s]
"""
@inline function get_liquid_integrals(n, ∂ₜV, m_liq, ρ′_rim, liq_bounds; quad)
    function liquid_integrals(Dᵢ)
        integrand = D -> begin
            ∂ₜV_D = ∂ₜV(Dᵢ, D)
            ∂ₜN = ∂ₜV_D * n(D)          # number collision rate
            ∂ₜM = ∂ₜN * m_liq(D)        # mass collision rate
            ∂ₜB = ∂ₜM / ρ′_rim(Dᵢ, D)   # rime volume collision rate
            return SA.SVector(∂ₜN, ∂ₜM, ∂ₜB)
        end
        bnds = crossing_integral_bounds(liq_bounds, ∂ₜV, Dᵢ)
        (∂ₜN_col, ∂ₜM_col, ∂ₜB_col) = integrate(integrand, bnds, quad)
        return ∂ₜN_col, ∂ₜM_col, ∂ₜB_col
    end
    return liquid_integrals
end

# `∂ₜV.v_i(Dᵢ)` and the cross-section coefficients derived from `ice_area(state, Dᵢ)`
# depend only on the outer diameter `Dᵢ`; hoist both once per outer node instead of
# reevaluating them at every inner quadrature node. `∂ₜV.v_l(D)` is likewise shared
# between the collision-rate and rime-density evaluations at each inner node.
@inline function get_liquid_integrals(
    n, ∂ₜV::VolumetricCollisionRate, m_liq, ρ′_rim::RimeDensityRate, liq_bounds; quad,
)
    function liquid_integrals(Dᵢ)
        v_i_at_Dᵢ = ∂ₜV.v_i(Dᵢ)
        coeffs = collision_cross_section_ice_liquid_coeffs(∂ₜV.state, Dᵢ)
        integrand = D -> begin
            v_l_at_D = ∂ₜV.v_l(D)
            v_term = abs(v_i_at_Dᵢ - v_l_at_D)
            E = one(v_term)  # TODO - Make collision efficiency a function of Dᵢ and Dₗ
            ∂ₜN = E * evalpoly(D, coeffs) * v_term * n(D)  # number collision rate
            ∂ₜM = ∂ₜN * m_liq(D)                           # mass collision rate
            ∂ₜB = ∂ₜM / rime_density_at(ρ′_rim, v_term, D)  # rime volume collision rate
            return SA.SVector(∂ₜN, ∂ₜM, ∂ₜB)
        end
        bnds = crossing_integral_bounds(liq_bounds, ∂ₜV, Dᵢ, v_i_at_Dᵢ)
        (∂ₜN_col, ∂ₜM_col, ∂ₜB_col) = integrate(integrand, bnds, quad)
        return ∂ₜN_col, ∂ₜM_col, ∂ₜB_col
    end
    return liquid_integrals
end

"""
    crossing_integral_bounds(liq_bounds, ∂ₜV, Dᵢ, [v_i_at_Dᵢ])

Insert the fall-speed crossing `∂ₜV.v_l(D) = ∂ₜV.v_i(Dᵢ)` into `liq_bounds`, so
that the derivative discontinuity of `|v_i(Dᵢ) - v_l(D)|` lies on a subinterval
boundary. `v_i_at_Dᵢ` defaults to `∂ₜV.v_i(Dᵢ)`; pass it explicitly to reuse an
already-computed value.

For a collision-rate integrand that does not carry the terminal-velocity
closures, the bounds are returned unchanged.

Called from [`get_liquid_integrals`](@ref).
"""
@inline function crossing_integral_bounds(
    liq_bounds::NTuple{2, Any}, ∂ₜV::VolumetricCollisionRate, Dᵢ, v_i_at_Dᵢ = ∂ₜV.v_i(Dᵢ),
)
    (D_min, D_max) = liq_bounds
    Dstar = crossover_diameter(v_i_at_Dᵢ, ∂ₜV.v_l, D_min, D_max)
    return (D_min, clamp(Dstar, D_min, D_max), D_max)
end
@inline crossing_integral_bounds(liq_bounds, ∂ₜV, Dᵢ, v_i_at_Dᵢ = nothing) = liq_bounds

"""
    crossover_diameter(v_target, v_l, D_min, D_max)

Find the diameter `D` in `[D_min, D_max]` where `v_l(D) = v_target`
"""
function crossover_diameter(v_target, v_l::F, D_min, D_max) where {F}
    FT = float(promote_type(typeof(v_target), typeof(D_min), typeof(D_max)))
    f(D) = v_l(D) - v_target
    maxiters = FT === Float32 ? 8 : 10
    sol = RS.find_zero(f,
        RS.BrentsMethod(FT(D_min), FT(D_max)), RS.CompactSolution(),
        FixedIterations{FT}(), maxiters,
    )
    return sol.root
end

"""
    liquid_species_segment(n_liq, m_liq, ρ′_rim, ∂ₜV, v_i_at_Dᵢ, coeffs, quad, a, b)

Hand-written Gauss quadrature sum of the `(∂ₜN, ∂ₜM, ∂ₜB)` collision-rate
integrand over one segment `[a, b]`, equivalent to `integrate` applied to the
integrand built by [`get_liquid_integrals`](@ref)'s `VolumetricCollisionRate`-specialized
method. `v_i_at_Dᵢ` and `coeffs` are the outer-node quantities hoisted by the caller.
"""
@inline function liquid_species_segment(
    n_liq, m_liq, ρ′_rim::RimeDensityRate, ∂ₜV::VolumetricCollisionRate,
    v_i_at_Dᵢ, coeffs, quad, a::FT, b::FT,
) where {FT}
    zero3 = SA.SVector(zero(FT), zero(FT), zero(FT))
    a < b || return zero3
    nnodes = quad.n
    scale_factor = (b - a) / 2
    shift = (a + b) / 2
    acc = zero3
    @inbounds for i in 1:nnodes
        y = node(quad, FT(i), nnodes)
        D = scale_factor * y + shift
        w = inv_weight_fun(quad, y) * weight(quad, FT(i), nnodes)
        v_term = abs(v_i_at_Dᵢ - ∂ₜV.v_l(D))
        E = one(v_term)  # TODO - Make collision efficiency a function of Dᵢ and Dₗ
        ∂ₜN = E * evalpoly(D, coeffs) * v_term * n_liq(D)  # number collision rate
        ∂ₜM = ∂ₜN * m_liq(D)                               # mass collision rate
        ∂ₜB = ∂ₜM / rime_density_at(ρ′_rim, v_term, D)      # rime volume collision rate
        acc += SA.SVector(∂ₜN, ∂ₜM, ∂ₜB) * w
    end
    return scale_factor * acc
end

"""
    rain_B_segment(n_liq, m_liq, ρ′_rim, ∂ₜV, v_i_at_Dᵢ, coeffs, quad, a, b)

Hand-written Gauss quadrature sum of the rime-volume-collision-rate integrand
over one segment `[a, b]`, equivalent to `integrate` applied to the B_rim
integrand in [`get_liquid_integrals_rain_closed`](@ref). `v_i_at_Dᵢ` and
`coeffs` are the outer-node quantities hoisted by the caller.
"""
@inline function rain_B_segment(
    n_liq, m_liq, ρ′_rim::RimeDensityRate, ∂ₜV::VolumetricCollisionRate,
    v_i_at_Dᵢ, coeffs, quad, a::FT, b::FT,
) where {FT}
    a < b || return zero(FT)
    nnodes = quad.n
    scale_factor = (b - a) / 2
    shift = (a + b) / 2
    acc = zero(FT)
    @inbounds for i in 1:nnodes
        y = node(quad, FT(i), nnodes)
        D = scale_factor * y + shift
        w = inv_weight_fun(quad, y) * weight(quad, FT(i), nnodes)
        v_term = abs(v_i_at_Dᵢ - ∂ₜV.v_l(D))
        E = one(v_term)  # TODO - Make collision efficiency a function of Dᵢ and Dₗ
        acc += E * evalpoly(D, coeffs) * v_term * n_liq(D) * m_liq(D) / rime_density_at(ρ′_rim, v_term, D) * w
    end
    return scale_factor * acc
end

"""
    closed_rain_inner_NM_setup(ai, bi, ci, D_min, D_max, λ)

Precompute, once per point, the [`gamma_inc_moment_channel_setup`](@ref) for
the ice-velocity channel (rate `λ`, moment orders `0:5`) and for each
rain-velocity channel `j` (rate `λ + ci[j]`, moment orders `bi[j] .+ (0:5)`).
Pass the result to [`closed_rain_inner_NM`](@ref)'s `channel_setups` argument
to avoid rebuilding it at every outer ice diameter.
"""
@inline function closed_rain_inner_NM_setup(ai, bi, ci, D_min, D_max, λ)
    ice_setup = gamma_inc_moment_channel_setup(0, λ, D_min, D_max)
    rain_setups = map((b, c) -> gamma_inc_moment_channel_setup(b, λ + c, D_min, D_max), bi, ci)
    return (ice_setup, rain_setups)
end

"""
    closed_rain_inner_NM(
        v_i_at_Dᵢ, Dstar, rᵢ, ρw, ai, bi, ci, D_min, D_max, N₀r, Dr_mean;
        [channel_setups],
    )

Closed-form `(∂ₜN_col, ∂ₜM_col)` for the rain inner integral at one outer ice
diameter, where `v_i_at_Dᵢ` is the ice particle terminal velocity there and
`Dstar` the fall-speed crossing from [`crossover_diameter`](@ref).
`channel_setups` defaults to a fresh [`closed_rain_inner_NM_setup`](@ref);
pass the per-point value from [`get_liquid_integrals_rain_closed`](@ref) to
avoid rebuilding it at every outer ice diameter.
"""
function closed_rain_inner_NM(
    v_i_at_Dᵢ, Dstar, rᵢ, ρw, ai, bi, ci, D_min, D_max, N₀r, Dr_mean;
    channel_setups = closed_rain_inner_NM_setup(ai, bi, ci, D_min, D_max, inv(Dr_mean)),
)
    FT = float(eltype(ai))
    (ice_setup, rain_setups) = channel_setups
    coeffs = SA.SVector(collision_cross_section_ice_liquid_coeffs(rᵢ))

    # Each channel's `gamma_inc_moment_channel_finish` gives the six
    # consecutive-order crossing-split moments at once (cross-section orders
    # 0,1,2 for the N moment, indices 1:3; the same orders shifted by the M
    # moment's base order 3, indices 4:6), weighted here by `coeffs` and the
    # channel's velocity-curve weight (`v_i_at_Dᵢ` for the ice channel,
    # `-ai[j]` for rain channel `j`).
    function channel_NM(setup, weight)
        moments = gamma_inc_moment_channel_finish(setup, D_min, Dstar, D_max)
        N_lo = @inbounds coeffs[1] * moments[1][1]
        N_hi = @inbounds coeffs[1] * moments[1][2]
        M_lo = @inbounds coeffs[1] * moments[4][1]
        M_hi = @inbounds coeffs[1] * moments[4][2]
        @inbounds for i in 2:lastindex(coeffs)
            N_lo += coeffs[i] * moments[i][1]
            N_hi += coeffs[i] * moments[i][2]
            M_lo += coeffs[i] * moments[i + 3][1]
            M_hi += coeffs[i] * moments[i + 3][2]
        end
        return (weight * N_lo, weight * N_hi, weight * M_lo, weight * M_hi)
    end

    (s_lo_N, s_hi_N, s_lo_M, s_hi_M) = channel_NM(ice_setup, v_i_at_Dᵢ)
    @inbounds for j in eachindex(ai)
        (lo_N, hi_N, lo_M, hi_M) = channel_NM(rain_setups[j], -ai[j])
        s_lo_N += lo_N
        s_hi_N += hi_N
        s_lo_M += lo_M
        s_hi_M += hi_M
    end
    mfac = ρw * CO.volume_sphere_D(one(FT))  # m_liq(D) = mfac Dₗ³
    return (N₀r * (s_lo_N - s_hi_N), N₀r * mfac * (s_lo_M - s_hi_M))  # number: D⁰, mass: D³
end

"""
    get_liquid_integrals_rain_closed(
        psd_r::RainParticlePDF_SB2006,
        n_r, ρₐ, L_r, N_r, state, ∂ₜV, m_liq, ρ′_rim, bounds_r; quad
    )

Return a function `liquid_integrals(Dᵢ) -> (∂ₜN_col, ∂ₜM_col, ∂ₜB_col)`
where N and M are the exact incomplete-gamma closed form and
B_rim is computed by quadrature, split at the fall-speed crossing.
The velocities and the rain velocity-curve coefficients come from the
[`VolumetricCollisionRate`](@ref) `∂ₜV`.
"""
@inline function get_liquid_integrals_rain_closed(
    psd_r::CMP.RainParticlePDF_SB2006,
    n_r, ρₐ, L_r, N_r, state, ∂ₜV::VolumetricCollisionRate, m_liq, ρ′_rim::RimeDensityRate, bounds_r; quad,
)
    FT = promote_type(eltype(state), UT.promote_typeof(ρₐ, L_r, N_r))
    ρw = psd_r.ρw
    (; N₀r, Dr_mean) = CM2.pdf_rain_parameters(psd_r, L_r / ρₐ, ρₐ, N_r)
    (; v_i, v_l) = ∂ₜV
    ai, bi, ci = SA.SVector(v_l.ai), SA.SVector(v_l.bi), SA.SVector(v_l.ci)
    D_min, D_max = bounds_r
    zero_rates = (zero(FT), zero(FT), zero(FT))
    # `D_min`, `D_max`, and the rain PSD slope `λ = inv(Dr_mean)` do not depend
    # on the outer ice diameter; build the closed-form channels' incomplete-gamma
    # setup once per point instead of once per outer node.
    channel_setups = closed_rain_inner_NM_setup(ai, bi, ci, D_min, D_max, inv(Dr_mean))
    function liquid_integrals(Dᵢ)
        if iszero(FD.value(N₀r)) || !(D_max > D_min)
            return zero_rates
        end
        v_i_at_Dᵢ = v_i(Dᵢ)
        rᵢ = sqrt(ice_area(state, Dᵢ) / π)
        coeffs = collision_cross_section_ice_liquid_coeffs(rᵢ)
        Dstar = crossover_diameter(v_i_at_Dᵢ, v_l, D_min, D_max)
        ∂ₜN_col, ∂ₜM_col = closed_rain_inner_NM(
            v_i_at_Dᵢ, Dstar, rᵢ, ρw, ai, bi, ci,
            D_min, D_max, N₀r, Dr_mean; channel_setups,
        )
        if !(isfinite(∂ₜN_col) && isfinite(∂ₜM_col))
            return zero_rates
        end
        # Reuse `v_i_at_Dᵢ` and `coeffs` (hoisted above) and share `v_l(D)` between the
        # collision rate and the rime density at each inner node.
        ∂ₜB_col = integrate(
            D -> begin
                v_l_at_D = v_l(D)
                v_term = abs(v_i_at_Dᵢ - v_l_at_D)
                E = one(v_term)  # TODO - Make collision efficiency a function of Dᵢ and Dₗ
                E * evalpoly(D, coeffs) * v_term * n_r(D) * m_liq(D) / rime_density_at(ρ′_rim, v_term, D)
            end,
            (D_min, clamp(Dstar, D_min, D_max), D_max),
            quad,
        )
        return (∂ₜN_col, ∂ₜM_col, ∂ₜB_col)
    end
    return liquid_integrals
end

"""
    get_combined_liquid_integrals(
        psd_r::RainParticlePDF_SB2006,
        n_c, n_r, ρₐ, L_r, N_r, state, ∂ₜV, m_liq, ρ′_rim, bounds_c, bounds_r; quad,
    )

Return a function `combined_integrals(Dᵢ) -> (∂ₜN_c_col, ∂ₜM_c_col, ∂ₜB_c_col, ∂ₜN_r_col, ∂ₜM_r_col, ∂ₜB_r_col)`
evaluating the cloud and rain inner integrals at a shared outer ice diameter `Dᵢ`.
Rain N and M are the exact incomplete-gamma closed form
([`closed_rain_inner_NM`](@ref), unchanged); cloud's three integrals and rain's
B_rim are the [`liquid_species_segment`](@ref)/[`rain_B_segment`](@ref) hand-written
per-segment quadrature sums, called in interleaved cloud/rain order and sharing
the outer-node quantities `v_i_at_Dᵢ` and `coeffs` between the two species.
"""
@inline function get_combined_liquid_integrals(
    psd_r::CMP.RainParticlePDF_SB2006,
    n_c, n_r, ρₐ, L_r, N_r, state, ∂ₜV::VolumetricCollisionRate, m_liq, ρ′_rim::RimeDensityRate,
    bounds_c, bounds_r; quad,
)
    FT = promote_type(eltype(state), UT.promote_typeof(ρₐ, L_r, N_r))
    ρw = psd_r.ρw
    (; N₀r, Dr_mean) = CM2.pdf_rain_parameters(psd_r, L_r / ρₐ, ρₐ, N_r)
    (; v_l) = ∂ₜV
    ai, bi, ci = SA.SVector(v_l.ai), SA.SVector(v_l.bi), SA.SVector(v_l.ci)
    D_min_r, D_max_r = bounds_r
    # `D_min_r`, `D_max_r`, and the rain PSD slope `λ = inv(Dr_mean)` do not depend on
    # the outer ice diameter; build the closed-form channels' incomplete-gamma setup
    # once per point instead of once per outer node.
    channel_setups = closed_rain_inner_NM_setup(ai, bi, ci, D_min_r, D_max_r, inv(Dr_mean))
    function combined_integrals(Dᵢ)
        v_i_at_Dᵢ = ∂ₜV.v_i(Dᵢ)
        rᵢ = sqrt(ice_area(state, Dᵢ) / π)
        coeffs = collision_cross_section_ice_liquid_coeffs(rᵢ)

        cb = crossing_integral_bounds(bounds_c, ∂ₜV, Dᵢ, v_i_at_Dᵢ)
        c1 = liquid_species_segment(n_c, m_liq, ρ′_rim, ∂ₜV, v_i_at_Dᵢ, coeffs, quad, cb[1], cb[2])

        rain_ok = !iszero(N₀r) && D_max_r > D_min_r
        Dstar_r = rain_ok ? crossover_diameter(v_i_at_Dᵢ, v_l, D_min_r, D_max_r) : D_min_r
        (∂ₜN_r, ∂ₜM_r) =
            rain_ok ?
            closed_rain_inner_NM(
                v_i_at_Dᵢ, Dstar_r, rᵢ, ρw, ai, bi, ci,
                D_min_r, D_max_r, N₀r, Dr_mean; channel_setups,
            ) : (zero(FT), zero(FT))
        rain_finite = rain_ok && isfinite(∂ₜN_r) && isfinite(∂ₜM_r)
        Dstar_r_clamped = clamp(Dstar_r, D_min_r, D_max_r)

        r1 =
            rain_finite ? rain_B_segment(n_r, m_liq, ρ′_rim, ∂ₜV, v_i_at_Dᵢ, coeffs, quad, D_min_r, Dstar_r_clamped) :
            zero(FT)
        c2 = liquid_species_segment(n_c, m_liq, ρ′_rim, ∂ₜV, v_i_at_Dᵢ, coeffs, quad, cb[2], cb[3])
        r2 =
            rain_finite ? rain_B_segment(n_r, m_liq, ρ′_rim, ∂ₜV, v_i_at_Dᵢ, coeffs, quad, Dstar_r_clamped, D_max_r) :
            zero(FT)

        (∂ₜN_c, ∂ₜM_c, ∂ₜB_c) = Tuple(c1 + c2)
        if !rain_finite
            ∂ₜN_r, ∂ₜM_r = zero(FT), zero(FT)
        end
        ∂ₜB_r = r1 + r2
        return (∂ₜN_c, ∂ₜM_c, ∂ₜB_c, ∂ₜN_r, ∂ₜM_r, ∂ₜB_r)
    end
    return combined_integrals
end

@inline _rain_inner_integrals(
    psd_r::CMP.RainParticlePDF_SB2006,
    n_r, ∂ₜV::VolumetricCollisionRate{<:Any, <:Any, <:CO.Chen2022VelocityCurve},
    m_liq, ρ′_rim, bounds_r, ρₐ, L_r, N_r, state; quad,
) = get_liquid_integrals_rain_closed(
    psd_r, n_r, ρₐ, L_r, N_r, state, ∂ₜV, m_liq, ρ′_rim, bounds_r;
    quad,
)
@inline _rain_inner_integrals(
    psd_r, n_r, ∂ₜV, m_liq, ρ′_rim, bounds_r, ρₐ, L_r, N_r, state; quad,
) = get_liquid_integrals(n_r, ∂ₜV, m_liq, ρ′_rim, bounds_r; quad)

"""
    ∫liquid_ice_collisions(
        n_i, ∂ₜM_max, cloud_integrals, rain_integrals, ice_bounds; [quad]
    )

Computes the bulk collision rate integrands between ice and liquid particles.

# Arguments
- `n_i`: ice particle size distribution function n_i(D)
- `∂ₜM_max`: maximum freezing rate function ∂ₜM_max(Dᵢ)
- `cloud_integrals`: an instance of [`get_liquid_integrals`](@ref) for cloud particles
- `rain_integrals`: an instance of [`get_liquid_integrals`](@ref) for rain particles
- `ice_bounds`: integration bounds for ice particles, from [`velocity_integral_bounds`](@ref)

# Keyword arguments
- `quad`: quadrature rule (a `Quadrature.QuadratureRule`)

# Returns
A tuple of 8 integrands, see [`∫liquid_ice_collisions`](@ref) for details.
"""
@inline function liquid_ice_collisions_partition(
    n, ∂ₜM_max_at_Dᵢ, ∂ₜN_c_col, ∂ₜM_c_col, ∂ₜB_c_col, ∂ₜN_r_col, ∂ₜM_r_col, ∂ₜB_r_col,
)
    # Partition the mass collisions between freezing and shedding
    ∂ₜM_col = ∂ₜM_c_col + ∂ₜM_r_col  # [kg / s]

    ∂ₜM_frz = min(∂ₜM_col, ∂ₜM_max_at_Dᵢ)
    # Branch on the primal so a differentiated zero selects the same branch as
    # its value under `ForwardDiff`.
    f_frz = iszero(FD.value(∂ₜM_col)) ? zero(∂ₜM_frz) : ∂ₜM_frz / ∂ₜM_col
    𝟙_wet = ∂ₜM_col > ∂ₜM_frz  # Used for wet densification

    # Integrating over `Dᵢ` gives another unit of `[m]`, so `[X / s / m]` --> `[X / s]`
    # ∂ₜX = ∫ ∂ₜX(Dᵢ) nᵢ(Dᵢ) dDᵢ
    return SA.SVector(
        n * ∂ₜM_c_col * f_frz,        # QCFRZ
        n * ∂ₜM_c_col * (1 - f_frz),  # QCSHD
        n * ∂ₜN_c_col,                # NCCOL
        n * ∂ₜM_r_col * f_frz,        # QRFRZ
        n * ∂ₜM_r_col * (1 - f_frz),  # QRSHD
        n * ∂ₜN_r_col,                # NRCOL
        n * ∂ₜM_col,                  # ∫M_col,      total collision rate
        n * ∂ₜB_c_col * f_frz,        # BCCOL,       ∂ₜB_rim source
        n * ∂ₜB_r_col * f_frz,        # BRCOL,       ∂ₜB_rim source
        n * 𝟙_wet * ∂ₜM_col,          # ∫𝟙_wet_M_col, wet growth indicator
    )
end

@inline function ∫liquid_ice_collisions(n_i, ∂ₜM_max, cloud_integrals, rain_integrals, ice_bounds; quad)
    @inline function liquid_ice_collisions_integrands(Dᵢ)
        # Inner integrals over liquid particle diameters
        ∂ₜN_c_col, ∂ₜM_c_col, ∂ₜB_c_col = cloud_integrals(Dᵢ)
        ∂ₜN_r_col, ∂ₜM_r_col, ∂ₜB_r_col = rain_integrals(Dᵢ)
        return liquid_ice_collisions_partition(
            n_i(Dᵢ), ∂ₜM_max(Dᵢ), ∂ₜN_c_col, ∂ₜM_c_col, ∂ₜB_c_col, ∂ₜN_r_col, ∂ₜM_r_col, ∂ₜB_r_col,
        )
    end
    return integrate(liquid_ice_collisions_integrands, ice_bounds, quad)
end

"""
    ∫liquid_ice_collisions(n_i, ∂ₜM_max, combined_integrals, ice_bounds; [quad])

Same as [`∫liquid_ice_collisions`](@ref)`(n_i, ∂ₜM_max, cloud_integrals, rain_integrals, ice_bounds; quad)`,
but with the cloud and rain inner integrals evaluated together by a single
`combined_integrals(Dᵢ) -> (∂ₜN_c_col, ∂ₜM_c_col, ∂ₜB_c_col, ∂ₜN_r_col, ∂ₜM_r_col, ∂ₜB_r_col)`
closure, e.g. [`get_combined_liquid_integrals`](@ref).
"""
@inline function ∫liquid_ice_collisions_combined(n_i, ∂ₜM_max, combined_integrals, ice_bounds; quad)
    # Inlined so the closure is not materialized. Its captured `ForwardDiff`
    # payload is roughly 1376*(N+1) bytes for an N-state layout, which crosses a
    # codegen stack-promotion boundary between the 24-state and 28-state layouts;
    # above it the object is heap-allocated once per category per substep. Inlining
    # lets the escape analysis delete the allocation entirely. Guarded by the
    # substep-solve allocation test in test/p3_multicategory_tests.jl.
    @inline function liquid_ice_collisions_integrands(Dᵢ)
        ∂ₜN_c_col, ∂ₜM_c_col, ∂ₜB_c_col, ∂ₜN_r_col, ∂ₜM_r_col, ∂ₜB_r_col = combined_integrals(Dᵢ)
        return liquid_ice_collisions_partition(
            n_i(Dᵢ), ∂ₜM_max(Dᵢ), ∂ₜN_c_col, ∂ₜM_c_col, ∂ₜB_c_col, ∂ₜN_r_col, ∂ₜM_r_col, ∂ₜB_r_col,
        )
    end
    return integrate(liquid_ice_collisions_integrands, ice_bounds, quad)
end

"""
    ∫liquid_ice_collisions(
        state, shape, psd_c, psd_r, L_c, N_c, L_r, N_r,
        aps, tps, vel, ρₐ, T, m_liq; [quad]
    )

Compute key liquid-ice collision rates and quantities. Used by [`bulk_liquid_ice_collision_sources`](@ref).

# Arguments
- `state`: [`P3State`](@ref)
- `shape`: the diagnosed [`P3Shape`](@ref)
- `psd_c`: [`CMP.CloudParticlePDF_SB2006`](@ref)
- `psd_r`: [`CMP.RainParticlePDF_SB2006`](@ref)
- `L_c`: cloud liquid water content [kg/m³]
- `N_c`: cloud liquid water number concentration [1/m³]
- `L_r`: rain water content [kg/m³]
- `N_r`: rain number concentration [1/m³]
- `aps`: [`CMP.AirProperties`](@ref)
- `tps`: `TDP.ThermodynamicsParameters`
- `vel`: velocity parameterization, e.g. [`CMP.Chen2022VelType`](@ref)
- `ρₐ`: air density [kg/m³]
- `T`: temperature [K]
- `m_liq`: liquid particle mass function `m_liq(D)`

# Keyword arguments
- `quad`: A `QuadratureRule` instance

# Returns
A tuple `(QCFRZ, QCSHD, NCCOL, QRFRZ, QRSHD, NRCOL, ∫M_col, BCCOL, BRCOL, ∫𝟙_wet_M_col)`, where:
1. `QCFRZ` - Cloud mass collision rate due to freezing [kg/s]
2. `QCSHD` - Cloud mass collision rate due to shedding [kg/s]
3. `NCCOL` - Cloud number collision rate [1/s]
4. `QRFRZ` - Rain mass collision rate due to freezing [kg/s]
5. `QRSHD` - Rain mass collision rate due to shedding [kg/s]
4. `NRCOL` - Rain number collision rate [1/s]
5. `∫M_col` - Total collision rate [kg/s]
6. `BCCOL` - Cloud rime volume source [m³/m³/s]
7. `BRCOL` - Rain rime volume source [m³/m³/s]
8. `∫𝟙_wet_M_col` - Wet growth indicator [kg/s]
"""
@inline function ∫liquid_ice_collisions(
    state, shape::P3Shape,
    psd_c, psd_r, L_c, N_c, L_r, N_r,
    aps, tps, vel, ρₐ, T, m_liq; quad,
)
    FT = eltype(state)

    # Particle size distributions
    n_c = DT.size_distribution(psd_c, L_c / ρₐ, ρₐ, N_c)  # n_c(Dₗ)
    n_r = DT.size_distribution(psd_r, L_r / ρₐ, ρₐ, N_r)  # n_r(Dₗ)
    n_i = DT.size_distribution(state, shape)              # n_i(Dᵢ)

    # Integrand components; the collision rate carries the ice and liquid
    # terminal-velocity closures used below
    # NOTE: We assume collision efficiency, shape (spherical), and terminal velocity is the
    #   same for cloud and precipitating liquid particles ⟹ same volumetric collision rate, ∂ₜV
    ∂ₜV = volumetric_collision_rate_integrand(vel, ρₐ, state)  # ∂ₜV(Dᵢ, Dₗ)
    ρ′_rim = compute_local_rime_density(vel, ρₐ, T, state)  # ρ′_rim(Dᵢ, Dₗ)
    ∂ₜM_max = compute_max_freeze_rate(aps, tps, vel, ρₐ, T, state)  # ∂ₜM_max(Dᵢ)

    p = FT(0.00001)
    ice_bounds = velocity_integral_bounds(state, shape, ∂ₜV.v_i; p)
    bounds_c = CM2.get_size_distribution_bounds(psd_c, L_c / ρₐ, ρₐ, N_c, p)
    bounds_r = CM2.get_size_distribution_bounds(psd_r, L_r / ρₐ, ρₐ, N_r, p)

    # The freeze/shed partition and the wet-growth indicator change branch at the
    # wet-growth onset diameter, so the onset is a subinterval boundary of the
    # outer integral
    (D_wet₁, D_wet₂) = wet_growth_onset_diameter(
        psd_c, psd_r, ∂ₜV, ∂ₜM_max, state,
        L_c, N_c, L_r, N_r, ρₐ, bounds_r,
        first(ice_bounds), last(ice_bounds),
    )
    ice_bounds = Tuple(
        SA.sort(
            SA.SVector(
                ice_bounds...,
                clamp(D_wet₁, first(ice_bounds), last(ice_bounds)),
                clamp(D_wet₂, first(ice_bounds), last(ice_bounds)),
            ),
        ),
    )

    return _∫liquid_ice_collisions_inner(
        psd_r, n_c, n_r, n_i, ∂ₜV, ρ′_rim, m_liq, ∂ₜM_max,
        bounds_c, bounds_r, ice_bounds, ρₐ, L_r, N_r, state; quad,
    )
end

# Fast path: cloud and rain-B_rim inner integrals evaluated by the interleaved
# `get_combined_liquid_integrals` restructure; rain N/M stay the exact closed form
# ([`closed_rain_inner_NM`](@ref)), for the (SB2006-exp rain PSD, Chen-2022 velocity) pair.
@inline function _∫liquid_ice_collisions_inner(
    psd_r::CMP.RainParticlePDF_SB2006, n_c, n_r, n_i,
    ∂ₜV::VolumetricCollisionRate{<:Any, <:Any, <:CO.Chen2022VelocityCurve},
    ρ′_rim::RimeDensityRate, m_liq, ∂ₜM_max, bounds_c, bounds_r, ice_bounds, ρₐ, L_r, N_r, state; quad,
)
    combined_integrals = get_combined_liquid_integrals(
        psd_r, n_c, n_r, ρₐ, L_r, N_r, state, ∂ₜV, m_liq, ρ′_rim, bounds_c, bounds_r; quad,
    )
    return ∫liquid_ice_collisions_combined(n_i, ∂ₜM_max, combined_integrals, ice_bounds; quad)
end
# Numerical fallback for any other PSD/velocity type: cloud and rain inner integrals
# evaluated by two independent `get_liquid_integrals`/`_rain_inner_integrals` closures.
@inline function _∫liquid_ice_collisions_inner(
    psd_r, n_c, n_r, n_i, ∂ₜV, ρ′_rim, m_liq, ∂ₜM_max,
    bounds_c, bounds_r, ice_bounds, ρₐ, L_r, N_r, state; quad,
)
    cloud_integrals = get_liquid_integrals(n_c, ∂ₜV, m_liq, ρ′_rim, bounds_c; quad)
    rain_integrals = _rain_inner_integrals(
        psd_r, n_r, ∂ₜV, m_liq, ρ′_rim, bounds_r, ρₐ, L_r, N_r, state; quad,
    )
    return ∫liquid_ice_collisions(n_i, ∂ₜM_max, cloud_integrals, rain_integrals, ice_bounds; quad)
end

"""
    wet_growth_onset_diameter(
        psd_c, psd_r, ∂ₜV, ∂ₜM_max, state,
        L_c, N_c, L_r, N_r, ρₐ, bounds_r, D_lo, D_hi,
    )

Return up to two ice diameters in `[D_lo, D_hi]` where the collected liquid
mass rate balances the freeze limit `∂ₜM_max` (the boundaries of the wet-growth
window; `D_lo` stands in for absent crossings). The balance is evaluated with
the liquid fall speeds simplified: the cloud collection term neglects the
droplet fall speed relative to the ice fall speed, and the rain term evaluates
the rain fall speed once at the mean rain diameter, so both reduce to polynomial
moments of their size distributions. Crossings are located on a log-spaced scan
of the interval and refined by fixed-iteration bisection.

This applies to the (`CMP.CloudParticlePDF_SB2006`,
`CMP.RainParticlePDF_SB2006`) distributions with a
[`CO.Chen2022VelocityCurve`](@ref) liquid velocity; for any other combination
`(D_lo, D_lo)` is returned.
"""
function wet_growth_onset_diameter(
    psd_c::CMP.CloudParticlePDF_SB2006, psd_r::CMP.RainParticlePDF_SB2006,
    ∂ₜV::VolumetricCollisionRate{<:Any, <:Any, <:CO.Chen2022VelocityCurve},
    ∂ₜM_max, state,
    L_c, N_c, L_r, N_r, ρₐ, bounds_r, D_lo, D_hi,
)
    (; v_i, v_l) = ∂ₜV
    FT = promote_type(eltype(state), UT.promote_typeof(L_c, N_c, L_r, N_r, ρₐ))
    πFT = FT(π)
    # Locate the onset window with the cloud droplet fall speed neglected (cloud
    # droplets fall far slower than the ice sizes that matter here) and the rain
    # fall speed evaluated once at the mean rain diameter (rain and ice fall
    # speeds are comparable, so dropping it entirely biases the search):
    #   ∫ K(D, Dₗ) n(Dₗ) m_liq(Dₗ) dDₗ = ∑ⱼ Kⱼ(rᵢ) (ρw π/6) M⁽ʲ⁺³⁾,
    # with K quadratic in Dₗ and M⁽ᵏ⁾ the (untruncated) size-distribution
    # moments. The onset diameters only bound the outer-integral subintervals;
    # the collision rates entering the physics keep the full fall-speed difference.
    (; λc, νcD, μcD) = CM2.pdf_cloud_parameters(psd_c, L_c / ρₐ, ρₐ, N_c)
    ρw = psd_c.ρw
    mfac = ρw * CO.volume_sphere_D(one(FT))
    M₃ = mfac * DT.generalized_gamma_Mⁿ(νcD, μcD, λc, N_c, 3)
    M₄ = mfac * DT.generalized_gamma_Mⁿ(νcD, μcD, λc, N_c, 4)
    M₅ = mfac * DT.generalized_gamma_Mⁿ(νcD, μcD, λc, N_c, 5)

    (; N₀r, Dr_mean) = CM2.pdf_rain_parameters(psd_r, L_r / ρₐ, ρₐ, N_r)
    mfac_r = psd_r.ρw * CO.volume_sphere_D(one(FT))
    M₃r = mfac_r * N₀r * FT(6) * Dr_mean^4    # k=3: k! = 6
    M₄r = mfac_r * N₀r * FT(24) * Dr_mean^5   # k=4: k! = 24
    M₅r = mfac_r * N₀r * FT(120) * Dr_mean^6  # k=5: k! = 120
    v_l_r = v_l(Dr_mean)

    function excess_mass_rate(D)
        v = v_i(D)
        rᵢ = sqrt(ice_area(state, D) / πFT)
        (k₀, k₁, k₂) = collision_cross_section_ice_liquid_coeffs(rᵢ)
        cloud_rate = v * (k₀ * M₃ + k₁ * M₄ + k₂ * M₅)
        rain_rate = abs(v - v_l_r) * (k₀ * M₃r + k₁ * M₄r + k₂ * M₅r)
        return cloud_rate + rain_rate - ∂ₜM_max(D)
    end
    # The balance can cross twice (a wet-growth window: collection outgrows the
    # freeze limit at intermediate sizes and falls behind again at large sizes),
    # so locate sign changes on a log-spaced grid, then refine each crossing in
    # log diameter within its grid bracket.
    llo, lhi = log(FT(D_lo)), log(FT(D_hi))
    n_scan = 16
    Δl = (lhi - llo) / n_scan
    maxiters = FT === Float32 ? 8 : 10
    tol = FixedIterations{FT}()
    function refine_crossing(l₁, l₂)
        sol = RS.find_zero(l -> excess_mass_rate(exp(l)),
            RS.BrentsMethod(l₁, l₂), RS.CompactSolution(),
            tol, maxiters,
        )
        return exp(sol.root)
    end
    onset₁ = FT(D_lo)
    onset₂ = FT(D_lo)
    l_prev = llo
    g_prev = excess_mass_rate(FT(D_lo))
    for i in 1:n_scan
        l = llo + i * Δl
        g = excess_mass_rate(exp(l))
        if g * g_prev < 0
            root = refine_crossing(l_prev, l)
            if onset₁ == FT(D_lo)
                onset₁ = root
            elseif onset₂ == FT(D_lo)
                onset₂ = root
            end
        end
        l_prev = l
        g_prev = g
    end
    return onset₁, onset₂
end
wet_growth_onset_diameter(
    psd_c, psd_r, ∂ₜV, ∂ₜM_max, state,
    L_c, N_c, L_r, N_r, ρₐ, bounds_r, D_lo, D_hi,
) = (D_lo, D_lo)

"""
    bulk_liquid_ice_collision_sources(
        state, shape,
        psd_c, psd_r, L_c, N_c, L_r, N_r,
        aps, tps, vel, ρₐ, T,
    )

Computes the bulk rates for ice and liquid particle collisions.

# Arguments
- `state`: the [`P3State`](@ref)
- `shape`: the diagnosed [`P3Shape`](@ref)
- `psd_c`: a [`CMP.CloudParticlePDF_SB2006`](@ref)
- `psd_r`: a [`CMP.RainParticlePDF_SB2006`](@ref)
- `L_c`: cloud liquid water content [kg/m³]
- `N_c`: cloud liquid water number concentration [1/m³]
- `L_r`: rain water content [kg/m³]
- `N_r`: rain number concentration [1/m³]
- `aps`: [`CMP.AirProperties`](@ref)
- `tps`: thermodynamics parameters
- `vel`: the velocity parameterization, e.g. [`CMP.Chen2022VelType`](@ref)
- `ρₐ`: air density [kg/m³]
- `T`: temperature [K]

# Returns
A `NamedTuple` of `(; ∂ₜq_c, ∂ₜq_r, ∂ₜN_c, ∂ₜN_r, ∂ₜL_rim, ∂ₜL_ice, ∂ₜB_rim)`, where:
1. `∂ₜq_c`: cloud liquid water content tendency [kg/kg/s]
2. `∂ₜq_r`: rain water content tendency [kg/kg/s]
3. `∂ₜN_c`: cloud number concentration tendency [1/m³/s]
4. `∂ₜN_r`: rain number concentration tendency [1/m³/s]
5. `∂ₜL_rim`: riming mass tendency [kg/m³/s]
6. `∂ₜL_ice`: ice water content tendency [kg/m³/s]
7. `∂ₜB_rim`: rime volume tendency [m³/m³/s]

Dispatches on the liquid-fraction treatment. Under
[`CMP.PredictedLiquidFraction`](@ref) the non-frozen collected liquid is
retained on the ice instead of shed to rain, and the wet-growth densification
is off; the return gains the retained-liquid source `∂ₜL_liq` [kg/m³/s]. See
the [P3 liquid-fraction documentation](@ref P3-liquid-fraction).
"""
@inline bulk_liquid_ice_collision_sources(
    state, shape::P3Shape,
    psd_c, psd_r, L_c, N_c, L_r, N_r,
    aps, tps, vel, ρₐ, T; quad,
) = _bulk_liquid_ice_collision_sources(
    state.params.liquid, state, shape,
    psd_c, psd_r, L_c, N_c, L_r, N_r,
    aps, tps, vel, ρₐ, T; quad,
)

@inline function _bulk_liquid_ice_collision_sources(
    ::CMP.NoLiquidFraction, state, shape::P3Shape,
    psd_c, psd_r, L_c, N_c, L_r, N_r,
    aps, tps, vel, ρₐ, T; quad,
)
    FT = promote_type(eltype(state), UT.promote_typeof(L_c, N_c, L_r, N_r, ρₐ, T))
    (; τ_wet, ρ_i) = state.params
    D_shd = FT(1e-3) # 1mm  # TODO: Externalize this parameter

    ρw = psd_c.ρw
    @assert ρw == psd_r.ρw "Cloud and rain should have the same liquid water density"
    m_liq(Dₗ) = ρw * CO.volume_sphere_D(Dₗ)

    rates = ∫liquid_ice_collisions(
        state, shape,
        psd_c, psd_r, L_c, N_c, L_r, N_r,
        aps, tps, vel, ρₐ, T, m_liq; quad,
    )
    (QCFRZ, QCSHD, NCCOL, QRFRZ, QRSHD, NRCOL, ∫∂ₜM_col, BCCOL, BRCOL, ∫𝟙_wet_M_col) = rates

    # Bulk wet growth fraction
    f_wet = iszero(FD.value(∫∂ₜM_col)) ? zero(∫∂ₜM_col) : ∫𝟙_wet_M_col / ∫∂ₜM_col

    # Shedding of rain
    # QRSHD = ∫∂ₜM_col - (QCFRZ + QRFRZ)
    NRSHD = QRSHD / m_liq(D_shd)
    # NCSHD = QCSHD / m_liq(D_shd)

    # Densification of rime
    (; ρq_ice, F_rim, ρ_rim) = state
    B_rim = iszero(FD.value(ρ_rim)) ? zero(ρ_rim) : (ρq_ice * F_rim) / ρ_rim  # from: ρ_rim = L_rim / B_rim
    QIWET = f_wet * ρq_ice * (1 - F_rim) / τ_wet   # densification of rime mass
    BIWET = f_wet * (ρq_ice / ρ_i - B_rim) / τ_wet  # densification of rime volume

    # Bulk rates
    ## Liquid phase
    ∂ₜq_c = (-QCFRZ - QCSHD) / ρₐ
    ∂ₜq_r = (-QRFRZ + QCSHD) / ρₐ
    ∂ₜN_c = -NCCOL
    ∂ₜN_r = -NRCOL + NRSHD
    ## Ice phase
    ∂ₜL_rim = QCFRZ + QRFRZ + QIWET
    ∂ₜL_ice = QCFRZ + QRFRZ
    # ∂ₜN_ice = 0
    ∂ₜB_rim = BCCOL + BRCOL + BIWET

    return @NamedTuple{∂ₜq_c::FT, ∂ₜq_r::FT, ∂ₜN_c::FT, ∂ₜN_r::FT, ∂ₜL_rim::FT, ∂ₜL_ice::FT, ∂ₜB_rim::FT}(
        (∂ₜq_c, ∂ₜq_r, ∂ₜN_c, ∂ₜN_r, ∂ₜL_rim, ∂ₜL_ice, ∂ₜB_rim)
    )
end

# Collection routing under predicted liquid fraction (C19 §3c): the non-frozen
# collected liquid is retained on the ice as `∂ₜL_liq` instead of shed to rain
# (above freezing the freeze fraction is zero, so all collected liquid is
# retained), and the wet-growth rime densification is off (the excess resides
# in the liquid mass).
@inline function _bulk_liquid_ice_collision_sources(
    ::CMP.PredictedLiquidFraction, state, shape::P3Shape,
    psd_c, psd_r, L_c, N_c, L_r, N_r,
    aps, tps, vel, ρₐ, T; quad,
)
    FT = promote_type(eltype(state), UT.promote_typeof(L_c, N_c, L_r, N_r, ρₐ, T))
    ρw = psd_c.ρw
    @assert ρw == psd_r.ρw "Cloud and rain should have the same liquid water density"
    m_liq(Dₗ) = ρw * CO.volume_sphere_D(Dₗ)

    rates = ∫liquid_ice_collisions(
        state, shape,
        psd_c, psd_r, L_c, N_c, L_r, N_r,
        aps, tps, vel, ρₐ, T, m_liq; quad,
    )
    (QCFRZ, QCSHD, NCCOL, QRFRZ, QRSHD, NRCOL, _, BCCOL, BRCOL, _) = rates

    # Bulk rates
    ## Liquid phase: all collected mass and number leave cloud and rain
    ∂ₜq_c = (-QCFRZ - QCSHD) / ρₐ
    ∂ₜq_r = (-QRFRZ - QRSHD) / ρₐ
    ∂ₜN_c = -NCCOL
    ∂ₜN_r = -NRCOL
    ## Ice phase: the frozen part rimes; the non-frozen part is retained liquid
    ∂ₜL_rim = QCFRZ + QRFRZ
    ∂ₜL_ice = QCFRZ + QRFRZ
    ∂ₜB_rim = BCCOL + BRCOL
    ∂ₜL_liq = QCSHD + QRSHD

    return @NamedTuple{∂ₜq_c::FT, ∂ₜq_r::FT, ∂ₜN_c::FT, ∂ₜN_r::FT, ∂ₜL_rim::FT, ∂ₜL_ice::FT, ∂ₜB_rim::FT, ∂ₜL_liq::FT}(
        (∂ₜq_c, ∂ₜq_r, ∂ₜN_c, ∂ₜN_r, ∂ₜL_rim, ∂ₜL_ice, ∂ₜB_rim, ∂ₜL_liq)
    )
end


"""
    collision_cross_section_ice_ice(state_1, D_1, state_2, D_2)
    collision_cross_section_ice_ice(state, D_1, D_2)

Ice-ice collision cross-section [m²], `π (r₁(D_1) + r₂(D_2))²`, where the ice
effective radius is `rₖ(D) = √(ice_area(stateₖ, D) / π)`; see [`ice_area`](@ref).
The three-argument method uses a single `state` for both particles (equal-state
collisions), used in [`ice_self_collection`](@ref); the four-argument method
takes distinct states for the two particles, used in inter-category collection
([`inter_category_collection`](@ref)).
"""
function collision_cross_section_ice_ice(state_1, D_1, state_2, D_2)
    r_eff(state, D) = √(ice_area(state, D) / π)
    return π * (r_eff(state_1, D_1) + r_eff(state_2, D_2))^2  # collision cross section
end
collision_cross_section_ice_ice(state, D_1, D_2) =
    collision_cross_section_ice_ice(state, D_1, state, D_2)

"""
    ice_self_collection(state, shape, vel, ρₐ; [quad])

Computes the ice self-collection (aggregation) rate, which decreases the ice number concentration
while leaving mass, rime mass, and rime volume unchanged.

# Arguments
- `state`: [`P3State`](@ref)
- `shape`: the diagnosed [`P3Shape`](@ref)
- `vel`: the velocity parameterization, e.g. [`CMP.Chen2022VelType`](@ref)
- `ρₐ`: air density [kg/m³]

# Keyword arguments
- `quad`: quadrature rule (a `Quadrature.QuadratureRule`)

# Returns
A `NamedTuple` of `(; dNdt)`, where:
1. `dNdt`: ice number concentration tendency due to self-collection [1/m³/s] (always positive or zero, represents a loss rate)
"""
@inline function ice_self_collection(state, shape::P3Shape, vel, ρₐ; quad)
    n_i = DT.size_distribution(state, shape)
    v_ice = ice_particle_terminal_velocity(vel, ρₐ, state)

    p = eps(one(ρₐ))
    ice_bounds = velocity_integral_bounds(state, shape, v_ice; p)
    D_min, D_max = ice_bounds[1], ice_bounds[end]

    function inner_integral(D_1)
        # Inner integral over D_2 ∈ [D_1, D_max] (the upper triangle). Its
        # subinterval boundaries are the P3 regime breakpoints restricted to that
        # window: clamping each breakpoint up to D_1 drops those below it to
        # zero-width (no-op) subintervals. v_ice(D_1) and n_i(D_1) do not vary
        # over the inner integral, so evaluate them once here.
        v_1 = v_ice(D_1)
        n_1 = n_i(D_1)
        collision_rate =
            D_2 -> collision_cross_section_ice_ice(state, D_1, D_2) * abs(v_1 - v_ice(D_2)) * n_i(D_2)
        D_lo = clamp(D_1, D_min, D_max)
        inner_bounds = map(D -> max(D, D_lo), ice_bounds)
        return n_1 * integrate(collision_rate, inner_bounds, quad)
    end

    # Integrate the upper triangle D_1 ≤ D_2, counting each unordered particle
    # pair once — the self-collection rate.
    dNdt = integrate(inner_integral, ice_bounds, quad)
    return (; dNdt)
end
