
# The aspect-ratio types live in `CMP` (`ParametersP3` stores the choice); the
# functor methods live here, where `ϕᵢ` is defined.
const AspectRatio = CMP.AspectRatio
const Oblate = CMP.Oblate
const NoAspectRatio = CMP.NoAspectRatio
@inline (::Oblate)(state, D) = cbrt(ϕᵢ(state, D))
@inline (::NoAspectRatio)(state, D) = one(D)

# Callable returned by `ice_particle_terminal_velocity`: piecewise small/large-ice
# Chen 2022 velocity scaled by the aspect-ratio factor from `state.params`.
struct P3IceParticleVelocityFunctor{FT, VS, VL, S} <: Function
    v_term_small::VS
    v_term_large::VL
    D_cutoff::FT
    state::S
end
@inline function (f::P3IceParticleVelocityFunctor)(D)
    vₜ = ifelse(D <= f.D_cutoff, f.v_term_small(D), f.v_term_large(D))
    v = vₜ * f.state.params.aspect_ratio(f.state, D)
    return min(v, f.state.params.v_term_ice_max)
end

"""
    velocity_breakpoints(v_term)

Diameters where the terminal-velocity closure `v_term` changes functional form.
Integrals with `v_term` in the integrand place these on subinterval boundaries,
see [`velocity_integral_bounds`](@ref).
"""
velocity_breakpoints(f::P3IceParticleVelocityFunctor) = (f.D_cutoff,)
velocity_breakpoints(::CO.Chen2022VelocityCurve) = ()

"""
    ice_particle_terminal_velocity(velocity_params, ρₐ, state::P3State)

Return a single-argument function `v_term(D)` that gives the Chen 2022
terminal velocity of an ice particle of maximum dimension `D`, scaled by the
aspect-ratio factor selected by `state.params.aspect_ratio`.

# Arguments
 - `velocity_params`: A [`CMP.Chen2022VelType`](@ref)
 - `ρₐ`: Air density [kg/m³]
 - `state`: A [`P3State`](@ref)
"""
@inline function ice_particle_terminal_velocity(
    velocity_params::CMP.Chen2022VelType, ρₐ, state::P3State,
)
    FT = typeof(ρₐ)
    (; small_ice, large_ice) = velocity_params
    D_cutoff = small_ice.cutoff
    ρᵢ = FT(916.7)  # TODO: Use parameter
    v_term_small = CO.particle_terminal_velocity(small_ice, ρₐ, ρᵢ)
    v_term_large = CO.particle_terminal_velocity(large_ice, ρₐ, ρᵢ)
    return P3IceParticleVelocityFunctor(v_term_small, v_term_large, D_cutoff, state)
end

# Callable returned by `mixed_particle_terminal_velocity`: the liquid-fraction
# blend of the ice-core and pure-drop (Chen rain) terminal velocities.
struct P3MixedParticleVelocityFunctor{VI, VD, FT} <: Function
    v_ice::VI
    v_drop::VD
    F_liq::FT
end
@inline (f::P3MixedParticleVelocityFunctor)(D) = liquid_blend(f.F_liq, f.v_ice(D), f.v_drop(D))
velocity_breakpoints(f::P3MixedParticleVelocityFunctor) = velocity_breakpoints(f.v_ice)

"""
    mixed_particle_terminal_velocity(velocity_params, ρₐ, state::P3State)

Return a single-argument function `v_term(D)` giving the whole (mixed-phase)
particle terminal velocity: the liquid-fraction blend
`(1 - F_liq) v_ice(D) + F_liq v_drop(D)` of the ice-core velocity
[`ice_particle_terminal_velocity`](@ref) and the Chen 2022 rain velocity
(C19 Eq 13). Reduces to the ice-core velocity at `F_liq = 0`.

# Arguments
 - `velocity_params`: A [`CMP.Chen2022VelType`](@ref)
 - `ρₐ`: Air density [kg/m³]
 - `state`: A [`P3State`](@ref)
"""
@inline function mixed_particle_terminal_velocity(
    velocity_params::CMP.Chen2022VelType, ρₐ, state::P3State,
)
    v_ice = ice_particle_terminal_velocity(velocity_params, ρₐ, state)
    v_drop = CO.particle_terminal_velocity(velocity_params.rain, ρₐ)
    return P3MixedParticleVelocityFunctor(v_ice, v_drop, state.F_liq)
end

struct P3NumberWeightedIntegrand{N, V} <: Function
    n::N
    v_term::V
end
@inline (f::P3NumberWeightedIntegrand)(D) = f.n(D) * f.v_term(D)

"""
    ice_terminal_velocity_number_weighted(
        velocity_params::CMP.Chen2022VelType, ρₐ, state::P3State, shape;
        p = 1e-6, quad,
    )

Return the terminal velocity of the number-weighted mean ice particle size.

# Arguments
- `velocity_params`: A [`CMP.Chen2022VelType`](@ref) with terminal velocity parameters
- `ρₐ`: Air density [kg/m³]
- `state`: A [`P3State`](@ref)
- `shape`: The diagnosed [`P3Shape`](@ref), or a real `logλ` from which the
  shape is diagnosed via [`get_distribution_shape`](@ref)

# Keyword arguments
 - `p`: Tolerance parameter for the integral bounds. Default is 1e-6.
 - `quad`: quadrature rule (a `Quadrature.QuadratureRule`)

See also [`ice_terminal_velocity_mass_weighted`](@ref)
"""
function ice_terminal_velocity_number_weighted(
    velocity_params::CMP.Chen2022VelType, ρₐ, state::P3State, logλ::Real;
    kw...,
)
    shape = get_distribution_shape(state, logλ)
    return ice_terminal_velocity_number_weighted(velocity_params, ρₐ, state, shape; kw...)
end
function ice_terminal_velocity_number_weighted(
    velocity_params::CMP.Chen2022VelType, ρₐ, state::P3State, shape::P3Shape;
    p = 1e-6, quad,
)
    (; ρn_ice) = state
    v_term = ice_particle_terminal_velocity(velocity_params, ρₐ, state)
    n = DT.size_distribution(state, shape)

    # ∫n(D) v(D) dD, normalized by the number concentration. The floored
    # denominator keeps an empty state (ρn_ice → 0, integ → 0) finite and the
    # mean velocity C0-continuous across ice onset.
    number_weighted_integrand = P3NumberWeightedIntegrand(n, v_term)
    bnds = velocity_integral_bounds(state, shape, v_term; p)
    integ = integrate(number_weighted_integrand, bnds, quad)
    return integ / max(ρn_ice, eps(one(ρn_ice)))
end

struct P3MassWeightedIntegrand{N, V, S} <: Function
    n::N
    v_term::V
    state::S
end
@inline (f::P3MassWeightedIntegrand)(D) = f.n(D) * f.v_term(D) * ice_mass(f.state, D)

"""
    ice_terminal_velocity_mass_weighted(
        velocity_params::CMP.Chen2022VelType, ρₐ, state::P3State, shape;
        p = 1e-6, quad,
    )

Return the terminal velocity of the mass-weighted mean ice particle size.

# Arguments
- `velocity_params`: A [`CMP.Chen2022VelType`](@ref) with terminal velocity parameters
- `ρₐ`: Air density [kg/m³]
- `state`: A [`P3State`](@ref)
- `shape`: The diagnosed [`P3Shape`](@ref), or a real `logλ` from which the
  shape is diagnosed via [`get_distribution_shape`](@ref)

# Keyword arguments
 - `p`: Tolerance parameter for the integral bounds. Default is 1e-6.
 - `quad`: quadrature rule (a `Quadrature.QuadratureRule`)

See also [`ice_terminal_velocity_number_weighted`](@ref)
"""
function ice_terminal_velocity_mass_weighted(
    velocity_params::CMP.Chen2022VelType, ρₐ, state::P3State, logλ::Real;
    kw...,
)
    shape = get_distribution_shape(state, logλ)
    return ice_terminal_velocity_mass_weighted(velocity_params, ρₐ, state, shape; kw...)
end
function ice_terminal_velocity_mass_weighted(
    velocity_params::CMP.Chen2022VelType, ρₐ, state::P3State, shape::P3Shape;
    p = 1e-6, quad,
)
    (; ρq_ice, ρn_ice) = state
    v_term = ice_particle_terminal_velocity(velocity_params, ρₐ, state)
    n = DT.size_distribution(state, shape)

    # ∫n(D) m(D) v(D) dD, normalized by the mass concentration floored by the
    # distribution's own represented mass ρn_ice·exp(logLdivN). Where the shape
    # solve is unclamped the floor sits at (or below) ρq_ice, so the mean is
    # unchanged; when ρq_ice → 0 with ρn_ice > 0 (logλ clamped) it keeps the
    # bare mean a bounded fall speed (≤ the largest particle speed) while the
    # sedimentation flux w·ρq stays conservative and vanishes with ρq_ice.
    bnds = velocity_integral_bounds(state, shape, v_term; p)
    integ = integrate(P3MassWeightedIntegrand(n, v_term, state), bnds, quad)
    represented_mass = ρn_ice * exp(logLdivN(state, shape))
    return integ / max(ρq_ice, represented_mass, floatmin(eltype(state)))
end

"""
    ice_terminal_velocity_number_weighted_from_prognostic(
        velocity_params, ρₐ, params, ρq_ice, ρn_ice, ρq_rim, ρb_rim, logλ; kw...
    )

Pointwise wrapper that takes the *raw prognostic* P3 ice state
(`ρq_ice`, `ρn_ice`, `ρq_rim`, `ρb_rim`) and returns the number-weighted
mean ice terminal velocity. Builds the per-cell `P3State` via
[`state_from_prognostic`](@ref), so `F_rim` is regularised to
`[0, 1 - eps(FT)]` and `ρ_rim` is clamped to `[0, 0.8 ρ_l]`.

Designed for `@.`-broadcast use from a host (CA, KiD, etc.) where the
state must be reconstructed from prognostic variables every cell.
"""
@inline function ice_terminal_velocity_number_weighted_from_prognostic(
    velocity_params, ρₐ, params, ρq_ice, ρn_ice, ρq_rim, ρb_rim, logλ; kw...,
)
    state = state_from_prognostic(params, ρq_ice, ρn_ice, ρq_rim, ρb_rim)
    return ice_terminal_velocity_number_weighted(velocity_params, ρₐ, state, logλ; kw...)
end

"""
    ice_terminal_velocity_mass_weighted_from_prognostic(
        velocity_params, ρₐ, params, ρq_ice, ρn_ice, ρq_rim, ρb_rim, logλ; kw...
    )

Mass-weighted counterpart to
[`ice_terminal_velocity_number_weighted_from_prognostic`](@ref). Builds
the per-cell `P3State` via the regularised
[`state_from_prognostic`](@ref).
"""
@inline function ice_terminal_velocity_mass_weighted_from_prognostic(
    velocity_params, ρₐ, params, ρq_ice, ρn_ice, ρq_rim, ρb_rim, logλ; kw...,
)
    state = state_from_prognostic(params, ρq_ice, ρn_ice, ρq_rim, ρb_rim)
    return ice_terminal_velocity_mass_weighted(velocity_params, ρₐ, state, logλ; kw...)
end
