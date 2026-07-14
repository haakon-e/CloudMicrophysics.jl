# Per-process reflectivity (sixth-moment, `Z`) tendencies for three-moment ice.
# The default closure holds μ fixed across growth/decay (Eq. 10 of Milbrandt et
# al. 2021) and adds an initiation term per new-ice process (Eq. 9).

"""
    ZContribution{FT}

Accumulator for a process's reflectivity contributions, grouped by the term the
post-pass evaluates.

# Fields
$(FIELDS)
"""
struct ZContribution{FT}
    "Net Group-2 mass rate feeding the constant-μ growth term [kg m⁻³ s⁻¹]"
    dL_growth::FT
    "Net Group-2 number rate feeding the constant-μ growth term [m⁻³ s⁻¹]"
    dN_growth::FT
    "Summed Group-1 initiation reflectivity rate [m³ s⁻¹]"
    dZ_init::FT
    "Moment-6-weighted rate, used only by the full-moment closure; zero otherwise [m³ s⁻¹]"
    dZ_full::FT
end
ZContribution{FT}() where {FT} = ZContribution{FT}(zero(FT), zero(FT), zero(FT), zero(FT))
Base.zero(::Type{ZContribution{FT}}) where {FT} = ZContribution{FT}()
Base.zero(::ZContribution{FT}) where {FT} = ZContribution{FT}()
@inline Base.:+(a::ZContribution, b::ZContribution) = ZContribution(
    a.dL_growth + b.dL_growth, a.dN_growth + b.dN_growth,
    a.dZ_init + b.dZ_init, a.dZ_full + b.dZ_full,
)
Base.broadcastable(z::ZContribution) = tuple(z)

"""
    ReflectivityCoefficients{FT}

Frozen coefficients of the constant-μ reflectivity growth tendency, computed
once per substep by [`reflectivity_growth_coefficients`](@ref) and held fixed
across the substep. Constructed with keywords only.

# Fields
$(FIELDS)
"""
struct ReflectivityCoefficients{FT}
    "Closure ratio `G(μ)` [-]"
    G::FT
    "Third-moment-to-number ratio `M₃/N` [m³]"
    M3divN::FT
    "Third-moment-to-mass ratio `M₃/L` [m³/kg]"
    M3divL::FT
    function ReflectivityCoefficients(; G, M3divN, M3divL)
        G, M3divN, M3divL = promote(G, M3divN, M3divL)
        return new{typeof(G)}(G, M3divN, M3divL)
    end
end
Base.broadcastable(c::ReflectivityCoefficients) = tuple(c)

"""
    reflectivity_growth_coefficients(state::P3State, shape)

Compute the frozen [`ReflectivityCoefficients`](@ref) of the constant-μ growth
tendency: `G(μ)`, the analytic ratio `M₃/N = Γ(μ+4)/Γ(μ+1)·λ⁻³`, and
`M₃/L = (M₃/N)/x̄` with the mean particle mass `x̄ = L/N` clamped into the
`[mean_mass_min, mean_mass_max]` band, so the coefficients stay bounded as the
category empties. Evaluate once per substep, outside any differentiated region;
[`reflectivity_growth_tendency`](@ref) then differentiates only through the
rates.
"""
function reflectivity_growth_coefficients(state::P3State{FT}, shape::P3Shape) where {FT}
    moments = state.params.moments
    μ = shape.μ
    G = G_of_μ(μ)
    M3divN = exp(loggamma_moment(μ, shape.logλ; k = 3) - loggamma_moment(μ, shape.logλ; k = 0))
    x̄ = _bounded_mean_mass(moments, state.ρq_ice, state.ρn_ice)
    return ReflectivityCoefficients(; G, M3divN, M3divL = M3divN / x̄)
end

"""
    reflectivity_growth_tendency(coeffs::ReflectivityCoefficients, dL_growth, dN_growth)

Compute the constant-μ growth/decay reflectivity tendency (Eq. 10 of
[Milbrandt2021](@cite)) as a linear combination of the net Group-2 mass and
number rates with the frozen coefficients:

```math
\\frac{dZ}{dt} = G(μ)\\left[2\\,\\frac{M₃}{N}\\frac{M₃}{L}\\,dL_g − \\left(\\frac{M₃}{N}\\right)² dN_g\\right].
```
"""
@inline function reflectivity_growth_tendency(
    (; G, M3divN, M3divL)::ReflectivityCoefficients, dL_growth, dN_growth,
)
    return G * (2 * M3divN * M3divL * dL_growth - M3divN^2 * dN_growth)
end

"""
    reflectivity_initiation_monodisperse(μ_init, D_nuc, dN_init)

Compute the reflectivity source for monodisperse initiation of new ice at a
single size `D_nuc` (deposition/heterogeneous nucleation, ice multiplication),
Eq. 9 of [Milbrandt2021](@cite) in its division-free form:

```math
dZ_{init} = G(μ_{init})\\, D_{nuc}⁶\\, dN_{init}.
```

`D_nuc⁶` is evaluated directly; in `Float32` it is representable for
`D_nuc ≳ 1 μm`.
"""
@inline reflectivity_initiation_monodisperse(μ_init, D_nuc, dN_init) =
    G_of_μ(μ_init) * D_nuc^6 * dN_init

"""
    reflectivity_initiation_freezing(moments::CMP.ThreeMoment, ρ_new, μ_source, dq_init, dN_init)

Compute the reflectivity source for freezing of a liquid drop distribution
(cloud or rain freezing), Eq. 9 of [Milbrandt2021](@cite) with the shape
parameter `μ_source` of the source drop PSD conserved:

```math
dZ_{init} = G(μ_{source})\\left(\\frac{6}{π ρ_{new}}\\right)² \\bar m\\, dq_{init},
```

with the frozen mean drop mass `m̄ = dq_init/dN_init` clamped into the
`[mean_mass_min, mean_mass_max]` band. The band is the ice band: its lower
bound can exceed the mean mass of small cloud drops, overestimating a
contribution that vanishes with `dq_init`, and it bounds the derivative of the
ratio at vanishing rates. `ρ_new` is the new-ice density.
"""
@inline function reflectivity_initiation_freezing(moments::CMP.ThreeMoment, ρ_new, μ_source, dq_init, dN_init)
    FT = typeof(dq_init)
    m̄ = _bounded_mean_mass(moments, dq_init, dN_init)
    return G_of_μ(μ_source) * (6 / (FT(π) * ρ_new))^2 * m̄ * dq_init
end

"""
    reflectivity_tendency(coeffs::ReflectivityCoefficients, zc::ZContribution)

Assemble the total constant-μ reflectivity tendency `dρz/dt` from an accumulated
[`ZContribution`](@ref): the Group-2 growth term evaluated with the frozen
`coeffs` plus the summed Group-1 initiation term. The `dZ_full` term is reserved
for the full-moment closure and ignored here.
"""
@inline reflectivity_tendency(coeffs::ReflectivityCoefficients, zc::ZContribution) =
    reflectivity_growth_tendency(coeffs, zc.dL_growth, zc.dN_growth) + zc.dZ_init

# Mean particle mass `q/n` clamped into the closure's mean-mass band. The
# denominator floor is `floatmin` (finiteness only); the physical bounding is on
# the ratio.
@inline function _bounded_mean_mass(moments::CMP.ThreeMoment, q, n)
    FT = typeof(q)
    return clamp(q / max(n, floatmin(FT)), FT(moments.mean_mass_min), FT(moments.mean_mass_max))
end

"""
    advected_reflectivity(ρn_ice, ρz_ice)

Transform the volumetric number and sixth moments into the advected variable
`√(ρn_ice · ρz_ice)` (Eq. 12 of [Milbrandt2021](@cite)). Advecting this
quantity, rather than `ρz_ice` directly, preserves the moment ratios (and hence
μ) under linear transport.
"""
@inline advected_reflectivity(ρn_ice, ρz_ice) = sqrt(max(ρn_ice * ρz_ice, zero(ρn_ice * ρz_ice)))

"""
    reflectivity_from_advected(moments::ThreeMoment, ρz_adv, ρn_ice)

Recover the volumetric sixth moment `ρz_ice` from the advected variable `ρz_adv`,
clamping `ρz_adv` into the admissible window `[√zn_lo, √zn_hi] · max(ρn_ice, 0)`
before inverting: `ρz_ice = ρz_adv² / max(ρn_ice, n_presence)`. The window keeps
the recovered moment consistent with the number content and finite when transport
drives `ρn_ice` toward zero while `ρz_adv` lags. The round trip with
[`advected_reflectivity`](@ref) is the identity for in-window `ρn_ice ≥ n_presence`.
"""
@inline function reflectivity_from_advected(moments::CMP.ThreeMoment, ρz_adv, ρn_ice)
    ρn⁺ = max(ρn_ice, zero(ρn_ice))
    ρz_adv_windowed = clamp(ρz_adv, sqrt(moments.zn_lo) * ρn⁺, sqrt(moments.zn_hi) * ρn⁺)
    return ρz_adv_windowed^2 / max(ρn_ice, moments.n_presence)
end
