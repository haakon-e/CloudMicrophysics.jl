import CloudMicrophysics.DistributionTools: size_distribution

"""
    P3Shape{FT}

Diagnosed ice size-distribution shape for one category, held fixed across a
substep. Constructed with keywords only.

# Fields
$(FIELDS)
"""
struct P3Shape{FT}
    "Whole-particle log-slope `log(λ)` [log(1/m)]"
    logλ::FT
    "Shape parameter μ [-]"
    μ::FT
    "Ice-core log-slope `log(λ)` [log(1/m)]; equals `logλ` when liquid is off"
    logλ_core::FT
    function P3Shape(; logλ, μ, logλ_core = logλ)
        logλ, μ, logλ_core = promote(logλ, μ, logλ_core)
        return new{typeof(logλ)}(logλ, μ, logλ_core)
    end
end

# Callable returned by `logN′ice`: evaluates `log(N′(D))` for a fixed state and slope.
# We store `λ = exp(logλ)` (computed once when the functor is built) rather than `logλ`
# so the slope term is a single multiply `λ * D` in the quadrature hot loop, instead of
# `exp(logλ + logD)` (a transcendental per evaluation). `logD = log(D)` is still needed
# for the `μ * logD` term.
struct P3LogNumberFunctor{FT} <: Function
    log_N₀::FT
    μ::FT
    λ::FT
end
@inline function (f::P3LogNumberFunctor)(D)
    logD = log(D)
    return f.log_N₀ + f.μ * logD - f.λ * D
end

"""
    logN′ice(state, shape)

Return a callable that computes `log(N′(D))`, the log of the ice particle number
concentration at diameter `D`, for the [`P3State`](@ref) `state` and the
diagnosed [`P3Shape`](@ref) `shape`.
"""
function logN′ice(state::P3State, shape::P3Shape)
    μ = shape.μ
    log_N₀ = get_logN₀(state.ρn_ice, μ, shape.logλ)
    # Promote to a common type: differentiating w.r.t. the ice number makes
    # `log_N₀` a `Dual` while `μ` (frozen on the shape) stays a plain float, and
    # `P3LogNumberFunctor` stores both in a single field type.
    λ = exp(shape.logλ)
    return P3LogNumberFunctor(promote(log_N₀, μ, λ)...)
end

# Callable returned by `size_distribution`: `n(D) = exp(logN′(D))`.
struct P3SizeDistributionFunctor{F} <: Function
    logN′::F
end
@inline (f::P3SizeDistributionFunctor)(D) = exp(f.logN′(D))

"""
    size_distribution(state::P3State, shape)

Return `n(D)`, a function that computes the size distribution for ice particles at diameter `D`

# Arguments
- `state`: The [`P3State`](@ref)
- `shape`: The diagnosed [`P3Shape`](@ref)
"""
DT.size_distribution(state::P3State, shape::P3Shape) = P3SizeDistributionFunctor(logN′ice(state, shape))

### ------------------------------------------------ ###
### ----- Obtaining P3 distribution parameters ----- ###
### ------------------------------------------------ ###

"""
    loggamma_inc_moment(D₁, D₂, μ, logλ, [k = 0], [scale = 1])

Compute `log(Iᵏ)` where `Iᵏ` is the following integral:

    ``I^k = ∫_{D₁}^{D₂} G(D) D^k dD``

 ``G(D) ≡ D^μ e^{-λD}`` is the (unnormalized) gamma kernel, and `k` is an arbitrary exponent.

 If `scale` is provided, `log(scale ⋅ Iᵏ)` is returned.

 With appropriate scaling, we can compute useful quantities like:
 - the `k`-th moment of the ice PSD,
    ``M^k = N₀ I^k``
 - combined power law and moment weighted integrals,
    ``∫_{D₁}^{D₂} (aD^b) D^n K(D) dD ≡ a I^(b + n)``

# Arguments
 - `D₁`: The minimum diameter [`m`]
 - `D₂`: The maximum diameter [`m`]
 - `μ`: The PSD shape parameter [`-`]
 - `logλ`: The log of the PSD slope parameter [`log(1/m)`]
 - `k`: An arbitrary exponent [`-`], default is `0`
 - `scale`: The scale factor [`-`], default is `1`

# Extended help
 ## Implementation details
 We can write `∫_D₁^D₂ G(D) D^k dD`, where `G(D) = D^μ e^{-λD}` as:
    `∫_D₁^∞ G(D) D^k dD - ∫_D₂^∞ G(D) D^k dD`
 with the transformation `x = λD`, and `z = μ+k+1`, each term can be written as:
    `∫_{Dᵢ}^∞ G(D) D^k dD = ∫_{λDᵢ}^∞ x^z e^{-x} dx / λ^z = Γ(z, λDᵢ) / λ^z`
 where `Γ(z, λDᵢ) = q ⋅ Γ(z)` and `q` is the incomplete gamma function ratio given by
    `(_, q) = UT.gamma_inc(z, x)`.
 This means that the integral `∫_{Dᵢ}^∞ G(D) D^k dD` is computed as:
    `Γ(z) ⋅ q / λ^z`
 The full integral from `D₁` to `D₂` is then:
    `Γ(z) ⋅ (q_D₁ - q_D₂) / λ^z`
 In log-space, this is:
    `- z log(λ) + logΓ(z) + log(q_D₁ - q_D₂)`

See also [`gamma_inc_moment`](@ref)
"""
function loggamma_inc_moment(D₁, D₂, μ, logλ, k = 0, scale = 1)
    FT = UT.promote_typeof(D₁, D₂, μ, logλ)
    D₁ < D₂ || return log(FT(0))  # return log(0) if D₁ ≥ D₂
    z = k + μ + 1
    # `λ⋅D ≡ xexpy(D, logλ) ≡ D * exp(logλ)` (numerically stable)
    x1 = LogExpFunctions.xexpy(D₁, logλ)
    x2 = LogExpFunctions.xexpy(D₂, logλ)
    (p1, q1) = UT.gamma_inc(z, x1)
    (p2, q2) = UT.gamma_inc(z, x2)
    Δq = x2 < z + 1 ? p2 - p1 : q1 - q2
    Δq = max(Δq, eps(FT))
    return -z * logλ + SF.loggamma(z) + log(Δq) + log(FT(scale))
end

"""
    gamma_inc_moment(D₁, D₂, p, α)

`∫_{D₁}^{D₂} D^p e^{-α D} dD = α^{-(p+1)} Γ(p+1) [Q(p+1,αD₁) - Q(p+1,αD₂)]`
with `Q` the regularized upper incomplete gamma.

Returns `0` if `D₂ ≤ D₁`, and `NaN` if `α ≤ 0`

See also [`loggamma_inc_moment`](@ref)
"""
@inline function gamma_inc_moment(D₁, D₂, p, α)
    FT = float(promote_type(typeof(D₁), typeof(D₂), typeof(α)))
    D₂ > D₁ || return zero(FT)
    α > 0 || return FT(NaN)
    z = p + 1
    x1 = α * D₁
    x2 = α * D₂
    (p1, q1) = UT.gamma_inc(z, x1)
    (p2, q2) = UT.gamma_inc(z, x2)
    Δq = x2 < z + 1 ? p2 - p1 : q1 - q2
    Δq = max(Δq, zero(FT))
    return SF.gamma(z) * Δq / α^z
end

# Analytic ForwardDiff partials of `M(D₁, D₂, p, α) = ∫_{D₁}^{D₂} Dᵖ e^{-αD} dD`:
# `∂M/∂D₁ = -D₁ᵖ e^{-αD₁}`, `∂M/∂D₂ = D₂ᵖ e^{-αD₂}`, `∂M/∂α = -M(D₁, D₂, p+1, α)`.
# The value is taken from the plain-`Real` method on the argument values.
@inline function gamma_inc_moment(D₁, D₂, p, α::FD.Dual{T}) where {T}
    v₁ = FD.value(T, D₁)
    v₂ = FD.value(T, D₂)
    vα = FD.value(T, α)
    M = gamma_inc_moment(v₁, v₂, p, vα)
    ∂D₁ = -v₁^p * exp(-vα * v₁)
    ∂D₂ = v₂^p * exp(-vα * v₂)
    ∂α = -gamma_inc_moment(v₁, v₂, p + 1, vα)
    Z = zero(FD.partials(α))
    part = ∂D₁ * _moment_partials(T, Z, D₁) + ∂D₂ * _moment_partials(T, Z, D₂) + ∂α * FD.partials(α)
    return FD.Dual{T}(M, part)
end

# Partials of an argument with respect to tag `T`; a non-`Dual` argument contributes `Z`.
@inline _moment_partials(::Type{T}, Z, x::FD.Dual{T}) where {T} = FD.partials(x)
@inline _moment_partials(::Type{T}, Z, x) where {T} = Z

"""
    loggamma_moment(μ, logλ; [k = 0], [scale = 1])

Compute `log(scale ⋅ ∫_0^∞ G(D) D^k dD)`, 
 where `G(D) ≡ D^μ e^{-λD}` is the (unnormalized) gamma kernel, 
 `k` is an arbitrary exponent, and `scale` is a scale factor.

# Arguments
 - `μ`: The PSD shape parameter [`-`]
 - `logλ`: The log of the PSD slope parameter [`log(1/m)`]

# Keyword arguments
- `k`: An arbitrary exponent [`-`], default is `0`
- `scale`: The scale factor [`-`], default is `1`.

The implementation follows the same logic as [`loggamma_inc_moment`](@ref),
    but with `D₁ = 0` and `D₂ = ∞`, which implies `q_D₁ = 1` and `q_D₂ = 0`.
"""
function loggamma_moment(μ, logλ; k = 0, scale = 1)
    FT = eltype(μ)
    z = k + μ + 1
    return -z * logλ + SF.loggamma(z) + log(FT(scale))
end

"""
    log_upper_incomplete_gamma(z, x)

Compute `log Γ(z, x) = log ∫_x^∞ tᶻ⁻¹ e⁻ᵗ dt`, the log of the unregularised
upper incomplete gamma function, as `loggamma(z) + log Q(z, x)` with `Q` the
regularised upper incomplete gamma ratio ([`UT.gamma_inc`](@ref)). Taking the
log directly (rather than the `eps`-floored `log(max(Q, eps))` of
[`loggamma_inc_moment`](@ref)) keeps the deep tail accurate: in Float32 `Q`
remains representable well past `x = 90` and only underflows to `0` (giving
`-Inf`) at extreme `x`, where the moment it feeds is physically negligible.
Used by the closed-form shedding moment.
"""
@inline function log_upper_incomplete_gamma(z, x)
    (_, Q) = UT.gamma_inc(z, x)
    return SF.loggamma(z) + log(max(Q, zero(Q)))
end

"""
    get_μ(slope::CMP.SlopeLaw, logλ)
    get_μ(state::P3State, logλ)
    
Compute the slope parameter μ

# Arguments
- `slope`: [`CMP.SlopeLaw`](@ref) object, or
- `state`: [`P3State`](@ref) object, or
- `params`: [`CMP.ParametersP3`](@ref) object
- `logλ`: The log of the slope parameter [log(1/m)]
"""
get_μ((; a, b, c, μ_max)::CMP.SlopePowerLaw, logλ) = clamp(a * exp(logλ)^b - c, 0, μ_max)
get_μ((; μ)::CMP.SlopeConstant, logλ...) = μ
get_μ((; params)::P3State, logλ) = get_μ(params.moments.slope, logλ)

"""
    logmass_gamma_moment(state, logλ; [n=0])

Compute `log(∫_0^∞ Dⁿ m(D) N′(D) dD)` given the `state` and `logλ`.
    This is the log of the `n`-th moment of the mass-weighted PSD.

# Arguments
- `state`: [`P3State`](@ref) object
- `μ`: The shape parameter [`-`]
- `logλ`: The log of the slope parameter [log(1/m)]

# Keyword arguments
- `n`: The order of the moment, default is `0`

# Note:
- For `n = 0`, this evaluates to `log(L/N₀)`
- For `n = 1`, this evaluates to the (unnormalized) mass-weighted mean particle size, see [`D_m`](@ref)
"""
function logmass_gamma_moment(state::P3State, μ, logλ; n = 0)
    bnds = segment_boundaries(state)
    moments = UU.unrolled_map(subintervals(bnds)) do (D_lo, D_hi)
        (a, b) = ice_mass_coeffs(state, (D_lo + D_hi) / 2)
        loggamma_inc_moment(D_lo, D_hi, μ, logλ, b + n, a)
    end
    return UT.unrolled_logsumexp(moments)
end

"""
    log_mixed_mass_moment(state, μ, logλ, F_liq; [n = 0])

Compute `log(∫_0^∞ Dⁿ mₜ(D) G(D) dD)`, the log of the `n`-th moment of the whole
(mixed-phase) mass-weighted kernel, where the mixed mass is the blend
`mₜ = (1 - F_liq) m_core(D) + F_liq (π/6) ρ_l D³` ([`mixed_mass`](@ref)).

The blend adds the single pure-`D³` drop moment `liq0` to the piecewise ice-core
moment `core` ([`logmass_gamma_moment`](@ref)) as a log-space convex combination
with `F_liq` outside every logarithm:

```math
\\log M = \\mathrm{core} + \\log\\!\\big(1 + F_{liq}\\,(e^{liq0 - core} - 1)\\big).
```

The value equals `core` exactly at `F_liq = 0` and its derivative with respect to
`F_liq` is finite there (`e^{liq0 - core} - 1`), avoiding the singular
`log(F_liq)` intermediate.
"""
function log_mixed_mass_moment(state::P3State, μ, logλ, F_liq; n = 0)
    core = logmass_gamma_moment(state, μ, logλ; n)
    liq0 = loggamma_moment(μ, logλ; k = 3 + n, scale = π * state.params.ρ_l / 6)
    return core + log1p(F_liq * expm1(liq0 - core))
end

"""
    logLdivN(state, logλ)
    logLdivN(state, shape::P3Shape)

Compute `log(L/N)` given the `state` and either a bare `logλ` (with μ evaluated
from the slope law, used inside the shape solve) or a diagnosed [`P3Shape`](@ref)
(reading `shape.μ`, used by external callers).

# Arguments
- `state`: [`P3State`](@ref) object
- `logλ`: The log of the slope parameter [log(1/m)]
"""
function logLdivN(state::P3State, logλ)
    μ = get_μ(state, logλ)
    logLdivN₀ = logmass_gamma_moment(state, μ, logλ; n = 0)
    logNdivN₀ = loggamma_moment(μ, logλ; k = 0)
    return logLdivN₀ - logNdivN₀
end
function logLdivN(state::P3State, shape::P3Shape)
    return logLdivN(state, shape.μ, shape.logλ)
end
# `log(L/N)` for the ice core at a fixed shape parameter μ (the shared
# whole-particle μ); see `get_distribution_logλ_core`.
function logLdivN(state::P3State, μ, logλ)
    logLdivN₀ = logmass_gamma_moment(state, μ, logλ; n = 0)
    logNdivN₀ = loggamma_moment(μ, logλ; k = 0)
    return logLdivN₀ - logNdivN₀
end

"""
    logLdivN_whole(state, logλ, F_liq)

Compute `log(q_tot/N)` for the whole (mixed-phase) particle PSD at slope `logλ`,
using the blended mass moment [`log_mixed_mass_moment`](@ref). Reduces to
[`logLdivN`](@ref) at `F_liq = 0`. Driven to the target `log(ρq_tot/ρn_ice)` by
[`get_distribution_logλ_whole`](@ref).
"""
function logLdivN_whole(state::P3State, logλ, F_liq)
    μ = get_μ(state, logλ)
    logLdivN₀ = log_mixed_mass_moment(state, μ, logλ, F_liq; n = 0)
    logNdivN₀ = loggamma_moment(μ, logλ; k = 0)
    return logLdivN₀ - logNdivN₀
end
# Whole-particle `log(q_tot/N)` at an explicit shape parameter μ (the shared
# whole-particle μ solved from `(L_whole, N, Z)` under three-moment ice); see the
# joint `_distribution_shape`.
function logLdivN_whole(state::P3State, μ, logλ, F_liq)
    logLdivN₀ = log_mixed_mass_moment(state, μ, logλ, F_liq; n = 0)
    logNdivN₀ = loggamma_moment(μ, logλ; k = 0)
    return logLdivN₀ - logNdivN₀
end

"""
    get_logN₀(N_ice, μ, logλ)

Compute `log(N₀)` given the `state`, `N`, and `logλ`,

        N  = N₀ ∫ G(D) dD
    log N₀ = log N - log(∫G(D) dD) 
           = log(N) - log( ∫D^μ e^{-λD} dD )
           = log(N) - M⁰

# Arguments
- `N_ice`: The number concentration [1/m³]
- `μ`: The shape parameter [`-`]
- `logλ`: The log of the slope parameter [log(1/m)]
"""
function get_logN₀(N_ice, μ, logλ)
    logNdivN₀ = loggamma_moment(μ, logλ; k = 0)
    logN₀ = log(N_ice) - logNdivN₀
    return logN₀
end

"""
    FixedIterations{FT}()

A `RootSolvers.AbstractTolerance` whose convergence predicate is always `false`,
so the bracketing solver never exits early and always runs the full iteration
budget. This makes the iteration count independent of the input, eliminating
warp divergence from data-dependent early-exit on the GPU (at the cost of the
warm-start speedup — a tighter initial bracket improves accuracy but not the
iteration count). The iteration budget itself is calibrated empirically; see
[`get_distribution_logλ`](@ref).
"""
struct FixedIterations{FT} <: RS.AbstractTolerance{FT} end
@inline (::FixedIterations)(x1, x2, y) = false

# Numerical size bounds on log(λ), shared by the two- and three-moment shape
# solves.
const LOGλ_MIN = CMP.P3_LOGλ_MIN
const LOGλ_MAX = CMP.P3_LOGλ_MAX

# Fixed root-solve budget for the shape solves. Calibrated empirically; see
# `get_distribution_logλ`.
@inline _shape_solver_maxiters(::Type{FT}) where {FT} = FT === Float32 ? 8 : 10

"""
    get_distribution_logλ(state, [logλ_guess, logλ_min, logλ_max])

Solve for the distribution parameters given the state, and the mass (`L`) and number (`N`) concentrations.

The assumed distribution is of the form

```math
N′(D) = N₀ D^μ e^{-λD}
```
where `N′(D)` is the number concentration at diameter `D` and `μ` is the slope parameter.
    The slope parameter is parameterized, e.g. [`CMP.SlopePowerLaw`](@ref) or [`CMP.SlopeConstant`](@ref).

This algorithm solves for `logλ = log(λ)` and `log_N₀ = log(N₀)`
    given `L_ice` and `N_ice` by solving the equations:

```math
\\begin{align*}
\\log(L) &= \\log ∫_0^∞ m(D) N′(D)\\ \\mathrm{d}D, \\\\
\\log(N) &= \\log ∫_0^∞ N′(D)\\ \\mathrm{d}D, \\\\
\\end{align*}
```
where `m(D)` is the mass of a particle at diameter `D` (see [`ice_mass`](@ref)).
    The procedure is decribed in detail in [the P3 docs](@ref "Parameterizations for the slope parameter \$μ\$").

# Arguments
- `state`: The [`P3State`](@ref)
- `logλ_guess`: Optional initial guess
- `logλ_min`: The minimum value of the search bounds [log(1/m)], default is `2`
- `logλ_max`: The maximum value of the search bounds [log(1/m)], default is `17`
"""
function get_distribution_logλ(state, logλ_guess = nothing, logλ_min = LOGλ_MIN, logλ_max = LOGλ_MAX)
    FT = eltype(state)
    ϵₘ = UT.ϵ_numerics_2M_M(FT)
    ϵₙ = UT.ϵ_numerics_2M_N(FT)
    (; ρn_ice, ρq_ice) = state
    lo, hi = FT(logλ_min), FT(logλ_max)
    # Floor the mass and number inside the logs so the mean-size target is
    # finite and C0-continuous across onset: below the ϵ thresholds the target
    # freezes at its limiting value and the solve returns a finite, bounded
    # logλ. The size distribution still vanishes with ρn_ice, so no spurious
    # ice appears; the number is relaxed toward a mass-consistent range by
    # `number_tendency_from_mass_limits`.
    target_log_LdN = log(max(ρq_ice, ϵₘ)) - log(max(ρn_ice, ϵₙ))

    shape_problem(logλ) = logLdivN(state, logλ) - target_log_LdN
    # Fixed iteration count (no early-exit) keeps GPU warps convergent. The
    # branchless Brent's method converges rapidly, and the shape problem
    # `logLdivN(logλ)` is close to linear over the [2,17] bracket, so these
    # counts empirically reach excellent accuracy across sampled physical
    # states. This is an empirical, curvature-dependent result, not a guaranteed
    # tolerance: a strongly-curved shape function (e.g. a future `get_μ` law)
    # could leave the root under-resolved with no runtime signal (the solver's
    # `converged` flag is unused). Accuracy is guarded end-to-end by the
    # `N ≈ ∫N′ dD` integral checks in `test/p3_tests.jl`; revisit the budget if
    # those tighten or the slope law changes.
    return _solve_shape_logλ(shape_problem, FT, logλ_guess, lo, hi, _shape_solver_maxiters(FT))
end

# Shared bracketing Brent solve for the shape problem `shape_problem(logλ) = 0`.
# Returns the bracket endpoint nearest the root if the bracket is invalid
# (non-finite or same-sign), else the fixed-iteration Brent root clamped into
# `[lo, hi]`. `maxiters` is chosen by the caller so each solve sets its own
# empirically calibrated iteration budget.
function _solve_shape_logλ(shape_problem::F, ::Type{FT}, logλ_guess, lo, hi, maxiters) where {F, FT}
    f_lo, f_hi = shape_problem(lo), shape_problem(hi)
    if !isfinite(f_lo) || !isfinite(f_hi) || f_lo * f_hi > 0
        return abs(f_lo) ≤ abs(f_hi) ? lo : hi
    end
    (lo, f_lo, hi, f_hi) = _narrow_bracket(shape_problem, lo, f_lo, hi, f_hi, logλ_guess)
    sol = RS.find_zero(
        shape_problem,
        RS.BrentsMethod(lo, hi),
        RS.CompactSolution(),
        FixedIterations{FT}(),
        maxiters,
    )
    return clamp(sol.root, lo, hi)  # logλ, within the search bounds
end

"""
    get_distribution_logλ_whole(state, [logλ_guess, logλ_min, logλ_max])

Solve for the whole-particle log-slope `logλ` under predicted liquid fraction:
the slope of the mixed-phase PSD normalised to the total mass
`ρq_tot = ρq_ice + ρq_liq` at the ice number `ρn_ice`, using the blended mass
target [`logLdivN_whole`](@ref). Reduces to [`get_distribution_logλ`](@ref) at
`F_liq = 0`. A fixed-iteration Brent solve identical in structure to the ice-core
solve; the iteration budget is calibrated separately (see the whole-particle
shape-solve study in `test/p3_liquid_fraction_tests.jl`).
"""
function get_distribution_logλ_whole(state, logλ_guess = nothing, logλ_min = 2, logλ_max = 17)
    FT = eltype(state)
    ϵₘ = UT.ϵ_numerics_2M_M(FT)
    ϵₙ = UT.ϵ_numerics_2M_N(FT)
    F_liq = state.F_liq
    q_tot = total_mass_concentration(state)
    lo, hi = FT(logλ_min), FT(logλ_max)
    target_log_LdN = log(max(q_tot, ϵₘ)) - log(max(state.ρn_ice, ϵₙ))
    shape_problem(logλ) = logLdivN_whole(state, logλ, F_liq) - target_log_LdN
    return _solve_shape_logλ(shape_problem, FT, logλ_guess, lo, hi, _whole_shape_maxiters(FT))
end

"""
    get_distribution_logλ_core(state, μ, [logλ_guess, logλ_min, logλ_max])

Solve for the ice-core log-slope `logλ` under predicted liquid fraction: the
slope of the frozen-core PSD normalised to the frozen mass `ρq_ice` at the ice
number `ρn_ice`, at the shared shape parameter `μ` (set from the whole-particle
solve, see [`get_distribution_logλ_whole`](@ref)). The core mass relation is the
frozen-core [`ice_mass`](@ref) via [`logLdivN`](@ref)`(state, μ, logλ)`. A
fixed-iteration Brent solve identical in structure to the whole-particle solve.
"""
function get_distribution_logλ_core(state, μ, logλ_guess = nothing, logλ_min = 2, logλ_max = 17)
    FT = eltype(state)
    ϵₘ = UT.ϵ_numerics_2M_M(FT)
    ϵₙ = UT.ϵ_numerics_2M_N(FT)
    (; ρn_ice, ρq_ice) = state
    lo, hi = FT(logλ_min), FT(logλ_max)
    target_log_LdN = log(max(ρq_ice, ϵₘ)) - log(max(ρn_ice, ϵₙ))
    shape_problem(logλ) = logLdivN(state, μ, logλ) - target_log_LdN
    return _solve_shape_logλ(shape_problem, FT, logλ_guess, lo, hi, _core_shape_maxiters(FT))
end

# Iteration budgets for the two liquid-fraction shape solves, widened over the
# ice-core dry-solve budget (8/10) and calibrated against a high-iteration
# reference in `test/p3_liquid_fraction_tests.jl`.
@inline _whole_shape_maxiters(::Type{Float32}) = 12
@inline _whole_shape_maxiters(::Type{FT}) where {FT} = 14
@inline _core_shape_maxiters(::Type{Float32}) = 12
@inline _core_shape_maxiters(::Type{FT}) where {FT} = 14

"""
    get_distribution_shape(params::ParametersP3{FT, <:TwoMoment}, logλ)
    get_distribution_shape(state::P3State, logλ)
    get_distribution_shape(state::P3State)

Return the [`P3Shape`](@ref) for a given `logλ`, or diagnose the full shape for a
`state`. The one-argument state form dispatches on both the moment closure and
the liquid treatment. Under [`CMP.TwoMoment`](@ref) ice, μ comes from the slope
law with `logλ` solved via [`get_distribution_logλ`](@ref); under
[`CMP.ThreeMoment`](@ref) ice, μ and `logλ` are solved jointly from the number,
mass, and sixth-moment content. Under [`CMP.PredictedLiquidFraction`](@ref) the
stored `logλ` is the whole-particle slope ([`get_distribution_logλ_whole`](@ref))
and `logλ_core` solves the frozen core at the shared μ
([`get_distribution_logλ_core`](@ref)); `logλ_core` equals `logλ` when liquid is
off.
"""
get_distribution_shape(params::CMP.ParametersP3{FT, <:CMP.TwoMoment}, logλ) where {FT} =
    P3Shape(; logλ, μ = get_μ(params.moments.slope, logλ))
get_distribution_shape(state::P3State, logλ) =
    _distribution_shape(state.params.moments, state.params.liquid, state, logλ)
get_distribution_shape(state::P3State) =
    _distribution_shape(state.params.moments, state.params.liquid, state)

# --- Two-moment ice ---
# Without liquid fraction the single slope is both whole and core.
_distribution_shape(::CMP.TwoMoment, ::CMP.NoLiquidFraction, state::P3State, logλ) =
    get_distribution_shape(state.params, logλ)
_distribution_shape(::CMP.TwoMoment, ::CMP.NoLiquidFraction, state::P3State) =
    get_distribution_shape(state, get_distribution_logλ(state))

# The stored `logλ` is the whole-particle slope. μ is diagnosed there and shared
# with the ice-core PSD, whose slope solves the frozen core at that fixed μ.
function _distribution_shape(::CMP.TwoMoment, ::CMP.PredictedLiquidFraction, state::P3State, logλ)
    μ = get_μ(state, logλ)
    return P3Shape(; logλ, μ, logλ_core = get_distribution_logλ_core(state, μ))
end
function _distribution_shape(::CMP.TwoMoment, ::CMP.PredictedLiquidFraction, state::P3State)
    logλ = get_distribution_logλ_whole(state)
    μ = get_μ(state, logλ)
    return P3Shape(; logλ, μ, logλ_core = get_distribution_logλ_core(state, μ))
end

"""
    reflectivity_number_ratio(state::P3State)

Return the sixth-moment-to-number ratio `Z/N = M₆/M₀` [m⁶], clamped as one
quantity into the admissible window `[zn_lo, zn_hi]` cached on the
[`CMP.ThreeMoment`](@ref) closure. At the lower bound, `logλ(μ)` exceeds the
maximum size bound for every μ, so the shape solve saturates `logλ` at that
bound and μ follows the mass target.
"""
@inline function reflectivity_number_ratio(state::P3State{FT}) where {FT}
    (; zn_lo, zn_hi) = state.params.moments
    ratio = state.ρz_ice / max(state.ρn_ice, floatmin(FT))
    return clamp(ratio, zn_lo, zn_hi)
end

"""
    _distribution_shape(moments::CMP.ThreeMoment, liquid::CMP.NoLiquidFraction, state::P3State)

Diagnose the [`P3Shape`](@ref) from `(L, N, Z)` for three-moment ice. The slope
is pinned analytically by `Z/N`,

```math
\\log λ(μ) = \\tfrac{1}{6}\\left[\\log Γ(μ+7) − \\log Γ(μ+1) − \\log(Z/N)\\right],
```

and μ solves the piecewise mass residual `logLdivN(state, μ, logλ(μ)) − log(L/N)`
on `μ ∈ [0, μ_max]` with Brent + [`FixedIterations`](@ref). `log(λ)` is clamped
into `[LOGλ_MIN, LOGλ_MAX]` inside the residual, so the returned `logλ`
reproduces the mass target when a size bound binds. The μ-clamp is the
reflectivity limiter.
"""
function _distribution_shape(moments::CMP.ThreeMoment, ::CMP.NoLiquidFraction, state::P3State{FT}) where {FT}
    (; ρn_ice, ρq_ice) = state
    μ_max = FT(moments.μ_max)
    lo, hi = FT(LOGλ_MIN), FT(LOGλ_MAX)
    ϵₘ = UT.ϵ_numerics_2M_M(FT)
    ϵₙ = UT.ϵ_numerics_2M_N(FT)
    target_logLdN = log(max(ρq_ice, ϵₘ)) - log(max(ρn_ice, ϵₙ))
    logZdN = log(reflectivity_number_ratio(state))
    logλ_of_μ(μ) = (SF.loggamma(μ + 7) - SF.loggamma(μ + 1) - logZdN) / 6
    residual(μ) = logLdivN(state, μ, clamp(logλ_of_μ(μ), lo, hi)) - target_logLdN
    μ = _solve_shape_μ(residual, FT(0), μ_max)
    logλ = clamp(logλ_of_μ(μ), lo, hi)
    return P3Shape(; logλ, μ)
end

"""
    _distribution_shape(moments::CMP.ThreeMoment, liquid::CMP.PredictedLiquidFraction, state::P3State)

Diagnose the joint three-moment predicted-liquid-fraction [`P3Shape`](@ref). The
whole-particle number and sixth moments are pure gamma moments regardless of
liquid, so the slope is pinned analytically by `Z/N` exactly as in the dry
three-moment solve. μ solves the whole-particle mass residual
[`logLdivN_whole`](@ref)`(state, μ, logλ(μ), F_liq)` against `log(q_tot/N)` on
`μ ∈ [0, μ_max]` with the in-residual `logλ` clamp (C23 shared-μ closure). The
ice-core slope then solves the frozen core at the shared μ
([`get_distribution_logλ_core`](@ref)). Reduces bit-for-bit to the dry
three-moment solve for `(μ, logλ)` at `F_liq = 0`, where `logλ_core = logλ`; as
`F_liq → 0⁺` the fixed-iteration core solve is tolerance-continuous with that
limit, not bitwise.
"""
function _distribution_shape(moments::CMP.ThreeMoment, ::CMP.PredictedLiquidFraction, state::P3State{FT}) where {FT}
    (; ρn_ice, F_liq) = state
    q_tot = total_mass_concentration(state)
    μ_max = FT(moments.μ_max)
    lo, hi = FT(LOGλ_MIN), FT(LOGλ_MAX)
    ϵₘ = UT.ϵ_numerics_2M_M(FT)
    ϵₙ = UT.ϵ_numerics_2M_N(FT)
    target_logLdN = log(max(q_tot, ϵₘ)) - log(max(ρn_ice, ϵₙ))
    logZdN = log(reflectivity_number_ratio(state))
    logλ_of_μ(μ) = (SF.loggamma(μ + 7) - SF.loggamma(μ + 1) - logZdN) / 6
    residual(μ) = logLdivN_whole(state, μ, clamp(logλ_of_μ(μ), lo, hi), F_liq) - target_logLdN
    μ = _solve_shape_μ(residual, FT(0), μ_max)
    logλ = clamp(logλ_of_μ(μ), lo, hi)
    # The frozen core shares μ; its slope solves the core mass. At F_liq = 0 the
    # core and whole solves coincide, so the shape matches the dry solve exactly.
    logλ_core = iszero(F_liq) ? logλ : get_distribution_logλ_core(state, μ)
    return P3Shape(; logλ, μ, logλ_core)
end

# Brent + fixed-iteration root find with a deterministic bracket (no warm start).
@inline function _solve_shape_μ(residual::F, lo::FT, hi::FT) where {F, FT}
    f_lo, f_hi = residual(lo), residual(hi)
    if !isfinite(f_lo) || !isfinite(f_hi) || f_lo * f_hi > 0
        return abs(f_lo) ≤ abs(f_hi) ? lo : hi
    end
    sol = RS.find_zero(
        residual, RS.BrentsMethod(lo, hi), RS.CompactSolution(),
        FixedIterations{FT}(), _shape_solver_maxiters(FT),
    )
    return clamp(sol.root, lo, hi)
end

"""
    get_distribution_logλ_from_prognostic(params, ρq_ice, ρn_ice, ρq_rim, ρb_rim)

Compute `log(λ)` for P3, using prognostic ice variables directly.

The P3 variables `F_rim` and `ρ_rim` are computed in a regularised way. This
helper solves the dry (ice-core) slope only; under
[`CMP.PredictedLiquidFraction`](@ref) hosts construct the state with the liquid
mass and use [`get_distribution_shape`](@ref) for the whole-particle slope.
"""
function get_distribution_logλ_from_prognostic(
    params, ρq_ice, ρn_ice, ρq_rim, ρb_rim, args...,
)
    state = state_from_prognostic(params, ρq_ice, ρn_ice, ρq_rim, ρb_rim)
    return get_distribution_logλ(state, args...)
end

@inline _narrow_bracket(_sp, lo, f_lo, hi, f_hi, ::Nothing) = (lo, f_lo, hi, f_hi)
@inline function _narrow_bracket(shape_problem, lo, f_lo, hi, f_hi, p::Real)
    p_ = oftype(lo, p)
    valid = isfinite(p_) & (lo < p_ < hi)
    p_clean = ifelse(valid, p_, lo)
    f_p = shape_problem(p_clean)
    valid &= isfinite(f_p)

    left = valid & (f_lo * f_p < 0)
    right = valid & !left

    new_hi = ifelse(left, p_clean, hi)
    new_f_hi = ifelse(left, f_p, f_hi)
    new_lo = ifelse(right, p_clean, lo)
    new_f_lo = ifelse(right, f_p, f_lo)

    return (new_lo, new_f_lo, new_hi, new_f_hi)
end

"""
    get_distribution_logλ_all_solutions(state)

Find all solutions for `logλ` given the `state` ([`P3State`](@ref)), `L`, and `N`.

!!! note "Usage"
    This function is experimental, and usually only relevant for the
    [`SlopePowerLaw`](@ref) parameterization, which can have multiple solutions
    for `logλ` for a given `log_L` and `log_N`.
"""
function get_distribution_logλ_all_solutions(state::P3State)
    # Find bounds by evaluating function incrementally, then apply root finding with bounds above and below zero-point
    target_log_LdN = log(state.ρq_ice) - log(state.ρn_ice)

    shape_problem(logλ) = logLdivN(state, logλ) - target_log_LdN

    Δλ = 0.01
    λs = 10.0 .^ (2.0:Δλ:6.0)
    logλ_bnds = Tuple[]
    # Loop over λs and find where shape_problem changes sign
    for i in 1:(length(λs) - 1)
        if shape_problem(log(λs[i])) * shape_problem(log(λs[i + 1])) < 0
            push!(logλ_bnds, (log(λs[i]), log(λs[i + 1])))
        end
    end

    # Apply root finding with bounds above and below zero-point
    logλs = [get_distribution_logλ(state, nothing, logλ_min, logλ_max) for (logλ_min, logλ_max) in logλ_bnds]
    return logλs
end
