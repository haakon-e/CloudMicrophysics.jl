"""
    BulkMicrophysicsTendencies

Fused bulk microphysics source terms for atmospheric models.

Provides a dispatch-based API to compute all microphysics tendencies
in a single function call, enabling:
- Simplified integration in atmospheric models
- Point-evaluation suitable for quadrature over subgrid-scale fluctuations
- GPU-friendly pure function design

# Usage

```julia
using CloudMicrophysics.BulkMicrophysicsTendencies

# For 1-moment microphysics
tendencies = bulk_microphysics_tendencies(
    Microphysics1Moment(), mp, tps,
    ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno
)
(; dq_lcl_dt, dq_icl_dt, dq_rai_dt, dq_sno_dt) = tendencies
```
"""
module BulkMicrophysicsTendencies

import ..Parameters as CMP
import ..Utilities as UT
import ..Microphysics0M as CM0
import ..Microphysics1M as CM1
import ..Microphysics2M as CM2
import ..MicrophysicsNonEq as CMNonEq
import ..P3Scheme as CMP3
import ..HetIceNucleation as CM_HetIce
import ...ThermodynamicsInterface as TDI
import ..Common as CO
import ForwardDiff as FD
import StaticArrays as SA

export MicrophysicsScheme,
    Microphysics0Moment,
    Microphysics1Moment,
    Microphysics2Moment,
    TendencyMode,
    Instantaneous,
    InstantaneousVerbose,
    LinearizedAverage,
    RosenbrockAverage,
    Verbose,
    Jacobian,
    DonorJacobian,
    CoupledDonorJacobian,
    ExactJacobian,
    GrowthTreatment,
    ImplicitGrowth,
    ExplicitGrowthDiagonal,
    TendencyLimiter,
    NoLimiter,
    EndStateSaturationAdjustment,
    rosenbrock_donor,
    rosenbrock_coupled,
    rosenbrock_exact,
    bulk_microphysics_tendencies

#####
##### Singleton types for dispatch
#####

"""
    MicrophysicsScheme

Abstract type for microphysics scheme dispatch.
"""
abstract type MicrophysicsScheme end

"""
    Microphysics0Moment <: MicrophysicsScheme

Singleton type for 0-moment microphysics scheme dispatch.
"""
struct Microphysics0Moment <: MicrophysicsScheme end

"""
    Microphysics1Moment <: MicrophysicsScheme

Singleton type for 1-moment microphysics scheme dispatch.
"""
struct Microphysics1Moment <: MicrophysicsScheme end

"""
    Microphysics2Moment <: MicrophysicsScheme

Singleton type for 2-moment microphysics scheme dispatch.

This unified scheme handles:
- Warm rain only (Seifert-Beheng 2006) when ice parameters are not provided
- Warm rain + P3 ice when ice state is provided
"""
struct Microphysics2Moment <: MicrophysicsScheme end

# --- Tendency output mode dispatch ---

"""
    TendencyMode

Abstract type for selecting the output mode of `bulk_microphysics_tendencies`.
"""
abstract type TendencyMode end

Base.broadcastable(m::TendencyMode) = tuple(m)

"""
    Instantaneous <: TendencyMode

Return raw nonlinear tendencies from a single evaluation of all microphysical
processes (no linearization, no time-averaging).
"""
struct Instantaneous <: TendencyMode end

"""
    InstantaneousVerbose <: TendencyMode

Return all individual source terms alongside aggregated `dq_*_dt` tendencies.
Useful for model diagnostics. Only works with instantaneous (nonlinear) evaluation.
"""
struct InstantaneousVerbose <: TendencyMode end

"""
    LinearizedAverage <: TendencyMode

Return time-averaged tendencies computed via repeated linearized implicit substeps.
This is the mode used operationally by ClimaAtmos.
"""
struct LinearizedAverage <: TendencyMode end

"""
    Jacobian

Abstract type selecting the matrix used in each linearized-implicit substep of
[`RosenbrockAverage`](@ref). The supported types and how to add another are
described in the P3 numerics documentation. The 1-moment scheme supports all
three options; the 2M+P3 scheme supports only [`ExactJacobian`](@ref).
"""
abstract type Jacobian end

"""
    DonorJacobian <: Jacobian

The donor-based linearization of the tendency: each transfer is linearized in its
donor species and rate-floored. The matrix used by [`LinearizedAverage`](@ref).
"""
struct DonorJacobian <: Jacobian end

"""
    CoupledDonorJacobian <: Jacobian

The donor-based linearization with the vapor-competition and collector couplings
of the exact derivative restored.
"""
struct CoupledDonorJacobian <: Jacobian end

"""
    ExactJacobian <: Jacobian

The exact derivative of the tendency, formed with `ForwardDiff`.
"""
struct ExactJacobian <: Jacobian end

"""
    ManualJacobian <: Jacobian

A hand-built 2M+P3 substep Jacobian: the stiff condensation/deposition and
number-adjustment couplings carried as closed-form analytic derivatives, the
warm-rain and freezing transfers as donor-based linearizations, and the
mixed-phase quadrature transfers (ice melt, liquid-ice collision)
donor-linearized on their own donor species with the quadrature rates held
frozen. Avoids the `ForwardDiff` pass and its `gamma_inc` shape-derivative
block.
"""
struct ManualJacobian <: Jacobian end

"""
    GrowthTreatment

Abstract type selecting how the positive (growth) diagonal of the Jacobian
enters the implicit operator.
"""
abstract type GrowthTreatment end

"""
    ImplicitGrowth <: GrowthTreatment

Use the Jacobian unchanged.
"""
struct ImplicitGrowth <: GrowthTreatment end

"""
    ExplicitGrowthDiagonal <: GrowthTreatment

Zero the positive diagonal of the Jacobian, so a growth mode is taken explicitly
and only the decay diagonal remains in the implicit operator.
"""
struct ExplicitGrowthDiagonal <: GrowthTreatment end

"""
    TendencyLimiter

Abstract type selecting a limiter applied to the realized substep increment.
"""
abstract type TendencyLimiter end

"""
    NoLimiter <: TendencyLimiter

Apply no limiter to the increment.
"""
struct NoLimiter <: TendencyLimiter end

"""
    EndStateSaturationAdjustment <: TendencyLimiter

Scale a substep increment so the latent-heated end state stays at or above
saturation over its more-supersaturated phase (ice when ice dominates, liquid when
liquid dominates), for cells that begin at or above saturation. Derived and analyzed
in the Rosenbrock substepping documentation.
"""
struct EndStateSaturationAdjustment <: TendencyLimiter end

"""
    RosenbrockAverage(jacobian, growth, limiter) <: TendencyMode
    RosenbrockAverage(; jacobian = DonorJacobian(), growth = ImplicitGrowth(), limiter = NoLimiter())

Time-averaged tendencies from repeated linearized-implicit (Rosenbrock-Euler)
substeps. The [`Jacobian`](@ref), [`GrowthTreatment`](@ref), and
[`TendencyLimiter`](@ref) options select the substep matrix, the growth-diagonal
treatment, and the increment limiter. See [`rosenbrock_donor`](@ref),
[`rosenbrock_coupled`](@ref), and [`rosenbrock_exact`](@ref) for the supported
configurations.
"""
struct RosenbrockAverage{J <: Jacobian, G <: GrowthTreatment, L <: TendencyLimiter} <: TendencyMode
    jacobian::J
    growth::G
    limiter::L
end
RosenbrockAverage(;
    jacobian = DonorJacobian(),
    growth = ImplicitGrowth(),
    limiter = NoLimiter(),
) = RosenbrockAverage(jacobian, growth, limiter)

"""
    rosenbrock_donor()

[`RosenbrockAverage`](@ref) with the donor-based Jacobian. Reproduces
[`LinearizedAverage`](@ref) within the unified framework.
"""
rosenbrock_donor() = RosenbrockAverage(DonorJacobian(), ImplicitGrowth(), NoLimiter())

"""
    rosenbrock_coupled()

[`RosenbrockAverage`](@ref) with the coupled donor-based Jacobian.
"""
rosenbrock_coupled() = RosenbrockAverage(CoupledDonorJacobian(), ImplicitGrowth(), NoLimiter())

"""
    rosenbrock_exact()

[`RosenbrockAverage`](@ref) with the exact Jacobian, an explicit growth diagonal,
and the end-state saturation adjustment.
"""
rosenbrock_exact() =
    RosenbrockAverage(ExactJacobian(), ExplicitGrowthDiagonal(), EndStateSaturationAdjustment())

"""
    rosenbrock_manual()

[`RosenbrockAverage`](@ref) with the hand-built 2M+P3 [`ManualJacobian`](@ref),
implicit growth, and the end-state saturation adjustment.
"""
rosenbrock_manual() =
    RosenbrockAverage(ManualJacobian(), ImplicitGrowth(), EndStateSaturationAdjustment())

"""
    Verbose(mode) <: TendencyMode

Diagnostic wrapper returning, alongside the net tendencies, the per-process
tendencies realized by the implicit solve of `mode`. Each process is attributed
through the same substep factorization, so the per-process tendencies sum to the
net of the unlimited solve; for a `mode` with a `TendencyLimiter`, the wrapped net
excludes the limiter. This is a diagnostic path, separate from the model time
step.
"""
struct Verbose{M <: TendencyMode} <: TendencyMode
    mode::M
end

# --- 1-Moment Microphysics ---

# --- Internal helpers ---

"""
Compute all individual 1-moment microphysics source terms in a single pass.

This is the **single source of truth** for which microphysical processes are
called and with what arguments. Both the raw tendency aggregation and the
linearized operator construction consume this output.

Constructs two `NamedTuple`s that are passed to all process functions
(see `Microphysics1M` module docs for the full convention):
- `micro = (; q_tot, q_lcl, q_icl, q_rai, q_sno)` — specific humidities (kg/kg)
- `thermo = (; ρ, T)` — air density (kg/m³) and temperature (K)

Naming convention: `S_process_species1_species2`
 - process: physical mechanism (phase_change, acnv, accr, melt, accr_melt, accr_freeze)
 - species1, species2: interacting pair (not from/to)
 - `_cold` / `_warm` suffix for two-sided collision arms (inactive arm = zero)

Returns a `NamedTuple` of ~19 scalar source terms.  All two-sided collision
processes are pre-routed by temperature, so consumers never need `is_warm`.
"""
@inline function _microphysics_source_terms(
    ::Microphysics1Moment, mp::CMP.Microphysics1MParams, tps,
    ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno,
)
    # Clamp negative inputs to zero (robustness against numerical errors)
    ρ = UT.clamp_to_nonneg(ρ)
    q_tot = UT.clamp_to_nonneg(q_tot)
    q_lcl = UT.clamp_to_nonneg(q_lcl)
    q_icl = UT.clamp_to_nonneg(q_icl)
    q_rai = UT.clamp_to_nonneg(q_rai)
    q_sno = UT.clamp_to_nonneg(q_sno)

    FT = UT.promote_typeof(ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno)
    opts = mp.options

    # Construct state tuples (reused across all process calls)
    micro = (; q_tot, q_lcl, q_icl, q_rai, q_sno)
    thermo = (; ρ, T)

    # Size-distribution / fall-speed quantities (λ⁻¹, n₀, v₀ for rain/snow/ice) are
    # pow/exp-heavy and shared by several processes per species: compute once and pass
    # to each process via its `sd` argument so they are not recomputed per process.
    sd = CM1.size_distr_parameters(mp, micro, thermo)

    # --- Phase change: vapor ↔ cloud condensate (bidirectional, ±) ---
    S_phase_change_vap_lcl = CMNonEq.conv_q_vap_to_q_lcl(opts.cloud_liquid_formation, mp, tps, micro, thermo)
    S_phase_change_vap_icl = CMNonEq.conv_q_vap_to_q_icl(opts.cloud_ice_formation, mp, tps, micro, thermo)

    # --- Autoconversion (cloud → precipitation, ≥ 0) ---
    S_acnv_lcl_rai = CM1.conv_q_lcl_to_q_rai(opts.rain_autoconversion, mp, tps, micro, thermo)
    S_acnv_icl_sno = CM1.conv_q_icl_to_q_sno(opts.snow_autoconversion, mp, tps, micro, thermo, sd)

    # --- Accretion (collisions between species) ---
    is_warm = T >= TDI.T_freeze(tps)

    # Cloud liquid + rain → rain
    S_accr_lcl_rai = CM1.accretion(opts.cloud_liquid_rain_accretion, mp, tps, micro, thermo, sd)

    # Cloud liquid + snow: product goes to sno (cold) or rai (warm), plus thermal melt
    (; S_accr, S_melt) = CM1.accretion(opts.cloud_liquid_snow_accretion, mp, tps, micro, thermo, sd)
    S_accr_lcl_sno_cold = ifelse(is_warm, zero(FT), S_accr)    # lcl → sno (cold)
    S_accr_lcl_sno_warm = ifelse(is_warm, S_accr, zero(FT))    # lcl → rai (warm)
    S_accr_melt_lcl_sno = S_melt                                # thermal melt of sno from warm lcl (already zero when cold)

    # Cloud ice + rain → snow (ice-side sink)
    S_accr_icl_rai = CM1.accretion(opts.cloud_ice_rain_accretion, mp, tps, micro, thermo, sd)

    # Rain frozen in cloud ice + rain collision → snow (rain sink)
    S_accr_freeze_icl_rai = CM1.accretion_rain_sink(opts.cloud_ice_rain_accretion, mp, tps, micro, thermo, sd)

    # Cloud ice + snow → snow
    S_accr_icl_sno = CM1.accretion(opts.cloud_ice_snow_accretion, mp, tps, micro, thermo, sd)

    # Rain-snow collisions: split into cold/warm arms (inactive arm = zero)
    (; S_rai_sno, S_sno_rai, S_melt) = CM1.accretion_snow_rain(opts.rain_snow_accretion, mp, tps, micro, thermo, sd)
    S_accr_rai_sno_cold = ifelse(is_warm, zero(FT), S_rai_sno) # cold arm: rai freezes → sno
    S_accr_rai_sno_warm = ifelse(is_warm, S_sno_rai, zero(FT)) # warm arm: sno melts → rai
    S_accr_melt_rai_sno = ifelse(is_warm, S_melt, zero(FT))    # thermal melt of sno from warm rai

    # --- Phase change: precipitation ↔ vapor ---
    S_phase_change_vap_rai = CM1.conv_q_rai_to_q_vap(opts.rain_condensation_evaporation, mp, tps, micro, thermo, sd)
    S_phase_change_vap_sno = CM1.conv_q_sno_to_q_vap(opts.snow_deposition_sublimation, mp, tps, micro, thermo, sd)

    # --- Melting ---
    S_melt_icl_lcl = CM1.conv_q_icl_to_q_lcl(opts.cloud_ice_melt, mp, tps, micro, thermo, sd)
    S_melt_sno_rai = CM1.conv_q_sno_to_q_rai(opts.snow_melt, mp, tps, micro, thermo, sd)

    return (;
        S_phase_change_vap_lcl, S_phase_change_vap_icl,
        S_acnv_lcl_rai, S_acnv_icl_sno,
        S_accr_lcl_rai, S_accr_lcl_sno_cold, S_accr_lcl_sno_warm, S_accr_melt_lcl_sno,
        S_accr_icl_rai, S_accr_freeze_icl_rai, S_accr_icl_sno,
        S_accr_rai_sno_cold, S_accr_rai_sno_warm, S_accr_melt_rai_sno,
        S_phase_change_vap_rai, S_phase_change_vap_sno,
        S_melt_icl_lcl, S_melt_sno_rai,
    )
end

"""
Aggregate individual source terms into the four hydrometeor tendency totals.

This is the **single location** where the sign convention of source terms
to tendency accumulators is defined.  All temperature-dependent routing is
pre-applied in `_microphysics_source_terms` (cold/warm arms), so every term
here appears with a fixed sign — no `ifelse` branching.
"""
@inline function _aggregate_tendencies(src)
    dq_lcl_dt =
        src.S_phase_change_vap_lcl - src.S_acnv_lcl_rai - src.S_accr_lcl_rai -
        src.S_accr_lcl_sno_cold - src.S_accr_lcl_sno_warm + src.S_melt_icl_lcl

    dq_icl_dt =
        src.S_phase_change_vap_icl - src.S_acnv_icl_sno - src.S_accr_icl_rai -
        src.S_accr_icl_sno - src.S_melt_icl_lcl

    dq_rai_dt =
        src.S_acnv_lcl_rai + src.S_accr_lcl_rai +
        src.S_accr_lcl_sno_warm + src.S_accr_melt_lcl_sno -
        src.S_accr_freeze_icl_rai -
        src.S_accr_rai_sno_cold + src.S_accr_rai_sno_warm + src.S_accr_melt_rai_sno +
        src.S_phase_change_vap_rai + src.S_melt_sno_rai

    dq_sno_dt =
        src.S_acnv_icl_sno +
        src.S_accr_lcl_sno_cold - src.S_accr_melt_lcl_sno +
        src.S_accr_icl_rai + src.S_accr_freeze_icl_rai +
        src.S_accr_icl_sno +
        src.S_accr_rai_sno_cold - src.S_accr_rai_sno_warm - src.S_accr_melt_rai_sno +
        src.S_phase_change_vap_sno - src.S_melt_sno_rai

    return (; dq_lcl_dt, dq_icl_dt, dq_rai_dt, dq_sno_dt)
end

"""
Construct a local linear approximation of 1-moment microphysics tendencies
from pre-computed source terms:

    dq/dt ≈ M * q + e

using a donor-based linearization:
- donor → receiver transfers are represented as `D * q_donor`
- vapor → condensate sources are treated as constants (`e`)
- condensate sinks are treated as linear sinks (`-D * q`)

All coefficients use `D = S / max(q_min, q_donor)` for robustness.

Returns a `NamedTuple` containing the nonzero entries of `M` and `e`.
"""
@inline function _linearize(src, q_lcl, q_icl, q_rai, q_sno, q_min)
    FT = typeof(src.S_phase_change_vap_lcl)

    M11 = zero(FT)
    M12 = zero(FT)
    M22 = zero(FT)
    M31 = zero(FT)
    M33 = zero(FT)
    M34 = zero(FT)
    M41 = zero(FT)
    M42 = zero(FT)
    M43 = zero(FT)
    M44 = zero(FT)
    e1 = zero(FT)
    e2 = zero(FT)
    e4 = zero(FT)

    # --- Phase change: vapor ↔ cloud condensate ---
    D = src.S_phase_change_vap_lcl / max(q_min, q_lcl)
    is_source = src.S_phase_change_vap_lcl >= zero(FT)
    e1 += ifelse(is_source, src.S_phase_change_vap_lcl, zero(FT))
    M11 += ifelse(is_source, zero(FT), D)

    D = src.S_phase_change_vap_icl / max(q_min, q_icl)
    is_source = src.S_phase_change_vap_icl >= zero(FT)
    e2 += ifelse(is_source, src.S_phase_change_vap_icl, zero(FT))
    M22 += ifelse(is_source, zero(FT), D)

    # --- Melt: ice cloud → liquid cloud ---
    D = src.S_melt_icl_lcl / max(q_min, q_icl)
    M22 -= D
    M12 += D

    # --- Autoconversion: donor-based transfer ---
    D = src.S_acnv_lcl_rai / max(q_min, q_lcl)
    M11 -= D
    M31 += D

    D = src.S_acnv_icl_sno / max(q_min, q_icl)
    M22 -= D
    M42 += D

    # --- Accretion: donor-based transfer ---
    D = src.S_accr_lcl_rai / max(q_min, q_lcl)
    M11 -= D
    M31 += D

    # lcl + sno accretion (cold/warm arms already zeroed)
    D_cold = src.S_accr_lcl_sno_cold / max(q_min, q_lcl)
    D_warm = src.S_accr_lcl_sno_warm / max(q_min, q_lcl)
    M11 -= D_cold + D_warm
    M31 += D_warm           # warm: lcl → rai
    M41 += D_cold           # cold: lcl → sno

    # thermal melt of sno from warm lcl
    D = src.S_accr_melt_lcl_sno / max(q_min, q_sno)
    M44 -= D
    M34 += D

    D = src.S_accr_icl_rai / max(q_min, q_icl)
    M22 -= D
    M42 += D

    D = src.S_accr_icl_sno / max(q_min, q_icl)
    M22 -= D
    M42 += D

    # rain frozen in icl + rai collision
    D = src.S_accr_freeze_icl_rai / max(q_min, q_rai)
    M33 -= D
    M43 += D

    # warm arm: sno melts → rai (already zero when cold)
    D = src.S_accr_rai_sno_warm / max(q_min, q_sno)
    M44 -= D
    M34 += D

    # thermal melt of sno from warm rai (already zero when cold)
    D = src.S_accr_melt_rai_sno / max(q_min, q_sno)
    M44 -= D
    M34 += D

    # cold arm: rai freezes → sno (already zero when warm)
    D = src.S_accr_rai_sno_cold / max(q_min, q_rai)
    M33 -= D
    M43 += D

    # --- Rain phase change: sink to vapor (always zero or negative) ---
    D = (-src.S_phase_change_vap_rai) / max(q_min, q_rai)
    M33 -= D

    # --- Snow phase change: deposition/sublimation ---
    D = src.S_phase_change_vap_sno / max(q_min, q_sno)
    is_source = src.S_phase_change_vap_sno >= zero(FT)
    e4 += ifelse(is_source, src.S_phase_change_vap_sno, zero(FT))
    M44 += ifelse(is_source, zero(FT), D)

    # --- Snow melt: snow → rain ---
    D = src.S_melt_sno_rai / max(q_min, q_sno)
    M44 -= D
    M34 += D

    return (
        M11 = M11, M12 = M12, M22 = M22,
        M31 = M31, M33 = M33, M34 = M34,
        M41 = M41, M42 = M42, M43 = M43, M44 = M44,
        e1 = e1, e2 = e2, e4 = e4,
    )
end

# --- Public API: bulk_microphysics_tendencies with TendencyMode dispatch ---

"""
    bulk_microphysics_tendencies(
        ::Instantaneous, ::Microphysics1Moment, mp, tps,
        ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno,
    )

Compute all 1-moment microphysics tendencies in one fused call.

Returns a NamedTuple with all source/sink terms for hydrometeor species.
This is a pure function of local thermodynamic state, suitable for:
- Point quadrature over subgrid-scale (T, q_tot) distributions
- GPU kernel evaluation
- Unit testing in isolation

# Arguments
- `mp`: Microphysics1MParams parameter container
- `tps`: Thermodynamics parameters
- `ρ`: Air density [kg/m³]
- `T`: Temperature [K]
- `q_tot`: Total water specific content [kg/kg]
- `q_lcl`: Cloud liquid water specific content [kg/kg]
- `q_icl`: Cloud ice specific content [kg/kg]
- `q_rai`: Rain specific content [kg/kg]
- `q_sno`: Snow specific content [kg/kg]

# Returns
`NamedTuple` with fields:
- `dq_lcl_dt`: Cloud liquid tendency [kg/kg/s]
- `dq_icl_dt`: Cloud ice tendency [kg/kg/s]
- `dq_rai_dt`: Rain tendency [kg/kg/s]
- `dq_sno_dt`: Snow tendency [kg/kg/s]

# Notes
- Negative specific contents are clamped to zero for robustness.
- Does NOT apply timestep-dependent limiters.
"""
@inline function bulk_microphysics_tendencies(
    ::Instantaneous, ::Microphysics1Moment, mp::CMP.Microphysics1MParams, tps,
    ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno,
)
    src = _microphysics_source_terms(
        Microphysics1Moment(), mp, tps,
        ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno,
    )
    return _aggregate_tendencies(src)
end

"""
    bulk_microphysics_tendencies(
        ::InstantaneousVerbose, ::Microphysics1Moment, mp, tps,
        ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno,
    )

Compute all 1-moment microphysics tendencies and return both aggregated
tendencies (`dq_*_dt`) and all individual source terms (`S_*`).

Useful for model diagnostics. The `dq_*_dt` fields are identical to those
returned by `Instantaneous()`.

# Returns
`NamedTuple` with all fields from `Instantaneous()` plus individual source
terms: `S_phase_change_vap_lcl`, `S_phase_change_vap_icl`, `S_acnv_lcl_rai`,
`S_acnv_icl_sno`, etc.
"""
@inline function bulk_microphysics_tendencies(
    ::InstantaneousVerbose, ::Microphysics1Moment, mp::CMP.Microphysics1MParams, tps,
    ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno,
)
    src = _microphysics_source_terms(
        Microphysics1Moment(), mp, tps,
        ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno,
    )
    agg = _aggregate_tendencies(src)
    return merge(agg, src)
end

"""
    bulk_microphysics_tendencies(
        ::LinearizedAverage, ::Microphysics1Moment, mp, tps,
        ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno, Δt, nsub = 1,
    )

Compute average 1-moment microphysics tendencies over `Δt` using repeated
linearized implicit substeps. Forwards to [`rosenbrock_donor`](@ref), the
donor-based configuration of [`RosenbrockAverage`](@ref).

# Returns
`NamedTuple` with fields:
- `dq_lcl_dt`: Cloud liquid tendency [kg/kg/s]
- `dq_icl_dt`: Cloud ice tendency [kg/kg/s]
- `dq_rai_dt`: Rain tendency [kg/kg/s]
- `dq_sno_dt`: Snow tendency [kg/kg/s]
"""
@inline bulk_microphysics_tendencies(
    ::LinearizedAverage, cm::Microphysics1Moment, mp::CMP.Microphysics1MParams, tps,
    ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno, Δt, nsub = 1,
) = bulk_microphysics_tendencies(
    rosenbrock_donor(), cm, mp, tps,
    ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno, Δt, nsub,
)


# --- 0-Moment Microphysics ---
"""
    bulk_microphysics_tendencies(::Microphysics0Moment, mp, tps, T, q_lcl, q_icl)
    bulk_microphysics_tendencies(::Microphysics0Moment, mp, tps, T, q_lcl, q_icl, q_vap_sat)

Compute 0-moment microphysics tendencies in one fused call.

Returns the total water tendency `dq_tot_dt` (a scalar, in kg/kg/s) from precipitation removal.

The first form uses the fixed condensate threshold `qc_0`;
the second form uses the supersaturation threshold `S_0 * q_vap_sat`.

# Arguments
- `mp`: Microphysics0MParams (contains τ_precip, qc_0, S_0)
- `tps`: Thermodynamics parameters
- `T`: Temperature [K]
- `q_lcl`: Cloud liquid specific content [kg/kg]
- `q_icl`: Cloud ice specific content [kg/kg]
- `q_vap_sat`: (second method only) Saturation specific humidity [kg/kg]

# Notes
- Does NOT apply limiters (caller applies based on timestep)
"""
@inline function bulk_microphysics_tendencies(
    ::Microphysics0Moment, mp::CMP.Microphysics0MParams, tps,
    T, q_lcl, q_icl,
)
    q_lcl = UT.clamp_to_nonneg(q_lcl)
    q_icl = UT.clamp_to_nonneg(q_icl)
    dq_tot_dt = CM0.remove_precipitation(mp.precip, q_lcl, q_icl)
    return dq_tot_dt
end
@inline function bulk_microphysics_tendencies(
    ::Microphysics0Moment,
    mp::CMP.Microphysics0MParams,
    tps,
    T,
    q_lcl,
    q_icl,
    q_vap_sat,
)
    q_lcl = UT.clamp_to_nonneg(q_lcl)
    q_icl = UT.clamp_to_nonneg(q_icl)
    dq_tot_dt = CM0.remove_precipitation(mp.precip, q_lcl, q_icl, q_vap_sat)
    return dq_tot_dt
end

# --- 2-Moment Microphysics Helper Functions ---

"""
    warm_rain_tendencies_2m(sb, q_lcl, q_rai, ρ, n_lcl, n_rai)

Internal helper function that computes 2M warm rain processes:
autoconversion, self-collection, accretion, and rain breakup.

Used by both warm-only and warm+ice dispatch methods to reduce code duplication.

# Arguments
- `sb`: SB2006 parameters
- `q_lcl`: Cloud liquid specific content (kg/kg)
- `q_rai`: Rain specific content (kg/kg)
- `ρ`: Air density (kg/m³)
- `n_lcl`: Cloud droplet number per kg air (1/kg)
- `n_rai`: Rain number per kg air (1/kg)

# Returns
`NamedTuple` with warm rain tendencies:
- `dq_lcl_dt`: Cloud liquid mass tendency (kg/kg/s)
- `dq_rai_dt`: Rain mass tendency (kg/kg/s)
- `dn_lcl_dt`: Cloud number tendency (1/kg/s)
- `dn_rai_dt`: Rain number tendency (1/kg/s)
"""
@inline function warm_rain_tendencies_2m(
    warm_rain, tps, T, q_tot, q_lcl, q_rai, q_ice, ρ, n_lcl, n_rai,
    w = zero(ρ), p = zero(ρ),
)

    # Unpack parameters
    sb = warm_rain.seifert_beheng
    aps = warm_rain.air_properties
    condevap = warm_rain.condevap

    # Convert to number densities for CM2 functions
    N_lcl = ρ * n_lcl
    N_rai = ρ * n_rai

    # Initialize tendencies
    FT = typeof(ρ)
    dq_lcl_dt = zero(FT)
    dq_rai_dt = zero(FT)
    dn_lcl_dt = zero(FT)
    dn_rai_dt = zero(FT)

    # --- Aerosol activation ---
    dn_lcl_activation_dt = zero(FT)

    # --- Condensation of vapor / evaporation of cloud liquid water ---
    micro_mock = (; q_tot, q_lcl, q_icl = q_ice, q_rai, q_sno = zero(q_ice))
    thermo_mock = (; ρ, T)
    ∂ₜq_lcl_cond = CMNonEq.conv_q_vap_to_q_lcl(
        CMP.CloudLiquidFormation(condevap.τ_relax), nothing, tps, micro_mock, thermo_mock,
    )
    ∂ₜn_lcl_cond = zero(∂ₜq_lcl_cond)  # neglect number change from condensation/evaporation
    dq_lcl_dt += ∂ₜq_lcl_cond
    dn_lcl_dt += ∂ₜn_lcl_cond

    # --- Evaporation of rain ---
    evap = CM2.rain_evaporation(sb, aps, tps, q_tot, q_lcl, q_ice, q_rai, zero(q_ice), ρ, N_rai, T)
    dq_rai_dt += evap.∂ₜq_rai
    dn_rai_dt += evap.∂ₜρn_rai / ρ

    # --- Autoconversion ---
    acnv = CM2.autoconversion(sb.acnv, sb.pdf_c, q_lcl, q_rai, ρ, N_lcl)
    dq_lcl_dt += acnv.dq_lcl_dt
    dq_rai_dt += acnv.dq_rai_dt
    dn_lcl_dt += acnv.dN_lcl_dt / ρ
    dn_rai_dt += acnv.dN_rai_dt / ρ

    # --- Cloud liquid self-collection ---
    ∂ₜN_lcl_sc = CM2.cloud_liquid_self_collection(sb.acnv, sb.pdf_c, q_lcl, ρ, acnv.dN_lcl_dt)
    dn_lcl_dt += ∂ₜN_lcl_sc / ρ

    # --- Accretion ---
    accr = CM2.accretion(sb, q_lcl, q_rai, ρ, N_lcl)
    dq_lcl_dt += accr.dq_lcl_dt
    dq_rai_dt += accr.dq_rai_dt
    dn_lcl_dt += accr.dN_lcl_dt / ρ

    # --- Rain self-collection ---
    ∂ρn_rai_sc_∂t = CM2.rain_self_collection(sb.pdf_r, sb.self, q_rai, ρ, N_rai)
    dn_rai_dt += ∂ρn_rai_sc_∂t / ρ

    # --- Rain breakup ---
    ∂ρn_rai_br_∂t = CM2.rain_breakup(sb.pdf_r, sb.brek, q_rai, ρ, N_rai, ∂ρn_rai_sc_∂t)
    dn_rai_dt += ∂ρn_rai_br_∂t / ρ

    # --- Number adjustment for mass limits ---
    # Cloud liquid
    numadj_lcl = (; sb.numadj.τ, x_min = sb.pdf_c.xc_min, x_max = sb.pdf_c.xc_max)
    ∂ₜn_lcl_numadj = CM2.number_tendency_from_mass_limits(numadj_lcl, q_lcl, n_lcl)
    dn_lcl_dt += ∂ₜn_lcl_numadj
    # Rain
    numadj_rai = (; sb.numadj.τ, x_min = sb.pdf_r.xr_min, x_max = sb.pdf_r.xr_max)
    ∂ₜn_rai_numadj = CM2.number_tendency_from_mass_limits(numadj_rai, q_rai, n_rai)
    dn_rai_dt += ∂ₜn_rai_numadj

    return (; dq_lcl_dt, dq_rai_dt, dn_lcl_dt, dn_rai_dt, dn_lcl_activation_dt)
end

# --- 2-Moment Microphysics (Unified Warm + Optional Ice) ---

"""
    _p3_ice_tendency_fields(dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt[, dq_liq_on_ice_dt][, dz_ice_dt])

The per-category P3 ice tendency fields, in the canonical order. The trailing
`dq_liq_on_ice_dt` (predicted liquid fraction) and `dz_ice_dt` (three-moment ice)
slots are appended when present; pass `nothing` to omit either.
"""
@inline _p3_ice_tendency_fields(
    dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_on_ice_dt = nothing, dz_ice_dt = nothing,
) = _append_z_field(
    _append_liq_field((; dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt), dq_liq_on_ice_dt),
    dz_ice_dt,
)
@inline _append_liq_field(nt, ::Nothing) = nt
@inline _append_liq_field(nt, dq_liq_on_ice_dt) = (; nt..., dq_liq_on_ice_dt)
@inline _append_z_field(nt, ::Nothing) = nt
@inline _append_z_field(nt, dz_ice_dt) = (; nt..., dz_ice_dt)

"""
    _categories_tendency_fields(blocks::NTuple{NCAT, <:NamedTuple})

Merge the per-category ice tendency blocks into the flat tendency fields. A
single category keeps the unsuffixed names; for `NCAT > 1` each block's fields
carry the category index before the `_dt` suffix (`dq_ice_dt` of category 2
becomes `dq_ice_2_dt`, matching the `_1.._N` suffixes of the host's per-category
prognostic scalars).
"""
@inline _categories_tendency_fields(b::NTuple{1, <:NamedTuple}) = b[1]
@inline _categories_tendency_fields(b::NTuple{2, <:NamedTuple}) =
    merge(_suffixed_category_fields(b[1], Val(1)), _suffixed_category_fields(b[2], Val(2)))
@inline _categories_tendency_fields(b::NTuple{3, <:NamedTuple}) = merge(
    _categories_tendency_fields((b[1], b[2])),
    _suffixed_category_fields(b[3], Val(3)),
)
@inline _categories_tendency_fields(b::NTuple{4, <:NamedTuple}) = merge(
    _categories_tendency_fields((b[1], b[2], b[3])),
    _suffixed_category_fields(b[4], Val(4)),
)

# Generated so the suffixed field names are compile-time constants (symbol
# construction cannot constant-fold through `ntuple`).
@generated function _suffixed_category_fields(nt::NamedTuple{names}, ::Val{j}) where {names, j}
    suffixed = map(names) do n
        s = String(n)
        endswith(s, "_dt") || error("tendency field `$n` does not end in `_dt`")
        Symbol(s[1:(end - 3)], :_, j, :_dt)
    end
    return :(NamedTuple{$suffixed}(values(nt)))
end

"""
    _moments(mp::Microphysics2MParams)

The [`CMP.MomentClosure`](@ref) of the P3 ice scheme carried by `mp`.
"""
@inline _moments(mp::CMP.Microphysics2MParams) = mp.ice.scheme.moments

"""
    _cat_ρz(moments, cat, ρ)

Volumetric sixth moment of the packed ice-category input `cat`: `cat.z_ice · ρ`
under [`CMP.ThreeMoment`](@ref) ice, `nothing` under [`CMP.TwoMoment`](@ref)
(the input carries no `z_ice` field).
"""
@inline _cat_ρz(::CMP.TwoMoment, cat, ρ) = nothing
@inline _cat_ρz(::CMP.ThreeMoment, cat, ρ) = UT.clamp_to_nonneg(cat.z_ice) * ρ

"""
    _reflectivity_coefficients(moments, mp, ρ, ice, shapes, zcoeffs)

Per-category frozen [`CMP3.ReflectivityCoefficients`](@ref) of the constant-μ
reflectivity closure: `nothing` under [`CMP.TwoMoment`](@ref) ice; under
[`CMP.ThreeMoment`](@ref), the passed `zcoeffs` when provided, else computed
from the packed inputs and the frozen shapes. Callers inside a differentiated
region must pass coefficients precomputed from the primal state.
"""
@inline _reflectivity_coefficients(::CMP.TwoMoment, mp, ρ, ice, shapes, zcoeffs) = nothing
@inline _reflectivity_coefficients(::CMP.ThreeMoment, mp, ρ, ice, shapes, zcoeffs) = zcoeffs
@inline function _reflectivity_coefficients(
    moments::CMP.ThreeMoment, mp, ρ, ice::NTuple{NCAT, <:NamedTuple}, shapes, ::Nothing,
) where {NCAT}
    liquid = _liquid(mp)
    return ntuple(Val(NCAT)) do j
        (; q_ice, n_ice, q_rim, b_rim) = ice[j]
        state = CMP3.state_from_prognostic(
            mp.ice.scheme, q_ice * ρ, n_ice * ρ, q_rim * ρ, b_rim * ρ,
            _cat_ρq_liq(liquid, ice[j], ρ), _cat_ρz(moments, ice[j], ρ),
        )
        CMP3.reflectivity_growth_coefficients(state, shapes[j])
    end
end

"""
    _liquid(mp::Microphysics2MParams)

The [`CMP.LiquidFractionTreatment`](@ref) of the P3 ice scheme carried by `mp`.
"""
@inline _liquid(mp::CMP.Microphysics2MParams) = mp.ice.scheme.liquid

"""
    _cat_ρq_liq(liquid, cat, ρ)

Volumetric liquid mass on ice of the packed ice-category input `cat`:
`cat.q_liq_on_ice · ρ` under [`CMP.PredictedLiquidFraction`](@ref), `nothing`
under [`CMP.NoLiquidFraction`](@ref) (the input has no `q_liq_on_ice`
field).
"""
@inline _cat_ρq_liq(::CMP.NoLiquidFraction, cat, ρ) = nothing
@inline _cat_ρq_liq(::CMP.PredictedLiquidFraction, cat, ρ) = UT.clamp_to_nonneg(cat.q_liq_on_ice) * ρ

"""
    _ice_condensate(liquid, q_ice, cat)

Ice-side condensate for the vapor and heat budgets: the frozen core `q_ice`,
plus the liquid on ice under [`CMP.PredictedLiquidFraction`](@ref) (the
reference total ice mass includes the liquid).
"""
@inline _ice_condensate(::CMP.NoLiquidFraction, q_ice, cat) = q_ice
@inline _ice_condensate(::CMP.PredictedLiquidFraction, q_ice, cat) =
    q_ice + UT.clamp_to_nonneg(cat.q_liq_on_ice)

"""
    _ice_tendency_fields(moments, liquid, zc_cat, p3, ρ, acc, init, dq_liq_dt)

Assemble one category's ice tendency fields from its accumulated specific rates
`acc`, in the canonical order. Appends `dq_liq_on_ice_dt` under
[`CMP.PredictedLiquidFraction`](@ref) and `dz_ice_dt` under
[`CMP.ThreeMoment`](@ref). The reflectivity term is the constant-μ growth over
the category's net-of-initiation rates with its frozen coefficient `zc_cat`,
plus the initiation terms of the rates in `init`. See the [three-moment
documentation](@ref P3-three-moment-ice) for the term forms.
"""
@inline function _ice_tendency_fields(moments, liquid, zc_cat, p3, ρ, acc, init, dq_liq_dt)
    dq_liq = _liquid_tendency_slot(liquid, dq_liq_dt)
    dz = _reflectivity_tendency_slot(moments, zc_cat, p3, ρ, acc, init)
    return _p3_ice_tendency_fields(
        acc.dq_ice_dt, acc.dn_ice_dt, acc.dq_rim_dt, acc.db_rim_dt, dq_liq, dz,
    )
end

@inline _liquid_tendency_slot(::CMP.NoLiquidFraction, dq_liq_dt) = nothing
@inline _liquid_tendency_slot(::CMP.PredictedLiquidFraction, dq_liq_dt) = dq_liq_dt

@inline _reflectivity_tendency_slot(::CMP.TwoMoment, zc_cat, p3, ρ, acc, init) = nothing
@inline function _reflectivity_tendency_slot(moments::CMP.ThreeMoment, zc_cat, p3, ρ, acc, init)
    μ_init = moments.μ_init
    dq_init = init.dq_nuc + init.dq_cldfrz + init.dq_raifrz
    dn_init = init.dn_nuc + init.dn_cldfrz + init.dn_raifrz
    dZ_init =
        CMP3.reflectivity_initiation_monodisperse(μ_init, init.D_nuc, init.dn_nuc * ρ) +
        CMP3.reflectivity_initiation_freezing(moments, p3.ρ_i, μ_init, init.dq_cldfrz * ρ, init.dn_cldfrz * ρ) +
        CMP3.reflectivity_initiation_freezing(moments, p3.ρ_i, zero(μ_init), init.dq_raifrz * ρ, init.dn_raifrz * ρ)
    dL_growth = (acc.dq_ice_dt - dq_init) * ρ
    dN_growth = (acc.dn_ice_dt - dn_init) * ρ
    return (CMP3.reflectivity_growth_tendency(zc_cat, dL_growth, dN_growth) + dZ_init) / ρ
end

# Per-category reflectivity coefficient of the frozen tuple; `nothing` under
# two-moment ice.
@inline _cat_zcoeff(::Nothing, j::Integer) = nothing
@inline _cat_zcoeff(zc::Tuple, j::Integer) = zc[j]

# Moment-closure-only assembly: appends `dz_ice_dt` under three-moment ice. The
# liquid-off entry point of the reflectivity assembly.
@inline _ice_tendency_fields(moments::CMP.MomentClosure, zcoeffs, p3, ρ, acc, init) =
    _p3_ice_tendency_fields(
        acc.dq_ice_dt, acc.dn_ice_dt, acc.dq_rim_dt, acc.db_rim_dt,
        nothing, _reflectivity_tendency_slot(moments, _cat_zcoeff(zcoeffs, 1), p3, ρ, acc, init),
    )

"""
    _collision_liquid_source(liquid, coll)

Retained-liquid source of the collision rates `coll` [kg/m³/s]: `coll.∂ₜL_liq`
under [`CMP.PredictedLiquidFraction`](@ref), zero otherwise.
"""
@inline _collision_liquid_source(::CMP.NoLiquidFraction, coll) = zero(coll.∂ₜq_c)
@inline _collision_liquid_source(::CMP.PredictedLiquidFraction, coll) = coll.∂ₜL_liq

"""
    _melt_accumulate(liquid, vel, aps, tps, T, T_freeze, ρ, state, shape, quad,
        dq_rai_dt, dn_rai_dt, dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt)

Accumulate the melting tendencies onto the passed specific-rate accumulators and
return them. Under [`CMP.NoLiquidFraction`](@ref) the whole melt converts ice to
rain; under [`CMP.PredictedLiquidFraction`](@ref) the `D_th` partition of
[`CMP3.ice_melt`](@ref) routes complete melt to rain and retained melt to the
liquid on ice, with the explicit frozen-core double entry.
"""
@inline function _melt_accumulate(
    ::CMP.NoLiquidFraction, vel, aps, tps, T, T_freeze, ρ, state, shape, quad,
    dq_rai_dt, dn_rai_dt, dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt,
)
    FT = eltype(ρ)
    melt = ifelse(T > T_freeze,
        CMP3.ice_melt(vel, aps, tps, T, ρ, state, shape; quad),
        (; dNdt = zero(ρ), dLdt = zero(ρ)),
    )
    # Specific (per-kg-air) ice-mass melt rate.
    ∂ₜq_ice_melt = melt.dLdt / ρ
    ∂ₜn_ice_melt = melt.dNdt / ρ
    # Melting converts ice to rain.
    dq_rai_dt += ∂ₜq_ice_melt
    dn_rai_dt += ∂ₜn_ice_melt  # Melted ice becomes rain drops
    dq_ice_dt -= ∂ₜq_ice_melt
    dn_ice_dt -= ∂ₜn_ice_melt  # Ice particles consumed by melting
    # Rim mass and rim volume drain proportionally to ice mass during melting
    dq_rim_dt -= ∂ₜq_ice_melt * state.F_rim
    db_rim_dt -= ifelse(state.ρ_rim > 0, ∂ₜq_ice_melt * state.F_rim / state.ρ_rim, zero(FT))
    return (dq_rai_dt, dn_rai_dt, dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt)
end
@inline function _melt_accumulate(
    ::CMP.PredictedLiquidFraction, vel, aps, tps, T, T_freeze, ρ, state, shape, quad,
    dq_rai_dt, dn_rai_dt, dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt,
)
    # The partitioned melt rates vanish below freezing (the temperature
    # driver is negative and the rates are floored at zero).
    melt = CMP3.ice_melt(vel, aps, tps, T, ρ, state, shape; quad)
    dq_rai_dt += melt.dLdt_rain / ρ
    dn_rai_dt += melt.dNdt_rain / ρ
    dq_ice_dt += melt.dLdt_ice / ρ
    dn_ice_dt -= melt.dNdt_rain / ρ
    dq_liq_dt += melt.dLdt_liq / ρ
    dq_rim_dt += melt.dLdt_rim / ρ
    db_rim_dt += melt.dBdt_rim / ρ
    return (dq_rai_dt, dn_rai_dt, dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt)
end

"""
    _refreeze_shed_accumulate(liquid, vel, aps, tps, T, ρ, state, shape, quad,
        dq_rai_dt, dn_rai_dt, dq_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt)

Accumulate refreezing ([`CMP3.ice_refreeze`](@ref)) and shedding
([`CMP3.ice_shed`](@ref), converted to a rate over `τ_shd`) onto the passed
specific-rate accumulators and return them. Both processes exist only under
[`CMP.PredictedLiquidFraction`](@ref).
"""
@inline _refreeze_shed_accumulate(
    ::CMP.NoLiquidFraction, vel, aps, tps, T, ρ, state, shape, quad,
    dq_rai_dt, dn_rai_dt, dq_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt,
) = (dq_rai_dt, dn_rai_dt, dq_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt)
@inline function _refreeze_shed_accumulate(
    liquid::CMP.PredictedLiquidFraction, vel, aps, tps, T, ρ, state, shape, quad,
    dq_rai_dt, dn_rai_dt, dq_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt,
)
    # Refreezing (below freezing): liquid on ice joins the rime.
    frz = CMP3.ice_refreeze(vel, aps, tps, T, ρ, state, shape; quad)
    dq_liq_dt += frz.dLdt_liq / ρ
    dq_ice_dt += frz.dLdt_ice / ρ
    dq_rim_dt += frz.dLdt_rim / ρ
    db_rim_dt += frz.dBdt_rim / ρ
    # Shedding: the shed-able liquid mass leaves as rain over τ_shd.
    shd = CMP3.ice_shed(state, shape)
    dq_liq_dt -= shd.L_shd / liquid.τ_shd / ρ
    dq_rai_dt += shd.L_shd / liquid.τ_shd / ρ
    dn_rai_dt += shd.N_shd / liquid.τ_shd / ρ
    return (dq_rai_dt, dn_rai_dt, dq_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt)
end

"""
    _residual_liquid_to_rain(liquid, p3, ρ, q_ice, cat,
        dq_rai_dt, dn_rai_dt, dq_liq_dt)

Drain residual liquid on an emptied ice core to rain. The liquid transfer
processes act only where the core is present (`q_ice` above the presence
threshold); below it, the liquid on ice has no ice to reside on and transfers
to rain as drops of diameter `D_shd_drop` over `τ_shd`. The rate ramps linearly
to zero at the presence threshold, so the sink is continuous in `q_ice` and
identically zero wherever the transfer processes are active. No-op under
[`CMP.NoLiquidFraction`](@ref).
"""
@inline _residual_liquid_to_rain(
    ::CMP.NoLiquidFraction, p3, ρ, q_ice, cat, dq_rai_dt, dn_rai_dt, dq_liq_dt,
) = (dq_rai_dt, dn_rai_dt, dq_liq_dt)
@inline function _residual_liquid_to_rain(
    liquid::CMP.PredictedLiquidFraction, p3, ρ, q_ice, cat, dq_rai_dt, dn_rai_dt, dq_liq_dt,
)
    FT = eltype(ρ)
    ϵₘ = UT.ϵ_numerics_2M_M(FT)
    q_liq = UT.clamp_to_nonneg(cat.q_liq_on_ice)
    r = UT.clamp_to_nonneg(1 - UT.clamp_to_nonneg(q_ice) / ϵₘ)
    drain = r * q_liq / liquid.τ_shd
    m_drop = p3.ρ_l * CO.volume_sphere_D(liquid.D_shd_drop)
    dq_liq_dt -= drain
    dq_rai_dt += drain
    dn_rai_dt += drain / m_drop
    return (dq_rai_dt, dn_rai_dt, dq_liq_dt)
end

"""
    _vapor_exchange_accumulate(liquid, subdep, tps, ρ, T, q_tot, q_lcl, q_rai,
        q_ice, n_ice, cat, state, q_other, share_core, share_liq,
        dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt)

Accumulate one ice category's vapor exchange onto the passed specific-rate
accumulators and return them. Under [`CMP.NoLiquidFraction`](@ref) this is the
core deposition/sublimation relaxation with its number and rime pathways. Under
[`CMP.PredictedLiquidFraction`](@ref) the exchange splits by the
[`CMP3.vapor_path_weight`](@ref) ramp `w`: the core deposition/sublimation
(scaled by `1 - w`, with the liquid on ice in the vapor and heat budgets, and
sublimation limited to the core) and the liquid-shell condensation/evaporation
(scaled by `w`, a source/sink of the liquid on ice, with evaporation reducing
the ice number in proportion to the whole mass). Both paths are bulk
relaxations toward ice and liquid saturation on the `SubDep2M` timescale, not
PSD-resolved integrals, consistent with the `NoLiquidFraction` treatment.

For multiple ice categories `q_other` carries the other categories' ice
condensate (so the vapor and heat budgets see the full condensate), and the
constant-timescale relaxation is partitioned among the categories by the mass
shares `share_core` (deposition/sublimation) and `share_liq` (shell
condensation/evaporation); see [`_category_share`](@ref). Both are one for a
single category.
"""
@inline function _vapor_exchange_accumulate(
    ::CMP.NoLiquidFraction, subdep, tps, ρ, T, q_tot, q_lcl, q_rai, q_ice, n_ice, cat, state,
    q_other, share_core, share_liq,
    dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt,
)
    FT = eltype(ρ)
    ϵₘ = UT.ϵ_numerics_2M_M(FT)
    n_per_q_ice = ifelse(q_ice > ϵₘ, n_ice / q_ice, zero(n_ice))
    # Deposition/sublimation of cloud ice
    micro_mock = (; q_tot, q_lcl, q_icl = q_ice, q_rai, q_sno = q_other)
    thermo_mock = (; ρ, T)
    ∂ₜq_ice_dep = CMNonEq.conv_q_vap_to_q_icl(
        CMP.ConstantTimescale(subdep.τ_relax), nothing, tps, micro_mock, thermo_mock,
    )
    # No ice deposition above freezing (lack of INPs)
    ∂ₜq_ice_dep = ifelse(T > tps.T_freeze, min(∂ₜq_ice_dep, zero(T)), ∂ₜq_ice_dep)
    # During sublimation, the number of ice particles decreases in proportion to the mean ice mass
    # During deposition, the number of ice particles remain unchanged
    ∂ₜn_ice_dep = ifelse(∂ₜq_ice_dep < 0, n_per_q_ice * ∂ₜq_ice_dep, zero(∂ₜq_ice_dep))
    dq_ice_dt += share_core * ∂ₜq_ice_dep
    dn_ice_dt += share_core * ∂ₜn_ice_dep
    ∂ₜq_ice_sub = min(∂ₜq_ice_dep, 0)   # ≤ 0; zero on the deposition branch
    dq_rim_dt += share_core * (∂ₜq_ice_sub * state.F_rim)
    db_rim_dt += share_core * ifelse(state.ρ_rim > 0, ∂ₜq_ice_sub * state.F_rim / state.ρ_rim, zero(FT))
    return (dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt)
end
@inline function _vapor_exchange_accumulate(
    liquid::CMP.PredictedLiquidFraction, subdep, tps, ρ, T, q_tot, q_lcl, q_rai, q_ice, n_ice, cat, state,
    q_other, share_core, share_liq,
    dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt,
)
    FT = eltype(ρ)
    ϵₘ = UT.ϵ_numerics_2M_M(FT)
    w = CMP3.vapor_path_weight(liquid, state.F_liq)
    q_liq = UT.clamp_to_nonneg(cat.q_liq_on_ice)
    thermo_mock = (; ρ, T)
    # Core deposition/sublimation, weighted by 1 - w. The liquid on ice enters
    # the vapor and heat budgets through the q_sno slot; sublimation stays
    # limited to the core mass q_icl.
    n_per_q_ice = ifelse(q_ice > ϵₘ, n_ice / q_ice, zero(n_ice))
    micro_ice = (; q_tot, q_lcl, q_icl = q_ice, q_rai, q_sno = q_liq + q_other)
    ∂ₜq_ice_dep = CMNonEq.conv_q_vap_to_q_icl(
        CMP.ConstantTimescale(subdep.τ_relax), nothing, tps, micro_ice, thermo_mock,
    )
    ∂ₜq_ice_dep = ifelse(T > tps.T_freeze, min(∂ₜq_ice_dep, zero(T)), ∂ₜq_ice_dep)
    ∂ₜn_ice_dep = ifelse(∂ₜq_ice_dep < 0, n_per_q_ice * ∂ₜq_ice_dep, zero(∂ₜq_ice_dep))
    dq_ice_dt += share_core * ((1 - w) * ∂ₜq_ice_dep)
    dn_ice_dt += share_core * ((1 - w) * ∂ₜn_ice_dep)
    ∂ₜq_ice_sub = min(∂ₜq_ice_dep, 0)
    dq_rim_dt += share_core * ((1 - w) * ∂ₜq_ice_sub * state.F_rim)
    db_rim_dt += share_core * ((1 - w) * ifelse(state.ρ_rim > 0, ∂ₜq_ice_sub * state.F_rim / state.ρ_rim, zero(FT)))
    # Liquid-shell condensation/evaporation, weighted by w, relaxing toward
    # liquid saturation; evaporation is limited to the liquid on ice (the
    # q_lcl slot of the shell state) and reduces the ice number in proportion
    # to the whole mass.
    micro_shell = (; q_tot, q_lcl = q_liq, q_icl = q_ice, q_rai = q_lcl + q_rai, q_sno = q_other)
    ∂ₜq_shell = CMNonEq.conv_q_vap_to_q_lcl(
        CMP.CloudLiquidFormation(subdep.τ_relax), nothing, tps, micro_shell, thermo_mock,
    )
    n_per_q_tot = ifelse(q_ice + q_liq > ϵₘ, n_ice / (q_ice + q_liq), zero(n_ice))
    dq_liq_dt += share_liq * (w * ∂ₜq_shell)
    dn_ice_dt += share_liq * (w * ifelse(∂ₜq_shell < 0, n_per_q_tot * ∂ₜq_shell, zero(∂ₜq_shell)))
    return (dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt)
end

"""
    _category_share(::Val{NCAT}, q, q_tot, ϵₘ)

Mass share of one ice category in the constant-timescale vapor exchange: the
category's mass fraction of the categories' total, regularized by `ϵₘ`. One for
a single category, where the relaxation applies unconditionally.
"""
@inline _category_share(::Val{1}, q, q_tot, ϵₘ) = one(q)
@inline _category_share(::Val{NCAT}, q, q_tot, ϵₘ) where {NCAT} = q / max(q_tot, ϵₘ)

@inline _liquid_category_share(::CMP.NoLiquidFraction, ncat, cat, q_liq_tot, ϵₘ) = one(ϵₘ)
@inline _liquid_category_share(::CMP.PredictedLiquidFraction, ::Val{1}, cat, q_liq_tot, ϵₘ) = one(ϵₘ)
@inline _liquid_category_share(::CMP.PredictedLiquidFraction, ::Val{NCAT}, cat, q_liq_tot, ϵₘ) where {NCAT} =
    UT.clamp_to_nonneg(cat.q_liq_on_ice) / max(q_liq_tot, ϵₘ)

@inline _cat_q_ice(c) = c.q_ice
@inline _cat_n_ice(c) = c.n_ice
@inline _cat_q_liq_clamped(c) = UT.clamp_to_nonneg(c.cat.q_liq_on_ice)

@inline _liquid_condensate_total(::CMP.NoLiquidFraction, cats, ::Type{FT}) where {FT} = zero(FT)
@inline _liquid_condensate_total(::CMP.PredictedLiquidFraction, cats, ::Type{FT}) where {FT} =
    reduce(+, map(_cat_q_liq_clamped, cats))

"""
    _entry_categories(mp, moments, liquid, ρ, ice, shapes)

Per-category clamped prognostic inputs, volumetric [`CMP3.P3State`](@ref), and
frozen shape of the packed entry, as an `NCAT`-tuple of `NamedTuple`s.
"""
@inline function _entry_categories(mp, moments, liquid, ρ, ice::NTuple{NCAT, <:NamedTuple}, shapes) where {NCAT}
    return ntuple(Val(NCAT)) do j
        cat = ice[j]
        q_ice = UT.clamp_to_nonneg(cat.q_ice)
        n_ice = UT.clamp_to_nonneg(cat.n_ice)
        q_rim = UT.clamp_to_nonneg(cat.q_rim)
        b_rim = UT.clamp_to_nonneg(cat.b_rim)
        state = CMP3.state_from_prognostic(
            mp.ice.scheme, q_ice * ρ, n_ice * ρ, q_rim * ρ, b_rim * ρ,
            _cat_ρq_liq(liquid, cat, ρ), _cat_ρz(moments, cat, ρ),
        )
        (; q_ice, n_ice, q_rim, b_rim, state, shape = shapes[j], cat)
    end
end

@inline _category_condensates(liquid, cats) = map(c -> _ice_condensate(liquid, c.q_ice, c.cat), cats)

"""
    _warm_ice_categories(ctx, cats, warm4)

Accumulate the warm-coupled per-category ice processes (liquid-ice collisions,
self-collection, melting, refreezing/shedding, and the residual-liquid drain)
over the categories, in index order. Every category consumes the same initial
cloud, rain, and vapor state within the step (rates are evaluated
simultaneously, as in the reference's per-category process loop); the warm-block
sinks accumulate across categories. Returns the updated warm accumulator and the
`NCAT`-tuple of per-category ice accumulators.
"""
@inline _warm_ice_categories(ctx, cats::Tuple{}, warm4) = (warm4, ())
@inline function _warm_ice_categories(ctx, cats::Tuple, warm4)
    (warm4, acc1) = _category_warm_ice_processes(ctx, first(cats), warm4)
    (warm4, rest) = _warm_ice_categories(ctx, Base.tail(cats), warm4)
    return (warm4, (acc1, rest...))
end

@inline function _category_warm_ice_processes(ctx, c, warm4)
    (; liquid, aps, tps, vel, pdf_c, pdf_r, quad, p3, ρ, T, T_freeze, L_lcl, N_lcl, L_rai, N_rai, ϵₘ) = ctx
    (; dq_lcl_dt, dn_lcl_dt, dq_rai_dt, dn_rai_dt) = warm4
    dq_ice_dt = zero(c.q_ice)
    dn_ice_dt = zero(c.q_ice)
    dq_rim_dt = zero(c.q_ice)
    db_rim_dt = zero(c.q_ice)
    dq_liq_dt = zero(c.q_ice)
    # Only compute ice processes if ice mass is present
    if c.q_ice > ϵₘ
        # --- Liquid-ice collisions
        coll = CMP3.bulk_liquid_ice_collision_sources(
            c.state, c.shape, pdf_c, pdf_r, L_lcl, N_lcl, L_rai, N_rai, aps, tps, vel, ρ, T;
            quad,
        )
        dq_lcl_dt += coll.∂ₜq_c
        dq_rai_dt += coll.∂ₜq_r
        dn_lcl_dt += coll.∂ₜN_c / ρ
        dn_rai_dt += coll.∂ₜN_r / ρ
        dq_ice_dt += coll.∂ₜL_ice / ρ
        dq_rim_dt += coll.∂ₜL_rim / ρ
        db_rim_dt += coll.∂ₜB_rim / ρ
        dq_liq_dt += _collision_liquid_source(liquid, coll) / ρ

        # --- Ice self-collection (aggregation)
        S_ice_agg = CMP3.ice_self_collection(c.state, c.shape, vel, ρ; quad)
        dn_ice_dt -= S_ice_agg.dNdt / ρ

        # Ice melting (above freezing temperature)
        (dq_rai_dt, dn_rai_dt, dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt) =
            _melt_accumulate(
                liquid, vel, aps, tps, T, T_freeze, ρ, c.state, c.shape, quad,
                dq_rai_dt, dn_rai_dt, dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt,
            )

        # Refreezing and shedding of the liquid on ice
        (dq_rai_dt, dn_rai_dt, dq_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt) =
            _refreeze_shed_accumulate(
                liquid, vel, aps, tps, T, ρ, c.state, c.shape, quad,
                dq_rai_dt, dn_rai_dt, dq_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt,
            )
    end
    # Residual liquid on an emptied core drains to rain (identically zero when
    # the core is above the presence threshold)
    (dq_rai_dt, dn_rai_dt, dq_liq_dt) = _residual_liquid_to_rain(
        liquid, p3, ρ, c.q_ice, c.cat, dq_rai_dt, dn_rai_dt, dq_liq_dt,
    )
    warm4 = (; dq_lcl_dt, dn_lcl_dt, dq_rai_dt, dn_rai_dt)
    acc = (; dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt)
    return (warm4, acc)
end

"""
    _intercategory_accumulate(icp, ctx, cats, acc)

Accumulate the inter-category collection transfers over every ordered category
pair as a source-sink double entry: the collectee loses ice mass, number, rime
mass, and rime volume ([`CMP3.inter_category_collection`](@ref)); the same
mass-like rates arrive at the collector, whose number is unchanged. No-op for a
single category (`icp === nothing`).

The reflectivity budget of both sides is carried by the per-category constant-μ
growth pass over the net rates ([`_reflectivity_tendency_slot`](@ref)): by
linearity, the collectee side realizes the documented `∂ₜz_j` sink with the
entry's frozen (mean-mass-band-clamped) coefficients, and the collector side
gains through its own Eq-10 growth from the gained mass at unchanged number, so
`Z` is not conserved pairwise (the documented behavior).
"""
@inline _intercategory_accumulate(::Nothing, ctx, cats, acc) = acc
@inline function _intercategory_accumulate(icp::CMP.InterCategoryParams, ctx, cats::NTuple{NCAT}, acc) where {NCAT}
    return _intercategory_pairs(icp, ctx, cats, acc, CMP3.ordered_category_pairs(Val(NCAT)))
end
@inline _intercategory_pairs(icp, ctx, cats, acc, ::Tuple{}) = acc
@inline function _intercategory_pairs(icp, ctx, cats, acc, pairs::Tuple)
    acc = _intercategory_pair(icp, ctx, cats, acc, first(pairs))
    return _intercategory_pairs(icp, ctx, cats, acc, Base.tail(pairs))
end
@inline function _intercategory_pair(icp, ctx, cats, acc, (i, j))
    (; vel, quad, ρ, ϵₘ) = ctx
    ci = cats[i]
    cj = cats[j]
    (ci.q_ice > ϵₘ && cj.q_ice > ϵₘ) || return acc
    r = CMP3.inter_category_collection(ci.state, ci.shape, cj.state, cj.shape, vel, ρ, icp; quad)
    aj = acc[j]
    aj = (;
        aj...,
        dq_ice_dt = aj.dq_ice_dt - r.∂ₜq_j / ρ,
        dn_ice_dt = aj.dn_ice_dt - r.∂ₜN_j / ρ,
        dq_rim_dt = aj.dq_rim_dt - r.∂ₜq_rim_j / ρ,
        db_rim_dt = aj.db_rim_dt - r.∂ₜb_rim_j / ρ,
    )
    acc = Base.setindex(acc, aj, j)
    ai = acc[i]
    ai = (;
        ai...,
        dq_ice_dt = ai.dq_ice_dt + r.∂ₜq_j / ρ,
        dq_rim_dt = ai.dq_rim_dt + r.∂ₜq_rim_j / ρ,
        db_rim_dt = ai.db_rim_dt + r.∂ₜb_rim_j / ρ,
    )
    return Base.setindex(acc, ai, i)
end

"""
    _destination_index(icp, cats, D_new)

Destination category for newly formed ice of mean-mass diameter `D_new`:
[`CMP3.icecat_destination`](@ref) over the per-category states and shapes, or 1
for a single category (`icp === nothing`).
"""
@inline _destination_index(::Nothing, cats, D_new) = 1
@inline function _destination_index(icp::CMP.InterCategoryParams, cats, D_new)
    states = map(c -> c.state, cats)
    shps = map(c -> c.shape, cats)
    return CMP3.icecat_destination(states, shps, D_new, icp)
end

# Mean-mass sphere diameter of a monodisperse frozen-drop population at ice
# density ρ_i; for a monodisperse population this equals the mass-weighted mean
# diameter required by the destination metric. `D_default` is returned when the
# number rate vanishes (the routed rate is zero).
@inline function _frozen_drop_diameter(ρ_i, ∂ₜq_frz, ∂ₜn_frz, D_default)
    D = cbrt(6 * ∂ₜq_frz / (π * ρ_i * ∂ₜn_frz))
    return ifelse(∂ₜn_frz > 0, D, oftype(D, D_default))
end

# Add the source rates `δ` (a `NamedTuple` of accumulator-field increments) to
# category `dest`.
@inline _add_source(acc::NTuple{NCAT, <:NamedTuple}, dest::Integer, δ::NamedTuple) where {NCAT} =
    Base.setindex(acc, _acc_plus(acc[dest], δ), dest)
@inline _acc_plus(a::NamedTuple, δ::NamedTuple{names}) where {names} = merge(
    a,
    NamedTuple{names}(ntuple(k -> getfield(a, names[k]) + getfield(δ, k), Val(length(names)))),
)

"""
    _vapor_numadj_categories(ctx, cats, conds, totals, acc)

Per-category vapor exchange ([`_vapor_exchange_accumulate`](@ref)) and ice
number adjustment, in index order. `conds` carries the per-category ice
condensates and `totals` the category totals for the vapor budget and the
partition shares.
"""
@inline _vapor_numadj_categories(ctx, cats::NTuple{NCAT}, conds, totals, acc) where {NCAT} =
    _vapor_numadj_rec(ctx, cats, conds, totals, acc, Val(1))
@inline function _vapor_numadj_rec(ctx, cats::NTuple{NCAT}, conds, totals, acc, ::Val{j}) where {NCAT, j}
    a = _category_vapor_numadj(ctx, cats[j], conds[j], totals, acc[j], Val(NCAT))
    acc = Base.setindex(acc, a, j)
    return j == NCAT ? acc : _vapor_numadj_rec(ctx, cats, conds, totals, acc, Val(j + 1))
end
@inline function _category_vapor_numadj(ctx, c, cond_j, totals, a, ncat::Val)
    (; liquid, moments, subdep, tps, ρ, T, q_tot, q_lcl, q_rai, ϵₘ) = ctx
    q_other = totals.q_icl_tot - cond_j
    share_core = _category_share(ncat, c.q_ice, totals.q_ice_tot, ϵₘ)
    share_liq = _liquid_category_share(liquid, ncat, c.cat, totals.q_liq_tot, ϵₘ)
    # --- Ice Sublimation / Deposition and liquid-shell condensation / evaporation
    (dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt) = _vapor_exchange_accumulate(
        liquid, subdep, tps, ρ, T, q_tot, q_lcl, q_rai, c.q_ice, c.n_ice, c.cat, c.state,
        q_other, share_core, share_liq,
        a.dq_ice_dt, a.dn_ice_dt, a.dq_rim_dt, a.db_rim_dt, a.dq_liq_dt,
    )
    # --- Ice number adjustment for mass limits
    # Nudges n_ice toward [q_ice / x_max, q_ice / x_min] over timescale τ.
    numadj = _ice_numadj_params(typeof(ϵₘ), moments)
    ∂ₜn_ice_numadj = CM2.number_tendency_from_mass_limits(numadj, c.q_ice, c.n_ice)
    dn_ice_dt += ∂ₜn_ice_numadj
    return (; dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt, dq_liq_dt)
end

# Per-category initiation rates: each source's rates land on its destination
# category, zero elsewhere.
@inline function _category_inits(
    acc::NTuple{NCAT, <:NamedTuple},
    D_nuc,
    dep,
    ∂ₜq_imm,
    ∂ₜn_imm,
    rain_frz,
    dests,
) where {NCAT}
    return ntuple(Val(NCAT)) do j
        z = acc[j].dq_ice_dt
        (;
            D_nuc,
            dq_nuc = oftype(z, ifelse(dests.nuc == j, dep.∂ₜq_frz, zero(dep.∂ₜq_frz))),
            dn_nuc = oftype(z, ifelse(dests.nuc == j, dep.∂ₜn_frz, zero(dep.∂ₜn_frz))),
            dq_cldfrz = oftype(z, ifelse(dests.imm == j, ∂ₜq_imm, zero(∂ₜq_imm))),
            dn_cldfrz = oftype(z, ifelse(dests.imm == j, ∂ₜn_imm, zero(∂ₜn_imm))),
            dq_raifrz = oftype(z, ifelse(dests.raifrz == j, rain_frz.∂ₜq_frz, zero(rain_frz.∂ₜq_frz))),
            dn_raifrz = oftype(z, ifelse(dests.raifrz == j, rain_frz.∂ₜn_frz, zero(rain_frz.∂ₜn_frz))),
        )
    end
end

@inline function _assemble_ice_blocks(moments, liquid, zc, p3, ρ, acc::NTuple{NCAT, <:NamedTuple}, inits) where {NCAT}
    blocks = ntuple(Val(NCAT)) do j
        _ice_tendency_fields(
            moments, liquid, _cat_zcoeff(zc, j), p3, ρ, acc[j], inits[j], acc[j].dq_liq_dt,
        )
    end
    return _categories_tendency_fields(blocks)
end

"""
    _bulk_2mp3_tendencies(warm, ice, dn_lcl_activation_dt)

Assemble the 2M+P3 tendency `NamedTuple` from the warm-rain block `warm`
(`dq_lcl_dt, dn_lcl_dt, dq_rai_dt, dn_rai_dt`), the ice block `ice`
(see [`_p3_ice_tendency_fields`](@ref)), and the trailing non-species
`dn_lcl_activation_dt` slot. Both 2M entry methods (warm-only and warm + P3 ice)
return through this single assembler.
"""
@inline _bulk_2mp3_tendencies(warm, ice, dn_lcl_activation_dt) =
    (; warm..., ice..., dn_lcl_activation_dt)

"""
    bulk_microphysics_tendencies(
        ::Microphysics2Moment,
        mp::Microphysics2MParams{WR, Nothing},
        ρ, T, q_tot, q_lcl, n_lcl, q_rai, n_rai,
    )

Compute 2-moment **warm rain only** microphysics tendencies (Seifert-Beheng 2006).

This method is type-stable and GPU-optimized for warm rain processes only.
For warm rain + P3 ice, see the method that accepts `Microphysics2MParams{FT, WR, <:P3IceParams}`.

# Arguments
- `mp`: Microphysics2MParams with `mp.ice == nothing` (warm rain only)
- `tps`: Thermodynamics parameters
- `ρ`: Air density (kg/m³)
- `T`: Temperature (K)
- `q_tot`: Total water specific content (kg/kg)
- `q_lcl`: Cloud liquid specific content (kg/kg)
- `n_lcl`: Cloud droplet number per kg air (1/kg)
- `q_rai`: Rain specific content (kg/kg)
- `n_rai`: Rain number per kg air (1/kg)

# Returns
`NamedTuple` with warm rain tendency fields:
- `dq_lcl_dt`: Cloud liquid tendency (kg/kg/s)
- `dn_lcl_dt`: Cloud number tendency (1/kg/s)
- `dq_rai_dt`: Rain tendency (kg/kg/s)
- `dn_rai_dt`: Rain number tendency (1/kg/s)
- `dq_ice_dt`: Ice tendency (always zero for warm-only)
- `dn_ice_dt`: Ice number tendency (always zero for warm-only)
- `dq_rim_dt`: Rime mass tendency (always zero for warm-only)
- `db_rim_dt`: Rime volume tendency (always zero for warm-only)
"""
@inline function bulk_microphysics_tendencies(  # TODO: Delete this function
    ::Microphysics2Moment, mp::CMP.Microphysics2MParams{WR, Nothing}, tps,
    ρ, T, q_tot, q_lcl, n_lcl, q_rai, n_rai,
    q_ice = zero(ρ), n_ice = zero(ρ), q_rim = zero(ρ), b_rim = zero(ρ), logλ = zero(ρ),
    inpc_log_shift = zero(ρ),
    w = zero(ρ), p = zero(ρ),
) where {WR}
    # Clamp negative inputs to zero (robustness against numerical errors)
    ρ = UT.clamp_to_nonneg(ρ)
    q_tot = UT.clamp_to_nonneg(q_tot)
    q_lcl = UT.clamp_to_nonneg(q_lcl)
    q_rai = UT.clamp_to_nonneg(q_rai)
    n_lcl = UT.clamp_to_nonneg(n_lcl)
    n_rai = UT.clamp_to_nonneg(n_rai)
    q_ice = UT.clamp_to_nonneg(q_ice)
    n_ice = UT.clamp_to_nonneg(n_ice)
    q_rim = UT.clamp_to_nonneg(q_rim)
    b_rim = UT.clamp_to_nonneg(b_rim)

    # Initialize ice-related tendencies (always zero for warm-only)
    dq_ice_dt = zero(ρ)
    dn_ice_dt = zero(ρ)
    dq_rim_dt = zero(ρ)
    db_rim_dt = zero(ρ)

    # --- Warm Rain Processes
    warm = warm_rain_tendencies_2m(mp.warm_rain, tps, T, q_tot, q_lcl, q_rai, q_ice, ρ, n_lcl, n_rai, w, p)
    dq_lcl_dt = warm.dq_lcl_dt
    dn_lcl_dt = warm.dn_lcl_dt
    dq_rai_dt = warm.dq_rai_dt
    dn_rai_dt = warm.dn_rai_dt
    dn_lcl_activation_dt = warm.dn_lcl_activation_dt

    return _bulk_2mp3_tendencies(
        (; dq_lcl_dt, dn_lcl_dt, dq_rai_dt, dn_rai_dt),
        _p3_ice_tendency_fields(dq_ice_dt, dn_ice_dt, dq_rim_dt, db_rim_dt),
        dn_lcl_activation_dt,
    )
end

"""
    _pack_2mp3_ice(mp, q_ice, n_ice, q_rim, b_rim, logλ)

Pack single-category positional ice arguments into the `(ice, shapes)` pair of
the packed 2M+P3 entry: a one-tuple of the prognostic `NamedTuple`
`(; q_ice, n_ice, q_rim, b_rim)` and a one-tuple of the [`CMP3.P3Shape`](@ref)
diagnosed from `logλ`.
"""
@inline _pack_2mp3_ice(mp, q_ice, n_ice, q_rim, b_rim, logλ) =
    _pack_2mp3_ice(_liquid(mp), mp, q_ice, n_ice, q_rim, b_rim, logλ)
@inline _pack_2mp3_ice(::CMP.NoLiquidFraction, mp, q_ice, n_ice, q_rim, b_rim, logλ) = (
    ((; q_ice, n_ice, q_rim, b_rim),),
    (CMP3.get_distribution_shape(mp.ice.scheme, logλ),),
)
@inline _pack_2mp3_ice(::CMP.PredictedLiquidFraction, mp, args...) = throw(
    ArgumentError(
        "positional ice inputs do not support predicted liquid fraction; use the packed entry with a q_liq_on_ice field",
    ),
)

"""
    _validate_packed_categories(mp, ice::NTuple{NCAT})

Throw an `ArgumentError` unless `NCAT` equals `n_categories(mp.ice)`.
"""
@inline function _validate_packed_categories(mp, ice::NTuple{NCAT}) where {NCAT}
    NCAT == CMP.n_categories(mp.ice) || throw(
        ArgumentError(
            "packed ice inputs carry $NCAT categories; `mp.ice` represents $(CMP.n_categories(mp.ice))",
        ),
    )
    return nothing
end

"""
    bulk_microphysics_tendencies(
        ::Microphysics2Moment,
        mp::Microphysics2MParams{WR, <:P3IceParams}, tps,
        ρ, T, q_tot, q_lcl, n_lcl, q_rai, n_rai,
        q_ice, n_ice, q_rim, b_rim, logλ,
    )

Single-category positional form of the packed warm rain + P3 ice entry: pack the
ice scalars and the shape diagnosed from `logλ` ([`_pack_2mp3_ice`](@ref)) and
delegate to the packed method.
"""
@inline bulk_microphysics_tendencies(
    cm::Microphysics2Moment, mp::CMP.Microphysics2MParams{WR, ICE}, tps,
    ρ, T, q_tot,
    q_lcl, n_lcl, q_rai, n_rai,
    q_ice, n_ice, q_rim, b_rim, logλ,
    inpc_log_shift = zero(ρ),
    w = zero(ρ), p = zero(ρ),
) where {WR, ICE <: CMP.P3IceParams} = bulk_microphysics_tendencies(
    cm, mp, tps, ρ, T, q_tot,
    q_lcl, n_lcl, q_rai, n_rai,
    _pack_2mp3_ice(mp, q_ice, n_ice, q_rim, b_rim, logλ)...,
    inpc_log_shift, w, p,
)

"""
    bulk_microphysics_tendencies(
        ::Microphysics2Moment,
        mp::Microphysics2MParams{WR, <:P3IceParams}, tps,
        ρ, T, q_tot, q_lcl, n_lcl, q_rai, n_rai,
        ice::NTuple{NCAT, <:NamedTuple}, shapes::NTuple{NCAT, <:P3Shape},
        inpc_log_shift = zero(ρ), w = zero(ρ), p = zero(ρ),
    )

Compute 2-moment warm rain + P3 ice microphysics tendencies from packed
per-category ice inputs.

# Arguments
- `mp`: Microphysics2MParams with P3 ice parameters present
- `tps`: Thermodynamics parameters
- `ρ`: Air density (kg/m³)
- `T`: Temperature (K)
- `q_tot`: Total water specific content (kg/kg)
- `q_lcl`: Cloud liquid specific content (kg/kg)
- `n_lcl`: Cloud droplet number per kg air (1/kg)
- `q_rai`: Rain specific content (kg/kg)
- `n_rai`: Rain number per kg air (1/kg)
- `ice`: per-category prognostic inputs, each a `NamedTuple` with fields
  `q_ice` (kg/kg), `n_ice` (1/kg), `q_rim` (kg/kg), `b_rim` (m³/kg), then,
  under [`CMP.PredictedLiquidFraction`](@ref), `q_liq_on_ice` (kg/kg), and,
  under [`CMP.ThreeMoment`](@ref) ice, `z_ice` (m⁶/kg)
- `shapes`: per-category frozen distribution shapes ([`CMP3.P3Shape`](@ref))

# Keyword Arguments
- `zcoeffs`: per-category frozen [`CMP3.ReflectivityCoefficients`](@ref) under
  [`CMP.ThreeMoment`](@ref) ice; computed from the packed inputs when not
  provided. Differentiated callers must pass coefficients precomputed from the
  primal state.

`NCAT` must equal `n_categories(mp.ice)`; up to four categories are supported.
For `NCAT > 1` the intra-category processes run per category from the same
initial cloud, rain, and vapor state (the warm-block sinks accumulate across
categories); the categories couple through the inter-category collection sweep
([`_intercategory_accumulate`](@ref)); the nucleation and freezing sources
route to the destination category ([`_destination_index`](@ref)); and the
constant-timescale vapor exchange is partitioned by the categories' mass shares
([`_category_share`](@ref)). Category merging is a separate host-callable
post-sedimentation step ([`CMP3.merge_categories`](@ref)), not part of this
tendency.

# Returns
`NamedTuple` with all tendency fields:
- `dq_lcl_dt`: Cloud liquid tendency (kg/kg/s)
- `dn_lcl_dt`: Cloud number tendency (1/kg/s)
- `dq_rai_dt`: Rain tendency (kg/kg/s)
- `dn_rai_dt`: Rain number tendency (1/kg/s)
- `dq_ice_dt`: Ice tendency (kg/kg/s)
- `dn_ice_dt`: Ice number tendency (1/kg/s)
- `dq_rim_dt`: Rime mass tendency (kg/kg/s)
- `db_rim_dt`: Rime volume tendency (m³/kg/s)
- `dq_liq_on_ice_dt`: Liquid-on-ice tendency (kg/kg/s), under
  [`CMP.PredictedLiquidFraction`](@ref) only
- `dz_ice_dt`: Reflectivity tendency (m⁶/kg/s), under [`CMP.ThreeMoment`](@ref) ice only
- `dn_lcl_activation_dt`: Droplet activation tendency (1/kg/s)

For `NCAT > 1` the per-category ice fields carry the category index before the
`_dt` suffix (`dq_ice_1_dt`, ..., `dz_ice_2_dt`); a single category keeps the
unsuffixed names.
"""
@inline function bulk_microphysics_tendencies(
    ::Microphysics2Moment, mp::CMP.Microphysics2MParams{WR, ICE}, tps,
    ρ, T, q_tot,
    q_lcl, n_lcl, q_rai, n_rai,
    ice::NTuple{NCAT, <:NamedTuple}, shapes::NTuple{NCAT, <:CMP3.P3Shape},
    inpc_log_shift = zero(ρ),
    w = zero(ρ), p = zero(ρ);
    zcoeffs = nothing,
) where {WR, ICE <: CMP.P3IceParams, NCAT}
    _validate_packed_categories(mp, ice)
    moments = _moments(mp)
    liquid = _liquid(mp)
    zc = _reflectivity_coefficients(moments, mp, ρ, ice, shapes, zcoeffs)
    FT = eltype(ρ)
    ϵₘ = UT.ϵ_numerics_2M_M(FT)
    # Clamp negative inputs to zero (robustness against numerical errors)
    ρ = UT.clamp_to_nonneg(ρ)
    q_tot = UT.clamp_to_nonneg(q_tot)
    q_lcl = UT.clamp_to_nonneg(q_lcl)
    q_rai = UT.clamp_to_nonneg(q_rai)
    n_lcl = UT.clamp_to_nonneg(n_lcl)
    n_rai = UT.clamp_to_nonneg(n_rai)

    # Convert to volumetric quantities for P3 functions
    L_lcl = q_lcl * ρ  # [kg lcl / m³ air]
    L_rai = q_rai * ρ  # [kg rai / m³ air]
    N_lcl = n_lcl * ρ  # [1 / m³ air]
    N_rai = n_rai * ρ  # [1 / m³ air]

    # Per-category clamped inputs, volumetric states, and frozen shapes
    cats = _entry_categories(mp, moments, liquid, ρ, ice, shapes)

    # Unpack warm rain parameters
    aps = mp.warm_rain.air_properties
    subdep = mp.warm_rain.subdep

    # --- P3 Ice Parameters
    p3 = mp.ice.scheme
    vel = mp.ice.terminal_velocity
    pdf_c = mp.ice.cloud_pdf
    pdf_r = mp.ice.rain_pdf
    ice_nucleation = mp.ice.ice_nucleation
    inp_depletion_model = mp.ice.inp_depletion_model
    quad = mp.ice.quad
    icp = mp.ice.inter_category
    T_freeze = TDI.TD.Parameters.T_freeze(tps)

    # --- Warm Rain Processes (the vapor and heat budgets see the total ice condensate)
    conds = _category_condensates(liquid, cats)
    q_icl_tot = reduce(+, conds)
    warm = warm_rain_tendencies_2m(mp.warm_rain, tps, T, q_tot, q_lcl, q_rai, q_icl_tot, ρ, n_lcl, n_rai, w, p)
    warm4 = (;
        dq_lcl_dt = warm.dq_lcl_dt, dn_lcl_dt = warm.dn_lcl_dt,
        dq_rai_dt = warm.dq_rai_dt, dn_rai_dt = warm.dn_rai_dt,
    )
    dn_lcl_activation_dt = warm.dn_lcl_activation_dt

    ctx = (;
        liquid, moments, aps, subdep, tps, vel, pdf_c, pdf_r, quad, p3,
        ρ, T, T_freeze, q_tot, q_lcl, q_rai, L_lcl, N_lcl, L_rai, N_rai, ϵₘ,
    )

    # --- Warm-coupled per-category ice processes (collisions, self-collection,
    # melting, refreezing/shedding, residual liquid); every category consumes
    # the same initial cloud, rain, and vapor state within the step
    (warm4, acc) = _warm_ice_categories(ctx, cats, warm4)

    # --- Inter-category collection (source-sink double entry)
    acc = _intercategory_accumulate(icp, ctx, cats, acc)

    (; dq_lcl_dt, dn_lcl_dt, dq_rai_dt, dn_rai_dt) = warm4

    # --- Ice nucleation (F23 + Bigg)
    τ_act = inp_depletion_model.τ_act
    # Vapor deposition nucleation size. TODO: put into ClimaParams.
    D_nuc = FT(10e-6)  # 10 μm nascent crystal - small-D tail of the P3
    m_nuc = p3.ρ_i * CO.volume_sphere_D(D_nuc)

    # F23 INP-activation depletion proxy over the categories' total ice number.
    n_ice_tot = reduce(+, map(_cat_n_ice, cats))
    n_active = CM_HetIce.n_active(inp_depletion_model, n_ice_tot)

    # --- deposition nucleation (vapor → pristine ice), routed to the
    # destination category of the nascent-crystal size
    dep = CM_HetIce.deposition_rate(
        ice_nucleation, tps, T, ρ, q_tot, q_lcl + q_rai, q_icl_tot, n_active;
        m_nuc, τ_act, inpc_log_shift,
    )
    dest_nuc = _destination_index(icp, cats, D_nuc)
    # No contribution to q_rim, b_rim — pristine deposition crystals have F_rim = 0.
    acc = _add_source(acc, dest_nuc, (; dn_ice_dt = dep.∂ₜn_frz, dq_ice_dt = dep.∂ₜq_frz))

    # --- F23-bounded Bigg immersion freezing of cloud drops
    cld_bigg = CM_HetIce.liquid_freezing_rate(
        mp.ice.rain_freezing, pdf_c, tps, q_lcl, ρ, N_lcl, T,
    )
    cld_cap = CM_HetIce.immersion_limit_rate(
        ice_nucleation, T, ρ; τ = τ_act, inpc_log_shift, n_active,
    )
    ∂ₜn_imm = min(cld_bigg.∂ₜn_frz, cld_cap.∂ₜn_frz)
    ∂ₜq_imm = ifelse(cld_bigg.∂ₜn_frz > 0, cld_bigg.∂ₜq_frz * ∂ₜn_imm / cld_bigg.∂ₜn_frz, zero(FT))

    # Drain liquid:
    dq_lcl_dt -= ∂ₜq_imm
    dn_lcl_dt -= ∂ₜn_imm
    # Add to ice as fully-rimed embryo graupel (F_rim = 1, solid-ice rime
    # volume), routed to the destination category of the frozen-drop size:
    dest_imm = _destination_index(icp, cats, _frozen_drop_diameter(p3.ρ_i, ∂ₜq_imm, ∂ₜn_imm, D_nuc))
    acc = _add_source(
        acc, dest_imm,
        (; dq_ice_dt = ∂ₜq_imm, dn_ice_dt = ∂ₜn_imm, dq_rim_dt = ∂ₜq_imm, db_rim_dt = ∂ₜq_imm / p3.ρ_i),
    )

    # --- Vapor exchange and ice number adjustment per category
    totals = (;
        q_icl_tot,
        q_ice_tot = reduce(+, map(_cat_q_ice, cats)),
        q_liq_tot = _liquid_condensate_total(liquid, cats, FT),
    )
    acc = _vapor_numadj_categories(ctx, cats, conds, totals, acc)

    # --- Rain Heterogeneous Freezing (Bigg 1953)
    rain_frz = CM_HetIce.liquid_freezing_rate(mp.ice.rain_freezing, pdf_r, tps, q_rai, ρ, N_rai, T)

    # Rain → ice (frozen rain is fully rimed, per MM15), routed to the
    # destination category of the frozen-drop size
    dq_rai_dt -= rain_frz.∂ₜq_frz
    dn_rai_dt -= rain_frz.∂ₜn_frz
    dest_raifrz = _destination_index(
        icp, cats, _frozen_drop_diameter(p3.ρ_i, rain_frz.∂ₜq_frz, rain_frz.∂ₜn_frz, D_nuc),
    )
    acc = _add_source(
        acc, dest_raifrz,
        (;
            dq_ice_dt = rain_frz.∂ₜq_frz, dn_ice_dt = rain_frz.∂ₜn_frz,
            dq_rim_dt = rain_frz.∂ₜq_frz,
            db_rim_dt = rain_frz.∂ₜq_frz / p3.ρ_i,  # ρ_i = 916.7 kg m⁻³, the density of solid bulk ice
        ),
    )

    # Aerosol activation is folded into `warm_rain_tendencies_2m` above —
    # `dn_lcl_activation_dt` from `warm` is already included in `dn_lcl_dt`.

    # Per-category initiation rates, in the accumulator element type.
    inits = _category_inits(
        acc, D_nuc, dep, ∂ₜq_imm, ∂ₜn_imm, rain_frz,
        (; nuc = dest_nuc, imm = dest_imm, raifrz = dest_raifrz),
    )
    return _bulk_2mp3_tendencies(
        (; dq_lcl_dt, dn_lcl_dt, dq_rai_dt, dn_rai_dt),
        _assemble_ice_blocks(moments, liquid, zc, p3, ρ, acc, inits),
        dn_lcl_activation_dt,
    )
end

include("BMT_state.jl")
include("BMT_rosenbrock.jl")

end # module BulkMicrophysicsTendencies
