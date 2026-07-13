#####
##### F0 de-risking spike: generic MicroState{FT, NCAT, LIQ, ZM, N} <: StaticVector
#####
##### Throwaway exploration for the P3 MMC2025 campaign (design synthesis D5/D12,
##### bmt-seam review items B4/S6/S9 and architecture item 6). Not for merge.
#####
##### Proves (or refutes) that a plain-bits-parameterized StaticVector with an
##### explicit accessor layer (no getproperty shim, no @generated) can replace the
##### MicroState2MP3 <: FieldVector{8} state through the Rosenbrock hot path with
##### bit-identical values, zero allocations, and clean inference.

import StaticArrays as SA
import ForwardDiff as FD
import CloudMicrophysics as CM
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.ThermodynamicsInterface as TDI

module Spike

import StaticArrays as SA
import ForwardDiff as FD
import CloudMicrophysics as CM
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.ThermodynamicsInterface as TDI

#####
##### The generic state vector
#####

"Flat length for a given layout: warm block (4) + NCAT ice blocks of (4 + LIQ + ZM)."
@inline state_length(NCAT::Int, LIQ::Bool, ZM::Bool) = 4 + NCAT * (4 + LIQ + ZM)

struct MicroState{FT, NCAT, LIQ, ZM, N} <: SA.StaticVector{N, FT}
    data::NTuple{N, FT}
    @inline function MicroState{FT, NCAT, LIQ, ZM, N}(data::Tuple) where {FT, NCAT, LIQ, ZM, N}
        N == state_length(NCAT, LIQ, ZM) || throw(DimensionMismatch(
            "MicroState length $N inconsistent with layout \
             (NCAT=$NCAT, LIQ=$LIQ, ZM=$ZM) => $(state_length(NCAT, LIQ, ZM))"))
        return new{FT, NCAT, LIQ, ZM, N}(convert(NTuple{N, FT}, data))
    end
end

# Outer convenience: derive N from the layout.
@inline MicroState{FT, NCAT, LIQ, ZM}(data::Tuple) where {FT, NCAT, LIQ, ZM} =
    MicroState{FT, NCAT, LIQ, ZM, state_length(NCAT, LIQ, ZM)}(data)

# StaticArrays plumbing.
Base.@propagate_inbounds function Base.getindex(x::MicroState, i::Int)
    @boundscheck checkbounds(x, i)
    return @inbounds getfield(x, :data)[i]
end
@inline Base.Tuple(x::MicroState) = getfield(x, :data)

# Length-preserving similar_type returns the layout-typed MicroState; the
# invariant is a hard throw (not @assert), so no off-size temporary can build a
# malformed MicroState (bmt-seam numerics S6). Non-vector sizes fall through to
# the StaticArrays SArray fallback (matrices, scalars).
@inline function SA.similar_type(
    ::Type{<:MicroState{FT0, NCAT, LIQ, ZM}}, ::Type{T}, ::SA.Size{S},
) where {FT0, NCAT, LIQ, ZM, T, S}
    if length(S) == 1
        n = S[1]
        n == state_length(NCAT, LIQ, ZM) || throw(DimensionMismatch(
            "similar_type length $n inconsistent with layout \
             (NCAT=$NCAT, LIQ=$LIQ, ZM=$ZM) => $(state_length(NCAT, LIQ, ZM))"))
        return MicroState{T, NCAT, LIQ, ZM, n}
    else
        return SA.SArray{Tuple{S...}, T, length(S), prod(S)}
    end
end

#####
##### Accessor layer: warm-block index constants + per-category accessors
#####

const IQ_LCL = 1
const IN_LCL = 2
const IQ_RAI = 3
const IN_RAI = 4

@inline n_ice_fields(::MicroState{FT, NCAT, LIQ, ZM}) where {FT, NCAT, LIQ, ZM} = 4 + LIQ + ZM
@inline n_categories(::MicroState{FT, NCAT}) where {FT, NCAT} = NCAT

# Base flat offset of ice category j (1-based); block fields follow at +1..+nif.
@inline _cat_offset(x::MicroState, ::Val{j}) where {j} = 4 + (j - 1) * n_ice_fields(x)

Base.@propagate_inbounds ice_q(x::MicroState, v::Val) = x[_cat_offset(x, v) + 1]
Base.@propagate_inbounds ice_n(x::MicroState, v::Val) = x[_cat_offset(x, v) + 2]
Base.@propagate_inbounds ice_qrim(x::MicroState, v::Val) = x[_cat_offset(x, v) + 3]
Base.@propagate_inbounds ice_brim(x::MicroState, v::Val) = x[_cat_offset(x, v) + 4]
Base.@propagate_inbounds ice_qliq(x::MicroState{FT, NCAT, true}, v::Val) where {FT, NCAT} =
    x[_cat_offset(x, v) + 5]
Base.@propagate_inbounds ice_zice(x::MicroState{FT, NCAT, LIQ, true}, v::Val) where {FT, NCAT, LIQ} =
    x[_cat_offset(x, v) + 4 + LIQ + 1]

# NamedTuple view of one category (readable tendency assembly). The LIQ/ZM
# branches are on plain-bits type parameters, so they constant-fold: the two
# arms have different NamedTuple types but the compiler picks the live one.
@inline function cat_view(x::MicroState{FT, NCAT, LIQ, ZM}, v::Val{j}) where {FT, NCAT, LIQ, ZM, j}
    off = _cat_offset(x, v)
    core = (; q_ice = x[off + 1], n_ice = x[off + 2], q_rim = x[off + 3], b_rim = x[off + 4])
    withliq = LIQ ? (; core..., q_liq_on_ice = x[off + 5]) : core
    return ZM ? (; withliq..., z_ice = x[off + 4 + LIQ + 1]) : withliq
end

# Compatibility alias for the default 2M+P3 layout (index-identical to today's
# MicroState2MP3 <: FieldVector{8}).
const MicroState2MP3{FT} = MicroState{FT, 1, false, false, 8}

# Conversions to/from the existing FieldVector state.
@inline from_fieldvector(x::BMT.MicroState2MP3{FT}) where {FT} = MicroState2MP3{FT}(Tuple(x))
@inline to_fieldvector(x::MicroState2MP3{FT}) where {FT} = BMT.MicroState2MP3{FT}(Tuple(x)...)

#####
##### Spike functor + Jacobian providers (reuse the real physics)
#####

# Reuses BMT._instantaneous_2mp3_tendency (the real per-substep raw rate,
# returned as an 8-field NamedTuple) and repacks into the layout-typed state.
struct SpikeTendency{P, H, F}
    mp::P
    tps::H
    ρ::F
    T::F
    q_tot::F
    logλ::F
end
@inline function (g::SpikeTendency)(x::MicroState2MP3{FT}) where {FT}
    (q_lcl, n_lcl, q_rai, n_rai, q_ice, n_ice, q_rim, b_rim) = Tuple(x)
    tend = BMT._instantaneous_2mp3_tendency(g.mp, g.tps,
        g.ρ, g.T, FT(g.q_tot),
        q_lcl, n_lcl, q_rai, n_rai, q_ice, n_ice, q_rim, b_rim, g.logλ,
    )
    return MicroState2MP3{FT}(values(tend))
end

# Exact (ForwardDiff) tendency + Jacobian on the plain StaticVector. This is the
# real _tendency_and_jacobian(::ExactJacobian, ...) body, but dispatched on
# StaticVector (not FieldVector) and reconstructing the layout-typed state.
@inline function spike_exact(g, x::MicroState{FT, NCAT, LIQ, ZM, N}) where {FT, NCAT, LIQ, ZM, N}
    Tag = typeof(FD.Tag(g, FT))
    DT = FD.Dual{Tag, FT, N}
    dx = MicroState{DT, NCAT, LIQ, ZM, N}(
        ntuple(i -> FD.Dual{Tag}(x[i], ntuple(s -> ifelse(s == i, one(FT), zero(FT)), Val(N))...), Val(N)),
    )
    y = g(dx)
    f = MicroState{FT, NCAT, LIQ, ZM, N}(ntuple(i -> @inbounds(FD.value(y[i])), Val(N)))
    J = SA.SMatrix{N, N, FT}(
        ntuple(k -> @inbounds(FD.partials(y[(k - 1) % N + 1], (k - 1) ÷ N + 1)), Val(N * N)),
    )
    return f, J
end

# Manual Jacobian: reuse the real _jacobian_2mp3_manual (an SMatrix{8,8} built
# from scalar fields; independent of the state vector's concrete type) by
# bridging through a real Instantaneous2MP3Tendency + FieldVector state.
@inline function spike_manual(g::SpikeTendency, x::MicroState2MP3{FT}) where {FT}
    f = g(x)
    greal = BMT.Instantaneous2MP3Tendency(g.mp, g.tps, g.ρ, g.T, g.q_tot, g.logλ)
    xreal = BMT.MicroState2MP3{FT}(Tuple(x)...)
    J = BMT._jacobian_2mp3_manual(greal, xreal)
    return f, J
end

#####
##### Species mask (accessor-based) and saturation-adjustment limiter
#####

@inline function spike_species_mask(x::MicroState2MP3{FT}) where {FT}
    ϵ = FT(1e-10)
    liq = ifelse(x[IQ_LCL] < ϵ, zero(FT), one(FT))
    rai = ifelse(x[IQ_RAI] < ϵ, zero(FT), one(FT))
    ice = ifelse(ice_q(x, Val(1)) < ϵ, zero(FT), one(FT))
    return MicroState2MP3{FT}((liq, liq, rai, rai, ice, ice, ice, ice))
end

@inline function spike_satadj(x::MicroState2MP3{FT}, d, ρ, Tsub, q_tot, Lv_over_cp, Ls_over_cp, tps) where {FT}
    Ssat(xx, TT) = max(
        TDI.supersaturation_over_ice(tps, q_tot, xx[IQ_LCL] + xx[IQ_RAI], ice_q(xx, Val(1)), ρ, TT),
        TDI.supersaturation_over_liquid(tps, q_tot, xx[IQ_LCL] + xx[IQ_RAI], ice_q(xx, Val(1)), ρ, TT),
    )
    latent(dd) = Lv_over_cp * (dd[IQ_LCL] + dd[IQ_RAI]) + Ls_over_cp * ice_q(dd, Val(1))
    return BMT._saturation_bisection(Ssat, latent, x, d, Tsub)
end

#####
##### Spike substep driver: mirrors bulk_microphysics_tendencies (lines 420-467)
#####

@inline function spike_driver(
    mode::BMT.RosenbrockAverage, mp, tps, ρ, T, q_tot,
    x0::NTuple{8, FT}, logλ, Δt, nsub;
    masked::Bool = false,
) where {FT}
    nsub_eff = max(Int(nsub), 1)
    h = Δt / FT(nsub_eff)
    cp_d = TDI.TD.Parameters.cp_d(tps)
    Lv_over_cp = TDI.TD.Parameters.LH_v0(tps) / cp_d
    Ls_over_cp = TDI.TD.Parameters.LH_s0(tps) / cp_d

    x = MicroState2MP3{FT}(x0)
    x₀ = x
    Tsub = T
    for _ in 1:nsub_eff
        g = SpikeTendency(mp, tps, ρ, Tsub, q_tot, logλ)
        x_prev = x
        if all(isfinite, x)
            f, J_raw = _spike_tj(mode.jacobian, g, x)
            J = BMT._apply_growth(mode.growth, J_raw)
            z = masked ? spike_species_mask(x) : ones(SA.SVector{8, FT})
            d = if all(isfinite, J)
                BMT._rosenbrock_update(x, f, J, z, h) - x
            else
                BMT._euler_update(x, f, h) - x
            end
            d = _spike_limit(mode.limiter, x, d, ρ, Tsub, q_tot, Lv_over_cp, Ls_over_cp, tps)
            x = max.(x .+ d, 0)
        else
            f = g(x)
            x = BMT._euler_update(x, f, h)
        end
        Δ = x - x_prev
        T_safe = max(150, Tsub)
        Tsub += (TDI.Lᵥ(tps, T_safe) * (Δ[IQ_LCL] + Δ[IQ_RAI]) + TDI.Lₛ(tps, T_safe) * ice_q(Δ, Val(1))) / cp_d
    end

    rates = (x - x₀) / Δt
    return NamedTuple{(
        :dq_lcl_dt, :dn_lcl_dt, :dq_rai_dt, :dn_rai_dt,
        :dq_ice_dt, :dn_ice_dt, :dq_rim_dt, :db_rim_dt, :dn_lcl_activation_dt,
    )}((Tuple(rates)..., zero(FT)))
end

@inline _spike_tj(::BMT.ExactJacobian, g, x) = spike_exact(g, x)
@inline _spike_tj(::BMT.ManualJacobian, g, x) = spike_manual(g, x)

@inline _spike_limit(::BMT.NoLimiter, x, d, ρ, Tsub, q_tot, Lv, Ls, tps) = d
@inline _spike_limit(::BMT.EndStateSaturationAdjustment, x, d, ρ, Tsub, q_tot, Lv, Ls, tps) =
    spike_satadj(x, d, ρ, Tsub, q_tot, Lv, Ls, tps)

end # module Spike
