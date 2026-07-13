#####
##### Layout-parameterized microphysics state vector
#####

"""
    state_length(NCAT, LIQ, ZM)

Flat length of a [`MicroState`](@ref) layout: the four-field warm block followed
by `NCAT` ice-category blocks of `4 + LIQ + ZM` fields each.
"""
@inline state_length(NCAT::Int, LIQ::Bool, ZM::Bool) = 4 + NCAT * (4 + LIQ + ZM)

"""
    MicroState{FT, NCAT, LIQ, ZM, N} <: StaticArrays.StaticVector{N, FT}

Prognostic microphysics species as a flat `StaticVector`. The layout is fixed by
the plain-bits type parameters `NCAT` (ice categories), `LIQ` (liquid-on-ice
fraction), and `ZM` (reflectivity), with `N = state_length(NCAT, LIQ, ZM)`. The
flat order is the warm block `(q_lcl, n_lcl, q_rai, n_rai)` followed by each ice
category block `(q_ice, n_ice, q_rim, b_rim[, q_liq_on_ice][, z_ice])`.

`NCAT = 1, LIQ = false, ZM = false` has the same flat indices as the default
2M+P3 state ([`MicroState2MP3`](@ref)). The warm-block index constants
([`IQ_LCL`](@ref), [`IN_LCL`](@ref), [`IQ_RAI`](@ref), [`IN_RAI`](@ref)), the
per-category accessors ([`ice_q`](@ref), [`ice_n`](@ref), [`ice_qrim`](@ref),
[`ice_brim`](@ref), [`ice_qliq`](@ref), [`ice_zice`](@ref)), and
[`cat_view`](@ref) read the individual species.
"""
struct MicroState{FT, NCAT, LIQ, ZM, N} <: SA.StaticVector{N, FT}
    data::NTuple{N, FT}
    @inline function MicroState{FT, NCAT, LIQ, ZM, N}(data::Tuple) where {FT, NCAT, LIQ, ZM, N}
        N == state_length(NCAT, LIQ, ZM) || throw(
            DimensionMismatch(
                "MicroState length $N inconsistent with layout (NCAT=$NCAT, LIQ=$LIQ, ZM=$ZM)",
            ),
        )
        return new{FT, NCAT, LIQ, ZM, N}(convert(NTuple{N, FT}, data))
    end
end

@inline MicroState{FT, NCAT, LIQ, ZM}(data::Tuple) where {FT, NCAT, LIQ, ZM} =
    MicroState{FT, NCAT, LIQ, ZM, state_length(NCAT, LIQ, ZM)}(data)

Base.@propagate_inbounds function Base.getindex(x::MicroState, i::Int)
    @boundscheck checkbounds(x, i)
    return @inbounds getfield(x, :data)[i]
end
@inline Base.Tuple(x::MicroState) = getfield(x, :data)

"""
    SA.similar_type(::Type{<:MicroState}, ::Type{T}, ::SA.Size{S})

Return the [`MicroState`](@ref) layout type for a vector shape of the same
length, enforcing the length invariant with a throw, and the StaticArrays
`SArray` for every other shape.
"""
@inline function SA.similar_type(
    ::Type{<:MicroState{FT0, NCAT, LIQ, ZM}}, ::Type{T}, ::SA.Size{S},
) where {FT0, NCAT, LIQ, ZM, T, S}
    if length(S) == 1
        n = S[1]
        n == state_length(NCAT, LIQ, ZM) || throw(
            DimensionMismatch(
                "MicroState length $n inconsistent with layout (NCAT=$NCAT, LIQ=$LIQ, ZM=$ZM)",
            ),
        )
        return MicroState{T, NCAT, LIQ, ZM, n}
    else
        return SA.SArray{Tuple{S...}, T, length(S), prod(S)}
    end
end

#####
##### Accessor layer
#####

"Flat index of the warm-block cloud liquid mass in a [`MicroState`](@ref)."
const IQ_LCL = 1
"Flat index of the warm-block cloud liquid number in a [`MicroState`](@ref)."
const IN_LCL = 2
"Flat index of the warm-block rain mass in a [`MicroState`](@ref)."
const IQ_RAI = 3
"Flat index of the warm-block rain number in a [`MicroState`](@ref)."
const IN_RAI = 4

@inline n_categories(::MicroState{FT, NCAT}) where {FT, NCAT} = NCAT
@inline n_ice_fields(::MicroState{FT, NCAT, LIQ, ZM}) where {FT, NCAT, LIQ, ZM} = 4 + LIQ + ZM
@inline _cat_offset(x::MicroState, ::Val{j}) where {j} = 4 + (j - 1) * n_ice_fields(x)

Base.@propagate_inbounds ice_q(x::MicroState, v::Val) = x[_cat_offset(x, v) + 1]
Base.@propagate_inbounds ice_n(x::MicroState, v::Val) = x[_cat_offset(x, v) + 2]
Base.@propagate_inbounds ice_qrim(x::MicroState, v::Val) = x[_cat_offset(x, v) + 3]
Base.@propagate_inbounds ice_brim(x::MicroState, v::Val) = x[_cat_offset(x, v) + 4]
Base.@propagate_inbounds ice_qliq(x::MicroState{FT, NCAT, true}, v::Val) where {FT, NCAT} =
    x[_cat_offset(x, v) + 5]
Base.@propagate_inbounds ice_zice(x::MicroState{FT, NCAT, LIQ, true}, v::Val) where {FT, NCAT, LIQ} =
    x[_cat_offset(x, v) + 4 + LIQ + 1]

"""
    ice_present_mass(x, ::Val{j})

Condensed ice mass that determines the presence of category `j` in the species
mask: the ice core mass, plus the liquid-on-ice mass when `LIQ` is set.
"""
@inline ice_present_mass(x::MicroState{FT, NCAT, false}, v::Val) where {FT, NCAT} = ice_q(x, v)
@inline ice_present_mass(x::MicroState{FT, NCAT, true}, v::Val) where {FT, NCAT} =
    ice_q(x, v) + ice_qliq(x, v)

"""
    cat_view(x, ::Val{j})

`NamedTuple` view of ice category `j`: `(q_ice, n_ice, q_rim, b_rim)`, extended
with `q_liq_on_ice` when `LIQ` is set and `z_ice` when `ZM` is set.
"""
@inline function cat_view(x::MicroState{FT, NCAT, LIQ, ZM}, v::Val{j}) where {FT, NCAT, LIQ, ZM, j}
    off = _cat_offset(x, v)
    core = (; q_ice = x[off + 1], n_ice = x[off + 2], q_rim = x[off + 3], b_rim = x[off + 4])
    withliq = LIQ ? (; core..., q_liq_on_ice = x[off + 5]) : core
    return ZM ? (; withliq..., z_ice = x[off + 4 + LIQ + 1]) : withliq
end

"""
    MicroState2MP3{FT}

The default eight-field 2M+P3 state layout: one ice category, no liquid fraction,
no reflectivity. Alias of `MicroState{FT, 1, false, false, 8}`. Internal to the
[`RosenbrockAverage`](@ref) implementation.
"""
const MicroState2MP3{FT} = MicroState{FT, 1, false, false, 8}
