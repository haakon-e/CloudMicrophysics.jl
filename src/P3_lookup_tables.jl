"""
    LinAxis{FT}
    LogAxis{FT}

A uniform interpolation axis. `LinAxis` is uniform in the coordinate; `LogAxis`
is uniform in `log10` of the coordinate. Both store the transformed lower bound
`lo`, the inverse grid spacing `inv_step = (n - 1) / (t_hi - t_lo)` in the
transformed coordinate, and the node count `n`.

Construct with the keyword form, giving the physical bounds:

    LinAxis(; lo, hi, n)
    LogAxis(; lo, hi, n)
"""
struct LinAxis{FT}
    lo::FT
    inv_step::FT
    n::Int
end

struct LogAxis{FT}
    lo::FT
    inv_step::FT
    n::Int
end

function LinAxis(; lo, hi, n)
    @assert n >= 2 "an interpolation axis needs at least two nodes"
    FT = float(promote_type(typeof(lo), typeof(hi)))
    return LinAxis{FT}(FT(lo), FT((n - 1) / (hi - lo)), Int(n))
end

function LogAxis(; lo, hi, n)
    @assert n >= 2 "an interpolation axis needs at least two nodes"
    FT = float(promote_type(typeof(lo), typeof(hi)))
    t_lo, t_hi = log10(FT(lo)), log10(FT(hi))
    return LogAxis{FT}(t_lo, FT((n - 1) / (t_hi - t_lo)), Int(n))
end

@inline naxis(ax::Union{LinAxis, LogAxis}) = ax.n

# Physical coordinate of node `i ∈ 1:n`, the inverse of the axis transform.
@inline node_coord(ax::LinAxis, i) = ax.lo + (i - 1) / ax.inv_step
@inline node_coord(ax::LogAxis, i) = exp10(ax.lo + (i - 1) / ax.inv_step)

# Fractional cell index and interpolation weight for the transformed coordinate
# `t`. The cell index is clamped to `1:n-1` and the weight to `[0, 1]`, so a
# coordinate outside the grid returns the nearest edge node (flat extrapolation).
@inline function split_index(t, n::Int)
    tc = clamp(t, zero(t), oftype(t, n - 1))
    i0 = clamp(floor(Int, tc) + 1, 1, n - 1)
    w = tc - (i0 - 1)
    return (i0, w)
end

"""
    fractional_index(ax, x) -> (i0, w)

Locate the physical coordinate `x` on the axis `ax`. Return the lower node index
`i0 ∈ 1:n-1` of the containing cell and the interpolation weight `w ∈ [0, 1]`,
with `x` reconstructed as the `w`-blend of nodes `i0` and `i0 + 1`. A coordinate
outside `[lo, hi]` clamps to the nearest edge (`w` saturates at `0` or `1`).
"""
@inline fractional_index(ax::LinAxis, x) = split_index((x - ax.lo) * ax.inv_step, ax.n)
@inline fractional_index(ax::LogAxis, x) = split_index((log10(x) - ax.lo) * ax.inv_step, ax.n)

"""
    RateTable{Names, FT, N}

Dense `N`-dimensional lookup table for the quantities named by the tuple
`Names`. The backing array has the quantity index as its first (fastest-varying)
dimension, so one multilinear corner read fetches every quantity contiguously.
Field `axes` is an `N`-tuple of [`LinAxis`](@ref)/[`LogAxis`](@ref); field `data`
has size `(length(Names), n_1, …, n_N)`.

Look up with [`lookup`](@ref).
"""
struct RateTable{Names, FT, N, AX <: Tuple, A <: AbstractArray}
    axes::AX
    data::A
end

function RateTable{Names}(axes::Tuple, data::AbstractArray{FT}) where {Names, FT}
    N = length(axes)
    @assert ndims(data) == N + 1
    @assert size(data, 1) == length(Names)
    return RateTable{Names, FT, N, typeof(axes), typeof(data)}(axes, data)
end

Adapt.adapt_structure(to, t::RateTable{Names}) where {Names} =
    RateTable{Names}(t.axes, Adapt.adapt(to, t.data))

quantity_names(::RateTable{Names}) where {Names} = Names
Base.eltype(::RateTable{Names, FT}) where {Names, FT} = FT

# Element stride (in array elements) of each table axis, with the quantity index
# occupying the first `Q` elements of every node.
@inline function node_strides(Q::Int, ns::NTuple{N, Int}) where {N}
    return ntuple(Val(N)) do d
        s = Q
        for k in 1:(d - 1)
            s *= ns[k]
        end
        s
    end
end

# Multilinear blend over the 2^N cell corners. Each leaf fetches the `Q`
# contiguous quantities of one corner as an `SVector`.
@inline function multilinear(data, ::Val{Q}, offset::Int, idx::Tuple, strides::Tuple) where {Q}
    (i0, w) = idx[1]
    s = strides[1]
    lo = multilinear(data, Val(Q), offset + (i0 - 1) * s, Base.tail(idx), Base.tail(strides))
    hi = multilinear(data, Val(Q), offset + i0 * s, Base.tail(idx), Base.tail(strides))
    return lo + (hi - lo) * w
end
@inline multilinear(data, ::Val{Q}, offset::Int, ::Tuple{}, ::Tuple{}) where {Q} =
    SA.SVector(ntuple(q -> @inbounds(data[offset + q]), Val(Q)))

"""
    lookup(table, x_1, …, x_N) -> NamedTuple

Multilinear interpolation of every quantity of `table` at the physical
coordinates `(x_1, …, x_N)`. Return a `NamedTuple` keyed by the table's quantity
names. Reproduces the node values at the `2^N` cell corners. The coordinates
should share the table element type, except for `ForwardDiff.Dual` coordinates
over that type; a coordinate of a wider type promotes the returned values.
"""
@inline function lookup(t::RateTable{Names, FT, N}, coords::Vararg{Any, N}) where {Names, FT, N}
    Q = length(Names)
    idx = map(fractional_index, t.axes, coords)
    ns = map(naxis, t.axes)
    strides = node_strides(Q, ns)
    vals = multilinear(t.data, Val(Q), 0, idx, strides)
    return NamedTuple{Names}(Tuple(vals))
end

"""
    P3LookupTables{R, S}

Container for the precomputed P3 rate tables: `rates`, a 4D [`RateTable`](@ref)
on `(log x_ice, F_rim, ρ_rim, log ρ_air)` holding `selfcol_g`, `melt_a_mom`,
`melt_b_mom`, `v_number`, and `v_mass`; and `shape`, a 3D `RateTable` on
`(log x_ice, F_rim, ρ_rim)` holding `logλ`.

Build with [`build_p3_lookup_tables`](@ref).
"""
struct P3LookupTables{R, S}
    rates::R
    shape::S
end

Adapt.adapt_structure(to, t::P3LookupTables) =
    P3LookupTables(Adapt.adapt(to, t.rates), Adapt.adapt(to, t.shape))

# selfcol and the two melt moments are stored as log; the velocities directly.
const RATE_QUANTITY_NAMES = (:log_selfcol_g, :log_melt_a_mom, :log_melt_b_mom, :v_number, :v_mass)
const SHAPE_QUANTITY_NAMES = (:logλ,)

"""
    P3TableGrid{FT}

Grid resolution and bounds for [`build_p3_lookup_tables`](@ref). The `F_rim`
upper bound and the `ρ_rim` upper bound are derived from the P3 parameters at
build time (the regularized `1 - eps(FT)` and `0.8 ρ_l`), so only the lower
`ρ_rim` bound and the `x_ice` / `ρ_air` bounds are set here.

# Fields
- `logλ_lo`: lower bound of the `logλ` rate axis [log(1/m)].
- `logλ_hi`: upper bound of the `logλ` rate axis [log(1/m)].
- `n_logλ`: number of `logλ` nodes on the rate table.
- `x_ice_lo`: lower bound of the `x_ice = L_ice / N_ice` axis of the `logλ` table [kg].
- `x_ice_hi`: upper bound of the `x_ice` axis of the `logλ` table [kg].
- `n_x_ice`: number of `x_ice` nodes on the `logλ` table.
- `n_F_rim`: number of `F_rim` nodes.
- `ρ_rim_lo`: lower bound of the `ρ_rim` axis [kg/m³].
- `n_ρ_rim`: number of `ρ_rim` nodes.
- `ρ_air_lo`: lower bound of the `ρ_air` axis [kg/m³].
- `ρ_air_hi`: upper bound of the `ρ_air` axis [kg/m³].
- `n_ρ_air`: number of `ρ_air` nodes.
- `build_order`: Gauss-Legendre order used to fill the nodes.
- `melt_bounds_tail`: tail probability setting the melt-moment integration bounds.
"""
Base.@kwdef struct P3TableGrid{FT}
    logλ_lo::FT = 2.0
    logλ_hi::FT = 17.0
    n_logλ::Int = 80
    x_ice_lo::FT = 1e-12
    x_ice_hi::FT = 2e-4
    n_x_ice::Int = 80
    n_F_rim::Int = 24
    ρ_rim_lo::FT = 100.0
    n_ρ_rim::Int = 18
    ρ_air_lo::FT = 0.05
    ρ_air_hi::FT = 1.5
    n_ρ_air::Int = 8
    build_order::Int = 12
    melt_bounds_tail::FT = 1e-6
end

"""
    build_p3_lookup_tables(params, velocity_params, aps; grid, quad)

Fill the P3 rate tables by evaluating the P3 process integrals at every grid
node with a high-order quadrature rule, and return a [`P3LookupTables`](@ref).

# Arguments
- `params`: [`CMP.ParametersP3`](@ref).
- `velocity_params`: [`CMP.Chen2022VelType`](@ref).
- `aps`: [`CMP.AirProperties`](@ref).

# Keyword Arguments
- `grid`: a [`P3TableGrid`](@ref). By default, `P3TableGrid{FT}()`.
- `quad`: the build quadrature rule. By default, `GaussLegendre(FT, grid.build_order)`.
"""
function build_p3_lookup_tables(
    params::CMP.ParametersP3, velocity_params, aps;
    grid::P3TableGrid = P3TableGrid{typeof(params.ρ_l)}(),
    quad = GaussLegendre(typeof(params.ρ_l), grid.build_order),
)
    FT = typeof(params.ρ_l)
    F_rim_hi = one(FT) - eps(FT)
    ρ_rim_hi = FT(0.8) * params.ρ_l
    ax_λ = LinAxis(; lo = grid.logλ_lo, hi = grid.logλ_hi, n = grid.n_logλ)
    ax_F = LinAxis(; lo = zero(FT), hi = F_rim_hi, n = grid.n_F_rim)
    ax_r = LinAxis(; lo = grid.ρ_rim_lo, hi = ρ_rim_hi, n = grid.n_ρ_rim)
    ax_a = LogAxis(; lo = grid.ρ_air_lo, hi = grid.ρ_air_hi, n = grid.n_ρ_air)
    ax_x = LogAxis(; lo = grid.x_ice_lo, hi = grid.x_ice_hi, n = grid.n_x_ice)

    nλ, nF, nr, na, nx = grid.n_logλ, grid.n_F_rim, grid.n_ρ_rim, grid.n_ρ_air, grid.n_x_ice
    data4 = Array{FT}(undef, length(RATE_QUANTITY_NAMES), nλ, nF, nr, na)
    data3 = Array{FT}(undef, length(SHAPE_QUANTITY_NAMES), nx, nF, nr)

    Threads.@threads for I in CartesianIndices((nλ, nF, nr))
        i, j, k = Tuple(I)
        logλ = node_coord(ax_λ, i)
        F_rim = node_coord(ax_F, j)
        ρ_rim = node_coord(ax_r, k)
        # Synthetic state with unit number and the mass that reproduces `logλ`.
        x_ice = exp(logLdivN(P3State(params, one(FT), one(FT), F_rim, ρ_rim), logλ))
        state = P3State(params, x_ice, one(FT), F_rim, ρ_rim)
        N′ = DT.size_distribution(state, logλ)
        for l in 1:na
            ρₐ = node_coord(ax_a, l)
            v_term = ice_particle_terminal_velocity(velocity_params, ρₐ, state)
            g = ice_self_collection(state, logλ, velocity_params, ρₐ; quad).dNdt
            v_number = ice_terminal_velocity_number_weighted(velocity_params, ρₐ, state, logλ; quad)
            v_mass = ice_terminal_velocity_mass_weighted(velocity_params, ρₐ, state, logλ; quad)
            bnds = velocity_integral_bounds(state, logλ, v_term; p = grid.melt_bounds_tail)
            melt_a = integrate(D -> ∂ice_mass_∂D(state, D) * N′(D) / D, bnds, quad)
            melt_b = integrate(D -> ∂ice_mass_∂D(state, D) * sqrt(D * v_term(D)) * N′(D) / D, bnds, quad)
            @inbounds begin
                data4[1, i, j, k, l] = log(g)
                data4[2, i, j, k, l] = log(melt_a)
                data4[3, i, j, k, l] = log(melt_b)
                data4[4, i, j, k, l] = v_number
                data4[5, i, j, k, l] = v_mass
            end
        end
    end

    Threads.@threads for I in CartesianIndices((nx, nF, nr))
        i, j, k = Tuple(I)
        x_ice = node_coord(ax_x, i)
        F_rim = node_coord(ax_F, j)
        ρ_rim = node_coord(ax_r, k)
        state = P3State(params, x_ice, one(FT), F_rim, ρ_rim)
        @inbounds data3[1, i, j, k] = get_distribution_logλ(state)
    end

    rates = RateTable{RATE_QUANTITY_NAMES}((ax_λ, ax_F, ax_r, ax_a), data4)
    shape = RateTable{SHAPE_QUANTITY_NAMES}((ax_x, ax_F, ax_r), data3)
    return P3LookupTables(rates, shape)
end

# Rate-table coordinates for a `state` at slope `logλ` and air density `ρₐ`.
@inline _rate_coords(state::P3State, logλ, ρₐ) = (logλ, state.F_rim, state.ρ_rim, ρₐ)

# ---------------------------------------------------------------------------- #
# Phase-2 liquid-ice collision tables
# ---------------------------------------------------------------------------- #

# Cloud and rain unpartitioned collision moments, normalized to unit ice number
# and unit liquid-number prefactor: `G_NC = NCCOL / (N_ice N_c)`,
# `G_MC = M_C / (N_ice N_c)`, `G_NR = NRCOL / (N_ice N₀r)`,
# `G_MR = M_R / (N_ice N₀r)`. Stored as `log`.
const COLLISION_CLOUD_NAMES = (:log_G_NC, :log_G_MC)
const COLLISION_RAIN_NAMES = (:log_G_NR, :log_G_MR)
# Musil ventilation moments of the unit-number ice PSD: `V_a = ∫ n̂ D dD`
# (air-density independent) and `V_b = ∫ n̂ D^{3/2} √v_i dD`. Stored as `log`.
const MUSIL_A_NAMES = (:log_V_a,)
const MUSIL_B_NAMES = (:log_V_b,)

"""
    P3CollisionTables{C, R, MA, MB}

Container for the Phase-2 liquid-ice collision tables:
- `cloud`: 5D [`RateTable`](@ref) on `(logλ, F_rim, ρ_rim, log ρ_air, log x_c)`
  holding `log_G_NC`, `log_G_MC`;
- `rain`: 5D `RateTable` on `(logλ, F_rim, ρ_rim, log ρ_air, log Dr_mean)`
  holding `log_G_NR`, `log_G_MR`;
- `musil_a`: 3D `RateTable` on `(logλ, F_rim, ρ_rim)` holding `log_V_a`;
- `musil_b`: 4D `RateTable` on `(logλ, F_rim, ρ_rim, log ρ_air)` holding `log_V_b`.

Build with [`build_p3_collision_tables`](@ref); assemble the bulk sources with the
table method of [`bulk_liquid_ice_collision_sources`](@ref).
"""
struct P3CollisionTables{C, R, MA, MB}
    cloud::C
    rain::R
    musil_a::MA
    musil_b::MB
end

Adapt.adapt_structure(to, t::P3CollisionTables) = P3CollisionTables(
    Adapt.adapt(to, t.cloud), Adapt.adapt(to, t.rain),
    Adapt.adapt(to, t.musil_a), Adapt.adapt(to, t.musil_b),
)

"""
    P3CollisionGrid{FT}

Grid resolution and bounds for [`build_p3_collision_tables`](@ref). The `F_rim`
upper bound (`1 - eps(FT)`) and the `ρ_rim` upper bound (`0.8 ρ_l`) are derived
from the P3 parameters at build time, matching [`P3TableGrid`](@ref). The shared
shape axes `(logλ, F_rim, ρ_rim, ρ_air)` reuse the [`P3TableGrid`](@ref) ranges;
the two 5D tables add a `log x_c` (cloud mean mass) axis and a `log Dr_mean`
(rain mean diameter) axis. The `Dr_mean` range is the interval into which the
limited rain PDF slope is clamped, so no realizable rain state leaves it.
"""
Base.@kwdef struct P3CollisionGrid{FT}
    logλ_lo::FT = 2.0
    logλ_hi::FT = 17.0
    n_logλ::Int = 40
    n_F_rim::Int = 12
    ρ_rim_lo::FT = 100.0
    n_ρ_rim::Int = 10
    ρ_air_lo::FT = 0.05
    ρ_air_hi::FT = 1.5
    n_ρ_air::Int = 6
    x_c_lo::FT = 1e-13
    x_c_hi::FT = 1e-10
    n_x_c::Int = 16
    Dr_lo::FT = 1e-4
    Dr_hi::FT = 1e-3
    n_Dr::Int = 16
    build_order::Int = 16
    bounds_tail::FT = 1e-5
end

# Cloud inner collision moments `(∂ₜN_c_col, ∂ₜM_c_col)` per unit cloud number,
# normalized so `NCCOL = N_ice N_c G_NC`. The ice PSD `n_i` is unit-number and the
# cloud PSD is built at unit number with `L_c = x_c`.
@inline function _cloud_unit_moments(state, logλ, n_i, ∂ₜV, ice_bounds, psd_c, ρₐ, x_c, m_liq, p; quad)
    FT = eltype(state)
    q_c = x_c / ρₐ
    n_c = DT.size_distribution(psd_c, q_c, ρₐ, one(FT))
    bounds_c = CM2.get_size_distribution_bounds(psd_c, q_c, ρₐ, one(FT), p)
    ρ′_unit = (_Dᵢ, _Dₗ) -> one(FT)
    cloud = get_liquid_integrals(n_c, ∂ₜV, m_liq, ρ′_unit, bounds_c; quad)
    G_NC = integrate(Dᵢ -> n_i(Dᵢ) * cloud(Dᵢ)[1], ice_bounds, quad)
    G_MC = integrate(Dᵢ -> n_i(Dᵢ) * cloud(Dᵢ)[2], ice_bounds, quad)
    return (G_NC, G_MC)
end

# Rain inner collision moments `(∂ₜN_r_col, ∂ₜM_r_col)` per unit `N₀r`, normalized
# so `NRCOL = N_ice N₀r G_NR`. The rain slope is `λr = 1 / Dr_mean`.
@inline function _rain_unit_moments(state, n_i, ∂ₜV, ice_bounds, psd_r, Dr_mean, p; quad)
    FT = eltype(state)
    (; v_i, v_l) = ∂ₜV
    ρw = psd_r.ρw
    ai, bi, ci = SA.SVector(v_l.ai), SA.SVector(v_l.bi), SA.SVector(v_l.ci)
    D_min = DT.exponential_quantile(Dr_mean, p)
    D_max = DT.exponential_quantile(Dr_mean, one(FT) - p)
    function inner(Dᵢ)
        v = v_i(Dᵢ)
        rᵢ = sqrt(ice_area(state, Dᵢ) / FT(π))
        Dstar = crossover_diameter(v, v_l, D_min, D_max)
        return closed_rain_inner_NM(v, Dstar, rᵢ, ρw, ai, bi, ci, D_min, D_max, one(FT), Dr_mean)
    end
    G_NR = integrate(Dᵢ -> n_i(Dᵢ) * inner(Dᵢ)[1], ice_bounds, quad)
    G_MR = integrate(Dᵢ -> n_i(Dᵢ) * inner(Dᵢ)[2], ice_bounds, quad)
    return (G_NR, G_MR)
end

# Musil ventilation moments of the unit-number ice PSD over `ice_bounds`.
@inline function _musil_unit_moments(n_i, v_term, ice_bounds; quad)
    V_a = integrate(D -> n_i(D) * D, ice_bounds, quad)
    V_b = integrate(D -> n_i(D) * D^(3 // 2) * sqrt(v_term(D)), ice_bounds, quad)
    return (V_a, V_b)
end

"""
    build_p3_collision_tables(params, velocity_params, aps, psd_c, psd_r; grid, quad)

Fill the Phase-2 liquid-ice collision tables by evaluating the unpartitioned
collision integrals and the Musil ventilation moments at every grid node, and
return a [`P3CollisionTables`](@ref). The integrals are built without the
wet-growth partition (temperature-free and bilinear in the liquid-number
prefactors); the temperature-dependent partition is applied at the use site by
the table method of [`bulk_liquid_ice_collision_sources`](@ref).

# Arguments
- `params`: [`CMP.ParametersP3`](@ref).
- `velocity_params`: [`CMP.Chen2022VelType`](@ref).
- `aps`: [`CMP.AirProperties`](@ref).
- `psd_c`: [`CMP.CloudParticlePDF_SB2006`](@ref).
- `psd_r`: [`CMP.RainParticlePDF_SB2006`](@ref).

# Keyword Arguments
- `grid`: a [`P3CollisionGrid`](@ref). By default, `P3CollisionGrid{FT}()`.
- `quad`: the build quadrature rule. By default, `GaussLegendre(FT, grid.build_order)`.
"""
function build_p3_collision_tables(
    params::CMP.ParametersP3, velocity_params, aps, psd_c, psd_r;
    grid::P3CollisionGrid = P3CollisionGrid{typeof(params.ρ_l)}(),
    quad = GaussLegendre(typeof(params.ρ_l), grid.build_order),
)
    FT = typeof(params.ρ_l)
    p = grid.bounds_tail
    F_rim_hi = one(FT) - eps(FT)
    ρ_rim_hi = FT(0.8) * params.ρ_l
    ρw = psd_c.ρw
    @assert ρw == psd_r.ρw "cloud and rain must share the liquid water density"
    m_liq(Dₗ) = ρw * CO.volume_sphere_D(Dₗ)

    ax_λ = LinAxis(; lo = grid.logλ_lo, hi = grid.logλ_hi, n = grid.n_logλ)
    ax_F = LinAxis(; lo = zero(FT), hi = F_rim_hi, n = grid.n_F_rim)
    ax_r = LinAxis(; lo = grid.ρ_rim_lo, hi = ρ_rim_hi, n = grid.n_ρ_rim)
    ax_a = LogAxis(; lo = grid.ρ_air_lo, hi = grid.ρ_air_hi, n = grid.n_ρ_air)
    ax_xc = LogAxis(; lo = grid.x_c_lo, hi = grid.x_c_hi, n = grid.n_x_c)
    ax_Dr = LogAxis(; lo = grid.Dr_lo, hi = grid.Dr_hi, n = grid.n_Dr)

    nλ, nF, nr, na = grid.n_logλ, grid.n_F_rim, grid.n_ρ_rim, grid.n_ρ_air
    nxc, nDr = grid.n_x_c, grid.n_Dr
    data_c = Array{FT}(undef, length(COLLISION_CLOUD_NAMES), nλ, nF, nr, na, nxc)
    data_R = Array{FT}(undef, length(COLLISION_RAIN_NAMES), nλ, nF, nr, na, nDr)
    data_ma = Array{FT}(undef, length(MUSIL_A_NAMES), nλ, nF, nr)
    data_mb = Array{FT}(undef, length(MUSIL_B_NAMES), nλ, nF, nr, na)

    Threads.@threads for I in CartesianIndices((nλ, nF, nr))
        i, j, k = Tuple(I)
        logλ = node_coord(ax_λ, i)
        F_rim = node_coord(ax_F, j)
        ρ_rim = node_coord(ax_r, k)
        x_ice = exp(logLdivN(P3State(params, one(FT), one(FT), F_rim, ρ_rim), logλ))
        state = P3State(params, x_ice, one(FT), F_rim, ρ_rim)
        n_i = DT.size_distribution(state, logλ)
        for l in 1:na
            ρₐ = node_coord(ax_a, l)
            ∂ₜV = volumetric_collision_rate_integrand(velocity_params, ρₐ, state)
            ice_bounds = velocity_integral_bounds(state, logλ, ∂ₜV.v_i; p)
            V_a, V_b = _musil_unit_moments(n_i, ∂ₜV.v_i, ice_bounds; quad)
            # `V_a` and `ice_bounds` are air-density independent; record `V_a` once
            l == 1 && (@inbounds data_ma[1, i, j, k] = log(V_a))
            @inbounds data_mb[1, i, j, k, l] = log(V_b)
            for m in 1:nxc
                x_c = node_coord(ax_xc, m)
                G_NC, G_MC = _cloud_unit_moments(state, logλ, n_i, ∂ₜV, ice_bounds, psd_c, ρₐ, x_c, m_liq, p; quad)
                @inbounds data_c[1, i, j, k, l, m] = log(G_NC)
                @inbounds data_c[2, i, j, k, l, m] = log(G_MC)
            end
            for m in 1:nDr
                Dr = node_coord(ax_Dr, m)
                G_NR, G_MR = _rain_unit_moments(state, n_i, ∂ₜV, ice_bounds, psd_r, Dr, p; quad)
                @inbounds data_R[1, i, j, k, l, m] = log(G_NR)
                @inbounds data_R[2, i, j, k, l, m] = log(G_MR)
            end
        end
    end

    cloud = RateTable{COLLISION_CLOUD_NAMES}((ax_λ, ax_F, ax_r, ax_a, ax_xc), data_c)
    rain = RateTable{COLLISION_RAIN_NAMES}((ax_λ, ax_F, ax_r, ax_a, ax_Dr), data_R)
    musil_a = RateTable{MUSIL_A_NAMES}((ax_λ, ax_F, ax_r), data_ma)
    musil_b = RateTable{MUSIL_B_NAMES}((ax_λ, ax_F, ax_r, ax_a), data_mb)
    return P3CollisionTables(cloud, rain, musil_a, musil_b)
end

"""
    ice_self_collection(tables::P3LookupTables, state, logλ, ρₐ)

Table-backed ice self-collection rate; see the quadrature method
[`ice_self_collection`](@ref).
"""
@inline function ice_self_collection(tables::P3LookupTables, state::P3State, logλ, ρₐ)
    q = lookup(tables.rates, _rate_coords(state, logλ, ρₐ)...)
    return (; dNdt = state.ρn_ice^2 * exp(q.log_selfcol_g))
end

"""
    ice_terminal_velocity_number_weighted(tables::P3LookupTables, state, logλ, ρₐ)

Table-backed number-weighted mean ice terminal velocity; see the quadrature
method [`ice_terminal_velocity_number_weighted`](@ref).
"""
@inline function ice_terminal_velocity_number_weighted(tables::P3LookupTables, state::P3State, logλ, ρₐ)
    FT = eltype(state)
    q = lookup(tables.rates, _rate_coords(state, logλ, ρₐ)...)
    below_ϵ = (state.ρn_ice < UT.ϵ_numerics_2M_N(FT)) | (state.ρq_ice < UT.ϵ_numerics_2M_M(FT))
    return ifelse(below_ϵ, zero(q.v_number), q.v_number)
end

"""
    ice_terminal_velocity_mass_weighted(tables::P3LookupTables, state, logλ, ρₐ)

Table-backed mass-weighted mean ice terminal velocity; see the quadrature method
[`ice_terminal_velocity_mass_weighted`](@ref).
"""
@inline function ice_terminal_velocity_mass_weighted(tables::P3LookupTables, state::P3State, logλ, ρₐ)
    FT = eltype(state)
    q = lookup(tables.rates, _rate_coords(state, logλ, ρₐ)...)
    below_ϵ = (state.ρn_ice < UT.ϵ_numerics_2M_N(FT)) | (state.ρq_ice < UT.ϵ_numerics_2M_M(FT))
    return ifelse(below_ϵ, zero(q.v_mass), q.v_mass)
end

"""
    ice_melt(tables::P3LookupTables, aps, tps, Tₐ, ρₐ, state, logλ)

Table-backed ice melting rate; see the quadrature method [`ice_melt`](@ref). The
tabulated ventilation moments are combined with the closed-form ventilation
factors `(aᵥ, bᵥ, Sc)` and the closed-form thermal prefactor
`4 K_therm (Tₐ - T_freeze) / L_f`.
"""
@inline function ice_melt(
    tables::P3LookupTables, aps::CMP.AirProperties, tps::TDI.PS, Tₐ, ρₐ, state::P3State, logλ,
)
    (; K_therm, ν_air, D_vapor) = aps
    L_f = TDI.Lf(tps, Tₐ)
    (; T_freeze, vent) = state.params
    (; aᵥ, bᵥ) = vent
    cbrt_Sc = cbrt(ν_air / D_vapor)

    q = lookup(tables.rates, _rate_coords(state, logλ, ρₐ)...)
    fac = 4 * K_therm / L_f * (Tₐ - T_freeze)
    integral = aᵥ * exp(q.log_melt_a_mom) + bᵥ * cbrt_Sc / sqrt(ν_air) * exp(q.log_melt_b_mom)
    dLdt = max(zero(fac), fac * state.ρn_ice * integral)
    dNdt = state.ρn_ice / state.ρq_ice * dLdt
    return (; dNdt, dLdt)
end

"""
    get_distribution_logλ(tables::P3LookupTables, state)

Table-backed slope parameter `logλ`; see the iterative solver
[`get_distribution_logλ`](@ref). A degenerate ice state returns `log(0)`.
"""
@inline function get_distribution_logλ(tables::P3LookupTables, state::P3State)
    FT = eltype(state)
    (; ρn_ice, ρq_ice) = state
    (ρn_ice < UT.ϵ_numerics_2M_N(FT) || ρq_ice < UT.ϵ_numerics_2M_M(FT)) &&
        return log(zero(ρq_ice))
    q = lookup(tables.shape, ρq_ice / ρn_ice, state.F_rim, state.ρ_rim)
    return q.logλ
end
