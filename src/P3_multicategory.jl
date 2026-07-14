#####
##### Multiple-ice-category physics: inter-category collection, destination
##### selection for new ice, and category merging.
#####
##### See the documentation page "P3 multiple ice categories" for the kernel,
##### transfer factors, and destination/merge algorithms.
#####

"""
    ordered_category_pairs(::Val{N})

Ordered `(collector, collectee)` index pairs for the `N`-category inter-category
collection sweep: every ordered pair of distinct categories, `N (N-1)` pairs
total. Empty for `N = 1`. Each pair `(i, j)` is passed to
[`inter_category_collection`](@ref) as `(state_i, …, state_j, …)`, i.e. category
`i` collects category `j`.
"""
@inline function ordered_category_pairs(::Val{N}) where {N}
    return ntuple(Val(N * (N - 1))) do k
        i = (k - 1) ÷ (N - 1) + 1
        r = (k - 1) % (N - 1)
        j = r < i - 1 ? r + 1 : r + 2
        (i, j)
    end
end

"""
    rime_collection_shutoff(icp::InterCategoryParams, F_rim)

Multiplicative factor in `[0, 1]` that shuts off inter-category collection for a
heavily rimed collector, as a function of the collector rime mass fraction
`F_rim`. Unity below `icp.F_rim_shutoff_start`, zero at and above
`icp.F_rim_shutoff_end`, and a `C¹` Hermite ramp (`1 - t²(3 - 2t)`) between.

The `C¹` ramp replaces the Fortran linear ramp `1 - (F_rim - 0.6)/0.3` between
the same `0.6`/`0.9` bounds, matching its endpoint values while removing the
slope discontinuities at the bounds.
"""
@inline function rime_collection_shutoff(icp::CMP.InterCategoryParams, F_rim)
    lo = icp.F_rim_shutoff_start
    hi = icp.F_rim_shutoff_end
    t = clamp((F_rim - lo) / (hi - lo), zero(F_rim), one(F_rim))
    return 1 - t * t * (3 - 2t)
end

"""
    inter_category_collection(state_i, shape_i, state_j, shape_j, vel, ρₐ, icp; quad)

Bulk gravitational-collection transfer for category `i` (collector) collecting
category `j` (collectee), evaluated by two-dimensional quadrature.

The kernel is `E · σ(Dᵢ, Dⱼ) · (vᵢ(Dᵢ) − vⱼ(Dⱼ))₊`, with the ice-ice collision
cross-section `σ` from [`collision_cross_section_ice_ice`](@ref), the one-sided
differential fall speed selecting collector-faster-than-collectee pairs, and the
efficiency `E = icp.E_ii · rime_collection_shutoff(icp, state_i.F_rim)` using the
collector rime fraction. The outer integral is over the collector diameter `Dᵢ`
and the inner over the collectee diameter `Dⱼ`; the fall-speed crossover
`vⱼ(D⋆) = vᵢ(Dᵢ)` is inserted as an inner subinterval boundary so the `(·)₊` kink
sits on a node (see [`crossover_diameter`](@ref)).

# Arguments
- `state_i`, `shape_i`: collector [`P3State`](@ref) and diagnosed [`P3Shape`](@ref)
- `state_j`, `shape_j`: collectee [`P3State`](@ref) and diagnosed [`P3Shape`](@ref)
- `vel`: velocity parameterization, e.g. [`CMP.Chen2022VelType`](@ref)
- `ρₐ`: air density [kg/m³]
- `icp`: [`CMP.InterCategoryParams`](@ref)

# Keyword arguments
- `quad`: quadrature rule (a `Quadrature.QuadratureRule`)

# Returns
A `NamedTuple` of the directed collectee-side rates. The mass-like rates are
also the collector-side gains: whatever ice mass, rime mass, and rime volume
leave the collectee arrive at the collector (the caller materializes the
source-sink pair); the collectee loses number while the collector number is
unchanged:
- `∂ₜN_j`: collectee number loss rate [1/m³/s]
- `∂ₜq_j`: ice-core mass transfer rate [kg/m³/s]
- `∂ₜq_rim_j`: rime mass transfer rate [kg/m³/s], `state_j.F_rim · ∂ₜq_j`
- `∂ₜb_rim_j`: rime volume transfer rate [m³/m³/s], `state_j.F_rim/state_j.ρ_rim · ∂ₜq_j`
- `∂ₜz_j`: collectee reflectivity loss rate [m⁶/m³/s] under the constant-μ
  closure `Z = c(μ) L²/N`, i.e. `∂ₜz_j = ρz_j (2 ∂ₜq_j/L_j − ∂ₜN_j/N_j)`; zero
  under two-moment ice. Reflectivity is not conserved pairwise: the collector
  reflectivity is recomputed from its updated mass and number by the
  three-moment shape solve.
"""
@inline function inter_category_collection(
    state_i::P3State, shape_i::P3Shape, state_j::P3State, shape_j::P3Shape,
    vel, ρₐ, icp::CMP.InterCategoryParams; quad,
)
    FT = eltype(state_i)

    n_i = DT.size_distribution(state_i, shape_i)
    n_j = DT.size_distribution(state_j, shape_j)
    v_i = ice_particle_terminal_velocity(vel, ρₐ, state_i)
    v_j = ice_particle_terminal_velocity(vel, ρₐ, state_j)

    p = FT(1e-6)
    bounds_i = velocity_integral_bounds(state_i, shape_i, v_i; p)
    bounds_j = velocity_integral_bounds(state_j, shape_j, v_j; p)
    Dj_min, Dj_max = first(bounds_j), last(bounds_j)

    function outer(D_i)
        v_i_Di = v_i(D_i)
        # collection of j by i occurs where the collector falls faster; the
        # crossover vⱼ(D⋆) = vᵢ(Dᵢ) is the kink of the (·)₊ integrand. The
        # outer integrand stays C¹ where the crossover enters or leaves the
        # inner domain (the (·)₊ integrand vanishes at D⋆), so only the
        # collector regime and velocity breakpoints bound the outer integral.
        Dstar = crossover_diameter(v_i_Di, v_j, Dj_min, Dj_max)
        inner_bounds = Tuple(SA.sort(SA.SVector(bounds_j..., clamp(Dstar, Dj_min, Dj_max))))
        integrand = D_j -> begin
            Δv = max(v_i_Di - v_j(D_j), zero(FT))
            base = collision_cross_section_ice_ice(state_i, D_i, state_j, D_j) * Δv * n_j(D_j)
            return SA.SVector(base, base * ice_mass(state_j, D_j))  # (number, mass)
        end
        return n_i(D_i) * integrate(integrand, inner_bounds, quad)
    end

    (∫N, ∫q) = integrate(outer, bounds_i, quad)
    E = icp.E_ii * rime_collection_shutoff(icp, state_i.F_rim)
    ∂ₜN_j = E * ∫N
    ∂ₜq_j = E * ∫q

    F_rim_j = state_j.F_rim
    ρ_rim_j = state_j.ρ_rim
    ∂ₜq_rim_j = F_rim_j * ∂ₜq_j
    ∂ₜb_rim_j = iszero(ρ_rim_j) ? zero(FT) : (F_rim_j / ρ_rim_j) * ∂ₜq_j

    # Constant-μ reflectivity sink: with μ frozen, Z = c(μ) L²/N, so
    # dZ = Z (2 dL/L − dN/N), evaluated on the collection mass and number rates.
    L_j = state_j.ρq_ice
    N_j = state_j.ρn_ice
    ϵ_pres = FT(UT.SPECIES_PRESENCE_THRESHOLD)
    ∂ₜz_j = (L_j > ϵ_pres) & (N_j > ϵ_pres) ?
            state_j.ρz_ice * (2 * ∂ₜq_j / L_j - ∂ₜN_j / N_j) : zero(FT)

    return (; ∂ₜN_j, ∂ₜq_j, ∂ₜq_rim_j, ∂ₜb_rim_j, ∂ₜz_j)
end

# Core-PSD view of a frozen shape: the shared μ with the ice-core slope. The
# destination and merge similarity metrics are evaluated on the frozen core
# (`logλ_core == logλ` when liquid is off).
@inline _core_shape(shape::P3Shape) =
    P3Shape(; logλ = shape.logλ_core, μ = shape.μ, logλ_core = shape.logλ_core)

"""
    icecat_destination(states, shapes, D_new, icp)

Return the index of the ice category that receives newly formed ice of
mean-mass diameter `D_new`. New ice goes to the populated category whose
mean-mass diameter [`D_m`](@ref) is closest to `D_new`, unless that closest
difference exceeds `icp.ΔD_init` and an empty category is available, in which
case the first empty category is opened
([Milbrandt and Morrison (2016)](@cite MilbrandtMorrison2016), section 2b(1)):

- all categories empty → category 1;
- all categories populated → smallest mean-diameter difference;
- otherwise → smallest difference if it is below `icp.ΔD_init`, else the first
  empty category.

A category is populated when `state.ρq_ice` exceeds the presence threshold
[`UT.SPECIES_PRESENCE_THRESHOLD`](@ref CloudMicrophysics.Utilities.SPECIES_PRESENCE_THRESHOLD).
`D_new` must use the same mean-diameter definition as [`D_m`](@ref) (the
mass-weighted mean diameter); the reference Fortran computes its `D_new` as the
equivalent mean-mass sphere diameter, a different moment of the distribution,
and mixing the two definitions degrades the selection metric. The per-category
mean diameters are evaluated on the frozen core (the shared μ with the ice-core
slope), so the liquid on ice does not enter the selection metric. Pure over the
`N`-tuples of states and shapes; `N = 1` returns 1.
"""
@inline function icecat_destination(
    states::NTuple{N, P3State}, shapes::NTuple{N, P3Shape}, D_new, icp::CMP.InterCategoryParams,
) where {N}
    FT = eltype(first(states))
    ΔD_init = FT(icp.ΔD_init)
    ϵ_pres = FT(UT.SPECIES_PRESENCE_THRESHOLD)
    i_mindiff = 1
    mindiff = FT(Inf)
    i_first_empty = 0
    n_present = 0
    for k in 1:N
        if states[k].ρq_ice > ϵ_pres
            n_present += 1
            diff = abs(D_m(states[k], _core_shape(shapes[k])) - D_new)
            if diff < mindiff
                mindiff = diff
                i_mindiff = k
            end
        elseif i_first_empty == 0
            i_first_empty = k
        end
    end
    n_present == 0 && return 1              # all empty
    n_present == N && return i_mindiff      # all populated
    return mindiff < ΔD_init ? i_mindiff : i_first_empty
end

"""
    mean_ice_density(state, shape)

Mean bulk density of an ice category [kg/m³]: the ice mass concentration divided
by the equivalent-sphere volume concentration `π/6 · M₃`, with `M₃` the third
moment of the number size distribution. Used as the density criterion in
[`merge_categories`](@ref).
"""
@inline function mean_ice_density(state::P3State, shape::P3Shape)
    FT = eltype(state)
    M₃ = exp(get_logN₀(state.ρn_ice, shape.μ, shape.logλ) + loggamma_moment(shape.μ, shape.logλ; k = 3))
    volume = FT(π) / 6 * M₃
    return state.ρq_ice / max(volume, floatmin(FT))
end

@inline function _category_prognostic(state::P3State)
    ρq_rim = state.F_rim * state.ρq_ice
    ρb_rim = iszero(state.ρ_rim) ? zero(ρq_rim) : ρq_rim / state.ρ_rim
    # from: F_liq = ρq_liq / (ρq_ice + ρq_liq); zero when liquid is off
    ρq_liq_on_ice = state.ρq_ice * (state.F_liq / (1 - state.F_liq))
    return SA.SVector(state.ρq_ice, state.ρn_ice, ρq_rim, ρb_rim, ρq_liq_on_ice, state.ρz_ice)
end
@inline _prognostic_namedtuple(v) = (;
    ρq_ice = v[1], ρn_ice = v[2], ρq_rim = v[3], ρb_rim = v[4],
    ρq_liq_on_ice = v[5], ρz_ice = v[6],
)

"""
    merge_categories(states, shapes, icp)

Post-sedimentation merge sweep: collapse adjacent ice categories that have
converged in mean size and bulk density
([Milbrandt and Morrison (2016)](@cite MilbrandtMorrison2016), section 2b(3)).
Two adjacent populated categories are merged when both the mean-mass-diameter
difference is below `icp.ΔD_merge` and the [`mean_ice_density`](@ref) difference
is below `icp.Δρ_merge`; both metrics are evaluated on the frozen core (the
shared μ with the ice-core slope), consistent with the destination metric, so
the liquid on ice does not enter the criterion. Merging sums the extensive
prognostic quantities (ice mass, number, rime mass, rime volume, liquid mass on
ice, and reflectivity) into the lower-index category and zeros the higher-index
one: the liquid rides along with its category. Reflectivity sums additively on
merge ([Milbrandt et al. (2021)](@cite Milbrandt2021)).

Pure over the `N`-tuples of states and shapes. Returns an `N`-tuple of
`NamedTuple`s `(; ρq_ice, ρn_ice, ρq_rim, ρb_rim, ρq_liq_on_ice, ρz_ice)`, the
merged prognostic quantities per category, with `ρq_liq_on_ice = 0` when the
liquid treatment is off and `ρz_ice = 0` under two-moment ice.
A category participates in merging when `state.ρq_ice` exceeds the presence
threshold
[`UT.SPECIES_PRESENCE_THRESHOLD`](@ref CloudMicrophysics.Utilities.SPECIES_PRESENCE_THRESHOLD).

The merge criterion uses the two-condition paper form; the Fortran reference
diverges here, reusing its `ΔD_init` and testing only the diameter condition.
"""
@inline function merge_categories(
    states::NTuple{N, P3State}, shapes::NTuple{N, P3Shape}, icp::CMP.InterCategoryParams,
) where {N}
    prog = map(_category_prognostic, states)
    N == 1 && return map(_prognostic_namedtuple, prog)

    FT = eltype(first(states))
    ΔD_merge = FT(icp.ΔD_merge)
    Δρ_merge = FT(icp.Δρ_merge)
    ϵ_pres = FT(UT.SPECIES_PRESENCE_THRESHOLD)
    present = map(s -> s.ρq_ice > ϵ_pres, states)
    # criterion metrics only for present categories; absent ones take inert zeros
    D = ntuple(k -> present[k] ? D_m(states[k], _core_shape(shapes[k])) : zero(FT), Val(N))
    ρ = ntuple(k -> present[k] ? mean_ice_density(states[k], _core_shape(shapes[k])) : zero(FT), Val(N))

    # merge_pair[k] (k ≥ 2): categories k and k-1 satisfy the merge criterion
    merge_pair = ntuple(Val(N)) do k
        k == 1 ? false :
        (present[k] && present[k - 1] &&
         abs(D[k] - D[k - 1]) < ΔD_merge && abs(ρ[k] - ρ[k - 1]) < Δρ_merge)
    end
    # root[k]: lowest index reachable from k by consecutive downward merges
    root = ntuple(Val(N)) do k
        r = k
        @inbounds while r ≥ 2 && merge_pair[r]
            r -= 1
        end
        r
    end
    # each category accumulates the prognostics of every category collapsing into it
    z = zero(SA.SVector{6, FT})
    return ntuple(Val(N)) do m
        acc = z
        for k in 1:N
            acc += ifelse(root[k] == m, prog[k], z)
        end
        _prognostic_namedtuple(acc)
    end
end
