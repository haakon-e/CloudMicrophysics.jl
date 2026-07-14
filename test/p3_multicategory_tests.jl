using Test: @testset, @test, @test_throws, @inferred
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.ThermodynamicsInterface as TDI
import CloudMicrophysics.DistributionTools as DT
import ClimaParams as CP
import BenchmarkTools as BT
import JET
import QuadGK as QGK
import StaticArrays as SA

# Copy a P3State with an injected sixth-moment (reflectivity) content, so the
# constant-μ inter-category Z hooks can be exercised without the three-moment
# core (which lands on a separate branch).
function _with_z(state::P3.P3State, ρz)
    return P3.P3State(
        state.params, state.ρq_ice, state.ρn_ice, state.F_rim, state.ρ_rim,
        state.F_liq, oftype(state.ρz_ice, ρz),
        state.ρ_g, state.D_th, state.D_gr, state.D_cr,
    )
end

# Independent nested-QuadGK reference for the directed inter-category collection
# (Float64), matching the kernel and one-sided fall-speed selection. The P3
# regime and velocity breakpoints are passed as QuadGK segment boundaries (outer
# over the collector, inner over the collectee with the crossover), so each
# subinterval is smooth and the adaptive rule converges cheaply.
function _intercat_reference(state_i, shape_i, state_j, shape_j, vel, ρₐ, icp)
    n_i = DT.size_distribution(state_i, shape_i)
    n_j = DT.size_distribution(state_j, shape_j)
    v_i = P3.ice_particle_terminal_velocity(vel, ρₐ, state_i)
    v_j = P3.ice_particle_terminal_velocity(vel, ρₐ, state_j)
    p = 1e-6  # matches the tail probability in `inter_category_collection`
    bi = P3.velocity_integral_bounds(state_i, shape_i, v_i; p)
    bj = P3.velocity_integral_bounds(state_j, shape_j, v_j; p)
    Dj_min, Dj_max = first(bj), last(bj)
    E = icp.E_ii * P3.rime_collection_shutoff(icp, state_i.F_rim)
    function inner(D_i, weight)
        vi = v_i(D_i)
        Dstar = clamp(P3.crossover_diameter(vi, v_j, Dj_min, Dj_max), Dj_min, Dj_max)
        segs = sort(collect(Float64.((bj..., Dstar))))
        f =
            D_j ->
                P3.collision_cross_section_ice_ice(state_i, D_i, state_j, D_j) *
                max(vi - v_j(D_j), 0) * n_j(D_j) * weight(D_j)
        return QGK.quadgk(f, segs...; rtol = 1e-8)[1]
    end
    outer_segs = Float64.(bi)
    ∫N = QGK.quadgk(D_i -> n_i(D_i) * inner(D_i, _ -> 1.0), outer_segs...; rtol = 1e-8)[1]
    ∫q = QGK.quadgk(D_i -> n_i(D_i) * inner(D_i, D_j -> P3.ice_mass(state_j, D_j)), outer_segs...; rtol = 1e-8)[1]
    return (∂ₜN_j = E * ∫N, ∂ₜq_j = E * ∫q)
end

function test_intercategory_params(FT)
    @testset "InterCategoryParams construction" begin
        # Val(N) selects the initiation threshold; the five Fortran values.
        expected = (2 => 5.0e-4, 3 => 4.0e-4, 4 => 2.35e-4, 5 => 1.75e-4, 6 => 1.5e-4)
        toml = CP.create_toml_dict(FT)
        for (N, ΔD) in expected
            icp = CMP.InterCategoryParams(toml, Val(N))
            @test icp.ΔD_init ≈ FT(ΔD)
            @test eltype(icp) == FT
        end
        # N ≥ 6 reuses the six-category value.
        @test CMP.InterCategoryParams(toml, Val(8)).ΔD_init ≈ FT(1.5e-4)
        # Keyword form and the FT convenience form agree with Val.
        @test CMP.InterCategoryParams(toml; n_categories = 3).ΔD_init ≈ FT(4.0e-4)
        @test CMP.InterCategoryParams(FT; n_categories = 4).ΔD_init ≈ FT(2.35e-4)
        # Shared fields.
        icp = CMP.InterCategoryParams(FT; n_categories = 2)
        @test icp.E_ii ≈ FT(0.1)
        @test icp.F_rim_shutoff_start ≈ FT(0.6)
        @test icp.F_rim_shutoff_end ≈ FT(0.9)
        @test icp.ΔD_merge ≈ FT(1.5e-4)
        @test icp.Δρ_merge ≈ FT(100)
        # A single category is not a multi-category configuration.
        @test_throws ArgumentError CMP.InterCategoryParams(toml, Val(1))

        # P3IceParams carries `nothing` by default; n_categories stays 1.
        mp = CMP.Microphysics2MParams(FT; with_ice = true)
        @test mp.ice.inter_category === nothing
        @test CMP.n_categories(mp.ice) == 1
        # An InterCategoryParams is attached with an explicit category count.
        ice = CMP.P3IceParams(FT; inter_category = icp, n_categories = 2)
        @test ice.inter_category === icp
        @test CMP.n_categories(ice) == 2
        @test_throws ArgumentError CMP.P3IceParams(FT; inter_category = icp)
    end
end

function test_ordered_category_pairs(FT)
    @testset "ordered_category_pairs" begin
        @test P3.ordered_category_pairs(Val(1)) == ()
        @test @inferred(P3.ordered_category_pairs(Val(2))) == ((1, 2), (2, 1))
        for N in 2:6
            pairs = P3.ordered_category_pairs(Val(N))
            @test length(pairs) == N * (N - 1)
            @test all(p -> p[1] != p[2], pairs)     # distinct categories
            @test length(unique(pairs)) == length(pairs)  # no duplicates
            # every ordered distinct pair appears exactly once
            @test Set(pairs) == Set((i, j) for i in 1:N, j in 1:N if i != j)
        end
    end
end

function test_rime_shutoff_ramp(FT)
    @testset "rime-fraction collection shutoff (C1 ramp)" begin
        icp = CMP.InterCategoryParams(FT; n_categories = 2)
        lo, hi = icp.F_rim_shutoff_start, icp.F_rim_shutoff_end
        # endpoint values match the Fortran linear ramp endpoints
        @test P3.rime_collection_shutoff(icp, FT(0)) == 1
        @test P3.rime_collection_shutoff(icp, lo) ≈ 1
        @test P3.rime_collection_shutoff(icp, (lo + hi) / 2) ≈ FT(0.5)
        @test P3.rime_collection_shutoff(icp, hi) ≈ 0 atol = eps(FT)
        @test P3.rime_collection_shutoff(icp, FT(0.99)) == 0
        # monotone non-increasing across the ramp
        Frs = FT.(range(0, 1; length = 40))
        vals = P3.rime_collection_shutoff.(Ref(icp), Frs)
        @test all(diff(vals) .<= eps(FT))
        # C1 continuity: near-zero slope at both bounds (the C1 improvement over
        # the Fortran linear ramp, whose slope jumps at 0.6 and 0.9)
        δ = FT(1e-3)
        slope(F) = (P3.rime_collection_shutoff(icp, F + δ) - P3.rime_collection_shutoff(icp, F - δ)) / (2δ)
        @test abs(slope(lo)) < FT(0.05)
        @test abs(slope(hi)) < FT(0.05)
    end
end

function test_intercategory_collection(FT)
    params = CMP.ParametersP3(FT)
    vel = CMP.Chen2022VelType(FT)
    icp = CMP.InterCategoryParams(FT; n_categories = 2)
    ρₐ = FT(1.2)
    quad = P3.GaussLegendre(FT, 6)

    # (ρq_ice, ρn_ice, F_rim, ρ_rim) for a grid spanning the rime ramp
    cats = (
        (FT(1e-4), FT(2e5), FT(0.0), FT(400)),    # small pristine, unrimed
        (FT(5e-4), FT(5e4), FT(0.5), FT(500)),    # mid, mid-rimed (ramp)
        (FT(2e-3), FT(1e4), FT(0.95), FT(800)),   # large, heavily rimed (near shutoff)
    )
    states = map(c -> P3.P3State(params, c...), cats)
    shapes = map(P3.get_distribution_shape, states)

    @testset "inter-category collection: transfer contract and Y-factors" begin
        for a in 1:3, b in 1:3
            a == b && continue
            si, shi = states[a], shapes[a]   # collector i
            sj, shj = states[b], shapes[b]   # collectee j
            r = @inferred P3.inter_category_collection(si, shi, sj, shj, vel, ρₐ, icp; quad)
            @test all(isfinite, values(r))
            @test r.∂ₜN_j ≥ 0 && r.∂ₜq_j ≥ 0

            # Double-entry contract: the returned collectee-side mass-like rates
            # are the collector-side gains. The wiring phase materializes the
            # +collector/-collectee pair; source-equals-sink conservation is
            # asserted there, on the assembled tendencies.
            @test propertynames(r) == (:∂ₜN_j, :∂ₜq_j, :∂ₜq_rim_j, :∂ₜb_rim_j, :∂ₜz_j)

            # Rime transfer uses the collectee bulk F_rim and ρ_rim (MM16
            # Y-factors); the proportionality is exact, and the transferred
            # rime arrives at the collectee rime density
            @test r.∂ₜq_rim_j == sj.F_rim * r.∂ₜq_j
            expected_b = iszero(sj.ρ_rim) ? zero(FT) : (sj.F_rim / sj.ρ_rim) * r.∂ₜq_j
            @test r.∂ₜb_rim_j == expected_b
            if !iszero(r.∂ₜb_rim_j)
                @test r.∂ₜq_rim_j / r.∂ₜb_rim_j ≈ sj.ρ_rim
            end
            # Two-moment ice carries no reflectivity transfer
            @test r.∂ₜz_j == 0
        end
    end

    @testset "inter-category collection: default-order convergence vs GL(64)" begin
        # The default order-6 rule agrees with the high-order rule to a few
        # percent across the state grid (both orderings, the rime-shutoff ramp).
        ref_quad = P3.GaussLegendre(FT, 64)
        for a in 1:3, b in 1:3
            a == b && continue
            args = (states[a], shapes[a], states[b], shapes[b], vel, ρₐ, icp)
            lo = P3.inter_category_collection(args...; quad)
            hi = P3.inter_category_collection(args...; quad = ref_quad)
            hi.∂ₜN_j == 0 && continue
            @test lo.∂ₜN_j ≈ hi.∂ₜN_j rtol = FT(3e-2)
            @test lo.∂ₜq_j ≈ hi.∂ₜq_j rtol = FT(3e-2)
        end
    end

    if FT == Float64
        @testset "inter-category collection: GL(64) vs independent nested QuadGK" begin
            # Validate the integrand and the crossover breakpoint independently:
            # the high-order rule matches an adaptive nested-QuadGK reference,
            # and the rime transfers are the exact Y-factor multiples of the
            # independently computed mass rate.
            for a in 1:3, b in 1:3
                a == b && continue
                sj = states[b]
                args = (states[a], shapes[a], sj, shapes[b], vel, ρₐ, icp)
                got = P3.inter_category_collection(args...; quad = P3.GaussLegendre(FT, 64))
                got.∂ₜN_j == 0 && continue
                ref = _intercat_reference(args...)
                @test got.∂ₜN_j ≈ ref.∂ₜN_j rtol = FT(5e-3)
                @test got.∂ₜq_j ≈ ref.∂ₜq_j rtol = FT(5e-3)
                @test got.∂ₜq_rim_j ≈ sj.F_rim * ref.∂ₜq_j rtol = FT(5e-3)
                if !iszero(sj.ρ_rim)
                    @test got.∂ₜb_rim_j ≈ (sj.F_rim / sj.ρ_rim) * ref.∂ₜq_j rtol = FT(5e-3)
                end
            end
        end
    end

    @testset "inter-category collection: constant-μ Eq-10 reflectivity sink" begin
        ρz = FT(1e-6)
        si, shi = states[2], shapes[2]  # collector (efficiency not shut off)
        sj, shj = states[1], shapes[1]  # collectee, unrimed
        sj_z = _with_z(sj, ρz)
        r = P3.inter_category_collection(si, shi, sj_z, shj, vel, ρₐ, icp; quad)
        # Statement of the implemented closure: Z = c(μ) L²/N with μ frozen, so
        # ∂ₜz = Z (2 ∂ₜq/L − ∂ₜN/N), using both quadrature results
        @test r.∂ₜz_j == ρz * (2 * r.∂ₜq_j / sj.ρq_ice - r.∂ₜN_j / sj.ρn_ice)
        @test r.∂ₜz_j > 0

        # Size-selective case: a dense small-graupel collector sweeping tiny
        # crystals removes mass faster than number (large crystals are
        # preferentially collected through the cross-section growth), so the
        # Eq-10 sink exceeds the number-proportional amplitude-scaling value
        s_grp = P3.P3State(params, FT(5e-4), FT(1e5), FT(0.55), FT(900))
        sh_grp = P3.get_distribution_shape(s_grp)
        s_tiny = _with_z(P3.P3State(params, FT(1e-5), FT(1e6), FT(0), FT(400)), ρz)
        sh_tiny = P3.get_distribution_shape(s_tiny)
        r_sel = P3.inter_category_collection(s_grp, sh_grp, s_tiny, sh_tiny, vel, ρₐ, icp; quad)
        amp_sel = (ρz / s_tiny.ρn_ice) * r_sel.∂ₜN_j
        @test r_sel.∂ₜz_j > amp_sel

        # Monodisperse limit: for a narrow collectee distribution (large μ at
        # fixed mean diameter) mass and number are removed in proportion, so
        # the Eq-10 sink reduces to the amplitude-scaling value
        D̄ = FT(100e-6)
        μ_narrow = FT(200)
        sh_n = P3.P3Shape(; logλ = log((μ_narrow + 1) / D̄), μ = μ_narrow)
        L_n = sj.ρn_ice * exp(P3.logLdivN(sj, sh_n))  # mass consistent with the narrow shape
        sj_n = _with_z(P3.P3State(params, L_n, sj.ρn_ice, FT(0), FT(400)), ρz)
        r_n = P3.inter_category_collection(si, shi, sj_n, sh_n, vel, ρₐ, icp; quad)
        amp_n = (ρz / sj_n.ρn_ice) * r_n.∂ₜN_j
        @test r_n.∂ₜz_j ≈ amp_n rtol = FT(2e-2)
    end

    @testset "inter-category collection: empty-category limit is continuous" begin
        si, shi = states[2], shapes[2]  # populated collector (efficiency not shut off)
        # Shrink the collectee amplitude at fixed mean mass (fixed shape). The
        # rate is then linear in the amplitude, so it vanishes continuously with
        # a finite slope: the ratio rate/scale is constant across scales.
        ratios_N = FT[]
        ratios_q = FT[]
        prev = FT(Inf)
        for s in FT.((1e-1, 1e-2, 1e-3))
            sj = P3.P3State(params, FT(1e-3) * s, FT(2e5) * s, FT(0), FT(400))
            shj = P3.get_distribution_shape(sj)
            r = P3.inter_category_collection(si, shi, sj, shj, vel, ρₐ, icp; quad)
            @test isfinite(r.∂ₜN_j) && r.∂ₜN_j > 0
            @test r.∂ₜN_j < prev           # decreases toward zero
            prev = r.∂ₜN_j
            push!(ratios_N, r.∂ₜN_j / s)
            push!(ratios_q, r.∂ₜq_j / s)
        end
        @test ratios_N[1] ≈ ratios_N[end] rtol = FT(1e-3)   # finite slope (C1)
        @test ratios_q[1] ≈ ratios_q[end] rtol = FT(1e-3)
        # exactly empty collectee → zero, no jump
        sj0 = P3.P3State(params, FT(0), FT(0), FT(0), FT(400))
        shj0 = P3.get_distribution_shape(sj0)
        r0 = P3.inter_category_collection(si, shi, sj0, shj0, vel, ρₐ, icp; quad)
        @test r0.∂ₜN_j == 0 && r0.∂ₜq_j == 0 && r0.∂ₜz_j == 0
        # empty collector → zero as well
        r0i = P3.inter_category_collection(sj0, shj0, si, shi, vel, ρₐ, icp; quad)
        @test r0i.∂ₜN_j == 0 && r0i.∂ₜq_j == 0
    end
end

function test_icecat_destination(FT)
    params = CMP.ParametersP3(FT)
    icp = CMP.InterCategoryParams(FT; n_categories = 3)   # ΔD_init = 400 µm

    small = P3.P3State(params, FT(1e-4), FT(3e5), FT(0), FT(400))   # small mean size
    large = P3.P3State(params, FT(3e-3), FT(5e3), FT(0.6), FT(700)) # large mean size
    empty = P3.P3State(params, FT(0), FT(0), FT(0), FT(400))
    sh_s = P3.get_distribution_shape(small)
    sh_l = P3.get_distribution_shape(large)
    sh_e = P3.get_distribution_shape(empty)
    Dm_s = P3.D_m(small, sh_s)
    Dm_l = P3.D_m(large, sh_l)

    @testset "icecat_destination truth table" begin
        # N = 1 degenerate: always category 1
        @test P3.icecat_destination((small,), (sh_s,), Dm_s, icp) == 1
        @test P3.icecat_destination((empty,), (sh_e,), FT(1e-4), icp) == 1

        # all empty → category 1
        @test P3.icecat_destination((empty, empty, empty), (sh_e, sh_e, sh_e), FT(1e-4), icp) == 1

        # all populated → smallest |D - D_new|
        states = (small, large)
        shapes = (sh_s, sh_l)
        @test P3.icecat_destination(states, shapes, Dm_s, icp) == 1
        @test P3.icecat_destination(states, shapes, Dm_l, icp) == 2
        @test @inferred(P3.icecat_destination(states, shapes, Dm_l, icp)) isa Int

        # partial (one empty): closest within ΔD_init → that populated category (3a)
        states3 = (small, large, empty)
        shapes3 = (sh_s, sh_l, sh_e)
        @test P3.icecat_destination(states3, shapes3, Dm_s + FT(1e-5), icp) == 1

        # partial: closest exceeds ΔD_init → first empty category (3b)
        D_far = Dm_l + FT(1e-2)  # far from both populated categories
        @test P3.icecat_destination(states3, shapes3, D_far, icp) == 3

        # first-empty selection uses the lowest empty index
        states_e = (empty, small, empty)
        shapes_e = (sh_e, sh_s, sh_e)
        @test P3.icecat_destination(states_e, shapes_e, Dm_s + FT(1e-2), icp) == 1

        # a category below the presence threshold counts as empty
        ϵ_pres = FT(P3.UT.SPECIES_PRESENCE_THRESHOLD)
        tiny = P3.P3State(params, ϵ_pres / 2, FT(10), FT(0), FT(400))
        sh_t = P3.get_distribution_shape(tiny)
        @test P3.icecat_destination((small, tiny), (sh_s, sh_t), Dm_s + FT(1e-2), icp) == 2
        @test P3.icecat_destination((tiny, tiny), (sh_t, sh_t), FT(1e-4), icp) == 1  # all empty
    end
end

function test_merge_categories(FT)
    params = CMP.ParametersP3(FT)
    icp = CMP.InterCategoryParams(FT; n_categories = 3)  # ΔD_merge=150µm, Δρ_merge=100

    _states(cats) = map(c -> P3.P3State(params, c...), cats)
    _shapes(states) = map(P3.get_distribution_shape, states)
    total(prog) = (
        ρq_ice = sum(p -> p.ρq_ice, prog),
        ρn_ice = sum(p -> p.ρn_ice, prog),
        ρq_rim = sum(p -> p.ρq_rim, prog),
        ρb_rim = sum(p -> p.ρb_rim, prog),
        ρz_ice = sum(p -> p.ρz_ice, prog),
    )

    @testset "merge: conservation and idempotence" begin
        # Two well-separated categories plus an empty slot → no merge
        cats =
            ((FT(1e-4), FT(3e5), FT(0), FT(400)), (FT(3e-3), FT(5e3), FT(0.6), FT(700)), (FT(0), FT(0), FT(0), FT(400)))
        states = _states(cats)
        shapes = _shapes(states)
        merged = @inferred P3.merge_categories(states, shapes, icp)
        @test merged[1].ρq_ice ≈ states[1].ρq_ice
        @test merged[2].ρq_ice ≈ states[2].ρq_ice
        @test merged[3].ρq_ice == 0

        # Idempotence for an active merge: two identical categories collapse into
        # one; reconstructing and merging again reproduces the merged state.
        c = (FT(5e-4), FT(1e5), FT(0.3), FT(400))
        mstates = (P3.P3State(params, c...), P3.P3State(params, c...))
        mshapes = _shapes(mstates)
        m1 = P3.merge_categories(mstates, mshapes, icp)
        @test m1[1].ρq_ice ≈ 2 * c[1] && m1[2].ρq_ice == 0
        states2 = map(p -> P3.state_from_prognostic(params, p.ρq_ice, p.ρn_ice, p.ρq_rim, p.ρb_rim), m1)
        m2 = P3.merge_categories(states2, _shapes(states2), icp)
        for k in 1:2
            @test m2[k].ρq_ice ≈ m1[k].ρq_ice
            @test m2[k].ρn_ice ≈ m1[k].ρn_ice
        end
    end

    @testset "merge: criterion boundaries" begin
        # Two categories close in size and density → merge into category 1
        base = (FT(5e-4), FT(1e5), FT(0.3), FT(400))
        s1 = P3.P3State(params, base...)
        sh1 = P3.get_distribution_shape(s1)
        Dm1 = P3.D_m(s1, sh1)
        ρ1 = P3.mean_ice_density(s1, sh1)

        # Construct a second category with nearly identical mean size and density
        # by reusing the same prognostic (guaranteed within both thresholds)
        s2 = P3.P3State(params, base...)
        states = (s1, s2)
        shapes = (sh1, P3.get_distribution_shape(s2))
        merged = P3.merge_categories(states, shapes, icp)
        @test merged[1].ρq_ice ≈ 2 * base[1]     # summed into category 1
        @test merged[1].ρn_ice ≈ 2 * base[2]
        @test merged[2].ρq_ice == 0              # category 2 emptied

        # A large category far in size from a small one → no merge
        big = P3.P3State(params, FT(4e-3), FT(2e3), FT(0.6), FT(800))
        states_far = (s1, big)
        shapes_far = (sh1, P3.get_distribution_shape(big))
        Dm_big = P3.D_m(big, P3.get_distribution_shape(big))
        @test abs(Dm_big - Dm1) > icp.ΔD_merge   # diameter criterion violated
        merged_far = P3.merge_categories(states_far, shapes_far, icp)
        @test merged_far[1].ρq_ice ≈ base[1]
        @test merged_far[2].ρq_ice ≈ FT(4e-3)

        # A category below the presence threshold does not merge, and its
        # prognostics pass through finite
        ϵ_pres = FT(P3.UT.SPECIES_PRESENCE_THRESHOLD)
        tiny = P3.P3State(params, ϵ_pres / 2, FT(10), FT(0), FT(400))
        states_t = (s1, tiny)
        shapes_t = (sh1, P3.get_distribution_shape(tiny))
        merged_t = P3.merge_categories(states_t, shapes_t, icp)
        @test merged_t[1].ρq_ice ≈ base[1]
        @test merged_t[2].ρq_ice == ϵ_pres / 2
        @test all(p -> all(isfinite, values(p)), merged_t)
    end

    @testset "merge: conservation of totals and reflectivity summation" begin
        # Three similar categories (all within thresholds) collapse into one;
        # totals are conserved and reflectivity sums additively (M21)
        c = (FT(5e-4), FT(1e5), FT(0.3), FT(400))
        s = P3.P3State(params, c...)
        sh = P3.get_distribution_shape(s)
        sz = _with_z(s, FT(2e-6))
        states = (sz, sz, sz)
        shapes = (sh, sh, sh)
        before = total(map(st -> P3._prognostic_namedtuple(P3._category_prognostic(st)), states))
        merged = P3.merge_categories(states, shapes, icp)
        after = total(merged)
        @test after.ρq_ice ≈ before.ρq_ice
        @test after.ρn_ice ≈ before.ρn_ice
        @test after.ρq_rim ≈ before.ρq_rim
        @test after.ρb_rim ≈ before.ρb_rim
        @test after.ρz_ice ≈ before.ρz_ice
        # collapsed into the lowest index
        @test merged[1].ρz_ice ≈ 3 * FT(2e-6)
        @test merged[2].ρz_ice == 0 && merged[3].ρz_ice == 0
    end
end

# Packed multi-category entry arguments: `cats` are the per-category prognostic
# NamedTuples; shapes are diagnosed per category.
function _ncat_entry_args(FT, cats, mode...; kw...)
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true, n_categories = length(cats), kw...)
    p3 = mp.ice.scheme
    ρ = FT(0.9)
    shapes = map(cats) do c
        ρq_liq = haskey(c, :q_liq_on_ice) ? c.q_liq_on_ice * ρ : nothing
        ρz = haskey(c, :z_ice) ? c.z_ice * ρ : nothing
        st = P3.state_from_prognostic(p3, c.q_ice * ρ, c.n_ice * ρ, c.q_rim * ρ, c.b_rim * ρ, ρq_liq, ρz)
        P3.get_distribution_shape(st)
    end
    tail = isempty(mode) ? () : (FT(60), 4)
    warm = (FT(1e-3), FT(1e8), FT(5e-4), FT(1e4))
    return (;
        mp,
        args = (mode..., BMT.Microphysics2Moment(), mp, tps, ρ, FT(263), FT(8e-3), warm..., cats, shapes, tail...),
    )
end

_ncat_cat1(FT) = (; q_ice = FT(1e-4), n_ice = FT(2e5), q_rim = FT(4e-5), b_rim = FT(6e-8))
_ncat_cat2(FT) = (; q_ice = FT(1e-3), n_ice = FT(1e4), q_rim = FT(4e-4), b_rim = FT(8e-7))
_ncat_empty(FT) = (; q_ice = FT(0), n_ice = FT(0), q_rim = FT(0), b_rim = FT(0))

# Rebuild the parameter set with a different InterCategoryParams.
function _replace_icp(mp, icp)
    ice = CMP.P3IceParams(;
        (
            k => getfield(mp.ice, k) for k in
            (:scheme, :terminal_velocity, :cloud_pdf, :rain_pdf, :ice_nucleation,
                :rain_freezing, :inp_depletion_model, :quad)
        )...,
        inter_category = icp, n_categories = CMP.n_categories(mp.ice),
    )
    return CMP.Microphysics2MParams(; warm_rain = mp.warm_rain, ice)
end

# Initiation threshold so wide that every source routes to the closest
# populated category, as in a single-category configuration.
_wide_init_icp(FT, icp) = CMP.InterCategoryParams(;
    icp.E_ii, icp.F_rim_shutoff_start, icp.F_rim_shutoff_end,
    ΔD_init = FT(1), icp.ΔD_merge, icp.Δρ_merge,
)

function test_ncat_params(FT)
    @testset "category-count type parameter" begin
        toml = CP.create_toml_dict(FT)
        mp1 = CMP.Microphysics2MParams(toml; with_ice = true)
        @test CMP.n_categories(mp1.ice) == 1
        @test mp1.ice.inter_category === nothing
        for N in 2:4
            mpN = CMP.Microphysics2MParams(toml; with_ice = true, n_categories = N)
            @test CMP.n_categories(mpN.ice) == N
            @test mpN.ice.inter_category isa CMP.InterCategoryParams
        end
        @test_throws ArgumentError CMP.Microphysics2MParams(toml; with_ice = true, n_categories = 5)
        # `inter_category` must be `nothing` exactly for a single category
        ice2 = CMP.Microphysics2MParams(toml; with_ice = true, n_categories = 2).ice
        kwfields = (;
            (
                k => getfield(ice2, k) for k in
                (:scheme, :terminal_velocity, :cloud_pdf, :rain_pdf, :ice_nucleation, :rain_freezing)
            )...
        )
        @test_throws ArgumentError CMP.P3IceParams(; kwfields..., n_categories = 2)
        @test_throws ArgumentError CMP.P3IceParams(;
            kwfields..., inter_category = ice2.inter_category, n_categories = 1,
        )
        # packed inputs must carry n_categories(mp.ice) categories
        (; args) = _ncat_entry_args(FT, (_ncat_cat1(FT), _ncat_cat2(FT)))
        bad = (args[1:10]..., (args[11][1],), (args[12][1],))
        @test_throws ArgumentError BMT.bulk_microphysics_tendencies(bad...)
    end
end

function test_ncat1_equivalence(FT)
    @testset "two categories with an empty slot reproduce the single category" begin
        cat1 = _ncat_cat1(FT)
        (; args) = _ncat_entry_args(FT, (cat1,))
        # Sources route to the closest populated category under the wide
        # initiation threshold (rain freezing otherwise opens the empty slot:
        # the largest drops freeze first, so the frozen-drop mean diameter sits
        # far from a small pristine population).
        r2 = _ncat_entry_args(FT, (cat1, _ncat_empty(FT)))
        mp2 = _replace_icp(r2.mp, _wide_init_icp(FT, r2.mp.ice.inter_category))
        args2 = (r2.args[1], mp2, r2.args[3:end]...)
        t1 = BMT.bulk_microphysics_tendencies(args...)
        t2 = BMT.bulk_microphysics_tendencies(args2...)
        # warm block and the populated category's block are bit-identical
        for k in (:dq_lcl_dt, :dn_lcl_dt, :dq_rai_dt, :dn_rai_dt, :dn_lcl_activation_dt)
            @test getproperty(t2, k) === getproperty(t1, k)
        end
        @test t2.dq_ice_1_dt === t1.dq_ice_dt
        @test t2.dn_ice_1_dt === t1.dn_ice_dt
        @test t2.dq_rim_1_dt === t1.dq_rim_dt
        @test t2.db_rim_1_dt === t1.db_rim_dt
        # the empty category receives nothing
        @test t2.dq_ice_2_dt == 0
        @test t2.dn_ice_2_dt == 0
        @test t2.dq_rim_2_dt == 0
        @test t2.db_rim_2_dt == 0
    end
end

function test_ncat2_intercategory_conservation(FT)
    @testset "inter-category double entry conserves mass, rime, and volume" begin
        cats = (_ncat_cat1(FT), _ncat_cat2(FT))
        (; mp, args) = _ncat_entry_args(FT, cats)
        # reference with collection off: E_ii = 0 zeroes only the
        # inter-category transfers, every other term is bit-identical
        icp = mp.ice.inter_category
        icp0 = CMP.InterCategoryParams(;
            E_ii = FT(0),
            icp.F_rim_shutoff_start, icp.F_rim_shutoff_end,
            icp.ΔD_init, icp.ΔD_merge, icp.Δρ_merge,
        )
        mp0 = _replace_icp(mp, icp0)
        args0 = (args[1], mp0, args[3:end]...)

        ta = BMT.bulk_microphysics_tendencies(args...)
        tb = BMT.bulk_microphysics_tendencies(args0...)

        # collection is active and moves mass between the categories
        @test ta.dq_ice_1_dt != tb.dq_ice_1_dt
        @test ta.dq_ice_2_dt != tb.dq_ice_2_dt
        # the warm block does not see inter-category collection
        for k in (:dq_lcl_dt, :dn_lcl_dt, :dq_rai_dt, :dn_rai_dt)
            @test getproperty(ta, k) === getproperty(tb, k)
        end
        # the category sums of the transferred quantities are unchanged
        tol(v...) = 32 * eps(FT) * maximum(abs, v)
        for (k1, k2) in ((:dq_ice_1_dt, :dq_ice_2_dt), (:dq_rim_1_dt, :dq_rim_2_dt), (:db_rim_1_dt, :db_rim_2_dt))
            Σa = getproperty(ta, k1) + getproperty(ta, k2)
            Σb = getproperty(tb, k1) + getproperty(tb, k2)
            @test abs(Σa - Σb) ≤ tol(getproperty(ta, k1), getproperty(ta, k2), Σa, Σb)
        end
        # number is a pure collectee sink: the category sum decreases
        @test ta.dn_ice_1_dt + ta.dn_ice_2_dt < tb.dn_ice_1_dt + tb.dn_ice_2_dt
    end
end

function test_ncat2_destination_routing(FT)
    @testset "destination routing through the entry" begin
        # large ice in category 1 (mean diameter far from the nascent-crystal
        # size) with an empty category 2: cold supersaturated deposition
        # nucleation opens category 2
        big = (; q_ice = FT(2e-3), n_ice = FT(1e3), q_rim = FT(2e-4), b_rim = FT(4e-7))
        (; args) = _ncat_entry_args(FT, (big, _ncat_empty(FT)))
        argsT = (args[1:4]..., FT(250), FT(8e-3), args[7:end]...)
        t = BMT.bulk_microphysics_tendencies(argsT...)
        @test t.dn_ice_2_dt > 0
        @test t.dq_ice_2_dt > 0
        # wide initiation threshold: every source stays in the populated
        # category and the empty slot receives nothing
        rs = _ncat_entry_args(FT, (_ncat_cat1(FT), _ncat_empty(FT)))
        mps = _replace_icp(rs.mp, _wide_init_icp(FT, rs.mp.ice.inter_category))
        args_sT = (rs.args[1], mps, rs.args[3:4]..., FT(250), FT(8e-3), rs.args[7:end]...)
        ts = BMT.bulk_microphysics_tendencies(args_sT...)
        @test ts.dn_ice_2_dt == 0
        @test ts.dq_ice_2_dt == 0
    end
end

function test_ncat_entry_gates(FT)
    @testset "two-category inference, allocations, and mode support" begin
        cats = (_ncat_cat1(FT), _ncat_cat2(FT))
        (; args) = _ncat_entry_args(FT, cats)
        @test (@inferred BMT.bulk_microphysics_tendencies(args...)) isa NamedTuple
        JET.@test_opt BMT.bulk_microphysics_tendencies(args...)
        trial = BT.@benchmark $(BMT.bulk_microphysics_tendencies)($args...) samples = 50 evals = 1
        @test trial.memory == 0

        argse = _ncat_entry_args(FT, cats, BMT.rosenbrock_exact()).args
        te = @inferred BMT.bulk_microphysics_tendencies(argse...)
        @test all(isfinite, values(te))
        JET.@test_opt BMT.bulk_microphysics_tendencies(argse...)
        trial = BT.@benchmark $(BMT.bulk_microphysics_tendencies)($argse...) samples = 20 evals = 1
        @test trial.memory == 0

        # rosenbrock_exact finiteness over a small grid, including the
        # near-empty cold corner where the substep trajectory reaches a
        # zero-number state with mass
        for T in (FT(250), FT(263), FT(275)), scale in (FT(1), FT(1e-2), FT(1e-3))
            c1 = map(x -> x * scale, _ncat_cat1(FT))
            c2 = map(x -> x * scale, _ncat_cat2(FT))
            a = _ncat_entry_args(FT, (c1, c2), BMT.rosenbrock_exact()).args
            aT = (a[1:5]..., T, a[7:end]...)
            @test all(isfinite, values(BMT.bulk_microphysics_tendencies(aT...)))
        end

        argsm = _ncat_entry_args(FT, cats, BMT.rosenbrock_manual()).args
        @test_throws ArgumentError BMT.bulk_microphysics_tendencies(argsm...)
        argsv = _ncat_entry_args(FT, cats, BMT.Verbose(BMT.rosenbrock_exact())).args
        @test_throws ArgumentError BMT.bulk_microphysics_tendencies(argsv...)
    end
end

function test_joint_ncat2_smoke(FT)
    @testset "two categories with three-moment liquid-fraction ice" begin
        c1 = (; _ncat_cat1(FT)..., q_liq_on_ice = FT(3e-5), z_ice = FT(1e-8))
        c2 = (; _ncat_cat2(FT)..., q_liq_on_ice = FT(1e-4), z_ice = FT(3e-7))
        (; args) = _ncat_entry_args(FT, (c1, c2); moments = :three_moment, liquid = :predicted)
        t = BMT.bulk_microphysics_tendencies(args...)
        @test all(isfinite, values(t))
        # canonical per-category field order: liquid slot before the
        # reflectivity slot, category blocks in index order
        ks = collect(keys(t))
        for j in 1:2
            iq = findfirst(==(Symbol(:dq_ice_, j, :_dt)), ks)
            il = findfirst(==(Symbol(:dq_liq_on_ice_, j, :_dt)), ks)
            iz = findfirst(==(Symbol(:dz_ice_, j, :_dt)), ks)
            @test iq < il < iz
        end
        @test findfirst(==(:dz_ice_1_dt), ks) < findfirst(==(:dq_ice_2_dt), ks)
        # water conservation of the internal transfers: total condensed water
        # change is finite and bounded by the vapor-side exchange scale
        S = t.dq_lcl_dt + t.dq_rai_dt +
            t.dq_ice_1_dt + t.dq_ice_2_dt + t.dq_liq_on_ice_1_dt + t.dq_liq_on_ice_2_dt
        @test isfinite(S)
        argse =
            _ncat_entry_args(FT, (c1, c2), BMT.rosenbrock_exact(); moments = :three_moment, liquid = :predicted).args
        te = BMT.bulk_microphysics_tendencies(argse...)
        @test all(isfinite, values(te))
    end
end


function test_substep_solver(FT)
    @testset "substep dense solve above the StaticArrays cutoff" begin
        # Cross-check the internal-API bypass (`BMT._static_lu_solve`, built on
        # `StaticArrays.__lu`) against the public heap-fallback path, so a
        # StaticArrays change that breaks the internal kernel fails loudly.
        s = Ref(UInt64(0x9E3779B97F4A7C15))
        draw() = (s[] = 6364136223846793005 * s[] + 1442695040888963407; Float64(s[] >> 11) * (2.0^-53))
        for N in (16, 20, 24, 28)
            A = SA.SMatrix{N, N, FT}(ntuple(i -> FT(0.2 * (draw() - 0.5)), N * N)) +
                2 * one(SA.SMatrix{N, N, FT})
            b = SA.SVector{N, FT}(ntuple(i -> FT(draw() - 0.5), N))
            x = BMT._static_lu_solve(A, b)
            xref = SA.SVector{N, FT}(Matrix(A) \ Vector(b))
            @test x isa SA.SVector{N, FT}
            @test x ≈ xref rtol = 500 * eps(FT)
        end
    end
end

function test_ncat_substep_alloc(FT)
    @testset "substep solve allocations across the state layouts" begin
        base = _ncat_cat1(FT)
        liqz = (; q_liq_on_ice = FT(3e-5), z_ice = FT(1e-8))
        joint2 = ((; moments = :three_moment, liquid = :predicted), 2, (; base..., liqz...))  # 16 states
        configs =
            FT === Float64 ?
            (
                joint2,
                ((;), 3, base),                                                               # 16 states
                ((;), 4, base),                                                               # 20 states
                ((; moments = :three_moment), 4, (; base..., z_ice = FT(1e-8))),              # 24 states
                ((; moments = :three_moment, liquid = :predicted), 4, (; base..., liqz...)),  # 28 states
            ) : (joint2,)  # the code path is float-type independent
        for (kw, ncat, cat) in configs
            argse = _ncat_entry_args(FT, ntuple(_ -> cat, ncat), BMT.rosenbrock_exact(); kw...).args
            te = BMT.bulk_microphysics_tendencies(argse...)
            @test all(isfinite, values(te))
            trial = BT.@benchmark $(BMT.bulk_microphysics_tendencies)($argse...) samples = 10 evals = 1
            @test trial.memory == 0
        end
    end
end

@testset "P3 multicategory tests ($FT)" for FT in (Float64, Float32)
    test_intercategory_params(FT)
    test_ordered_category_pairs(FT)
    test_rime_shutoff_ramp(FT)
    test_intercategory_collection(FT)
    test_icecat_destination(FT)
    test_merge_categories(FT)
    test_ncat_params(FT)
    test_ncat1_equivalence(FT)
    test_ncat2_intercategory_conservation(FT)
    test_ncat2_destination_routing(FT)
    test_ncat_entry_gates(FT)
    test_joint_ncat2_smoke(FT)
    test_substep_solver(FT)
    test_ncat_substep_alloc(FT)
end
nothing
