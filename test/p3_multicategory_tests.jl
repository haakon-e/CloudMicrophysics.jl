using Test: @testset, @test, @test_throws, @inferred
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.DistributionTools as DT
import ClimaParams as CP
import QuadGK as QGK

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
        # An InterCategoryParams can be attached explicitly.
        ice = CMP.P3IceParams(FT; inter_category = icp)
        @test ice.inter_category === icp
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

@testset "P3 multicategory tests ($FT)" for FT in (Float64, Float32)
    test_intercategory_params(FT)
    test_ordered_category_pairs(FT)
    test_rime_shutoff_ramp(FT)
    test_intercategory_collection(FT)
    test_icecat_destination(FT)
    test_merge_categories(FT)
end
nothing
