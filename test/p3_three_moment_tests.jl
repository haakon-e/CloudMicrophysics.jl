using Test: @testset, @test, @test_throws, @inferred
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.Parameters as CMP
import SpecialFunctions as SF
import ForwardDiff as FD

# Model log(L/N) of the three-moment residual at (state, μ, log(Z/N)).
# See /home/haakon/notes/p3-mmc2025/impl/3mom-monotonicity-study.md
function _mono_φ(state, μ, logZdN)
    lλ = (SF.loggamma(μ + 7) - SF.loggamma(μ + 1) - logZdN) / 6
    return P3.logmass_gamma_moment(state, μ, lλ; n = 0) - P3.loggamma_moment(μ, lλ; k = 0)
end

# Build a three-moment state whose true shape is (μt, λt) for the given rime
# state and number concentration.
function _state_from_shape(params, μt, λt, F_rim, ρ_rim, N)
    FT = eltype(params.ρ_i)
    logλt = log(λt)
    ref = P3.P3State(params, FT(0), FT(0), F_rim, ρ_rim)
    logN₀ = P3.get_logN₀(N, μt, logλt)
    L = exp(logN₀ + P3.logmass_gamma_moment(ref, μt, logλt; n = 0))
    Z = N * exp(SF.loggamma(μt + 7) - SF.loggamma(μt + 1) - 6 * logλt)
    return (; state = P3.P3State(params, L, N, F_rim, ρ_rim, Z), L, Z)
end

# Rime states matching the monotonicity study grid.
function _rime_states(FT)
    states = Tuple{FT, FT}[(FT(0), FT(0))]
    for Fr in (FT(0.3), FT(0.6), FT(0.9), FT(0.99)), ρr in (FT(100), FT(400), FT(900))
        push!(states, (Fr, ρr))
    end
    return states
end

# log(Z/N) targets placing the logλ(μ) window relative to the [2, 17] rails.
function _zn_targets(FT)
    base = SF.loggamma(FT(17)) - SF.loggamma(FT(11))
    return FT[base - 6 * vc for vc in (FT(0), FT(2), FT(4), FT(9.5), FT(15), FT(17), FT(19))]
end

# Independent bisection reference for a monotone residual `r`, run to a tight
# tolerance to isolate the fixed-iteration budget error of the production solve.
function _bisect(r, lo, hi; tol = 1e-13, maxit = 200)
    a, b = lo, hi
    fa = r(a)
    fa * r(b) > 0 && return abs(fa) ≤ abs(r(b)) ? a : b
    for _ in 1:maxit
        m = (a + b) / 2
        fm = r(m)
        abs(b - a) < tol && return m
        if fa * fm ≤ 0
            b = m
        else
            a = m
            fa = fm
        end
    end
    return (a + b) / 2
end

function test_3m_monotonicity(FT)
    @testset "Residual monotonicity r(μ) over the (rime, Z/N) grid" begin
        params = CMP.ParametersP3(FT; moments = :three_moment)
        μs = collect(FT(0):FT(0.02):FT(20))
        n_nonmono = 0
        min_margin = FT(Inf)
        for (Fr, ρr) in _rime_states(FT), logZdN in _zn_targets(FT)
            st = P3.P3State(params, FT(1e-6), FT(1e5), Fr, ρr)
            φs = [_mono_φ(st, μ, logZdN) for μ in μs]
            dφ = diff(φs)
            all(>(0), dφ) || (n_nonmono += 1)
            min_margin = min(min_margin, minimum(dφ))
        end
        @test n_nonmono == 0
        @test min_margin > 0
    end
end

function test_3m_solve_accuracy(FT)
    @testset "Solve accuracy vs 200-iteration reference" begin
        params = CMP.ParametersP3(FT; moments = :three_moment)
        μ_targets = (FT(2), FT(5), FT(8), FT(11), FT(14), FT(17))
        lo, hi = FT(P3.LOGλ_MIN), FT(P3.LOGλ_MAX)
        # The shipped residual clamps logλ into the size rails inside r(μ); the
        # reference solves the same residual to a tight tolerance, so the metric
        # is the pure fixed-iteration budget error. Restricted to the physical
        # mean-mass band where the study calibrated the 8/10 budget.
        inband(logLdN) = -13 * log(10) ≤ logLdN ≤ -4 * log(10)
        worst_μ = FT(0)
        for (Fr, ρr) in _rime_states(FT), logZdN in _zn_targets(FT)
            st = P3.P3State(params, FT(1e-6), FT(1e5), Fr, ρr)
            resid(μ, target) =
                P3.logLdivN(st, μ, clamp((SF.loggamma(μ + 7) - SF.loggamma(μ + 1) - logZdN) / 6, lo, hi)) - target
            φs = [resid(μ, FT(0)) for μ in FT(0):FT(0.02):FT(20)]
            all(>(0), diff(φs)) || continue
            for μt in μ_targets
                target = resid(μt, FT(0))  # log(L/N) making the true root μt
                inband(Float64(target)) || continue
                μ_ref = _bisect(μ -> resid(μ, target), FT(0), FT(20))
                μ_prod = P3._solve_shape_μ(μ -> resid(μ, target), FT(0), FT(20))
                worst_μ = max(worst_μ, abs(μ_prod - μ_ref))
            end
        end
        tol = FT === Float32 ? FT(5e-3) : FT(1e-6)
        @test worst_μ < tol
    end
end

function test_3m_roundtrip(FT)
    @testset "Round trip (L, N, Z) → shape → (L, N, Z) in window" begin
        params = CMP.ParametersP3(FT; moments = :three_moment)
        N = FT(1e5)
        max_μ = FT(0)
        max_lλ = FT(0)
        max_Z = FT(0)
        for μt in FT.((0.5, 2, 5, 8, 11, 14, 18)), λt in FT.((1e3, 1e4, 1e5)),
            (Fr, ρr) in ((FT(0), FT(0)), (FT(0.5), FT(500)), (FT(0.9), FT(900)))

            (; state, Z) = _state_from_shape(params, μt, λt, Fr, ρr, N)
            shape = P3.get_distribution_shape(state)
            max_μ = max(max_μ, abs(shape.μ - μt))
            max_lλ = max(max_lλ, abs(shape.logλ - log(λt)))
            # Analytic sixth moment recomputed from the solved shape.
            Z_solved = N * exp(SF.loggamma(shape.μ + 7) - SF.loggamma(shape.μ + 1) - 6 * shape.logλ)
            max_Z = max(max_Z, abs(Z_solved - Z) / Z)
        end
        μ_tol = FT === Float32 ? FT(5e-3) : FT(1e-8)
        @test max_μ < μ_tol
        @test max_lλ < (FT === Float32 ? FT(1e-3) : FT(1e-9))
        @test max_Z < (FT === Float32 ? FT(2e-2) : FT(1e-6))
    end
end

function test_3m_rails(FT)
    @testset "Rail behavior (C0 in the mass target, clean μ saturation)" begin
        params = CMP.ParametersP3(FT; moments = :three_moment)
        μ_max = params.moments.μ_max
        # A Z/N straddling the lower logλ rail; sweep the mass target across it.
        base = SF.loggamma(FT(17)) - SF.loggamma(FT(11))
        N = FT(1e5)
        st_ref = P3.P3State(params, FT(1e-6), FT(1e5), FT(0.6), FT(400))
        μ_prev = nothing
        max_jump = FT(0)
        for lr in ((FT(2)), (FT(17)))
            logZdN = base - 6 * lr
            φlo = _mono_φ(st_ref, FT(0.001), logZdN)
            φhi = _mono_φ(st_ref, μ_max - FT(0.001), logZdN)
            targets = range(φlo, φhi; length = 400)
            μs = FT[]
            for t in targets
                Z = N * exp(logZdN)
                L = N * exp(t)
                st = P3.P3State(params, L, N, FT(0.6), FT(400), Z)
                shape = P3.get_distribution_shape(st)
                @test isfinite(shape.μ) && isfinite(shape.logλ)
                @test FT(0) ≤ shape.μ ≤ μ_max
                @test FT(P3.LOGλ_MIN) ≤ shape.logλ ≤ FT(P3.LOGλ_MAX)
                push!(μs, shape.μ)
            end
            max_jump = max(max_jump, maximum(abs.(diff(μs))))
        end
        # A continuous target→μ map has no O(1) jumps at this sweep density.
        @test max_jump < FT(0.5)
    end
end

function test_3m_onset(FT)
    @testset "Onset continuity and ForwardDiff derivatives (Z, N, L → 0)" begin
        params = CMP.ParametersP3(FT; moments = :three_moment)
        Chen = CMP.Chen2022VelType(FT)
        ρa = FT(1.0)
        N = FT(1e5)
        (; state) = _state_from_shape(params, FT(5), FT(1e4), FT(0.5), FT(500), N)
        L0 = state.ρq_ice
        Z0 = state.ρz_ice

        # Value continuity: shape and V_z finite as each moment → 0.
        for scale in FT.((1.0, 1e-2, 1e-4, 1e-8, 0.0))
            for st in (
                P3.P3State(params, L0, N, FT(0.5), FT(500), Z0 * scale),
                P3.P3State(params, L0, N * scale, FT(0.5), FT(500), Z0),
                P3.P3State(params, L0 * scale, N, FT(0.5), FT(500), Z0),
            )
                shape = P3.get_distribution_shape(st)
                @test isfinite(shape.μ) && isfinite(shape.logλ)
                v = P3.ice_terminal_velocity_reflectivity_weighted(Chen, ρa, st, shape)
                @test isfinite(v) && v ≥ 0
            end
        end

        # ForwardDiff: recovery ρz = ρz_adv²/max(ρn, n_presence) has a bounded
        # derivative in ρn near onset, floored by the number presence scale.
        n_presence = FT(1e-3)
        ρz_adv = P3.advected_reflectivity(N, Z0)
        drecov = FD.derivative(ρn -> P3.reflectivity_from_advected(ρz_adv, ρn, n_presence), FT(0))
        @test isfinite(drecov)

        # The growth coefficients stay finite as ρq_ice → 0 (mean-mass band).
        for q in FT.((0.0, 1e-12, 1e-8))
            st = P3.P3State(params, q, N, FT(0.5), FT(500), Z0)
            sh = P3.get_distribution_shape(st)
            coeffs = P3.reflectivity_growth_coefficients(st, sh)
            @test isfinite(coeffs.G) && isfinite(coeffs.M3divN) && isfinite(coeffs.M3divL)
            @test isfinite(P3.reflectivity_growth_tendency(coeffs, FT(1e-6), FT(0)))
        end
    end
end

function test_3m_z_bounds(FT)
    @testset "State-side reflectivity admissibility clamp" begin
        params3 = CMP.ParametersP3(FT; moments = :three_moment)
        params2 = CMP.ParametersP3(FT)
        N = FT(1e5)
        (; state, Z) = _state_from_shape(params3, FT(5), FT(1e4), FT(0.5), FT(500), N)
        # A physical Z is left untouched.
        @test state.ρz_ice ≈ Z rtol = 10 * eps(FT)
        # An out-of-window Z is clamped into the admissible window.
        (; zn_lo, zn_hi) = params3.moments
        @test 0 < zn_lo < zn_hi < FT(Inf)
        st_hi = P3.P3State(params3, state.ρq_ice, N, FT(0.5), FT(500), FT(1e30))
        @test st_hi.ρz_ice ≤ zn_hi * N * (1 + 10 * eps(FT))
        # Under two-moment ice ρz_ice stays zero regardless of input.
        st2 = P3.P3State(params2, state.ρq_ice, N, FT(0.5), FT(500), FT(1e-10))
        @test st2.ρz_ice == FT(0)
        # Empty state → zero.
        st0 = P3.P3State(params3, FT(0), FT(0), FT(0.5), FT(500), FT(1e-10))
        @test st0.ρz_ice == FT(0)
    end
end

function test_3m_tendencies(FT)
    @testset "Per-process reflectivity tendencies" begin
        params = CMP.ParametersP3(FT; moments = :three_moment)
        moments = params.moments
        N = FT(1e5)
        (; state) = _state_from_shape(params, FT(5), FT(1e4), FT(0.5), FT(500), N)
        shape = P3.get_distribution_shape(state)

        # ZContribution accumulator.
        z0 = P3.ZContribution{FT}()
        @test iszero(z0.dL_growth) && iszero(z0.dZ_init)
        za = P3.ZContribution{FT}(FT(1), FT(2), FT(3), FT(4))
        zsum = z0 + za + za
        @test zsum.dL_growth == FT(2) && zsum.dZ_init == FT(6)
        @test zero(za) isa P3.ZContribution{FT}
        @test isbits(za)

        # Growth tendency (Eq. 10) is linear in (dL, dN) and finite.
        coeffs = P3.reflectivity_growth_coefficients(state, shape)
        @test isbits(coeffs)
        dZ_g = P3.reflectivity_growth_tendency(coeffs, FT(1e-6), FT(1e2))
        @test isfinite(dZ_g)
        @test P3.reflectivity_growth_tendency(coeffs, FT(0), FT(0)) == FT(0)
        # Linearity check.
        dZ_a = P3.reflectivity_growth_tendency(coeffs, FT(2e-6), FT(0))
        dZ_b = P3.reflectivity_growth_tendency(coeffs, FT(1e-6), FT(0))
        @test dZ_a ≈ 2 * dZ_b rtol = 10 * eps(FT)

        # Frozen-coefficient contract: the tendency differentiates only through
        # the rates (dL, dN); the derivative is the frozen linear combination.
        a, b = FT(3e-7), FT(5e1)
        d = FD.derivative(x -> P3.reflectivity_growth_tendency(coeffs, a * x, b * x), FT(1))
        @test d ≈ coeffs.G * (2 * coeffs.M3divN * coeffs.M3divL * a - coeffs.M3divN^2 * b) rtol =
            10 * eps(FT)
        c = FT(1e-20)
        dt = FD.derivative(
            x -> P3.reflectivity_tendency(coeffs, P3.ZContribution(a * x, b * x, c * x, FT(0) * x)),
            FT(1),
        )
        @test dt ≈ d + c rtol = 10 * eps(FT)

        # Monodisperse initiation (Eq. 9), division-free and linear in dN.
        D_nuc = FT(1e-5)
        dZ_i = P3.reflectivity_initiation_monodisperse(FT(moments.μ_init), D_nuc, FT(1e3))
        @test dZ_i ≈ P3.G_of_μ(FT(moments.μ_init)) * D_nuc^6 * FT(1e3) rtol = 10 * eps(FT)
        @test P3.reflectivity_initiation_monodisperse(FT(moments.μ_init), D_nuc, FT(0)) == FT(0)

        # Drop-freezing initiation; bounded and C0 as dN → 0 (dZ → 0 with dq).
        dZ_f = P3.reflectivity_initiation_freezing(moments, params.ρ_i, FT(2), FT(1e-6), FT(1e3))
        @test isfinite(dZ_f) && dZ_f > 0
        @test P3.reflectivity_initiation_freezing(moments, params.ρ_i, FT(2), FT(0), FT(0)) == FT(0)

        # Post-pass assembly consumes the accumulator terms.
        zc = P3.ZContribution{FT}(FT(1e-6), FT(1e2), dZ_i, FT(0))
        dρz = P3.reflectivity_tendency(coeffs, zc)
        @test dρz ≈ P3.reflectivity_growth_tendency(coeffs, FT(1e-6), FT(1e2)) + dZ_i rtol = 10 * eps(FT)

        # G(μ) identity anchors.
        @test P3.G_of_μ(FT(0)) ≈ FT(20)
        @test P3.G_of_μ(FT(20)) ≈ FT(15600 / 10626)
    end
end

function test_3m_velocity(FT)
    @testset "Reflectivity-weighted terminal velocity" begin
        params = CMP.ParametersP3(FT; moments = :three_moment)
        Chen = CMP.Chen2022VelType(FT)
        ρa = FT(1.0)
        N = FT(1e5)
        ref = P3.GaussLegendre(FT, 96)
        quad6 = P3.GaussLegendre(FT, 6)
        worst = FT(0)
        worst6 = FT(0)
        for μt in FT.((0.0, 5.0, 18.0)), λt in FT.((5e2, 1e3, 1e4, 1e5)),
            (Fr, ρr) in ((FT(0), FT(0)), (FT(0.9), FT(900)))

            (; state) = _state_from_shape(params, μt, λt, Fr, ρr, N)
            shape = P3.get_distribution_shape(state)
            v_ref = P3.ice_terminal_velocity_reflectivity_weighted(Chen, ρa, state, shape; quad = ref)
            v = P3.ice_terminal_velocity_reflectivity_weighted(Chen, ρa, state, shape)
            @test isfinite(v) && v ≥ 0
            (isfinite(v_ref) && v_ref > 0) || continue
            worst = max(worst, abs(v - v_ref) / v_ref)
            v6 = P3.ice_terminal_velocity_reflectivity_weighted(Chen, ρa, state, shape; quad = quad6)
            worst6 = max(worst6, abs(v6 - v_ref) / v_ref)
            # V_z weights the largest particles, so it is not below the
            # number-weighted mean fall speed.
            v_n = P3.ice_terminal_velocity_number_weighted(Chen, ρa, state, shape; quad = ref)
            @test v_ref ≥ v_n * (1 - 10 * sqrt(eps(FT)))
        end
        # Production order accuracy, and the insufficiency of the mass/number
        # default order 6, per the reflectivity level of
        # test/p3_quadrature_error_study.jl.
        @test worst < (FT === Float32 ? FT(1e-4) : FT(1e-8))
        @test worst6 > FT(1e-5)
    end
end

function test_3m_corners(FT)
    @testset "No NaN/Inf on corner states" begin
        params = CMP.ParametersP3(FT; moments = :three_moment)
        Chen = CMP.Chen2022VelType(FT)
        ρa = FT(1.0)
        Ls = FT.((0, 1e-12, 1e-6, 1e-3))
        Ns = FT.((0, 1e2, 1e5, 1e8))
        Zs = FT.((0, 1e-30, 1e-15, 1e-3, 1e30))
        rimes = ((FT(0), FT(0)), (FT(0.5), FT(500)), (FT(0.99), FT(900)))
        for L in Ls, N in Ns, Z in Zs, (Fr, ρr) in rimes
            st = P3.P3State(params, L, N, Fr, ρr, Z)
            @test isfinite(st.ρz_ice)
            shape = P3.get_distribution_shape(st)
            @test isfinite(shape.μ) && isfinite(shape.logλ)
            v = P3.ice_terminal_velocity_reflectivity_weighted(Chen, ρa, st, shape)
            @test isfinite(v)
            coeffs = P3.reflectivity_growth_coefficients(st, shape)
            dZ = P3.reflectivity_growth_tendency(coeffs, FT(1e-6), FT(1e2))
            @test isfinite(dZ)
        end
    end
end

function test_3m_onset_decoupled(FT)
    @testset "Decoupled onset sweeps (ρn → 0 at fixed ρz, ρz → 0 at fixed ρn)" begin
        params = CMP.ParametersP3(FT; moments = :three_moment)
        Chen = CMP.Chen2022VelType(FT)
        ρa = FT(1.0)
        (; zn_lo, zn_hi) = params.moments
        L = FT(1e-6)
        Z_sat = FT(1e10)  # above the window for every swept ρn

        # ρn → 0⁺ with a saturated sixth moment: the stored ρz pins to the
        # window's upper edge and the ratio stays at zn_hi; at exactly ρn = 0
        # the stored ρz collapses to zero and the ratio evaluates at zn_lo. The
        # shape-only V_z changes discontinuously there, but the sedimentation
        # flux V_z·ρz vanishes continuously with ρn.
        for ρn in FT.((1e5, 1e2, 1e-2, 1e-10, 1e-30))
            st = P3.P3State(params, L, ρn, FT(0.5), FT(500), Z_sat)
            @test P3.reflectivity_number_ratio(st) ≈ zn_hi rtol = 10 * eps(FT)
            shape = P3.get_distribution_shape(st)
            v = P3.ice_terminal_velocity_reflectivity_weighted(Chen, ρa, st, shape)
            @test isfinite(v) && FT(0) ≤ v ≤ params.v_term_ice_max
            # flux → 0 with ρn
            @test v * st.ρz_ice ≤ params.v_term_ice_max * zn_hi * ρn * (1 + 10 * eps(FT))
        end
        st0 = P3.P3State(params, L, FT(0), FT(0.5), FT(500), Z_sat)
        @test st0.ρz_ice == FT(0)
        @test P3.reflectivity_number_ratio(st0) == zn_lo
        shape0 = P3.get_distribution_shape(st0)
        v0 = P3.ice_terminal_velocity_reflectivity_weighted(Chen, ρa, st0, shape0)
        @test isfinite(v0) && FT(0) ≤ v0 ≤ params.v_term_ice_max

        # ρz → 0 at fixed ρn: the ratio decreases continuously onto zn_lo, and
        # the shape map has no jump across the window's lower edge.
        ρn = FT(1e5)
        for Z in (zn_hi * ρn) .* FT.((1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 0))
            st = P3.P3State(params, L, ρn, FT(0.5), FT(500), Z)
            r = P3.reflectivity_number_ratio(st)
            @test zn_lo ≤ r ≤ zn_hi
            shape = P3.get_distribution_shape(st)
            @test isfinite(shape.μ) && isfinite(shape.logλ)
        end
        st_lo = P3.P3State(params, L, ρn, FT(0.5), FT(500), zn_lo * ρn)
        st_zero = P3.P3State(params, L, ρn, FT(0.5), FT(500), FT(0))
        sh_lo = P3.get_distribution_shape(st_lo)
        sh_zero = P3.get_distribution_shape(st_zero)
        @test sh_zero.μ ≈ sh_lo.μ atol = sqrt(eps(FT))
        @test sh_zero.logλ ≈ sh_lo.logλ atol = sqrt(eps(FT))
    end
end

@testset "P3 three-moment tests ($FT)" for FT in (Float64, Float32)
    test_3m_monotonicity(FT)
    test_3m_solve_accuracy(FT)
    test_3m_roundtrip(FT)
    test_3m_rails(FT)
    test_3m_onset(FT)
    test_3m_onset_decoupled(FT)
    test_3m_z_bounds(FT)
    test_3m_tendencies(FT)
    test_3m_velocity(FT)
    test_3m_corners(FT)
end
nothing
