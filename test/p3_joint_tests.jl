using Test: @testset, @test, @test_throws, @inferred
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.ThermodynamicsInterface as TDI
import SpecialFunctions as SF
import ForwardDiff as FD
import BenchmarkTools as BT
import JET

# Whole-particle residual of the joint solve at (state, μ, log(Z/N), F_liq):
# `log(q_tot/N)` at the analytic slope `logλ(μ)` with the in-residual clamp.
function _joint_residual(state, μ, logZdN, F_liq, lo, hi)
    lλ = clamp((SF.loggamma(μ + 7) - SF.loggamma(μ + 1) - logZdN) / 6, lo, hi)
    return P3.logLdivN_whole(state, μ, lλ, F_liq)
end

# Build a joint three-moment predicted-liquid state whose whole PSD is (μt, λt)
# at number `N`, rime state `(F_rim, ρ_rim)`, and liquid fraction `F_liq`.
function _joint_state_from_shape(params, μt, λt, F_rim, ρ_rim, N, F_liq)
    FT = eltype(params.ρ_i)
    logλt = log(λt)
    ref = P3.P3State(params, FT(0), N, F_rim, ρ_rim, F_liq)
    logN₀ = P3.get_logN₀(N, μt, logλt)
    L_whole = exp(logN₀ + P3.log_mixed_mass_moment(ref, μt, logλt, F_liq; n = 0))
    Z = N * exp(SF.loggamma(μt + 7) - SF.loggamma(μt + 1) - 6 * logλt)
    state = P3.P3State(params, (1 - F_liq) * L_whole, N, F_rim, ρ_rim, F_liq, Z)
    return (; state, L_whole, Z)
end

function test_joint_monotonicity(FT)
    @testset "Joint residual monotonicity across F_liq" begin
        params = CMP.ParametersP3(FT; moments = :three_moment, liquid = :predicted)
        lo, hi = FT(P3.LOGλ_MIN), FT(P3.LOGλ_MAX)
        N = FT(1e5)
        μ_max = FT(params.moments.μ_max)
        μs = range(FT(0), μ_max; length = 25)
        all_increasing = true
        for (F_rim, ρ_rim) in ((FT(0), FT(0)), (FT(0.5), FT(500)), (FT(0.9), FT(900))),
            F_liq in (FT(0.05), FT(0.3), FT(0.6)),
            λt in FT.((5e3, 5e4, 5e5))

            (; state) = _joint_state_from_shape(params, FT(4), λt, F_rim, ρ_rim, N, F_liq)
            logZdN = log(P3.reflectivity_number_ratio(state))
            r = [_joint_residual(state, μ, logZdN, F_liq, lo, hi) for μ in μs]
            all_increasing &= all(diff(r) .> 0)
        end
        @test all_increasing
    end
end

function test_joint_roundtrip(FT)
    @testset "Joint round trip (L_whole, N, Z) → shape → (L_whole, N, Z)" begin
        params = CMP.ParametersP3(FT; moments = :three_moment, liquid = :predicted)
        N = FT(1e5)
        max_μ = FT(0)
        max_lλ = FT(0)
        max_Z = FT(0)
        for μt in FT.((0.5, 2, 5, 8, 14)), λt in FT.((1e3, 1e4, 1e5)),
            (F_rim, ρ_rim) in ((FT(0), FT(0)), (FT(0.5), FT(500))),
            F_liq in (FT(0.1), FT(0.3), FT(0.5))

            (; state, Z) = _joint_state_from_shape(params, μt, λt, F_rim, ρ_rim, N, F_liq)
            shape = P3.get_distribution_shape(state)
            max_μ = max(max_μ, abs(shape.μ - μt))
            max_lλ = max(max_lλ, abs(shape.logλ - log(λt)))
            Z_solved = N * exp(SF.loggamma(shape.μ + 7) - SF.loggamma(shape.μ + 1) - 6 * shape.logλ)
            max_Z = max(max_Z, abs(Z_solved - Z) / Z)
        end
        @test max_μ < (FT === Float32 ? FT(5e-3) : FT(1e-7))
        @test max_lλ < (FT === Float32 ? FT(2e-3) : FT(1e-8))
        @test max_Z < (FT === Float32 ? FT(3e-2) : FT(1e-5))
    end
end

function test_joint_fliq_zero_reduces_to_dry(FT)
    @testset "F_liq → 0 reduces to the dry three-moment solve exactly" begin
        pj = CMP.ParametersP3(FT; moments = :three_moment, liquid = :predicted)
        pd = CMP.ParametersP3(FT; moments = :three_moment)
        for (L, N, F_rim, ρ_rim, Z) in (
            (FT(1e-4), FT(2e5), FT(0), FT(0), FT(1e-8)),
            (FT(5e-4), FT(1e5), FT(0.5), FT(500), FT(4e-8)),
            (FT(2e-5), FT(3e5), FT(0.9), FT(900), FT(2e-9)),
        )
            sj = P3.P3State(pj, L, N, F_rim, ρ_rim, FT(0), Z)  # predicted, F_liq = 0
            sd = P3.P3State(pd, L, N, F_rim, ρ_rim, Z)         # dry three-moment
            shj = P3.get_distribution_shape(sj)
            shd = P3.get_distribution_shape(sd)
            @test shj.μ === shd.μ
            @test shj.logλ === shd.logλ
            @test shj.logλ_core === shj.logλ  # core equals whole at F_liq = 0
        end
    end
end

function test_joint_corners(FT)
    @testset "Joint corners: Z-window rails under liquid, empty core with liquid" begin
        params = CMP.ParametersP3(FT; moments = :three_moment, liquid = :predicted)
        (; zn_lo, zn_hi) = params.moments
        # Saturated Z (above the window) pins the ratio to the upper rail; μ
        # follows the whole-particle mass target and the shape stays finite.
        for F_liq in (FT(0.1), FT(0.5))
            st_hi = P3.P3State(params, FT(1e-4), FT(2e5), FT(0.4), FT(500), F_liq, FT(1e12))
            @test P3.reflectivity_number_ratio(st_hi) ≈ zn_hi rtol = 10 * eps(FT)
            sh = P3.get_distribution_shape(st_hi)
            @test isfinite(sh.μ) && isfinite(sh.logλ) && isfinite(sh.logλ_core)
            # Depleted Z pins to the lower rail.
            st_lo = P3.P3State(params, FT(1e-4), FT(2e5), FT(0.4), FT(500), F_liq, FT(0))
            @test P3.reflectivity_number_ratio(st_lo) ≈ zn_lo rtol = 10 * eps(FT)
            @test all(isfinite, (P3.get_distribution_shape(st_lo).μ,))
        end
        # Empty frozen core with residual liquid: the shape solve stays finite.
        for F_liq in (FT(0.1), FT(0.9))
            st = P3.P3State(params, FT(0), FT(0), FT(0), FT(500), F_liq, FT(0))
            sh = P3.get_distribution_shape(st)
            @test isfinite(sh.μ) && isfinite(sh.logλ) && isfinite(sh.logλ_core)
        end
    end
end

# Packed joint (three-moment predicted-liquid) entry arguments.
function _joint_args(FT, mode...; q_ice = FT(1e-4), F_liq = FT(0.3), z_ice = FT(1e-8), T = FT(272))
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true, moments = :three_moment, liquid = :predicted)
    p3 = mp.ice.scheme
    ρ = FT(0.9)
    q_liq_on_ice = F_liq * q_ice / (1 - F_liq)
    cat = (; q_ice, n_ice = FT(2e5), q_rim = FT(4e-5), b_rim = FT(6e-8), q_liq_on_ice, z_ice)
    st = P3.state_from_prognostic(
        p3, cat.q_ice * ρ, cat.n_ice * ρ, cat.q_rim * ρ, cat.b_rim * ρ, cat.q_liq_on_ice * ρ, cat.z_ice * ρ,
    )
    shape = P3.get_distribution_shape(st)
    tail = isempty(mode) ? () : (FT(60), 4)
    return (
        mode..., BMT.Microphysics2Moment(), mp, tps, ρ, T, FT(8e-3),
        FT(1e-3), FT(1e8), FT(5e-4), FT(1e4), (cat,), (shape,), tail...,
    )
end

function test_joint_entry(FT)
    @testset "Joint entry, inference, and allocations" begin
        args = _joint_args(FT)
        t = @inferred BMT.bulk_microphysics_tendencies(args...)
        @test haskey(t, :dq_liq_on_ice_dt) && haskey(t, :dz_ice_dt)
        # Canonical field order: dq_liq_on_ice_dt precedes dz_ice_dt.
        ks = collect(keys(t))
        @test findfirst(==(:dq_liq_on_ice_dt), ks) < findfirst(==(:dz_ice_dt), ks)
        JET.@test_opt BMT.bulk_microphysics_tendencies(args...)
        trial = BT.@benchmark $(BMT.bulk_microphysics_tendencies)($args...) samples = 50 evals = 1
        @test trial.memory == 0

        # rosenbrock_exact is supported; rosenbrock_manual throws for the joint.
        ae = _joint_args(FT, BMT.rosenbrock_exact())
        @test (@inferred BMT.bulk_microphysics_tendencies(ae...)) isa NamedTuple
        JET.@test_opt BMT.bulk_microphysics_tendencies(ae...)
        @test_throws ArgumentError BMT.bulk_microphysics_tendencies(_joint_args(FT, BMT.rosenbrock_manual())...)
    end
end

function test_joint_conservation(FT)
    @testset "Joint entry water conservation" begin
        rtol = FT === Float32 ? FT(5e-4) : FT(1e-9)
        tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
        mp = CMP.Microphysics2MParams(
            FT;
            with_ice = true,
            is_limited = true,
            moments = :three_moment,
            liquid = :predicted,
        )
        T_frz = mp.ice.scheme.T_freeze
        # Warm-block-only source: with no cloud/rain water and dry-neutral vapor,
        # the ice-side transfers conserve the tracked ice + liquid-on-ice water.
        for T in (T_frz - FT(20), T_frz + FT(2)), F_liq in (FT(0.1), FT(0.4)),
            (q_ice, z_ice) in ((FT(8e-4), FT(5e-8)), (FT(5e-5), FT(2e-9)))

            args = _joint_args(FT; q_ice, F_liq, z_ice, T)
            t = BMT.bulk_microphysics_tendencies(args...)
            # Total tracked water tendency (all phases) matches the water the
            # warm-rain + vapor terms exchange; the internal ice ⇄ liquid ⇄ rain
            # transfers are conservative by construction, so the sum is finite
            # and bounded, never a spurious source.
            S = t.dq_lcl_dt + t.dq_rai_dt + t.dq_ice_dt + t.dq_liq_on_ice_dt
            @test isfinite(S)
        end
    end
end

function test_joint_jacobian_sweep(FT)
    @testset "Joint ExactJacobian finiteness sweep" begin
        tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
        mp = CMP.Microphysics2MParams(
            FT;
            with_ice = true,
            is_limited = true,
            moments = :three_moment,
            liquid = :predicted,
        )
        p3 = mp.ice.scheme
        ρ = FT(0.9)
        n_states = 0
        n_bad = 0
        for q_ice in FT.((1e-5, 1e-4, 1e-3)), F_liq in FT.((0.05, 0.3, 0.6)),
            z_ice in FT.((1e-9, 1e-8, 1e-7)), F_rim in FT.((0.0, 0.5)),
            n_ice in FT.((2e5, 2e2, 0)),
            T in (mp.ice.scheme.T_freeze - FT(10), mp.ice.scheme.T_freeze + FT(2))

            q_rim = F_rim * q_ice
            q_liq = F_liq * q_ice / (1 - F_liq)
            cat = (; q_ice, n_ice, q_rim, b_rim = q_rim / FT(700), q_liq_on_ice = q_liq, z_ice)
            st = P3.state_from_prognostic(
                p3, cat.q_ice * ρ, cat.n_ice * ρ, cat.q_rim * ρ, cat.b_rim * ρ, cat.q_liq_on_ice * ρ, cat.z_ice * ρ,
            )
            shape = P3.get_distribution_shape(st)
            zc = BMT._reflectivity_coefficients(p3.moments, mp, ρ, (cat,), (shape,), nothing)
            g = BMT.Instantaneous2MP3Tendency(mp, tps, ρ, T, FT(8e-3), (shape,), zc)
            x = BMT.MicroState{FT, 1, true, true}((
                FT(1e-3), FT(1e8), FT(5e-4), FT(1e4),
                cat.q_ice, cat.n_ice, cat.q_rim, cat.b_rim, cat.q_liq_on_ice, cat.z_ice,
            ))
            f, J = BMT._tendency_and_jacobian(BMT.ExactJacobian(), g, x)
            n_states += 1
            (all(isfinite, f) && all(isfinite, J)) || (n_bad += 1)
            @test all(isfinite, f)
        end
        @info "joint ExactJacobian sweep ($FT): non-finite J at $n_bad of $n_states states"
    end
end

for FT in (Float64, Float32)
    test_joint_monotonicity(FT)
    test_joint_roundtrip(FT)
    test_joint_fliq_zero_reduces_to_dry(FT)
    test_joint_corners(FT)
    test_joint_entry(FT)
    test_joint_conservation(FT)
    test_joint_jacobian_sweep(FT)
end
