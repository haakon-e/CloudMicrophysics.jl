#####
##### F0 spike driver: parity / alloc / inference / timing harness.
##### Run: JULIA_NUM_THREADS=8 julia +release --project=test test/spike_run.jl
#####

import ClimaParams as CP
import StaticArrays as SA
import ForwardDiff as FD
import CloudMicrophysics as CM
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.ThermodynamicsInterface as TDI
using BenchmarkTools: @belapsed
using JET: JET
using Test

include(joinpath(@__DIR__, "spike_microstate.jl"))

const TEND_KEYS = (
    :dq_lcl_dt, :dn_lcl_dt, :dq_rai_dt, :dn_rai_dt,
    :dq_ice_dt, :dn_ice_dt, :dq_rim_dt, :db_rim_dt, :dn_lcl_activation_dt,
)

# Strict bit equality of two 9-field tendency NamedTuples.
function bit_identical(a, b)
    for k in TEND_KEYS
        va, vb = a[k], b[k]
        # === is bit-exact for Float (distinguishes -0.0/0.0, equal NaN bits)
        va === vb || return (false, k, va, vb)
    end
    return (true, :none, nothing, nothing)
end

function max_ulp(a, b)
    m = 0.0
    for k in TEND_KEYS
        va, vb = a[k], b[k]
        (isnan(va) && isnan(vb)) && continue
        d = abs(Float64(va) - Float64(vb))
        s = max(abs(Float64(va)), abs(Float64(vb)))
        m = max(m, s == 0 ? d : d / (s * eps(Float64)))
    end
    return m
end

# Validated states drawn from test/rosenbrock_mode_tests.jl.
function regimes(::Type{FT}) where {FT}
    return (
        (; name = "warm rain", ρ = FT(1.05), T = FT(288), q_tot = FT(0.015),
            x = (FT(4e-4), FT(8e7), FT(2.1e-3), FT(5e4), FT(0), FT(0), FT(0), FT(0)), logλ = FT(-Inf)),
        (; name = "mixed phase", ρ = FT(0.78), T = FT(273.5), q_tot = FT(0.009),
            x = (FT(2e-4), FT(5e7), FT(1e-4), FT(4e4), FT(1e-4), FT(2e5), FT(4e-5), FT(6e-8)), logλ = nothing),
        (; name = "ice sublimation", ρ = FT(0.45), T = FT(253), q_tot = FT(4e-4),
            x = (FT(0), FT(0), FT(0), FT(0), FT(8e-4), FT(5e5), FT(5e-4), FT(9e-7)), logλ = nothing),
        (; name = "cond-freeze stress", ρ = FT(0.45), T = FT(233), q_tot = FT(0.003),
            x = (FT(1e-6), FT(1e6), FT(1e-12), FT(1e-2), FT(8e-4), FT(5e5), FT(5e-4), FT(9e-7)), logλ = nothing),
        (; name = "near-empty band + ice", ρ = FT(0.45), T = FT(253), q_tot = FT(4e-4),
            x = (FT(1e-13), FT(1e2), FT(0), FT(0), FT(8e-4), FT(5e5), FT(5e-4), FT(9e-7)), logλ = nothing),
    )
end

function consistent_logλ(p3, ρ, x)
    st = P3.state_from_prognostic(p3, ρ * x[5], ρ * x[6], ρ * x[7], ρ * x[8])
    return P3.get_distribution_logλ(st)
end

function real_call(mode, mp, tps, r, logλ, Δt, nsub)
    return BMT.bulk_microphysics_tendencies(
        mode, BMT.Microphysics2Moment(), mp, tps,
        r.ρ, r.T, r.q_tot, r.x..., logλ, Δt, nsub,
    )
end

function run_parity(::Type{FT}) where {FT}
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true)
    p3 = mp.ice.scheme
    Δt = FT(30)

    modes = (
        (; label = "rosenbrock_exact", mode = BMT.rosenbrock_exact(), masked = false),
        (; label = "rosenbrock_manual", mode = BMT.rosenbrock_manual(), masked = false),
        # ExactJacobian + ImplicitGrowth: the only mode using the field-access
        # species mask, so it exercises spike_species_mask / accessors.
        (; label = "exact+implicit(masked)",
            mode = BMT.RosenbrockAverage(BMT.ExactJacobian(), BMT.ImplicitGrowth(), BMT.NoLimiter()),
            masked = true),
    )

    println("\n================ PARITY  FT = $FT ================")
    allok = true
    ncmp = 0
    nskip = 0
    worst = 0.0
    for m in modes
        for nsub in (1, 4, 16)
            for r in regimes(FT)
                logλ = isnothing(r.logλ) ? consistent_logλ(p3, r.ρ, r.x) : r.logλ
                ref = try
                    real_call(m.mode, mp, tps, r, logλ, Δt, nsub)
                catch e
                    # reference itself hit a thermo domain error at this state;
                    # require the spike to fail the same way, then skip.
                    spikefail = try
                        Spike.spike_driver(m.mode, mp, tps, r.ρ, r.T, r.q_tot, r.x, logλ, Δt, nsub; masked = m.masked)
                        false
                    catch
                        true
                    end
                    spikefail || (allok = false; @warn "reference threw but spike succeeded" mode = m.label r.name)
                    nskip += 1
                    continue
                end
                got = Spike.spike_driver(m.mode, mp, tps, r.ρ, r.T, r.q_tot, r.x, logλ, Δt, nsub; masked = m.masked)
                (ok, k, va, vb) = bit_identical(ref, got)
                u = max_ulp(ref, got)
                worst = max(worst, u)
                ncmp += 1
                if !ok
                    allok = false
                    @warn "MISMATCH" mode = m.label nsub r.name field = k ref = va got = vb ulp = u
                end
            end
        end
    end
    println("  comparisons: $ncmp   skipped (ref threw): $nskip")
    println("  bit-identical over all compared modes/regimes/nsub: ", allok)
    println("  worst-case ulp gap: ", worst)
    return allok
end

function run_alloc(::Type{FT}) where {FT}
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true)
    p3 = mp.ice.scheme
    Δt = FT(30)
    r = regimes(FT)[2]  # mixed phase, all species active
    logλ = consistent_logλ(p3, r.ρ, r.x)

    println("\n================ ALLOCATIONS  FT = $FT ================")
    for (label, mode, masked) in (
        ("real  exact ", BMT.rosenbrock_exact(), nothing),
        ("real  manual", BMT.rosenbrock_manual(), nothing),
        ("spike exact ", BMT.rosenbrock_exact(), false),
        ("spike manual", BMT.rosenbrock_manual(), true),
    )
        if isnothing(masked)
            f() = real_call(mode, mp, tps, r, logλ, Δt, 4)
            f()  # warmup
            a = @allocated f()
        else
            g() = Spike.spike_driver(mode, mp, tps, r.ρ, r.T, r.q_tot, r.x, logλ, Δt, 4; masked = masked)
            g()  # warmup
            a = @allocated g()
        end
        println("  $label : $a bytes")
    end
end

function run_timing(::Type{FT}) where {FT}
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true)
    p3 = mp.ice.scheme
    Δt = FT(30)
    r = regimes(FT)[2]
    logλ = consistent_logλ(p3, r.ρ, r.x)

    ρ, T, q_tot, x = r.ρ, r.T, r.q_tot, r.x
    println("\n================ TIMING (ns)  FT = $FT ================")
    for (label, jac, masked) in (
        ("exact", BMT.rosenbrock_exact(), false),
        ("manual", BMT.rosenbrock_manual(), true),
    )
        tr = @belapsed real_call($jac, $mp, $tps, $r, $logλ, $Δt, 4)
        ts = @belapsed Spike.spike_driver($jac, $mp, $tps, $ρ, $T, $q_tot, $x, $logλ, $Δt, 4; masked = $masked)
        println("  $label : real = $(round(tr*1e9, digits=1)) ns   spike = $(round(ts*1e9, digits=1)) ns   ratio = $(round(ts/tr, digits=3))")
    end
end

function run_inference(::Type{FT}) where {FT}
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true)
    p3 = mp.ice.scheme
    Δt = FT(30)
    r = regimes(FT)[2]
    logλ = consistent_logλ(p3, r.ρ, r.x)

    println("\n================ INFERENCE  FT = $FT ================")
    for (label, mode, masked) in (
        ("spike exact ", BMT.rosenbrock_exact(), false),
        ("spike manual", BMT.rosenbrock_manual(), true),
    )
        println("  --- @report_opt $label (spike) ---")
        JET.@report_opt Spike.spike_driver(mode, mp, tps, r.ρ, r.T, r.q_tot, r.x, logλ, Δt, 4; masked = masked)
    end
    println("  --- @report_opt real exact (baseline) ---")
    JET.@report_opt real_call(BMT.rosenbrock_exact(), mp, tps, r, logλ, Δt, 4)

    # @test_opt gate on the spike hot path
    @testset "spike @test_opt FT=$FT" begin
        JET.@test_opt Spike.spike_driver(BMT.rosenbrock_exact(), mp, tps, r.ρ, r.T, r.q_tot, r.x, logλ, Δt, 4; masked = false)
        JET.@test_opt Spike.spike_driver(BMT.rosenbrock_manual(), mp, tps, r.ρ, r.T, r.q_tot, r.x, logλ, Δt, 4; masked = true)
    end

    # inferred return type of the driver
    rt = Base.return_types(Spike.spike_driver,
        (typeof(BMT.rosenbrock_exact()), typeof(mp), typeof(tps), FT, FT, FT, NTuple{8, FT}, FT, FT, Int))
    println("  inferred return type (spike_driver, exact): ", rt)
end

# Minimal feature-on instantiations: construct, index, cat_view, and pass through
# the generic _rosenbrock_system SMatrix path with a dummy Jacobian.
function run_feature_instantiations(::Type{FT}) where {FT}
    println("\n================ FEATURE INSTANTIATIONS  FT = $FT ================")

    function exercise(NCAT, LIQ, ZM)
        N = Spike.state_length(NCAT, LIQ, ZM)
        data = ntuple(i -> FT(i) * FT(1e-4), N)
        x = Spike.MicroState{FT, NCAT, LIQ, ZM, N}(data)
        # index round-trip
        @assert Tuple(x) === data
        @assert x[1] == data[1] && x[N] == data[N]
        # accessors on category 1
        _ = Spike.ice_q(x, Val(1))
        _ = Spike.ice_n(x, Val(1))
        cv = Spike.cat_view(x, Val(1))
        # dummy tendency + dummy diagonally-dominant Jacobian
        f = x
        J = SA.SMatrix{N, N, FT}(ntuple(k -> (((k - 1) % N + 1) == ((k - 1) ÷ N + 1) ? FT(-1) : FT(0)), Val(N * N)))
        z = ones(SA.SVector{N, FT})
        h = FT(1)
        S, S⁻¹, A = BMT._rosenbrock_system(x, f, J, z, h)
        Δx = BMT._rosenbrock_update(x, f, J, z, h)
        # inference check
        rt = Base.return_types(BMT._rosenbrock_update, (typeof(x), typeof(f), typeof(J), typeof(z), typeof(h)))
        infok = isconcretetype(only(rt))
        println("  NCAT=$NCAT LIQ=$LIQ ZM=$ZM => N=$N  cat_view keys=$(keys(cv))")
        println("      update type = $(typeof(Δx))")
        println("      _rosenbrock_update return concrete? $infok  ($(only(rt)))")
        # verify the update stayed a MicroState of the right layout
        @assert Δx isa Spike.MicroState{FT, NCAT, LIQ, ZM, N} "update lost layout type: $(typeof(Δx))"
        return N
    end

    exercise(1, false, false)   # N = 8  (today)
    exercise(1, true, false)    # N = 9  (liquid fraction)
    exercise(1, false, true)    # N = 9  (three-moment)
    exercise(1, true, true)     # N = 10 (both)
    exercise(2, false, false)   # N = 12 (two categories)
    exercise(2, true, true)     # N = 16 (two categories, both features)

    # similar_type throws on an inconsistent length
    threw = false
    try
        SA.similar_type(Spike.MicroState{FT, 1, false, false, 8}, FT, SA.Size((7,)))
    catch e
        threw = e isa DimensionMismatch
    end
    println("  similar_type throws on inconsistent length: ", threw)

    # constructor throws on inconsistent N
    threw2 = false
    try
        Spike.MicroState{FT, 1, false, false, 7}(ntuple(i -> FT(i), 7))
    catch e
        threw2 = e isa DimensionMismatch
    end
    println("  constructor throws on inconsistent N: ", threw2)
end

function main()
    ok64 = run_parity(Float64)
    ok32 = run_parity(Float32)
    run_alloc(Float64)
    run_alloc(Float32)
    run_inference(Float64)
    run_inference(Float32)
    run_feature_instantiations(Float64)
    run_feature_instantiations(Float32)
    run_timing(Float64)
    run_timing(Float32)
    println("\n================ SUMMARY ================")
    println("  parity Float64: ", ok64)
    println("  parity Float32: ", ok32)
end

main()
