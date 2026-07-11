using Test
import Random
import Statistics: quantile
import ForwardDiff as FD
import Adapt
using KernelAbstractions

import CloudMicrophysics as CM
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.ThermodynamicsInterface as TDI

include("p3_quadrature_error_study.jl")  # generate_column_states, hail_core_states

const FLOOR = 1e-12
relerr(a, b) = abs(a - b) / max(abs(a), abs(b), FLOOR)

# Physically realizable states for the interpolation sweep. `logλ` is drawn on
# the shape-solver bracket, `F_rim` across the full tabulated range including the
# `F_rim → 1` corner, `ρ_rim` above a rime-density floor (rimed particles are
# denser than that), and each `logλ` is mapped to the consistent `x_ice`.
function sweep_states(::Type{FT}, params, n; seed = 0xBEEF) where {FT}
    rng = Random.MersenneTwister(seed)
    r_lo, r_hi = FT(100), FT(0.8) * params.ρ_l
    F_hi = one(FT) - eps(FT)
    states = Tuple{P3.P3State{FT}, FT, FT}[]
    while length(states) < n
        logλ = FT(3 + rand(rng) * 10)
        F_rim = FT(rand(rng)) * F_hi
        ρ_rim = r_lo + rand(rng) * (r_hi - r_lo)
        ρₐ = FT(exp10(log10(0.1) + rand(rng) * (log10(1.4) - log10(0.1))))
        x = exp(P3.logLdivN(P3.P3State(params, one(FT), one(FT), F_rim, ρ_rim), logλ))
        (isfinite(x) && x > 0) || continue
        state = P3.P3State(params, x * FT(1e5), FT(1e5), F_rim, ρ_rim)
        push!(states, (state, logλ, ρₐ))
    end
    return states
end

function harness_states(::Type{FT}, params) where {FT}
    out = Tuple{P3.P3State{FT}, FT, FT}[]
    for s in vcat(generate_column_states(FT), hail_core_states(FT, STUDY_HAIL_CORES))
        (s.q_ice > 0 && s.n_ice > 0) || continue
        state = P3.state_from_prognostic(params, s.ρ * s.q_ice, s.ρ * s.n_ice, s.ρ * s.q_rim, s.ρ * s.b_rim)
        push!(out, (state, P3.get_distribution_logλ(state), FT(s.ρ)))
    end
    return out
end

@testset "P3 lookup tables" begin
    FT = Float64
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, quad = CM.Quadrature.GaussLegendre(FT, 12))
    params, vel, aps = mp.ice.scheme, mp.ice.terminal_velocity, mp.warm_rain.air_properties
    quad = mp.ice.quad
    qref = CM.Quadrature.GaussLegendre(FT, 128)

    grid = P3.P3TableGrid{FT}()
    tables = P3.build_p3_lookup_tables(params, vel, aps; grid)

    @testset "axes and fractional index" begin
        ax = P3.LinAxis(; lo = 0.0, hi = 10.0, n = 11)
        @test P3.fractional_index(ax, 0.0) == (1, 0.0)
        @test P3.fractional_index(ax, 3.5) == (4, 0.5)
        @test P3.fractional_index(ax, 10.0) == (10, 1.0)
        @test P3.fractional_index(ax, -5.0) == (1, 0.0)   # clamp low
        @test P3.fractional_index(ax, 99.0) == (10, 1.0)  # clamp high
        lax = P3.LogAxis(; lo = 1e-3, hi = 1e3, n = 7)
        i, w = P3.fractional_index(lax, 1.0)
        @test (i, w) == (4, 0.0)
        for ix in 1:11
            @test P3.node_coord(ax, ix) ≈ FT(ix - 1)
        end
        @test P3.node_coord(lax, 1) ≈ 1e-3
        @test P3.node_coord(lax, 7) ≈ 1e3
    end

    @testset "lookup exactness at nodes and inference" begin
        axλ, axF, axr, axa = tables.rates.axes
        for (i, j, k, l) in ((1, 1, 1, 1), (7, 3, 4, 2), (axλ.n, axF.n, axr.n, axa.n))
            c = (P3.node_coord(axλ, i), P3.node_coord(axF, j), P3.node_coord(axr, k), P3.node_coord(axa, l))
            q = P3.lookup(tables.rates, c...)
            for (qi, name) in enumerate(P3.RATE_QUANTITY_NAMES)
                @test getproperty(q, name) ≈ tables.rates.data[qi, i, j, k, l] rtol = 1e-12
            end
        end
        c = (7.0, 0.3, 400.0, 0.9)
        @test (@inferred P3.lookup(tables.rates, c...)) isa NamedTuple{P3.RATE_QUANTITY_NAMES}
        lk(t, a, b, cc, d) = P3.lookup(t.rates, a, b, cc, d)
        lk(tables, c...)
        @test (@allocated lk(tables, c...)) == 0
    end

    @testset "lookup is differentiable in the coordinates" begin
        f(x) = P3.lookup(tables.rates, x, 0.3, 400.0, 0.9).v_mass
        d = FD.derivative(f, 6.0)
        fd = (f(6.0 + 1e-6) - f(6.0 - 1e-6)) / 2e-6
        @test d ≈ fd rtol = 1e-5
    end

    @testset "rate node exactness vs direct evaluation at build order" begin
        axλ, axF, axr, axa = tables.rates.axes
        maxerr = 0.0
        for (i, j, k, l) in ((5, 2, 3, 2), (20, 5, 6, 4), (60, 10, 12, 6))
            logλ = P3.node_coord(axλ, i)
            F_rim = P3.node_coord(axF, j)
            ρ_rim = P3.node_coord(axr, k)
            ρₐ = P3.node_coord(axa, l)
            x = exp(P3.logLdivN(P3.P3State(params, one(FT), one(FT), F_rim, ρ_rim), logλ))
            state = P3.P3State(params, x, one(FT), F_rim, ρ_rim)
            e = max(
                relerr(P3.ice_self_collection(tables, state, logλ, ρₐ).dNdt,
                    P3.ice_self_collection(state, logλ, vel, ρₐ; quad).dNdt),
                relerr(P3.ice_terminal_velocity_number_weighted(tables, state, logλ, ρₐ),
                    P3.ice_terminal_velocity_number_weighted(vel, ρₐ, state, logλ; quad)),
                relerr(P3.ice_terminal_velocity_mass_weighted(tables, state, logλ, ρₐ),
                    P3.ice_terminal_velocity_mass_weighted(vel, ρₐ, state, logλ; quad)),
                relerr(P3.ice_melt(tables, aps, tps, FT(280), ρₐ, state, logλ).dLdt,
                    P3.ice_melt(vel, aps, tps, FT(280), ρₐ, state, logλ; quad).dLdt),
            )
            maxerr = max(maxerr, e)
        end
        @test maxerr < 1e-10
    end

    @testset "no clamping for harness states" begin
        axλ, axF, axr, axa = tables.rates.axes
        axx = tables.shape.axes[1]
        for (state, logλ, ρₐ) in harness_states(FT, params)
            @test P3.node_coord(axλ, 1) <= logλ <= P3.node_coord(axλ, axλ.n)
            @test 0 <= state.F_rim <= P3.node_coord(axF, axF.n)
            @test P3.node_coord(axa, 1) <= ρₐ <= P3.node_coord(axa, axa.n)
            @test P3.node_coord(axx, 1) <= state.ρq_ice / state.ρn_ice <= P3.node_coord(axx, axx.n)
            # ρ_rim only for rimed states; unrimed states clamp harmlessly (the
            # rate is independent of ρ_rim at F_rim = 0).
            if state.F_rim > 0
                @test P3.node_coord(axr, 1) <= state.ρ_rim <= P3.node_coord(axr, axr.n)
            end
        end
    end

    @testset "interpolation accuracy vs GL(128)" begin
        # Multilinear interpolation is O(h) near the `μ(logλ)` clamp kinks and in
        # the `F_rim → 1` regime, where the partially-rimed size range vanishes.
        # The 95th-percentile tolerance is the accepted accuracy target of the
        # default grid; the maximum is a per-quantity bound on the `F_rim → 1`
        # corner, where the mass-weighted velocity is the worst case. See
        # docs/src/P3LookupTables.md and `test/p3_lookup_error_study.jl` for the
        # resolution sweep behind these values.
        pts = vcat(harness_states(FT, params), sweep_states(FT, params, 200))
        E = Dict(k => FT[] for k in (:selfcol, :vN, :vM, :melt))
        for (state, logλ, ρₐ) in pts
            push!(
                E[:selfcol],
                relerr(P3.ice_self_collection(tables, state, logλ, ρₐ).dNdt,
                    P3.ice_self_collection(state, logλ, vel, ρₐ; quad = qref).dNdt),
            )
            push!(
                E[:vN],
                relerr(P3.ice_terminal_velocity_number_weighted(tables, state, logλ, ρₐ),
                    P3.ice_terminal_velocity_number_weighted(vel, ρₐ, state, logλ; quad = qref)),
            )
            push!(
                E[:vM],
                relerr(P3.ice_terminal_velocity_mass_weighted(tables, state, logλ, ρₐ),
                    P3.ice_terminal_velocity_mass_weighted(vel, ρₐ, state, logλ; quad = qref)),
            )
            push!(
                E[:melt],
                relerr(P3.ice_melt(tables, aps, tps, FT(280), ρₐ, state, logλ).dLdt,
                    P3.ice_melt(vel, aps, tps, FT(280), ρₐ, state, logλ; quad = qref).dLdt),
            )
        end
        max_tol = (selfcol = 1.5e-1, vN = 1.5e-1, vM = 5e-1, melt = 1.5e-1)
        for k in keys(E)
            @test quantile(E[k], 0.95) < 4e-2
            @test maximum(E[k]) < getproperty(max_tol, k)
        end
    end

    @testset "shape (logλ) table: node exactness and interpolation" begin
        axx, axF, axr = tables.shape.axes
        for (i, j, k) in ((1, 1, 1), (20, 5, 6), (40, 12, 10), (axx.n, axF.n, axr.n))
            x_ice = P3.node_coord(axx, i)
            F_rim = P3.node_coord(axF, j)
            ρ_rim = P3.node_coord(axr, k)
            state = P3.P3State(params, x_ice, one(FT), F_rim, ρ_rim)
            @test P3.get_distribution_logλ(tables, state) ≈ tables.shape.data[1, i, j, k] rtol = 1e-12
            @test P3.get_distribution_logλ(tables, state) ≈ P3.get_distribution_logλ(state) rtol = 1e-12
        end
        # First-order interpolation across the `x_ice → logλ` fold of the
        # SlopePowerLaw shape law, so the bound is looser than the rate tables;
        # see docs/src/P3LookupTables.md.
        xlo, xhi = P3.node_coord(axx, 1), P3.node_coord(axx, axx.n)
        E = FT[]
        for (state, _, _) in sweep_states(FT, params, 400)
            xlo <= state.ρq_ice / state.ρn_ice <= xhi || continue
            push!(E, relerr(P3.get_distribution_logλ(tables, state), P3.get_distribution_logλ(state)))
        end
        @test quantile(E, 0.95) < 1e-1
        @test maximum(E) < 2.5e-1
    end

    @testset "interpolant smoothness under mesh refinement" begin
        # A continuous multilinear interpolant halves its maximum adjacent
        # sample difference when the sampling doubles; a discontinuity (e.g. a
        # non-finite node) would keep it near constant. See SCREAM's P3 test.
        axλ, axF, axr, axa = tables.rates.axes
        bounds = (
            (P3.node_coord(axλ, 1), P3.node_coord(axλ, axλ.n)),
            (P3.node_coord(axF, 1), P3.node_coord(axF, axF.n)),
            (P3.node_coord(axr, 1), P3.node_coord(axr, axr.n)),
            (P3.node_coord(axa, 1), P3.node_coord(axa, axa.n)),
        )
        base = (6.0, 0.3, 400.0, 0.9)
        maxdiff(dim, n) = begin
            lo, hi = bounds[dim]
            xs = range(lo, hi, n)
            vals = map(xs) do x
                c = ntuple(d -> d == dim ? x : base[d], 4)
                v = P3.lookup(tables.rates, c...)
                (v.log_selfcol_g, v.v_mass)
            end
            maximum(
                max(abs(vals[i + 1][1] - vals[i][1]), abs(vals[i + 1][2] - vals[i][2]))
                for i in 1:(length(vals) - 1)
            )
        end
        for dim in 1:4
            @test maxdiff(dim, 401) < 0.75 * maxdiff(dim, 201)
        end
    end

    @testset "table-backed rate calls: inference and allocations" begin
        state = P3.P3State(params, FT(1e-2), FT(1e5), FT(0.4), FT(400))
        logλ = FT(6.0)
        ρₐ = FT(0.9)
        @test (@inferred P3.ice_self_collection(tables, state, logλ, ρₐ)) isa NamedTuple
        @test (@inferred P3.ice_melt(tables, aps, tps, FT(280), ρₐ, state, logλ)) isa NamedTuple
        sc(t, s, l, r) = P3.ice_self_collection(t, s, l, r).dNdt
        vn(t, s, l, r) = P3.ice_terminal_velocity_number_weighted(t, s, l, r)
        vm(t, s, l, r) = P3.ice_terminal_velocity_mass_weighted(t, s, l, r)
        ll(t, s) = P3.get_distribution_logλ(t, s)
        melt(t, ap, tp, T, r, s, l) = P3.ice_melt(t, ap, tp, T, r, s, l).dLdt
        lf(tp, T) = TDI.Lf(tp, T)
        for f in (() -> sc(tables, state, logλ, ρₐ), () -> vn(tables, state, logλ, ρₐ),
            () -> vm(tables, state, logλ, ρₐ), () -> ll(tables, state))
            f()
            @test (@allocated f()) == 0
        end
        # `ice_melt` inherits the thermodynamics `Lf` allocation; the table path
        # adds nothing on top of it.
        melt(tables, aps, tps, FT(280), ρₐ, state, logλ)
        lf(tps, FT(280))
        @test (@allocated melt(tables, aps, tps, FT(280), ρₐ, state, logλ)) <=
              (@allocated lf(tps, FT(280)))
    end

    @testset "Float32 build and lookup" begin
        F32 = Float32
        p32 = CMP.Microphysics2MParams(F32; with_ice = true, quad = CM.Quadrature.GaussLegendre(F32, 12))
        g32 = P3.P3TableGrid{F32}(n_logλ = 24, n_F_rim = 8, n_ρ_rim = 6, n_ρ_air = 5, n_x_ice = 24)
        t32 = P3.build_p3_lookup_tables(
            p32.ice.scheme,
            p32.ice.terminal_velocity,
            p32.warm_rain.air_properties;
            grid = g32,
        )
        @test eltype(t32.rates) === F32
        @test all(isfinite, t32.rates.data)
        axλ, axF, axr, axa = t32.rates.axes
        logλ, F_rim, ρ_rim, ρₐ =
            P3.node_coord(axλ, 12), P3.node_coord(axF, 3), P3.node_coord(axr, 3), P3.node_coord(axa, 3)
        x = exp(P3.logLdivN(P3.P3State(p32.ice.scheme, one(F32), one(F32), F_rim, ρ_rim), logλ))
        state = P3.P3State(p32.ice.scheme, x, one(F32), F_rim, ρ_rim)
        tab = P3.ice_self_collection(t32, state, logλ, ρₐ).dNdt
        ref = P3.ice_self_collection(state, logλ, p32.ice.terminal_velocity, ρₐ; quad = p32.ice.quad).dNdt
        @test tab ≈ ref rtol = 1e-4
    end

    @testset "device kernel (KernelAbstractions)" begin
        # Runs on the CPU backend here, exercising the isbits/Adapt/kernel path;
        # on a CUDA-capable node the same kernel runs on the device.
        backend = CPU()
        dtables = Adapt.adapt(Array, tables)
        n = 8
        logλs = collect(range(FT(4), FT(12), n))
        Fs = fill(FT(0.3), n)
        ρrs = fill(FT(400), n)
        ρas = fill(FT(0.9), n)
        out_v = similar(logλs)
        out_g = similar(logλs)
        @kernel function lut_kernel!(out_v, out_g, tbl, logλs, Fs, ρrs, ρas)
            i = @index(Global, Linear)
            q = P3.lookup(tbl.rates, logλs[i], Fs[i], ρrs[i], ρas[i])
            out_v[i] = q.v_number
            out_g[i] = exp(q.log_selfcol_g)
        end
        lut_kernel!(backend)(out_v, out_g, dtables, logλs, Fs, ρrs, ρas; ndrange = n)
        synchronize(backend)
        for i in 1:n
            ref = P3.lookup(tables.rates, logλs[i], Fs[i], ρrs[i], ρas[i])
            @test out_v[i] ≈ ref.v_number
            @test out_g[i] ≈ exp(ref.log_selfcol_g)
        end
    end
end
nothing
