using Test
import Random
import Statistics: quantile
import Adapt

import CloudMicrophysics as CM
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.ThermodynamicsInterface as TDI
import CloudMicrophysics.Common as CO
import CloudMicrophysics.DistributionTools as DT
import CloudMicrophysics.Microphysics2M as CM2
import StaticArrays: SVector

include("p3_quadrature_error_study.jl")  # generate_column_states, hail_core_states

const CFLOOR = 1e-12
crelerr(a, b) = abs(a - b) / max(abs(a), abs(b), CFLOOR)

# Harness states carrying their liquid loadings, restricted to states with both
# ice and some liquid so the collision sources are non-trivial.
function collision_harness(::Type{FT}, params) where {FT}
    out = NamedTuple[]
    for s in vcat(generate_column_states(FT), hail_core_states(FT, STUDY_HAIL_CORES))
        (s.q_ice > 0 && s.n_ice > 0) || continue
        (s.q_lcl > 0 || s.q_rai > 0) || continue
        state = P3.state_from_prognostic(params, s.ρ * s.q_ice, s.ρ * s.n_ice, s.ρ * s.q_rim, s.ρ * s.b_rim)
        push!(
            out,
            (;
                state, logλ = P3.get_distribution_logλ(state), ρₐ = FT(s.ρ),
                L_c = FT(s.ρ * s.q_lcl), N_c = FT(s.n_lcl), L_r = FT(s.ρ * s.q_rai), N_r = FT(s.n_rai),
            ),
        )
    end
    return out
end

# Physically realizable states with randomized liquid loadings, so the `x_c` and
# `Dr_mean` axes are exercised and the 95th percentile is a statistic over the
# domain rather than the few harness points. `F_rim` is drawn up to the realizable
# `0.95` (the harness maximum); the interpolation error grows without bound as
# `F_rim → 1`, where the partially-rimed size range vanishes, and that corner is
# covered by the node-exactness test rather than by this accuracy statistic.
function collision_sweep(::Type{FT}, params, n; seed = 0xC0FFEE) where {FT}
    rng = Random.MersenneTwister(seed)
    r_lo, r_hi = FT(100), FT(0.8) * params.ρ_l
    F_hi = FT(0.95)
    out = NamedTuple[]
    while length(out) < n
        logλ = FT(3 + rand(rng) * 10)
        F_rim = FT(rand(rng)) * F_hi
        ρ_rim = r_lo + rand(rng) * (r_hi - r_lo)
        ρₐ = FT(exp10(log10(0.1) + rand(rng) * (log10(1.4) - log10(0.1))))
        x = exp(P3.logLdivN(P3.P3State(params, one(FT), one(FT), F_rim, ρ_rim), logλ))
        (isfinite(x) && x > 0) || continue
        state = P3.P3State(params, x * FT(1e5), FT(1e5), F_rim, ρ_rim)
        N_c = FT(exp10(7 + rand(rng) * 1.5))
        q_c = FT(exp10(-4 + rand(rng) * 0.7))
        N_r = FT(exp10(3.5 + rand(rng) * 1.5))
        q_r = FT(exp10(-4.3 + rand(rng) * 0.8))
        push!(out, (; state, logλ, ρₐ, L_c = ρₐ * q_c, N_c, L_r = ρₐ * q_r, N_r))
    end
    return out
end

@testset "P3 collision lookup tables" begin
    FT = Float64
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, quad = CM.Quadrature.GaussLegendre(FT, 12))
    params, vel, aps = mp.ice.scheme, mp.ice.terminal_velocity, mp.warm_rain.air_properties
    psd_c, psd_r = mp.ice.cloud_pdf, mp.ice.rain_pdf
    build_quad = CM.Quadrature.GaussLegendre(FT, 12)
    qref = CM.Quadrature.GaussLegendre(FT, 32)
    m_liq(Dₗ) = psd_c.ρw * CO.volume_sphere_D(Dₗ)

    rgrid = P3.P3TableGrid{FT}(n_logλ = 48, n_F_rim = 14, n_ρ_rim = 10, n_ρ_air = 5)
    rate_tables = P3.build_p3_lookup_tables(params, vel, aps; grid = rgrid)
    # A CI-sized grid, coarser than the default; the accuracy thresholds are its
    # interpolation error, dominated by the cloud channel in the F_rim → 1 and
    # μ(logλ) kink corners. The default grid is finer; see docs/src/P3LookupTables.md.
    cgrid = P3.P3CollisionGrid{FT}(
        n_logλ = 32, n_F_rim = 14, n_ρ_rim = 10, n_ρ_air = 5, n_x_c = 12, n_Dr = 12, build_order = 12,
    )
    ctables = P3.build_p3_collision_tables(params, vel, aps, psd_c, psd_r; grid = cgrid, quad = build_quad)

    @testset "node exactness vs direct quadrature at build order" begin
        axλ, axF, axr, axa, axxc = ctables.cloud.axes
        axDr = ctables.rain.axes[5]
        maxerr = 0.0
        for (i, j, k, l) in ((3, 2, 2, 2), (12, 5, 4, 3), (axλ.n, axF.n, axr.n, axa.n))
            logλ = P3.node_coord(axλ, i)
            F_rim = P3.node_coord(axF, j)
            ρ_rim = P3.node_coord(axr, k)
            ρₐ = P3.node_coord(axa, l)
            x_ice = exp(P3.logLdivN(P3.P3State(params, one(FT), one(FT), F_rim, ρ_rim), logλ))
            state = P3.P3State(params, x_ice, one(FT), F_rim, ρ_rim)  # unit ice number
            for m in (1, 5, axxc.n)
                x_c = P3.node_coord(axxc, m)
                qc = P3.lookup(ctables.cloud, logλ, F_rim, ρ_rim, ρₐ, x_c)
                # cloud unit-moment reference: full collision at cold T, unit prefactors
                rc = P3.∫liquid_ice_collisions(
                    state, logλ, psd_c, psd_r, x_c, one(FT), zero(FT), zero(FT),
                    aps, tps, vel, ρₐ, FT(205), m_liq; quad = build_quad,
                )
                maxerr = max(
                    maxerr,
                    crelerr(exp(qc.log_G_NC), rc[3]), crelerr(exp(qc.log_G_MC), rc[1] + rc[2]),
                )
            end
            # Musil moments at this shape node
            qa = P3.lookup(ctables.musil_a, logλ, F_rim, ρ_rim)
            qb = P3.lookup(ctables.musil_b, logλ, F_rim, ρ_rim, ρₐ)
            n_i = DT.size_distribution(state, logλ)
            v_i = P3.ice_particle_terminal_velocity(vel, ρₐ, state)
            bnds = P3.velocity_integral_bounds(state, logλ, v_i; p = FT(1e-5))
            V_a = CM.Quadrature.integrate(D -> n_i(D) * D, bnds, build_quad)
            V_b = CM.Quadrature.integrate(D -> n_i(D) * D^(3 // 2) * sqrt(v_i(D)), bnds, build_quad)
            maxerr = max(maxerr, crelerr(exp(qa.log_V_a), V_a), crelerr(exp(qb.log_V_b), V_b))
        end
        @test maxerr < 1e-9
    end

    @testset "rain node exactness via unit-N₀r closed form" begin
        # The rain closed form scales exactly in N₀r; verify the tabulated shape.
        axλ, axF, axr, axa, axDr = ctables.rain.axes
        maxerr = 0.0
        for (i, j, k, l, m) in ((4, 2, 2, 2, 3), (axλ.n, axF.n, axr.n, axa.n, axDr.n))
            logλ = P3.node_coord(axλ, i)
            F_rim = P3.node_coord(axF, j)
            ρ_rim = P3.node_coord(axr, k)
            ρₐ = P3.node_coord(axa, l)
            Dr = P3.node_coord(axDr, m)
            x_ice = exp(P3.logLdivN(P3.P3State(params, one(FT), one(FT), F_rim, ρ_rim), logλ))
            state = P3.P3State(params, x_ice, one(FT), F_rim, ρ_rim)
            n_i = DT.size_distribution(state, logλ)
            ∂ₜV = P3.volumetric_collision_rate_integrand(vel, ρₐ, state)
            bnds = P3.velocity_integral_bounds(state, logλ, ∂ₜV.v_i; p = FT(1e-5))
            v_l = ∂ₜV.v_l
            ai, bi, ci = SVector(v_l.ai), SVector(v_l.bi), SVector(v_l.ci)
            Dmin = DT.exponential_quantile(Dr, FT(1e-5))
            Dmax = DT.exponential_quantile(Dr, 1 - FT(1e-5))
            inner(D) = P3.closed_rain_inner_NM(
                ∂ₜV.v_i(D), P3.crossover_diameter(∂ₜV.v_i(D), v_l, Dmin, Dmax),
                sqrt(P3.ice_area(state, D) / FT(π)), psd_r.ρw, ai, bi, ci, Dmin, Dmax, one(FT), Dr,
            )
            G_NR = CM.Quadrature.integrate(D -> n_i(D) * inner(D)[1], bnds, build_quad)
            G_MR = CM.Quadrature.integrate(D -> n_i(D) * inner(D)[2], bnds, build_quad)
            q = P3.lookup(ctables.rain, logλ, F_rim, ρ_rim, ρₐ, Dr)
            maxerr = max(maxerr, crelerr(exp(q.log_G_NR), G_NR), crelerr(exp(q.log_G_MR), G_MR))
        end
        @test maxerr < 1e-9
    end

    @testset "no clamping for harness states" begin
        axλ, axF, axr, axa, axxc = ctables.cloud.axes
        axDr = ctables.rain.axes[5]
        for h in collision_harness(FT, params)
            @test P3.node_coord(axλ, 1) <= h.logλ <= P3.node_coord(axλ, axλ.n)
            @test 0 <= h.state.F_rim <= P3.node_coord(axF, axF.n)
            @test P3.node_coord(axa, 1) <= h.ρₐ <= P3.node_coord(axa, axa.n)
            if h.state.F_rim > 0
                @test P3.node_coord(axr, 1) <= h.state.ρ_rim <= P3.node_coord(axr, axr.n)
            end
            if h.N_c > 0 && h.L_c > 0
                @test P3.node_coord(axxc, 1) <= h.L_c / h.N_c <= P3.node_coord(axxc, axxc.n)
            end
            if h.N_r > 0 && h.L_r > 0
                (; Dr_mean) = CM2.pdf_rain_parameters(psd_r, h.L_r / h.ρₐ, h.ρₐ, h.N_r)
                @test P3.node_coord(axDr, 1) <= Dr_mean <= P3.node_coord(axDr, axDr.n)
            end
        end
    end

    tab(h, T) = P3.bulk_liquid_ice_collision_sources(
        rate_tables, ctables, h.state, h.logλ, psd_c, psd_r,
        h.L_c, h.N_c, h.L_r, h.N_r, aps, tps, vel, h.ρₐ, T,
    )
    quadr(h, T) = P3.bulk_liquid_ice_collision_sources(
        h.state, h.logλ, psd_c, psd_r, h.L_c, h.N_c, h.L_r, h.N_r,
        aps, tps, vel, h.ρₐ, T; quad = qref,
    )
    OUT = (:∂ₜq_c, :∂ₜq_r, :∂ₜN_c, :∂ₜN_r, :∂ₜL_rim, :∂ₜL_ice, :∂ₜB_rim)

    pts = vcat(collision_harness(FT, params), collision_sweep(FT, params, 150))

    @testset "cold state: all 7 outputs match quadrature to interpolation error" begin
        # At T = 205 K the Musil limit does not bind (f_frz = 1), so every output
        # is a table interpolation of the quadrature value. The 95th percentile is
        # the accepted interpolation accuracy; the maximum bounds the F_rim → 1,
        # low-logλ corner, matching the Phase-1 rate tables.
        E = FT[]
        for h in pts
            a = tab(h, FT(205))
            b = quadr(h, FT(205))
            for k in OUT
                push!(E, crelerr(getfield(a, k), getfield(b, k)))
            end
        end
        @test quantile(E, 0.95) < 1.2e-1
        @test maximum(E) < 4e-1
    end

    @testset "∂ₜq_c and ∂ₜN_c are temperature-independent" begin
        # `∂ₜq_c = -M_C/ρₐ` and `∂ₜN_c = -NCCOL` carry no temperature dependence,
        # so the table assembly reproduces them across temperatures to round-off.
        for h in pts
            ref = tab(h, FT(230))
            for T in (FT(205), FT(250), FT(263), FT(268), FT(272))
                r = tab(h, T)
                @test r.∂ₜq_c ≈ ref.∂ₜq_c rtol = 1e-12
                @test r.∂ₜN_c ≈ ref.∂ₜN_c rtol = 1e-12
            end
        end
    end

    @testset "∂ₜq_c and ∂ₜN_c match quadrature at all temperatures" begin
        # Partition-free: independent of the freeze/shed split.
        Eq = FT[]
        EN = FT[]
        for h in pts, T in (FT(205), FT(250), FT(263), FT(268), FT(272))
            a = tab(h, T)
            b = quadr(h, T)
            push!(Eq, crelerr(a.∂ₜq_c, b.∂ₜq_c))
            push!(EN, crelerr(a.∂ₜN_c, b.∂ₜN_c))
        end
        @test quantile(Eq, 0.95) < 1.8e-1
        @test maximum(Eq) < 4e-1
        @test quantile(EN, 0.95) < 1.8e-1
        @test maximum(EN) < 4e-1
    end

    @testset "inference and allocations" begin
        h = first(collision_harness(FT, params))
        T = FT(263)
        @test (@inferred P3.bulk_max_freeze_rate(ctables, aps, tps, h.state, h.logλ, h.ρₐ, T)) isa FT
        src(rt, ct, hh, TT) = P3.bulk_liquid_ice_collision_sources(
            rt, ct, hh.state, hh.logλ, psd_c, psd_r, hh.L_c, hh.N_c, hh.L_r, hh.N_r, aps, tps, vel, hh.ρₐ, TT,
        )
        @test (@inferred src(rate_tables, ctables, h, T)) isa NamedTuple{OUT}
        cap(ct, hh, TT) = P3.bulk_max_freeze_rate(ct, aps, tps, hh.state, hh.logλ, hh.ρₐ, TT)
        # The freeze-capacity and full assembly inherit the thermodynamics `Lf`/`Lᵥ`
        # allocation; the table lookups add nothing on top of it.
        lf(tp, TT) = TDI.Lf(tp, TT) + TDI.Lᵥ(tp, TT)
        cap(ctables, h, T)
        lf(tps, T)
        @test (@allocated cap(ctables, h, T)) <= (@allocated lf(tps, T))
        src(rate_tables, ctables, h, T)
        @test (@allocated src(rate_tables, ctables, h, T)) <= (@allocated lf(tps, T))
    end

    @testset "Float32 build and assembly" begin
        F32 = Float32
        p32 = CMP.Microphysics2MParams(F32; with_ice = true, quad = CM.Quadrature.GaussLegendre(F32, 12))
        params32, vel32, aps32 = p32.ice.scheme, p32.ice.terminal_velocity, p32.warm_rain.air_properties
        psd_c32, psd_r32 = p32.ice.cloud_pdf, p32.ice.rain_pdf
        rg32 = P3.P3TableGrid{F32}(n_logλ = 24, n_F_rim = 8, n_ρ_rim = 6, n_ρ_air = 5)
        rt32 = P3.build_p3_lookup_tables(params32, vel32, aps32; grid = rg32)
        cg32 = P3.P3CollisionGrid{F32}(
            n_logλ = 16, n_F_rim = 6, n_ρ_rim = 5, n_ρ_air = 4, n_x_c = 8, n_Dr = 8, build_order = 12,
        )
        ct32 = P3.build_p3_collision_tables(params32, vel32, aps32, psd_c32, psd_r32; grid = cg32)
        @test eltype(ct32.cloud) === F32
        @test all(isfinite, ct32.cloud.data)
        @test all(isfinite, ct32.rain.data)
        @test all(isfinite, ct32.musil_a.data)
        @test all(isfinite, ct32.musil_b.data)
        state =
            P3.state_from_prognostic(params32, F32(0.9 * 5e-4), F32(0.9 * 1e5), F32(0.9 * 1e-4), F32(0.9 * 1e-4 / 300))
        logλ = P3.get_distribution_logλ(state)
        r = P3.bulk_liquid_ice_collision_sources(
            rt32, ct32, state, logλ, psd_c32, psd_r32,
            F32(0.9 * 5e-4), F32(1e8), F32(0.9 * 2e-4), F32(1e4), aps32,
            TDI.TD.Parameters.ThermodynamicsParameters(F32), vel32, F32(0.9), F32(263),
        )
        @test all(isfinite, values(r))
        @test r.∂ₜN_c <= 0
    end

    @testset "device kernel (Adapt round-trip)" begin
        dct = Adapt.adapt(Array, ctables)
        q1 = P3.lookup(ctables.cloud, 6.0, 0.3, 400.0, 0.9, 1e-11)
        q2 = P3.lookup(dct.cloud, 6.0, 0.3, 400.0, 0.9, 1e-11)
        @test q1.log_G_NC ≈ q2.log_G_NC
        @test q1.log_G_MC ≈ q2.log_G_MC
    end
end
nothing
