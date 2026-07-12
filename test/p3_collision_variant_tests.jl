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
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import StaticArrays: SVector

include("p3_quadrature_error_study.jl")  # generate_column_states, hail_core_states

const VFLOOR = 1e-12
vrelerr(a, b) = abs(a - b) / max(abs(a), abs(b), VFLOOR)

# Marker adaptor: wraps every backing `Array` it reaches in a `TagArray`, so a
# round-trip proves `Adapt` traverses the wrapper down to the tables' arrays.
struct TagArray{T, N} <: AbstractArray{T, N}
    parent::Array{T, N}
end
Base.size(a::TagArray) = size(a.parent)
Base.getindex(a::TagArray, i::Int...) = getindex(a.parent, i...)
struct MarkAdaptor end
Adapt.adapt_storage(::MarkAdaptor, x::Array) = TagArray(x)

function variant_harness(::Type{FT}, params) where {FT}
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

function variant_sweep(::Type{FT}, params, n; seed = 0xBADA55) where {FT}
    rng = Random.MersenneTwister(seed)
    r_lo, r_hi = FT(100), FT(0.8) * params.ρ_l
    out = NamedTuple[]
    while length(out) < n
        logλ = FT(3 + rand(rng) * 10)
        F_rim = FT(rand(rng)) * FT(0.95)
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

@testset "P3 collision variant C and hybrid" begin
    FT = Float64
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, quad = CM.Quadrature.GaussLegendre(FT, 12))
    params, vel, aps = mp.ice.scheme, mp.ice.terminal_velocity, mp.warm_rain.air_properties
    psd_c, psd_r = mp.ice.cloud_pdf, mp.ice.rain_pdf
    build_quad = CM.Quadrature.GaussLegendre(FT, 16)
    out_quad = CM.Quadrature.GaussLegendre(FT, 8)
    qref = CM.Quadrature.GaussLegendre(FT, 32)
    m_liq(Dₗ) = psd_c.ρw * CO.volume_sphere_D(Dₗ)

    rgrid = P3.P3TableGrid{FT}(n_logλ = 48, n_F_rim = 14, n_ρ_rim = 10, n_ρ_air = 5)
    rate_tables = P3.build_p3_lookup_tables(params, vel, aps; grid = rgrid)
    cgrid = P3.P3CollisionGrid{FT}(
        n_logλ = 32, n_F_rim = 14, n_ρ_rim = 10, n_ρ_air = 5, n_x_c = 12, n_Dr = 12, build_order = 12,
    )
    ctables = P3.build_p3_collision_tables(params, vel, aps, psd_c, psd_r; grid = cgrid, quad = build_quad)
    itables = P3.build_p3_collision_inner_tables(params, vel, aps, psd_c, psd_r; quad = build_quad)

    OUT = (:∂ₜq_c, :∂ₜq_r, :∂ₜN_c, :∂ₜN_r, :∂ₜL_rim, :∂ₜL_ice, :∂ₜB_rim)
    vC(h, T) = P3.bulk_liquid_ice_collision_sources(
        rate_tables, itables, h.state, h.logλ, psd_c, psd_r,
        h.L_c, h.N_c, h.L_r, h.N_r, aps, tps, vel, h.ρₐ, T; quad = out_quad,
    )
    vA(h, T) = P3.bulk_liquid_ice_collision_sources(
        rate_tables, ctables, h.state, h.logλ, psd_c, psd_r,
        h.L_c, h.N_c, h.L_r, h.N_r, aps, tps, vel, h.ρₐ, T,
    )
    vH(h, T, θ) = P3.bulk_liquid_ice_collision_sources(
        rate_tables, ctables, itables, h.state, h.logλ, psd_c, psd_r,
        h.L_c, h.N_c, h.L_r, h.N_r, aps, tps, vel, h.ρₐ, T; quad = out_quad, θ,
    )
    quadr(h, T) = P3.bulk_liquid_ice_collision_sources(
        h.state, h.logλ, psd_c, psd_r, h.L_c, h.N_c, h.L_r, h.N_r, aps, tps, vel, h.ρₐ, T; quad = qref,
    )
    pts = vcat(variant_harness(FT, params), variant_sweep(FT, params, 120))

    @testset "inner-table node exactness" begin
        axv, axr, axa, axxc = itables.cloud_inner.axes
        axDr = itables.rain_inner.axes[4]
        p = FT(1e-5)
        maxerr = 0.0
        for (i, j, l) in ((3, 4, 2), (10, 12, 3), (axv.n, axr.n, axa.n))
            v_i = P3.node_coord(axv, i)
            r_i = P3.node_coord(axr, j)
            ρₐ = P3.node_coord(axa, l)
            v_l = CO.particle_terminal_velocity(vel.rain, ρₐ)
            ai, bi, ci = SVector(v_l.ai), SVector(v_l.bi), SVector(v_l.ci)
            for m in (1, 6, axDr.n)
                Dr = P3.node_coord(axDr, m)
                Dmin = DT.exponential_quantile(Dr, p)
                Dmax = DT.exponential_quantile(Dr, 1 - p)
                Dstar = P3.crossover_diameter(v_i, v_l, Dmin, Dmax)
                ref = P3.closed_rain_inner_NM(v_i, Dstar, r_i, psd_r.ρw, ai, bi, ci, Dmin, Dmax, one(FT), Dr)
                q = P3.lookup(itables.rain_inner, v_i, r_i, ρₐ, Dr)
                maxerr = max(maxerr, vrelerr(exp(q.log_H_NR), ref[1]), vrelerr(exp(q.log_H_MR), ref[2]))
            end
        end
        @test maxerr < 1e-9
    end

    @testset "(v_i, r_i) collapse: runtime inner factors through the table basis" begin
        # The rain inner moment depends on the ice state only through (v_i, r_i).
        # The runtime closed-form path (get_liquid_integrals_rain_closed) and the
        # tabulated basis (closed_rain_inner_NM) evaluate the same function of
        # (v_i, r_i); a build-order lookup reproduces the runtime inner to the
        # interpolation error, and there is no state-dependent residual.
        p = FT(1e-5)
        leak = 0.0
        for h in first(pts, 20)
            h.N_r > 0 && h.L_r > 0 || continue
            (; N₀r, Dr_mean) = CM2.pdf_rain_parameters(psd_r, h.L_r / h.ρₐ, h.ρₐ, h.N_r)
            (
                N₀r > 0 &&
                P3.node_coord(itables.rain_inner.axes[4], 1) <= Dr_mean <=
                P3.node_coord(itables.rain_inner.axes[4], itables.rain_inner.axes[4].n)
            ) || continue
            ∂ₜV = P3.volumetric_collision_rate_integrand(vel, h.ρₐ, h.state)
            bounds_r = CM2.get_size_distribution_bounds(psd_r, h.L_r / h.ρₐ, h.ρₐ, h.N_r, p)
            n_r = DT.size_distribution(psd_r, h.L_r / h.ρₐ, h.ρₐ, h.N_r)
            ρ1 = (a, b) -> one(FT)
            runtime = P3.get_liquid_integrals_rain_closed(
                psd_r,
                n_r,
                h.ρₐ,
                h.L_r,
                h.N_r,
                h.state,
                ∂ₜV,
                m_liq,
                ρ1,
                bounds_r;
                quad = build_quad,
            )
            bnds = P3.velocity_integral_bounds(h.state, h.logλ, ∂ₜV.v_i; p)
            gm = sqrt(bnds[1] * bnds[end])
            for Dᵢ in (gm, sqrt(bnds[1] * gm))
                ∂ₜN, ∂ₜM, _ = runtime(Dᵢ)
                q = P3.lookup(itables.rain_inner, ∂ₜV.v_i(Dᵢ), sqrt(P3.ice_area(h.state, Dᵢ) / FT(π)), h.ρₐ, Dr_mean)
                leak = max(leak, vrelerr(∂ₜN, N₀r * exp(q.log_H_NR)), vrelerr(∂ₜM, N₀r * exp(q.log_H_MR)))
            end
        end
        @test leak < 1e-1
    end

    @testset "variant C: no warm-band bias, within a closure/interpolation envelope" begin
        # The p95 is temperature independent (no warm-band bias); the max is a
        # closure/interpolation envelope, set by the F_rim → 1 corner of the
        # representative rime-density ∂ₜB_rim closure on this CI grid, not pure
        # interpolation error.
        Ecold = FT[]
        Ewarm = FT[]
        for h in pts
            ac = vC(h, FT(230))
            bc = quadr(h, FT(230))
            aw = vC(h, FT(272))
            bw = quadr(h, FT(272))
            for k in OUT
                push!(Ecold, vrelerr(getfield(ac, k), getfield(bc, k)))
                push!(Ewarm, vrelerr(getfield(aw, k), getfield(bw, k)))
            end
        end
        @test quantile(Ecold, 0.95) < 1.0e-1
        @test quantile(Ewarm, 0.95) < 1.0e-1
        @test maximum(Ewarm) < 6.0e-1
    end

    @testset "variant C beats variant A on the warm-band partition outputs" begin
        # The five partition-dependent outputs carry the bulk-partition bias of
        # variant A in the warm band; variant C removes it.
        partition_out = (:∂ₜq_r, :∂ₜN_r, :∂ₜL_ice)
        EA = FT[]
        EC = FT[]
        for h in pts, T in (FT(266), FT(270), FT(272))
            a = vA(h, T)
            c = vC(h, T)
            b = quadr(h, T)
            for k in partition_out
                push!(EA, vrelerr(getfield(a, k), getfield(b, k)))
                push!(EC, vrelerr(getfield(c, k), getfield(b, k)))
            end
        end
        @test quantile(EC, 0.95) < quantile(EA, 0.95)
    end

    @testset "hybrid selects the branch by θ" begin
        for h in first(pts, 40), T in (FT(230), FT(268), FT(272))
            a = vA(h, T)
            c = vC(h, T)
            # θ = 0: ratio ≥ 0 always, so the variant-A branch is always taken.
            @test vH(h, T, FT(0)).∂ₜq_r == a.∂ₜq_r
            # θ = Inf: the variant-C branch is always taken.
            @test vH(h, T, FT(Inf)).∂ₜq_r == c.∂ₜq_r
            # θ = 1: one of the two branches, exactly.
            hq = vH(h, T, FT(1)).∂ₜq_r
            @test hq == a.∂ₜq_r || hq == c.∂ₜq_r
        end
    end

    @testset "inference and allocations" begin
        h = first(variant_harness(FT, params))
        T = FT(268)
        @test (@inferred vC(h, T)) isa NamedTuple{OUT}
        @test (@inferred vH(h, T, FT(1))) isa NamedTuple{OUT}
        lf(tp, TT) = TDI.Lf(tp, TT) + TDI.Lᵥ(tp, TT)
        vC(h, T)
        lf(tps, T)
        @test (@allocated vC(h, T)) <= (@allocated lf(tps, T))
        vH(h, T, FT(1))
        @test (@allocated vH(h, T, FT(1))) <= (@allocated lf(tps, T))
    end

    @testset "Float32 build and assembly" begin
        F32 = Float32
        p32 = CMP.Microphysics2MParams(F32; with_ice = true, quad = CM.Quadrature.GaussLegendre(F32, 12))
        params32, vel32, aps32 = p32.ice.scheme, p32.ice.terminal_velocity, p32.warm_rain.air_properties
        psd_c32, psd_r32 = p32.ice.cloud_pdf, p32.ice.rain_pdf
        rg32 = P3.P3TableGrid{F32}(n_logλ = 24, n_F_rim = 8, n_ρ_rim = 6, n_ρ_air = 5)
        rt32 = P3.build_p3_lookup_tables(params32, vel32, aps32; grid = rg32)
        ig32 = P3.P3CollisionInnerGrid{F32}(n_v_i = 20, n_r_i = 20, n_ρ_air = 4, n_x_c = 8, n_Dr = 8, build_order = 12)
        it32 = P3.build_p3_collision_inner_tables(params32, vel32, aps32, psd_c32, psd_r32; grid = ig32)
        @test eltype(it32.cloud_inner) === F32
        @test all(isfinite, it32.cloud_inner.data)
        @test all(isfinite, it32.rain_inner.data)
        state =
            P3.state_from_prognostic(params32, F32(0.9 * 5e-4), F32(0.9 * 1e5), F32(0.9 * 1e-4), F32(0.9 * 1e-4 / 300))
        logλ = P3.get_distribution_logλ(state)
        r = P3.bulk_liquid_ice_collision_sources(
            rt32, it32, state, logλ, psd_c32, psd_r32,
            F32(0.9 * 5e-4), F32(1e8), F32(0.9 * 2e-4), F32(1e4),
            aps32, TDI.TD.Parameters.ThermodynamicsParameters(F32), vel32, F32(0.9), F32(263);
            quad = CM.Quadrature.GaussLegendre(F32, 8),
        )
        @test all(isfinite, values(r))
        @test r.∂ₜN_c <= 0
        # Ice present, cloud and rain absent: rime-volume closure stays finite.
        r0 = P3.bulk_liquid_ice_collision_sources(
            rt32, it32, state, logλ, psd_c32, psd_r32,
            F32(0), F32(0), F32(0), F32(0),
            aps32, TDI.TD.Parameters.ThermodynamicsParameters(F32), vel32, F32(0.9), F32(263);
            quad = CM.Quadrature.GaussLegendre(F32, 8),
        )
        @test all(isfinite, values(r0))
        @test iszero(r0.∂ₜB_rim)
    end

    @testset "device kernel (Adapt round-trip)" begin
        dit = Adapt.adapt(Array, itables)
        q1 = P3.lookup(itables.rain_inner, 2.0, 1e-4, 0.9, 3e-4)
        q2 = P3.lookup(dit.rain_inner, 2.0, 1e-4, 0.9, 3e-4)
        @test q1.log_H_NR ≈ q2.log_H_NR
        @test q1.log_H_MR ≈ q2.log_H_MR
    end

    @testset "P3IceTables Adapt traverses to backing arrays" begin
        eng = BMT.P3IceTables(rate_tables, ctables, itables, out_quad, BMT.P3CollisionVariantC())
        m = Adapt.adapt(MarkAdaptor(), eng)
        @test m.rate_tables.rates.data isa TagArray
        @test m.coll_tables.cloud.data isa TagArray
        @test m.inner_tables.rain_inner.data isa TagArray
        @test eng.coll_tables.cloud.data isa Array  # the original is untouched
        # A variant-A-only engine with `nothing` sub-tables adapts to `nothing`.
        engA0 = BMT.P3IceTables(rate_tables, ctables, nothing, out_quad, BMT.P3CollisionVariantA())
        @test Adapt.adapt(MarkAdaptor(), engA0).inner_tables === nothing
    end

    @testset "BMT table path dispatches the collision variant" begin
        engA = BMT.P3IceTables(rate_tables, ctables, itables, out_quad, BMT.P3CollisionVariantA())
        engC = BMT.P3IceTables(rate_tables, ctables, itables, out_quad, BMT.P3CollisionVariantC())
        engH = BMT.P3IceTables(rate_tables, ctables, itables, out_quad, BMT.P3CollisionHybrid(FT(1)))
        s = P3.state_from_prognostic(params, 0.85 * 5e-4, 0.85 * 1e5, 0.85 * 1e-4, 0.85 * 1e-4 / 300)
        logλ = P3.get_distribution_logλ(s)
        args = (
            0.85,
            FT(230),
            FT(5e-3),
            FT(5e-4),
            FT(1e8 / 0.85),
            FT(2e-4),
            FT(1e4 / 0.85),
            FT(5e-4),
            FT(1e5 / 0.85),
            FT(1e-4),
            FT(1e-4 / 300),
            logλ,
        )
        base = BMT.bulk_microphysics_tendencies(BMT.Microphysics2Moment(), mp, tps, args...)
        rA = BMT.bulk_microphysics_tendencies(BMT.Microphysics2Moment(), mp, tps, args...; p3_tables = engA)
        rC = BMT.bulk_microphysics_tendencies(BMT.Microphysics2Moment(), mp, tps, args...; p3_tables = engC)
        rH = BMT.bulk_microphysics_tendencies(BMT.Microphysics2Moment(), mp, tps, args...; p3_tables = engH)
        @test keys(base) == keys(rA) == keys(rC) == keys(rH)
        @test typeof(base) === typeof(rA) === typeof(rC) === typeof(rH)
        @test all(isfinite, values(rC))
        # the ice-cloud sink is partition-free; the table path tracks quadrature
        @test rC.dn_lcl_dt ≈ base.dn_lcl_dt rtol = 1.5e-1
    end

    @testset "table path stays finite with ice but no liquid" begin
        # Ice present, cloud and rain absent: the representative rime-volume
        # closure must not divide by an indeterminate mean collision size.
        engA = BMT.P3IceTables(rate_tables, ctables, itables, out_quad, BMT.P3CollisionVariantA())
        engC = BMT.P3IceTables(rate_tables, ctables, itables, out_quad, BMT.P3CollisionVariantC())
        engH = BMT.P3IceTables(rate_tables, ctables, itables, out_quad, BMT.P3CollisionHybrid(FT(1)))
        for (F_rim, ρ_rim) in ((FT(0), FT(0)), (FT(0.4), FT(400)))
            ρ = FT(0.4)
            q_ice, n_ice = FT(1e-6), FT(1e5)
            q_rim = F_rim * q_ice
            b_rim = ρ_rim > 0 ? q_rim / ρ_rim : FT(0)
            s = P3.state_from_prognostic(params, ρ * q_ice, ρ * n_ice, ρ * q_rim, ρ * b_rim)
            logλ = P3.get_distribution_logλ(s)
            for T in (FT(240), FT(275))
                args = (
                    ρ, T, q_ice + FT(1e-3),
                    FT(0), FT(0), FT(0), FT(0),  # no cloud, no rain
                    q_ice, n_ice, q_rim, b_rim, logλ,
                )
                for eng in (engA, engC, engH)
                    r = BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics2Moment(), mp, tps, args...; p3_tables = eng,
                    )
                    @test all(isfinite, values(r))
                end
            end
        end
    end
end
nothing
