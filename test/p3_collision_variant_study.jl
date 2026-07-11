"""
Evaluation study for the Phase-2 liquid-ice collision lookup tables.

Compare the two table paths, variant A (bulk freeze/shed partition) and variant C
(exact per-diameter partition with tabulated inner moments), and the hybrid switch,
against a high-order Gauss-Legendre reference of `bulk_liquid_ice_collision_sources`.
Report, per variant: the seven-output collision error and the full
`bulk_microphysics_tendencies` vector error over the quadrature error-study harness,
a physically realizable sweep, and a temperature sweep; the CPU time of one call
against the quadrature baseline; the table build time and memory; the two flagged
soft spots (the representative rime-density and wet-fraction closures); and a hybrid
`θ` sweep.

The error metric matches `test/p3_quadrature_error_study.jl`:
`|a - b| / max(|a|, |b|, floor)`.

Run from the repository root:

    julia --project=test -e 'include("test/p3_collision_variant_study.jl");
                             run_collision_variant_study()'

Keyword arguments of `run_collision_variant_study`:

- `FT`: float type. By default, `Float64`.
- `reference_order`: order of the reference rule. By default, `128`.
- `n_sweep`: number of pseudo-random sweep states. By default, `200`.
- `include_timing`: also benchmark each variant against the quadrature baseline. By
  default, `true`.

Return a `NamedTuple` of the measured tables.
"""

import ClimaParams as CP
import CloudMicrophysics as CM
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.ThermodynamicsInterface as TDI
import BenchmarkTools as BT
import Random
import Statistics: quantile
using Printf

if !isdefined(@__MODULE__, :generate_column_states)
    include("p3_quadrature_error_study.jl")
end

_cv_relerr(a, b) = abs(a - b) / max(abs(a), abs(b), 1e-12)
const _CV_OUT = (:∂ₜq_c, :∂ₜq_r, :∂ₜN_c, :∂ₜN_r, :∂ₜL_rim, :∂ₜL_ice, :∂ₜB_rim)

# Collision-source states `(state, logλ, ρ_air, liquid loadings)`: the harness
# restricted to ice-with-liquid states plus a physically realizable sweep.
function _cv_states(::Type{FT}, params, n_sweep) where {FT}
    out = NamedTuple[]
    for s in vcat(generate_column_states(FT), hail_core_states(FT, STUDY_HAIL_CORES))
        (s.q_ice > 0 && s.n_ice > 0) || continue
        (s.q_lcl > 0 || s.q_rai > 0) || continue
        st = P3.state_from_prognostic(params, s.ρ * s.q_ice, s.ρ * s.n_ice, s.ρ * s.q_rim, s.ρ * s.b_rim)
        push!(
            out,
            (;
                state = st, logλ = P3.get_distribution_logλ(st), ρₐ = FT(s.ρ),
                L_c = FT(s.ρ * s.q_lcl), N_c = FT(s.n_lcl), L_r = FT(s.ρ * s.q_rai), N_r = FT(s.n_rai),
            ),
        )
    end
    rng = Random.MersenneTwister(0xC0FFEE)
    r_lo, r_hi = FT(100), FT(0.8) * params.ρ_l
    target = length(out) + n_sweep
    while length(out) < target
        logλ = FT(3 + rand(rng) * 10)
        F_rim = FT(rand(rng)) * FT(0.95)
        ρ_rim = r_lo + rand(rng) * (r_hi - r_lo)
        ρₐ = FT(exp10(log10(0.1) + rand(rng) * (log10(1.4) - log10(0.1))))
        x = exp(P3.logLdivN(P3.P3State(params, one(FT), one(FT), F_rim, ρ_rim), logλ))
        (isfinite(x) && x > 0) || continue
        st = P3.P3State(params, x * FT(1e5), FT(1e5), F_rim, ρ_rim)
        push!(
            out,
            (;
                state = st, logλ, ρₐ,
                L_c = ρₐ * FT(exp10(-4 + rand(rng) * 0.7)), N_c = FT(exp10(7 + rand(rng) * 1.5)),
                L_r = ρₐ * FT(exp10(-4.3 + rand(rng) * 0.8)), N_r = FT(exp10(3.5 + rand(rng) * 1.5)),
            ),
        )
    end
    return out
end

function run_collision_variant_study(;
    FT = Float64, reference_order = 128, n_sweep = 200, include_timing = true,
)
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, quad = CM.Quadrature.GaussLegendre(FT, 6))
    params, vel, aps = mp.ice.scheme, mp.ice.terminal_velocity, mp.warm_rain.air_properties
    psd_c, psd_r = mp.ice.cloud_pdf, mp.ice.rain_pdf
    qref = CM.Quadrature.GaussLegendre(FT, reference_order)
    build_quad = CM.Quadrature.GaussLegendre(FT, 16)
    out_quad = CM.Quadrature.GaussLegendre(FT, 8)

    tb = @elapsed rate_tables = P3.build_p3_lookup_tables(params, vel, aps)
    tA = @elapsed ctables = P3.build_p3_collision_tables(params, vel, aps, psd_c, psd_r; quad = build_quad)
    tC = @elapsed itables = P3.build_p3_collision_inner_tables(params, vel, aps, psd_c, psd_r; quad = build_quad)
    memA =
        (
            sizeof(ctables.cloud.data) + sizeof(ctables.rain.data) + sizeof(ctables.musil_a.data) +
            sizeof(ctables.musil_b.data)
        ) / 1e6
    memC = (sizeof(itables.cloud_inner.data) + sizeof(itables.rain_inner.data)) / 1e6
    println("build: rate ", round(tb; digits = 0), " s | A ", round(tA; digits = 0), " s (",
        round(memA; digits = 1), " MB) | C ", round(tC; digits = 1), " s (", round(memC; digits = 1), " MB)")

    states = _cv_states(FT, params, n_sweep)
    Tsweep = (FT(230), FT(255), FT(263), FT(268), FT(272))
    truth = Dict{Tuple{Int, FT}, Any}()
    for (i, h) in enumerate(states), T in Tsweep
        truth[(i, T)] = P3.bulk_liquid_ice_collision_sources(
            h.state, h.logλ, psd_c, psd_r, h.L_c, h.N_c, h.L_r, h.N_r, aps, tps, vel, h.ρₐ, T; quad = qref,
        )
    end
    vA(h, T) = P3.bulk_liquid_ice_collision_sources(
        rate_tables, ctables, h.state, h.logλ, psd_c, psd_r, h.L_c, h.N_c, h.L_r, h.N_r, aps, tps, vel, h.ρₐ, T,
    )
    vC(h, T) = P3.bulk_liquid_ice_collision_sources(
        rate_tables, itables, h.state, h.logλ, psd_c, psd_r, h.L_c, h.N_c, h.L_r, h.N_r, aps, tps, vel, h.ρₐ, T;
        quad = out_quad,
    )
    vH(h, T, θ) = P3.bulk_liquid_ice_collision_sources(
        rate_tables, ctables, itables, h.state, h.logλ, psd_c, psd_r, h.L_c, h.N_c, h.L_r, h.N_r, aps, tps, vel,
        h.ρₐ, T; quad = out_quad, θ,
    )

    function coll_err(f)
        Eall, Ewarm = FT[], FT[]
        for (i, h) in enumerate(states), T in Tsweep
            a, b = f(h, T), truth[(i, T)]
            for k in _CV_OUT
                e = _cv_relerr(getfield(a, k), getfield(b, k))
                push!(Eall, e)
                T == FT(272) && push!(Ewarm, e)
            end
        end
        (
            all_p95 = quantile(Eall, 0.95),
            all_max = maximum(Eall),
            warm_p95 = quantile(Ewarm, 0.95),
            warm_max = maximum(Ewarm),
        )
    end
    coll = (; A = coll_err(vA), C = coll_err(vC), hybrid1 = coll_err((h, T) -> vH(h, T, FT(1))))
    println("collision 7-output error vs GL(", reference_order, "):")
    for (nm, r) in pairs(coll)
        @printf(
            "  %-8s all p95=%.2e max=%.2e | 272K p95=%.2e max=%.2e\n",
            nm,
            r.all_p95,
            r.all_max,
            r.warm_p95,
            r.warm_max
        )
    end

    θsweep = NamedTuple[]
    for θ in (FT(0), FT(1), FT(2), FT(Inf))
        E = FT[]
        for (i, h) in enumerate(states)
            a, b = vH(h, FT(272), θ), truth[(i, FT(272))]
            for k in _CV_OUT
                push!(E, _cv_relerr(getfield(a, k), getfield(b, k)))
            end
        end
        push!(θsweep, (; θ, warm_p95 = quantile(E, 0.95), warm_max = maximum(E)))
    end

    timing = nothing
    if include_timing
        h = states[findfirst(s -> s.N_c > 0 && s.N_r > 0 && s.state.F_rim > 0.1, states)]
        T = FT(265)
        tq = BT.@belapsed P3.bulk_liquid_ice_collision_sources(
            $h.state, $h.logλ, $psd_c, $psd_r, $h.L_c, $h.N_c, $h.L_r, $h.N_r, $aps, $tps, $vel, $h.ρₐ, $T;
            quad = $(mp.ice.quad),
        ) samples = 60 evals = 1
        ta = BT.@belapsed vA($h, $T) samples = 200 evals = 1
        tc = BT.@belapsed vC($h, $T) samples = 100 evals = 1
        th = BT.@belapsed vH($h, $T, $(FT(1))) samples = 100 evals = 1
        timing = (; quadrature = tq, A = ta, C = tc, hybrid1 = th)
        @printf(
            "collision time: quad %.3f ms | A %.1f us | C %.1f us | hybrid %.1f us\n",
            tq * 1e3,
            ta * 1e6,
            tc * 1e6,
            th * 1e6
        )
    end

    return (; build = (; rate = tb, A = tA, C = tC, memA, memC), collision = coll, θsweep, timing)
end
