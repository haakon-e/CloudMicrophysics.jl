"""
Resolution study for the P3 clean-quantity lookup tables.

For each tabulated quantity (`ice_self_collection`, the number- and mass-weighted
ice terminal velocities, and `ice_melt`), report the relative error against a
high-order Gauss-Legendre reference as a function of the per-axis grid
resolution, and a lookup-versus-quadrature timing comparison on the CPU.

The error metric matches `test/p3_quadrature_error_study.jl`:
`|a - b| / max(|a|, |b|, floor)`. The evaluation states are the quadrature
error-study harness (`generate_column_states` plus `hail_core_states`) together
with a deterministic, physically realizable pseudo-random sweep of the table
input space; the reference evaluates each quantity at the same `logλ` passed to
the table, so the study isolates interpolation error.

Run from the repository root:

    julia --project=test -e 'include("test/p3_lookup_error_study.jl");
                             run_lookup_error_study(; include_timing = true)'

Keyword arguments of `run_lookup_error_study`:

- `FT`: float type. By default, `Float64`.
- `reference_order`: order of the reference rule. By default, `128`.
- `n_sweep`: number of pseudo-random sweep states. By default, `400`.
- `axis_levels`: the per-axis resolutions to sweep. By default, three levels per
  axis around the default grid.
- `include_timing`: also benchmark a table lookup against a quadrature call. By
  default, `false`.

Return a vector of `(; axis, n, quantity, p95, max)` rows.
"""

import ClimaParams as CP
import CloudMicrophysics as CM
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.ThermodynamicsInterface as TDI
import BenchmarkTools as BT
import Random
import Statistics: quantile

if !isdefined(@__MODULE__, :generate_column_states)
    include("p3_quadrature_error_study.jl")
end

# Evaluation states as `(state, logλ, ρ_air)` triples: the error-study harness
# plus a physically realizable pseudo-random sweep (rimed particles carry a rime
# density above a floor; each `logλ` maps to a consistent `x_ice`).
function _lut_eval_states(::Type{FT}, params, n_sweep) where {FT}
    states = Tuple{P3.P3State{FT}, FT, FT}[]
    for s in vcat(generate_column_states(FT), hail_core_states(FT, STUDY_HAIL_CORES))
        (s.q_ice > 0 && s.n_ice > 0) || continue
        st = P3.state_from_prognostic(params, s.ρ * s.q_ice, s.ρ * s.n_ice, s.ρ * s.q_rim, s.ρ * s.b_rim)
        push!(states, (st, P3.get_distribution_logλ(st), FT(s.ρ)))
    end
    rng = Random.MersenneTwister(0xBEEF)
    r_lo, r_hi = FT(100), FT(0.8) * params.ρ_l
    n_target = length(states) + n_sweep
    while length(states) < n_target
        logλ = FT(3 + rand(rng) * 10)
        F_rim = FT(rand(rng) * 0.95)
        ρ_rim = r_lo + rand(rng) * (r_hi - r_lo)
        ρₐ = FT(exp10(log10(0.1) + rand(rng) * (log10(1.4) - log10(0.1))))
        x = exp(P3.logLdivN(P3.P3State(params, one(FT), one(FT), F_rim, ρ_rim), logλ))
        (isfinite(x) && x > 0) || continue
        push!(states, (P3.P3State(params, x * FT(1e5), FT(1e5), F_rim, ρ_rim), logλ, ρₐ))
    end
    return states
end

_lut_relerr(a, b) = abs(a - b) / max(abs(a), abs(b), 1e-12)

# Reference values (one per state) at the reference quadrature order.
function _lut_references(states, vel, aps, tps, qref, FT)
    return map(states) do (st, logλ, ρₐ)
        (
            selfcol = P3.ice_self_collection(st, logλ, vel, ρₐ; quad = qref).dNdt,
            vN = P3.ice_terminal_velocity_number_weighted(vel, ρₐ, st, logλ; quad = qref),
            vM = P3.ice_terminal_velocity_mass_weighted(vel, ρₐ, st, logλ; quad = qref),
            melt = P3.ice_melt(vel, aps, tps, FT(280), ρₐ, st, logλ; quad = qref).dLdt,
        )
    end
end

function _lut_quantity_errors(tables, states, refs, aps, tps, FT)
    E = Dict(k => FT[] for k in (:selfcol, :vN, :vM, :melt))
    for ((st, logλ, ρₐ), ref) in zip(states, refs)
        push!(E[:selfcol], _lut_relerr(P3.ice_self_collection(tables, st, logλ, ρₐ).dNdt, ref.selfcol))
        push!(E[:vN], _lut_relerr(P3.ice_terminal_velocity_number_weighted(tables, st, logλ, ρₐ), ref.vN))
        push!(E[:vM], _lut_relerr(P3.ice_terminal_velocity_mass_weighted(tables, st, logλ, ρₐ), ref.vM))
        push!(E[:melt], _lut_relerr(P3.ice_melt(tables, aps, tps, FT(280), ρₐ, st, logλ).dLdt, ref.melt))
    end
    return E
end

function run_lookup_error_study(;
    FT = Float64,
    reference_order = 128,
    n_sweep = 400,
    axis_levels = (
        n_logλ = (48, 80, 128),
        n_F_rim = (12, 24, 40),
        n_ρ_rim = (10, 18, 30),
        n_ρ_air = (5, 8, 12),
    ),
    include_timing = false,
)
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, quad = CM.Quadrature.GaussLegendre(FT, 12))
    params, vel, aps = mp.ice.scheme, mp.ice.terminal_velocity, mp.warm_rain.air_properties
    qref = CM.Quadrature.GaussLegendre(FT, reference_order)
    states = _lut_eval_states(FT, params, n_sweep)
    refs = _lut_references(states, vel, aps, tps, qref, FT)
    base = P3.P3TableGrid{FT}()
    base_nt = NamedTuple{fieldnames(P3.P3TableGrid)}(map(f -> getfield(base, f), fieldnames(P3.P3TableGrid)))

    results = NamedTuple[]
    println("axis        | n    | quantity | p95        | max")
    for axis in (:n_logλ, :n_F_rim, :n_ρ_rim, :n_ρ_air)
        for n in getproperty(axis_levels, axis)
            grid = P3.P3TableGrid{FT}(; merge(base_nt, NamedTuple{(axis,)}((n,)))...)
            tables = P3.build_p3_lookup_tables(params, vel, aps; grid)
            E = _lut_quantity_errors(tables, states, refs, aps, tps, FT)
            for q in (:selfcol, :vN, :vM, :melt)
                p95, mx = quantile(E[q], 0.95), maximum(E[q])
                push!(results, (; axis, n, quantity = q, p95, max = mx))
                println(rpad(axis, 11), " | ", rpad(n, 4), " | ", rpad(q, 8), " | ",
                    rpad(round(p95, sigdigits = 3), 10), " | ", round(mx, sigdigits = 3))
            end
        end
    end

    if include_timing
        tables = P3.build_p3_lookup_tables(params, vel, aps; grid = base)
        st, logλ, ρₐ = states[end]
        tlut = BT.@benchmark P3.ice_self_collection($tables, $st, $logλ, $ρₐ) samples = 1000
        tqua = BT.@benchmark P3.ice_self_collection($st, $logλ, $vel, $ρₐ; quad = $(mp.ice.quad)) samples = 200
        println("\nice_self_collection CPU time:")
        println("  table lookup : ", round(BT.minimum(tlut).time, digits = 1), " ns")
        println("  GL(12) quad  : ", round(BT.minimum(tqua).time / 1000, digits = 2), " μs")
        println("  speedup      : ", round(BT.minimum(tqua).time / BT.minimum(tlut).time, digits = 1), "x")
    end

    return results
end
