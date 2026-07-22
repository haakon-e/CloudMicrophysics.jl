using Test
import Statistics: median, quantile

import ClimaParams as CP
import CloudMicrophysics as CM
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.Common as CO
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.ThermodynamicsInterface as TDI

"""
Accuracy regression guard for the production default quadrature order (`GL(6)`)
over [`generate_comprehensive_states`](@ref) (`p3_state_battery.jl`): a
deterministic, systematic Latin-hypercube sweep of the full P3 prognostic and
environment space plus explicit stress corners, 361 states total.

`bulk_tendencies_quadrature_tests.jl` locks the same comparison over the thin
~17-state `generate_column_states`/`hail_core_states` battery. That battery
missed a real accuracy issue this session (`MomentMatched(2)`'s mass-regime
`N`-component failure only appeared once the ice mass-regime thresholds and
riming-density contrast were swept broadly — see
`derivation_thread3_massregime.md`/`derivation_thread5_comprehensive_battery.md`
in the perf-quadmath scratch history). This file adds the same style of lock
over the broad battery, so a future regression restricted to a state the thin
battery does not sample is still caught.

# Locked tolerances (`GL(6)` vs `GL(128)`, `Float64`)

`error_study_level` groups (median / p95, from `run_quadrature_error_study`
with `states = generate_comprehensive_states(...)`):

| level | median tol | p95 tol |
|---|---|---|
| `transport` | 1e-6 | 1e-2 |
| `collision_efficiency` | 5e-3 | 3e-2 |
| `collision` | 5e-3 | 3e-2 |
| `selfcol` | 2e-2 | 3e-2 |
| `bulk` | 1e-6 | 1e-2 |

`max` is intentionally not locked at the level aggregate: `collision`/
`collision_efficiency` legitimately spike toward 1.0 in a small number of
states because the wet-growth onset search itself (not the quadrature order)
occasionally misses a real but negligible crossing — a known, non-consequential
limitation (`derivation_thread5_comprehensive_battery.md`,
`derivation_thread7_categorization_and_lessons.md`), not a regression signal.
`median`/`p95` are the sensitive, low-noise statistics; a real regression
(e.g. the mass-regime `N` failure) shifts those, not just the tail.

Per-raw-collision-component tolerances (`∫liquid_ice_collisions`'s 10 raw
outputs, median / p95): the mass/number-rate components (`QCFRZ`, `NCCOL`,
`QRFRZ`, `NRCOL`, `∫M_col`, `BCCOL`, `BRCOL`) are locked tightly on both
statistics — any of these regressing is a real bug. The wet-growth-gated shed
components (`QCSHD`, `QRSHD`, `∫𝟙_wet_M_col`) are locked only on the median
(near-zero across most states) — their `p95`/`max` inherit the onset-search
limitation above and are not useful regression signals at the default order.

Rerun `measure_ci_lock_tolerances.jl`-style analysis (see the perf-quadmath
scratch history) and update this table when changing the quadrature rule,
integral bounds, breakpoints, or P3 process integrands — same policy as
`bulk_tendencies_quadrature_tests.jl`.
"""

include("p3_quadrature_error_study.jl")
include("p3_state_battery.jl")

const COMPREHENSIVE_LEVEL_TOLERANCES = (
    transport = (median = 1e-6, p95 = 1e-2),
    collision_efficiency = (median = 5e-3, p95 = 3e-2),
    collision = (median = 5e-3, p95 = 3e-2),
    selfcol = (median = 2e-2, p95 = 3e-2),
    bulk = (median = 1e-6, p95 = 1e-2),
)

# (median tol, p95 tol); p95 omitted (`nothing`) for the wet-growth-gated shed
# components, whose tail is governed by the onset-search limitation above, not
# quadrature order.
const COMPREHENSIVE_COMPONENT_TOLERANCES = (
    QCFRZ = (median = 2e-2, p95 = 3e-2),
    QCSHD = (median = 1e-2, p95 = nothing),
    NCCOL = (median = 1e-2, p95 = 1.5e-2),
    QRFRZ = (median = 5e-3, p95 = 1e-2),
    QRSHD = (median = 5e-3, p95 = nothing),
    NRCOL = (median = 5e-3, p95 = 5e-3),
    ∫M_col = (median = 2e-3, p95 = 2e-2),
    BCCOL = (median = 2e-2, p95 = 3e-2),
    BRCOL = (median = 2e-3, p95 = 2e-2),
    ∫𝟙_wet_M_col = (median = 2e-2, p95 = nothing),
)

function comprehensive_raw_collision(mp, tps, s)
    (; quad, terminal_velocity, cloud_pdf, rain_pdf) = mp.ice
    aps = mp.warm_rain.air_properties
    (; ρ, T, q_lcl, n_lcl, q_rai, n_rai, q_ice, n_ice, q_rim, b_rim) = s
    FT = typeof(ρ)
    (q_ice > 0 && n_ice > 0) || return nothing
    state = P3.state_from_prognostic(mp.ice.scheme, ρ * q_ice, ρ * n_ice, ρ * q_rim, ρ * b_rim)
    logλ = P3.get_distribution_logλ(state)
    m_liq(Dₗ) = cloud_pdf.ρw * CO.volume_sphere_D(Dₗ)
    rates = P3.∫liquid_ice_collisions(
        state, logλ, cloud_pdf, rain_pdf, ρ * q_lcl, n_lcl, ρ * q_rai, n_rai,
        aps, tps, terminal_velocity, ρ, T, m_liq; quad,
    )
    return collect(FT, rates)
end

function test_comprehensive_battery_accuracy_guard(FT)
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    battery = generate_comprehensive_states(FT; n_lhs = 300, seed = 1)
    states = vcat(generate_column_states(FT), hail_core_states(FT, STUDY_HAIL_CORES), battery)
    @test length(states) >= 300

    @testset "Level aggregates (median/p95), GL(6) vs GL(128)" begin
        results = run_quadrature_error_study(; FT, orders = (6,), reference_order = 128, states)
        for row in results
            tol = getfield(COMPREHENSIVE_LEVEL_TOLERANCES, row.level)
            @test row.median < tol.median
            @test row.p95 < tol.p95
        end
    end

    @testset "Per-raw-collision-component (median/p95), GL(6) vs GL(128)" begin
        collision_labels = (:QCFRZ, :QCSHD, :NCCOL, :QRFRZ, :QRSHD, :NRCOL, :∫M_col, :BCCOL, :BRCOL, :∫𝟙_wet_M_col)
        mp6 = CMP.Microphysics2MParams(FT; with_ice = true, quad = CM.Quadrature.GaussLegendre(FT, 6))
        mp128 = CMP.Microphysics2MParams(FT; with_ice = true, quad = CM.Quadrature.GaussLegendre(FT, 128))
        refs = [comprehensive_raw_collision(mp128, tps, s) for s in states]
        floors = FT(1e-9) .* max.(
            [maximum(abs(r[i]) for r in refs if r !== nothing) for i in 1:length(collision_labels)],
            eps(FT),
        )
        errs = Dict(l => FT[] for l in collision_labels)
        for (s, ref) in zip(states, refs)
            ref === nothing && continue
            val = comprehensive_raw_collision(mp6, tps, s)
            for (i, l) in enumerate(collision_labels)
                push!(errs[l], abs(val[i] - ref[i]) / max(abs(val[i]), abs(ref[i]), floors[i]))
            end
        end
        for l in collision_labels
            tol = getfield(COMPREHENSIVE_COMPONENT_TOLERANCES, l)
            v = errs[l]
            @test median(v) < tol.median
            tol.p95 === nothing || @test quantile(v, 0.95) < tol.p95
        end
    end
end

@testset "Comprehensive-battery accuracy guard (Float64)" begin
    test_comprehensive_battery_accuracy_guard(Float64)
end
nothing
