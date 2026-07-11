using Test
import ClimaParams as CP
import StaticArrays: SVector

import CloudMicrophysics as CM
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.ThermodynamicsInterface as TDI
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

include("p3_quadrature_error_study.jl")  # generate_column_states, hail_core_states, STUDY_HAIL_CORES

const TABLE_FLOOR_FRAC = 1e-3
const RATE_KEYS =
    (:dq_lcl_dt, :dn_lcl_dt, :dq_rai_dt, :dn_rai_dt, :dq_ice_dt, :dn_ice_dt, :dq_rim_dt, :db_rim_dt)

# Relative error, normalizing near-zero entries by a fraction of the field's
# dynamic range so a negligible component cannot report a spurious 100% error.
relerr(table, quad, scale) = abs(table - quad) / max(abs(quad), TABLE_FLOOR_FRAC * scale)

function logλ_of(mp, s)
    state = P3.state_from_prognostic(
        mp.ice.scheme, s.ρ * s.q_ice, s.ρ * s.n_ice, s.ρ * s.q_rim, s.ρ * s.b_rim,
    )
    return P3.get_distribution_logλ(state)
end

function variant_c_tables(::Type{FT}) where {FT}
    mp = CMP.Microphysics2MParams(FT; with_ice = true)
    params, vel, aps = mp.ice.scheme, mp.ice.terminal_velocity, mp.warm_rain.air_properties
    psd_c, psd_r = mp.ice.cloud_pdf, mp.ice.rain_pdf
    rgrid = P3.P3TableGrid{FT}(n_logλ = 48, n_F_rim = 14, n_ρ_rim = 10, n_ρ_air = 5)
    rate_tables = P3.build_p3_lookup_tables(params, vel, aps; grid = rgrid)
    inner_tables = P3.build_p3_collision_inner_tables(
        params, vel, aps, psd_c, psd_r; quad = CM.Quadrature.GaussLegendre(FT, 16),
    )
    tables = BMT.P3IceTables(
        rate_tables, nothing, inner_tables, CM.Quadrature.GaussLegendre(FT, 8), BMT.P3CollisionVariantC(),
    )
    return mp, tables
end

# Compare the manual RosenbrockAverage mode with variant-C tables against the
# quadrature-backed mode over mixed-phase and hail-core states, and check the
# table-backed substep is allocation-free. See #741 and p3_collision_variant_tests.jl.
function run_p3_tables_case(::Type{FT}, p95_tol) where {FT}
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp, tables = variant_c_tables(FT)
    mode = BMT.rosenbrock_manual()
    Δt = FT(1)

    states = filter(
        s -> s.q_ice > 0 && s.n_ice > 0 && (s.q_lcl > 0 || s.q_rai > 0),
        vcat(generate_column_states(FT), hail_core_states(FT, STUDY_HAIL_CORES)),
    )
    @test !isempty(states)

    quad_of(s) = BMT.bulk_microphysics_tendencies(
        mode, BMT.Microphysics2Moment(), mp, tps,
        s.ρ, s.T, s.q_tot, s.q_lcl, s.n_lcl, s.q_rai, s.n_rai,
        s.q_ice, s.n_ice, s.q_rim, s.b_rim, logλ_of(mp, s), Δt,
    )
    table_of(s) = BMT.bulk_microphysics_tendencies(
        mode, BMT.Microphysics2Moment(), mp, tps,
        s.ρ, s.T, s.q_tot, s.q_lcl, s.n_lcl, s.q_rai, s.n_rai,
        s.q_ice, s.n_ice, s.q_rim, s.b_rim, logλ_of(mp, s), Δt; p3_tables = tables,
    )
    # Positional p3_tables form used by broadcasts on GPU fields.
    table_pos_of(s) = BMT.bulk_microphysics_tendencies(
        mode, BMT.Microphysics2Moment(), mp, tps,
        s.ρ, s.T, s.q_tot, s.q_lcl, s.n_lcl, s.q_rai, s.n_rai,
        s.q_ice, s.n_ice, s.q_rim, s.b_rim, logλ_of(mp, s), Δt, 1, tables,
    )
    @test all(s -> table_pos_of(s) === table_of(s), states)
    # P3IceTables must broadcast as a scalar so it can be a positional broadcast
    # argument (as ClimaAtmos passes it on GPU fields).
    @test Base.broadcastable(tables) === (tables,)
    take_tables(_, t) = t
    @test all(take_tables.([1, 2, 3], tables) .=== tables)

    quad_res = map(quad_of, states)
    table_res = map(table_of, states)

    @test all(r -> keys(r) === RATE_KEYS, table_res)
    @test all(r -> all(isfinite, values(r)), table_res)
    @test all(i -> typeof(table_res[i]) === typeof(quad_res[i]), eachindex(states))

    errs = FT[]
    for k in RATE_KEYS
        scale = maximum(r -> abs(getfield(r, k)), quad_res)
        for i in eachindex(states)
            push!(errs, relerr(getfield(table_res[i], k), getfield(quad_res[i], k), scale))
        end
    end
    sort!(errs)
    p95 = errs[max(1, ceil(Int, 0.95 * length(errs)))]
    @info "RosenbrockAverage manual variant-C vs quadrature" FT p95 max = last(errs)
    @test p95 < p95_tol

    s = first(states)
    call() = table_of(s)
    call()
    @test iszero(@allocated call())
    call_pos() = table_pos_of(s)
    call_pos()
    @test iszero(@allocated call_pos())
    return nothing
end

@testset "RosenbrockAverage manual: p3_tables matches quadrature mode" begin
    # Documented variant-C envelope (p95 < 1e-1, see p3_collision_variant_tests.jl).
    @testset "Float64" begin
        run_p3_tables_case(Float64, 1e-1)
    end
    @testset "Float32" begin
        run_p3_tables_case(Float32, 1e-1)
    end
end
nothing
