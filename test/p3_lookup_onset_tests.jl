using Test

import CloudMicrophysics as CM
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.Utilities as UT

# The table-backed P3 slope and rates stay finite and C0-continuous across ice
# onset: no collapse to logλ = -Inf, no clamp to the largest-particle table edge.

function onset_tables(::Type{FT}) where {FT}
    mp = CMP.Microphysics2MParams(FT; with_ice = true, quad = CM.Quadrature.GaussLegendre(FT, 12))
    params, vel, aps = mp.ice.scheme, mp.ice.terminal_velocity, mp.warm_rain.air_properties
    grid = P3.P3TableGrid{FT}(n_logλ = 24, n_F_rim = 8, n_ρ_rim = 6, n_ρ_air = 4)
    return params, P3.build_p3_lookup_tables(params, vel, aps; grid)
end

maxrelstep(r) = maximum(abs, diff(r)) / (maximum(abs, r) + eps(eltype(r)))

@testset "P3 table onset continuity" begin
    for FT in (Float64, Float32)
        params, tables = onset_tables(FT)
        ϵₘ = UT.ϵ_numerics_2M_M(FT)
        F_rim, ρ_rim, ρₐ = FT(0.4), FT(400), FT(1)

        @testset "$FT tables finite at the onset edge" begin
            @test all(isfinite, tables.shape.data)
            @test all(isfinite, tables.rates.data)
        end

        # Sweep ρq_ice across the mass threshold ϵₘ at fixed number.
        ρn_ice = FT(100) * ϵₘ / FT(1e-8)
        ρq = FT.(exp10.(range(log10(FT(0.1) * ϵₘ), log10(FT(100) * ϵₘ); length = 80)))
        sweep = [P3.P3State(params, q, ρn_ice, F_rim, ρ_rim) for q in ρq]
        lλ_solver = [P3.get_distribution_logλ(s) for s in sweep]
        lλ_table = [P3.get_distribution_logλ(tables, s) for s in sweep]

        @testset "$FT logλ finite and continuous across onset" begin
            @test all(isfinite, lλ_solver)
            @test all(isfinite, lλ_table)
            @test maximum(abs, diff(lλ_solver)) < FT(2)
            @test maximum(abs, diff(lλ_table)) < FT(0.5)
        end

        @testset "$FT table rates continuous, no edge read" begin
            selfcol = [P3.ice_self_collection(tables, s, l, ρₐ).dNdt for (s, l) in zip(sweep, lλ_table)]
            vN = [P3.ice_terminal_velocity_number_weighted(tables, s, l, ρₐ) for (s, l) in zip(sweep, lλ_table)]
            vM = [P3.ice_terminal_velocity_mass_weighted(tables, s, l, ρₐ) for (s, l) in zip(sweep, lλ_table)]
            for r in (selfcol, vN, vM)
                @test all(isfinite, r)
                @test maxrelstep(r) < FT(0.2)
            end
            # The largest-particle edge reads ≈6 m/s; onset fall speeds are well below.
            @test all(<(FT(3)), vN)
            @test all(<(FT(4)), vM)
        end

        # Mean mass above the Float32 shape-table floor, where the table resolves.
        x = FT.(exp10.(range(log10(FT(2) * ϵₘ + FT(3e-7)), log10(FT(1e-4)); length = 60)))
        interior = [P3.P3State(params, xi, one(FT), F_rim, ρ_rim) for xi in x]
        @testset "$FT table smooth over the resolvable interior" begin
            lλ = [P3.get_distribution_logλ(tables, s) for s in interior]
            vN = [P3.ice_terminal_velocity_number_weighted(tables, s, l, ρₐ) for (s, l) in zip(interior, lλ)]
            vM = [P3.ice_terminal_velocity_mass_weighted(tables, s, l, ρₐ) for (s, l) in zip(interior, lλ)]
            @test all(isfinite, lλ) && all(isfinite, vN) && all(isfinite, vM)
            @test maximum(abs, diff(lλ)) < FT(0.6)
            @test vN[end] > vN[1] && vM[end] > vM[1]
        end
    end
end
nothing
