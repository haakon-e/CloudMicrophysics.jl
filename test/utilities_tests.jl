using Test
import CloudMicrophysics.Utilities as UT

# Regression test for the `eps(FT)`-tied regularization bug in `rime_density`/
# `rime_mass_fraction`: their old default smoothing scale, `eps(typeof(denominator))`,
# is ~9 orders of magnitude larger in `Float32` than `Float64`, so an ordinary
# (non-degenerate) rime ratio was spuriously suppressed by ~10% in `Float32` alone.
# Fixed by physically-motivated, `FT`-independent default scales.
@testset "rime_density / rime_mass_fraction: Float32-Float64 consistency" begin
    @testset "exact state that exposed the bug" begin
        # The `(q_rim, b_rim)` pair from the comprehensive-battery state whose
        # `Float32` `TruncatedMM` full-tendency error (17.9%) traced back to this
        # regularization: `ρ_rim` differed by ~10% between precisions before the
        # fix (100.36 vs 90.32), not ordinary rounding.
        q_rim, b_rim = 1.2176073047297676e-4, 1.2132270859560276e-6
        ρ_rim64 = UT.rime_density(q_rim, b_rim)
        ρ_rim32 = UT.rime_density(Float32(q_rim), Float32(b_rim))
        @test isapprox(Float64(ρ_rim32), ρ_rim64; rtol = 1e-5)
    end

    @testset "sweep over a physically plausible (F_rim, ρ_rim) grid" begin
        # `ρ_rim` should agree between `Float32` and `Float64` to ordinary
        # single-precision rounding (not the ~10% regularization artifact) across
        # the full physical range: light-to-heavy riming, low-to-high rime density.
        for q_ice in (1e-6, 1e-4, 1e-2), F_rim in (0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99),
            ρ_rim_true in (50.0, 100.0, 400.0, 700.0, 900.0)

            q_rim = F_rim * q_ice
            b_rim = q_rim / ρ_rim_true
            ρ_rim64 = UT.rime_density(q_rim, b_rim)
            ρ_rim32 = UT.rime_density(Float32(q_rim), Float32(b_rim))
            relerr = abs(Float64(ρ_rim32) - ρ_rim64) / ρ_rim64
            @test relerr < 1e-4
        end
    end

    @testset "sweep over a physically plausible F_rim grid (rime_mass_fraction)" begin
        for q_ice in (1e-6, 1e-4, 1e-2), F_rim_true in (0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
            q_rim = F_rim_true * q_ice
            F_rim64 = UT.rime_mass_fraction(q_rim, q_ice)
            F_rim32 = UT.rime_mass_fraction(Float32(q_rim), Float32(q_ice))
            relerr = abs(Float64(F_rim32) - F_rim64) / F_rim64
            @test relerr < 1e-4
        end
    end

    @testset "genuinely negligible rime/ice still regularizes to zero" begin
        # The fix must not weaken the degenerate-case protection: a truly
        # negligible denominator (far below the new physical scales) still
        # returns a finite, small result, in both precisions.
        for FT in (Float64, Float32)
            @test isfinite(UT.rime_density(FT(0), FT(0)))
            @test UT.rime_density(FT(0), FT(0)) == 0
            @test isfinite(UT.rime_mass_fraction(FT(0), FT(0)))
            @test UT.rime_mass_fraction(FT(0), FT(0)) == 0
        end
    end
end
nothing
