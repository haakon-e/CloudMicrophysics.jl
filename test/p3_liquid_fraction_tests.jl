using Test: @testset, @test, @test_throws, @inferred
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.Common as CO
import CloudMicrophysics.DistributionTools as DT
import CloudMicrophysics.Utilities as UT
import CloudMicrophysics.ThermodynamicsInterface as TDI
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.MicrophysicsNonEq as CMNonEq
import CloudMicrophysics.Microphysics2M as CM2
import CloudMicrophysics.HetIceNucleation as CM_HetIce
import SpecialFunctions as SF
import QuadGK as QGK
import ForwardDiff as FD

# ρq_liq consistent with a target F_liq on a frozen core ρq_ice.
_ρq_liq_from_F(ρq_ice, F_liq) = ρq_ice * F_liq / (1 - F_liq)

function test_liquid_fraction_state(FT)
    @testset "state construction and F_liq" begin
        p = CMP.ParametersP3(FT; liquid = :predicted)
        ρq_ice, ρn_ice, ρq_rim, ρb_rim = FT(8e-4), FT(2e5), FT(3e-4), FT(1e-6)

        # F_liq = 0 recovers the dry state exactly (feature on, no liquid)
        st0 = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, FT(0))
        @test st0.F_liq == 0
        @test P3.total_mass_concentration(st0) == ρq_ice

        # regularised ratio at physical magnitudes and the q_tot = sum recovery
        ρq_liq = _ρq_liq_from_F(ρq_ice, FT(0.3))
        st = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, ρq_liq)
        @test st.F_liq ≈ FT(0.3) rtol = 1e-4
        @test P3.total_mass_concentration(st) ≈ ρq_ice + ρq_liq rtol = 1e-4
        # F_rim keeps the frozen-core normalisation (independent of F_liq)
        @test st.F_rim ≈ ρq_rim / ρq_ice rtol = 1e-4
        @test st.F_rim == st0.F_rim
        # thresholds independent of F_liq
        @test (st.D_th, st.D_gr, st.D_cr) == (st0.D_th, st0.D_gr, st0.D_cr)

        # clamp into [0, F_melt] before use
        big = _ρq_liq_from_F(ρq_ice, FT(0.999))
        st_hi = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, big)
        @test st_hi.F_liq == p.liquid.F_melt
        # negative liquid floored to zero
        st_neg = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, FT(-1e-3))
        @test st_neg.F_liq == 0

        # NoLiquidFraction forces F_liq = 0
        pn = CMP.ParametersP3(FT)
        stn = P3.P3State(pn, ρq_ice, ρn_ice, FT(0.3), FT(500), FT(0.5))
        @test stn.F_liq == 0
    end
end

function test_liquid_mass_fraction_onset(FT)
    @testset "liquid_mass_fraction onset value + derivative" begin
        q_present = FT(1e-10)
        # value + derivative at ρq_liq → 0 for a range of ρq_ice, including thin ice
        for ρq_ice in FT.((1e-3, 1e-5, 1e-7, 1e-9))
            F(ρq_liq) = UT.liquid_mass_fraction(ρq_liq, ρq_ice + ρq_liq, q_present)
            v0 = F(zero(FT))
            d0 = FD.derivative(F, zero(FT))
            @test v0 == 0
            @test isfinite(d0)
            # ∂F/∂ρq_liq |₀ = 1/(ρq_ice + q_present)
            @test d0 ≈ 1 / (ρq_ice + q_present) rtol = 1e-4
            # a small liquid perturbation is finite and non-negative
            v1 = F(FT(1e-2) * ρq_ice)
            @test isfinite(v1) && 0 ≤ v1 < 1
        end
        # finite as ρq_tot → 0 (no eps-scale step)
        @test UT.liquid_mass_fraction(zero(FT), zero(FT), q_present) == 0
        @test isfinite(UT.liquid_mass_fraction(FT(1e-12), FT(1e-12), q_present))
    end
end

function test_particle_property_blends(FT)
    @testset "mixed_mass / mixed_area / velocity blends" begin
        p = CMP.ParametersP3(FT; liquid = :predicted)
        vel = CMP.Chen2022VelType(FT)
        ρₐ = FT(1)
        ρq_ice, ρn_ice, ρq_rim, ρb_rim = FT(5e-4), FT(1e5), FT(2e-4), FT(6e-7)

        # liquid_blend endpoints
        @test P3.liquid_blend(zero(FT), FT(2), FT(9)) == 2
        @test P3.liquid_blend(one(FT), FT(2), FT(9)) == 9

        st0 = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, FT(0))
        Ds = FT.((1e-6, 5e-5, 1e-4, 5e-4, 2e-3, 1e-2))
        for D in Ds
            # F_liq = 0 ⇒ blends reduce to the ice-core relations
            @test P3.mixed_mass(st0, D) == P3.ice_mass(st0, D)
            @test P3.mixed_area(st0, D) == P3.ice_area(st0, D)
        end

        # drop-branch finiteness over the quadrature support and the drop limit
        st = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, _ρq_liq_from_F(ρq_ice, FT(0.6)))
        v_mix = P3.mixed_particle_terminal_velocity(vel, ρₐ, st)
        for D in (FT(0), Ds...)
            m = P3.mixed_mass(st, D)
            a = P3.mixed_area(st, D)
            @test isfinite(m) && m ≥ 0
            @test isfinite(a) && a ≥ 0
            @test isfinite(v_mix(D)) && v_mix(D) ≥ 0
            # blend lies between the ice-core and pure-drop values
            m_drop = p.ρ_l * CO.volume_sphere_D(D)
            @test min(P3.ice_mass(st, D), m_drop) - eps(FT) ≤ m ≤ max(P3.ice_mass(st, D), m_drop) + eps(FT)
        end

        # onset value + ForwardDiff derivative w.r.t. F_liq and ρq_liq at F_liq → 0,
        # including a thin core (ρq_ice ~ 1e-9); shape frozen (not differentiated).
        D = FT(3e-4)
        for ρqi in FT.((1e-3, 1e-9))
            # w.r.t. F_liq directly (via the P3State constructor)
            mF(f) = P3.mixed_mass(P3.P3State(p, ρqi, ρn_ice, FT(0.4), FT(500), f), D)
            @test mF(zero(FT)) == P3.ice_mass(P3.P3State(p, ρqi, ρn_ice, FT(0.4), FT(500), FT(0)), D)
            @test isfinite(FD.derivative(mF, zero(FT)))
            # w.r.t. ρq_liq through the regularised ratio
            mL(ρl) = P3.mixed_mass(P3.state_from_prognostic(p, ρqi, ρn_ice, ρqi * FT(0.4), ρqi * FT(0.4) / 500, ρl), D)
            @test isfinite(FD.derivative(mL, zero(FT)))
            @test isfinite(FD.derivative(mL, FT(1e-2) * ρqi))
        end
    end
end

function test_whole_mass_moment(FT)
    @testset "whole-particle mass moment" begin
        p = CMP.ParametersP3(FT; liquid = :predicted)
        ρq_ice, ρn_ice, ρq_rim, ρb_rim = FT(6e-4), FT(1e5), FT(1e-4), FT(3e-7)
        st = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, _ρq_liq_from_F(ρq_ice, FT(0.5)))
        μ, logλ = FT(1.5), FT(9.0)
        λ = exp(logλ)

        # value == core exactly at F_liq = 0
        core = P3.logmass_gamma_moment(st, μ, logλ; n = 0)
        @test P3.log_mixed_mass_moment(st, μ, logλ, zero(FT); n = 0) == core

        # derivative finite at F_liq = 0 and equals expm1(liq0 - core)
        liq0 = P3.loggamma_moment(μ, logλ; k = 3, scale = π * p.ρ_l / 6)
        d0 = FD.derivative(f -> P3.log_mixed_mass_moment(st, μ, logλ, f; n = 0), zero(FT))
        @test isfinite(d0)
        @test d0 ≈ expm1(liq0 - core) rtol = 1e-4

        # value vs QuadGK reference for several F_liq
        for F_liq in FT.((0.0, 0.1, 0.5, 0.9))
            stF = P3.P3State(p, ρq_ice, ρn_ice, st.F_rim, st.ρ_rim, F_liq)
            for n in (0, 1)
                lhs = exp(P3.log_mixed_mass_moment(stF, μ, logλ, stF.F_liq; n))
                rhs, _ = QGK.quadgk(D -> P3.mixed_mass(stF, D) * D^n * D^μ * exp(-λ * D), FT(0), FT(0.1))
                @test lhs ≈ rhs rtol = (FT === Float32 ? 1e-3 : 1e-6)
            end
        end

        # exponent margin: `exp(liq0 - core)` needs no overflow shift because
        # `liq0 - core` stays far below the Float32 overflow threshold (~88.7)
        # over the slope-solve bracket, μ range, and rime states; assert the
        # bound directly, with the F_liq = 0 identity and derivative alongside.
        for F_rim in FT.((0, 0.5, 0.9)), ρqi in FT.((1e-6, 5e-3)), μx in (FT(0), p.moments.slope.μ_max)
            st_ext = P3.state_from_prognostic(
                p, ρqi, FT(1e3), F_rim * ρqi, F_rim * ρqi / 400, _ρq_liq_from_F(ρqi, FT(0.9)),
            )
            for lλ in FT.((2, 5, 10, 17))
                core = P3.logmass_gamma_moment(st_ext, μx, lλ; n = 0)
                liq0 = P3.loggamma_moment(μx, lλ; k = 3, scale = π * p.ρ_l / 6)
                @test liq0 - core < 80
                @test isfinite(P3.log_mixed_mass_moment(st_ext, μx, lλ, st_ext.F_liq; n = 0))
                @test P3.log_mixed_mass_moment(st_ext, μx, lλ, zero(FT); n = 0) == core
                @test isfinite(FD.derivative(f -> P3.log_mixed_mass_moment(st_ext, μx, lλ, f; n = 0), zero(FT)))
            end
        end
    end
end

function test_two_slope_shape(FT)
    @testset "two-slope shape machinery" begin
        p = CMP.ParametersP3(FT; liquid = :predicted)
        pn = CMP.ParametersP3(FT)
        rtol = FT === Float32 ? 1e-2 : 1e-4

        for (F_rim, ρ_rim, ρq_ice, ρn_ice, F_liq) in (
            (FT(0), FT(0), FT(1e-4), FT(1e5), FT(0.3)),
            (FT(0.6), FT(500), FT(1e-3), FT(2e5), FT(0.5)),
            (FT(0.9), FT(800), FT(2e-4), FT(5e4), FT(0.8)),
        )
            ρq_rim, ρb_rim = F_rim * ρq_ice, (ρ_rim > 0 ? F_rim * ρq_ice / ρ_rim : FT(0))
            st = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, _ρq_liq_from_F(ρq_ice, F_liq))
            sh = P3.get_distribution_shape(st)
            @test sh.μ == P3.get_μ(st, sh.logλ)  # shared μ from the whole slope

            # whole PSD carries q_tot with the blended mass
            n_whole = DT.size_distribution(st, sh)
            Iw, _ = QGK.quadgk(D -> P3.mixed_mass(st, D) * n_whole(D), FT(0), FT(0.1))
            @test Iw ≈ P3.total_mass_concentration(st) rtol = rtol
            # core PSD (logλ_core, shared μ) carries the frozen core mass
            core = P3.P3Shape(; logλ = sh.logλ_core, μ = sh.μ)
            n_core = DT.size_distribution(st, core)
            Ic, _ = QGK.quadgk(D -> P3.ice_mass(st, D) * n_core(D), FT(0), FT(0.1))
            @test Ic ≈ ρq_ice rtol = rtol
        end

        # F_liq → 0 limit: logλ_core ≈ logλ_whole (intended physics, not exact reproduction)
        st_dry = P3.state_from_prognostic(p, FT(1e-3), FT(2e5), FT(3e-4), FT(6e-7), FT(1e-12))
        sh_dry = P3.get_distribution_shape(st_dry)
        @test sh_dry.logλ_core ≈ sh_dry.logλ rtol = 1e-3

        # NoLiquidFraction: logλ_core == logλ exactly
        st_no = P3.state_from_prognostic(pn, FT(1e-3), FT(2e5), FT(3e-4), FT(6e-7))
        sh_no = P3.get_distribution_shape(st_no)
        @test sh_no.logλ_core == sh_no.logλ
    end
end

function test_shape_iteration_budget(FT)
    @testset "whole/core shape-solve iteration budget" begin
        p = CMP.ParametersP3(FT; liquid = :predicted)
        Bw = FT === Float32 ? 12 : 14
        Bc = Bw
        function solve_whole(st, iters)
            ϵₘ, ϵₙ = UT.ϵ_numerics_2M_M(FT), UT.ϵ_numerics_2M_N(FT)
            tgt = log(max(P3.total_mass_concentration(st), ϵₘ)) - log(max(st.ρn_ice, ϵₙ))
            sp(logλ) = P3.logLdivN_whole(st, logλ, st.F_liq) - tgt
            return P3._solve_shape_logλ(sp, FT, nothing, FT(2), FT(17), iters)
        end
        function solve_core(st, μ, iters)
            ϵₘ, ϵₙ = UT.ϵ_numerics_2M_M(FT), UT.ϵ_numerics_2M_N(FT)
            tgt = log(max(st.ρq_ice, ϵₘ)) - log(max(st.ρn_ice, ϵₙ))
            sp(logλ) = P3.logLdivN(st, μ, logλ) - tgt
            return P3._solve_shape_logλ(sp, FT, nothing, FT(2), FT(17), iters)
        end
        errs = Float64[]
        for F_rim in FT.((0, 0.3, 0.6, 0.9)), ρ_rim in FT.((200, 500, 800)),
            ρq_ice in FT.((1e-6, 1e-5, 1e-4, 1e-3, 5e-3)), ρn_ice in FT.((1e3, 1e5, 1e7)),
            F_liq in FT.((0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95))

            ρq_rim, ρb_rim = F_rim * ρq_ice, F_rim * ρq_ice / ρ_rim
            st = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, _ρq_liq_from_F(ρq_ice, F_liq))
            wref = solve_whole(st, 40)
            μref = P3.get_μ(st, wref)
            cref = solve_core(st, μref, 40)
            ws = solve_whole(st, Bw)
            μs = P3.get_μ(st, ws)
            cs = solve_core(st, μs, Bc)
            push!(errs, abs(Float64(ws) - Float64(wref)))
            push!(errs, abs(Float64(cs) - Float64(cref)))
        end
        # A few extreme mean-mass corners stay ill-conditioned regardless of
        # budget (as does the dry solve); bound the corners and require the
        # remaining states to match the high-iteration reference tightly.
        sorted = sort(errs)
        @test sorted[end] < 0.05
        @test count(>(1e-2), errs) ≤ 4
        @test sorted[end - 4] < 1e-3
    end
end

function test_ice_melt_liquid(FT)
    @testset "ice_melt liquid-fraction rework" begin
        p = CMP.ParametersP3(FT; liquid = :predicted)
        vel = CMP.Chen2022VelType(FT)
        aps = CMP.AirProperties(FT)
        tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
        quad = P3.GaussLegendre(FT, 12)
        ρₐ = FT(1)
        T_frz = p.T_freeze
        ρq_ice, ρn_ice, ρq_rim, ρb_rim = FT(8e-4), FT(2e5), FT(3e-4), FT(1e-6)
        st = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, _ρq_liq_from_F(ρq_ice, FT(0.3)))
        sh = P3.get_distribution_shape(st)

        # below freezing: no melt
        m_cold = P3.ice_melt(vel, aps, tps, T_frz - FT(0.01), ρₐ, st, sh; quad)
        @test m_cold.dLdt_rain == 0 && m_cold.dLdt_liq == 0 && m_cold.dLdt_ice == 0

        # above freezing: partition, signs, and rime drain
        m = P3.ice_melt(vel, aps, tps, T_frz + FT(2), ρₐ, st, sh; quad)
        @test m.dLdt_rain ≥ 0 && m.dLdt_liq ≥ 0 && m.dNdt_rain ≥ 0
        @test m.dLdt_rim ≤ 0 && m.dBdt_rim ≤ 0
        # explicit double entry: the frozen core loses what rain and liquid gain
        @test m.dLdt_ice == -(m.dLdt_rain + m.dLdt_liq)
        # rime drains proportionally to the total core-mass loss at fixed F_rim
        ΔL = m.dLdt_rain + m.dLdt_liq
        @test m.dLdt_rim ≈ -ΔL * st.F_rim rtol = 1e-5
        @test m.dBdt_rim ≈ -ΔL * st.F_rim / st.ρ_rim rtol = 1e-5
        # number reduced only by the complete-melt (rain) part
        @test m.dNdt_rain ≈ st.ρn_ice / st.ρq_ice * m.dLdt_rain rtol = 1e-5

        # the split total equals a single unsplit core-PSD melt integral
        # (ventilation at the blended whole-particle fall speed, C19 Eq A3)
        L_f = TDI.Lf(tps, T_frz + FT(2))
        v_term = P3.mixed_particle_terminal_velocity(vel, ρₐ, st)
        F_v = CO.ventilation_factor(p.vent, aps, v_term)
        core = P3.P3Shape(; logλ = sh.logλ_core, μ = sh.μ)
        N′ = DT.size_distribution(st, core)
        fac = 4 * aps.K_therm / L_f * FT(2)
        bnds = P3.velocity_integral_bounds(st, core, v_term; p = 1e-6)
        total = fac * P3.integrate(D -> P3.∂ice_mass_∂D(st, D) * F_v(D) * N′(D) / D, bnds, quad)
        @test m.dLdt_rain + m.dLdt_liq ≈ total rtol = 1e-5
        # wet ventilation melts faster than the dry-core ventilation would
        v_dry = P3.ice_particle_terminal_velocity(vel, ρₐ, st)
        F_v_dry = CO.ventilation_factor(p.vent, aps, v_dry)
        total_dry = fac * P3.integrate(D -> P3.∂ice_mass_∂D(st, D) * F_v_dry(D) * N′(D) / D, bnds, quad)
        @test total > total_dry

        # unrimed ice does not drain rime
        st_u = P3.state_from_prognostic(p, ρq_ice, ρn_ice, FT(0), FT(0), _ρq_liq_from_F(ρq_ice, FT(0.3)))
        sh_u = P3.get_distribution_shape(st_u)
        m_u = P3.ice_melt(vel, aps, tps, T_frz + FT(2), ρₐ, st_u, sh_u; quad)
        @test m_u.dLdt_rim == 0 && m_u.dBdt_rim == 0
    end
end

function test_ice_refreeze(FT)
    @testset "ice_refreeze" begin
        p = CMP.ParametersP3(FT; liquid = :predicted)
        vel = CMP.Chen2022VelType(FT)
        aps = CMP.AirProperties(FT)
        tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
        quad = P3.GaussLegendre(FT, 12)
        ρₐ = FT(1)
        T_frz = p.T_freeze
        ρq_ice, ρn_ice, ρq_rim, ρb_rim = FT(6e-4), FT(1e5), FT(2e-4), FT(5e-7)
        st = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, _ρq_liq_from_F(ρq_ice, FT(0.5)))
        sh = P3.get_distribution_shape(st)

        # above freezing: no refreeze
        r_warm = P3.ice_refreeze(vel, aps, tps, T_frz + FT(1), ρₐ, st, sh; quad)
        @test r_warm.dLdt_ice == 0

        # below freezing: conserves total mass, grows rime at ρ_i
        r = P3.ice_refreeze(vel, aps, tps, T_frz - FT(5), ρₐ, st, sh; quad)
        @test r.dLdt_ice ≥ 0
        @test r.dLdt_liq == -r.dLdt_ice          # q_tot conserved
        @test r.dLdt_rim == r.dLdt_ice           # frozen liquid joins the rime
        @test r.dBdt_rim ≈ r.dLdt_rim / p.ρ_i rtol = 1e-6

        # continuous to zero at the freezing temperature
        r_edge = P3.ice_refreeze(vel, aps, tps, T_frz - FT(1e-4), ρₐ, st, sh; quad)
        @test 0 ≤ r_edge.dLdt_ice < r.dLdt_ice
        # scales with F_liq (dry ice does not refreeze)
        st_dry = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, FT(0))
        sh_dry = P3.get_distribution_shape(st_dry)
        @test P3.ice_refreeze(vel, aps, tps, T_frz - FT(5), ρₐ, st_dry, sh_dry; quad).dLdt_ice == 0
    end
end

function test_ice_shed(FT)
    @testset "ice_shed closed form" begin
        p = CMP.ParametersP3(FT; liquid = :predicted)
        pn = CMP.ParametersP3(FT)
        D₀, D_drop = p.liquid.D_shd_onset, p.liquid.D_shd_drop
        m_drop = p.ρ_l * CO.volume_sphere_D(D_drop)

        # a state with hail-sized ice so shedding is non-negligible (small λ)
        ρq_ice, ρn_ice, ρq_rim, ρb_rim = FT(3e-3), FT(1e3), FT(2.4e-3), FT(3e-6)
        st = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, _ρq_liq_from_F(ρq_ice, FT(0.4)))
        sh = P3.get_distribution_shape(st)
        s = P3.ice_shed(st, sh)

        # closed form vs QuadGK reference for ∫_{D₀}^∞ D³ n dD
        n_whole = DT.size_distribution(st, sh)
        Iref, _ = QGK.quadgk(D -> D^3 * n_whole(D), D₀, FT(1))
        L_ref = st.F_rim * st.F_liq * (π * p.ρ_l / 6) * Iref
        @test s.L_shd ≈ L_ref rtol = (FT === Float32 ? 1e-3 : 1e-6)
        @test s.N_shd ≈ s.L_shd / m_drop rtol = 1e-6
        @test s.L_shd > 0

        # the closed-form moment is bilinear in (F_rim, F_liq) at a fixed PSD:
        # pass the same frozen shape and vary only the state prefactors
        st_a = P3.P3State(p, ρq_ice, ρn_ice, FT(0.4), FT(600), FT(0.2))
        st_b = P3.P3State(p, ρq_ice, ρn_ice, FT(0.8), FT(600), FT(0.4))
        s_a = P3.ice_shed(st_a, sh)
        s_b = P3.ice_shed(st_b, sh)
        @test s_b.L_shd / s_a.L_shd ≈ (st_b.F_rim * st_b.F_liq) / (st_a.F_rim * st_a.F_liq) rtol = 1e-4

        # deep-tail depths (D10): fix the slope so the onset sits x decay lengths
        # in, and compare the closed form against a Float64 QuadGK reference. At
        # x ≈ 90 the Float32 linear-space output is subnormal-quantized (the
        # log-space value stays accurate, see the BigFloat checks below), so
        # only the x = 60 depth carries a tight Float32 tolerance.
        p64 = CMP.ParametersP3(Float64; liquid = :predicted)
        for (x_decay, rtol32) in ((60, 1e-3), (90, 2e-2))
            logλx = FT(log(x_decay / 0.009))
            shx = P3.P3Shape(; logλ = logλx, μ = FT(1))
            stx = P3.P3State(p, ρq_ice, ρn_ice, FT(0.5), FT(600), FT(0.4))
            sx = P3.ice_shed(stx, shx)
            nx = DT.size_distribution(
                P3.P3State(p64, Float64(ρq_ice), Float64(ρn_ice), 0.5, 600.0, 0.4),
                P3.P3Shape(; logλ = Float64(logλx), μ = 1.0),
            )
            Irefx, _ = QGK.quadgk(D -> D^3 * nx(D), Float64(D₀), 1.0)
            L_refx = Float64(stx.F_rim) * Float64(stx.F_liq) * (π * Float64(p.ρ_l) / 6) * Irefx
            @test L_refx > 0
            @test Float64(sx.L_shd) ≈ L_refx rtol = (FT === Float32 ? rtol32 : 1e-8)
        end

        # deep tail: small ice sheds ~0; the log-space moment underflows rather
        # than producing the eps-floored overestimate.
        st_sm = P3.state_from_prognostic(p, FT(5e-4), FT(1e6), FT(4e-4), FT(1e-6), _ρq_liq_from_F(FT(5e-4), FT(0.4)))
        sh_sm = P3.get_distribution_shape(st_sm)
        s_sm = P3.ice_shed(st_sm, sh_sm)
        n_sm = DT.size_distribution(st_sm, sh_sm)
        Iref_sm, _ = QGK.quadgk(D -> D^3 * n_sm(D), D₀, FT(1))
        L_ref_sm = st_sm.F_rim * st_sm.F_liq * (π * p.ρ_l / 6) * Iref_sm
        @test s_sm.L_shd ≥ 0
        @test s_sm.L_shd ≤ ρq_ice          # never exceeds the ice mass
        @test isapprox(s_sm.L_shd, L_ref_sm; atol = FT(1e-30), rtol = 1e-1)

        # BigFloat reference for the log-space upper incomplete gamma, including
        # the λ D_shd ~ 90 range (D10)
        for (z, x) in (
            (FT(5), FT(1)), (FT(8), FT(20)), (FT(11), FT(60)),
            (FT(5), FT(85)), (FT(8), FT(90)), (FT(11), FT(95)),
        )
            ref = Float64(log(SF.gamma(big(z)) * SF.gamma_inc(big(z), big(x))[2]))
            @test Float64(P3.log_upper_incomplete_gamma(z, x)) ≈ ref rtol = 1e-4
        end

        # NoLiquidFraction returns zero
        st_no = P3.state_from_prognostic(pn, ρq_ice, ρn_ice, ρq_rim, ρb_rim)
        @test P3.ice_shed(st_no, P3.get_distribution_shape(st_no)) == (; L_shd = zero(FT), N_shd = zero(FT))
    end
end

function test_vapor_ramp(FT)
    @testset "vapor-path ramp" begin
        p = CMP.ParametersP3(FT; liquid = :predicted)
        liq = p.liquid
        (; F_dry, ΔF_switch) = liq
        @test F_dry == FT(0.01)
        @test ΔF_switch == FT(0.02)
        band_hi = F_dry + ΔF_switch
        mid = F_dry + ΔF_switch / 2
        w(f) = P3.vapor_path_weight(liq, f)

        # w(0) = 0 and w'(0) = 0 (dry-ice baseline untouched)
        @test w(zero(FT)) == 0
        @test FD.derivative(w, zero(FT)) == 0
        # zero on the dry side up to the band start, one above the band
        @test w(F_dry) == 0
        @test w(band_hi) == 1
        @test w(FT(0.5)) == 1
        # monotone increasing across the band, midpoint = 0.5
        @test w(mid) ≈ FT(0.5)
        @test w(F_dry + ΔF_switch / 4) < w(F_dry + 3 * ΔF_switch / 4)
        # C1: zero slope at both band edges, positive slope inside
        @test FD.derivative(w, F_dry) ≈ 0 atol = 1e-6
        @test FD.derivative(w, band_hi) ≈ 0 atol = 1e-6
        @test FD.derivative(w, mid) > 0

        # band asserted strictly inside (0, F_melt) at construction
        # (args: F_dry, ΔF_switch, F_melt, q_liq_present, D_shd_onset, D_shd_drop, τ_shd)
        good = (FT(0.01), FT(0.02), FT(0.99), FT(1e-10), FT(9e-3), FT(1e-3), FT(1))
        bad(i, v) = ntuple(j -> j == i ? v : good[j], length(good))
        @test CMP.PredictedLiquidFraction{FT}(good...) isa CMP.PredictedLiquidFraction{FT}
        @test_throws AssertionError CMP.PredictedLiquidFraction{FT}(bad(2, FT(0.99))...)  # band past F_melt
        @test_throws AssertionError CMP.PredictedLiquidFraction{FT}(bad(1, FT(0))...)     # F_dry = 0
        @test_throws AssertionError CMP.PredictedLiquidFraction{FT}(bad(2, FT(0))...)     # zero width
        @test_throws AssertionError CMP.PredictedLiquidFraction{FT}(bad(7, FT(0))...)     # zero timescale
    end
end

function test_no_nan_corners(FT)
    @testset "no NaN/Inf in degenerate corners" begin
        p = CMP.ParametersP3(FT; liquid = :predicted)
        vel = CMP.Chen2022VelType(FT)
        aps = CMP.AirProperties(FT)
        tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
        quad = P3.GaussLegendre(FT, 12)
        ρₐ = FT(1)
        T_frz = p.T_freeze
        # (ρq_ice, ρn_ice, ρq_rim, ρb_rim, ρq_liq)
        corners = (
            (FT(0), FT(0), FT(0), FT(0), FT(0)),           # empty
            (FT(0), FT(1e3), FT(0), FT(0), FT(0)),         # empty core, positive number
            (FT(1e-9), FT(1e3), FT(0), FT(0), FT(1e-9)),   # thin core with liquid
            (FT(1e-3), FT(1e5), FT(9e-4), FT(1e-6), FT(1e-12)),  # near-dry
            (FT(1e-3), FT(1e5), FT(9e-4), FT(1e-6), FT(2e-1)),   # liquid clamped to F_melt
        )
        for c in corners
            st = P3.state_from_prognostic(p, c...)
            sh = P3.get_distribution_shape(st)
            @test isfinite(sh.logλ) && isfinite(sh.logλ_core) && isfinite(sh.μ)
            @test isfinite(P3.total_mass_concentration(st))
            for D in FT.((0, 1e-5, 1e-3, 1e-2))
                @test isfinite(P3.mixed_mass(st, D)) && isfinite(P3.mixed_area(st, D))
            end
            r = P3.ice_refreeze(vel, aps, tps, T_frz - FT(5), ρₐ, st, sh; quad)
            @test all(isfinite, values(r))
            s = P3.ice_shed(st, sh)
            @test all(isfinite, values(s))
            m = P3.ice_melt(vel, aps, tps, T_frz + FT(2), ρₐ, st, sh; quad)
            @test all(isfinite, values(m))
        end

        # melt number rate through the mean-mass band: bounded at an empty core
        # with positive number, and vanishing continuously with the mass rate on
        # the near-empty approach (value and ForwardDiff derivative w.r.t. ρq_ice
        # at a frozen shape); see the liqfrac-core review
        sh0 = P3.get_distribution_shape(P3.state_from_prognostic(p, FT(1e-13), FT(1e3), FT(0), FT(0), FT(0)))
        melt_dNdt(x) = P3.ice_melt(
            vel, aps, tps, T_frz + FT(2), ρₐ,
            P3.state_from_prognostic(p, x, FT(1e3), FT(0), FT(0), FT(0)), sh0; quad,
        ).dNdt_rain
        m0 = P3.ice_melt(
            vel, aps, tps, T_frz + FT(2), ρₐ,
            P3.state_from_prognostic(p, FT(0), FT(1e3), FT(0), FT(0), FT(0)), sh0; quad,
        )
        @test isfinite(m0.dNdt_rain) && m0.dNdt_rain ≥ 0
        @test m0.dNdt_rain == m0.dLdt_rain / p.mean_mass_min
        for x in FT.((0, 1e-15, 1e-14, 1e-13, 1e-12))
            v = melt_dNdt(x)
            @test isfinite(v) && v ≥ 0
            @test isfinite(FD.derivative(melt_dNdt, x))
        end
    end
end

function _liq_entry_inputs(FT; F_liq = FT(0.3), q_ice = FT(8e-4))
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true, liquid = :predicted)
    ρ = FT(1)
    warm = (; q_tot = FT(8e-3), q_lcl = FT(1e-3), n_lcl = FT(1e8), q_rai = FT(5e-4), n_rai = FT(1e4))
    q_liq = _ρq_liq_from_F(q_ice, F_liq)
    cat = (;
        q_ice,
        n_ice = FT(2e5),
        q_rim = FT(3e-4) * q_ice / FT(8e-4),
        b_rim = FT(1e-6) * q_ice / FT(8e-4),
        q_liq_on_ice = q_liq,
    )
    st = P3.state_from_prognostic(
        mp.ice.scheme, cat.q_ice * ρ, cat.n_ice * ρ, cat.q_rim * ρ, cat.b_rim * ρ, q_liq * ρ,
    )
    shape = P3.get_distribution_shape(st)
    return (; tps, mp, ρ, warm, cat, shape)
end

function _liq_entry(FT, T, inp; mode = ())
    (; tps, mp, ρ, warm, cat, shape) = inp
    tail = isempty(mode) ? () : (FT(60), 4)
    return BMT.bulk_microphysics_tendencies(
        mode..., BMT.Microphysics2Moment(), mp, tps, ρ, T, warm.q_tot,
        warm.q_lcl, warm.n_lcl, warm.q_rai, warm.n_rai, (cat,), (shape,), tail...,
    )
end

function test_liquid_entry_wiring(FT)
    @testset "tendency entry wiring" begin
        inp = _liq_entry_inputs(FT)
        (; tps, mp, ρ, warm, cat, shape) = inp
        T_frz = mp.ice.scheme.T_freeze
        names10 = (
            :dq_lcl_dt, :dn_lcl_dt, :dq_rai_dt, :dn_rai_dt,
            :dq_ice_dt, :dn_ice_dt, :dq_rim_dt, :db_rim_dt,
            :dq_liq_on_ice_dt, :dn_lcl_activation_dt,
        )
        for T in (T_frz - FT(10), T_frz + FT(2))
            inst = _liq_entry(FT, T, inp)
            @test keys(inst) == names10
            @test all(isfinite, values(inst))
            ros = _liq_entry(FT, T, inp; mode = (BMT.rosenbrock_exact(),))
            @test keys(ros) == names10
            @test all(isfinite, values(ros))
        end

        # warm melting and collection fill the liquid; cold refreezing drains it
        # into rime
        inst_warm = _liq_entry(FT, T_frz + FT(2), inp)
        @test inst_warm.dq_liq_on_ice_dt > 0
        inst_cold = _liq_entry(FT, T_frz - FT(10), inp)
        @test inst_cold.dq_liq_on_ice_dt < 0
        @test inst_cold.dq_rim_dt > 0

        # unsupported paths throw
        @test_throws ArgumentError _liq_entry(FT, T_frz + FT(2), inp; mode = (BMT.rosenbrock_manual(),))
        @test_throws ArgumentError _liq_entry(FT, T_frz + FT(2), inp; mode = (BMT.Verbose(BMT.rosenbrock_exact()),))
        @test_throws ArgumentError BMT.bulk_microphysics_tendencies(
            BMT.Microphysics2Moment(), mp, tps, ρ, T_frz + FT(2), warm.q_tot,
            warm.q_lcl, warm.n_lcl, warm.q_rai, warm.n_rai,
            cat.q_ice, cat.n_ice, cat.q_rim, cat.b_rim, shape.logλ,
        )
    end
end

function test_liquid_collision_routing(FT)
    @testset "collection routing" begin
        p = CMP.ParametersP3(FT; liquid = :predicted)
        pn = CMP.ParametersP3(FT)
        vel = CMP.Chen2022VelType(FT)
        aps = CMP.AirProperties(FT)
        tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
        mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true, liquid = :predicted)
        pdf_c = mp.ice.cloud_pdf
        pdf_r = mp.ice.rain_pdf
        quad = P3.GaussLegendre(FT, 12)
        ρₐ = FT(1)
        T_frz = p.T_freeze
        L_c, N_c, L_r, N_r = FT(1e-3), FT(1e8), FT(5e-4), FT(1e4)
        ρq_ice, ρn_ice, ρq_rim, ρb_rim = FT(8e-4), FT(2e5), FT(3e-4), FT(1e-6)
        ρq_liq = _ρq_liq_from_F(ρq_ice, FT(0.3))
        st = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, ρq_liq)
        sh = P3.get_distribution_shape(st)

        # above freezing: no freezing, all collected liquid retained on the ice
        cw = P3.bulk_liquid_ice_collision_sources(
            st, sh, pdf_c, pdf_r, L_c, N_c, L_r, N_r, aps, tps, vel, ρₐ, T_frz + FT(2); quad,
        )
        @test cw.∂ₜL_liq > 0
        @test cw.∂ₜL_rim == 0 && cw.∂ₜL_ice == 0 && cw.∂ₜB_rim == 0
        @test cw.∂ₜq_c ≤ 0 && cw.∂ₜq_r ≤ 0 && cw.∂ₜN_c ≤ 0 && cw.∂ₜN_r ≤ 0
        # mass closure of the routing: everything removed from cloud and rain
        # arrives in the retained liquid
        @test cw.∂ₜL_liq ≈ -(cw.∂ₜq_c + cw.∂ₜq_r) * ρₐ rtol = 1e-6

        # far below freezing (dry growth): collected liquid freezes to rime
        # exactly as under NoLiquidFraction, and nothing is retained
        cc = P3.bulk_liquid_ice_collision_sources(
            st, sh, pdf_c, pdf_r, L_c, N_c, L_r, N_r, aps, tps, vel, ρₐ, T_frz - FT(30); quad,
        )
        @test cc.∂ₜL_liq ≥ 0
        @test cc.∂ₜL_rim > 0
        @test cc.∂ₜL_liq < cc.∂ₜL_rim  # deep cold: freezing dominates retention
        # mass closure below freezing: collected mass splits between rime and
        # retained liquid
        @test cc.∂ₜL_rim + cc.∂ₜL_liq ≈ -(cc.∂ₜq_c + cc.∂ₜq_r) * ρₐ rtol = 1e-6
        stn = P3.P3State(pn, ρq_ice, ρn_ice, st.F_rim, st.ρ_rim)
        shn = P3.P3Shape(; logλ = sh.logλ, μ = sh.μ)
        cn = P3.bulk_liquid_ice_collision_sources(
            stn, shn, pdf_c, pdf_r, L_c, N_c, L_r, N_r, aps, tps, vel, ρₐ, T_frz - FT(30); quad,
        )
        @test cc.∂ₜL_ice ≈ cn.∂ₜL_ice rtol = 1e-4
    end
end

function test_liquid_entry_conservation(FT)
    @testset "entry water conservation" begin
        tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
        mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true, liquid = :predicted)
        p3 = mp.ice.scheme
        liq = p3.liquid
        sb = mp.warm_rain.seifert_beheng
        aps = mp.warm_rain.air_properties
        condevap = mp.warm_rain.condevap
        subdep = mp.warm_rain.subdep
        T_frz = p3.T_freeze
        ρ = FT(1)
        rtol = FT === Float32 ? 5e-4 : 1e-9

        # Closure of the transfers: the vapor-exchange terms are reconstructed
        # from the same public primitives (white-box), so this testset validates
        # that every non-vapor process conserves the tracked water sum, not the
        # vapor-path physics itself (see the vapor anchor testset).
        function vapor_exchange(T, warm, cat, state)
            (; q_tot, q_lcl, n_lcl, q_rai, n_rai) = warm
            q_liq = cat.q_liq_on_ice
            q_ice = cat.q_ice
            q_icl_tot = q_ice + q_liq
            thermo = (; ρ, T)
            # cloud condensation/evaporation and rain evaporation (warm block)
            cond = CMNonEq.conv_q_vap_to_q_lcl(
                CMP.CloudLiquidFormation(condevap.τ_relax), nothing, tps,
                (; q_tot, q_lcl, q_icl = q_icl_tot, q_rai, q_sno = zero(q_ice)), thermo,
            )
            evap = CM2.rain_evaporation(
                sb, aps, tps, q_tot, q_lcl, q_icl_tot, q_rai, zero(q_ice), ρ, n_rai * ρ, T,
            ).∂ₜq_rai
            # deposition nucleation
            τ_act = mp.ice.inp_depletion_model.τ_act
            D_nuc = FT(10e-6)
            m_nuc = p3.ρ_i * CO.volume_sphere_D(D_nuc)
            n_active = CM_HetIce.n_active(mp.ice.inp_depletion_model, cat.n_ice)
            dep_nuc = CM_HetIce.deposition_rate(
                mp.ice.ice_nucleation, tps, T, ρ, q_tot, q_lcl + q_rai, q_icl_tot, n_active;
                m_nuc, τ_act, inpc_log_shift = zero(ρ),
            ).∂ₜq_frz
            # ramped core deposition/sublimation and shell condensation/evaporation
            w = P3.vapor_path_weight(liq, state.F_liq)
            depsub = CMNonEq.conv_q_vap_to_q_icl(
                CMP.ConstantTimescale(subdep.τ_relax), nothing, tps,
                (; q_tot, q_lcl, q_icl = q_ice, q_rai, q_sno = q_liq), thermo,
            )
            depsub = ifelse(T > tps.T_freeze, min(depsub, zero(T)), depsub)
            shell = CMNonEq.conv_q_vap_to_q_lcl(
                CMP.CloudLiquidFormation(subdep.τ_relax), nothing, tps,
                (; q_tot, q_lcl = q_liq, q_icl = q_ice, q_rai = q_lcl + q_rai, q_sno = zero(q_ice)), thermo,
            )
            return cond + evap + dep_nuc + (1 - w) * depsub + w * shell
        end

        for T in (T_frz - FT(20), T_frz - FT(5), T_frz + FT(2)),
            F_liq in (FT(0), FT(0.005), FT(0.3), FT(0.7)),
            q_ice in (FT(8e-4), FT(5e-5))

            inp = _liq_entry_inputs(FT; F_liq, q_ice)
            tend = _liq_entry(FT, T, inp)
            S = tend.dq_lcl_dt + tend.dq_rai_dt + tend.dq_ice_dt + tend.dq_liq_on_ice_dt
            st = P3.state_from_prognostic(
                p3, inp.cat.q_ice * ρ, inp.cat.n_ice * ρ, inp.cat.q_rim * ρ, inp.cat.b_rim * ρ,
                inp.cat.q_liq_on_ice * ρ,
            )
            V = vapor_exchange(T, inp.warm, inp.cat, st)
            scale = max(abs(S), abs(V), FT(1e-8))
            @test abs(S - V) / scale < rtol
        end

        # sub-threshold core with residual liquid: the drain to rain is a
        # transfer and the sum still closes on the vapor terms
        warm0 = (; q_tot = FT(1e-3), q_lcl = FT(0), n_lcl = FT(0), q_rai = FT(1e-5), n_rai = FT(1e3))
        cat0 = (; q_ice = FT(0), n_ice = FT(0), q_rim = FT(0), b_rim = FT(0), q_liq_on_ice = FT(1e-4))
        st0 = P3.state_from_prognostic(p3, FT(0), FT(0), FT(0), FT(0), cat0.q_liq_on_ice * ρ)
        sh0 = P3.get_distribution_shape(st0)
        for T in (T_frz - FT(10), T_frz + FT(2))
            tend = _liq_entry(FT, T, (; tps, mp, ρ, warm = warm0, cat = cat0, shape = sh0))
            S = tend.dq_lcl_dt + tend.dq_rai_dt + tend.dq_ice_dt + tend.dq_liq_on_ice_dt
            V = vapor_exchange(T, warm0, cat0, st0)
            scale = max(abs(S), abs(V), FT(1e-8))
            @test abs(S - V) / scale < rtol
        end
    end
end

function test_residual_liquid_drain(FT)
    @testset "residual liquid drain on an emptied core" begin
        tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
        mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true, liquid = :predicted)
        p3 = mp.ice.scheme
        liq = p3.liquid
        ρ = FT(1)
        ϵₘ = UT.ϵ_numerics_2M_M(FT)
        m_drop = p3.ρ_l * CO.volume_sphere_D(liq.D_shd_drop)
        q_liq = FT(1e-4)
        z = zero(FT)
        cat(q_ice) = (; q_ice, n_ice = FT(0), q_rim = FT(0), b_rim = FT(0), q_liq_on_ice = q_liq)

        # the ramp: full drain at an empty core, zero at the presence threshold,
        # continuous and conserving in between
        drain_of(q_ice) = begin
            (dq_rai, dn_rai, dq_liq) =
                BMT._residual_liquid_to_rain(liq, p3, ρ, q_ice, cat(q_ice), z, z, z)
            @test dq_rai + dq_liq == 0             # conserving transfer
            @test dn_rai ≈ dq_rai / m_drop rtol = 1e-6
            dq_rai
        end
        @test drain_of(FT(0)) ≈ q_liq / liq.τ_shd rtol = 1e-6
        @test drain_of(ϵₘ / 2) ≈ q_liq / liq.τ_shd / 2 rtol = 1e-6
        @test drain_of(ϵₘ) == 0
        @test drain_of(FT(1e-4)) == 0              # active-core region: no-op
        # continuity at the threshold
        @test drain_of(ϵₘ * (1 - FT(1e-3))) < q_liq / liq.τ_shd * FT(2e-3)

        # accumulation scenario through the entry: sub-threshold core, liquid
        # above; dry air so the shell term reinforces the drain
        warm0 = (; q_tot = FT(1e-3), q_lcl = FT(0), n_lcl = FT(0), q_rai = FT(1e-5), n_rai = FT(1e3))
        st0 = P3.state_from_prognostic(p3, FT(0), FT(0), FT(0), FT(0), q_liq * ρ)
        sh0 = P3.get_distribution_shape(st0)
        T = p3.T_freeze - FT(10)
        tend = _liq_entry(FT, T, (; tps, mp, ρ, warm = warm0, cat = cat(FT(0)), shape = sh0))
        @test all(isfinite, values(tend))
        @test tend.dq_liq_on_ice_dt ≤ -q_liq / liq.τ_shd * (1 - FT(1e-4))
    end
end

function test_liquid_jacobian_sweep(FT)
    @testset "ExactJacobian finiteness sweep" begin
        tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
        mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true, liquid = :predicted)
        p3 = mp.ice.scheme
        T_frz = p3.T_freeze
        ρ = FT(1)
        (q_lcl, n_lcl, q_rai, n_rai) = (FT(1e-3), FT(1e8), FT(5e-4), FT(1e4))
        n_states = 0
        n_bad_J = 0
        for T in (T_frz - FT(20), T_frz - FT(2), T_frz + FT(2)),
            F_liq in (FT(0), FT(0.05), FT(0.3), FT(0.7), FT(0.95)),
            q_ice in (FT(1e-6), FT(1e-4), FT(2e-3)),
            F_rim in (FT(0), FT(0.5), FT(0.9))

            q_rim = F_rim * q_ice
            b_rim = q_rim / FT(500)
            # liquid capped at a physical load (the uncapped F_liq = 0.95 point
            # at the largest ice content implies ~40 g/kg of liquid, whose
            # latent release over a substep leaves the thermodynamic domain)
            q_liq = min(_ρq_liq_from_F(q_ice, F_liq), FT(4e-3))
            # total water consistent with the condensates
            q_tot = max(FT(8e-3), 2 * (q_lcl + q_rai + q_ice + q_liq))
            st = P3.state_from_prognostic(p3, q_ice * ρ, FT(2e5) * ρ, q_rim * ρ, b_rim * ρ, q_liq * ρ)
            shape = P3.get_distribution_shape(st)
            g = BMT.Instantaneous2MP3Tendency(mp, tps, ρ, T, q_tot, (shape,))
            x = BMT.MicroState{FT, 1, true, false}((
                q_lcl, n_lcl, q_rai, n_rai, q_ice, FT(2e5), q_rim, b_rim, q_liq,
            ))
            f, J = BMT._tendency_and_jacobian(BMT.ExactJacobian(), g, x)
            n_states += 1
            all(isfinite, J) || (n_bad_J += 1)
            @test all(isfinite, f)
            # the substep driver stays finite either way (Euler fallback)
            ros = BMT.bulk_microphysics_tendencies(
                BMT.rosenbrock_exact(), BMT.Microphysics2Moment(), mp, tps, ρ, T, q_tot,
                q_lcl, n_lcl, q_rai, n_rai,
                ((; q_ice, n_ice = FT(2e5), q_rim, b_rim, q_liq_on_ice = q_liq),), (shape,),
                FT(60), 4,
            )
            @test all(isfinite, values(ros))
        end
        @info "liquid ExactJacobian sweep ($FT): non-finite J at $n_bad_J of $n_states states"
        @test n_bad_J / n_states < 0.25
    end
end

function test_liquid_jacobian_fd(FT)
    @testset "ExactJacobian finite-difference cross-check" begin
        tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
        mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true, liquid = :predicted)
        p3 = mp.ice.scheme
        ρ = FT(1)
        T = p3.T_freeze + FT(2)
        q_tot = FT(8e-3)
        (q_ice, n_ice, q_rim, b_rim) = (FT(8e-4), FT(2e5), FT(3e-4), FT(1e-6))
        q_liq = _ρq_liq_from_F(q_ice, FT(0.3))
        st = P3.state_from_prognostic(p3, q_ice * ρ, n_ice * ρ, q_rim * ρ, b_rim * ρ, q_liq * ρ)
        shape = P3.get_distribution_shape(st)
        g = BMT.Instantaneous2MP3Tendency(mp, tps, ρ, T, q_tot, (shape,))
        x = BMT.MicroState{FT, 1, true, false}((
            FT(1e-3), FT(1e8), FT(5e-4), FT(1e4), q_ice, n_ice, q_rim, b_rim, q_liq,
        ))
        _, J = BMT._tendency_and_jacobian(BMT.ExactJacobian(), g, x)
        # Float32 central differences through the quadrature carry ~5e-3
        # relative steps against curvature; Float64 validates the derivative
        # tightly and Float32 coarsely
        rtol = FT === Float32 ? 1e-1 : 1e-5
        for j in 1:9
            h = cbrt(eps(FT)) * abs(x[j])
            xp = Base.setindex(x, x[j] + h, j)
            xm = Base.setindex(x, x[j] - h, j)
            col_fd = (g(xp) - g(xm)) / (xp[j] - xm[j])
            atol = rtol * maximum(abs, col_fd)
            for i in 1:9
                @test isapprox(J[i, j], col_fd[i]; rtol, atol)
            end
        end
    end
end

function test_vapor_anchor(FT)
    @testset "vapor-path anchor" begin
        # Independent anchor for the ramped vapor exchange: the state sits in the
        # mass-limited sublimation/evaporation branches, where the relaxations
        # reduce to -q/(τ Γ), with w, F_liq, and Γ recomputed from first
        # principles rather than the entry helpers.
        tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
        mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true, liquid = :predicted)
        p3 = mp.ice.scheme
        liq = p3.liquid
        subdep = mp.warm_rain.subdep
        ρ = FT(1)
        T = p3.T_freeze - FT(10)
        (q_tot, q_lcl, q_rai) = (FT(5e-4), FT(0), FT(0))
        (q_ice, n_ice, q_liq) = (FT(1e-4), FT(2e5), FT(5e-5))
        cat = (; q_ice, n_ice, q_rim = FT(0), b_rim = FT(0), q_liq_on_ice = q_liq)
        st = P3.state_from_prognostic(p3, q_ice * ρ, n_ice * ρ, FT(0), FT(0), q_liq * ρ)

        # the deficits exceed the condensate masses, so both relaxations sit in
        # the mass-limited branch
        qs_ice = TDI.saturation_vapor_specific_content_over_ice(tps, T, ρ)
        qs_liq = TDI.saturation_vapor_specific_content_over_liquid(tps, T, ρ)
        q_vap = q_tot - q_lcl - q_rai - q_ice - q_liq
        @test qs_ice - q_vap > q_ice
        @test qs_liq - q_vap > q_liq

        # hand-computed ramp weight and relaxation factors
        F_liq_hand = (q_liq * ρ) / ((q_ice + q_liq) * ρ + liq.q_liq_present)
        t = clamp((F_liq_hand - liq.F_dry) / liq.ΔF_switch, FT(0), FT(1))
        w = t^2 * (3 - 2t)
        Rᵥ = TDI.Rᵥ(tps)
        Lₛ = TDI.Lₛ(tps, T)
        Lᵥ = TDI.Lᵥ(tps, T)
        τ = subdep.τ_relax
        cp_core = TDI.cpₘ(tps, q_tot, q_lcl + q_rai, q_ice + q_liq)
        cp_shell = TDI.cpₘ(tps, q_tot, q_liq, q_ice)
        Γᵢ = 1 + (Lₛ / cp_core) * qs_ice * (Lₛ / (Rᵥ * T^2) - 1 / T)
        Γₗ = 1 + (Lᵥ / cp_shell) * qs_liq * (Lᵥ / (Rᵥ * T^2) - 1 / T)
        expected_core = -(1 - w) * q_ice / (τ * Γᵢ)
        expected_shell = -w * q_liq / (τ * Γₗ)
        expected_n =
            (1 - w) * (n_ice / q_ice) * (-q_ice / (τ * Γᵢ)) +
            w * (n_ice / (q_ice + q_liq)) * (-q_liq / (τ * Γₗ))

        z = zero(FT)
        (dq_ice, dn_ice, dq_rim, db_rim, dq_liq) = BMT._vapor_exchange_accumulate(
            liq, subdep, tps, ρ, T, q_tot, q_lcl, q_rai, q_ice, n_ice, cat, st,
            zero(FT), one(FT), one(FT),  # single category: no other condensate, unit shares
            z, z, z, z, z,
        )
        rtol = FT === Float32 ? 1e-3 : 1e-6
        @test dq_ice ≈ expected_core rtol = rtol
        @test dq_liq ≈ expected_shell rtol = rtol
        @test dn_ice ≈ expected_n rtol = rtol
        @test dq_rim == 0 && db_rim == 0  # unrimed state
    end
end


function test_liquid_entry_jacobian(FT)
    @testset "entry ExactJacobian (liquid on)" begin
        tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
        mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true, liquid = :predicted)
        T_frz = mp.ice.scheme.T_freeze
        ρ = FT(1)
        (q_tot, q_lcl, n_lcl, q_rai, n_rai) = (FT(8e-3), FT(1e-3), FT(1e8), FT(5e-4), FT(1e4))
        (q_ice, n_ice, q_rim, b_rim) = (FT(8e-4), FT(2e5), FT(3e-4), FT(1e-6))

        # primal/Jacobian lockstep at the healthy state, at F_liq → 0, and near
        # F_melt, above and below freezing; the 3-slot shape stays frozen on the
        # substep context
        for T in (T_frz - FT(10), T_frz + FT(2)), F_liq in (FT(0), FT(0.3), FT(0.95))
            q_liq = _ρq_liq_from_F(q_ice, F_liq)
            st = P3.state_from_prognostic(
                mp.ice.scheme, q_ice * ρ, n_ice * ρ, q_rim * ρ, b_rim * ρ, q_liq * ρ,
            )
            shape = P3.get_distribution_shape(st)
            g = BMT.Instantaneous2MP3Tendency(mp, tps, ρ, T, q_tot, (shape,))
            x = BMT.MicroState{FT, 1, true, false}((
                q_lcl, n_lcl, q_rai, n_rai, q_ice, n_ice, q_rim, b_rim, q_liq,
            ))
            f, J = BMT._tendency_and_jacobian(BMT.ExactJacobian(), g, x)
            @test all(isfinite, f)
            @test all(isfinite, J)
            @test f == g(x)  # primal recovered from the dual pass
        end
    end
end

function test_liquid_sedimentation(FT)
    @testset "mixed-particle sedimentation velocities" begin
        p = CMP.ParametersP3(FT; liquid = :predicted)
        pn = CMP.ParametersP3(FT)
        vel = CMP.Chen2022VelType(FT)
        quad = P3.GaussLegendre(FT, 12)
        ρₐ = FT(1)
        ρq_ice, ρn_ice, ρq_rim, ρb_rim = FT(8e-4), FT(2e5), FT(3e-4), FT(1e-6)
        ρq_liq = _ρq_liq_from_F(ρq_ice, FT(0.5))
        st = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, ρq_liq)
        sh = P3.get_distribution_shape(st)

        vn = P3.ice_terminal_velocity_number_weighted(vel, ρₐ, st, sh; quad)
        vm = P3.ice_terminal_velocity_mass_weighted(vel, ρₐ, st, sh; quad)
        @test isfinite(vn) && vn > 0
        @test isfinite(vm) && vm > 0
        @test vn < vm  # mass weighting emphasizes larger, faster particles

        # prognostic wrappers with the trailing liquid argument
        vn2 = P3.ice_terminal_velocity_number_weighted_from_prognostic(
            vel, ρₐ, p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, ρq_liq, sh.logλ; quad,
        )
        vm2 = P3.ice_terminal_velocity_mass_weighted_from_prognostic(
            vel, ρₐ, p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, ρq_liq, sh.logλ; quad,
        )
        @test vn2 ≈ vn rtol = 1e-6
        @test vm2 ≈ vm rtol = 1e-6

        # F_liq = 0 under the predicted treatment equals the dry velocities
        st0 = P3.state_from_prognostic(p, ρq_ice, ρn_ice, ρq_rim, ρb_rim, FT(0))
        sh0 = P3.get_distribution_shape(st0)
        stn = P3.state_from_prognostic(pn, ρq_ice, ρn_ice, ρq_rim, ρb_rim)
        shn = P3.P3Shape(; logλ = sh0.logλ, μ = sh0.μ)
        @test P3.ice_terminal_velocity_number_weighted(vel, ρₐ, st0, sh0; quad) ==
              P3.ice_terminal_velocity_number_weighted(vel, ρₐ, stn, shn; quad)
        @test P3.ice_terminal_velocity_mass_weighted(vel, ρₐ, st0, sh0; quad) ==
              P3.ice_terminal_velocity_mass_weighted(vel, ρₐ, stn, shn; quad)
    end
end

@testset "P3 liquid-fraction tests ($FT)" for FT in (Float64, Float32)
    test_liquid_fraction_state(FT)
    test_liquid_mass_fraction_onset(FT)
    test_particle_property_blends(FT)
    test_whole_mass_moment(FT)
    test_two_slope_shape(FT)
    test_shape_iteration_budget(FT)
    test_ice_melt_liquid(FT)
    test_ice_refreeze(FT)
    test_ice_shed(FT)
    test_vapor_ramp(FT)
    test_no_nan_corners(FT)

    # tendency-entry wiring
    test_liquid_entry_wiring(FT)
    test_liquid_collision_routing(FT)
    test_liquid_entry_conservation(FT)
    test_residual_liquid_drain(FT)
    test_liquid_entry_jacobian(FT)
    test_liquid_jacobian_sweep(FT)
    test_liquid_jacobian_fd(FT)
    test_vapor_anchor(FT)
    test_liquid_sedimentation(FT)
end
