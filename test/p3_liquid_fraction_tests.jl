using Test: @testset, @test, @test_throws, @inferred
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.Common as CO
import CloudMicrophysics.DistributionTools as DT
import CloudMicrophysics.Utilities as UT
import CloudMicrophysics.ThermodynamicsInterface as TDI
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
end
