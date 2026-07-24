using Test

import JET
import BenchmarkTools as BT

import ClimaParams as CP
import CloudMicrophysics as CM
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.ThermodynamicsInterface as TDI
import CloudMicrophysics.MicrophysicsNonEq as CMNonEq
import CloudMicrophysics.Microphysics2M as CM2
import CloudMicrophysics.HetIceNucleation as CM_HetIce
import CloudMicrophysics.Common as CO
import CloudMicrophysics.Utilities as UT
import ForwardDiff as FD
import StaticArrays: SVector
import StaticArrays as SA

# The unified `RosenbrockAverage{Jacobian, GrowthTreatment, TendencyLimiter}`
# framework: presets (`rosenbrock_donor`, `rosenbrock_coupled`,
# `rosenbrock_exact`, `rosenbrock_manual`), the keyword constructor, the
# `LinearizedAverage` ≡ donor equivalence, the `Verbose` wrapper, and the 2M+P3
# `ExactJacobian`/`ManualJacobian` contract.

net_vec_1m(t) = SVector(t.dq_lcl_dt, t.dq_icl_dt, t.dq_rai_dt, t.dq_sno_dt)

function test_framework_1m(FT)
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics1MParams(FT)
    T_frz = TDI.T_freeze(tps)

    # x = [q_lcl, q_icl, q_rai, q_sno]
    regimes = (
        (; ρ = FT(1.0), T = T_frz + FT(17), q_tot = FT(0.018), x = FT[2e-3, 0, 5e-4, 0]),  # warm rain
        (; ρ = FT(1.2), T = T_frz + FT(5), q_tot = FT(0.012), x = FT[5e-4, 2e-4, 3e-4, 3e-4]),  # mixed warm
        (; ρ = FT(1.2), T = T_frz - FT(10), q_tot = FT(0.012), x = FT[3e-4, 5e-4, 2e-4, 4e-4]),  # mixed cold
    )

    @testset "rosenbrock_donor() ≡ LinearizedAverage() ($FT)" begin
        for r in regimes, nsub in (1, 4, 16)
            Δt = FT(20)
            donor = BMT.bulk_microphysics_tendencies(
                BMT.rosenbrock_donor(), BMT.Microphysics1Moment(), mp, tps,
                r.ρ, r.T, r.q_tot, r.x..., Δt, nsub,
            )
            lin = BMT.bulk_microphysics_tendencies(
                BMT.LinearizedAverage(), BMT.Microphysics1Moment(), mp, tps,
                r.ρ, r.T, r.q_tot, r.x..., Δt, nsub,
            )
            @test net_vec_1m(donor) == net_vec_1m(lin)
        end
    end

    @testset "keyword constructor matches rosenbrock_donor() ($FT)" begin
        kw = BMT.RosenbrockAverage(
            jacobian = BMT.DonorJacobian(),
            growth = BMT.ImplicitGrowth(),
            limiter = BMT.NoLimiter(),
        )
        @test kw == BMT.rosenbrock_donor()
        for r in regimes
            Δt = FT(20)
            t_kw = BMT.bulk_microphysics_tendencies(
                kw, BMT.Microphysics1Moment(), mp, tps,
                r.ρ, r.T, r.q_tot, r.x..., Δt, 4,
            )
            t_preset = BMT.bulk_microphysics_tendencies(
                BMT.rosenbrock_donor(), BMT.Microphysics1Moment(), mp, tps,
                r.ρ, r.T, r.q_tot, r.x..., Δt, 4,
            )
            @test net_vec_1m(t_kw) == net_vec_1m(t_preset)
        end
    end

    @testset "1M presets give finite tendencies ($FT)" begin
        presets = (BMT.rosenbrock_donor(), BMT.rosenbrock_coupled(), BMT.rosenbrock_exact())
        for mode in presets, r in regimes, nsub in (1, 2, 8), Δt in (FT(20), FT(120))
            t = BMT.bulk_microphysics_tendencies(
                mode, BMT.Microphysics1Moment(), mp, tps,
                r.ρ, r.T, r.q_tot, r.x..., Δt, nsub,
            )
            @test all(isfinite, net_vec_1m(t))
        end
    end

    @testset "EndStateSaturationAdjustment limits the more-supersaturated phase ($FT)" begin
        Lv = TDI.TD.Parameters.LH_v0(tps) / TDI.TD.Parameters.cp_d(tps)
        Ls = TDI.TD.Parameters.LH_s0(tps) / TDI.TD.Parameters.cp_d(tps)
        S_liq(x, T, ρ, qt) = TDI.supersaturation_over_liquid(tps, qt, x[1] + x[3], x[2] + x[4], ρ, T)
        S_ice(x, T, ρ, qt) = TDI.supersaturation_over_ice(tps, qt, x[1] + x[3], x[2] + x[4], ρ, T)
        function end_state_S(mode, ρ, T, qt, x, Δt, nsub)
            d = net_vec_1m(
                BMT.bulk_microphysics_tendencies(
                    mode, BMT.Microphysics1Moment(), mp, tps, ρ, T, qt, x..., Δt, nsub,
                ),
            )
            xe = x .+ Δt .* d
            Te = T + Lv * ((xe[1] - x[1]) + (xe[3] - x[3])) + Ls * ((xe[2] - x[2]) + (xe[4] - x[4]))
            (S_liq(xe, Te, ρ, qt), S_ice(xe, Te, ρ, qt))
        end
        exact = BMT.rosenbrock_exact()
        unlimited = BMT.RosenbrockAverage(BMT.ExactJacobian(), BMT.ExplicitGrowthDiagonal(), BMT.NoLimiter())
        tol = FT == Float64 ? FT(1e-3) : FT(3e-2)
        # warm, liquid-supersaturated, coarse single step
        let ρ = FT(1.0), T = T_frz + FT(8), qt = FT(0.022), x = FT[1e-4, 0, 1e-4, 0], Δt = FT(240)
            Sl_exact, _ = end_state_S(exact, ρ, T, qt, x, Δt, 1)
            Sl_unlim, _ = end_state_S(unlimited, ρ, T, qt, x, Δt, 1)
            @test abs(Sl_exact) < tol
            @test Sl_unlim < -tol
        end
        # cold, ice-supersaturated, coarse single step
        let ρ = FT(0.9), T = T_frz - FT(20), qt = FT(2.0e-3), x = FT[0, 1e-4, 0, 1e-4], Δt = FT(240)
            _, Si_exact = end_state_S(exact, ρ, T, qt, x, Δt, 1)
            _, Si_unlim = end_state_S(unlimited, ρ, T, qt, x, Δt, 1)
            @test abs(Si_exact) < tol
            @test Si_unlim < -tol
        end
    end

    @testset "Verbose(rosenbrock_donor()) per-process sums to net ($FT)" begin
        rtol = FT == Float64 ? FT(1e-10) : FT(1e-4)
        for r in regimes, nsub in (1, 4, 16)
            Δt = FT(20)
            v = BMT.bulk_microphysics_tendencies(
                BMT.Verbose(BMT.rosenbrock_donor()), BMT.Microphysics1Moment(), mp, tps,
                r.ρ, r.T, r.q_tot, r.x..., Δt, nsub,
            )
            net = net_vec_1m(v)
            recon = SVector((sum(values(v.processes)) + v.clamp_correction)...)
            @test all(isfinite, recon)
            scale = maximum(abs.(net)) + eps(FT)
            @test maximum(abs.(recon - net)) ≤ rtol * scale
        end
    end
end

function test_framework_2m(FT)
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true)
    p3 = mp.ice.scheme
    T_frz = TDI.T_freeze(tps)

    consistent_logλ(ρ, x) =
        P3.get_distribution_logλ(P3.state_from_prognostic(p3, ρ * x[5], ρ * x[6], ρ * x[7], ρ * x[8]))

    # x = [q_lcl, n_lcl, q_rai, n_rai, q_ice, n_ice, q_rim, b_rim]
    ρ = FT(0.78)
    T = FT(273.5)
    q_tot = FT(0.009)
    x = FT[2e-4, 5e7, 1e-4, 4e4, 1e-4, 2e5, 4e-5, 6e-8]
    logλ = consistent_logλ(ρ, x)

    @testset "rosenbrock_exact() on Microphysics2Moment works ($FT)" begin
        for nsub in (1, 2, 8), Δt in (FT(60), FT(300))
            t = BMT.bulk_microphysics_tendencies(
                BMT.rosenbrock_exact(), BMT.Microphysics2Moment(), mp, tps,
                ρ, T, q_tot, x..., logλ, Δt, nsub,
            )
            @test all(isfinite, values(t))
        end
    end

    @testset "rosenbrock_manual() on Microphysics2Moment works ($FT)" begin
        for nsub in (1, 2, 8), Δt in (FT(60), FT(300))
            t = BMT.bulk_microphysics_tendencies(
                BMT.rosenbrock_manual(), BMT.Microphysics2Moment(), mp, tps,
                ρ, T, q_tot, x..., logλ, Δt, nsub,
            )
            @test all(isfinite, values(t))
        end
    end

    @testset "_jacobian_2mp3_manual Tier-1 closed-form entries match ForwardDiff ($FT)" begin
        # Validates the closed-form Tier-1 pieces (`_condevap_derivs`, the
        # ice-number sublimation pathway, the F23 pathway, and `_numadj_derivs`)
        # against ForwardDiff of the primal functions they linearize, across the
        # branches `_jacobian_2mp3_manual` selects between. Full-matrix agreement
        # is not asserted here: Tier 2 (donor-diagonal) and Tier 3 (dropped
        # quadrature couplings) are approximations by design. See #743.
        rtol = FT == Float64 ? FT(1e-9) : FT(1e-2)
        atol = FT == Float64 ? FT(1e-12) : FT(1e-6)
        qmin = UT.ϵ_numerics_2M_M(FT)

        cp_l = TDI.TD.Parameters.cp_l(tps)
        cp_v = TDI.TD.Parameters.cp_v(tps)
        cp_i = TDI.TD.Parameters.cp_i(tps)
        dcp_dliq = cp_l - cp_v
        dcp_dice = cp_i - cp_v
        τ_l = mp.warm_rain.condevap.τ_relax
        τ_i = mp.warm_rain.subdep.τ_relax

        micro(v) = (; q_tot = v[5], q_lcl = v[1], q_icl = v[3], q_rai = v[2], q_sno = zero(v[1]))

        function condevap_manual_and_fd(ρ, T, q_tot, q_lcl, q_rai, q_ice, is_ice)
            τ = is_ice ? τ_i : τ_l
            L = is_ice ? TDI.Lₛ(tps, T) : TDI.Lᵥ(tps, T)
            qᵥ_sat =
                is_ice ?
                TDI.saturation_vapor_specific_content_over_ice(tps, T, ρ) :
                TDI.saturation_vapor_specific_content_over_liquid(tps, T, ρ)
            dqs_dT = CMNonEq.dqcld_dT(qᵥ_sat, L, TDI.Rᵥ(tps), T)
            cp_air = TDI.cpₘ(tps, q_tot, q_lcl + q_rai, q_ice)
            Γ = CMNonEq.gamma_helper(L, cp_air, dqs_dT)
            sat_excess = TDI.q_vap(q_tot, q_lcl + q_rai, q_ice) - qᵥ_sat
            q_limit = is_ice ? q_ice : q_lcl
            d = BMT._condevap_derivs(τ, sat_excess, Γ, cp_air, L, dqs_dT, q_limit, is_ice, dcp_dliq, dcp_dice)
            dep_active = !(is_ice && (T > T_frz) && (sat_excess > 0))
            manual = dep_active ? [d.∂s_liq, d.∂s_rai, d.∂s_ice] : zeros(FT, 3)
            h(v) =
                if is_ice
                    raw = CMNonEq.conv_q_vap_to_q_icl(CMP.ConstantTimescale(τ), nothing, tps, micro(v), (; ρ, T))
                    ifelse(T > T_frz, min(raw, zero(raw)), raw)
                else
                    CMNonEq.conv_q_vap_to_q_lcl(CMP.CloudLiquidFormation(τ), nothing, tps, micro(v), (; ρ, T))
                end
            fd = FD.gradient(h, FT[q_lcl, q_rai, q_ice, zero(q_ice), q_tot])[1:3]
            return manual, fd, dep_active
        end

        # cap_binds true/false for cloud and ice; dep_suppressed for ice.
        condevap_states = (
            (;
                ρ = FT(1.0),
                T = FT(290),
                q_tot = FT(0.02),
                q_lcl = FT(2e-4),
                q_rai = FT(1e-4),
                q_ice = FT(1e-4),
                is_ice = false,
            ),   # cloud, vapor branch (limit_binds=false)
            (;
                ρ = FT(0.9),
                T = FT(290),
                q_tot = FT(0.0005),
                q_lcl = FT(2e-5),
                q_rai = FT(2e-5),
                q_ice = FT(1e-8),
                is_ice = false,
            ), # cloud, limited branch (limit_binds=true)
            (;
                ρ = FT(0.9),
                T = FT(260),
                q_tot = FT(0.008),
                q_lcl = FT(2e-4),
                q_rai = FT(1e-4),
                q_ice = FT(1e-4),
                is_ice = true,
            ),   # ice, dep active, vapor branch
            (;
                ρ = FT(0.9),
                T = FT(250),
                q_tot = FT(0.0003),
                q_lcl = FT(2e-5),
                q_rai = FT(2e-5),
                q_ice = FT(1e-8),
                is_ice = true,
            ),  # ice, dep active, limited branch
            (;
                ρ = FT(0.9),
                T = FT(280),
                q_tot = FT(0.02),
                q_lcl = FT(2e-4),
                q_rai = FT(1e-4),
                q_ice = FT(1e-4),
                is_ice = true,
            ),    # ice, dep_suppressed
        )
        dep_active_flags = Bool[]
        for s in condevap_states
            manual, fd, dep_active = condevap_manual_and_fd(s.ρ, s.T, s.q_tot, s.q_lcl, s.q_rai, s.q_ice, s.is_ice)
            push!(dep_active_flags, dep_active)
            @test all(isapprox.(manual, fd; rtol, atol))
        end
        # confirm both branches of `dep_active` were exercised
        @test any(dep_active_flags)
        @test any(!f for f in dep_active_flags)

        # ice-number sublimation pathway: n_per_q · ∂ₜq_ice_dep, active only when
        # subsaturated (sublimating) with q_ice above the numerics floor.
        function n_sub_manual_and_fd(ρ, T, q_tot, q_lcl, q_rai, q_ice, n_ice)
            qᵥ_sat = TDI.saturation_vapor_specific_content_over_ice(tps, T, ρ)
            dqs_dT = CMNonEq.dqcld_dT(qᵥ_sat, TDI.Lₛ(tps, T), TDI.Rᵥ(tps), T)
            cp_air = TDI.cpₘ(tps, q_tot, q_lcl + q_rai, q_ice)
            Γ = CMNonEq.gamma_helper(TDI.Lₛ(tps, T), cp_air, dqs_dT)
            sat_excess = TDI.q_vap(q_tot, q_lcl + q_rai, q_ice) - qᵥ_sat
            ci = BMT._condevap_derivs(
                τ_i,
                sat_excess,
                Γ,
                cp_air,
                TDI.Lₛ(tps, T),
                dqs_dT,
                q_ice,
                true,
                dcp_dliq,
                dcp_dice,
            )
            dep_active = !((T > T_frz) && (sat_excess > 0))
            raw = CMNonEq.conv_q_vap_to_q_icl(CMP.ConstantTimescale(τ_i), nothing, tps,
                (; q_tot, q_lcl, q_icl = q_ice, q_rai, q_sno = zero(q_ice)), (; ρ, T))
            ∂ₜq_ice_dep = ifelse(T > T_frz, min(raw, zero(raw)), raw)
            n_sub_active = ∂ₜq_ice_dep < 0 && q_ice > qmin && dep_active
            n_per_q = n_ice / max(qmin, q_ice)
            manual =
                n_sub_active ?
                [
                    n_per_q * ci.∂s_liq,
                    n_per_q * ci.∂s_rai,
                    n_per_q * (ci.∂s_ice - ∂ₜq_ice_dep / max(qmin, q_ice)),
                    ∂ₜq_ice_dep / max(qmin, q_ice),
                ] :
                zeros(FT, 4)
            h(v) = begin
                raw_h = CMNonEq.conv_q_vap_to_q_icl(CMP.ConstantTimescale(τ_i), nothing, tps,
                    (; q_tot, q_lcl = v[1], q_icl = v[3], q_rai = v[2], q_sno = zero(v[1])), (; ρ, T))
                dqdep = ifelse(T > T_frz, min(raw_h, zero(raw_h)), raw_h)
                n_per_q_h = v[4] / max(qmin, v[3])
                ifelse(dqdep < 0, n_per_q_h * dqdep, zero(dqdep))
            end
            fd = FD.gradient(h, FT[q_lcl, q_rai, q_ice, n_ice])
            return manual, fd, n_sub_active
        end
        n_sub_states = (
            (;
                ρ = FT(0.9),
                T = FT(250),
                q_tot = FT(0.0005),
                q_lcl = FT(2e-5),
                q_rai = FT(2e-5),
                q_ice = FT(1e-5),
                n_ice = FT(1e-3),
            ), # subsaturated ⇒ active
            (;
                ρ = FT(0.9),
                T = FT(260),
                q_tot = FT(0.008),
                q_lcl = FT(2e-4),
                q_rai = FT(1e-4),
                q_ice = FT(1e-4),
                n_ice = FT(1e-3),
            ),  # supersaturated (growth) ⇒ inactive
        )
        # The nice_ice component subtracts two comparable-magnitude terms
        # (∂s_ice and ∂ₜq_ice_dep / q_ice); the residual is amplified by
        # n_ice / q_ice, loosening the achievable tolerance relative to the
        # other Tier-1 entries.
        atol_nsub = FT == Float64 ? FT(1e-9) : FT(1e-3)
        for s in n_sub_states
            manual, fd, nsa = n_sub_manual_and_fd(s.ρ, s.T, s.q_tot, s.q_lcl, s.q_rai, s.q_ice, s.n_ice)
            @test all(isapprox.(manual, fd; rtol, atol = atol_nsub))
        end
        @test n_sub_manual_and_fd(n_sub_states[1]...)[3]
        @test !n_sub_manual_and_fd(n_sub_states[2]...)[3]

        # F23 deposition-nucleation number pathway.
        ice_nucleation = mp.ice.ice_nucleation
        τ_act = mp.ice.inp_depletion_model.τ_act
        D_nuc = FT(10e-6)
        m_nuc = p3.ρ_i * CO.volume_sphere_D(D_nuc)
        function f23_manual_and_fd(ρ, T, q_tot, q_liq, q_ice, n_ice)
            h(n) = CM_HetIce.deposition_rate(
                ice_nucleation,
                tps,
                T,
                ρ,
                q_tot,
                q_liq,
                q_ice,
                n;
                m_nuc,
                τ_act,
                inpc_log_shift = zero(ρ),
            ).∂ₜn_frz
            fd = FD.derivative(h, n_ice)
            f23_active = h(n_ice) > 0
            manual = f23_active ? -1 / τ_act : zero(FT)
            return manual, fd, f23_active
        end
        f23_active_state =
            (; ρ = FT(0.9), T = FT(250), q_tot = FT(0.005), q_liq = FT(2e-4), q_ice = FT(1e-4), n_ice = FT(1e2))
        f23_inactive_state =
            (; ρ = FT(0.9), T = FT(290), q_tot = FT(0.02), q_liq = FT(2e-4), q_ice = FT(1e-4), n_ice = FT(2e5))
        for s in (f23_active_state, f23_inactive_state)
            manual, fd, _ = f23_manual_and_fd(s.ρ, s.T, s.q_tot, s.q_liq, s.q_ice, s.n_ice)
            @test isapprox(manual, fd; rtol, atol)
        end
        @test f23_manual_and_fd(f23_active_state...)[3]
        @test !f23_manual_and_fd(f23_inactive_state...)[3]

        # `_numadj_derivs` against `CM2.number_tendency_from_mass_limits`, across
        # the interior/low/high/empty regimes, for the cloud, rain, and ice bounds.
        sb = mp.warm_rain.seifert_beheng
        numadj_species = (
            (sb.pdf_c.xc_min, sb.pdf_c.xc_max, sb.numadj.τ),
            (sb.pdf_r.xr_min, sb.pdf_r.xr_max, sb.numadj.τ),
            (FT(1e-12), FT(1e-5), FT(100)),
        )
        for (x_min, x_max, τ) in numadj_species
            q = FT(2e-4)
            n_mid = q / sqrt(x_min * x_max)
            for (q_test, n_test) in (
                (q, q / x_max * FT(0.5)),  # low clamp
                (q, n_mid),                # interior
                (q, q / x_min * FT(2)),    # high clamp
                (qmin / 2, n_mid),         # empty
            )
                manual = collect(BMT._numadj_derivs(FT, q_test, n_test, x_min, x_max, τ, qmin))
                h(v) = CM2.number_tendency_from_mass_limits((; x_min, x_max, τ), v[1], v[2])
                fd = FD.gradient(h, FT[q_test, n_test])
                # the mass coupling is treated explicitly; the implicit part is
                # the relaxation diagonal
                @test manual[1] == 0
                @test isapprox(manual[2], fd[2]; rtol, atol)
            end
        end
    end

    @testset "non-Exact RosenbrockAverage on Microphysics2Moment throws ($FT)" begin
        for mode in (BMT.rosenbrock_donor(), BMT.rosenbrock_coupled())
            @test_throws ArgumentError BMT.bulk_microphysics_tendencies(
                mode, BMT.Microphysics2Moment(), mp, tps,
                ρ, T, q_tot, x..., logλ, FT(60), 4,
            )
        end
    end

    @testset "RosenbrockAverage on warm-rain-only parameters throws ($FT)" begin
        mp_warm = CMP.Microphysics2MParams(FT; with_ice = false)
        @test_throws "requires P3 ice parameters" BMT.bulk_microphysics_tendencies(
            BMT.rosenbrock_exact(), BMT.Microphysics2Moment(), mp_warm, tps,
            ρ, T, q_tot, x..., logλ, FT(60), 4,
        )
    end
end

function _framework_2mp3_args(FT, mode; packed = false)
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true)
    p3 = mp.ice.scheme
    st = P3.state_from_prognostic(
        p3, FT(0.78) * FT(1e-4), FT(0.78) * FT(2e5), FT(0.78) * FT(4e-5), FT(0.78) * FT(6e-8),
    )
    logλ = P3.get_distribution_logλ(st)
    ice_args = (FT(1e-4), FT(2e5), FT(4e-5), FT(6e-8), logλ)
    ice_tail = packed ? BMT._pack_2mp3_ice(mp, ice_args...) : ice_args
    return (
        mode, BMT.Microphysics2Moment(), mp, tps,
        FT(0.78), FT(273.5), FT(0.009),
        FT(2e-4), FT(5e7), FT(1e-4), FT(4e4), ice_tail...,
        FT(60), 4,
    )
end

function test_framework_2mp3_inference(FT)
    for (name, mode) in
        (("rosenbrock_exact()", BMT.rosenbrock_exact()), ("rosenbrock_manual()", BMT.rosenbrock_manual())),
        packed in (false, true)

        args = _framework_2mp3_args(FT, mode; packed)
        form = packed ? "packed" : "positional"
        @testset "$name $form inference and allocations ($FT)" begin
            @test (@inferred BMT.bulk_microphysics_tendencies(args...)) isa NamedTuple
            JET.@test_opt BMT.bulk_microphysics_tendencies(args...)
            trail = BT.@benchmark $(splat(BMT.bulk_microphysics_tendencies))($args) samples = 100 evals = 1
            @test trail.memory == 0
        end
    end
end

function _framework_3m_args(FT, mode...)
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true, moments = :three_moment)
    p3 = mp.ice.scheme
    ρ = FT(0.78)
    (q_ice, n_ice, q_rim, b_rim, z_ice) = (FT(1e-4), FT(2e5), FT(4e-5), FT(6e-8), FT(1e-8))
    st = P3.state_from_prognostic(p3, q_ice * ρ, n_ice * ρ, q_rim * ρ, b_rim * ρ, z_ice * ρ)
    shape = P3.get_distribution_shape(st)
    ice = ((; q_ice, n_ice, q_rim, b_rim, z_ice),)
    tail = isempty(mode) ? () : (FT(60), 4)
    return (
        mode..., BMT.Microphysics2Moment(), mp, tps,
        ρ, FT(273.5), FT(0.009),
        FT(2e-4), FT(5e7), FT(1e-4), FT(4e4), ice, (shape,),
        tail...,
    )
end

function _framework_liq_args(FT, mode...)
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true, liquid = :predicted)
    p3 = mp.ice.scheme
    ρ = FT(0.78)
    (q_ice, n_ice, q_rim, b_rim, q_liq_on_ice) = (FT(1e-4), FT(2e5), FT(4e-5), FT(6e-8), FT(3e-5))
    st = P3.state_from_prognostic(p3, q_ice * ρ, n_ice * ρ, q_rim * ρ, b_rim * ρ, q_liq_on_ice * ρ)
    shape = P3.get_distribution_shape(st)
    ice = ((; q_ice, n_ice, q_rim, b_rim, q_liq_on_ice),)
    tail = isempty(mode) ? () : (FT(60), 4)
    return (
        mode..., BMT.Microphysics2Moment(), mp, tps,
        ρ, FT(273.5), FT(0.009),
        FT(2e-4), FT(5e7), FT(1e-4), FT(4e4), ice, (shape,),
        tail...,
    )
end

function test_framework_3m_inference(FT)
    for (name, mode) in (
        ("instantaneous", ()),
        ("rosenbrock_exact()", (BMT.rosenbrock_exact(),)),
        ("rosenbrock_manual()", (BMT.rosenbrock_manual(),)),
    )
        args = _framework_3m_args(FT, mode...)
        @testset "$name three-moment packed inference and allocations ($FT)" begin
            @test (@inferred BMT.bulk_microphysics_tendencies(args...)) isa NamedTuple
            JET.@test_opt BMT.bulk_microphysics_tendencies(args...)
            trail = BT.@benchmark $(splat(BMT.bulk_microphysics_tendencies))($args) samples = 100 evals = 1
            @test trail.memory == 0
        end
    end
end

function test_framework_liq_inference(FT)
    for (name, mode) in (
        ("instantaneous", ()),
        ("rosenbrock_exact()", (BMT.rosenbrock_exact(),)),
    )
        args = _framework_liq_args(FT, mode...)
        @testset "$name liquid-fraction packed inference and allocations ($FT)" begin
            @test (@inferred BMT.bulk_microphysics_tendencies(args...)) isa NamedTuple
            JET.@test_opt BMT.bulk_microphysics_tendencies(args...)
            trail = BT.@benchmark $(splat(BMT.bulk_microphysics_tendencies))($args) samples = 100 evals = 1
            @test trail.memory == 0
        end
    end
end

# Packed-vs-positional 2M+P3 entry parity: the positional form packs its ice
# scalars and shape via `_pack_2mp3_ice` and must reproduce the packed form
# bit-for-bit.
function test_packed_entry_parity(FT)
    tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)
    mp = CMP.Microphysics2MParams(FT; with_ice = true, is_limited = true)
    p3 = mp.ice.scheme
    Δt = FT(60)

    # deterministic 64-bit LCG state sweep
    s = Ref(UInt64(0x9E3779B97F4A7C15))
    draw() = (s[] = 6364136223846793005 * s[] + 1442695040888963407; Float64(s[] >> 11) * (2.0^-53))
    uni(lo, hi) = lo + (hi - lo) * draw()

    @testset "packed vs positional 2M+P3 entries ($FT)" begin
        for _ in 1:5
            ρ = FT(uni(0.4, 1.3))
            T = FT(TDI.T_freeze(tps) + uni(-25, 20))
            q_tot = FT(uni(1e-4, 2e-2))
            q_lcl = FT(uni(0, 3e-3))
            n_lcl = FT(uni(1e6, 1e9))
            q_rai = FT(uni(0, 1e-3))
            n_rai = FT(uni(1e2, 1e5))
            q_ice = FT(uni(1e-6, 2e-3))
            n_ice = FT(uni(1e3, 1e6))
            q_rim = FT(uni(0, 0.9)) * q_ice
            b_rim = q_rim / FT(uni(100, 800))
            logλ = P3.get_distribution_logλ_from_prognostic(p3, q_ice * ρ, n_ice * ρ, q_rim * ρ, b_rim * ρ)
            ice, shapes = BMT._pack_2mp3_ice(mp, q_ice, n_ice, q_rim, b_rim, logλ)
            common = (mp, tps, ρ, T, q_tot, q_lcl, n_lcl, q_rai, n_rai)
            inst_pos = BMT.bulk_microphysics_tendencies(
                BMT.Microphysics2Moment(), common..., q_ice, n_ice, q_rim, b_rim, logλ,
            )
            inst_pack = BMT.bulk_microphysics_tendencies(BMT.Microphysics2Moment(), common..., ice, shapes)
            @test all(map(===, values(inst_pos), values(inst_pack)))
            for mode in (BMT.rosenbrock_exact(), BMT.rosenbrock_manual()), nsub in (1, 4)
                r_pos = BMT.bulk_microphysics_tendencies(
                    mode, BMT.Microphysics2Moment(), common..., q_ice, n_ice, q_rim, b_rim, logλ, Δt, nsub,
                )
                r_pack = BMT.bulk_microphysics_tendencies(
                    mode, BMT.Microphysics2Moment(), common..., ice, shapes, Δt, nsub,
                )
                @test all(map(===, values(r_pos), values(r_pack)))
            end
        end
    end

    @testset "packed ice input round-trip and category count ($FT)" begin
        @test CMP.n_categories(mp.ice) == 1
        q_ice = FT(1e-4)
        n_ice = FT(2e5)
        q_rim = FT(4e-5)
        b_rim = FT(6e-8)
        logλ = P3.get_distribution_logλ_from_prognostic(p3, q_ice, n_ice, q_rim, b_rim)
        ice, shapes = BMT._pack_2mp3_ice(mp, q_ice, n_ice, q_rim, b_rim, logλ)
        @test ice === ((; q_ice, n_ice, q_rim, b_rim),)
        @test shapes === (P3.get_distribution_shape(p3, logλ),)
        @test isbits(ice) && isbits(shapes)
        @test shapes[1].logλ === logλ === shapes[1].logλ_core
        # category count mismatch throws
        args = (mp, tps, FT(1), FT(273), FT(0.01), FT(1e-4), FT(1e8), FT(1e-4), FT(1e4))
        ice2 = (ice[1], ice[1])
        shapes2 = (shapes[1], shapes[1])
        @test_throws ArgumentError BMT.bulk_microphysics_tendencies(
            BMT.Microphysics2Moment(), args..., ice2, shapes2,
        )
        @test_throws ArgumentError BMT.bulk_microphysics_tendencies(
            BMT.rosenbrock_exact(), BMT.Microphysics2Moment(), args..., ice2, shapes2, FT(60), 1,
        )
        @test_throws ArgumentError BMT.bulk_microphysics_tendencies(
            BMT.Verbose(BMT.rosenbrock_exact()), BMT.Microphysics2Moment(), args..., ice2, shapes2, FT(60), 1,
        )
    end
end

# Structural tests of the layout-parameterized `MicroState` over the default
# (N=8) and extended (N=9/10/12) layouts.
function test_microstate_layout(FT)
    layouts = (
        (; NCAT = 1, LIQ = false, ZM = false, N = 8),
        (; NCAT = 1, LIQ = true, ZM = false, N = 9),
        (; NCAT = 1, LIQ = false, ZM = true, N = 9),
        (; NCAT = 1, LIQ = true, ZM = true, N = 10),
        (; NCAT = 2, LIQ = false, ZM = false, N = 12),
    )

    @testset "MicroState construction, indexing, cat_view ($FT)" begin
        for L in layouts
            @test BMT.state_length(L.NCAT, L.LIQ, L.ZM) == L.N
            MS = BMT.MicroState{FT, L.NCAT, L.LIQ, L.ZM, L.N}
            x = MS(ntuple(i -> FT(i), L.N))
            @test x isa MS
            @test length(x) == L.N
            @test all(x[i] == FT(i) for i in 1:L.N)
            @test (x[BMT.IQ_LCL], x[BMT.IN_LCL], x[BMT.IQ_RAI], x[BMT.IN_RAI]) == (FT(1), FT(2), FT(3), FT(4))
            @test BMT.n_categories(x) == L.NCAT
            @test BMT.n_ice_fields(x) == 4 + L.LIQ + L.ZM
            keys_core = (:q_ice, :n_ice, :q_rim, :b_rim)
            keys_liq = L.LIQ ? (:q_liq_on_ice,) : ()
            keys_z = L.ZM ? (:z_ice,) : ()
            expected_keys = (keys_core..., keys_liq..., keys_z...)
            for j in 1:L.NCAT
                cv = BMT.cat_view(x, Val(j))
                @test keys(cv) == expected_keys
                off = 4 + (j - 1) * (4 + L.LIQ + L.ZM)
                @test (BMT.ice_q(x, Val(j)), BMT.ice_n(x, Val(j)), BMT.ice_qrim(x, Val(j)), BMT.ice_brim(x, Val(j))) ==
                      (FT(off + 1), FT(off + 2), FT(off + 3), FT(off + 4))
                @test cv.q_ice == FT(off + 1)
                if L.LIQ
                    @test BMT.ice_qliq(x, Val(j)) == cv.q_liq_on_ice == FT(off + 5)
                    @test BMT.ice_present_mass(x, Val(j)) == cv.q_ice + cv.q_liq_on_ice
                else
                    @test BMT.ice_present_mass(x, Val(j)) == cv.q_ice
                end
                if L.ZM
                    @test BMT.ice_zice(x, Val(j)) == cv.z_ice == FT(off + 4 + L.LIQ + 1)
                end
            end
        end
    end

    @testset "MicroState length invariant throws ($FT)" begin
        @test_throws DimensionMismatch BMT.MicroState{FT, 1, false, false, 7}(ntuple(i -> FT(i), 7))
        @test_throws DimensionMismatch SA.similar_type(BMT.MicroState2MP3{FT}, FT, SA.Size((7,)))
    end

    @testset "MicroState similar_type and solve preserve the layout type ($FT)" begin
        for L in layouts
            MS = BMT.MicroState{FT, L.NCAT, L.LIQ, L.ZM, L.N}
            x = MS(ntuple(i -> FT(i), L.N))
            N = L.N
            @test SA.similar_type(typeof(x), FT, SA.Size((N,))) == MS
            @test typeof(abs.(x)) == MS
            @test typeof(max.(x .+ x, FT(0))) == MS
            J = FT(2) .* one(SA.SMatrix{N, N, FT})
            @test typeof(J * x) == MS
            @test typeof(J \ x) == MS
            @test typeof(J * (J \ (J * x))) == MS
            @test typeof(x * x') <: SA.SMatrix{N, N, FT}
        end
    end
end

test_framework_1m(Float64)
test_framework_1m(Float32)
test_framework_2m(Float64)
test_framework_2m(Float32)

test_microstate_layout(Float64)
test_microstate_layout(Float32)

test_packed_entry_parity(Float64)
test_packed_entry_parity(Float32)

test_framework_2mp3_inference(Float64)
test_framework_2mp3_inference(Float32)

test_framework_3m_inference(Float64)
test_framework_3m_inference(Float32)
test_framework_liq_inference(Float64)
test_framework_liq_inference(Float32)
