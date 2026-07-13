# Numerical study: monotonicity and solver behavior of the three-moment P3
# shape residual r(μ). Deliverable report:
#   /home/haakon/notes/p3-mmc2025/impl/3mom-monotonicity-study.md
#
# For targets (L/N, Z/N), the slope is pinned analytically
#   logλ(μ) = (1/6)[logΓ(μ+7) − logΓ(μ+1) − log(Z/N)],
# and μ ∈ [0, μ_max] solves the piecewise mass residual
#   r(μ) = logmass_gamma_moment(state, μ, logλ(μ); n=0)
#          − loggamma_moment(μ, logλ(μ); k=0) − log(L/N).
# Define φ(μ) = logmass_gamma_moment − loggamma_moment (the model log(L/N));
# then r(μ) = φ(μ) − log(L/N), so log(L/N) is a vertical shift and monotonicity
# of r is a property of φ alone.

import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.Parameters as CMP
import SpecialFunctions as SF
const RS = P3.RS  # RootSolvers is a transitive dep, reached via the P3 module
using Printf

# ------------------------------------------------------------------ #
# Residual building blocks
# ------------------------------------------------------------------ #

logλ_of_μ(μ, logZdN) = (SF.loggamma(μ + 7) - SF.loggamma(μ + 1) - logZdN) / 6

# φ(μ) = model log(L/N) for a given state and log(Z/N)
function φ(state, μ, logZdN)
    lλ = logλ_of_μ(μ, logZdN)
    return P3.logmass_gamma_moment(state, μ, lλ; n = 0) - P3.loggamma_moment(μ, lλ; k = 0)
end

# ------------------------------------------------------------------ #
# Grid definition
# ------------------------------------------------------------------ #

# Rime states: F_rim = 0 needs only one entry (ρ_rim irrelevant, D_gr=D_cr=Inf).
function rime_states(params, FT)
    states = Tuple{String, FT, FT}[]
    push!(states, ("Frim=0.00", FT(0), FT(0)))
    for Fr in (FT(0.3), FT(0.6), FT(0.9), FT(0.99)), ρr in (FT(100), FT(400), FT(900))
        push!(states, (@sprintf("Frim=%.2f,ρr=%d", Fr, ρr), Fr, ρr))
    end
    return states
end

# Z/N targets, parametrized by the target logλ at μ=10 (v_c). The logλ(μ) window
# is ≈ [v_c−1.50, v_c+0.56] over μ∈[0,20], so these positions place it relative
# to the [2,17] rails: fully below, straddling low rail, inside, straddling high
# rail, fully above.
function zn_targets(FT)
    # logZdN such that logλ(μ=10) = v_c  ⇔  logZdN = (logΓ(17)−logΓ(11)) − 6 v_c
    base = SF.loggamma(FT(17)) - SF.loggamma(FT(11))
    vcs = (
        ("below[<2]", FT(0.0)),
        ("straddle_lo", FT(2.0)),
        ("inside_lo", FT(4.0)),
        ("inside_mid", FT(9.5)),
        ("inside_hi", FT(15.0)),
        ("straddle_hi", FT(17.0)),
        ("above[>17]", FT(19.0)),
    )
    return [(name, base - 6 * vc, vc) for (name, vc) in vcs]
end

# L/N probes: mean particle mass [kg], the numadj band [1e-12,1e-5] extended a
# decade each way.
LN_targets(FT) = FT.((1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-4))

# ------------------------------------------------------------------ #
# Monotonicity analysis of φ over a μ grid
# ------------------------------------------------------------------ #

struct MonoResult{FT}
    strictly_increasing::Bool
    n_reversals::Int          # number of contiguous non-increasing runs
    worst_drop::FT            # largest peak-to-trough decrease across a reversal
    worst_drop_loc::FT        # μ at the peak of the worst reversal
    worst_drop_width::FT      # μ width of the worst reversal
    min_dφ::FT                # tightest monotonicity margin (smallest per-step increment)
    φmin::FT
    φmax::FT
end

function analyze_mono(μs, φs::AbstractVector{FT}) where {FT}
    n = length(φs)
    dφ = diff(φs)
    inc = all(>(0), dφ)
    min_dφ = minimum(dφ)
    # Find contiguous runs where φ is non-increasing (dφ <= 0)
    n_rev = 0
    worst_drop = zero(FT)
    worst_loc = zero(FT)
    worst_width = zero(FT)
    i = 1
    while i <= n - 1
        if dφ[i] <= 0
            j = i
            while j <= n - 1 && dφ[j] <= 0
                j += 1
            end
            # run of non-increasing steps spans indices i..j (φ peak at i, trough at j)
            n_rev += 1
            drop = φs[i] - φs[j]
            if drop > worst_drop
                worst_drop = drop
                worst_loc = μs[i]
                worst_width = μs[j] - μs[i]
            end
            i = j
        else
            i += 1
        end
    end
    return MonoResult{FT}(inc, n_rev, worst_drop, worst_loc, worst_width, min_dφ, minimum(φs), maximum(φs))
end

# ------------------------------------------------------------------ #
# Production solver pattern (Brent + FixedIterations) on r(μ)=φ(μ)−logLdN
# ------------------------------------------------------------------ #

function solve_μ(state, logZdN, logLdN, maxiters; lo, hi)
    FT = eltype(state)
    r(μ) = φ(state, μ, logZdN) - logLdN
    flo, fhi = r(FT(lo)), r(FT(hi))
    if !isfinite(flo) || !isfinite(fhi) || flo * fhi > 0
        return abs(flo) <= abs(fhi) ? FT(lo) : FT(hi)
    end
    sol = RS.find_zero(
        r,
        RS.BrentsMethod(FT(lo), FT(hi)),
        RS.CompactSolution(),
        P3.FixedIterations{FT}(),
        maxiters,
    )
    return clamp(sol.root, FT(lo), FT(hi))
end

# Independent bisection reference to cross-check the 200-iteration Brent solve
function bisect_ref(state, logZdN, logLdN; lo, hi, tol = 1e-13, maxit = 200)
    FT = Float64
    r(μ) = Float64(φ(state, μ, logZdN)) - Float64(logLdN)
    a, b = FT(lo), FT(hi)
    fa = r(a)
    (fa * r(b) > 0) && return abs(fa) <= abs(r(b)) ? a : b
    for _ in 1:maxit
        m = (a + b) / 2
        fm = r(m)
        (abs(b - a) < tol) && return m
        if fa * fm <= 0
            b = m
        else
            a, fa = m, fm
        end
    end
    return (a + b) / 2
end

# ------------------------------------------------------------------ #
# Bracket-interaction designs (Task 4)
# ------------------------------------------------------------------ #

# μ where logλ(μ)=c on [0,μmax], clamped (logλ monotone increasing in μ)
function μ_at_logλ(logZdN, c, FT; μmax = FT(20))
    logλ_of_μ(FT(0), logZdN) >= c && return FT(0)
    logλ_of_μ(μmax, logZdN) <= c && return μmax
    a, b = FT(0), μmax
    for _ in 1:80
        m = (a + b) / 2
        if logλ_of_μ(m, logZdN) < c
            a = m
        else
            b = m
        end
    end
    return (a + b) / 2
end

# (a1) clamp logλ inside the residual, solve μ∈[0,μmax]
function design_a1(state, logZdN, logLdN; FT, μmax, maxiters, lλlo = FT(2), lλhi = FT(17))
    r(μ) = begin
        lλ = clamp(logλ_of_μ(μ, logZdN), lλlo, lλhi)
        P3.logmass_gamma_moment(state, μ, lλ; n = 0) - P3.loggamma_moment(μ, lλ; k = 0) - logLdN
    end
    lo, hi = FT(0), μmax
    flo, fhi = r(lo), r(hi)
    μ = if !isfinite(flo) || !isfinite(fhi) || flo * fhi > 0
        abs(flo) <= abs(fhi) ? lo : hi
    else
        clamp(RS.find_zero(r, RS.BrentsMethod(lo, hi), RS.CompactSolution(), P3.FixedIterations{FT}(), maxiters).root, lo, hi)
    end
    return (μ, clamp(logλ_of_μ(μ, logZdN), lλlo, lλhi))
end

# (a2) unclamped residual solve μ∈[0,μmax]; clamp logλ only on output (design sketch 2.6)
function design_a2(state, logZdN, logLdN; FT, μmax, maxiters, lλlo = FT(2), lλhi = FT(17))
    μ = solve_μ(state, logZdN, logLdN, maxiters; lo = FT(0), hi = μmax)
    return (μ, clamp(logλ_of_μ(μ, logZdN), lλlo, lλhi))
end

# (b) restrict μ bracket to logλ(μ)∈[2,17]; return unclamped logλ(μ)
function design_b(state, logZdN, logLdN; FT, μmax, maxiters, lλlo = FT(2), lλhi = FT(17))
    μlo = μ_at_logλ(logZdN, lλlo, FT; μmax)
    μhi = μ_at_logλ(logZdN, lλhi, FT; μmax)
    (μlo >= μhi) && (return (μlo, logλ_of_μ(μlo, logZdN)))  # empty bracket, degenerate
    μ = solve_μ(state, logZdN, logLdN, maxiters; lo = μlo, hi = μhi)
    return (μ, logλ_of_μ(μ, logZdN))
end

# ------------------------------------------------------------------ #
# Driver
# ------------------------------------------------------------------ #

function run_study(::Type{FT}; Δμ = 0.02, μmax = 20.0) where {FT}
    params = CMP.ParametersP3(FT)
    μs = collect(FT(0):FT(Δμ):FT(μmax))
    states = rime_states(params, FT)
    zns = zn_targets(FT)
    # reference states/params in Float64 for the F32 comparison
    paramsF64 = CMP.ParametersP3(Float64)

    println("\n########## FT = $FT ##########")
    println("μ grid: 0:$(Δμ):$(μmax)  ($(length(μs)) points)")

    # ---- Report thresholds for each rime state ----
    println("\n-- rime-state thresholds [m] --")
    @printf("%-18s %12s %12s %12s\n", "state", "D_th", "D_gr", "D_cr")
    for (nm, Fr, ρr) in states
        st = P3.P3State(params, FT(1e-6), FT(1e5), Fr, ρr)
        @printf("%-18s %12.4e %12.4e %12.4e\n", nm, st.D_th, st.D_gr, st.D_cr)
    end

    # ---- Monotonicity sweep ----
    println("\n-- monotonicity of φ(μ) over the (rime, Z/N) grid --")
    @printf("%-18s %-12s %6s %5s %11s %10s %9s | %10s %10s\n",
        "state", "ZN_target", "v_c", "incr", "worst_drop", "drop_μ", "F32-err", "φmin", "φmax")
    n_total = 0
    n_nonmono = 0
    worst_global = (FT(0), "", "")
    min_margin = (FT(Inf), "", "")
    for (snm, Fr, ρr) in states
        st = P3.P3State(params, FT(1e-6), FT(1e5), Fr, ρr)
        stF64 = P3.P3State(paramsF64, 1e-6, 1e5, Float64(Fr), Float64(ρr))
        for (znm, logZdN, vc) in zns
            n_total += 1
            φs = [φ(st, μ, logZdN) for μ in μs]
            mr = analyze_mono(μs, φs)
            if mr.min_dφ < min_margin[1]
                min_margin = (mr.min_dφ, snm, znm)
            end
            # F32-vs-F64 reference discrepancy (only meaningful for FT=Float32)
            f32err = FT(0)
            if FT === Float32
                φs64 = [φ(stF64, Float64(μ), Float64(logZdN)) for μ in μs]
                f32err = FT(maximum(abs.(Float64.(φs) .- φs64)))
            end
            if !mr.strictly_increasing
                n_nonmono += 1
                if mr.worst_drop > worst_global[1]
                    worst_global = (mr.worst_drop, snm, znm)
                end
            end
            @printf("%-18s %-12s %6.1f %5s %11.3e %10.3f %9.2e | %10.3f %10.3f\n",
                snm, znm, vc, mr.strictly_increasing ? "yes" : "NO",
                mr.worst_drop, mr.worst_drop_loc, f32err, mr.φmin, mr.φmax)
        end
    end
    @printf("\nsummary FT=%s: %d combos, %d non-monotone. worst drop = %.3e at (%s, %s)\n",
        FT, n_total, n_nonmono, worst_global[1], worst_global[2], worst_global[3])
    @printf("tightest monotonicity margin (min per-step Δφ over grid) = %.3e at (%s, %s)\n",
        min_margin[1], min_margin[2], min_margin[3])

    return (; params, paramsF64, μs, states, zns)
end

# Fine-grid confirmation that no narrow reversal is missed at the default step.
function confirm_fine(::Type{FT}; Δμ = 0.002, μmax = 20.0) where {FT}
    params = CMP.ParametersP3(FT)
    μs = collect(FT(0):FT(Δμ):FT(μmax))
    n_nonmono = 0
    minmarg = FT(Inf)
    for (_, Fr, ρr) in rime_states(params, FT), (_, logZdN, _) in zn_targets(FT)
        st = P3.P3State(params, FT(1e-6), FT(1e5), Fr, ρr)
        φs = [φ(st, μ, logZdN) for μ in μs]
        mr = analyze_mono(μs, φs)
        mr.strictly_increasing || (n_nonmono += 1)
        minmarg = min(minmarg, mr.min_dφ)
    end
    @printf("fine-grid (Δμ=%.3f, FT=%s): %d non-monotone; min per-step Δφ = %.3e\n",
        Δμ, FT, n_nonmono, minmarg)
end

# ------------------------------------------------------------------ #
# Task 3: solver-budget check
# ------------------------------------------------------------------ #

# Reference is always Float64. `FTs` is the solver precision under test. The
# target log(L/N) is defined in Float64 (root = μt by construction) and passed
# to the FTs solver as `FTs(logLdN)`, so the error is the honest FTs-vs-Float64
# discrepancy.
function run_budget(::Type{FTs}, ctx; μmax = 20.0) where {FTs}
    (; params, states, zns) = ctx            # ctx is the Float64 context
    prod_iters = FTs === Float32 ? 8 : 10
    μ_targets = (2.0, 5.0, 8.0, 11.0, 14.0, 17.0)
    println("\n-- Task 3: solver budget (production $(prod_iters) iters, $(FTs) solve vs Float64 ref) --")
    # A target's mean particle mass L/N = exp(logLdN); physical band = [1e-13, 1e-4] kg.
    inband(logLdN) = -13 * log(10) <= logLdN <= -4 * log(10)
    # `restrict`: :all or :phys (physical mean-mass band only)
    function scan(iters, restrict)
        m = 0.0; mλ = 0.0; mM6 = 0.0; refchk = 0.0
        worst = ("", "", 0.0, 0.0)
        for (snm, Fr, ρr) in states
            st64 = P3.P3State(params, 1e-6, 1e5, Fr, ρr)
            sts = P3.P3State(CMP.ParametersP3(FTs), FTs(1e-6), FTs(1e5), FTs(Fr), FTs(ρr))
            for (znm, logZdN, _) in zns
                φs = [φ(st64, μ, logZdN) for μ in 0.0:0.02:μmax]
                all(>(0), diff(φs)) || continue
                for μt in μ_targets
                    logLdN = φ(st64, μt, logZdN)
                    (restrict === :phys && !inband(logLdN)) && continue
                    μ_ref = bisect_ref(st64, logZdN, logLdN; lo = 0.0, hi = μmax)
                    refchk = max(refchk, abs(Float64(solve_μ(st64, logZdN, logLdN, 200; lo = 0.0, hi = μmax)) - μ_ref))
                    μp = Float64(solve_μ(sts, FTs(logZdN), FTs(logLdN), iters; lo = FTs(0), hi = FTs(μmax)))
                    dμ = abs(μp - μ_ref)
                    dλ = abs(exp(logλ_of_μ(μp, logZdN)) - exp(logλ_of_μ(μ_ref, logZdN))) / exp(logλ_of_μ(μ_ref, logZdN))
                    m6p = exp(Float64(P3.logmass_gamma_moment(sts, FTs(μp), FTs(logλ_of_μ(μp, logZdN)); n = 6)))
                    m6r = exp(P3.logmass_gamma_moment(st64, μ_ref, logλ_of_μ(μ_ref, logZdN); n = 6))
                    dM6 = abs(m6p - m6r) / max(m6r, floatmin(Float64))
                    (dμ > m) && (m = dμ; worst = (snm, znm, μt, exp(logLdN)))
                    mλ = max(mλ, dλ); mM6 = max(mM6, dM6)
                end
            end
        end
        return (m, mλ, mM6, refchk, worst)
    end

    for restrict in (:all, :phys)
        (m, mλ, mM6, refchk, worst) = scan(prod_iters, restrict)
        @printf("[%s]  max|μ_prod−μ_ref| = %.3e at (%s, %s, μt=%.1f, L/N=%.2e kg)\n",
            restrict, m, worst[1], worst[2], worst[3], worst[4])
        @printf("       max rel err λ = %.3e   max rel err M6(mass) = %.3e   ref self-check = %.3e\n",
            mλ, mM6, refchk)
    end

    println("budget scan  max|μ_iters − μ_ref(F64)|   [all | physical band]:")
    for iters in 2:1:12
        ma = scan(iters, :all)[1]
        mp = scan(iters, :phys)[1]
        @printf("  iters=%2d : all=%.3e %-9s phys=%.3e %s\n",
            iters, ma, ma < 0.01 ? "(<0.01)" : "", mp, mp < 0.01 ? "(<0.01)" : "")
    end
end

# ------------------------------------------------------------------ #
# Task 4: bracket-interaction continuity across a rail
# ------------------------------------------------------------------ #

# C0 diagnostic: a continuous target->result map has max consecutive jump that
# shrinks in proportion to the sweep step, so jump(coarse)/jump(fine) ≈ (n_fine/n_coarse).
# A genuine discontinuity keeps an O(1) jump at both resolutions (ratio ≈ 1).
function sweep_design(f, st, logZdN, logLdNs; FT, μmax, maxiters)
    pts = [f(st, logZdN, FT(x); FT = FT, μmax = FT(μmax), maxiters = maxiters) for x in logLdNs]
    jμ = maximum(abs(pts[i+1][1] - pts[i][1]) for i in 1:(length(pts)-1))
    jλ = maximum(abs(pts[i+1][2] - pts[i][2]) for i in 1:(length(pts)-1))
    # target reproduction error at strictly-interior roots (limiter-saturated
    # endpoints excluded: those intentionally do not reproduce the target)
    terr = FT(0)
    for (k, x) in enumerate(logLdNs)
        (μ, lλ) = pts[k]
        (FT(0.02) < μ < FT(μmax) - FT(0.02)) || continue
        model = P3.logmass_gamma_moment(st, μ, lλ; n = 0) - P3.loggamma_moment(μ, lλ; k = 0)
        terr = max(terr, abs(model - FT(x)))
    end
    return (jμ, jλ, terr)
end

function run_bracket(::Type{FT}; μmax = 20.0, nsweep = 400) where {FT}
    params = CMP.ParametersP3(FT)
    prod_iters = FT === Float32 ? 8 : 10
    st = P3.P3State(params, FT(1e-6), FT(1e5), FT(0.6), FT(400))
    base = SF.loggamma(FT(17)) - SF.loggamma(FT(11))
    designs = ((:a1, design_a1), (:a2, design_a2), (:b, design_b))
    println("\n-- Task 4: bracket interaction, FT=$FT  (state Frim=0.60 ρr=400) --")

    for (rail_name, vc, rail) in (("lower rail logλ=2", FT(2.0), FT(2)), ("upper rail logλ=17", FT(17.0), FT(17)))
        logZdN = base - 6 * vc
        μstar = μ_at_logλ(logZdN, rail, FT; μmax = FT(μmax))
        @printf("\n%s: crosses at μ*=%.3f (logλ(0)=%.3f, logλ(20)=%.3f)\n",
            rail_name, μstar, logλ_of_μ(FT(0), logZdN), logλ_of_μ(FT(20), logZdN))
        # a1-clamped residual monotonicity check on the μ grid
        φ_a1 = map(FT(0):FT(0.02):FT(μmax)) do μ
            lλ = clamp(logλ_of_μ(μ, logZdN), FT(2), FT(17))
            P3.logmass_gamma_moment(st, μ, lλ; n = 0) - P3.loggamma_moment(μ, lλ; k = 0)
        end
        @printf("  a1-clamped φ strictly increasing on μ-grid: %s\n", all(>(0), diff(φ_a1)) ? "yes" : "NO")

        φlo = φ(st, FT(0.001), logZdN)
        φhi = φ(st, FT(μmax) - FT(0.001), logZdN)
        @printf("  %-6s %12s %12s | %12s %12s | %10s %10s\n",
            "design", "jμ(coarse)", "jμ(fine)", "jλ(coarse)", "jλ(fine)", "jμ_ratio", "tgt_err")
        for (nm, f) in designs
            xc = range(φlo, φhi; length = nsweep)
            xf = range(φlo, φhi; length = 5 * nsweep)
            (jμc, jλc, terr) = sweep_design(f, st, logZdN, xc; FT, μmax, maxiters = prod_iters)
            (jμf, jλf, _) = sweep_design(f, st, logZdN, xf; FT, μmax, maxiters = prod_iters)
            @printf("  %-6s %12.3e %12.3e | %12.3e %12.3e | %10.2f %10.3e\n",
                nm, jμc, jμf, jλc, jλf, jμc / max(jμf, eps(FT)), terr)
        end
    end
end

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

ctx64 = run_study(Float64)
ctx32 = run_study(Float32)
confirm_fine(Float64)
confirm_fine(Float32)
run_budget(Float64, ctx64)
run_budget(Float32, ctx32)
run_bracket(Float64)
run_bracket(Float32)
println("\nDONE")
