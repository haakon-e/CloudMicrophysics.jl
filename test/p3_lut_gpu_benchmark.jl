# GPU microbenchmark harness for the P3 lookup tables (Phase-3).
#
# Measures device throughput (cells/s and us/cell) for:
#   (a) warm-rain-only 2-moment bulk tendency baseline,
#   (b) full 2M+P3 bulk tendency with the P3 tables in variant A, C, and hybrid,
#   (c) the quadrature reference path (p3_tables = nothing), optional.
#
# Each path is run over three state mixes that stress memory access realism:
#   :uniform  - every cell holds the same mixed-phase state,
#   :block    - contiguous warps share a regime (warm / mixed / hail),
#   :shuffled - a per-cell random regime,
# in both Float64 and Float32.
#
# Runs on the CUDA backend when a device is present, and on the
# KernelAbstractions CPU backend otherwise (the same code path), so the CPU run
# is a smoke test of the device kernels before the batch job.
#
# Usage: julia --project=<env> test/p3_lut_gpu_benchmark.jl [outdir] [smoke]

import Test as TT
using KernelAbstractions
using ClimaComms
ClimaComms.@import_required_backends
import Adapt
import Random
import Printf: @sprintf

import CloudMicrophysics as CM
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.ThermodynamicsInterface as TDI
import CloudMicrophysics.P3Scheme as P3
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

const work_groups = 256

if ClimaComms.device() isa ClimaComms.CUDADevice
    using CUDA
    const backend = CUDABackend()
    CUDA.allowscalar(false)
    const ArrayType = CuArray
    const ON_GPU = true
    @info "P3 LUT GPU benchmark running on CUDA GPU" backend ArrayType
else
    const backend = CPU()
    const ArrayType = Array
    const ON_GPU = false
    @info "No CUDA GPU found. Running on the KernelAbstractions CPU backend (smoke test)" backend ArrayType
end

# --- Kernels -----------------------------------------------------------------
# The output is a scalar reduction of the tendency tuple, so the kernel forces
# the full computation while keeping the output layout independent of the
# tendency field set (warm-only returns eight fields, the P3 path nine).

@inline _reduce_tendency(r) = sum(values(r))

@kernel inbounds = true function warm_only_kernel!(
    mp, tps, out, ρ, T, q_tot, q_lcl, n_lcl, q_rai, n_rai,
)
    i = @index(Global, Linear)
    r = BMT.bulk_microphysics_tendencies(
        BMT.Microphysics2Moment(), mp, tps,
        ρ[i], T[i], q_tot[i], q_lcl[i], n_lcl[i], q_rai[i], n_rai[i],
    )
    out[i] = _reduce_tendency(r)
end

@kernel inbounds = true function full_p3_kernel!(
    mp, tps, tables, out,
    ρ, T, q_tot, q_lcl, n_lcl, q_rai, n_rai, q_ice, n_ice, q_rim, b_rim, logλ,
)
    i = @index(Global, Linear)
    r = BMT.bulk_microphysics_tendencies(
        BMT.Microphysics2Moment(), mp, tps,
        ρ[i], T[i], q_tot[i], q_lcl[i], n_lcl[i], q_rai[i], n_rai[i],
        q_ice[i], n_ice[i], q_rim[i], b_rim[i], logλ[i];
        p3_tables = tables,
    )
    out[i] = _reduce_tendency(r)
end

# --- State archetypes and mixes ----------------------------------------------
# Three physically distinct regimes. `q_ice = 0` (warm) skips the P3 ice branch;
# mixed and hail exercise it at a light and a heavily-rimed loading. Numbers are
# specific contents (per kg air); the tendency multiplies by ρ internally.

const REGIMES = (
    # (ρ, T, q_tot, q_lcl, n_lcl, q_rai, n_rai, q_ice, n_ice, q_rim, b_rim)
    warm = (1.10, 290.0, 1.20e-2, 1.0e-3, 1.0e8, 1.0e-4, 1.0e4, 0.0, 0.0, 0.0, 0.0),
    mixed = (0.70, 263.0, 3.0e-3, 3.0e-4, 5.0e7, 1.0e-4, 8.0e3, 5.0e-4, 1.0e5, 1.0e-4, 1.0e-4 / 300),
    hail = (0.90, 270.0, 4.0e-3, 5.0e-4, 3.0e7, 3.0e-4, 1.0e4, 2.0e-3, 5.0e4, 1.8e-3, 1.8e-3 / 800),
)

# Field order matching REGIMES tuples.
const FIELDS =
    (:ρ, :T, :q_tot, :q_lcl, :n_lcl, :q_rai, :n_rai, :q_ice, :n_ice, :q_rim, :b_rim)

# logλ for a regime; 0 when there is no ice (branch is skipped).
function regime_logλ(::Type{FT}, params, reg) where {FT}
    r = REGIMES[reg]
    (; ρ, q_ice, n_ice, q_rim, b_rim) = NamedTuple{FIELDS}(FT.(r))
    q_ice > 0 || return zero(FT)
    state = P3.state_from_prognostic(params, ρ * q_ice, ρ * n_ice, ρ * q_rim, ρ * b_rim)
    return P3.get_distribution_logλ(state)
end

# Per-cell regime assignment for a mix.
function regime_indices(N, mix; seed = 0xC0FFEE)
    regs = (:warm, :mixed, :hail)
    if mix === :uniform
        return fill(:mixed, N)
    elseif mix === :block
        # Contiguous warps (32 lanes) share a regime.
        return [regs[mod(fld(i - 1, 32), 3) + 1] for i in 1:N]
    elseif mix === :shuffled
        rng = Random.MersenneTwister(seed)
        return [regs[rand(rng, 1:3)] for i in 1:N]
    else
        error("unknown mix $mix")
    end
end

# Build device input arrays for a mix, plus the precomputed logλ per cell.
function build_inputs(::Type{FT}, N, mix, params) where {FT}
    idx = regime_indices(N, mix)
    logλ_by_reg = Dict(reg => regime_logλ(FT, params, reg) for reg in keys(REGIMES))
    cols = Dict(f => Vector{FT}(undef, N) for f in FIELDS)
    logλ = Vector{FT}(undef, N)
    for i in 1:N
        reg = idx[i]
        vals = NamedTuple{FIELDS}(FT.(REGIMES[reg]))
        for f in FIELDS
            cols[f][i] = vals[f]
        end
        logλ[i] = logλ_by_reg[reg]
    end
    dev(v) = ArrayType(v)
    return (;
        ρ = dev(cols[:ρ]), T = dev(cols[:T]), q_tot = dev(cols[:q_tot]),
        q_lcl = dev(cols[:q_lcl]), n_lcl = dev(cols[:n_lcl]),
        q_rai = dev(cols[:q_rai]), n_rai = dev(cols[:n_rai]),
        q_ice = dev(cols[:q_ice]), n_ice = dev(cols[:n_ice]),
        q_rim = dev(cols[:q_rim]), b_rim = dev(cols[:b_rim]),
        logλ = dev(logλ),
    )
end

# --- Table build -------------------------------------------------------------
# Grids match the Phase-2 variant tests (validated); build is one-time on the
# host with -t auto threads.
function build_tables(::Type{FT}) where {FT}
    mp = CMP.Microphysics2MParams(FT; with_ice = true, quad = CM.Quadrature.GaussLegendre(FT, 6))
    params, vel, aps = mp.ice.scheme, mp.ice.terminal_velocity, mp.warm_rain.air_properties
    psd_c, psd_r = mp.ice.cloud_pdf, mp.ice.rain_pdf
    build_quad = CM.Quadrature.GaussLegendre(FT, 16)
    out_quad = CM.Quadrature.GaussLegendre(FT, 8)

    rgrid = P3.P3TableGrid{FT}(n_logλ = 48, n_F_rim = 14, n_ρ_rim = 10, n_ρ_air = 5)
    rate_tables = P3.build_p3_lookup_tables(params, vel, aps; grid = rgrid)
    cgrid = P3.P3CollisionGrid{FT}(
        n_logλ = 32, n_F_rim = 14, n_ρ_rim = 10, n_ρ_air = 5, n_x_c = 12, n_Dr = 12, build_order = 12,
    )
    ctables = P3.build_p3_collision_tables(params, vel, aps, psd_c, psd_r; grid = cgrid, quad = build_quad)
    itables = P3.build_p3_collision_inner_tables(params, vel, aps, psd_c, psd_r; quad = build_quad)

    engA = BMT.P3IceTables(rate_tables, ctables, itables, out_quad, BMT.P3CollisionVariantA())
    engC = BMT.P3IceTables(rate_tables, ctables, itables, out_quad, BMT.P3CollisionVariantC())
    engH = BMT.P3IceTables(rate_tables, ctables, itables, out_quad, BMT.P3CollisionHybrid(FT(1)))
    return (; mp, params, engA, engC, engH)
end

# --- Timing ------------------------------------------------------------------
# Minimum wall time over K synchronized launches, excluding the first
# (compile) launch, which is timed separately.
function time_launch(launch!, K)
    tmin = Inf
    for _ in 1:K
        t = @elapsed begin
            launch!()
            KernelAbstractions.synchronize(backend)
        end
        tmin = min(tmin, t)
    end
    return tmin
end

# --- Main --------------------------------------------------------------------
function run_benchmarks(; outdir, smoke)
    N_table = smoke ? 4096 : (ON_GPU ? 1 << 20 : 8192)
    N_quad = smoke ? 512 : (ON_GPU ? 1 << 14 : 512)
    K = smoke ? 2 : 5
    run_quad = true
    mixes = (:uniform, :block, :shuffled)

    rows = NamedTuple[]
    isdir(outdir) || mkpath(outdir)

    kwarm! = warm_only_kernel!(backend, work_groups)
    kfull! = full_p3_kernel!(backend, work_groups)

    for FT in (Float64, Float32)
        tps = TDI.TD.Parameters.ThermodynamicsParameters(FT)

        @info "=== Building P3 tables (FT = $FT) ==="
        t_build = @elapsed tb = build_tables(FT)
        (; mp, params) = tb
        mp_warm = CMP.Microphysics2MParams(FT; with_ice = false)
        table_MB = Base.summarysize(tb.engA) / (1024^2)
        @info "Tables built" FT t_build_s=round(t_build, digits = 2) table_MB=round(table_MB, digits = 2)

        # Move the table-backed engines to the device array type. On the CPU
        # backend this is an Array->Array round-trip through the same Adapt path.
        engA = Adapt.adapt(ArrayType, tb.engA)
        engC = Adapt.adapt(ArrayType, tb.engC)
        engH = Adapt.adapt(ArrayType, tb.engH)

        for mix in mixes
            inp = build_inputs(FT, N_table, mix, params)
            out = KernelAbstractions.allocate(backend, FT, N_table)

            # Each path: (name, N, launch closure, table footprint MB)
            warm_launch() = kwarm!(
                mp_warm, tps, out,
                inp.ρ, inp.T, inp.q_tot, inp.q_lcl, inp.n_lcl, inp.q_rai, inp.n_rai;
                ndrange = N_table,
            )
            p3_launch(tables) = kfull!(
                mp, tps, tables, out,
                inp.ρ, inp.T, inp.q_tot, inp.q_lcl, inp.n_lcl, inp.q_rai, inp.n_rai,
                inp.q_ice, inp.n_ice, inp.q_rim, inp.b_rim, inp.logλ;
                ndrange = N_table,
            )

            paths = [
                (name = "warm-only", N = N_table, launch = warm_launch, tab = 0.0),
                (name = "variantA", N = N_table, launch = () -> p3_launch(engA), tab = table_MB),
                (name = "variantC", N = N_table, launch = () -> p3_launch(engC), tab = table_MB),
                (name = "hybrid", N = N_table, launch = () -> p3_launch(engH), tab = table_MB),
            ]

            for p in paths
                try
                    t_compile = @elapsed begin
                        p.launch()
                        KernelAbstractions.synchronize(backend)
                    end
                    tmin = time_launch(p.launch, K)
                    push_row!(rows, FT, p.name, mix, p.N, tmin, t_compile, p.tab)
                    report_row(FT, p.name, mix, p.N, tmin, t_compile)
                catch err
                    @error "path failed" path=p.name FT mix exception=(err, catch_backtrace())
                end
            end
        end

        # (c) quadrature reference path: smaller N, uniform+shuffled only.
        if run_quad
            for mix in (:uniform, :shuffled)
                try
                    inp = build_inputs(FT, N_quad, mix, params)
                    out = KernelAbstractions.allocate(backend, FT, N_quad)
                    quad_launch() = kfull!(
                        mp, tps, nothing, out,
                        inp.ρ, inp.T, inp.q_tot, inp.q_lcl, inp.n_lcl, inp.q_rai, inp.n_rai,
                        inp.q_ice, inp.n_ice, inp.q_rim, inp.b_rim, inp.logλ;
                        ndrange = N_quad,
                    )
                    t_compile = @elapsed begin
                        quad_launch()
                        KernelAbstractions.synchronize(backend)
                    end
                    tmin = time_launch(quad_launch, max(2, K - 2))
                    push_row!(rows, FT, "quadrature", mix, N_quad, tmin, t_compile, 0.0)
                    report_row(FT, "quadrature", mix, N_quad, tmin, t_compile)
                catch err
                    @error "quadrature path failed" FT mix exception=(err, catch_backtrace())
                end
            end
        end
    end

    write_results(rows, outdir)
    return rows
end

function push_row!(rows, FT, name, mix, N, tmin, t_compile, tab_MB)
    cells_per_s = N / tmin
    us_per_cell = tmin * 1e6 / N
    # Effective table-read bandwidth estimate: table footprint streamed once per
    # kernel launch is a lower bound; the achieved DRAM read rate needs ncu. The
    # reported figure is the table footprint divided by the per-launch time.
    bw_GBs = tab_MB > 0 ? (tab_MB / 1024) / tmin : 0.0
    push!(
        rows,
        (;
            FT = string(FT), path = name, mix = string(mix), N,
            us_per_cell, cells_per_s, t_compile_s = t_compile, table_MB = tab_MB, bw_GBs,
        ),
    )
end

function report_row(FT, name, mix, N, tmin, t_compile)
    @info @sprintf(
        "%-10s %-8s %-8s N=%-8d  %8.3f us/cell  %10.3e cells/s  (compile %6.2fs)",
        name, string(FT), string(mix), N, tmin * 1e6 / N, N / tmin, t_compile,
    )
end

function write_results(rows, outdir)
    csv = joinpath(outdir, "results.csv")
    open(csv, "w") do io
        println(io, "FT,path,mix,N,us_per_cell,cells_per_s,compile_s,table_MB,bw_GBs")
        for r in rows
            println(
                io,
                join(
                    (
                        r.FT, r.path, r.mix, r.N,
                        @sprintf("%.4f", r.us_per_cell), @sprintf("%.4e", r.cells_per_s),
                        @sprintf("%.3f", r.t_compile_s), @sprintf("%.3f", r.table_MB),
                        @sprintf("%.4f", r.bw_GBs),
                    ), ","),
            )
        end
    end

    md = joinpath(outdir, "results.md")
    open(md, "w") do io
        println(io, "# P3 lookup-table GPU microbenchmark results\n")
        println(io, "Backend: ", ON_GPU ? "CUDA GPU" : "KernelAbstractions CPU (smoke)", "\n")
        println(io, "| path | FT | mix | N | us/cell | cells/s | compile (s) | table (MB) | table BW (GB/s) |")
        println(io, "|:-----|:---|:----|--:|--------:|--------:|------------:|-----------:|----------------:|")
        for r in rows
            println(
                io,
                @sprintf(
                    "| %s | %s | %s | %d | %.3f | %.3e | %.2f | %.2f | %s |",
                    r.path, r.FT, r.mix, r.N, r.us_per_cell, r.cells_per_s,
                    r.t_compile_s, r.table_MB, r.bw_GBs > 0 ? @sprintf("%.2f", r.bw_GBs) : "-",
                )
            )
        end
        println(io, "\nTable BW is the table footprint divided by per-launch time (a lower bound on ")
        println(io, "streamed table bytes). The achieved DRAM read rate requires ncu/nsys profiling.")
    end
    @info "Results written" csv md
    return (; csv, md)
end

# Entry point
let
    outdir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "p3_lut_gpu_results")
    smoke = "smoke" in ARGS
    run_benchmarks(; outdir, smoke)
end

nothing
