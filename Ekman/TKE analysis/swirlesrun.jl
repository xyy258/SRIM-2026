# Swirles run driver for the Ekman TKE / K_T analysis.
#
# The counterpart of Stokes/3D/swirlesrun.jl, same structure and same purpose:
# one resumable, self-reporting job that a 12 h Slurm allocation can be
# re-submitted against until the work is done.
#
# Launched by swirles.sh in this folder as
#     srun julia --project=$PROJECT_DIR "Ekman/TKE analysis/swirlesrun.jl"
#
# ---------------------------------------------------------------------------
# WHAT IT RUNS: ONE CASE
# ---------------------------------------------------------------------------
#   r = N/f = 1,  softplus background,  T = 20 m,  sharp = 6
#   grid 100 × 100 × 500 over 75 × 75 × 120 m,  max_Δt = 7.5 s
#   duration 4e5 s = 6.37 inertial periods
#
# Every one of those numbers is Ekman/3D Simulation/Parameters.jl's, unchanged.
# Nothing here is a sweep: the whole point is to reproduce, for the Ekman case,
# the TKE and K_T analysis built for the tidal case.
#
# ---------------------------------------------------------------------------
# THE 12 h BUDGET
# ---------------------------------------------------------------------------
# Measured throughput for the Stokes study on the same partition:
# 434,800 iterations of a 100 × 100 × 300 grid in 2.111 h, i.e. 17.5 ms per
# iteration (logs/P4_T5_sqrtRi2.log of the archived sweep).
#
# This grid is 100 × 100 × 500 = 5.0 M cells against 3.0 M, so ~29 ms/iteration.
# The iteration COUNT is the reason this case is cheap: the Stokes runs were
# CFL-limited to Δt ≈ 3 s by a 0.0086 m wall cell, whereas this grid's first cell
# is 0.133 m and max_Δt = 7.5 s binds instead. So
#
#     4e5 s / 7.5 s  =  53,300 iterations  ×  29 ms  ≈  0.43 h
#
# Call it ~1 h with the output writers, the u* fit and GPU contention — against a
# 12 h wall. The figures add a few minutes. THIS IS AN ESTIMATE FROM A DIFFERENT
# CASE'S TIMINGS: the job prints its own wall time, and the first submission is
# what turns the estimate into a measurement.
#
# Because it is one short case there is no wall-clock guard and no staging. If it
# does not finish, raise --time or run the steps separately with EKMAN_STAGE.
#
# ---------------------------------------------------------------------------
# STAGES (EKMAN_STAGE, default "auto")
# ---------------------------------------------------------------------------
#   sim      the simulation only
#   post     MixedLayerDiffusivity.jl + measure_h0.jl, from data on disk
#   figures  Ekman_anim.jl + Ekman_plot.jl + TKE.jl, from data on disk
#   auto     sim → post → figures
#
# RESUMABLE: the simulation writes outputs into Data/<case>/ and drops a marker
# when it completes, so a re-submission skips straight to post-processing. Delete
# the marker to force a re-run.
#
# Knobs: EKMAN_STAGE, R, T_STRAT, SHARP, GRAD_FLOOR, SMOOTH, SKIP_PERIODS,
#        TKE_FRAC, H_PEAK, CD, SKIP_PREFLIGHT, SKIP_FIGURES, EKMAN_SMOKE.

using Printf, Dates

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, "..", ".."))     # repo root: holds Project.toml

const stage    = get(ENV, "EKMAN_STAGE", "auto")
const r_case   = get(ENV, "R", "1")
const T_case   = get(ENV, "T_STRAT", "20")
const sharp    = get(ENV, "SHARP", "6")
const casename = get(ENV, "CASE_NAME",
                     @sprintf("r=%.1f, T=%.1f", parse(Float64, r_case), parse(Float64, T_case)))

const datadir  = joinpath(HERE, "Data", casename)
const moments  = joinpath(datadir, "Moments.jld2")
const marker   = joinpath(datadir, ".done_sim")

const started = time()
elapsed_h() = (time() - started) / 3600
log(msg) = (@printf("[%s | %.2f h] %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
                    elapsed_h(), msg); flush(stdout))

mkpath(joinpath(HERE, "logs"))

# One child process per step. Parameters.jl declares its values `const` (GPU
# kernels capture them), so a process can only ever set up one case.
#
# `dir = ROOT` matters: Ekman3D.jl anchors its own paths to @__DIR__, but the
# Julia project is the repo root's Project.toml, and Pkg.instantiate() inside the
# script resolves against the active project. GKSwstype keeps GR headless.
function run_julia(payload::AbstractVector{<:AbstractString}, env::AbstractDict, logfile;
                   threads = "auto", label = first(payload))
    cmd = Cmd([joinpath(Sys.BINDIR, "julia"), "--project=$ROOT", "-t", threads,
               String.(payload)...])
    envs = Dict{String,String}(String(k) => String(v) for (k, v) in env)
    full = merge(Dict{String,String}(ENV), Dict("GKSwstype" => "100"), envs)
    path = joinpath(HERE, "logs", logfile)

    # APPEND, never truncate: a resumed job appends to its own log, and nothing a
    # child prints carries a wall-clock date, so this banner is what separates the
    # attempts. `grep -n '^==== ' logs/<file>` lists them, newest last.
    open(path, "a") do io
        println(io, "\n", "="^78)
        println(io, "==== ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "  ", label)
        println(io, "==== ", join(("$k=$v" for (k, v) in sort(collect(envs))), "  "))
        println(io, "="^78)
        flush(io)
    end
    return success(pipeline(setenv(cmd, full; dir = ROOT),
                            stdout = path, stderr = path, append = true))
end

run_script(script, env::AbstractDict, logfile) =
    run_julia([joinpath(HERE, script)], env, logfile; label = script)

# Knobs forwarded only when actually set, so an unset one keeps the script's own
# default rather than being overridden with an empty string.
passthrough(keys...) = Dict{String,String}(k => ENV[k] for k in keys if haskey(ENV, k))
const case_env = merge(Dict("R" => r_case, "T_STRAT" => T_case, "SHARP" => sharp),
                       passthrough("CASE_NAME", "CD"))

# ---------------- 0. Preflight ----------------
# Checks what this job needs BEFORE spending an hour on it: the environment
# instantiates, CUDA works, and — the thing most likely to break on a version
# bump — the SGS diffusivity is where Moments.jl looks for it. Serial
# precompilation, because parallel precompilation contends on pidfile locks over
# cephfs.
#
# NOTE the closure order: this case builds (ScalarDiffusivity, AMD), the opposite
# of the Stokes case, which is exactly why Moments.jl searches instead of indexing.
const preflight_code = """
using Pkg
Pkg.instantiate()
Pkg.precompile()
using Oceananigans, CUDA, JLD2, Plots
println("preflight: packages loaded, Oceananigans ", pkgversion(Oceananigans))

m = NonhydrostaticModel(RectilinearGrid(CPU(); size = (2, 2, 2), extent = (1, 1, 1));
                        tracers = :b,
                        closure = (ScalarDiffusivity(ν = 1e-6, κ = 1e-7),
                                   AnisotropicMinimumDissipation()))
prop = hasproperty(m, :closure_fields) ? :closure_fields :
       hasproperty(m, :diffusivity_fields) ? :diffusivity_fields : nothing
prop === nothing && (println("preflight: NO closure-field container on the model"); exit(1))
cf = getproperty(m, prop)
entries = cf isa Tuple ? collect(cf) : Any[cf]
for (i, c) in enumerate(entries)
    println("preflight: model.\$prop[\$i] = ", c === nothing ? "nothing" : propertynames(c))
end
any(c -> c !== nothing && :κₑ in propertynames(c), entries) ||
    println("preflight: WARNING — κₑ not found; Moments.jl will fall back or error")

if CUDA.functional()
    println("preflight: CUDA functional — ", CUDA.name(CUDA.device()))
else
    println("preflight: CUDA NOT functional — the simulation cannot run")
    exit(1)
end
"""

function preflight()
    if get(ENV, "SKIP_PREFLIGHT", "0") == "1"
        log("SKIP_PREFLIGHT=1 — not checking the environment")
        return
    end
    log("preflight: instantiate + precompile + CUDA + SGS access path, see logs/preflight.log")
    ok = run_julia(["-e", preflight_code],
                   Dict("JULIA_NUM_PRECOMPILE_TASKS" => "1"),
                   "preflight.log"; threads = "1", label = "preflight")
    ok || error("Preflight failed — see logs/preflight.log. If it is a Pkg or precompile " *
                "error, run `JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=$ROOT -e " *
                "'using Pkg; Pkg.instantiate(); Pkg.precompile()'` once on a login node. " *
                "Nothing has been run yet.")
    log("preflight OK")
end

# ---------------- 1. Simulation ----------------
function do_sim()
    if isfile(marker)
        log("simulation already complete for \"$casename\" — skipping (delete $(relpath(marker, HERE)) to redo)")
        return true
    end
    log("simulation: r = $r_case, T = $T_case m, sharp = $sharp, 6.37 inertial periods (~1 h expected)")
    log("  output → Data/$casename/, log → logs/sim.log")
    t0 = time()
    # PLOTS=0: the figures are their own step, so a failure there cannot cost the
    # simulation, and they can be redrawn without re-running it.
    ok = run_script("Ekman3D.jl", merge(case_env, Dict("MOMENTS" => "1", "PLOTS" => "0"),
                                        passthrough("EKMAN_SMOKE", "SMOKE_ITERS")),
                    "sim.log")
    if ok && isfile(moments)
        # The marker, not the moments file, is the completion test: a run cut short
        # leaves a valid but truncated file behind.
        write(marker, string(now()))
        log(@sprintf("  simulation done in %.2f h", (time() - t0) / 3600))
        return true
    end
    log("  SIMULATION FAILED (or wrote no moments file) — see logs/sim.log")
    return false
end

# ---------------- 2. Post-processing ----------------
function do_post()
    if !isfile(moments)
        log("no Moments.jld2 at $(relpath(moments, HERE)) — nothing to post-process")
        return
    end

    # h0 FIRST. It answers whether the turbulent layer ever reached the pycnocline
    # at z = T, and if it did not, the VERIFICATION block below is measuring noise.
    log("measure_h0.jl — did the turbulent layer reach the pycnocline at z = $T_case m?")
    run_julia([joinpath(HERE, "measure_h0.jl")],
              merge(case_env, passthrough("TKE_FRAC", "H_PEAK")),
              "h0.log"; threads = "1", label = "measure_h0.jl")
    surface("h0.log", "h0 MEASUREMENT")

    log("MixedLayerDiffusivity.jl → Data/$casename/mixing_*.jld2 and Plots/K_T_*.png")
    ok = run_script("MixedLayerDiffusivity.jl",
                    merge(case_env, passthrough("GRAD_FLOOR", "SMOOTH", "SKIP_PERIODS",
                                                "RESULT_SUFFIX", "N_PHASE")),
                    "post.log")
    ok || log("  post-processing returned an error — see logs/post.log")
    surface("post.log", "VERIFICATION")
end

# The h0 report and the VERIFICATION block are the deliverables, so lift them into
# the job log rather than leaving them buried in a child log nobody opens.
function surface(logfile, marker_line)
    path = joinpath(HERE, "logs", logfile)
    isfile(path) || return
    lines = readlines(path)
    i = findlast(l -> startswith(l, marker_line), lines)
    i === nothing && return
    println()
    foreach(println, lines[max(1, i - 2):end])
    flush(stdout)
end

# ---------------- 3. Figures ----------------
function do_figures()
    get(ENV, "SKIP_FIGURES", "0") == "1" && (log("SKIP_FIGURES=1 — no figures"); return)
    isfile(moments) || (log("no data yet — no figures"); return)
    log("Figures.jl — animations, averaged profiles, and the slice-based TKE comparison")
    run_script("Figures.jl", merge(case_env, passthrough("FIGURES")), "figures.log") ||
        log("  Figures.jl returned an error — see logs/figures.log")
end

# ---------------- Drive ----------------
log("=== Ekman TKE run: stage \"$stage\", case \"$casename\" ===")
preflight()

if stage == "sim"
    do_sim()
elseif stage == "post"
    do_post()
elseif stage == "figures"
    do_figures()
elseif stage == "auto"
    if do_sim()
        do_post()
        do_figures()
    else
        log("stopping: the simulation did not complete, so there is nothing to analyse")
    end
else
    error("Unknown EKMAN_STAGE \"$stage\" — use sim, post, figures or auto")
end

log(@sprintf("=== %s done in %.2f h ===", stage, elapsed_h()))
