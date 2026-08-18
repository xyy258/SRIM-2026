# ==============================================================================
# Driver script for Ekman TKE / K_T analysis
#
# Resumable, self-reporting driver designed to run within a Slurm allocation.
# Executes simulation, post-processing, and figure generation sequentially.
#
# Usage:
#     srun julia --project=$PROJECT_DIR "Ekman/TKE analysis/swirlesrun.jl"
#
# Environment Configuration (Knobs):
#   EKMAN_STAGE   : Execution target -> "sim", "post", "figures", or "auto" (default)
#   R             : N/f ratio (default: 1)
#   T_STRAT       : Thermocline depth in meters (default: 20)
#   SHARP         : Pycnocline sharpness parameter (default: 6)
#   CASE_NAME     : Custom case identifier directory (default: "r=<R>, T=<T_STRAT>")
#   SKIP_PREFLIGHT: Set to "1" to bypass environment & CUDA check
#   SKIP_FIGURES  : Set to "1" to skip rendering plots/animations
# ==============================================================================

using Printf, Dates

# ------------------------------------------------------------------------------
# Paths & Environment Setup
# ------------------------------------------------------------------------------

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, "..", ".."))     # Repository root (holds Project.toml)

const stage    = get(ENV, "EKMAN_STAGE", "auto")
const r_case   = get(ENV, "R", "1")
const T_case   = get(ENV, "T_STRAT", "20")
const sharp    = get(ENV, "SHARP", "6")
const casename = get(ENV, "CASE_NAME",
                     @sprintf("r=%.1f, T=%.1f", parse(Float64, r_case), parse(Float64, T_case)))

# Case output paths
const datadir  = joinpath(HERE, "Data", casename)
const moments  = joinpath(datadir, "Moments.jld2")
const marker   = joinpath(datadir, ".done_sim") # Marks successful simulation completion

# Global timing and logging setup
const started = time()
elapsed_h() = (time() - started) / 3600
log(msg) = (@printf("[%s | %.2f h] %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
                    elapsed_h(), msg); flush(stdout))

# Ensure local log directory exists
mkpath(joinpath(HERE, "logs"))

# Forward select environment variables to child processes only if explicitly set
passthrough(keys...) = Dict{String,String}(k => ENV[k] for k in keys if haskey(ENV, k))

const case_env = merge(Dict("R" => r_case, "T_STRAT" => T_case, "SHARP" => sharp),
                       passthrough("CASE_NAME", "CD"))

# ------------------------------------------------------------------------------
# Process Runner Utilities
# ------------------------------------------------------------------------------

"""
    run_julia(payload, env, logfile; threads="auto", label=first(payload))

Spawns a isolated Julia subprocess targeting the repository project environment.
Appends output to `logs/<logfile>` with a timestamp header block.
"""
function run_julia(payload::AbstractVector{<:AbstractString}, env::AbstractDict, logfile;
                   threads = "auto", label = first(payload))
    cmd = Cmd([joinpath(Sys.BINDIR, "julia"), "--project=$ROOT", "-t", threads,
               String.(payload)...])
    envs = Dict{String,String}(String(k) => String(v) for (k, v) in env)

    # GKSwstype=100 forces headless rendering for GR/Plots.jl in HPC environments
    full = merge(Dict{String,String}(ENV), Dict("GKSwstype" => "100"), envs)
    path = joinpath(HERE, "logs", logfile)

    # Append output so resumed job history remains intact across multiple runs
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

"""
    run_script(script, env, logfile)

Convenience wrapper to run a target Julia script file located in `HERE`.
"""
run_script(script, env::AbstractDict, logfile) =
    run_julia([joinpath(HERE, script)], env, logfile; label = script)

# ------------------------------------------------------------------------------
# Stage 0: Environment Preflight Check
# ------------------------------------------------------------------------------

# Inline check script: Instantiates env, checks GPU availability, and verifies
# that Oceananigans' closure fields export the expected SGS diffusivity (:κₑ).
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
        log("SKIP_PREFLIGHT=1 — skipping environment checks")
        return
    end

    log("preflight: verifying package dependencies, CUDA, and closure fields...")

    # Use single-threaded precompilation to prevent lock contention on shared filesystems (e.g., CephFS)
    ok = run_julia(["-e", preflight_code],
                   Dict("JULIA_NUM_PRECOMPILE_TASKS" => "1"),
                   "preflight.log"; threads = "1", label = "preflight")

    if !ok
        error("Preflight failed — check logs/preflight.log. " *
              "If Pkg/precompile failed, run `Pkg.instantiate()` on a login node.")
    end
    log("preflight OK")
end

# ------------------------------------------------------------------------------
# Stage 1: Main Simulation Execution
# ------------------------------------------------------------------------------

function do_sim()
    # Check marker file (not just data file presence) to ensure full completion
    if isfile(marker)
        log("Simulation complete for \"$casename\" — skipping. (Delete $(relpath(marker, HERE)) to re-run)")
        return true
    end

    log("Starting simulation: r = $r_case, T = $T_case m, sharp = $sharp")
    log("  output → Data/$casename/, log → logs/sim.log")
    t0 = time()

    # Disable internal plotting (PLOTS=0) so visualizer failures don't abort the simulation
    ok = run_script("Ekman3D.jl", merge(case_env, Dict("MOMENTS" => "1", "PLOTS" => "0"),
                                        passthrough("EKMAN_SMOKE", "SMOKE_ITERS")),
                    "sim.log")

    if ok && isfile(moments)
        write(marker, string(now())) # Mark successful execution
        log(@sprintf("  simulation completed successfully in %.2f h", (time() - t0) / 3600))
        return true
    end

    log("  SIMULATION FAILED — see logs/sim.log for details")
    return false
end

# ------------------------------------------------------------------------------
# Stage 2: Post-Processing & Boundary Layer Diagnostics
# ------------------------------------------------------------------------------

function do_post()
    if !isfile(moments)
        log("Missing moments file at $(relpath(moments, HERE)) — aborting post-processing")
        return
    end

    # Measure mixed layer depth (h0) first to verify if turbulence interacted with pycnocline
    log("Running measure_h0.jl (boundary layer depth at z = $T_case m)...")
    run_julia([joinpath(HERE, "measure_h0.jl")],
              merge(case_env, passthrough("TKE_FRAC", "H_PEAK")),
              "h0.log"; threads = "1", label = "measure_h0.jl")
    surface("h0.log", "h0 MEASUREMENT")

    # Calculate turbulent diffusivities (K_T)
    log("Running MixedLayerDiffusivity.jl...")
    ok = run_script("MixedLayerDiffusivity.jl",
                    merge(case_env, passthrough("GRAD_FLOOR", "SMOOTH", "SKIP_PERIODS",
                                                "RESULT_SUFFIX", "N_PHASE")),
                    "post.log")
    ok || log("  post-processing encountered errors — see logs/post.log")
    surface("post.log", "VERIFICATION")
end

"""
    surface(logfile, marker_line)

Extracts and prints key milestone results directly into stdout from child log files.
"""
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

# ------------------------------------------------------------------------------
# Stage 3: Visual Diagnostics & Plots
# ------------------------------------------------------------------------------

function do_figures()
    if get(ENV, "SKIP_FIGURES", "0") == "1"
        log("SKIP_FIGURES=1 — skipping visualization")
        return
    end
    if !isfile(moments)
        log("No diagnostic data found — skipping visualization")
        return
    end

    log("Generating figures, animations, and TKE profiles via Figures.jl...")
    run_script("Figures.jl", merge(case_env, passthrough("FIGURES")), "figures.log") ||
        log("  Figures.jl encountered errors — see logs/figures.log")
end

# ------------------------------------------------------------------------------
# Main Entry Point & Orchestration
# ------------------------------------------------------------------------------

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
        log("Aborting pipeline: simulation stage failed.")
    end
else
    error("Invalid EKMAN_STAGE \"$stage\" — supported values: sim, post, figures, auto")
end

log(@sprintf("=== Finished stage \"%s\" in %.2f h ===", stage, elapsed_h()))