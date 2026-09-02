#!/usr/bin/env julia
# ekmanrun.jl — the Ekman T = 10 m column, re-run with the subgrid buoyancy flux
# saved, so that K_T on the Ekman side is built the same way as on the Stokes side.
#
# Launched by Combined/swirles.sh as
#     srun julia --project=$PROJECT_DIR Combined/ekmanrun.jl
#
# ---------------- Why this run exists ----------------
# The combined figure (figures/l_vs_q_over_N_ath_T10_combined.png) puts the Ekman
# T = 10 column on the Stokes l-vs-√TKE/N axes. The two sides are not built the
# same way, because "Ekman 3D.jl" has its diffusivity_fields writer commented out
# (line 253) and therefore saves no subgrid buoyancy flux. So
#
#     Stokes:   K_T = −(⟨w′b′⟩ + F_sgs) / ⟨∂b/∂z⟩
#     Ekman:    K_T = −⟨w′b′⟩          / ⟨∂b/∂z⟩       (resolved only)
#
# and the size of that gap, read off the Stokes runs at z = h, is
#
#     N/ω          1     2     5    10    25    50
#     K_sgs/K_T  0.03  0.03  0.06  0.10  0.30  0.59
#
# Negligible where the stratification is weak, a factor of two at the strong end —
# and the strong end is exactly where the two columns are being compared. The
# present figure works around this by ALSO plotting a resolved-only Stokes
# comparator (the open squares), which is honest but throws away the Stokes
# subgrid flux rather than measuring the Ekman one.
#
# This cannot be fixed offline. AMD sets κₑ from the full three-dimensional
# velocity and buoyancy gradients, and Velocity.jld2/Buoyancy.jld2 hold one y
# slice, so there is nothing on disk to reconstruct it from. The column has to be
# re-run.
#
# ---------------- What is different from "Ekman 3D.jl" ----------------
# The physics is identical: same grid, same stretching, same quadratic-drag
# bottom, same sponge, same AMD + molecular closure, same softplus profile at
# sharp = 6, same duration. Parameters.jl is INCLUDED from the Ekman folder
# rather than copied, so the two cannot drift apart.
#
# What changes is the output. Instead of turning the commented-out
# diffusivity_fields writer back on — which writes κₑ and νₑ as raw 3D fields,
# hundreds of GB, and still leaves the correlation ⟨κₑ ∂b/∂z⟩ to be guessed at
# from a y slice — this writes the same thirteen plane-averaged profiles the
# Stokes runs write from Moments.jl:
#
#     means:   U, V, W, B, dBdz          (W is the ⟨w⟩_xy ≈ 0 check)
#     moments: uu, vv, ww                (Centers)
#              uw, vw, wb                (Faces)
#     subgrid: kappa_sgs, F_sgs          (Faces)
#
# from which the post-processing forms
#
#     TKE = ½(⟨uu⟩ − U² + ⟨vv⟩ − V² + ⟨ww⟩)
#     F_b = ⟨wb⟩ + F_sgs
#     K_T = −F_b / dBdz
#
# exactly as MixedLayerDiffusivity.jl does. F_sgs is written as the average of
# the product, −⟨(κₑ + κ₀) ∂b/∂z⟩, not the product of the averages: AMD responds
# to local strain, which peaks where the gradient is sharp, so the two are
# correlated and the two orderings differ.
#
# That also closes the SECOND asymmetry, which is not about the subgrid flux at
# all. Velocity.jld2 is written with indices = (:, 1, :), so the reduction in
# reduce_ekman_T10.jl averages over 100 x points at one y. These are full-plane
# averages over all 10 000 columns, the same Reynolds average the Stokes side
# takes. Both sides then differ only in the forcing, which is the point.
#
# Three smaller differences, all deliberate:
#   * output goes to Data/Ekman_moments/, NOT Data/Ekman/ — the existing runs are
#     not touched, and neither is anything under Ekman/;
#   * the u* fit writes its parameters file into Combined/logs/, not into
#     Ekman/3D Simulation/Parameters/;
#   * Ekman_anim.jl and Ekman_plot.jl are not run — they are per-case figures
#     from the slice files, and the slices are off by default here.
#
# The random kick is seeded per case (SEED, default 20260902 + 100r), so this run
# is reproducible. It is NOT the same realisation as the runs already on disk,
# which were unseeded: expect the same statistics, not the same numbers.
#
# ---------------- The budget ----------------
# The whole column is about 3 h on the ampere partition, so it fits one 12 h wall
# with room to spare and needs no staging, no restarts and no array.
#
# That number comes from the Stokes column's own cluster logs rather than from a
# benchmark here, because it is the same code on the same partition:
#
#     logs/P4_T10_sqrtRi{0,2,10,50}.log   403 000 steps in 1.93-1.96 h
#                                         = 0.0173 s/step on 100x100x300
#                                         = 0.0058 s per step per Mcell
#
# This grid is 100x100x500 = 5.0 Mcell, and the time step pins at max_Δt = 8 s
# with the advective CFL sitting at 0.8 — the ceiling sets the step, not the CFL,
# so N does not change it and 40e4 s is 50 000 steps flat. Hence
#
#     50 000 x 0.0058 x 5.0 = 1450 s = 0.40 h per case, 2.8 h for the column.
#
# CASE_HOURS defaults to 1.0 h, which is 2.5x that, because scaling by cell count
# is only approximate: the pressure solve is an FFT in z, so going from 300 to
# 500 points is slightly worse than linear. The first case to finish replaces the
# estimate with what the machine actually did.
#
# Do NOT calibrate this against a desktop GPU. Benchmarked here on a shared RTX
# 4000 Ada the same case runs at 0.21-0.43 s/step depending on who else is on the
# card and how hot it is — a factor of two spread, and about 10x the cluster
# rate. An earlier version of this header extrapolated from that machine and put
# a case at 2.85 h, which is why the number above is anchored on the cluster logs
# instead.
#
# The wall-clock guard and the per-case completion markers are kept even though
# nothing is expected to hit them: if a case does overrun, the job declines the
# next one rather than starting something it cannot finish, and re-submitting
# swirles.sh continues rather than restarts.
#
# RATIOS is ordered STRONGEST FIRST — 50, 25, 10 before 5, 2, 1, 0.5 — so that a
# job which does get cut short returns the cases where the subgrid share is large
# enough to move the points, rather than the weakly stratified ones whose
# resolved-only numbers are already within a few per cent.
#
# Output costs essentially nothing in time. Benchmarked A/B on the same card
# minutes apart, all thirteen moments at 200 s plus the four Avg_* writers came
# to 0.4320 s/step against 0.4294 s/step for moments alone at 314 s — 0.6 %, i.e.
# inside the noise. So the writers are configured for what is wanted, not for
# speed: the Avg_* files stay on by default so the existing Ekman analysis
# scripts run unchanged on this output, even though every profile in them is
# duplicated in the moments file.
#
# Disk, measured: 57 MB of moments a case (2001 samples x 28 kB) plus 54 MB of
# Avg_*, so about 110 MB a case and 0.8 GB for the column. The x-z slices are off
# by default (SLICES=1 turns them on, at 1.4 GB a case) since they are already on
# disk from the existing runs and nothing here reads them.
#
# ---------------- Stages (SWEEP_STAGE, default "auto") ----------------
#   preflight  environment + CUDA + the SGS access path, no GPU work
#   cases      the column only
#   check      re-report the health numbers from the moments files on disk
#   auto       preflight -> cases -> check
#
# ENV: SWEEP_STAGE, RATIOS, T_STRAT, DURATION, SHARP, ARCH, SEED, SLICES,
#      AVG_WRITERS, MOMENT_INTERVAL, OUT_ROOT, GRID_TAG, WALL_HOURS, CASE_HOURS,
#      SKIP_PREFLIGHT, DRY_RUN.
#
# ---------------- One case per job, if it is ever wanted ----------------
# The seven cases share nothing — no spin-up, no restart file, no ordering — so
# RATIOS=<r> restricts a job to a single case and the column can be run as a
# seven-task Slurm array instead (swirles.sh does this when given --array=0-6).
# At 0.4 h a case that is not needed to fit the wall; it is only worth it to turn
# a 3 h column into a 25 min one, and it costs seven queue waits to do so. The
# markers make the two modes interchangeable either way.
#
# DRY_RUN=1 prints the plan — which cases are complete, which would run, what
# they are estimated to cost — and exits without touching the GPU. Worth doing on
# a login node before submitting:
#
#     cd /cephfs/store/damtp/tll46/SRIM-2026
#     DRY_RUN=1 julia --project=. Combined/ekmanrun.jl
#
# ---------------- Layout of this file ----------------
# The driver comes first and ends in `exit()`, so in sweep mode the simulation
# section below it is never even parsed. That ordering is not cosmetic: Julia
# expands macros in both branches of a top-level `if`, so `@at` in the simulation
# section would be looked up before `using Oceananigans` had run.

using Printf, Dates

const HERE  = @__DIR__
const REPO  = dirname(HERE)
const EKMAN = joinpath(REPO, "Ekman", "3D Simulation")

const CASE_MODE = !isempty(ARGS) && ARGS[1] == "case"

const T_STRAT  = parse(Float64, get(ENV, "T_STRAT", "10"))
const GRID_TAG = get(ENV, "GRID_TAG", "100x100x500_drag")
const OUT_ROOT = get(ENV, "OUT_ROOT", joinpath(HERE, "Data", "Ekman_moments", "4"))

# Case folders keep "Ekman 3D.jl"'s own naming, so that Data/Ekman_moments/4 is a
# drop-in sibling of Data/Ekman/4 and reduce_ekman_T10.jl can be pointed at
# either by changing one path.
case_dir(r)   = joinpath(OUT_ROOT, @sprintf("r=%.1f, T=%.1f", r, T_STRAT))
moments_of(r) = joinpath(case_dir(r), "Moments.jld2")
marker_of(r)  = joinpath(case_dir(r), ".done_moments_$GRID_TAG")
case_tag(r)   = @sprintf("P4_T%g_r%g", T_STRAT, r)

# =============================================================================
#  SWEEP MODE — the driver. One child process per case, then exit.
# =============================================================================
if !CASE_MODE

const stage      = get(ENV, "SWEEP_STAGE", "auto")
const ratios     = [parse(Float64, s) for s in split(get(ENV, "RATIOS", "50 25 10 5 2 1 0.5"))]
const wall_hours = parse(Float64, get(ENV, "WALL_HOURS", "11.5"))
const dry_run    = get(ENV, "DRY_RUN", "0") == "1"
const PROJECT    = dirname(Base.active_project())

const started = time()
elapsed_h() = (time() - started) / 3600
log(msg) = (@printf("[%s | %.2f h] %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
                    elapsed_h(), msg); flush(stdout))

mkpath(joinpath(HERE, "logs"))
mkpath(OUT_ROOT)

# Children inherit whatever project the driver was started in — on the cluster
# that is the repository root, the environment the existing Ekman runs used.
# `dir = REPO` matches the convention that "Ekman 3D.jl" is run from the repo
# root; GKSwstype keeps anything that pulls in Plots headless on a compute node.
function run_julia(payload, env, logfile; label = "julia")
    cmd  = Cmd([joinpath(Sys.BINDIR, "julia"), "--project=$PROJECT", "-t", "auto", payload...])
    envs = Dict{String,String}(String(k) => String(v) for (k, v) in env)
    full = merge(Dict{String,String}(ENV), Dict("GKSwstype" => "100"), envs)
    path = joinpath(HERE, "logs", logfile)

    # Append, not truncate: a resumed job adds to its own log and this banner is
    # what separates the attempts. List them with `grep -n '^==== ' <log>`.
    open(path, "a") do io
        println(io, "\n", "="^78)
        println(io, "==== ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "  ", label)
        isempty(envs) || println(io, "==== ", join(("$k=$v" for (k, v) in sort(collect(envs))), "  "))
        println(io, "="^78)
    end
    return success(pipeline(setenv(cmd, full; dir = REPO),
                            stdout = path, stderr = path, append = true))
end

# ---------------- 0. Preflight ----------------
# Fail in the first two minutes rather than three hours in. The SGS access path
# is the one thing here most likely to break on a version bump, and every case
# needs it.
const preflight_code = """
using Pkg; Pkg.instantiate(); Pkg.precompile()
using Oceananigans, CUDA, JLD2
println("preflight: Oceananigans ", pkgversion(Oceananigans))
m = NonhydrostaticModel(RectilinearGrid(CPU(); size = (2, 2, 2), extent = (1, 1, 1));
                        tracers = :b,
                        closure = (ScalarDiffusivity(ν = 1e-6, κ = 1e-7),
                                   AnisotropicMinimumDissipation()))
prop = hasproperty(m, :closure_fields) ? :closure_fields :
       hasproperty(m, :diffusivity_fields) ? :diffusivity_fields : nothing
prop === nothing && (println("preflight: NO closure-field container on the model"); exit(1))
cf = getproperty(m, prop)
found = false
for (i, c) in enumerate(cf isa Tuple ? collect(cf) : Any[cf])
    c isa NamedTuple || continue
    println("preflight: model.\$prop[\$i] has ", propertynames(c))
    :κₑ in propertynames(c) && (global found = true)
end
found || println("preflight: WARNING — κₑ not found by name; the fallback search will be used")
BulkDrag(coefficient = 1e-3)
println("preflight: BulkDrag constructs")
if CUDA.functional()
    println("preflight: CUDA functional — ", CUDA.name(CUDA.device()))
else
    println("preflight: CUDA NOT functional — the simulations cannot run"); exit(1)
end
"""

function preflight()
    get(ENV, "SKIP_PREFLIGHT", "0") == "1" && (log("SKIP_PREFLIGHT=1 — not checking the environment"); return)
    log("preflight: instantiate + precompile + CUDA + SGS access path → logs/preflight_ekman.log")
    run_julia(["-e", preflight_code], Dict("JULIA_NUM_PRECOMPILE_TASKS" => "1"),
              "preflight_ekman.log"; label = "preflight") ||
        error("Preflight failed — see Combined/logs/preflight_ekman.log. If it is a Pkg " *
              "or precompile error, run `JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. " *
              "-e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'` once on a login node. " *
              "Nothing has been run yet.")
    log("preflight OK")
end

# ---------------- 1. Cases ----------------
# The estimate starts generous and is replaced by a measurement after the first
# case finishes: starting a case that cannot finish wastes all of it, declining
# one costs only a re-submission.
case_estimate_h = parse(Float64, get(ENV, "CASE_HOURS", "1.0"))
done_cases, failed_cases, skipped_cases = String[], String[], String[]

function run_case(rv)
    tag = case_tag(rv)
    if isfile(marker_of(rv))
        log("  $tag already complete on this grid+BC — skipping")
        tag in done_cases || push!(done_cases, tag)
        return true
    end
    if elapsed_h() + case_estimate_h > wall_hours
        log(@sprintf("  %s not started — %.2f h left of the %.1f h budget, a case needs ~%.1f h",
                     tag, wall_hours - elapsed_h(), wall_hours, case_estimate_h))
        push!(skipped_cases, tag)
        return false
    end

    log(@sprintf("run %s  (T = %g m, N/f = %g, N = %.1e s⁻¹, Ri = %g)",
                 tag, T_STRAT, rv, rv * 1e-4, rv^2))
    t0 = time()
    ok = run_julia([@__FILE__, "case", string(rv)], Dict{String,String}(),
                   "$tag.log"; label = "$tag  (r = $rv, T = $T_STRAT)")

    if ok && isfile(moments_of(rv))
        # The marker, not the file, is the test for completion: a run cut short
        # by the wall leaves a valid but truncated Moments.jld2 behind.
        write(marker_of(rv), string(now()))
        push!(done_cases, tag)
        global case_estimate_h = 1.1 * (time() - t0) / 3600
        log(@sprintf("  %s done in %.2f h — estimate for the rest is now %.2f h",
                     tag, (time() - t0) / 3600, case_estimate_h))
        return true
    end
    push!(failed_cases, tag)
    log("  $tag FAILED — see Combined/logs/$tag.log")
    return false
end

# ---------------- 2. Check ----------------
# Reads each finished moments file back and reports the three numbers that say
# whether it is usable, so a bad column is caught on the cluster rather than
# after 740 MB has been pulled back. Cheap: plane-averaged profiles, not fields,
# and only the last quarter of the record.
const check_code = raw"""
using JLD2, Statistics, Printf
f  = ARGS[1]; r = parse(Float64, ARGS[2]); f₀ = 1e-4; N² = (r * f₀)^2
io = jldopen(f, "r")
its = sort(parse.(Int, keys(io["timeseries/t"])))
ts  = [io["timeseries/t/$i"] for i in its]
grab(v, i) = io["timeseries/$v/$i"][1, 1, :]
sel = its[max(1, ceil(Int, 0.75 * length(its))):end]

wmax = maximum(maximum(abs, grab("W", i)) for i in sel)
kmin = minimum(minimum(grab("kappa_sgs", i)) for i in sel)

shares = Float64[]
for i in sel
    G, wb, Fs = grab("dBdz", i), grab("wb", i), grab("F_sgs", i)
    for k in eachindex(G)
        G[k] > 0.05 * N² || continue
        Fb = wb[k] + Fs[k]
        abs(Fb) > 0 && push!(shares, abs(Fs[k]) / abs(Fb))
    end
end
close(io)

@printf("  samples %d over %.0f-%.0f s (%.2f-%.2f T_f)\n", length(its), first(ts), last(ts),
        first(ts) / (2π / f₀), last(ts) / (2π / f₀))
@printf("  max|<w>_xy| = %.2e m/s (= %.1e U∞)   %s\n", wmax, wmax / 0.04,
        wmax / 0.04 > 1e-10 ? "SUSPECT" : "ok")
@printf("  min kappa_sgs = %.3e m²/s            %s\n", kmin,
        kmin < 0 ? "NEGATIVE — mask it in post-processing" : "ok")
if isempty(shares)
    println("  |F_sgs|/|F_b|: no cells above the gradient floor")
else
    @printf("  |F_sgs|/|F_b| over the stratified cells: median %.3f, 90th pct %.3f  (n = %d)\n",
            median(shares), quantile(shares, 0.9), length(shares))
end
"""

function check()
    have = [rv for rv in sort(ratios) if isfile(moments_of(rv))]
    isempty(have) && (log("no moments files to check yet"); return)
    log("checking $(length(have)) moments file(s) — ⟨w⟩_xy, κ_sgs sign, subgrid share")
    jl = joinpath(Sys.BINDIR, "julia")
    for rv in have
        println("\n", case_tag(rv), "   ", relpath(moments_of(rv), REPO)); flush(stdout)
        run(ignorestatus(`$jl --project=$PROJECT -e $check_code $(moments_of(rv)) $rv`))
    end
    println()
    log("check done. The subgrid share is what this run exists to measure: on the " *
        "Stokes side it rises from 0.03 at N/ω = 1 to 0.59 at N/ω = 50.")
end

# ---------------- The plan ----------------
function show_plan()
    todo = [rv for rv in ratios if !isfile(marker_of(rv))]
    println("\n", "#"^78)
    @printf("#  Ekman N/f column at T = %g m, grid+BC tag %s\n", T_STRAT, GRID_TAG)
    @printf("#  N/f ∈ {%s},  run order {%s}  (strongest first — see the header)\n",
            join(sort(ratios), ", "), join(ratios, ", "))
    @printf("#  output → %s\n", relpath(OUT_ROOT, REPO))
    println("#")
    println("#  case              N/f      Ri   status")
    for rv in sort(ratios)
        st = isfile(marker_of(rv))  ? "complete — will be skipped" :
             isfile(moments_of(rv)) ? "moments on disk but NO marker — will be re-run" : "TO RUN"
        @printf("#  %-16s %5g %7g   %s\n", case_tag(rv), rv, rv^2, st)
    end
    println("#")
    @printf("#  %d to run x ~%.1f h = %.1f h in a %.1f h budget\n",
            length(todo), case_estimate_h, length(todo) * case_estimate_h, wall_hours)
    if length(todo) * case_estimate_h > wall_hours
        n_fit = max(0, floor(Int, wall_hours / case_estimate_h))
        @printf("#  THAT DOES NOT FIT: about %d of the %d will run and the rest will be\n",
                n_fit, length(todo))
        println("#  declined by the wall-clock guard. Every case carries its own marker, so")
        println("#  re-submitting swirles.sh continues rather than restarts. Nothing is lost.")
        @printf("#  Expect roughly %d submissions in all.\n",
                ceil(Int, length(todo) * case_estimate_h / wall_hours))
    end
    println("#  Per case: 0.40 h expected (50 000 steps at 8 s, scaled from the Stokes")
    println("#  column's 0.0058 s/step/Mcell on this partition); CASE_HOURS holds 2.5x")
    println("#  that as margin. Replaced by the measurement once the first case finishes.")
    println("#"^78, "\n")
    flush(stdout)
end

log("=== Ekman T = $T_STRAT column with F_sgs, stage \"$stage\", grid+BC tag $GRID_TAG ===")
show_plan()

if dry_run
    log("DRY_RUN=1 — the plan above is all this does. Nothing has run.")
elseif stage == "preflight"
    preflight()
elseif stage == "check"
    check()
elseif stage in ("cases", "auto")
    preflight()
    for rv in ratios; run_case(rv); end
    stage == "auto" && check()
else
    error("Unknown SWEEP_STAGE \"$stage\" — use preflight, cases, check or auto")
end

log(@sprintf("=== %s: %d complete, %d failed, %d not started ===",
             stage, length(done_cases), length(failed_cases), length(skipped_cases)))
isempty(failed_cases)  || log("  failed:      " * join(failed_cases, ", "))
isempty(skipped_cases) || log("  not started: " * join(skipped_cases, ", ") *
                              "  — re-submit swirles.sh to continue")

# Non-zero if any case failed, so Slurm records the job as failed. Cases the
# wall-clock guard declined are not failures — they are what the next submission
# is for.
exit(isempty(failed_cases) ? 0 : 1)

end # sweep mode

# =============================================================================
#  CASE MODE — one simulation, one process.
#
#  Parameters.jl declares almost everything `const` and GPU kernels capture those
#  globals, so a process can only ever set up one case; the driver above spawns
#  one child per r rather than looping in here.
# =============================================================================

using Oceananigans, JLD2, CUDA, Statistics, Random
using Oceananigans.AbstractOperations: ∂z, @at

# --- parameters -------------------------------------------------------------
# r, profile and T are the three values Parameters.jl leaves assignable; setting
# them first makes its `if !@isdefined` guards fall through to these.
r       = parse(Float64, ARGS[2])
profile = 4
T       = T_STRAT
include(joinpath(EKMAN, "Parameters.jl"))

# SHARP is exposed only so a mismatch with the Stokes side can be tested; leave
# it alone to reproduce the runs already on disk.
sharp = parse(Float64, get(ENV, "SHARP", string(sharp)))

const run_duration = parse(Float64, get(ENV, "DURATION", string(duration)))
const moment_dt    = parse(Float64, get(ENV, "MOMENT_INTERVAL", "314"))
const want_slices  = get(ENV, "SLICES", "0") == "1"
const want_avg     = get(ENV, "AVG_WRITERS", "1") == "1"

Random.seed!(parse(Int, get(ENV, "SEED", string(20260902 + round(Int, 100r)))))

arch = get(ENV, "ARCH", "gpu") == "cpu" ? CPU() :
       (CUDA.functional() ? GPU() :
        error("ekmanrun.jl: ARCH=gpu but CUDA is not functional on this node."))

# --- grid: verbatim from "Ekman 3D.jl" --------------------------------------
refinement = 1.8            # controls spacing near surface (higher means finer)
stretching = 10             # controls rate of stretching at bottom
h(k) = (Nz + 1 - k) / Nz
ζ(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))
z_faces(k) = -H * (ζ(k) * Σ(k) - 1)

grid = RectilinearGrid(arch;
    topology = (Periodic, Periodic, Bounded),
    size     = (Nx, Ny, Nz),
    x = (0, Lx), y = (0, Ly), z = z_faces)

z₁ = abs(Array(znodes(grid, Center()))[1])      # closest grid centre to the bottom
cᴰ = (κ / log(z₁ / z₀))^2                       # drag coefficient (κ = von Karman)

# --- boundary and initial conditions ----------------------------------------
u_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = cᴰ))
v_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = cᴰ))
b_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(0))

uᵢ(x, y, z) = U∞ + kick * randn()
vᵢ(x, y, z) = kick * randn()
wᵢ(x, y, z) = kick * randn()

scale = (r == 0 || isnothing(r)) ? 1 : N²
bᵢ(x, y, z) = (scale / sharp) * log(1 + exp(sharp * (z - T)))   # profile 4, softplus

# --- forcing and sponge ------------------------------------------------------
v_forcing_fn(x, y, z, t, p) = p.f * p.s          # balances the initial geostrophy
v_forcing = Forcing(v_forcing_fn, parameters = (s = U∞, f = f₀))

sponge_rate = r * f₀
sponge_mask = mask == 0 ? PiecewiseLinearMask{:z}(center = H, width = S) :
                          GaussianMask{:z}(center = H, width = 0.85S)

u_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask, target = U∞)
v_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask)
w_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask)

b_target = Field{Center, Center, Center}(grid)
set!(b_target, bᵢ)
b_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask, target = b_target)

# --- model -------------------------------------------------------------------
model = NonhydrostaticModel(grid;
    advection   = Centered(order = 4),
    timestepper = :RungeKutta3,
    tracers     = :b,
    buoyancy    = (r == 0 || isnothing(r)) ? nothing : BuoyancyTracer(),
    closure     = (ScalarDiffusivity(ν = ν₀, κ = κ₀), AnisotropicMinimumDissipation()),
    boundary_conditions = (u = u_bcs, v = v_bcs, b = b_bcs),
    coriolis    = FPlane(f = f₀),
    forcing     = (u = u_sponge, v = (v_forcing, v_sponge), w = w_sponge, b = b_sponge))

set!(model, u = uᵢ, v = vᵢ, w = wᵢ, b = bᵢ)

@printf("""
ekmanrun.jl case %s
  architecture      %s
  dimensions        %.1f m x %.1f m x %.1f m  (+ %.0f m sponge, H = %.0f m)
  grid              %d x %d x %d   (first centre at z = %.4f m)
  U∞  = %.4f   f₀ = %.2e   r = N/f = %.1f   N² = %.3e
  profile 4 (softplus), T = %.1f m, sharp = %.1f
  cᴰ  = %.5f   ν₀ = %.1e   κ₀ = %.1e   Pr = %.0f
  duration %.3e s = %.2f inertial periods, max Δt = %.0f s
  moments every %.0f s, x-z slices %s, Avg_* writers %s
""", case_tag(r), arch isa GPU ? "GPU" : "CPU", Lx, Ly, Lz, S, H, Nx, Ny, Nz, z₁,
     U∞, f₀, r, N², T, sharp, cᴰ, ν₀, κ₀, Pr,
     run_duration, run_duration / (2π / f₀), max_Δt, moment_dt,
     want_slices ? "ON" : "off", want_avg ? "ON" : "off")
flush(stdout)

simulation = Simulation(model, Δt = 0.1 * max_Δt, stop_time = run_duration)
simulation.callbacks[:wizard] =
    Callback(TimeStepWizard(cfl = 0.9, max_change = 1.2, max_Δt = max_Δt),
             IterationInterval(5))

start_time = time_ns()
progress(sim) = @printf("i: % 7d, sim time: % 9.0f (%.2f T_f), wall: % 10s, Δt: % 6.2f, CFL: %.2e\n",
    sim.model.clock.iteration, sim.model.clock.time,
    sim.model.clock.time / (2π / f₀),
    prettytime(1e-9 * (time_ns() - start_time)),
    sim.Δt, AdvectiveCFL(sim.Δt)(sim.model))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# --- output ------------------------------------------------------------------
u, v, w = model.velocities
b = model.tracers.b

const OUTDIR = case_dir(r)
mkpath(OUTDIR)
fn(name) = joinpath(OUTDIR, name)

# The colleague's own writers, kept so his analysis scripts run unchanged on this
# output. The four Avg_* files are 36 MB together and Avg_vel is needed for the
# u* fit below, so they always run; the two slice files are 1.4 GB and are
# already on disk from the existing column, so they are off unless asked for.
if want_slices
    simulation.output_writers[:velocity] =
        JLD2Writer(model, (; u, v, w), filename = fn("Velocity.jld2"),
                   indices = (:, 1, :), schedule = TimeInterval(200),
                   overwrite_existing = true, with_halos = false)
    simulation.output_writers[:b] =
        JLD2Writer(model, (; b), filename = fn("Buoyancy.jld2"),
                   indices = (:, 1, :), schedule = TimeInterval(200),
                   overwrite_existing = true, with_halos = false)
end

# The four Avg_* writers are OFF by default (AVG_WRITERS=1 restores them). Every
# profile in them is already in the moments file — Avg_b is B, Avg_vel is U, V, W
# and Avg_grad_b is dBdz — so they are pure duplication, and Avg_grad_b runs at
# twice the cadence of everything else while Avg_vort costs six derivative
# operations that nothing in this analysis reads. Together they are 8 plane
# reductions per sample plus 4001 more for the gradient, against 13 for the
# moments: keeping them roughly doubles the output cost of the run.
#
# The u* fit below needs U and V near the wall, and takes them from the moments
# file when these are off, so nothing downstream depends on the switch.
if want_avg
    u_avg = Field(Average(u, dims = (1, 2)))
    v_avg = Field(Average(v, dims = (1, 2)))
    w_avg = Field(Average(w, dims = (1, 2)))
    b_avg = Field(Average(b, dims = (1, 2)))
    db_dz_avg = Field(Average(∂z(b), dims = (1, 2)))
    ωx_avg = Field(Average(∂y(w) - ∂z(v), dims = (1, 2)))
    ωy_avg = Field(Average(∂z(u) - ∂x(w), dims = (1, 2)))
    ωz_avg = Field(Average(∂x(v) - ∂y(u), dims = (1, 2)))

    simulation.output_writers[:avg_db_dz] =
        JLD2Writer(model, (; db_dz = db_dz_avg), filename = fn("Avg_grad_b.jld2"),
                   schedule = TimeInterval(100), overwrite_existing = true)
    simulation.output_writers[:avg_b] =
        JLD2Writer(model, (; b = b_avg), filename = fn("Avg_b.jld2"),
                   schedule = TimeInterval(200), overwrite_existing = true)
    simulation.output_writers[:avg_velocity] =
        JLD2Writer(model, (; u_avg, v_avg, w_avg), filename = fn("Avg_vel.jld2"),
                   schedule = TimeInterval(200), overwrite_existing = true)
    simulation.output_writers[:avg_vorticity] =
        JLD2Writer(model, (; ωx_avg, ωy_avg, ωz_avg), filename = fn("Avg_vort.jld2"),
                   schedule = TimeInterval(200), overwrite_existing = true)
end

# --- the subgrid diffusivity -------------------------------------------------
# The closure is a tuple, (ScalarDiffusivity(), AnisotropicMinimumDissipation()),
# so the closure auxiliary fields are a tuple in the same order: the
# ScalarDiffusivity entry is `nothing` and AMD carries νₑ and a κₑ NamedTuple
# keyed by tracer name. Both the container name and the spelling of κₑ have moved
# between Oceananigans versions, so neither is hardcoded — this is the resolver
# from Stokes/3D/Moments.jl, which searches and logs what it found. It searches
# every entry, so it does not care that AMD is second here and first there.
const _CLOSURE_FIELD_PROPERTIES = (:closure_fields, :diffusivity_fields)
const _KAPPA_NAMES = (:κₑ, :κ_e, :κₜ, :kappa_e, :kappaₑ, :κ)
_is_field(x) = x isa Oceananigans.Fields.AbstractField

function find_sgs_diffusivity(model, tracer_name::Symbol)
    prop = findfirst(p -> hasproperty(model, p), _CLOSURE_FIELD_PROPERTIES)
    prop === nothing && error("""
        ekmanrun.jl: could not find the closure auxiliary fields on the model.
        Tried $(join(string.("model.", _CLOSURE_FIELD_PROPERTIES), ", ")).
        Available properties: $(join(string.(propertynames(model)), ", ")).""")

    propname  = _CLOSURE_FIELD_PROPERTIES[prop]
    container = getproperty(model, propname)
    @info "closure auxiliary fields at `model.$propname` ($(typeof(container).name.name))"
    candidates = container isa Tuple ? collect(container) : Any[container]

    for (i, c) in enumerate(candidates)
        (c === nothing || !(c isa NamedTuple)) && continue
        for name in propertynames(c)
            name in _KAPPA_NAMES || continue
            e = getproperty(c, name)
            fld = e isa NamedTuple ? get(e, tracer_name, nothing) : e
            if _is_field(fld)
                @info "SGS diffusivity resolved as `model.$propname[$i].$name` — $(summary(fld))"
                return fld
            end
        end
        for name in propertynames(c)          # fallback: a rename we do not know
            e = getproperty(c, name)
            e isa NamedTuple || continue
            fld = get(e, tracer_name, nothing)
            if _is_field(fld)
                @warn "SGS diffusivity found by FALLBACK at `model.$propname[$i].$name[:$tracer_name]` — add :$name to _KAPPA_NAMES."
                return fld
            end
        end
    end
    error("""
        ekmanrun.jl: `model.$propname` holds no SGS diffusivity for tracer :$tracer_name.
        Entries: $(join(string.(typeof.(candidates)), ", "))
        Saving F_sgs is the entire purpose of this run, so this is fatal rather
        than a degraded mode: with Pr = $Pr the buoyancy field is effectively
        SGS-controlled on this grid, and dropping the subgrid flux would
        underestimate K_T most badly inside the pycnocline — the one region the
        whole calculation depends on.""")
end

κₑ_b = find_sgs_diffusivity(model, :b)

# --- the moments -------------------------------------------------------------
# Raw moments, written instantaneously on TimeInterval and never on
# AveragedTimeInterval: the mean has to be subtracted per sample and only then
# smoothed, because time-averaging first leaves Var_t(U) — the variance of the
# mean flow over the window — inside what is then called TKE.
#
# Variances at Centers so TKE is one Center profile; fluxes at Faces so
# K_T = −F/(dB/dz) is a ratio of two Face profiles needing no interpolation.
# dBdz uses the model's own ∂z, the same stretched-grid operator the solver uses,
# rather than an offline difference.
plane_avg(op) = Field(Average(op, dims = (1, 2)))

M_U    = plane_avg(u)
M_V    = plane_avg(v)
M_W    = plane_avg(w)                                        # the ⟨w⟩ ≈ 0 check
M_B    = plane_avg(b)
M_dBdz = plane_avg(∂z(b))

M_uu = plane_avg(@at (Center, Center, Center) u * u)
M_vv = plane_avg(@at (Center, Center, Center) v * v)
M_ww = plane_avg(@at (Center, Center, Center) w * w)

M_uw = plane_avg(@at (Center, Center, Face) u * w)
M_vw = plane_avg(@at (Center, Center, Face) v * w)
M_wb = plane_avg(@at (Center, Center, Face) w * b)

# Signed so post-processing simply adds: F_b = ⟨wb⟩ + F_sgs, positive upward.
# κ₀ is the molecular diffusivity — Parameters.jl uses κ for the von Karman
# constant, so getting these two the wrong way round would put 0.41 m²/s of
# diffusivity into the flux.
M_F_sgs     = plane_avg(@at (Center, Center, Face) -(κₑ_b + κ₀) * ∂z(b))
M_kappa_sgs = plane_avg(@at (Center, Center, Face) κₑ_b + κ₀)

simulation.output_writers[:moments] =
    JLD2Writer(model,
               (U = M_U, V = M_V, W = M_W, B = M_B, dBdz = M_dBdz,
                uu = M_uu, vv = M_vv, ww = M_ww,
                uw = M_uw, vw = M_vw, wb = M_wb,
                kappa_sgs = M_kappa_sgs, F_sgs = M_F_sgs),
               filename = fn("Moments.jld2"),
               schedule = TimeInterval(moment_dt),
               overwrite_existing = true, with_halos = false)

@info @sprintf("writing 13 plane-averaged profiles to %s every %.0f s (%d samples)",
               fn("Moments.jld2"), moment_dt, floor(Int, run_duration / moment_dt))

# --- runtime health check ----------------------------------------------------
# (a) ⟨w⟩_xy should be ~1e-18 U∞. If it is not, the plane average is not a clean
#     Reynolds average and every flux in the moments file is wrong.
# (b) κₑ must not go negative. AMD backscatter that the version does not clip
#     gives nonsense K_T in isolated cells, which then has to be masked.
function health_check(sim)
    compute!(M_W)
    w_max = maximum(abs, interior(M_W))
    κ_min, κ_max = extrema(interior(κₑ_b))
    @info @sprintf("health @ %.2f T_f: max|⟨w⟩_xy| = %.3e m/s (= %.1e U∞); κₑ ∈ [%.3e, %.3e] m²/s",
                   sim.model.clock.time / (2π / f₀), w_max, w_max / U∞, κ_min, κ_max)
    w_max / U∞ > 1e-10 &&
        @warn @sprintf("⟨w⟩_xy is %.2e U∞, not ~1e-18 — the fluxes in the moments file are suspect.", w_max / U∞)
    κ_min < 0 &&
        @warn @sprintf("κₑ went negative (min %.3e): this Oceananigans version does not clip AMD backscatter. Mask it in post-processing.", κ_min)
    return nothing
end
simulation.callbacks[:health] =
    Callback(health_check, get(ENV, "EKMAN_SMOKE", "0") == "1" ?
                           IterationInterval(5) : TimeInterval(run_duration / 20))

run!(simulation)

# --- u*, from the same log-layer fit "Ekman 3D.jl" uses ----------------------
# Written into Combined/logs/, not into Ekman/3D Simulation/Parameters/.
function fit_log_layer(z, U; κv = 0.41)
    X = [log.(z) ones(length(z))]
    A, B = X \ U
    r2 = 1 - sum((U .- X * [A, B]) .^ 2) / sum((U .- mean(U)) .^ 2)
    return A * κv, exp(-B / A), r2
end

src, un, vn = want_avg ? (fn("Avg_vel.jld2"), "u_avg", "v_avg") :
                        (fn("Moments.jld2"), "U", "V")
vel = jldopen(src, "r")
last_iter = parse(Int, keys(vel["timeseries/t"])[end])
ua = vel["timeseries/$un/$last_iter"][1, 1, :]
va = vel["timeseries/$vn/$last_iter"][1, 1, :]
close(vel)

const n_fit = 5
zc = Array(znodes(grid, Center()))
u_star, z₀_fit, r2 = fit_log_layer(zc[1:n_fit], @.(sqrt(ua[1:n_fit]^2 + va[1:n_fit]^2)); κv = κ)
δ = u_star / f₀

mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", @sprintf("params_%s.txt", case_tag(r))), "w") do io
    @printf(io, """
Combined/ekmanrun.jl — %s
Written %s

Dimensions                      %.1f m x %.1f m x %.1f m (+ %.0f m sponge)
Grid size                       %d x %d x %d
Far stream velocity             U∞  = %.4f
Square buoyancy frequency       N²  = %.3e
Coriolis parameter              f   = %.2e
Ratio                           r   = N/f = %.1f
Pycnocline height / sharpness   T   = %.1f m, sharp = %.1f
Molecular kinematic viscosity   ν₀  = %.2e
Molecular diffusivity           κ₀  = %.2e
Prandtl number                  Pr  = %.1f
Reynolds number                 Re∞ = %.3e
Drag coefficient                cᴰ  = %.5f
Friction velocity               u*  = %.4e  (log fit over %d points, R² = %.4f)
Fitted roughness                z₀  = %.4e  (imposed %.4e)
Layer lengthscale               δ   = %.2f m
Friction Reynolds               Re* = %.3e
Friction Richardson             Ri* = %.1f
""", case_tag(r), Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
     Lx, Ly, Lz, S, Nx, Ny, Nz, U∞, N², f₀, r, T, sharp, ν₀, κ₀, Pr, Re∞, cᴰ,
     u_star, n_fit, r2, z₀_fit, z₀, δ, u_star * δ / ν₀, N² / f₀^2)
end

@printf("u* = %.4e m/s (R² = %.4f), z₀ fit %.3e vs imposed %.3e, δ = %.2f m\n",
        u_star, r2, z₀_fit, z₀, δ)
@printf("case %s finished at %s\n", case_tag(r), Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
flush(stdout)
