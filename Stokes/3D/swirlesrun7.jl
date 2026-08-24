#!/usr/bin/env julia
# Batch driver for the N/ω column: one pycnocline height, seven stratifications.
# Launched by swirles.sh as
#     srun julia --project=$PROJECT_DIR Stokes/3D/swirlesrun7.jl
#
# ---------------- What this run is for ----------------
# swirlesrun4.jl inverts K_T = √TKE·l for the mixing length l = K_T/√TKE and
# asks which candidate scale it follows. With h taken as the 0.1-of-background
# crossing, the previous column left this table (logs/length_scales.log,
# archived under h_defs/crossing 0.1 background N/):
#
#     N/ω        1       2      10
#     median l/(√TKE/N)  0.013   0.017   0.024      (T = 10)
#                        0.009   0.016   0.016      (T = 5)
#
# The ratio climbs with N/ω instead of sitting still, so at these stratifications
# √TKE/N is not yet the scale that sets the mixing — but three points cannot say
# whether it is heading for a plateau (the buoyancy scale taking over, as it
# should at large N) or simply rising without limit (the wrong scale). The other
# candidates fall the other way: median l/h drops from 0.003 to 0.001 between
# N/ω = 1 and 10, and l/(u*/ω) does the same.
#
# So this run widens the column to N/ω ∈ {0, 1, 2, 5, 10, 25, 50} at fixed
# T = 10 m. Half a decade beyond the old top end is enough to tell a plateau from
# a ramp, and N/ω = 50 is Ri = 2500, the strongest case in Gayen, Sarkar & Taylor
# (2010), so it is a stratification the reference covers rather than an
# extrapolation.
#
# T is fixed at 10 m because stage 0 of the previous sweep measured h0 = 9.215 m
# (logs/h0_P4_T10_sqrtRi0.log), so T/h0 = 1.09: the pycnocline sits at the top of
# the turbulent layer, where the layer reaches it in every case and how far it
# entrains past it is what varies. That is inside the usable 0.3-2.0 h0 window,
# and it is the T four of the seven cases are already run at.
#
# ---------------- The budget ----------------
# swirles.sh asks for 12 h. Measured on this grid and this bottom boundary
# condition, a case is 1.95 h for 8 periods including the moments writer
# (1.92-1.97 h across N/ω = 0, 1, 2, 10 — the time step is set by the advective
# CFL and settles at 3-12 s, so it barely moves with N). The wave limit is no
# constraint either: at N/ω = 50, 1/N = 200 s, far above the step in use.
#
#   4 cases already complete at T = 10 (N/ω = 0, 1, 2, 10)     0 h
#   3 new cases x 1.95 h  (N/ω = 5, 25, 50)                    5.9 h
#   post-processing                                            0.3 h
#                                                            -------
#                                                              6.2 h   in a 12 h wall
#
# That assumes outputs/P4_T10_sqrtRi{0,1,2,10}/.done_moments_<grid_tag> are on
# this filesystem. If they are not — a fresh checkout, or a scratch that has been
# cleared — all seven run, which is 13.6 h and does not fit. Nothing is lost when
# that happens: every case writes its own marker, the wall-clock guard declines a
# case it cannot finish rather than starting one, and re-submitting picks up
# where this left off. Expect five cases in the first 12 h and the rest in a
# short second submission.
#
# For that reason SQRT_RI is ordered with the three NEW stratifications first.
# Cases already on disk are skipped in milliseconds whatever the order, so this
# costs nothing when the old ones are present, and when they are not it means the
# first submission returns the points this run exists to add rather than
# re-deriving ones already published in the last figure.
#
# ---------------- What it produces ----------------
#   outputs/<tag>/TidalBL3D_<tag>_moments.jld2   second moments, per case
#   outputs/<tag>/mixing_<tag>_hcross.jld2       K_T, h, TKE at h, per case
#   h_defs/crossing_0p1/figures/K_T_<tag>.png    the four-panel check, per case
#   h_defs/crossing_0p1/figures/l_vs_*.png       l against each candidate scale
#   h_defs/crossing_0p1/figures/l_over_L_vs_N.png   median l/L against N/ω
#
# The post-processing runs with H_DEF=crossing H_LEVEL=0.1, the definition the
# trend above was read from. The other two definitions of h are not re-run here:
# they are a separate question, and run_h_definitions.sh in "Code running/" does
# all three over whatever is on disk once this has finished.
#
# l_over_L_vs_N.png is the figure this run is for. A candidate that sets the
# mixing length gives a flat line; the ramp in the table above is a sloping one.
#
# ---------------- Stages (SWEEP_STAGE, default "auto") ----------------
#   spinup   turbulent spin-up with the drag bottom, if absent      ~1.5 h
#   cases    the column only, no post-processing
#   post     post-processing only, over whatever is on disk
#   auto     spin-up -> cases -> post-processing
#
# ENV: SWEEP_STAGE, T_STRAT, SQRT_RI, N_PERIODS, SPIN_PERIODS, WALL_HOURS,
#      CASE_HOURS, POST_RESERVE_H, GRID_TAG, FIELDS3D_CASE, H_LEVEL, CD,
#      Z_DRAG_REF, SHARP, GRAD_FLOOR, SMOOTH, SKIP_PERIODS, SKIP_PREFLIGHT,
#      SKIP_SPIN_CHECK, SKIP_POST, DRY_RUN.
#
# DRY_RUN=1 prints the plan — which cases are complete, which would run, and what
# they are estimated to cost — and exits without touching the GPU. Worth doing on
# a login node before submitting, since it is the cheapest way to find out
# whether the four old cases are visible from here.

using Printf, Dates

const HERE = @__DIR__

const stage = get(ENV, "SWEEP_STAGE", "auto")

# One pycnocline height. See the header: T/h0 = 1.09 at the measured h0.
const T_strat = get(ENV, "T_STRAT", "10")

# New stratifications first — see the ordering note in the header. The list is
# sorted numerically before post-processing, so the figures and tables come out
# in N/ω order regardless of the order the cases ran in.
const sqrt_Ri = split(get(ENV, "SQRT_RI", "5 25 50 0 1 2 10"))

# 8 periods, matching the four cases already on disk. Changing this would make
# the new points incomparable with the old ones, which is the whole purpose of
# reusing them; if it is changed, change GRID_TAG too so nothing is reused.
const n_periods    = get(ENV, "N_PERIODS", "8")
const spin_periods = get(ENV, "SPIN_PERIODS", "5")

# The bottom boundary condition is quadratic drag, as in the previous sweep. The
# tag carries "_drag" so a marker from the older no-slip runs cannot make this
# job skip a case that was never run on this wall layer.
const grid_tag = get(ENV, "GRID_TAG", "100x100x300_drag")

const wall_hours   = parse(Float64, get(ENV, "WALL_HOURS", "11.3"))
const post_reserve = parse(Float64, get(ENV, "POST_RESERVE_H", "0.4"))
const dry_run      = get(ENV, "DRY_RUN", "0") == "1"
const skip_post    = get(ENV, "SKIP_POST", "0") == "1"

# The mixed-layer height definition this run is about: h is the lowest z at which
# the plane-averaged gradient recovers to H_LEVEL of the background, the foot of
# the interface. See mixed_layer_height.jl.
const h_def   = "crossing"
const h_level = get(ENV, "H_LEVEL", "0.1")
const h_sfx   = "_hcross"
const h_root  = joinpath(HERE, "h_defs", "crossing_0p1")

ri_case(s)   = "sqrtRi" * replace(s, "." => "p")
t_lbl(t)     = replace(replace(t, r"\.0$" => ""), "." => "p")
tag_of(T, s) = "P4_T$(t_lbl(T))_$(ri_case(s))"

# One case keeps its 3D snapshots, as the only route to an analysis this sweep
# did not plan for. P4_T10_sqrtRi2 already has them from the previous column, so
# the one worth adding is the strongest new stratification: it is the case
# furthest from anything on disk and the one whose interface structure the
# column-averaged moments describe least well. It costs about 750 MB.
# FIELDS3D_CASE=none turns it off.
const fields3d_case = get(ENV, "FIELDS3D_CASE",
                          tag_of(T_strat, string(argmax(x -> something(tryparse(Float64, x), -Inf),
                                                        sqrt_Ri))))

marker_of(tag)  = joinpath(HERE, "outputs", tag, ".done_moments_$grid_tag")
moments_of(tag) = joinpath(HERE, "outputs", tag, "TidalBL3D_$(tag)_moments.jld2")
mixing_of(tag)  = joinpath(HERE, "outputs", tag, "mixing_$(tag)$(h_sfx).jld2")

const started = time()
elapsed_h() = (time() - started) / 3600
log(msg) = (@printf("[%s | %.2f h] %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
                    elapsed_h(), msg); flush(stdout))

mkpath(joinpath(HERE, "logs"))
mkpath(joinpath(h_root, "figures"))
mkpath(joinpath(h_root, "logs"))

# One child process per step. case_params.jl declares its parameters `const`,
# since GPU kernels capture them, so one process can only ever set up one case.
# `dir = HERE` makes the relative outputs/ path resolve here, and GKSwstype keeps
# the plotting headless on a node with no display.
function run_julia(payload::AbstractVector{<:AbstractString}, env::AbstractDict, logfile;
                   threads = "auto", label = first(payload))
    cmd = Cmd([joinpath(Sys.BINDIR, "julia"), "--project=$HERE", "-t", threads,
               String.(payload)...])
    envs = Dict{String,String}(String(k) => String(v) for (k, v) in env)
    full = merge(Dict{String,String}(ENV), Dict("GKSwstype" => "100"), envs)
    path = joinpath(HERE, "logs", logfile)

    # Append rather than truncate: logs/ is committed, so a file may already hold
    # earlier attempts and a resumed job appends to its own. Nothing a child
    # prints carries a date, so this banner is what separates them; list the
    # attempts with `grep -n '^==== ' logs/<file>`, newest last.
    open(path, "a") do io
        println(io, "\n", "="^78)
        println(io, "==== ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "  ", label)
        println(io, "==== ", join(("$k=$v" for (k, v) in sort(collect(envs))), "  "))
        println(io, "="^78)
        flush(io)
    end

    return success(pipeline(setenv(cmd, full; dir = HERE),
                            stdout = path, stderr = path, append = true))
end

run_script(script, args::AbstractVector{<:AbstractString}, env::AbstractDict, logfile) =
    run_julia([joinpath(HERE, script), String.(args)...], env, logfile;
              label = script * " " * join(args, " "))

# Settings forwarded only when they are actually set, so an unset one keeps the
# child script's own default rather than being overridden with an empty string.
passthrough(keys...) = Dict{String,String}(k => ENV[k] for k in keys if haskey(ENV, k))

# ---------------- 0. Preflight ----------------
# Fail in the first two minutes rather than after the spin-up. The SGS
# diffusivity access path is the most likely thing here to break on a version
# bump, and Moments.jl needs it for every case.
const preflight_code = """
using Pkg
Pkg.instantiate()
Pkg.precompile()
using Oceananigans, CUDA, JLD2, Plots
println("preflight: packages loaded, Oceananigans ", pkgversion(Oceananigans))

# The grid is POSITIONAL in this version — NonhydrostaticModel(grid = ...) is a
# MethodError, not a deprecation.
m = NonhydrostaticModel(RectilinearGrid(CPU(); size = (2, 2, 2), extent = (1, 1, 1));
                        tracers = :b,
                        closure = (AnisotropicMinimumDissipation(),
                                   ScalarDiffusivity(ν = 1e-6, κ = 1e-7)))
prop = hasproperty(m, :closure_fields) ? :closure_fields :
       hasproperty(m, :diffusivity_fields) ? :diffusivity_fields : nothing
prop === nothing && (println("preflight: NO closure-field container on the model"); exit(1))
cf = getproperty(m, prop)
c1 = cf isa Tuple ? cf[1] : cf
println("preflight: model.\$prop entry 1 has ", propertynames(c1))
:κₑ in propertynames(c1) || println("preflight: WARNING — κₑ not found; Moments.jl will fall back or error")

if CUDA.functional()
    println("preflight: CUDA functional — ", CUDA.name(CUDA.device()))
else
    println("preflight: CUDA NOT functional — the simulations cannot run")
    exit(1)
end
"""

function preflight()
    if get(ENV, "SKIP_PREFLIGHT", "0") == "1"
        log("SKIP_PREFLIGHT=1 — not checking the environment")
        return
    end
    log("preflight: instantiate + precompile + CUDA + SGS access path, see logs/preflight_sweep7.log")
    ok = run_julia(["-e", preflight_code],
                   Dict("JULIA_NUM_PRECOMPILE_TASKS" => "1"),
                   "preflight_sweep7.log"; threads = "1", label = "preflight")
    ok || error("Preflight failed — see logs/preflight_sweep7.log. If it is a Pkg or " *
                "precompile error, run `JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. " *
                "-e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'` once on a login " *
                "node. Nothing has been run yet.")
    log("preflight OK")
end

# ---------------- 1. Spin-up ----------------
# One Ri = 0 spin-up serves every case: the velocity field does not depend on the
# buoyancy profile, so all seven restart from the same turbulent state and the
# only thing separating them is N.
const spin_tag    = "spinup_$grid_tag"
const spin_fields = joinpath("outputs", spin_tag, "TidalBL3D_$(spin_tag)_fields.jld2")

function do_spinup()
    if isfile(joinpath(HERE, spin_fields))
        log("drag spin-up already present ($spin_tag) — skipping")
        return true
    end
    log("spin-up: Ri0 from rest, $spin_periods periods, DRAG bottom (~1.5 h expected)")
    log("  the no-slip snapshots under outputs/spinup_* are a different wall layer and are left alone")
    # FIELDS3D=1 is required, since the 3D snapshot is what the spin-up produces
    # and LIGHT_OUTPUT would otherwise switch it off. MOMENTS=0 because the
    # spin-up is unstratified and its second moments are of no use.
    ok = run_script("Tidal3D.jl", ["Ri0"],
                    merge(Dict("PROFILE"      => "4",
                               "RUN_TAG"      => spin_tag,
                               "T_STRAT"      => T_strat,
                               "N_PERIODS"    => spin_periods,
                               "LIGHT_OUTPUT" => "1",
                               "FIELDS3D"     => "1",
                               "MOMENTS"      => "0"),
                          passthrough("CD", "Z_DRAG_REF")),
                    "$spin_tag.log")
    ok || log("  spin-up returned an error — see logs/$spin_tag.log")
    ok && foreach(rm, filter(f -> occursin("_checkpoint", f),
                             readdir(joinpath(HERE, "outputs", spin_tag), join = true)))
    return isfile(joinpath(HERE, spin_fields))
end

# A spin-up that never became turbulent still writes a valid snapshot, and every
# case restarted from it would run to completion looking normal while being
# laminar. w is a cheap test: a laminar Stokes layer depends on z only, so w is
# zero there, while turbulence keeps it at a few per cent of U₀.
const spin_check_code = """
using Oceananigans, Statistics, Printf
f, U0, floor_ = ARGS[1], parse(Float64, ARGS[2]), parse(Float64, ARGS[3])
ts = FieldTimeSeries(f, "w")
n  = length(ts.times)
w  = Array(interior(ts[n]))
r  = sqrt(mean(w .^ 2)) / U0
@printf("spin-up check: w_rms/U0 = %.4f at snapshot %d/%d (t = %.2f periods)\\n",
        r, n, n, ts.times[n] / (2π / 1e-4))
@printf("  turbulent reference 0.009-0.012, floor %.4f -> %s\\n",
        floor_, r < floor_ ? "LAMINAR" : "turbulent")
exit(r < floor_ ? 1 : 0)
"""

function check_spinup()
    get(ENV, "SKIP_SPIN_CHECK", "0") == "1" && return
    ok = run_julia(["-e", spin_check_code, joinpath(HERE, spin_fields),
                    get(ENV, "U0_REF", "0.04"), get(ENV, "SPIN_W_FLOOR", "0.003")],
                   Dict{String,String}(), "$spin_tag.log";
                   threads = "1", label = "spin-up turbulence check")
    ok || error("Spin-up is NOT turbulent (see logs/$spin_tag.log). The snapshot is left " *
                "in place for inspection; delete outputs/$spin_tag/ and re-submit to redo " *
                "it, with SPIN_PERIODS raised. No cases have run.")
end

# ---------------- 2. Cases ----------------
# The estimate starts generous and is replaced by a measurement after the first
# case: starting a case that cannot finish wastes all of it, while declining one
# costs only a re-submission. 1.95 h per 8 periods is what the previous column
# measured; 2.4 h is that plus a margin for a contended node.
case_estimate_h = parse(Float64, get(ENV, "CASE_HOURS", string(0.30 * parse(Int, n_periods))))
done_cases, failed_cases, skipped_cases = String[], String[], String[]

function run_case(s)
    tag  = tag_of(T_strat, s)
    dir  = joinpath(HERE, "outputs", tag)
    mark = marker_of(tag)

    if isfile(mark)
        log("  $tag already complete on this grid+BC — skipping")
        tag in done_cases || push!(done_cases, tag)
        return true
    end
    if elapsed_h() + case_estimate_h > wall_hours - post_reserve
        log(@sprintf("  %s not started — %.2f h left of the budget (%.1f h minus %.1f h held for post-processing), a case needs ~%.1f h",
                     tag, wall_hours - post_reserve - elapsed_h(), wall_hours,
                     post_reserve, case_estimate_h))
        push!(skipped_cases, tag)
        return false
    end

    fields3d = tag == fields3d_case ? "1" : "0"
    Ri = something(tryparse(Float64, s), NaN)^2
    log(@sprintf("run %s  (T = %s m, N/ω = %s, Ri = %g, N = %.1e s⁻¹, %s periods, FIELDS3D=%s)",
                 tag, T_strat, s, Ri, something(tryparse(Float64, s), NaN) * 1e-4,
                 n_periods, fields3d))
    case_start = time()
    ok = run_script("Tidal3D.jl", [ri_case(s)],
                    merge(Dict("PROFILE"      => "4",
                               "T_STRAT"      => T_strat,
                               "N_PERIODS"    => n_periods,
                               "LIGHT_OUTPUT" => "1",
                               "FIELDS3D"     => fields3d,
                               "MOMENTS"      => "1",
                               "SPINUP_FILE"  => spin_fields),
                          passthrough("CD", "Z_DRAG_REF", "SHARP")),
                    "$tag.log")

    if ok && isfile(moments_of(tag))
        # The marker, not the moments file, is the test for completion: a run cut
        # short leaves a valid but truncated file behind.
        write(mark, string(now()))
        foreach(rm, filter(f -> occursin("_checkpoint", f), readdir(dir, join = true)))
        push!(done_cases, tag)
        global case_estimate_h = 1.1 * (time() - case_start) / 3600
        log(@sprintf("  %s done in %.2f h", tag, (time() - case_start) / 3600))
        return true
    else
        push!(failed_cases, tag)
        log("  $tag FAILED — see logs/$tag.log")
        return false
    end
end

# ---------------- 3. Post-processing ----------------
# Sorted, so the figures and the tables come out in N/ω order whatever order the
# cases ran in.
sorted_Ri() = sort(sqrt_Ri, by = s -> something(tryparse(Float64, s), Inf))

function post()
    skip_post && (log("SKIP_POST=1 — no post-processing"); return)
    ss = sorted_Ri()
    have = [s for s in ss if isfile(moments_of(tag_of(T_strat, s)))]
    isempty(have) && (log("no moments files to post-process yet"); return)
    log("post-processing $(length(have)) case(s), H_DEF=$h_def H_LEVEL=$h_level → " *
        "mixing_<tag>$(h_sfx).jld2 and $(relpath(h_root, HERE))/figures/")

    hopts = Dict("H_DEF" => h_def, "H_LEVEL" => h_level, "MIX_SUFFIX" => h_sfx,
                 "FIG_DIR" => joinpath(h_root, "figures"),
                 "LOG_DIR" => joinpath(h_root, "logs"))

    # 1. K_T, h and the four-panel check, one case at a time.
    ok = run_script("MixedLayerDiffusivity.jl", String[],
                    merge(hopts,
                          Dict("T_VALUES" => T_strat, "N_OVER_OMEGA" => join(have, " ")),
                          passthrough("GRAD_FLOOR", "SMOOTH", "SKIP_PERIODS")),
                    "post_sweep7.log")
    ok || log("  MixedLayerDiffusivity returned an error — see logs/post_sweep7.log")
    # The verification block is the result, so print it into the job log rather
    # than leaving it in a child log.
    path = joinpath(HERE, "logs", "post_sweep7.log")
    if isfile(path)
        lines = readlines(path)
        i = findlast(l -> startswith(l, "VERIFICATION"), lines)
        i === nothing || foreach(println, lines[max(1, i - 2):end])
        flush(stdout)
    end

    # 2. The length-scale question itself. K_T_bulk is averaged across the
    #    interface while TKE is local at z = h, so the pairing is a genuine
    #    mismatch; the at_h pass repeats everything with the local K_T, and a
    #    conclusion that survives both is the one to believe.
    for (src, sfx, lf) in (("bulk", "", "length_scales_sweep7.log"),
                           ("at_h", "_ath", "length_scales_ath_sweep7.log"))
        ok = run_script("swirlesrun4.jl", String[],
                        merge(hopts, Dict("K_SOURCE" => src, "RESULT_SUFFIX" => sfx),
                              passthrough("SKIP_PERIODS")),
                        lf)
        ok || log("  swirlesrun4.jl ($src) returned an error — see logs/$lf")
    end
    log("post-processing done — the figure this run is for is " *
        "$(relpath(h_root, HERE))/figures/l_over_L_vs_N.png")
end

# ---------------- The plan ----------------
# Printed at the start of every run, and the whole of DRY_RUN=1.
function show_plan()
    ss = sorted_Ri()
    todo = [s for s in sqrt_Ri if !isfile(marker_of(tag_of(T_strat, s)))]
    println("\n", "#"^78)
    @printf("#  N/ω column at T = %s m, %s periods, grid+BC tag %s\n", T_strat, n_periods, grid_tag)
    @printf("#  N/ω ∈ {%s},  run order {%s}\n", join(ss, ", "), join(sqrt_Ri, ", "))
    println("#")
    println("#  case                    N/ω      Ri   status")
    for s in ss
        tag = tag_of(T_strat, s)
        v = something(tryparse(Float64, s), NaN)
        st = isfile(marker_of(tag))  ? "complete — will be skipped" :
             isfile(moments_of(tag)) ? "moments on disk but NO marker — will be re-run" :
                                       "TO RUN" * (tag == fields3d_case ? "  (keeps 3D fields)" : "")
        @printf("#  %-22s %5g %7g   %s\n", tag, v, v^2, st)
    end
    println("#")
    @printf("#  %d to run x ~%.1f h = %.1f h, plus %.1f h held for post-processing, in a %.1f h budget\n",
            length(todo), case_estimate_h, length(todo) * case_estimate_h,
            post_reserve, wall_hours)
    if length(todo) * case_estimate_h + post_reserve > wall_hours
        n_fit = max(0, floor(Int, (wall_hours - post_reserve) / case_estimate_h))
        @printf("#  THAT DOES NOT FIT: about %d of the %d will run and the rest will be\n",
                n_fit, length(todo))
        println("#  declined by the wall-clock guard. Every case carries its own marker, so")
        println("#  re-submitting swirles.sh continues rather than restarts. Nothing is lost.")
    end
    println("#"^78, "\n")
    flush(stdout)
end

# ---------------- Drive ----------------
log("=== N/ω column, stage \"$stage\", T = $T_strat m, grid+BC tag $grid_tag ===")
show_plan()

if dry_run
    log("DRY_RUN=1 — the plan above is all this does. Nothing has run.")
elseif stage == "post"
    post()
else
    # Before the spin-up, not after: the point of the preflight is to fail in the
    # first two minutes rather than 1.5 h in. `post` never reaches here, since it
    # needs no GPU and is expected to be run on a login node.
    preflight()
    do_spinup() || error("No spin-up snapshot at $spin_fields — see logs/$spin_tag.log. Nothing else has run.")
    check_spinup()

    if stage == "spinup"
        log("spin-up only — stopping")
    elseif stage in ("cases", "auto")
        log("--- the column: T = $T_strat m, N/ω ∈ {$(join(sorted_Ri(), ", "))}, $n_periods periods ---")
        for s in sqrt_Ri; run_case(s); end
        stage == "auto" && post()
    else
        error("Unknown SWEEP_STAGE \"$stage\" — use spinup, cases, post or auto")
    end
end

log(@sprintf("=== %s: %d complete, %d failed, %d not started ===",
             stage, length(done_cases), length(failed_cases), length(skipped_cases)))
isempty(failed_cases)  || log("  failed:      " * join(failed_cases, ", "))
isempty(skipped_cases) || log("  not started: " * join(skipped_cases, ", ") *
                              "  — re-submit swirles.sh to continue")
