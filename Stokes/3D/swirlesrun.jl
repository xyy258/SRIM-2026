# Swirles run driver: the softplus (profile 4) T = 5 m column on the 100×100×300
# grid, 8 tidal periods, producing figure 4 for the column and a figure 5 per case.
#
# Launched by swirles.sh as
#     srun julia --project=$PROJECT_DIR Stokes/3D/swirlesrun.jl
#
# Steps:
#   1. one Ri = 0 spin-up from rest (skipped if its snapshot is already there)
#   1b. measure that the spin-up actually went turbulent
#   2. the six cases N/ω = √Ri ∈ {0, 0.5, 1, 2, 5, 10}, 8 periods each
#   3. Figure4_metres.jl → figures/Figure4_softplus_T5.png
#   4. Figure5.jl        → figures/Figure5_P4_T5_sqrtRi<s>.png, one per case
#
# ONE SPIN-UP SERVES EVERY CASE. Tidal3D.jl's grid no longer depends on T,
# PROFILE or any run knob, so a single snapshot is a valid restart for all six —
# and for any other T added later. The per-T spin-ups the earlier drivers made
# were forced by a grid that was refined at each T; that refinement is gone.
#
# EXPECT TO SUBMIT THIS MORE THAN ONCE. Scaling run 3's measured T = 5 timings by
# the grid change (585k → 3.0M cells, and Δx halved, which roughly halves the
# CFL-limited Δt) puts a case near 4.3 h and the spin-up near 2.6 h, so the column
# is ~28 h of GPU against a 12 h allocation. The job stops launching cases it
# cannot finish, marks what completed, and picks up where it left off — budget
# about four submissions of swirles.sh. If that is too slow, N_PERIODS=4 fits far
# more per job, at the cost of not being comparable with the 8-period columns.
#
# NOTHING HERE REUSES THE OLD 48×48 DATA. The markers carry the grid in their
# name, so the .done files left by the earlier drivers cannot make this job skip a
# case that now needs redoing; the spin-up sits under its own grid-stamped tag;
# and Tidal3D.jl's check_snapshot refuses a snapshot off any other grid outright.
# Case output files ARE overwritten in place, one case at a time.
#
# NO ANIMATIONS, and no rebuild of the all-T overview. The overview stacks T as
# rows and every other column on disk (T = 10, 15, 20, 30) is still 48×48 data, so
# a row from this grid beside them would read as physics rather than as a change
# of discretization. Figure4_metres.jl is therefore run with SKIP_SWEEP=1, which
# leaves the existing Figure4_softplus_sweep.png untouched. To animate a case
# afterwards:
#     PROFILE=4 T_STRAT=5 julia --project=. Tidal3Danimation.jl sqrtRi2
#
# FIGURE 4 NOW SHOWS 0–30 m, not the 0–15 m the earlier submissions drew. That
# window is a plotting choice and nothing more: the profiles files hold the plane
# average over the whole 50 m physical column (plus a 10 m sponge, which is never
# plotted), so the taller figure comes out of data already on disk. Every case of
# the T = 5 column carries its marker, so re-submitting swirles.sh unchanged skips
# straight past the spin-up and the six cases and redraws the figures — minutes,
# not hours. FIGURES_ONLY=1 skips even the checks in between.
#
# Knobs (all optional):
#   T_VALUES="5"        which columns to run (the grid is the same for any T)
#   SQRT_RI="0 1 2"     subset of the N/ω cases — see the note on parallel jobs below
#   N_PERIODS=8         tidal periods per production case
#   SPIN_PERIODS=5      tidal periods of spin-up
#   WALL_HOURS=11       stop launching new cases past this
#   CASE_HOURS=4.4      first-case time estimate, before the job measures its own
#   GRID_TAG=...        bump when Tidal3D.jl's grid changes, to retire old markers
#   SPIN_W_FLOOR=0.003  turbulence threshold for the spin-up, in w_rms/U₀
#   FIG4_ZMAX=30        top of the figure-4 depth axis, in metres (≤ 50)
#   FIG4_WIDTH=640      figure-4 canvas width in pixels, PER case column
#   FIG4_HEIGHT=390     figure-4 canvas height in pixels — with the width, the aspect
#   FIGURES_ONLY=1      draw the figures from the existing profiles, run nothing
#   SKIP_SPIN_CHECK=1   accept the spin-up without measuring it
#   SKIP_FIGURES=1      simulations only
#   SKIP_PREFLIGHT=1    skip the environment check
#
# ONCE THE SPIN-UP EXISTS the six cases are independent, so several jobs can share
# the column: give each a disjoint SQRT_RI. Do not do that on the first submission
# — two jobs would race to build the same spin-up.

using Printf, Dates

const HERE = @__DIR__

const T_values     = split(get(ENV, "T_VALUES", "5"))
const sqrt_Ri      = split(get(ENV, "SQRT_RI", "0 0.5 1 2 5 10"))
const n_periods    = get(ENV, "N_PERIODS", "8")
const spin_periods = get(ENV, "SPIN_PERIODS", "5")
const skip_figures = get(ENV, "SKIP_FIGURES", "0") == "1"

# Figure 4's depth window, in metres, and the canvas it is drawn on. Passed to
# Figure4_metres.jl, which clamps the window to the 50 m physical domain so the
# sponge is never drawn. Doubling the window without widening the canvas turns each
# panel into a portrait strip, so the width goes up and the height comes down with
# it: 640 × 390 per case column keeps the landscape shape the 0–15 m figures had.
const fig4_zmax   = get(ENV, "FIG4_ZMAX", "30")
const fig4_width  = get(ENV, "FIG4_WIDTH", "640")
const fig4_height = get(ENV, "FIG4_HEIGHT", "390")

# Redraw only. The cases are already on disk and marked complete, so this is the
# submission to make when the change is to the figures rather than to the physics:
# it skips the spin-up, the turbulence check and the case loop outright.
const figures_only = get(ENV, "FIGURES_ONLY", "0") == "1"

# Stamped into the completion markers and the spin-up tag so that work done on a
# different grid is never mistaken for work done on this one. Bump it whenever
# Nx/Ny/Nz_total or the stretching in Tidal3D.jl changes.
const grid_tag = get(ENV, "GRID_TAG", "100x100x300")

# Slurm kills the job at --time; hold back enough for the figures, which are the
# deliverable and are cheap compared with a case.
const wall_hours   = parse(Float64, get(ENV, "WALL_HOURS", "11.0"))
const post_reserve = parse(Float64, get(ENV, "POST_RESERVE_H", "0.3"))

ri_case(s) = "sqrtRi" * replace(s, "." => "p")
t_lbl(t)   = replace(replace(t, r"\.0$" => ""), "." => "p")
tag_of(T, s) = "P4_T$(t_lbl(T))_$(ri_case(s))"
marker_of(tag) = joinpath(HERE, "outputs", tag, ".done_$grid_tag")

const started = time()
elapsed_h() = (time() - started) / 3600
log(msg) = (@printf("[%s | %.2f h] %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
                    elapsed_h(), msg); flush(stdout))

mkpath(joinpath(HERE, "logs"))

# One child process per step. case_params.jl declares the case parameters `const`
# (GPU kernels capture them), so a process can only ever set up one case.
# `dir = HERE` makes the relative outputs/ path resolve here; GKSwstype keeps GR
# headless on a node with no display.
# `env` is AbstractDict, not Dict{String,String}: T_VALUES and SQRT_RI are split
# into SubStrings, so a dict built from them is Dict{String,AbstractString} and
# would not match a concrete signature. Values are converted on the way in.
function run_julia(payload::AbstractVector{<:AbstractString}, env::AbstractDict, logfile;
                   threads = "auto", label = first(payload))
    cmd = Cmd([joinpath(Sys.BINDIR, "julia"), "--project=$HERE", "-t", threads,
               String.(payload)...])
    envs = Dict{String,String}(String(k) => String(v) for (k, v) in env)
    full = merge(Dict{String,String}(ENV), Dict("GKSwstype" => "100"), envs)
    path = joinpath(HERE, "logs", logfile)

    # APPEND, never truncate: logs/ is committed, so these names may already hold
    # earlier attempts, and a resumed job appends to its own. Nothing a child
    # prints carries a wall-clock date, so this banner is what separates them —
    # `grep -n '^==== ' logs/<file>` lists the attempts, newest last.
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

# ---------------- 0. Preflight ----------------
# Checks what this job needs BEFORE spending hours on it: the environment
# instantiates and CUDA works. Serial precompilation, because parallel
# precompilation contends on pidfile locks over cephfs.
const preflight_code = """
using Pkg
Pkg.instantiate()
Pkg.precompile()
using Oceananigans, CUDA, JLD2, Plots
println("preflight: packages loaded")

if CUDA.functional()
    println("preflight: CUDA functional — ", CUDA.name(CUDA.device()))
else
    println("preflight: CUDA NOT functional — the simulations cannot run")
    exit(1)
end
"""

if get(ENV, "SKIP_PREFLIGHT", "0") == "1"
    log("SKIP_PREFLIGHT=1 — not checking the environment")
else
    log("preflight: instantiate + precompile + CUDA, see logs/preflight.log")
    ok = run_julia(["-e", preflight_code],
                   Dict("JULIA_NUM_PRECOMPILE_TASKS" => "1"),
                   "preflight.log"; threads = "1", label = "preflight")
    ok || error("Preflight failed — see logs/preflight.log. If it is a Pkg or precompile " *
                "error, run `JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. -e 'using Pkg; " *
                "Pkg.instantiate(); Pkg.precompile()'` once on a login node. " *
                "Nothing has been run yet.")
    log("preflight OK")
end

# ---------------- 1. Spin-up ----------------
# T_STRAT is set only because case_params.jl wants a value; the spin-up is Ri = 0,
# so its buoyancy field is passive and the grid is T-independent — this snapshot
# is a valid restart for every case below whatever their T.
const spin_tag    = "spinup_$grid_tag"
const spin_fields = joinpath("outputs", spin_tag, "TidalBL3D_$(spin_tag)_fields.jld2")

if figures_only
    log("FIGURES_ONLY=1 — no spin-up, no cases; the figures are redrawn from the profiles on disk")
elseif isfile(joinpath(HERE, spin_fields))
    log("spin-up snapshot already present ($spin_tag) — skipping")
else
    log("spin-up: Ri0 from rest, $spin_periods periods (~2.6 h expected)")
    # FIELDS3D=1 is mandatory: the 3D snapshot IS the spin-up's product, and
    # LIGHT_OUTPUT would otherwise switch it off.
    spin_ok = run_script("Tidal3D.jl", ["Ri0"],
                         Dict("PROFILE"   => "4",
                              "RUN_TAG"   => spin_tag,
                              "T_STRAT"   => first(T_values),
                              "N_PERIODS" => spin_periods,
                              "FIELDS3D"  => "1"),
                         "$spin_tag.log")
    spin_ok || log("  spin-up returned an error — see logs/$spin_tag.log")
    # Its checkpoint is dead weight once the snapshot exists.
    spin_ok && foreach(rm, filter(f -> occursin("_checkpoint", f),
                                  readdir(joinpath(HERE, "outputs", spin_tag), join = true)))
end

# Without the snapshot every case would start from rest with noise (Tidal3D.jl
# warns and carries on), so most of each run would be spin-up rather than the
# stratified evolution the figures are about. Stop instead.
figures_only || isfile(joinpath(HERE, spin_fields)) ||
    error("No spin-up snapshot at $spin_fields — see logs/$spin_tag.log. Nothing else has run.")

# ---------------- 1b. Spin-up turbulence check ----------------
# A spin-up that never transitioned still writes a perfectly valid snapshot, and
# every case restarted from it would run to completion looking normal while being
# laminar — six wasted cases, discovered only when the figures come out
# featureless. w is the cheap discriminator: a laminar Stokes layer is z-dependent
# only, so w ≡ 0 there, while turbulence keeps it at a few percent of U₀. The
# reference band is from the four 48×48 spin-ups (w_rms/U₀ = 0.009–0.012); the
# floor sits a factor of ~3 below it, far above anything decaying noise reaches,
# which leaves room for this finer grid to land somewhat off that band.
#
# Re-checked on every submission, not only after a fresh spin-up: a resumed job
# inherits whatever the previous attempt left, and that is exactly when a bad
# snapshot would go unnoticed. Costs one Julia startup.
const w_floor = get(ENV, "SPIN_W_FLOOR", "0.003")

# U₀ must match case_params.jl (`const U₀ = 0.04`). It only sets the scale the
# ratio is reported in, so a stale value here mis-scales the check rather than
# breaking it — but keep them together.
const U0_ref = get(ENV, "U0_REF", "0.04")

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

if !figures_only && get(ENV, "SKIP_SPIN_CHECK", "0") != "1" &&
   !run_julia(["-e", spin_check_code, joinpath(HERE, spin_fields), U0_ref, w_floor],
              Dict{String,String}(), "$spin_tag.log";
              threads = "1", label = "spin-up turbulence check")
    error("Spin-up is NOT turbulent (see logs/$spin_tag.log). The snapshot is left in " *
          "place for inspection; delete outputs/$spin_tag/ and re-submit to redo it, " *
          "with SPIN_PERIODS raised or a stronger kick in Tidal3D.jl. No cases have run.")
end

# ---------------- 2. Cases ----------------
done_cases, failed_cases, skipped_cases = String[], String[], String[]

# Generous until the job measures its own throughput: starting a case that cannot
# finish wastes all of it, whereas declining one costs only a re-submission.
case_estimate_h = parse(Float64, get(ENV, "CASE_HOURS", string(0.55 * parse(Int, n_periods))))

for Tv in T_values, s in sqrt_Ri
    figures_only && break
    tag  = tag_of(Tv, s)
    dir  = joinpath(HERE, "outputs", tag)
    mark = marker_of(tag)

    if isfile(mark)
        log("  $tag already complete on this grid — skipping"); push!(done_cases, tag); continue
    end
    if elapsed_h() + case_estimate_h > wall_hours - post_reserve
        # One literal format string: @sprintf resolves at macro-expansion time, so
        # a `*`-concatenated format is not a format string at all.
        log(@sprintf("  %s not started — %.2f h left of the budget (%.1f h minus %.1f h held for figures), a case needs ~%.1f h",
                     tag, wall_hours - post_reserve - elapsed_h(), wall_hours,
                     post_reserve, case_estimate_h))
        push!(skipped_cases, tag); continue
    end

    log("run $tag  (N/ω = $s, Ri = $(parse(Float64, s)^2), $n_periods periods)")
    case_start = time()
    # LIGHT_OUTPUT drops the 3D snapshots, which figures 4 and 5 never open, at no
    # cost in runtime. It keeps the x–z slices, so a case can still be animated.
    ok = run_script("Tidal3D.jl", [ri_case(s)],
                    Dict("PROFILE"      => "4",
                         "T_STRAT"      => Tv,
                         "N_PERIODS"    => n_periods,
                         "LIGHT_OUTPUT" => "1",
                         "SPINUP_FILE"  => spin_fields),
                    "$tag.log")

    if ok && isfile(joinpath(dir, "TidalBL3D_$(tag)_profiles.jld2"))
        # The marker, not the profiles file, is the completion test: a run cut
        # short leaves a valid but truncated profiles file behind.
        write(mark, string(now()))
        foreach(rm, filter(f -> occursin("_checkpoint", f), readdir(dir, join = true)))
        push!(done_cases, tag)
        global case_estimate_h = 1.1 * (time() - case_start) / 3600
        log(@sprintf("  %s done in %.2f h", tag, (time() - case_start) / 3600))
    else
        push!(failed_cases, tag)
        log("  $tag FAILED — see logs/$tag.log")
    end
end

figures_only ||
    log("simulations: $(length(done_cases)) complete, $(length(failed_cases)) failed, " *
        "$(length(skipped_cases)) not started")

# ---------------- 3–4. Figures ----------------
# Held back until EVERY case of the column carries this grid's marker. The figure
# scripts happily plot whatever profiles files they find, and a case not yet redone
# still has its 48×48 file sitting there — so plotting early would mix two grids
# into one panel and label it as one experiment. Nothing is lost by waiting: the
# markers persist, so the submission that finishes the column draws the figures.
const column_complete = all(isfile(marker_of(tag_of(T, s))) for T in T_values, s in sqrt_Ri)

if skip_figures
    log("SKIP_FIGURES=1 — stopping before the figures")
elseif !column_complete
    pending = [tag_of(T, s) for T in T_values, s in sqrt_Ri if !isfile(marker_of(tag_of(T, s)))]
    log("figures deferred — $(length(pending)) case(s) still to run on this grid: " *
        join(pending, ", "))
    log("  re-submit swirles.sh to continue; the figures are drawn once the column is complete")
else
    ri_list = join(sqrt_Ri, " ")
    T_list  = join(T_values, " ")
    np = parse(Int, n_periods)
    # Figure 5 draws one curve per whole period listed, four at a time to match its
    # colour ramp: the LAST four periods, i.e. the state actually reached.
    periods_plot = join(max(1, np - 3):np, " ")
    fig_env = Dict("T_VALUES" => T_list, "SQRT_RI" => ri_list,
                   "N_PERIODS_PLOT" => periods_plot)

    # Figure 4 only. Figure 5 plots profiles against z on its own axis and is left
    # exactly as it was, so fig_env stays clean of these.
    fig4_env = merge(fig_env, Dict("SKIP_SWEEP"  => "1",
                                   "FIG4_ZMAX"   => fig4_zmax,
                                   "FIG4_WIDTH"  => fig4_width,
                                   "FIG4_HEIGHT" => fig4_height))

    # SKIP_SWEEP is not optional here: without it Figure4_metres.jl rebuilds the
    # combined overview from T_VALUES alone, replacing the multi-T figure with a
    # single row — and see the header on why that figure should not be rebuilt at
    # all while the other columns are 48×48 data.
    log("Figure 4 (T ∈ {$T_list}, N/ω ∈ {$ri_list}, depth axis 0–$fig4_zmax m)")
    run_script("Figure4_metres.jl", String[], fig4_env, "post_figures.log") ||
        log("  Figure4_metres.jl FAILED — see logs/post_figures.log")

    log("Figure 5 (one per case, periods $periods_plot)")
    run_script("Figure5.jl", String[], fig_env, "post_figures.log") ||
        log("  Figure5.jl FAILED — see logs/post_figures.log")
end

# ---------------- Summary ----------------
log("=== DONE ===")
isempty(failed_cases)  || log("failed:      " * join(failed_cases, ", "))
isempty(skipped_cases) || log("not started: " * join(skipped_cases, ", "))

if column_complete && !skip_figures
    log("push the figures from here:")
    log("  git add -A && git commit -m 'T = $(join(T_values, ", ")) on $grid_tag' && git push")
elseif !isempty(skipped_cases)
    log("re-submit swirles.sh to continue — $(length(skipped_cases)) case(s) left")
end
