# Swirles run driver: the softplus (profile 4) T = 30 m column.
#
# Runs, in order:
#   1. one Ri = 0 spin-up on the T = 30 grid (skipped if its snapshot exists),
#   2. the six production cases N/ω = √Ri ∈ {0, 0.5, 1, 2, 5, 10}, each restarted
#      from that snapshot,
#   3. Figure4_metres.jl  → figures/Figure4_softplus_T30.png (all six columns),
#   4. Figure5.jl         → figures/Figure5_P4_T30_sqrtRi<s>.png, one per case.
#
# Launched by swirles.sh as
#     srun julia --project=$PROJECT_DIR Stokes/3D/swirlestestrun.jl
# This file itself needs nothing but stdlib: every step is a SEPARATE julia
# process. That is not incidental — case_params.jl declares the case parameters
# `const` (they are captured by GPU kernels), so a second case cannot be set up
# in a process that has already run one. Each child is launched with its working
# directory set to this file's directory and --project pointed at it, because
# Tidal3D.jl calls Pkg.activate(".") and case_params.jl writes to a RELATIVE
# outputs/ path; run from anywhere else the data lands beside the wrong project.
#
# RESUMABLE. A case counts as done only once it has written a .done marker, so a
# job killed by the wall clock mid-case redoes that case rather than skipping a
# truncated one. Re-submit swirles.sh to continue; the figures step re-runs every
# time and simply draws whatever is on disk.
#
# Knobs (all optional):
#   T_STRAT=30          pycnocline height, metres
#   SQRT_RI="0 1 2"     subset of the N/ω column
#   N_PERIODS=8         tidal periods per production case
#   SPIN_PERIODS=5      tidal periods of spin-up
#   WALL_HOURS=8.5      stop launching new cases past this; figures still run
#   SKIP_FIGURES=1      simulations only

using Printf, Dates

const HERE = @__DIR__

const T_strat      = get(ENV, "T_STRAT", "30")
const sqrt_Ri      = split(get(ENV, "SQRT_RI", "0 0.5 1 2 5 10"))
const n_periods    = get(ENV, "N_PERIODS", "8")
const spin_periods = get(ENV, "SPIN_PERIODS", "5")
const skip_figures = get(ENV, "SKIP_FIGURES", "0") == "1"

# Slurm kills the job at --time; leave enough room to finish the figures rather
# than losing them with the allocation. Keep in step with swirles.sh's --time.
const wall_hours = parse(Float64, get(ENV, "WALL_HOURS", "8.5"))

# The vertical grid is refined at this T alone (T_SWEEP), giving Nz ≈ 254–290
# instead of the 458 a grid banded at all four sweep values needs. The cost is
# that this column's discretization is its own — fine here, since every figure
# produced below compares cases at ONE T, but it means these runs are not
# directly comparable with a T = 5/10/20 block built on a different grid.
const T_sweep = T_strat

# case label → directory suffix, matching case_params.jl: 0.5 → sqrtRi0p5
ri_case(s) = "sqrtRi" * replace(s, "." => "p")
t_lbl(t)   = replace(replace(t, r"\.0$" => ""), "." => "p")

const TL       = t_lbl(T_strat)
const spin_tag = "spinup_T" * TL
const spin_fields = joinpath("outputs", spin_tag, "TidalBL3D_$(spin_tag)_fields.jld2")

const started = time()
elapsed_h() = (time() - started) / 3600
log(msg) = (@printf("[%s | %.2f h] %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
                    elapsed_h(), msg); flush(stdout))

mkpath(joinpath(HERE, "logs"))

# One child process. `dir = HERE` is what makes the relative outputs/ path in
# case_params.jl resolve here; GKSwstype = 100 keeps GR headless on a node with
# no display (the figure steps abort without it).
function run_julia(script, args::Vector{String}, env::Dict{String,String}, logfile)
    cmd = Cmd([joinpath(Sys.BINDIR, "julia"), "--project=$HERE", "-t", "auto",
               joinpath(HERE, script), args...])
    full = merge(Dict{String,String}(ENV), Dict("GKSwstype" => "100"), env)
    path = joinpath(HERE, "logs", logfile)
    return success(pipeline(setenv(cmd, full; dir = HERE),
                            stdout = path, stderr = path, append = true))
end

# ---------------- 1. Spin-up ----------------
# Unstratified, from rest, and the ONLY thing it exists to write is the 3D
# snapshot the six cases restart from — hence FIELDS3D=1, which LIGHT_OUTPUT
# would otherwise switch off. The paper's protocol: spin up unstratified to
# turbulence, then switch stratification on.
if isfile(joinpath(HERE, spin_fields))
    log("spin-up snapshot already present — skipping ($spin_fields)")
else
    log("T = $T_strat: spin-up (Ri0 from rest, $spin_periods periods)")
    spin_ok = run_julia("Tidal3D.jl", ["Ri0"],
                        Dict("PROFILE"   => "4",
                             "RUN_TAG"   => spin_tag,
                             "T_STRAT"   => T_strat,
                             "T_SWEEP"   => T_sweep,
                             "N_PERIODS" => spin_periods,
                             "FIELDS3D"  => "1"),
                        "$spin_tag.log")
    spin_ok || log("  spin-up returned an error — see logs/$spin_tag.log")
    # Its checkpoint is ~0.15 GB and is dead weight: the spin-up's product is the
    # fields snapshot, and nothing ever resumes the spin-up itself.
    if spin_ok
        foreach(rm, filter(f -> occursin("_checkpoint", f),
                           readdir(joinpath(HERE, "outputs", spin_tag), join = true)))
    end
end

# Without the snapshot every case would start from rest with noise (Tidal3D.jl
# warns and carries on), so eight periods of each run would be spin-up rather
# than the stratified evolution the figures are about. Stop instead.
isfile(joinpath(HERE, spin_fields)) ||
    error("No spin-up snapshot at $spin_fields — cannot start the stratified cases")

# ---------------- 2. The six cases ----------------
done_cases, failed_cases, skipped_cases = String[], String[], String[]

# A case is ~1 h per 4 periods on this grid, so the guess below is deliberately
# generous: starting a case that cannot finish wastes the whole of it, whereas
# declining one costs only a re-submission.
case_estimate_h = parse(Float64, get(ENV, "CASE_HOURS", string(0.3 * parse(Int, n_periods))))

for s in sqrt_Ri
    tag  = "P4_T$(TL)_$(ri_case(s))"
    dir  = joinpath(HERE, "outputs", tag)
    mark = joinpath(dir, ".done")

    if isfile(mark)
        log("  $tag already complete — skipping"); push!(done_cases, tag); continue
    end
    if elapsed_h() + case_estimate_h > wall_hours
        log(@sprintf("  %s not started — %.2f h left of the %.1f h budget, a case needs ~%.1f h; re-submit to continue",
                     tag, wall_hours - elapsed_h(), wall_hours, case_estimate_h))
        push!(skipped_cases, tag); continue
    end

    log("run $tag  (N/ω = $s, Ri = $(parse(Float64, s)^2), $n_periods periods)")
    case_start = time()
    # LIGHT_OUTPUT drops the vorticity/x-y files and the 3D snapshots, i.e.
    # everything figures 4 and 5 never open, at no cost in runtime.
    ok = run_julia("Tidal3D.jl", [ri_case(s)],
                   Dict("PROFILE"      => "4",
                        "T_STRAT"      => T_strat,
                        "T_SWEEP"      => T_sweep,
                        "N_PERIODS"    => n_periods,
                        "LIGHT_OUTPUT" => "1",
                        "SPINUP_FILE"  => spin_fields),
                   "$tag.log")

    profiles = joinpath(dir, "TidalBL3D_$(tag)_profiles.jld2")
    if ok && isfile(profiles)
        # The marker, not the profiles file, is the completion test: a run cut
        # short by the wall clock leaves a valid but truncated profiles file.
        write(mark, string(now()))
        # The resume checkpoint is ~0.15 GB and is dead weight once a run ends.
        foreach(rm, filter(f -> occursin("_checkpoint", f),
                           readdir(dir, join = true)))
        push!(done_cases, tag)
        # Measured cost replaces the guess, so the budget check tightens as soon
        # as this node's actual throughput is known.
        global case_estimate_h = 1.1 * (time() - case_start) / 3600
        log(@sprintf("  %s done in %.2f h", tag, (time() - case_start) / 3600))
    else
        push!(failed_cases, tag)
        log("  $tag FAILED — see logs/$tag.log")
    end
end

log("simulations: $(length(done_cases)) complete, $(length(failed_cases)) failed, " *
    "$(length(skipped_cases)) not started")

# ---------------- 3. Figures ----------------
# Both scripts skip cases whose profiles file is missing (figure 4 renders them
# as empty panels), so this is worth running even after a partial sweep.
if skip_figures
    log("SKIP_FIGURES=1 — stopping before the figures")
elseif isempty(done_cases)
    log("no completed cases — nothing to plot")
else
    ri_list = join(sqrt_Ri, " ")
    # Figure 5 draws one curve per whole period listed here, four at a time to
    # match its colour ramp: the last four periods of the run, so the curves show
    # the state the mixed layer has actually reached.
    np = parse(Int, n_periods)
    periods_plot = join(max(1, np - 3):np, " ")
    fig_env = Dict("T_VALUES" => T_strat, "SQRT_RI" => ri_list,
                   "N_PERIODS_PLOT" => periods_plot)

    log("Figure 4 (T = $T_strat, N/ω ∈ {$ri_list})")
    run_julia("Figure4_metres.jl", String[], fig_env, "post_figures.log") ||
        log("  Figure4_metres.jl FAILED — see logs/post_figures.log")

    log("Figure 5 (one per case, periods $periods_plot)")
    run_julia("Figure5.jl", String[], fig_env, "post_figures.log") ||
        log("  Figure5.jl FAILED — see logs/post_figures.log")
end

log("=== DONE ===")
isempty(failed_cases)  || log("failed:      " * join(failed_cases, ", "))
isempty(skipped_cases) || log("not started: " * join(skipped_cases, ", "))
