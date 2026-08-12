# Swirles run driver #3: the softplus (profile 4) T = 5 m and T = 10 m columns,
# 8 tidal periods, with figures and animations.
#
# These are the two T values where the pycnocline actually sits inside the
# turbulent boundary layer (δ = u*/f ≈ 9.2 m), so the mixed layer erodes it and
# was still growing at period 4 in the earlier runs: h_m gained +3.5 m in the
# last period at T = 5, N/ω = 0, and +1.5 m at T = 10, N/ω = 1. T = 20 and 30
# are not repeated here — their pycnocline is above the boundary layer's reach,
# h_m never leaves its initial value, and the 8-period T = 30 run confirmed it
# (periods 5–8 lie on top of one another).
#
# Runs, in order:
#   1. per T: an Ri = 0 spin-up on that T's grid (skipped if its snapshot exists)
#   2. per T: the six cases N/ω = √Ri ∈ {0, 0.5, 1, 2, 5, 10}, 8 periods each
#   3. Figure4_metres.jl → figures/Figure4_softplus_T5.png and ..._T10.png
#   4. Figure5.jl        → figures/Figure5_P4_T<T>_sqrtRi<s>.png, one per run
#   5. Tidal3Danimation.jl → outputs/<tag>/animation_<tag>.mp4, one per run
#
# Launched by swirles3.sh as
#     srun julia --project=$PROJECT_DIR Stokes/3D/swirlesrun3.jl
# Budget on an A100, measured from the T = 30 job: ~0.47 h per case, ~0.3 h per
# spin-up, ~0.05 h per animation → 2 spin-ups + 12 cases + 12 animations ≈ 7 h,
# inside the 12 h allocation with room to spare.
#
# T IS THE OUTER LOOP and each T gets a vertical grid refined at its own
# pycnocline (T_SWEEP = that T, giving Nz ≈ 254–290 rather than 458). A spin-up
# snapshot only restarts onto an identical grid, so one spin-up per T is forced
# by that choice; interleaving T would mean redoing them. It also means a
# difference between T = 5 and T = 10 is not purely physical — wall spacing and
# pycnocline resolution are matched, but the grids differ in how coarsely they
# cover regions with no feature in them.
#
# RESUMABLE. Each case writes .done and each animation .anim_done only after the
# child exits cleanly, so a job killed by the wall clock redoes that item rather
# than skipping a truncated one. Re-submit swirles3.sh to continue.
#
# Knobs (all optional):
#   T_VALUES="5 10"     which columns to run
#   SQRT_RI="0 1 2"     subset of the N/ω cases
#   N_PERIODS=8         tidal periods per production case
#   SPIN_PERIODS=5      tidal periods of spin-up
#   WALL_HOURS=11       stop launching new simulations past this
#   SKIP_ANIM=1         simulations and figures only
#   SKIP_FIGURES=1      no figures
#   SKIP_PREFLIGHT=1    skip the environment check

using Printf, Dates

const HERE = @__DIR__

const T_values     = split(get(ENV, "T_VALUES", "5 10"))
const sqrt_Ri      = split(get(ENV, "SQRT_RI", "0 0.5 1 2 5 10"))
const n_periods    = get(ENV, "N_PERIODS", "8")
const spin_periods = get(ENV, "SPIN_PERIODS", "5")
const skip_anim    = get(ENV, "SKIP_ANIM", "0") == "1"
const skip_figures = get(ENV, "SKIP_FIGURES", "0") == "1"

# Slurm kills the job at --time; hold back enough for the figures and the
# animations, which are the deliverables and are cheap compared with a case.
const wall_hours   = parse(Float64, get(ENV, "WALL_HOURS", "11.0"))
const post_reserve = parse(Float64, get(ENV, "POST_RESERVE_H", "0.8"))

ri_case(s) = "sqrtRi" * replace(s, "." => "p")
t_lbl(t)   = replace(replace(t, r"\.0$" => ""), "." => "p")
tag_of(T, s) = "P4_T$(t_lbl(T))_$(ri_case(s))"

const started = time()
elapsed_h() = (time() - started) / 3600
log(msg) = (@printf("[%s | %.2f h] %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
                    elapsed_h(), msg); flush(stdout))

mkpath(joinpath(HERE, "logs"))

# One child process per step. case_params.jl declares the case parameters
# `const` (GPU kernels capture them), so a process can only ever set up one
# case. `dir = HERE` makes the relative outputs/ path resolve here; GKSwstype
# keeps GR headless on a node with no display.
# `env` is AbstractDict, not Dict{String,String}: T_VALUES and SQRT_RI are split
# into SubStrings, so a dict built from them is Dict{String,AbstractString} and
# would not match a concrete signature. Values are converted on the way in.
function run_julia(payload::AbstractVector{<:AbstractString}, env::AbstractDict, logfile;
                   threads = "auto", label = first(payload))
    # String.() throughout: case names come from splitting T_VALUES/SQRT_RI and
    # arrive as SubStrings, which Cmd and the env dict will not take directly.
    cmd = Cmd([joinpath(Sys.BINDIR, "julia"), "--project=$HERE", "-t", threads,
               String.(payload)...])
    envs = Dict{String,String}(String(k) => String(v) for (k, v) in env)
    full = merge(Dict{String,String}(ENV), Dict("GKSwstype" => "100"), envs)
    path = joinpath(HERE, "logs", logfile)

    # APPEND, never truncate: logs/ is committed, so these names already hold
    # earlier runs, and a resumed job appends to its own earlier attempt.
    # Nothing a child prints carries a wall-clock date, so this banner is what
    # separates attempts — `grep -n '^==== ' logs/<file>` lists them, newest last.
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
# Checks everything this job needs BEFORE spending hours: the environment
# instantiates, CUDA works (the simulations need a GPU), and mp4 encoding works
# (the animations need GR headless plus the FFMPEG artifact). Serial, because
# parallel precompilation contends on pidfile locks over cephfs.
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

anim = @animate for i in 1:3
    plot([0, 1], [0, i], legend = false)
end
out = mp4(anim, joinpath(tempdir(), "preflight_\$(getpid()).mp4"), fps = 2)
println("preflight: mp4 encoding works")
rm(out.filename, force = true)
"""

if get(ENV, "SKIP_PREFLIGHT", "0") == "1"
    log("SKIP_PREFLIGHT=1 — not checking the environment")
else
    log("preflight: instantiate + precompile + CUDA + test encode, see logs/preflight3.log")
    ok = run_julia(["-e", preflight_code],
                   Dict("JULIA_NUM_PRECOMPILE_TASKS" => "1"),
                   "preflight3.log"; threads = "1", label = "preflight (run 3)")
    ok || error("Preflight failed — see logs/preflight3.log. If it is a Pkg or precompile " *
                "error, run `JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. -e 'using Pkg; " *
                "Pkg.instantiate(); Pkg.precompile()'` once on a login node. " *
                "Nothing has been run yet.")
    log("preflight OK")
end

# ---------------- 1–2. Spin-up and cases, per T ----------------
done_cases, failed_cases, skipped_cases = String[], String[], String[]

# Measured at 0.47 h on the A100 for an 8-period case; deliberately generous
# until this job measures its own throughput. Starting a case that cannot finish
# wastes all of it, whereas declining one costs only a re-submission.
case_estimate_h = parse(Float64, get(ENV, "CASE_HOURS", string(0.075 * parse(Int, n_periods))))

for Tv in T_values
    TL = t_lbl(Tv)
    # This block's grid: refined at THIS T only (see the header note).
    spin_tag    = "spinup_T$TL"
    spin_fields = joinpath("outputs", spin_tag, "TidalBL3D_$(spin_tag)_fields.jld2")

    if isfile(joinpath(HERE, spin_fields))
        log("T = $Tv: spin-up snapshot already present — skipping")
    else
        log("T = $Tv: spin-up (Ri0 from rest, $spin_periods periods)")
        # FIELDS3D=1 is mandatory: the 3D snapshot IS the spin-up's product, and
        # LIGHT_OUTPUT would otherwise switch it off.
        spin_ok = run_script("Tidal3D.jl", ["Ri0"],
                             Dict("PROFILE"   => "4",
                                  "RUN_TAG"   => spin_tag,
                                  "T_STRAT"   => Tv,
                                  "T_SWEEP"   => Tv,
                                  "N_PERIODS" => spin_periods,
                                  "FIELDS3D"  => "1"),
                             "$spin_tag.log")
        spin_ok || log("  T = $Tv: spin-up returned an error — see logs/$spin_tag.log")
        # Its checkpoint is ~0.15 GB of dead weight once the snapshot exists.
        spin_ok && foreach(rm, filter(f -> occursin("_checkpoint", f),
                                      readdir(joinpath(HERE, "outputs", spin_tag), join = true)))
    end

    # Without the snapshot every case starts from rest with noise (Tidal3D.jl
    # warns and carries on), so most of the run would be spin-up rather than the
    # stratified evolution the figures are about. Skip the column instead.
    if !isfile(joinpath(HERE, spin_fields))
        log("  T = $Tv: no spin-up snapshot — skipping this column")
        append!(skipped_cases, [tag_of(Tv, s) for s in sqrt_Ri])
        continue
    end

    for s in sqrt_Ri
        tag  = tag_of(Tv, s)
        dir  = joinpath(HERE, "outputs", tag)
        mark = joinpath(dir, ".done")

        if isfile(mark)
            log("  $tag already complete — skipping"); push!(done_cases, tag); continue
        end
        if elapsed_h() + case_estimate_h > wall_hours - post_reserve
            # One literal format string: @sprintf resolves at macro-expansion
            # time, so a `*`-concatenated format is not a format string at all.
            log(@sprintf("  %s not started — %.2f h left of the budget (%.1f h minus %.1f h held for figures/animations), a case needs ~%.1f h",
                         tag, wall_hours - post_reserve - elapsed_h(), wall_hours,
                         post_reserve, case_estimate_h))
            push!(skipped_cases, tag); continue
        end

        log("run $tag  (N/ω = $s, Ri = $(parse(Float64, s)^2), $n_periods periods)")
        case_start = time()
        # LIGHT_OUTPUT drops the vorticity/x-y files and the 3D snapshots — all
        # of which figures 4/5 and the animation never open — at no cost in
        # runtime. It KEEPS the x–z slices, which the animations read.
        ok = run_script("Tidal3D.jl", [ri_case(s)],
                        Dict("PROFILE"      => "4",
                             "T_STRAT"      => Tv,
                             "T_SWEEP"      => Tv,
                             "N_PERIODS"    => n_periods,
                             "LIGHT_OUTPUT" => "1",
                             "SPINUP_FILE"  => spin_fields),
                        "$tag.log")

        if ok && isfile(joinpath(dir, "TidalBL3D_$(tag)_profiles.jld2"))
            # The marker, not the profiles file, is the completion test: a run
            # cut short leaves a valid but truncated profiles file behind.
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
end

log("simulations: $(length(done_cases)) complete, $(length(failed_cases)) failed, " *
    "$(length(skipped_cases)) not started")

# ---------------- 3–4. Figures ----------------
# Both scripts skip cases whose profiles file is missing, so these are worth
# running after a partial sweep too.
if skip_figures
    log("SKIP_FIGURES=1 — stopping before the figures")
elseif isempty(done_cases)
    log("no completed cases — nothing to plot")
else
    ri_list = join(sqrt_Ri, " ")
    T_list  = join(T_values, " ")
    np = parse(Int, n_periods)
    # Figure 5 draws one curve per whole period listed, four at a time to match
    # its colour ramp: the LAST four periods, i.e. the state actually reached.
    periods_plot = join(max(1, np - 3):np, " ")
    fig_env = Dict("T_VALUES" => T_list, "SQRT_RI" => ri_list,
                   "N_PERIODS_PLOT" => periods_plot)

    # One invocation covers both columns and writes one figure per T. SKIP_SWEEP
    # suppresses the combined all-T overview, which would otherwise be rebuilt
    # from T = 5 and 10 alone and replace the existing four-T version.
    log("Figure 4 (one per T ∈ {$T_list}, N/ω ∈ {$ri_list})")
    run_script("Figure4_metres.jl", String[],
               merge(fig_env, Dict("SKIP_SWEEP" => "1")), "post_figures3.log") ||
        log("  Figure4_metres.jl FAILED — see logs/post_figures3.log")

    log("Figure 5 (one per case, periods $periods_plot)")
    run_script("Figure5.jl", String[], fig_env, "post_figures3.log") ||
        log("  Figure5.jl FAILED — see logs/post_figures3.log")
end

# ---------------- 5. Animations ----------------
anim_done, anim_failed = String[], String[]

if skip_anim
    log("SKIP_ANIM=1 — stopping before the animations")
else
    for tag in done_cases
        dir  = joinpath(HERE, "outputs", tag)
        mark = joinpath(dir, ".anim_done")
        mp4f = joinpath(dir, "animation_$tag.mp4")
        # The x–z slice file, not the profiles file the figures read.
        isfile(joinpath(dir, "TidalBL3D_$tag.jld2")) || continue
        isfile(mark) && (log("  $tag already animated — skipping");
                         push!(anim_done, tag); continue)

        # T_STRAT must match the run: it selects the directory AND rebuilds the
        # b_bg(z) that is subtracted to form b'. The tag carries it, so recover
        # it from there rather than from the loop variable.
        Tv = match(r"^P4_T([0-9p]+)_", tag)[1]
        log("animate $tag")
        ok = run_script("Tidal3Danimation.jl", [split(tag, "_")[end]],
                        Dict("PROFILE" => "4",
                             "T_STRAT" => replace(Tv, "p" => "."),
                             "STRIDE"  => get(ENV, "STRIDE", "1"),
                             "BPLIM"   => get(ENV, "BPLIM", "20")),
                        "anim_$tag.log")
        if ok && isfile(mp4f)
            write(mark, string(now()))
            push!(anim_done, tag)
            log(@sprintf("  %s animated (%.1f MB)", tag, filesize(mp4f) / 1e6))
        else
            push!(anim_failed, tag)
            log("  $tag animation FAILED — see logs/anim_$tag.log")
        end
    end
    log("animations: $(length(anim_done)) complete, $(length(anim_failed)) failed")
end

# ---------------- Summary ----------------
log("=== DONE ===")
isempty(failed_cases)  || log("failed:      " * join(failed_cases, ", "))
isempty(skipped_cases) || log("not started: " * join(skipped_cases, ", "))
isempty(anim_failed)   || log("animations failed: " * join(anim_failed, ", "))

# Figures are tracked by git and come across with a commit; the mp4s are under
# outputs/ and are gitignored, so they need rsync. Print the command rather than
# leaving it to be looked up.
if !isempty(anim_done)
    log("figures: commit figures/ and push, then pull on the workstation")
    log("animations: rsync them off the cluster with")
    log("  rsync -av --include='*/' --include='animation_*.mp4' --exclude='*' \\")
    log("    tll46@sw-mn01:$(joinpath(HERE, "outputs"))/ <local>/Stokes/3D/outputs/")
end
