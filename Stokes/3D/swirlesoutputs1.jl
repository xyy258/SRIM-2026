# Swirles animation driver: one x–z animation per case in the T = 30 m column.
#
# Runs Tidal3Danimation.jl for N/ω = √Ri ∈ {0, 0.5, 1, 2, 5, 10}, producing
#   outputs/P4_T30_sqrtRi<s>/animation_P4_T30_sqrtRi<s>.mp4
# two stacked panels each: u, and the thermal perturbation b' = b − b_bg(z).
#
# Launched by swirles2.sh as
#     srun julia --project=$PROJECT_DIR Stokes/3D/swirlesoutputs1.jl
# Post-processing only — it reads what the simulations already wrote and runs no
# model, so it needs NO GPU. `--gres=gpu:1` can come out of swirles2.sh to let it
# schedule sooner; the preflight below reports CUDA but does not require it.
#
# Structure mirrors swirlestestrun.jl deliberately: one child julia process per
# case (case_params.jl declares its parameters `const`, so a process can only
# ever set up one case), logs appended under a timestamped banner, and a .done
# marker per case so a job cut short by the wall clock resumes instead of
# skipping a half-written mp4.
#
# Knobs (all optional):
#   T_STRAT=30          which column to animate — must match the runs on disk
#   SQRT_RI="0 1 2"     subset of the N/ω cases
#   STRIDE=2            use every Nth frame (halves length and render time)
#   BPLIM=20            b' colour limit, in units of N²_ref·δ
#   WALL_HOURS=11.5     stop launching new cases past this
#   SKIP_PREFLIGHT=1    skip the environment check

using Printf, Dates

const HERE = @__DIR__

const T_strat = get(ENV, "T_STRAT", "30")
const sqrt_Ri = split(get(ENV, "SQRT_RI", "0 0.5 1 2 5 10"))
const stride  = get(ENV, "STRIDE", "1")
const bplim   = get(ENV, "BPLIM", "20")
const wall_hours = parse(Float64, get(ENV, "WALL_HOURS", "11.5"))

t_lbl(t)   = replace(replace(t, r"\.0$" => ""), "." => "p")
ri_case(s) = "sqrtRi" * replace(s, "." => "p")
const TL = t_lbl(T_strat)

const started = time()
elapsed_h() = (time() - started) / 3600
log(msg) = (@printf("[%s | %.2f h] %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
                    elapsed_h(), msg); flush(stdout))

mkpath(joinpath(HERE, "logs"))

# One child process. `dir = HERE` makes the relative outputs/ path in
# case_params.jl resolve here; GKSwstype = 100 keeps GR headless on a node with
# no display, without which every frame fails.
function run_julia(payload::Vector{String}, env::Dict{String,String}, logfile;
                   threads = "auto", label = first(payload))
    cmd = Cmd([joinpath(Sys.BINDIR, "julia"), "--project=$HERE", "-t", threads,
               payload...])
    full = merge(Dict{String,String}(ENV), Dict("GKSwstype" => "100"), env)
    path = joinpath(HERE, "logs", logfile)

    # APPEND, never truncate: logs/ is committed, so these names already hold
    # earlier runs. Nothing a child prints carries a wall-clock date, so the
    # banner is what separates one attempt from the next —
    # `grep -n '^==== ' logs/<file>` lists them, newest last.
    open(path, "a") do io
        println(io, "\n", "="^78)
        println(io, "==== ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "  ", label)
        println(io, "==== ", join(("$k=$v" for (k, v) in sort(collect(env))), "  "))
        println(io, "="^78)
        flush(io)
    end

    return success(pipeline(setenv(cmd, full; dir = HERE),
                            stdout = path, stderr = path, append = true))
end

run_script(script, args::Vector{String}, env::Dict{String,String}, logfile) =
    run_julia([joinpath(HERE, script), args...], env, logfile;
              label = script * " " * join(args, " "))

# ---------------- 0. Preflight ----------------
# Rendering is the whole job here, so the check actually ENCODES a throwaway
# clip rather than just loading Plots: that exercises GR headless plus the
# FFMPEG artifact, which is the pair that fails on a fresh depot. Finding out
# now costs ten seconds; finding out from mp4() costs a full render first.
const preflight_code = """
using Pkg
Pkg.instantiate()
Pkg.precompile()
using Oceananigans, JLD2, Plots
println("preflight: packages loaded")

# No model is built here, so a GPU is not needed — report it and carry on.
try
    using CUDA
    println("preflight: CUDA ", CUDA.functional() ? "functional (unused for animations)" :
                                                    "not functional (fine — no GPU needed)")
catch err
    println("preflight: CUDA unavailable (fine — no GPU needed): ", err)
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
    log("preflight: instantiate + precompile + test encode, see logs/preflight_anim.log")
    ok = run_julia(["-e", preflight_code],
                   Dict("JULIA_NUM_PRECOMPILE_TASKS" => "1"),
                   "preflight_anim.log"; threads = "1", label = "preflight (animation)")
    ok || error("Preflight failed — see logs/preflight_anim.log. If it is a Pkg or " *
                "precompile error, run `JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. " *
                "-e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'` once on a login " *
                "node. Nothing has been rendered yet.")
    log("preflight OK")
end

# ---------------- Animations, one per case ----------------
done_cases, failed_cases, skipped_cases, missing_cases = String[], String[], String[], String[]

# Rendered 1601 frames (8 periods × 200/period) in a few minutes per case last
# time; generous, and replaced by a measurement after the first case.
anim_estimate_h = parse(Float64, get(ENV, "ANIM_HOURS", "0.25"))

for s in sqrt_Ri
    tag  = "P4_T$(TL)_$(ri_case(s))"
    dir  = joinpath(HERE, "outputs", tag)
    mark = joinpath(dir, ".anim_done")
    mp4f = joinpath(dir, "animation_$tag.mp4")

    # The x–z slice file, NOT the profiles file the figures read. LIGHT_OUTPUT=1
    # keeps it (u and b only), so the production runs have it — but a case that
    # never ran, or ran with the slice writer disabled, would otherwise fail
    # deep inside FieldTimeSeries with a much less obvious message.
    slices = joinpath(dir, "TidalBL3D_$tag.jld2")
    if !isfile(slices)
        log("  $tag has no x–z slice file ($(basename(slices))) — nothing to animate")
        push!(missing_cases, tag); continue
    end
    if isfile(mark)
        log("  $tag already animated — skipping"); push!(done_cases, tag); continue
    end
    if elapsed_h() + anim_estimate_h > wall_hours
        log(@sprintf("  %s not started — %.2f h left of the %.1f h budget; re-submit to continue",
                     tag, wall_hours - elapsed_h(), wall_hours))
        push!(skipped_cases, tag); continue
    end

    log("animate $tag  (N/ω = $s)")
    case_start = time()
    # PROFILE and T_STRAT must match the run: they select the output directory
    # AND rebuild the b_bg(z) that gets subtracted to form b'. Wrong values here
    # animate the wrong case, or subtract the wrong background from the right one.
    ok = run_script("Tidal3Danimation.jl", [ri_case(s)],
                    Dict("PROFILE" => "4",
                         "T_STRAT" => T_strat,
                         "STRIDE"  => stride,
                         "BPLIM"   => bplim),
                    "anim_$tag.log")

    if ok && isfile(mp4f)
        write(mark, string(now()))
        push!(done_cases, tag)
        global anim_estimate_h = 1.5 * (time() - case_start) / 3600
        log(@sprintf("  %s done in %.2f h (%.1f MB)", tag, (time() - case_start) / 3600,
                     filesize(mp4f) / 1e6))
    else
        push!(failed_cases, tag)
        log("  $tag FAILED — see logs/anim_$tag.log")
    end
end

log("animations: $(length(done_cases)) complete, $(length(failed_cases)) failed, " *
    "$(length(missing_cases)) with no slice data, $(length(skipped_cases)) not started")
isempty(failed_cases)  || log("failed:      " * join(failed_cases, ", "))
isempty(missing_cases) || log("no data:     " * join(missing_cases, ", "))
isempty(skipped_cases) || log("not started: " * join(skipped_cases, ", "))

# The mp4s live under outputs/, which is gitignored (and *.mp4 is ignored too),
# so they will NOT arrive on a workstation via git pull — print the copy command
# rather than leaving them to be hunted for.
if !isempty(done_cases)
    log("=== DONE — $(length(done_cases)) animation(s) under outputs/<tag>/ ===")
    log("copy them off the cluster with:")
    log("  rsync -av --include='*/' --include='animation_*.mp4' --exclude='*' \\")
    log("    <user>@<cluster>:$(joinpath(HERE, "outputs"))/ ./animations/")
else
    log("=== DONE — nothing rendered ===")
end
