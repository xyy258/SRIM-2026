# Batch driver for the K_T / TKE study, launched by swirles.sh as
#     srun julia --project=$PROJECT_DIR Stokes/3D/swirlesrun.jl
#
# run_moments_sweep.sh is the plain staged driver and is what to use
# interactively. This one adds the three things a batch allocation needs: a
# preflight that fails immediately rather than after hours, a completion marker
# per case so a re-submission resumes rather than restarts, and a wall-clock
# budget so a case that cannot finish is never started. Both drive the same
# scripts and share the marker files, so they can be used interchangeably.
#
# ---------------- The 12 h budget and the default run ----------------
# swirles.sh asks for 12 hours. Measured throughput on this grid is about
# 0.26 h per tidal period, and the moments writer adds roughly 15 %, so budget
# 0.30 h per period.
#
# The default is one column: T = 10 m, N/ω ∈ {0, 1, 2, 10}, 8 periods.
#
#   spin-up, 5 periods (no moments writer)               1.3 h
#   4 cases × 8 periods × 0.30 h                         9.6 h
#   post-processing                                      0.2 h
#                                                      -------
#                                                       11.1 h   in a 12 h wall
#
# That fits with about an hour to spare. If the node runs slowly the wall-clock
# guard may still defer the last case, which costs one short re-submission rather
# than a redo, since every case carries its own marker. The job also measures its
# own throughput after the first case and replaces the estimate above.
#
# T = 10 m comes from the previous sweep's figure 5 panels, archived under
# "-Softplus T sweep, no-slip bottom/figures/": the mixed-layer top reaches
# 12.5 m at N/ω = 2 and 10.4 m at N/ω = 10 by period 8. So at T = 10 the
# turbulent layer reaches the pycnocline in every case, and how far it entrains
# past it varies strongly with N/ω, which is the contrast K_T is measured from.
# Those runs had a no-slip bottom, and drag raises u* and deepens the layer, so
# stage 0 checks T against the measured h0 before the rest of the column is run
# (`t_vs_h0`, window 0.3 h0 to 2 h0, T_H0_FORCE=1 to override).
#
# Three things are cut from the original plan of 16 periods over a 4 × 5 grid of
# cases, which would be about 96 h:
#
#   T_VALUES   {2,3,5,8} → {10}   one column instead of a T-sweep. The K_T(TKE)
#                       exponent comes from varying N at fixed T, which is what
#                       a column gives; what is lost is the geometry test and
#                       stage 3 with it.
#   N_PERIODS  16 → 8   7 periods survive SKIP_PERIODS=1, which is enough for
#                       the boxcar, the phase bins and the log-log fit.
#   SQRT_RI    {0,1,2,5,10} → {0,1,2,10}
#                       drops 5, which lies between 2 and 10 on a log axis. The
#                       full decade in N and the N = 0 control both survive.
#
# Add a second column later with MOMENTS_STAGE=stage1 STAGE1_T=<T>; stages 2 and
# 3 become meaningful once more than one T is on disk.
#
# ---------------- Stages (MOMENTS_STAGE, default "auto") ----------------
#   spinup   turbulent spin-up with the drag bottom               ~1.3 h
#   stage0   the column's first case (N/ω = 0), h0 and the T check ~2.4 h
#   stage1   the rest of the column, or any other T               ~9.6 h
#   stage2   geometry: N/ω = 2 across several T (needs a T list)
#   stage3   fill: every (T, N/ω) not yet done (needs a T list)
#   post     re-run the post-processing only
#   h0       re-print h0 from the stage-0 case
#   auto     spin-up → stage 0 → h0 → the T check → the rest of the column
#
# `auto` runs the whole column, but only past the T check. Stage 0 is the
# column's own N/ω = 0 case, kept and marked, so nothing is paid for twice. If h0
# says T is above the turbulent layer, or deep inside it, the job stops there and
# reports the T to use instead, having spent one case rather than four.
#
# Panel (c) is the check that matters: K_T_bulk (from wb and F_sgs) and K_T_pe
# (from B alone) share no code, so if they disagree the subgrid flux is wrong and
# the panel (d) slope describes the closure rather than the flow. `auto` reports
# it at the end of the column and stage 3 refuses to start without it;
# STAGE3_FORCE=1 overrides.
#
# The bottom boundary condition is quadratic drag. Every run under outputs/
# predates that and used no-slip, so its spin-up is the wrong wall layer to
# restart from. The grid tag carries "_drag" and a fresh spin-up is built under
# its own name, leaving the existing runs untouched.
#
# ENV: MOMENTS_STAGE, T_VALUES, SQRT_RI, N_PERIODS, SPIN_PERIODS, WALL_HOURS,
#      CASE_HOURS, GRID_TAG, CD, Z_DRAG_REF, GRAD_FLOOR, SMOOTH, STAGE3_FORCE,
#      T_H0_FORCE, SKIP_PREFLIGHT, SKIP_POST, SKIP_SPIN_CHECK, SPIN_W_FLOOR.

using Printf, Dates
using JLD2          # the stage-3 gate reads `checks` from mixing_<tag>.jld2

const HERE = @__DIR__

const stage        = get(ENV, "MOMENTS_STAGE", "auto")
# One column by default: a single pycnocline height and four stratifications.
# See the header for where T = 10 m comes from; stage 0 checks it against the h0
# it measures before the rest of the column is run (see `t_vs_h0`).
const T_values     = split(get(ENV, "T_VALUES", "10"))
# 0.5 is dropped because Δb is tiny there and K_T = −∫F dz/Δb is then badly
# conditioned; 5 is dropped to fit the budget. 0 is kept, since b is a passive
# scalar there and it is the unstratified control. SQRT_RI="0 1 2 5 10" restores
# the full list.
const sqrt_Ri      = split(get(ENV, "SQRT_RI", "0 1 2 10"))
# 8 periods rather than 16, to fit the budget. 7 survive SKIP_PERIODS=1, which
# is enough for the boxcar and the log-log fit.
const n_periods    = get(ENV, "N_PERIODS", "8")
const spin_periods = get(ENV, "SPIN_PERIODS", "5")
const skip_post    = get(ENV, "SKIP_POST", "0") == "1"

const stage1_T = get(ENV, "STAGE1_T", String(first(T_values)))
const stage2_s = get(ENV, "STAGE2_S", "2")

# Stage 0 is the first case of the column rather than a throwaway run at some
# other T. It still exists to measure h0, but N/ω = 0 at the column's own T is
# both the unstratified control the column needs anyway and the cleanest run to
# read a turbulent-layer height from, since nothing suppresses the layer there.
# It carries a marker like any other case, so the column loop skips it.
const stage0_T, stage0_s = stage1_T, String(first(sqrt_Ri))
const stage0_periods = get(ENV, "N_PERIODS_STAGE0", n_periods)

# The grid is unchanged from the earlier sweeps but the bottom boundary condition
# is not, so the tag carries "_drag" and the old markers cannot make this job skip
# a case that needs redoing on a different wall layer.
const grid_tag = get(ENV, "GRID_TAG", "100x100x300_drag")

# 11.3 h rather than 11.0: the default column is 11.1 h of work in a 12 h wall,
# and a guard at 11.0 would defer the last case and cost an extra submission.
# This is a ceiling rather than a schedule — the job replaces its per-case
# estimate with a measurement after the first case.
const wall_hours   = parse(Float64, get(ENV, "WALL_HOURS", "11.3"))
const post_reserve = parse(Float64, get(ENV, "POST_RESERVE_H", "0.4"))

ri_case(s) = "sqrtRi" * replace(s, "." => "p")
t_lbl(t)   = replace(replace(t, r"\.0$" => ""), "." => "p")
tag_of(T, s) = "P4_T$(t_lbl(T))_$(ri_case(s))"

# One case keeps its 3D snapshots, as the only route to analysis this sweep did
# not plan for; every other case runs FIELDS3D=0 LIGHT_OUTPUT=1 and saves about
# 1.5 GB. It is the case both stage 1 and stage 2 pass through, derived from
# stage1_T and stage2_s rather than written out, since stage 0 may change T and a
# literal tag would then name a case that does not exist.
const fields3d_case = get(ENV, "FIELDS3D_CASE", tag_of(stage1_T, stage2_s))
marker_of(tag) = joinpath(HERE, "outputs", tag, ".done_moments_$grid_tag")
moments_of(tag) = joinpath(HERE, "outputs", tag, "TidalBL3D_$(tag)_moments.jld2")
mixing_of(tag)  = joinpath(HERE, "outputs", tag, "mixing_$(tag).jld2")

const started = time()
elapsed_h() = (time() - started) / 3600
log(msg) = (@printf("[%s | %.2f h] %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
                    elapsed_h(), msg); flush(stdout))

mkpath(joinpath(HERE, "logs"))

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

    # Append rather than truncate: logs/ is committed, so a file may already
    # hold earlier attempts and a resumed job appends to its own. Nothing a child
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
const preflight_code = """
using Pkg
Pkg.instantiate()
Pkg.precompile()
using Oceananigans, CUDA, JLD2, Plots
println("preflight: packages loaded, Oceananigans ", pkgversion(Oceananigans))

# The SGS diffusivity access path is the most likely thing in this study to break
# on a version bump, and it is cheap to check here rather than after a spin-up.
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
    log("preflight: instantiate + precompile + CUDA + SGS access path, see logs/preflight_moments.log")
    ok = run_julia(["-e", preflight_code],
                   Dict("JULIA_NUM_PRECOMPILE_TASKS" => "1"),
                   "preflight_moments.log"; threads = "1", label = "preflight")
    ok || error("Preflight failed — see logs/preflight_moments.log. If it is a Pkg or " *
                "precompile error, run `JULIA_NUM_PRECOMPILE_TASKS=1 julia --project=. " *
                "-e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'` once on a login " *
                "node. Nothing has been run yet.")
    log("preflight OK")
end

# ---------------- 1. Spin-up ----------------
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
                    merge(Dict("PROFILE"   => "4",
                               "RUN_TAG"   => spin_tag,
                               "T_STRAT"   => stage0_T,
                               "N_PERIODS" => spin_periods,
                               "LIGHT_OUTPUT" => "1",
                               "FIELDS3D"  => "1",
                               "MOMENTS"   => "0"),
                          passthrough("CD", "Z_DRAG_REF")),
                    "$spin_tag.log")
    ok || log("  spin-up returned an error — see logs/$spin_tag.log")
    # The checkpoint is no longer needed once the snapshot exists.
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
                "it, with SPIN_PERIODS raised. No cases have run. NOTE the drag bottom " *
                "may simply need longer than the no-slip one did to transition.")
end

# ---------------- 2. Cases ----------------
# The estimate is deliberately generous until the job measures its own
# throughput: starting a case that cannot finish wastes all of it, while
# declining one costs only a re-submission. 0.30 h per period is the 0.26 h
# measured on this grid plus about 15 % for the profiles Moments.jl adds.
case_estimate_h = parse(Float64, get(ENV, "CASE_HOURS", string(0.30 * parse(Int, n_periods))))
done_cases, failed_cases, skipped_cases = String[], String[], String[]

function run_case(Tv, s, np)
    tag  = tag_of(Tv, s)
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
    log("run $tag  (T = $Tv m, N/ω = $s, Ri = $(parse(Float64, s)^2), $np periods, FIELDS3D=$fields3d)")
    case_start = time()
    ok = run_script("Tidal3D.jl", [ri_case(s)],
                    merge(Dict("PROFILE"      => "4",
                               "T_STRAT"      => Tv,
                               "N_PERIODS"    => np,
                               "LIGHT_OUTPUT" => "1",
                               "FIELDS3D"     => fields3d,
                               "MOMENTS"      => "1",
                               "SPINUP_FILE"  => spin_fields),
                          passthrough("CD", "Z_DRAG_REF", "SHARP")),
                    "$tag.log")

    if ok && isfile(moments_of(tag))
        # The marker, not the moments file, is the test for completion: a run
        # cut short leaves a valid but truncated file behind.
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
function post(Tvs, ss)
    skip_post && (log("SKIP_POST=1 — no post-processing"); return)
    have = [(T, s) for T in Tvs, s in ss if isfile(moments_of(tag_of(T, s)))]
    isempty(have) && (log("no moments files to post-process yet"); return)
    log("post-processing $(length(have)) case(s) → mixing_<tag>.jld2 and figures/K_T_<tag>.png")
    ok = run_script("MixedLayerDiffusivity.jl", String[],
                    merge(Dict("T_VALUES" => join(Tvs, " "),
                               "N_OVER_OMEGA" => join(ss, " ")),
                          passthrough("GRAD_FLOOR", "SMOOTH", "SKIP_PERIODS", "RESULT_SUFFIX")),
                    "post_moments.log")
    ok || log("  post-processing returned an error — see logs/post_moments.log")
    # The verification block is the result, so print it into the job log rather
    # than leaving it in a child log.
    path = joinpath(HERE, "logs", "post_moments.log")
    if isfile(path)
        lines = readlines(path)
        i = findlast(l -> startswith(l, "VERIFICATION"), lines)
        i === nothing || foreach(println, lines[max(1, i-2):end])
        flush(stdout)
    end
end

# ---------------- 4. h0 ----------------
function report_h0()
    tag = tag_of(stage0_T, stage0_s)
    isfile(moments_of(tag)) || (log("no stage-0 moments file — h0 cannot be measured"); return NaN)
    run_julia([joinpath(HERE, "measure_h0.jl"), moments_of(tag)],
              passthrough("TKE_FRAC", "H_PEAK"), "h0_$tag.log";
              threads = "1", label = "measure_h0.jl")
    path = joinpath(HERE, "outputs", tag, "h0.txt")
    h0 = isfile(path) ? something(tryparse(Float64, strip(read(path, String))), NaN) : NaN
    println("\n" * "#"^78)
    if isnan(h0)
        println("#  h0 COULD NOT BE MEASURED. measure_h0.jl's own report follows; it falls")
        println("#  back on the mixed-layer height h and prints a T_VALUES line from it,")
        println("#  which is a usable but UNVERIFIED substitute. Full log: logs/h0_$tag.log")
        println("#"^78)
        lg = joinpath(HERE, "logs", "h0_$tag.log")
        if isfile(lg)
            lines = readlines(lg)
            i = findlast(l -> startswith(l, "h0 MEASUREMENT"), lines)
            i === nothing || foreach(println, lines[max(1, i-1):end])
        end
        println("#"^78)
    else
        @printf("#  h0 = %.3f m   (TKE down to 1 %% of its near-wall peak at peak phase)\n", h0)
        println("#")
        println("#  RE-PARAMETERISE STAGES 1-3 ON THIS before submitting them. Re-submit with")
        @printf("#      MOMENTS_STAGE=stage1 T_VALUES=\"%.1f %.1f %.1f %.1f\" sbatch swirles.sh\n",
                0.4h0, 0.7h0, 1.2h0, 2.0h0)
        println("#  A pycnocline above the turbulent layer is never reached, and the softplus")
        println("#  background is unstratified below z = T, so such a case measures nothing.")
    end
    println("#"^78 * "\n")
    flush(stdout)
    return h0
end

# ---------------- 4b. Is T within reach of the turbulence? ----------------
# The question stage 0 exists to answer. The background is unstratified below
# z = T, so a pycnocline the turbulent layer never reaches leaves Δb at its
# initial value and K_T = −∫F dz/Δb measures nothing. A pycnocline far below the
# top of the layer is destroyed in the first period or two instead, and Δb
# collapses towards zero, which is badly conditioned in the other direction.
#
# The usable window is roughly 0.3 h0 to 2 h0. Outside it the job says so and
# stops rather than spending the rest of the budget; T_H0_FORCE=1 overrides.
function t_vs_h0(h0)
    T = something(tryparse(Float64, String(stage1_T)), NaN)
    (isnan(h0) || isnan(T)) && return true      # nothing measured; report_h0 says so
    r = T / h0
    log(@sprintf("T / h0 = %.1f / %.1f = %.2f", T, h0, r))
    if r > 2.0
        log("  T IS ABOVE THE TURBULENT LAYER. The pycnocline is never reached, so every")
        log(@sprintf("  case in this column would measure nothing. Re-submit with T_VALUES=\"%.1f\".", h0))
        return get(ENV, "T_H0_FORCE", "0") == "1"
    elseif r < 0.3
        log("  T IS DEEP INSIDE THE TURBULENT LAYER. The pycnocline is wiped out early and Δb")
        log(@sprintf("  collapses, so K_T = −∫F dz/Δb is ill-conditioned. Re-submit with T_VALUES=\"%.1f\".", h0))
        return get(ENV, "T_H0_FORCE", "0") == "1"
    end
    log("  T sits inside the usable window 0.3 h0 - 2 h0 — the column is worth running")
    return true
end

# ---------------- 5. The stage-3 gate ----------------
function gate_ok()
    if get(ENV, "STAGE3_FORCE", "0") == "1"
        log("STAGE3_FORCE=1 — gate bypassed")
        return true
    end
    worst, nseen = 0.0, 0
    for s in sqrt_Ri
        tag = tag_of(stage1_T, s)
        f = mixing_of(tag)
        isfile(f) || continue
        c = try
            JLD2.jldopen(d -> d["checks"], f)
        catch err
            log("  could not read $f ($err)")
            continue
        end
        r = get(c, "pe_rel_diff", NaN)
        log(@sprintf("  %-20s K_T_bulk vs K_T_pe: %6.1f %%   K_sgs/K_T: %.2f",
                     tag, 100r, get(c, "K_sgs_over_K_T", NaN)))
        isfinite(r) || continue
        worst = max(worst, r)
        nseen += 1
    end
    nseen == 0 && (log("  no stage-1 results — run stage 1 and its post-processing first"); return false)
    log(@sprintf("  worst disagreement %.1f %% over %d case(s)", 100worst, nseen))
    return worst < 0.3
end

# ---------------- Drive ----------------
log("=== moments sweep: stage \"$stage\", grid+BC tag $grid_tag ===")
preflight()

if stage == "post"
    post(T_values, sqrt_Ri)
elseif stage == "h0"
    report_h0()
else
    do_spinup() || error("No spin-up snapshot at $spin_fields — see logs/$spin_tag.log. Nothing else has run.")
    check_spinup()

    if stage in ("spinup",)
        log("spin-up only — stopping")
    elseif stage in ("stage0", "auto")
        log("--- STAGE 0 (probe = first case of the column): T = $stage0_T m, N/ω = $stage0_s, $stage0_periods periods ---")
        run_case(stage0_T, stage0_s, stage0_periods)
        post([stage0_T], [stage0_s])
        h0 = report_h0()

        if stage == "stage0"
            t_vs_h0(h0)
            log("stage0 only — stopping. The case is kept and marked, so the column skips it.")
            log("  This is the UNSTRATIFIED CONTROL: at N/ω = 0 the b field is a passive")
            log("  scalar, so K_T is a genuine tracer diffusivity and checks 1-4 all apply,")
            log("  but K_T ~ TKE/N is undefined at N = 0 — the panel (d) slope here tests the")
            log("  machinery against the √TKE·l branch rather than discriminating between the")
            log("  two. Add the stratified cases with:  sbatch swirles.sh   (no arguments)")
        elseif !t_vs_h0(h0)
            log("auto: STOPPING. h0 says this T is the wrong height — see the lines above.")
            log("  The probe case is kept and marked; nothing else has run.")
        else
            log("--- continuing into the rest of the column: T = $stage1_T m, N/ω ∈ {$(join(sqrt_Ri, ", "))} ---")
            for s in sqrt_Ri; run_case(stage1_T, s, n_periods); end
            post([stage1_T], sqrt_Ri)
            log("--- panel (c) check (what a T-sweep would be gated on) ---")
            gate_ok() ? log("CHECK PASS — the two K_T routes agree; the panel (d) slope can be read as physics") :
                        log("CHECK FAIL — the two K_T routes disagree; do not read the slope until that is fixed")
        end
    elseif stage == "stage1"
        log("--- STAGE 1 (closure): T = $stage1_T m, N/ω ∈ {$(join(sqrt_Ri, ", "))}, $n_periods periods ---")
        log("    this is the stage the K_T(TKE) exponent comes from")
        for s in sqrt_Ri; run_case(stage1_T, s, n_periods); end
        post([stage1_T], sqrt_Ri)
        log("--- stage 1 gate (what stage 3 waits on) ---")
        gate_ok() ? log("GATE PASS — panel (c) agrees; stage 3 may be submitted") :
                    log("GATE FAIL — the two K_T routes disagree; fix the SGS flux before stage 3")
    elseif stage == "stage2"
        log("--- STAGE 2 (geometry): N/ω = $stage2_s, T ∈ {$(join(T_values, ", "))}, $n_periods periods ---")
        log("    the PE / geometry test: does delta_eff track h across T?")
        for T in T_values; run_case(T, stage2_s, n_periods); end
        post(T_values, [stage2_s])
    elseif stage == "stage3"
        if !gate_ok()
            error("STAGE 3 GATE FAILED. Stage 1's K_T_bulk and K_T_pe do not agree, so the " *
                  "subgrid flux is not trustworthy and 28 h of stage 3 would characterise " *
                  "the AMD closure rather than the flow. Fix that first, or set " *
                  "STAGE3_FORCE=1 if the disagreement is understood. Nothing has run.")
        end
        log("--- STAGE 3 (fill): the remaining $(length(T_values) * length(sqrt_Ri)) of the grid ---")
        for T in T_values, s in sqrt_Ri; run_case(T, s, n_periods); end
        post(T_values, sqrt_Ri)
    else
        error("Unknown MOMENTS_STAGE \"$stage\" — use spinup, stage0, stage1, stage2, stage3, post, h0 or auto")
    end
end

log(@sprintf("=== %s: %d complete, %d failed, %d not started ===",
             stage, length(done_cases), length(failed_cases), length(skipped_cases)))
isempty(failed_cases)  || log("  failed:      " * join(failed_cases, ", "))
isempty(skipped_cases) || log("  not started: " * join(skipped_cases, ", ") *
                              "  — re-submit swirles.sh to continue")
