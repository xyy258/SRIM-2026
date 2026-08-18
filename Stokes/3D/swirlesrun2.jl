# swirlesrun2.jl — RE-RUN THE POST-PROCESSING ONLY, on the fixed analysis code.
#
# Tidal (Stokes) cases only.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------------------------------------------------------------------
# The 18 Aug sweep simulated all four cases correctly, but its post-processing
# step ran against a MixedLayerDiffusivity.jl that predated two guards:
#
#   * R_MIN  — a correlation floor below which the panel (d) slope is not
#     quotable as an exponent. A least-squares slope through a round cloud is
#     still a number, and with n in the thousands its formal error bar is small,
#     which is exactly how a null result gets written up as a finding.
#   * the N/ω = 0 branch — at zero stratification K_T ~ TKE/N is UNDEFINED, so
#     that case cannot discriminate between the two scalings at all. It is the
#     unstratified control, not a measurement.
#
# The cluster checkout was made at 15:47 and the guards were pushed at 15:58, so
# logs/post_moments.log reports N/ω = 0 (slope +0.17, r = +0.23) as
# "all checks pass; the slope above can be read as physics" and picks a nearest
# scaling for it. Both statements are wrong, and neither is a numerical error —
# the fit itself is fine, the interpretation on top of it was not guarded.
#
# So this job re-derives nothing about the physics. It re-runs the SAME analysis
# on the SAME moments files with the fixed code, and the slopes are expected to
# come back IDENTICAL. What changes is which of them carry a caveat. The script
# checks that expectation explicitly at the end: if a slope moves, something
# other than the guards changed and the run needs looking at, not quoting.
#
# ---------------------------------------------------------------------------
# WHAT IT READS AND WRITES
# ---------------------------------------------------------------------------
# Reads   outputs/<tag>/TidalBL3D_<tag>_moments.jld2   (already on the cluster)
# Writes  outputs/<tag>/mixing_<tag>_fixed.jld2
#         figures/K_T_<tag>_fixed.png
#         logs/post_moments_fixed.log
#
# RESULT_SUFFIX defaults to "_fixed" so nothing from the first pass is
# overwritten and the two can be compared side by side. Set RESULT_SUFFIX="" to
# replace the originals in place once you are happy with these.
#
# No simulation, no spin-up, no GPU work — this is minutes, not hours. The 12 h
# wall in swirles2.sh is inherited from the simulation job; dropping it to
# --time=00:30:00 will clear the queue considerably faster.
#
#     sbatch Stokes/3D/swirles2.sh
#
# Knobs: T_VALUES, SQRT_RI/N_OVER_OMEGA, RESULT_SUFFIX, GRAD_FLOOR, SMOOTH,
#        SKIP_PERIODS, R_MIN, OLD_LOG
# ---------------------------------------------------------------------------

using Printf, Dates

const HERE = @__DIR__

const T_values = split(get(ENV, "T_VALUES", "10"))
const sqrt_Ri  = split(get(ENV, "SQRT_RI", get(ENV, "N_OVER_OMEGA", "0 1 2 10")))

# Non-empty by default: the brief forbids overwriting anything under outputs/,
# and the first pass's figures are what the caveats need to be compared against.
const SUFFIX  = get(ENV, "RESULT_SUFFIX", "_fixed")
const OLD_LOG = get(ENV, "OLD_LOG", joinpath(HERE, "logs", "post_moments.log"))
const NEW_LOG = "post_moments_fixed.log"

ri_case(s)   = "sqrtRi" * replace(s, "." => "p")
t_lbl(t)     = replace(replace(t, r"\.0$" => ""), "." => "p")
tag_of(T, s) = "P4_T$(t_lbl(T))_$(ri_case(s))"
moments_of(tag) = joinpath(HERE, "outputs", tag, "TidalBL3D_$(tag)_moments.jld2")

const started = time()
log(msg) = (@printf("[%s | %.2f h] %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
                    (time() - started) / 3600, msg); flush(stdout))

mkpath(joinpath(HERE, "logs"))

# ---------------------------------------------------------------------------
# 1. Prove the fixed code is the code that will run.
# ---------------------------------------------------------------------------
# This is the whole reason the first pass produced unquotable verdicts, so it is
# checked before anything is spent rather than discovered afterwards in a log.
# A textual check on the source, not a version string: the guards either are in
# the file this job is about to run or they are not.
function assert_fixed()
    src_path = joinpath(HERE, "MixedLayerDiffusivity.jl")
    isfile(src_path) || error("MixedLayerDiffusivity.jl not found at $src_path")
    src = read(src_path, String)

    # Contiguous substrings only. An earlier version of this check looked for
    # "CANNOT discriminate", which is split across a line break in the source and
    # so never matched — the guard then reported the fixed file as stale.
    needed = ["R_MIN"                        => "the correlation floor",
              "unstratified = c.Ri == 0"     => "the N/\u03c9 = 0 branch",
              "weak_fit = isfinite(c.rcorr)" => "the weak-fit branch",
              "is UNDEFINED (N = 0)"         => "the N/\u03c9 = 0 verdict text",
              "WEAK FIT: r ="                => "the weak-correlation verdict text"]
    missing = [what for (marker, what) in needed if !occursin(marker, src)]

    if !isempty(missing)
        println()
        println("="^78)
        println("STALE CHECKOUT — the fixes are NOT in MixedLayerDiffusivity.jl.")
        println("Missing: ", join(missing, ", "))
        println()
        println("This job would reproduce the first pass's verdicts exactly and")
        println("waste the allocation. On the cluster, in the project directory:")
        println("    git pull")
        println("then resubmit. Nothing has been written.")
        println("="^78)
        error("MixedLayerDiffusivity.jl predates the verification fixes")
    end

    log("MixedLayerDiffusivity.jl carries all four guard markers — OK")

    # Record provenance in the job log so this pass can never be ambiguous the
    # way the first one was.
    try
        head  = strip(read(`git -C $HERE rev-parse --short HEAD`, String))
        dirty = !isempty(strip(read(`git -C $HERE status --porcelain -- MixedLayerDiffusivity.jl`, String)))
        log("analysis code at commit $head" * (dirty ? "  (UNCOMMITTED local edits)" : ""))
    catch
        log("git provenance unavailable (not a checkout?) — continuing")
    end
end

# ---------------------------------------------------------------------------
# 2. Which cases actually have data.
# ---------------------------------------------------------------------------
function available_cases()
    want = [(T, s) for T in T_values for s in sqrt_Ri]
    have = [(T, s) for (T, s) in want if isfile(moments_of(tag_of(T, s)))]
    gone = setdiff(want, have)

    for (T, s) in gone
        log("MISSING  $(tag_of(T, s)) — no moments file, skipped")
    end
    isempty(have) && error("no moments files found under outputs/ — nothing to post-process")
    log("$(length(have)) case(s) to re-analyse: " * join((tag_of(T, s) for (T, s) in have), ", "))
    return have
end

# ---------------------------------------------------------------------------
# 3. Run it.
# ---------------------------------------------------------------------------
function run_post(cases)
    Tvs = unique(first.(cases))
    ss  = unique(last.(cases))

    env = Dict("T_VALUES"      => join(Tvs, " "),
               "N_OVER_OMEGA"  => join(ss, " "),
               "RESULT_SUFFIX" => SUFFIX,
               "GKSwstype"     => "100")
    for k in ("GRAD_FLOOR", "SMOOTH", "SKIP_PERIODS", "R_MIN")
        haskey(ENV, k) && (env[k] = ENV[k])
    end

    path = joinpath(HERE, "logs", NEW_LOG)
    open(path, "a") do io       # append, never truncate — logs/ is committed
        println(io, "\n", "="^78)
        println(io, "==== ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
                    "  MixedLayerDiffusivity.jl (re-run on fixed code)")
        println(io, "==== ", join(("$k=$v" for (k, v) in sort(collect(env))), "  "))
        println(io, "="^78)
    end

    cmd = Cmd([joinpath(Sys.BINDIR, "julia"), "--project=$HERE", "-t", "auto",
               joinpath(HERE, "MixedLayerDiffusivity.jl")])
    full = merge(Dict{String,String}(ENV), env)

    log("running MixedLayerDiffusivity.jl → logs/$NEW_LOG")
    ok = success(pipeline(setenv(cmd, full; dir = HERE),
                          stdout = path, stderr = path, append = true))
    ok || log("  post-processing returned an error — see logs/$NEW_LOG")
    return ok, path
end

# ---------------------------------------------------------------------------
# 4. Surface the VERIFICATION block, and compare against the first pass.
# ---------------------------------------------------------------------------
const SLOPE_RE = r"slope .*? z = h = ([+-][0-9.]+)\s+\(r = ([+-][0-9.]+), n = ([0-9]+)\)"
const TAG_RE   = r"^VERIFICATION\s+(\S+)"

# Last occurrence per tag wins: these logs are appended to, so a file may hold
# several attempts and only the newest describes the code that just ran.
function slopes_in(path)
    out = Dict{String,NTuple{3,Float64}}()
    isfile(path) || return out
    tag = ""
    for line in eachline(path)
        m = match(TAG_RE, line)
        m !== nothing && (tag = m.captures[1]; continue)
        m = match(SLOPE_RE, line)
        if m !== nothing && !isempty(tag)
            out[tag] = (parse(Float64, m.captures[1]),
                        parse(Float64, m.captures[2]),
                        parse(Float64, m.captures[3]))
        end
    end
    return out
end

function report(new_path)
    lines = isfile(new_path) ? readlines(new_path) : String[]
    i = findlast(l -> startswith(l, "VERIFICATION"), lines)
    if i !== nothing
        println()
        foreach(println, lines[max(1, i - 2):end])
    end

    old, new = slopes_in(OLD_LOG), slopes_in(new_path)
    isempty(old) && (log("no first-pass log at $OLD_LOG — nothing to compare against"); return)

    println()
    println("="^78)
    println("SLOPE COMPARISON — first pass vs fixed code")
    println("="^78)
    println("The guards are interpretive: they change which slopes may be quoted,")
    println("not the slopes themselves. Every row should read UNCHANGED.")
    println()
    @printf("  %-22s %8s %8s   %8s %8s\n",
            "case", "old", "old r", "new", "new r")
    moved = String[]
    for tag in sort(collect(keys(new)))
        if haskey(old, tag)
            (so, ro, _) = old[tag]
            (sn, rn, _) = new[tag]
            same = isapprox(so, sn; atol = 0.005) && isapprox(ro, rn; atol = 0.005)
            same || push!(moved, tag)
            @printf("  %-22s %+8.2f %+8.2f   %+8.2f %+8.2f   %s\n",
                    tag, so, ro, sn, rn, same ? "unchanged" : "*** MOVED ***")
        else
            (sn, rn, _) = new[tag]
            @printf("  %-22s %8s %8s   %+8.2f %+8.2f   %s\n",
                    tag, "-", "-", sn, rn, "new")
        end
    end
    println()
    if isempty(moved)
        println("  All slopes reproduced. The re-run differs from the first pass only in")
        println("  the caveats printed above, which is what was intended.")
    else
        println("  *** ", join(moved, ", "), " MOVED.")
        println("  The guards cannot shift a fit. Something else differs between the two")
        println("  passes — check GRAD_FLOOR, SMOOTH and SKIP_PERIODS in both log headers")
        println("  before quoting either number.")
    end
    println("="^78)
end

# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------
log("=== post-processing re-run on fixed analysis code (tidal cases only) ===")
log("RESULT_SUFFIX = \"$SUFFIX\"" * (isempty(SUFFIX) ?
    "  — WRITING OVER THE FIRST PASS'S RESULTS" : ""))

assert_fixed()
cases = available_cases()
ok, path = run_post(cases)
report(path)

log(ok ? "done" : "done, WITH ERRORS — see logs/$NEW_LOG")
ok || exit(1)
