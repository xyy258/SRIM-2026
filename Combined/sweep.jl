# What the T = 10 m sweep consists of, where each case's files live, and the
# colour ramp that encodes r in every figure. Included by the reductions and by
# every plot script, so that adding a case is one edit rather than eight.
#
# It exists because of the weakly stratified re-runs (swirles.sh, 2026-09-10/11,
# LOG.txt). Those three cases did NOT land beside the originals — they were run
# on the cluster into one folder for scp and sit in Combined/Data/lowN. So a
# case's root is no longer a constant, and the tag for a non-integer r is not
# "sqrtRi0.5" but "sqrtRi0p5". Both of those were open-coded in six files.
#
# Nothing here reads data; it only says which cases exist and where.

using Printf

const SWEEP_HERE = @__DIR__

# ---------------- the sweep ----------------
# Stokes r = N/ω and Ekman r = N/f. ω = f = 1e-4 deliberately, so the same r
# means the same background N in both columns. 0.2 and 0.5 are the low-N
# re-runs; Ekman r = 0.5 predates them and is in the original folder.
const SVALS  = [0.2, 0.5, 1, 2, 5, 10, 25, 50]     # Stokes
const RATIOS = [0.2, 0.5, 1, 2, 5, 10, 25, 50]     # Ekman

# ---------------- where the files are ----------------
# Searched in order; Data/lowN wins where a case exists in both. It has to:
# Stokes/3D/outputs/P4_T10_sqrtRi0p5 predates the moments pipeline and holds
# only the raw field file, no *_moments.jld2 and no mixing_*, which is why
# r = 0.5 was dropped from the original sweep in the first place.
const STOKES_ROOTS = [joinpath(SWEEP_HERE, "Data", "raw", "lowN", "stokes"),
                      "/home/tll46/SRIM-2026/Stokes/3D/outputs"]
const EKMAN_ROOTS  = [joinpath(SWEEP_HERE, "Data", "raw", "lowN", "ekman"),
                      joinpath(SWEEP_HERE, "Data", "raw", "Ekman_moments", "4")]
# EKMAN_ROOT puts a root in front of those, for a variant run whose case folder
# has the same name as one already here — the rough-bed r = 25, say, since
# case_dir() names by r alone.
haskey(ENV, "EKMAN_ROOT") && pushfirst!(EKMAN_ROOTS, ENV["EKMAN_ROOT"])

# "P4_T10_sqrtRi5", "P4_T10_sqrtRi0p5" — the decimal point is written "p", which
# is how Stokes/3D/case_params.jl has always named them.
sqrtRi_tag(s) = "P4_T10_sqrtRi" *
    (isinteger(s) ? string(Int(s)) : replace(string(float(s)), "." => "p"))

# The two files every Stokes analysis here needs, or nothing if neither root has
# a complete case. `root` is returned so the caller can say where it read from.
function stokes_case(s)
    tag = sqrtRi_tag(s)
    for root in STOKES_ROOTS
        mix = joinpath(root, tag, "mixing_$(tag)_hcross.jld2")
        mom = joinpath(root, tag, "TidalBL3D_$(tag)_moments.jld2")
        isfile(mix) && isfile(mom) && return (tag = tag, mix = mix, mom = mom, root = root)
    end
    return nothing
end

# The Ekman case folder, or nothing. ekmanrun.jl's case_dir() formats r as
# %.1f, so this must match it exactly — and an r with two decimals would
# collide, which swirles.sh refuses rather than allows.
function ekman_case(r)
    sub = @sprintf("r=%.1f, T=10.0", r)
    for root in EKMAN_ROOTS
        d = joinpath(root, sub)
        isfile(joinpath(d, "Moments.jld2")) && return (dir = d, root = root)
    end
    return nothing
end

# ---------------- colour by r ----------------
# One anchor per sweep value, placed at log10(r), so the ramp is even in the
# sweep rather than in r. The two low anchors were added with the lowN runs;
# everything at r >= 1 keeps the colour it had in the earlier figures.
const RAMP = [(-0.699, (120, 200, 215)), (-0.301, ( 58, 140, 180)),
              ( 0.0,   ( 27,  78, 143)), ( 0.301, ( 46, 139,  87)),
              ( 0.699, (200, 150,  30)), ( 1.0,   (180,  80,  44)),
              ( 1.398, (142,  27,  78)), ( 1.699, ( 75,  16,  96))]

function ramp_colour(s)
    x = clamp(log10(s), RAMP[1][1], RAMP[end][1])
    for i in 1:length(RAMP)-1
        (x0, c0), (x1, c1) = RAMP[i], RAMP[i+1]
        x <= x1 || continue
        f = x1 == x0 ? 0.0 : (x - x0) / (x1 - x0)
        chan(k) = clamp(round(Int, c0[k] + f * (c1[k] - c0[k])), 0, 255)
        return "#" * join(string(chan(k), base = 16, pad = 2) for k in 1:3)
    end
    return "#000000"
end
