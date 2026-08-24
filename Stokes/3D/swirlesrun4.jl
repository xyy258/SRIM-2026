#!/usr/bin/env julia
# Which length scale does the mixing length follow?
#
# From K_T = √TKE · l, invert for the length itself,
#
#     l(t) = K_T_bulk(t) / √TKE(t)          [metres]
#
# and plot it against each candidate scale in turn. If l follows a candidate L,
# the cloud lies on the 1:1 line and the ratio l/L is the same constant in every
# case. Slope 1 on its own is not enough: a prefactor that changes from case to
# case means the scale is not the one setting the mixing.
#
# What is expected: at weak stratification l should follow a geometric scale, set
# by the flow and indifferent to N, and at strong stratification it should switch
# to the buoyancy scale √TKE/N. The points are coloured by N/ω, so that switch
# shows up as a separation of colours rather than needing a fit.
#
# CANDIDATES
#   h          mixed-layer depth                       geometric, varies in time
#   δ_s        Stokes layer thickness √(2ν/ω)           geometric, constant
#   u*/ω       friction velocity over tidal frequency   geometric, varies in time
#   √TKE/N     buoyancy scale                           undefined at N = 0
#
# δ_s is the same number in every run, so its plot only answers whether l is
# constant. It is kept because it is the natural wall scale to rule out.
#
# Note the pairing: K_T_bulk is averaged across the interface while TKE is local,
# at z = h. They are paired because the bulk integral is dominated by the
# interface, but it is still a mismatch — set K_SOURCE=at_h to use the local K_T
# instead and check that the conclusion does not depend on the choice.
#
# USAGE  GKSwstype=100 julia --project=. swirlesrun4.jl
# ENV    SKIP_PERIODS (3)   K_SOURCE = bulk | at_h (bulk)
#        OUT_ROOT / FIG_DIR / RESULT_SUFFIX

using JLD2, Plots, Printf, Statistics, Dates

get!(ENV, "GKSwstype", "100")

const HERE    = @__DIR__
const ω       = 1e-4
const ν       = 1.0e-6
const T_tide  = 2π / ω
const δ_s     = sqrt(2ν / ω)                    # 0.1414 m
const SKIP    = parse(Float64, get(ENV, "SKIP_PERIODS", "3"))
const KSRC    = get(ENV, "K_SOURCE", "bulk")
const outroot = joinpath(HERE, get(ENV, "OUT_ROOT", "outputs"))
const figdir  = get(ENV, "FIG_DIR", joinpath(HERE, "figures"))
const logdir  = get(ENV, "LOG_DIR", joinpath(HERE, "logs"))
const SUFFIX  = get(ENV, "RESULT_SUFFIX", "")
KSRC in ("bulk", "at_h") || error("K_SOURCE must be bulk or at_h — got \"$KSRC\"")

mkpath(figdir); mkpath(logdir)
default(fontfamily = "sans-serif", framestyle = :box, grid = true, gridalpha = 0.15,
        tickfontsize = 8, guidefontsize = 9, legendfontsize = 7, titlefontsize = 10)

# Three definitions of h can be on disk at once (see mixed_layer_height.jl) and
# they are not interchangeable, so a figure must not mix them. Files written
# before h_def existed all used the 0.1 crossing.
include(joinpath(@__DIR__, "mixed_layer_height.jl"))
h_def_of(f) = try jldopen(io -> haskey(io, "h_def") ? io["h_def"] : "crossing", f, "r")
              catch; "crossing" end

function loglog_slope(x, y)          # copied from MixedLayerDiffusivity.jl
    m = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
    n = count(m)
    n < 8 && return (NaN, NaN, n)
    lx, ly = log10.(x[m]), log10.(y[m])
    sx, sy = mean(lx), mean(ly)
    slope = sum((lx .- sx) .* (ly .- sy)) / sum((lx .- sx) .^ 2)
    r = sum((lx .- sx) .* (ly .- sy)) / sqrt(sum((lx .- sx) .^ 2) * sum((ly .- sy) .^ 2))
    return (slope, r, n)
end

# --- load the newest file per case, so a re-run replaces the older one -------
function load_cases()
    isdir(outroot) || error("no $outroot/ — run this where the case folders are")
    best = Dict{String,String}()
    for d in readdir(outroot; join = true)
        isdir(d) || continue
        for f in readdir(d; join = true)
            startswith(basename(f), "mixing_") && endswith(f, ".jld2") || continue
            h_def_of(f) == H_DEF || continue
            tag = try jldopen(io -> io["tag"], f, "r") catch; continue end
            (!haskey(best, tag) || mtime(f) > mtime(best[tag])) && (best[tag] = f)
        end
    end
    out = []
    for (tag, f) in best
        d = jldopen(f, "r") do io
            (; tag, T = io["T"], s = io["n_over_omega"], times = io["times"],
               Kb = io["K_T_bulk"], Kh = io["K_at_h"], TKE = io["TKE_at_h"],
               h = io["h"], ustar = io["ustar"])
        end
        length(d.times) == length(d.Kb) || (@warn "$tag: sample axis is not time — skipping"; continue)
        push!(out, d)
    end
    return sort(out, by = c -> (c.s, c.T))
end

cases = load_cases()
isempty(cases) && (@warn "no mixing_*.jld2 under $outroot/"; exit(0))
@printf("%d case(s): %s\n", length(cases), join((c.tag for c in cases), ", "))

# l = K_T/√TKE and every candidate scale, on the samples kept from one case.
function series(c)
    m = c.times .>= SKIP * T_tide
    K = (KSRC == "bulk" ? c.Kb : c.Kh)[m]
    E = c.TKE[m]
    q = sqrt.(max.(E, 0))                      # q = √TKE, the velocity scale
    l = K ./ q
    N = c.s * ω
    return (l = l,
            cand = Dict("h"       => c.h[m],
                        "delta_s" => fill(δ_s, count(m)),
                        "ustar_over_omega" => c.ustar[m] ./ ω,
                        "q_over_N" => c.s > 0 ? q ./ N : fill(NaN, count(m))))
end
S = Dict(c.tag => series(c) for c in cases)

const NAMES = ["h" => "h  (mixed-layer depth, m)",
               "delta_s" => "δ_s = √(2ν/ω)  (Stokes thickness, m)",
               "ustar_over_omega" => "u*/ω  (m)",
               "q_over_N" => "√TKE / N  (buoyancy scale, m)"]
# Colour is how the figures answer the question: the switch from a geometric
# scale to the buoyancy scale should show up as the colours separating. That
# needs a colour for every N/ω actually run, so this is a ramp in log₁₀(N/ω)
# rather than a lookup table. The old table only held {0,1,2,10} and returned
# black for anything else, which put N/ω = 5, 25 and 50 in one indistinguishable
# cloud.
#
# The anchors keep N/ω = 1 and 10 at the colours the earlier figures used, so a
# new plot can be laid beside an old one. N/ω = 0 stays grey and off the ramp:
# there b is a passive scalar, so it is a control rather than the weak end of a
# trend.
# The channels are Int, not the UInt8 that a 0x.. literal would give: the
# interpolation takes differences, and in UInt8 a negative one wraps to a large
# positive value, which silently produces colours from the wrong end of the ramp.
# The anchors sit at log₁₀ of the N/ω values actually being run, so each case
# gets its anchor hue exactly and anything in between interpolates. The hues run
# blue → green → amber → rust → crimson → violet, which stays ordered and
# separable; earlier drafts crowded 10, 25 and 50 into three shades of red.
# N/ω = 1 and 10 keep the colours the previous figures used.
const RAMP = [(0.0,   ( 27,  78, 143)),       # N/ω = 1    deep blue  #1B4E8F
              (0.301, ( 46, 139,  87)),       # N/ω = 2    green      #2E8B57
              (0.699, (200, 150,  30)),       # N/ω = 5    amber      #C8961E
              (1.0,   (180,  80,  44)),       # N/ω = 10   rust       #B4502C
              (1.398, (142,  27,  78)),       # N/ω = 25   crimson    #8E1B4E
              (1.699, ( 75,  16,  96))]       # N/ω = 50   violet     #4B1060

function ramp_colour(s)
    s > 0 || return "#8C8C8C"
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
clr(c) = ramp_colour(c.s)

# --- one plot per candidate -------------------------------------------------
written = String[]
for (key, axlabel) in NAMES
    p = plot(xscale = :log10, yscale = :log10, legend = :topleft,
             xlabel = axlabel, ylabel = "l = K_T/√TKE  (m)",
             size = (760, 620), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
             title = key == "delta_s" ?
                 "l vs δ_s — δ_s is the same in every run, so this asks only:\nis l constant?" :
                 "does the mixing length follow $key ?   (1:1 dashed)")
    any_pt = false
    for c in cases
        x, y = S[c.tag].cand[key], S[c.tag].l
        g = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
        count(g) >= 8 || continue
        any_pt = true
        scatter!(p, x[g], y[g]; ms = 1.6, msw = 0, alpha = 0.45, color = clr(c),
                 label = @sprintf("T=%g  N/ω=%g", c.T, c.s))
    end
    if any_pt && key != "delta_s"
        # The 1:1 line, l = L exactly. A prefactor moves a cloud off it
        # vertically without changing the slope, which is why it is drawn.
        xs = reduce(vcat, [filter(v -> isfinite(v) && v > 0, S[c.tag].cand[key]) for c in cases])
        lo, hi = minimum(xs), maximum(xs)
        plot!(p, [lo, hi], [lo, hi]; color = :black, lw = 1.5, ls = :dash, label = "l = L (1:1)")
    end
    f = joinpath(figdir, "l_vs_$(key)$(SUFFIX).png")
    savefig(p, f); push!(written, f)
end

# --- the trend across the column ------------------------------------------
# The scatter plots above show one cloud per case, so a trend across N/ω has to
# be read by eye from where the clouds sit. With a whole column of N/ω on disk
# the trend is the result, so it gets its own figure: the median l/L of each
# case against N/ω, one line per candidate.
#
# How to read it: a candidate that SETS the mixing length gives a flat line —
# same proportionality at every stratification. A sloping line means l is merely
# correlated with that scale. The expected picture is a geometric scale flat at
# small N/ω and l/(√TKE/N) flat at large N/ω, crossing somewhere in between.
#
# N/ω = 0 is left out: it cannot go on a log axis, √TKE/N is undefined there,
# and b is a passive scalar, so it is a control rather than the weak end of the
# trend. Its ratios are in the table below.
function fig_ratio_trend()
    Ts = sort(unique(c.T for c in cases))
    # Ticks at the N/ω actually run. Left to itself a log axis labels
    # 10^0.0, 10^0.3, 10^0.6 ..., which is unreadable as a stratification.
    svals = sort(unique(c.s for c in cases if c.s > 0))
    p = plot(xscale = :log10, yscale = :log10, legend = :outerright,
             xticks = (svals, [@sprintf("%g", v) for v in svals]),
             xlabel = "N/ω", ylabel = "median l / L over the case",
             size = (900, 620), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
             title = "Which candidate scale is l proportional to, and is the\n" *
                     "proportionality the same at every N/ω?  (flat = yes)")
    CAND_CLR = Dict("h" => "#1B4E8F", "delta_s" => "#8C8C8C",
                    "ustar_over_omega" => "#2E8B57", "q_over_N" => "#B4502C")
    styles = [:solid, :dash, :dot, :dashdot]
    drew = false
    for (key, _) in NAMES, (i, T) in enumerate(Ts)
        xs, ys = Float64[], Float64[]
        for c in cases
            c.T == T && c.s > 0 || continue
            x, y = S[c.tag].cand[key], S[c.tag].l
            g = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
            count(g) >= 8 || continue
            push!(xs, c.s); push!(ys, median(y[g] ./ x[g]))
        end
        length(xs) >= 2 || continue
        drew = true
        plot!(p, xs, ys; lw = 2, marker = :circle, ms = 4, msw = 0,
              color = CAND_CLR[key], ls = styles[min(i, end)],
              label = @sprintf("%s,  T = %g m", key, T))
    end
    drew || return nothing
    f = joinpath(figdir, "l_over_L_vs_N$(SUFFIX).png")
    savefig(p, f)
    return f
end
let f = fig_ratio_trend()
    f === nothing || push!(written, f)
end

# --- table ------------------------------------------------------------------
# A slope near 1 says l is proportional to L within the case, and the same median
# l/L across cases says it is the same proportionality. Both are needed.
open(joinpath(logdir, "length_scales$SUFFIX.log"), "a") do io
    for out in (io, stdout)
        println(out, "\n", "="^96)
        @printf(out, "==== %s  swirlesrun4.jl   l = K_T(%s)/√TKE   SKIP_PERIODS=%g\n",
                Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), KSRC, SKIP)
        println(out, "="^96)
        for (key, _) in NAMES
            @printf(out, "\n-- %s --\n%-22s %6s | %10s %8s %6s | %10s\n",
                    key, "case", "N/ω", "slope", "r", "n", "median l/L")
            for c in cases
                x, y = S[c.tag].cand[key], S[c.tag].l
                g = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
                sl, r, n = loglog_slope(x, y)
                rat = count(g) > 0 ? median(y[g] ./ x[g]) : NaN
                # A regression on a candidate with no spread in x returns a
                # number, but that number is noise, so say so instead.
                flat = count(g) > 0 && std(log10.(x[g])) < 1e-9
                if flat
                    @printf(out, "%-22s %6g | %10s %8s %6d | %10.3f\n",
                            c.tag, c.s, "n/a", "(const)", n, rat)
                else
                    @printf(out, "%-22s %6g | %+10.2f %8.2f %6d | %10.3f\n",
                            c.tag, c.s, sl, r, n, rat)
                end
            end
        end
        println(out, "\n", "-"^96)
        println(out, "A candidate is THE scale only if slope ≈ 1 AND median l/L is the same")
        println(out, "across cases. Slope 1 with a case-dependent ratio means l is merely")
        println(out, "correlated with that scale, not set by it.")
        for f in written; @printf(out, "wrote %s\n", relpath(f, HERE)); end
        println(out, "="^96)
    end
end
