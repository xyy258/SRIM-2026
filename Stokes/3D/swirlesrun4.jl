#!/usr/bin/env julia
# =============================================================================
# swirlesrun4.jl — what length scale does the mixing length actually follow?
#
# From K_T = √TKE · l, invert for the length itself:
#
#       l(t) = K_T_bulk(t) / √TKE(t)          [metres]
#
# then plot it against each candidate scale in turn. If l tracks a candidate L,
# the cloud lies on the 1:1 line (slope 1 in log-log) AND the ratio l/L is the
# SAME constant in every case. Slope 1 alone is not enough — a case-dependent
# prefactor means the scale is not the one setting the mixing.
#
# The expectation being tested: below the transition l follows a GEOMETRIC scale
# (set by the flow's geometry, indifferent to N), and above it swaps to the
# buoyancy scale q/N. So the weakly stratified cases should collapse on h /
# δ_s / u*/ω and the strongly stratified ones on √TKE/N — points are coloured by
# N/ω so that swap is visible as a colour separation rather than needing a fit.
#
# CANDIDATES
#   h          mixed-layer depth                       geometric, time-varying
#   δ_s        Stokes layer thickness √(2ν/ω)           geometric, CONSTANT
#   u*/ω       friction velocity over tidal frequency   geometric, time-varying
#   √TKE/N     buoyancy scale                           NOT geometric; undefined at N = 0
#
# δ_s is the same number in every run, so its plot is degenerate as a scatter:
# every point sits at x = 0.1414 m and the only question it answers is whether l
# is constant. Kept because it is the natural wall scale to rule out.
#
# PAIRING NOTE. K_T_bulk is the interface-averaged diffusivity (−∫F_b dz / Δb)
# while TKE is local, at z = h. They are paired because the bulk integral is
# dominated by the interface, but it IS a mismatch — set K_SOURCE=at_h to use the
# local K_T at z = h instead and check the conclusion does not depend on it.
#
# USAGE  GKSwstype=100 julia --project=. swirlesrun4.jl
# ENV    SKIP_PERIODS (3)   K_SOURCE = bulk | at_h (bulk)
#        OUT_ROOT / FIG_DIR / RESULT_SUFFIX
# =============================================================================

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

# Three definitions of h can be on disk at once (see mixed_layer_height.jl), and
# they are not interchangeable: a figure built from a mixture of them would be
# meaningless. Files written before h_def existed were all the 0.1-crossing.
include(joinpath(@__DIR__, "mixed_layer_height.jl"))
h_def_of(f) = try jldopen(io -> haskey(io, "h_def") ? io["h_def"] : "crossing", f, "r")
              catch; "crossing" end

function loglog_slope(x, y)          # verbatim from MixedLayerDiffusivity.jl
    m = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
    n = count(m)
    n < 8 && return (NaN, NaN, n)
    lx, ly = log10.(x[m]), log10.(y[m])
    sx, sy = mean(lx), mean(ly)
    slope = sum((lx .- sx) .* (ly .- sy)) / sum((lx .- sx) .^ 2)
    r = sum((lx .- sx) .* (ly .- sy)) / sqrt(sum((lx .- sx) .^ 2) * sum((ly .- sy) .^ 2))
    return (slope, r, n)
end

# --- load: newest file per tag, so a _fixed re-run supersedes the original -----
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

# l = K_T/√TKE, plus every candidate, on the retained samples of one case.
function series(c)
    m = c.times .>= SKIP * T_tide
    K = (KSRC == "bulk" ? c.Kb : c.Kh)[m]
    E = c.TKE[m]
    q = sqrt.(max.(E, 0))                      # q ≡ √TKE, the velocity scale
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
const CLR = Dict(0.0 => "#8C8C8C", 1.0 => "#1B4E8F", 2.0 => "#2E8B57", 10.0 => "#B4502C")
clr(c) = get(CLR, c.s, "#000000")

# --- one plot per candidate ---------------------------------------------------
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
        # 1:1 over the drawn range: l = L exactly. Prefactors move a cloud off it
        # vertically without changing its slope, which is the point of drawing it.
        xs = reduce(vcat, [filter(v -> isfinite(v) && v > 0, S[c.tag].cand[key]) for c in cases])
        lo, hi = minimum(xs), maximum(xs)
        plot!(p, [lo, hi], [lo, hi]; color = :black, lw = 1.5, ls = :dash, label = "l = L (1:1)")
    end
    f = joinpath(figdir, "l_vs_$(key)$(SUFFIX).png")
    savefig(p, f); push!(written, f)
end

# --- table --------------------------------------------------------------------
# slope ≈ 1 says l is PROPORTIONAL to L within the case; median l/L being the same
# number across cases says it is the SAME proportionality — both are needed.
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
                # A regression on a candidate with no spread in x is degenerate —
                # it returns a number, and that number is noise. Say so instead.
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
