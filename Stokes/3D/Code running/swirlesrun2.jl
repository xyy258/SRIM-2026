#!/usr/bin/env julia
# Panel (d) on its own, for every case on disk.
#
# ---------------- What it answers ----------------
# Panel (d) fits log K_T against log TKE, both at z = h, within a single case.
# The two candidate closures are
#
#     K_T = c · √TKE · l        (l a fixed geometric length)   → slope 1/2
#     K_T = Γ · TKE / N         (N fixed by the background)    → slope 1
#
# The fit can tell them apart only because l and N are constant within a case
# while TKE varies over the tidal cycle. If l were the buoyancy scale √TKE/N the
# two would be algebraically the same. So slope 1/2 means the mixing length is
# set by geometry, and slope 1 means it is set by the stratification.
#
# Across the sweep both are limits of one closure,
#
#     K_T = √TKE · l_eff ,      l_eff = min( l_geom , c·√TKE/N )
#
# which crosses over at a turbulent Froude number Fr = √TKE/(N·l_geom) of about
# 1. The last figure tests that, and it is a stronger test than eight separate
# exponents, since every case must collapse onto one curve.
#
# ---------------- Why a cutoff ----------------
# K_T swings over a decade during the first few periods, from the restart, the
# change of bottom boundary condition, and h settling to equilibrium. None of
# that is the physics being fitted, and it has a large influence on a
# least-squares slope. MixedLayerDiffusivity.jl already discards SKIP_PERIODS;
# this script sweeps the cutoff so the answer can be checked against it. A slope
# that levels off is a measurement, a slope that drifts is the transient.
#
# ---------------- Input ----------------
# The small mixing_*.jld2 files written by MixedLayerDiffusivity.jl rather than
# the large moments files, since K_at_h, TKE_at_h, times, h and delta_eff are all
# in there already. Nothing is recomputed, so these numbers cannot drift from the
# main analysis. Where a case has more than one file, the newest wins, so each
# case appears once.
#
# USAGE
#   GKSwstype=100 julia --project=. swirlesrun2.jl
# ENV
#   SKIP_PERIODS   cutoff for the main grid and tables    (default 3)
#   SKIP_SWEEP     cutoffs for the stability figure       (default "0 1 2 3 4 5 6")
#   L_GEOM         geometric length for the collapse: h | delta_eff  (default h)
#   OUT_ROOT       where the case folders live            (default outputs)
#   FIG_DIR        where figures go                       (default figures)
#   RESULT_SUFFIX  appended to every output filename      (default "")

using JLD2, Plots, Printf, Statistics, Dates

get!(ENV, "GKSwstype", "100")           # headless-safe before any plotting

const HERE    = @__DIR__
const ω       = 1e-4
const T_tide  = 2π / ω
const outroot = joinpath(HERE, get(ENV, "OUT_ROOT", "outputs"))
const figdir  = get(ENV, "FIG_DIR", joinpath(HERE, "figures"))
const logdir  = get(ENV, "LOG_DIR", joinpath(HERE, "logs"))
const SUFFIX  = get(ENV, "RESULT_SUFFIX", "")
const SKIP    = parse(Float64, get(ENV, "SKIP_PERIODS", "3"))
const SWEEP   = parse.(Float64, split(get(ENV, "SKIP_SWEEP", "0 1 2 3 4 5 6")))
const L_GEOM  = get(ENV, "L_GEOM", "h")
L_GEOM in ("h", "delta_eff") || error("L_GEOM must be h or delta_eff — got \"$L_GEOM\"")

mkpath(figdir); mkpath(logdir)
default(fontfamily = "sans-serif", framestyle = :box, grid = true, gridalpha = 0.15,
        tickfontsize = 7, guidefontsize = 8, legendfontsize = 6, titlefontsize = 8)

# --- the fit, copied unchanged from MixedLayerDiffusivity.jl -----------------
# Do not change one copy alone: the point of reading the saved arrays is that
# this script and the main analysis agree by construction.
# Three definitions of h can be on disk at once (see mixed_layer_height.jl) and
# they are not interchangeable, so a figure must not mix them. Files written
# before h_def existed all used the 0.1 crossing.
include(joinpath(@__DIR__, "mixed_layer_height.jl"))
h_def_of(f) = try jldopen(io -> haskey(io, "h_def") ? io["h_def"] : "crossing", f, "r")
              catch; "crossing" end

function loglog_slope(x, y)
    m = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
    n = count(m)
    n < 8 && return (NaN, NaN, n)
    lx, ly = log10.(x[m]), log10.(y[m])
    sx, sy = mean(lx), mean(ly)
    slope = sum((lx .- sx) .* (ly .- sy)) / sum((lx .- sx) .^ 2)
    r = sum((lx .- sx) .* (ly .- sy)) / sqrt(sum((lx .- sx) .^ 2) * sum((ly .- sy) .^ 2))
    return (slope, r, n)
end

# --- discovery ---------------------------------------------------------------
function discover()
    isdir(outroot) || error("no $outroot/ — run this where the case folders are")
    best = Dict{String,String}()
    for d in readdir(outroot; join = true)
        isdir(d) || continue
        for f in readdir(d; join = true)
            startswith(basename(f), "mixing_") && endswith(f, ".jld2") || continue
            h_def_of(f) == H_DEF || continue
            tag = try
                jldopen(io -> io["tag"], f, "r")
            catch e
                @warn "cannot read $f — skipping" exception = e
                continue
            end
            # The newest file wins, so a re-run replaces the older one and each
            # case is plotted once.
            (!haskey(best, tag) || mtime(f) > mtime(best[tag])) && (best[tag] = f)
        end
    end
    return best
end

function load_case(f)
    d = jldopen(f, "r") do io
        (; tag = io["tag"], T = io["T"], s = io["n_over_omega"], Ri = io["Ri"],
           times = io["times"], K = io["K_at_h"], TKE = io["TKE_at_h"],
           h = io["h"], δ = io["delta_eff"], smooth = io["smooth"])
    end
    # With SMOOTH=phase the sample axis is phase rather than time, so a cutoff
    # in periods means nothing. Refuse rather than cut in the wrong place.
    if length(d.times) != length(d.K)
        @warn "$(d.tag): $(length(d.times)) times vs $(length(d.K)) samples " *
              "(smooth = $(d.smooth)) — a period cutoff needs the time axis. Skipping."
        return nothing
    end
    return d
end

keep_mask(c, skip) = (c.times .>= skip * T_tide)
l_geom(c) = L_GEOM == "h" ? c.h : c.δ

fit_at(c, skip) = (m = keep_mask(c, skip); loglog_slope(c.TKE[m], c.K[m]))

# =============================================================================
cases_by_tag = discover()
if isempty(cases_by_tag)
    @warn "no mixing_*.jld2 found under $outroot/ — run MixedLayerDiffusivity.jl first"
    exit(0)
end
cases = filter(!isnothing, [load_case(f) for f in values(cases_by_tag)])
# T outer and N/ω inner, so the grid reads as one row per interface depth.
sort!(cases, by = c -> (c.T, c.s))
@printf("found %d case(s): %s\n", length(cases), join((c.tag for c in cases), ", "))

const CLR = Dict(0.0 => "#8C8C8C", 1.0 => "#1B4E8F", 2.0 => "#2E8B57", 10.0 => "#B4502C")
clr(c) = get(CLR, c.s, "#000000")

# --- Figure 1: the grid, one panel (d) per case ------------------------------
function grid_figure(skip)
    panels = Any[]
    for c in cases
        m = keep_mask(c, skip)
        x, y, t = c.TKE[m], c.K[m], c.times[m] ./ T_tide
        g = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
        slope, r, n = loglog_slope(x, y)
        ttl = @sprintf("%s\nT=%g  N/ω=%g   slope %.2f (r %.2f, n %d)",
                       c.tag, c.T, c.s, slope, r, n)
        p = plot(xscale = :log10, yscale = :log10, title = ttl,
                 xlabel = "TKE at z = h (m² s⁻²)", ylabel = "K_T at z = h (m² s⁻¹)",
                 legend = :topleft, colorbar = false)
        if count(g) >= 8
            # Shaded by time, so a trajectory that is still drifting shows up as
            # a colour gradient across the cloud rather than hiding in it.
            scatter!(p, x[g], y[g]; marker_z = t[g], c = :viridis, ms = 1.6,
                     msw = 0, alpha = 0.55, label = "")
            x0, y0 = 10^mean(log10.(x[g])), 10^mean(log10.(y[g]))
            xr = 10 .^ range(log10(minimum(x[g])), log10(maximum(x[g])); length = 2)
            plot!(p, xr, y0 .* (xr ./ x0) .^ 0.5; color = "#B4502C", lw = 1.4,
                  ls = :dash, label = "½ → geometric l")
            plot!(p, xr, y0 .* (xr ./ x0) .^ 1.0; color = "#7A3117", lw = 1.4,
                  ls = :dot, label = "1 → TKE/N")
            isfinite(slope) && plot!(p, xr, y0 .* (xr ./ x0) .^ slope;
                                     color = :black, lw = 2,
                                     label = @sprintf("fit %.2f", slope))
        else
            annotate!(p, 0.5, 0.5, text("too few samples", 8))
        end
        push!(panels, p)
    end
    nT = length(unique(c.T for c in cases))
    ncol = max(1, cld(length(cases), max(nT, 1)))
    plot(panels...; layout = (max(nT, 1), ncol), size = (420 * ncol, 380 * max(nT, 1)),
         plot_title = @sprintf("panel (d) — first %g tidal periods discarded", skip),
         plot_titlefontsize = 11, left_margin = 11Plots.mm, bottom_margin = 7Plots.mm)
end

f1 = joinpath(figdir, "panelD_grid$SUFFIX.png")
savefig(grid_figure(SKIP), f1)

# --- Figure 2: does the slope depend on where the cut is made? ---------------
# A flat line here means the slope is a measurement.
p2 = plot(xlabel = "tidal periods discarded", ylabel = "slope  d(log K_T)/d(log TKE)",
          title = "panel (d) slope vs spin-up cutoff", legend = :outerright,
          size = (900, 500), ylims = (-0.6, 1.6), left_margin = 5Plots.mm,
          bottom_margin = 5Plots.mm)
hline!(p2, [0.5]; color = "#B4502C", ls = :dash, lw = 1.2, label = "½  geometric l")
hline!(p2, [1.0]; color = "#7A3117", ls = :dot,  lw = 1.2, label = "1  TKE/N")
stability = Dict{String,Any}()
for c in cases
    rows = [(sk, fit_at(c, sk)...) for sk in SWEEP]   # (cutoff, slope, r, n)
    stability[c.tag] = rows
    sl = [r[2] for r in rows]
    # Dashed for the shallower interface, so the two T columns can be told apart.
    plot!(p2, SWEEP, sl; color = clr(c), lw = 2, ls = c.T <= 5 ? :dash : :solid,
          marker = :circle, ms = 3,
          label = @sprintf("T=%g  N/ω=%g", c.T, c.s))
end
f2 = joinpath(figdir, "panelD_stability$SUFFIX.png")
savefig(p2, f2)

# --- Figure 3: the crossover, and the collapse that tests it -----------------
p3a = plot(xlabel = "N/ω", ylabel = "slope", title = @sprintf("(a) exponent vs stratification (skip %g)", SKIP),
           legend = :bottomright, xscale = :log10)
hline!(p3a, [0.5]; color = "#B4502C", ls = :dash, lw = 1.2, label = "½")
hline!(p3a, [1.0]; color = "#7A3117", ls = :dot,  lw = 1.2, label = "1")
for Tv in sort(unique(c.T for c in cases))
    sub = [c for c in cases if c.T == Tv && c.s > 0]      # N/ω = 0 has no x value
    isempty(sub) && continue
    sl = [fit_at(c, SKIP)[1] for c in sub]
    rr = [abs(fit_at(c, SKIP)[2]) for c in sub]
    # The marker size carries |r|, so a weak fit is a small point rather than
    # sitting on the curve looking as firm as a tight one.
    plot!(p3a, [c.s for c in sub], sl; lw = 2, marker = :circle,
          ms = 3 .+ 7 .* rr, label = @sprintf("T = %g m", Tv))
end

# The collapse: y = K_T/(√TKE·l) = l_eff/l_geom = min(1, c·Fr), so
#   Fr ≫ 1 (weak stratification)  → y → 1, flat        → K_T ~ √TKE·l
#   Fr ≪ 1 (strong stratification) → y → c·Fr, slope 1 → K_T ~ TKE/N
# It is undefined at N = 0, so the unstratified control does not appear here.
p3b = plot(xscale = :log10, yscale = :log10, legend = :topleft, colorbar = false,
           xlabel = "Fr = √TKE / (N · $(L_GEOM))",
           ylabel = "K_T / (√TKE · $(L_GEOM))  = l_eff / l_geom",
           title = "(b) do all cases collapse onto min(1, c·Fr)?")
for c in cases
    c.s > 0 || continue
    m = keep_mask(c, SKIP)
    N = c.s * ω
    l = l_geom(c)[m]
    x = sqrt.(max.(c.TKE[m], 0)) ./ (N .* l)
    y = c.K[m] ./ (sqrt.(max.(c.TKE[m], 0)) .* l)
    g = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
    count(g) >= 8 || continue
    scatter!(p3b, x[g], y[g]; ms = 1.4, msw = 0, alpha = 0.35, color = clr(c),
             label = @sprintf("T=%g N/ω=%g", c.T, c.s))
end
f3 = joinpath(figdir, "panelD_crossover$SUFFIX.png")
savefig(plot(p3a, p3b; layout = (1, 2), size = (1100, 460),
             left_margin = 6Plots.mm, bottom_margin = 6Plots.mm), f3)

# --- the table --------------------------------------------------------------
open(joinpath(logdir, "panelD$SUFFIX.log"), "a") do io
    for out in (io, stdout)
        println(out, "\n", "="^94)
        @printf(out, "==== %s  swirlesrun2.jl (panel d only)  SKIP_PERIODS=%g  L_GEOM=%s\n",
                Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), SKIP, L_GEOM)
        println(out, "="^94)
        @printf(out, "%-22s %6s %6s | %8s %7s %7s | %s\n",
                "case", "T", "N/ω", "slope", "r", "n", "slope at each cutoff " * join(SWEEP, " "))
        for c in cases
            sl, r, n = fit_at(c, SKIP)
            sw = join((@sprintf("%+.2f", row[2]) for row in stability[c.tag]), " ")
            @printf(out, "%-22s %6g %6g | %+8.2f %7.2f %7d | %s\n",
                    c.tag, c.T, c.s, sl, r, n, sw)
        end
        println(out, "-"^94)
        # A slope that moves with the cutoff is not a property of the flow.
        println(out, "A slope that changes across the cutoff columns is the spin-up transient,")
        println(out, "not the closure. Read the exponent only where those columns are flat.")
        @printf(out, "wrote %s\n      %s\n      %s\n",
                relpath(f1, HERE), relpath(f2, HERE), relpath(f3, HERE))
        println(out, "="^94)
    end
end
