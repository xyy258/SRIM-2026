#!/usr/bin/env julia
# =============================================================================
# swirlesrun2.jl — panel (d) alone, for every case on disk.
#
# WHAT THIS ANSWERS
# -----------------
# Panel (d) fits  log K_T  against  log TKE, both at z = h, within one case.
# The two candidate closures are
#
#     K_T = c · √TKE · l        (l a FIXED GEOMETRIC length)   → slope 1/2
#     K_T = Γ · TKE / N         (N FIXED by the background)    → slope 1
#
# The fit discriminates ONLY because l and N are constants within a case while
# TKE varies over the tidal cycle. If l were instead the buoyancy scale √TKE/N
# the two forms would be algebraically identical (√TKE·√TKE/N = TKE/N) and the
# fit would be vacuous. So "slope 1/2" is really the statement THE MIXING LENGTH
# IS SET BY GEOMETRY, and "slope 1" is THE MIXING LENGTH IS SET BY STRATIFICATION.
#
# Within one case only one can hold: c√TKE·l = ΓTKE/N has the single solution
# √TKE = clN/Γ, so the two lines cross at one point rather than coinciding.
# Across the sweep both are limits of ONE closure,
#
#     K_T = √TKE · l_eff ,      l_eff = min( l_geom , c·√TKE/N )
#
# whose crossover is at the turbulent Froude number Fr = √TKE/(N·l_geom) ≈ 1.
# The last figure below tests exactly that, and it is a far stronger test than
# eight separate exponents: every case must collapse onto one curve.
#
# WHY A CUTOFF
# ------------
# Panel (c) shows K_T swinging over a decade during the first few periods — the
# restart from the spin-up snapshot, plus the no-slip → drag change at the
# bottom, plus h migrating to its equilibrium. None of that is the physics being
# fitted, and because it is a large excursion in BOTH axes it has high leverage
# on a least-squares slope. MixedLayerDiffusivity.jl already discards
# SKIP_PERIODS (default 1.0); this script sweeps the cutoff so you can see
# whether the answer depends on where you put it. A slope that plateaus is a
# measurement; a slope that drifts with the cutoff is the transient talking.
#
# INPUT
# -----
# The small mixing_*.jld2 written by MixedLayerDiffusivity.jl, NOT the large
# moments files: K_at_h, TKE_at_h, times, h and delta_eff are all already in
# there. Nothing is re-derived, so these numbers cannot drift from the main
# analysis. Every outputs/*/mixing_*.jld2 is picked up; where a case has both a
# plain and a _fixed copy the newer file wins, so each tag appears once.
#
# USAGE
#   GKSwstype=100 julia --project=. swirlesrun2.jl
# ENV
#   SKIP_PERIODS   cutoff for the main grid + tables      (default 3)
#   SKIP_SWEEP     cutoffs for the stability figure       (default "0 1 2 3 4 5 6")
#   L_GEOM         geometric length for the collapse: h | delta_eff  (default h)
#   OUT_ROOT       where the case folders live            (default outputs)
#   FIG_DIR        where figures go                       (default figures)
#   RESULT_SUFFIX  appended to every output filename      (default "")
# =============================================================================

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

# --- the fit, COPIED VERBATIM from MixedLayerDiffusivity.jl -------------------
# Do not "improve" one copy alone: the whole point of reading the saved arrays
# is that this script and the main analysis agree by construction.
# Three definitions of h can be on disk at once (see mixed_layer_height.jl), and
# they are not interchangeable: a figure built from a mixture of them would be
# meaningless. Files written before h_def existed were all the 0.1-crossing.
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
            # Newest wins, so a _fixed re-run supersedes the stale original and
            # each case is plotted once.
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
    # SMOOTH=phase bins over phase, so the sample axis is no longer time and a
    # cutoff in periods is meaningless. Refuse rather than silently mis-cut.
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
# T outer, N/ω inner: the grid then reads as one row per interface depth.
sort!(cases, by = c -> (c.T, c.s))
@printf("found %d case(s): %s\n", length(cases), join((c.tag for c in cases), ", "))

const CLR = Dict(0.0 => "#8C8C8C", 1.0 => "#1B4E8F", 2.0 => "#2E8B57", 10.0 => "#B4502C")
clr(c) = get(CLR, c.s, "#000000")

# --- FIGURE 1: the grid, one panel (d) per case ------------------------------
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
            # Shaded by time so a trajectory that is still drifting is visible as
            # a colour gradient across the cloud rather than hiding in the scatter.
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

# --- FIGURE 2: does the slope depend on where you cut? -----------------------
# The point of the whole exercise. A flat line is a measurement.
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
    # Dashed for the shallower interface, so the two T columns separate by eye.
    plot!(p2, SWEEP, sl; color = clr(c), lw = 2, ls = c.T <= 5 ? :dash : :solid,
          marker = :circle, ms = 3,
          label = @sprintf("T=%g  N/ω=%g", c.T, c.s))
end
f2 = joinpath(figdir, "panelD_stability$SUFFIX.png")
savefig(p2, f2)

# --- FIGURE 3: the crossover, and the collapse that tests it -----------------
p3a = plot(xlabel = "N/ω", ylabel = "slope", title = @sprintf("(a) exponent vs stratification (skip %g)", SKIP),
           legend = :bottomright, xscale = :log10)
hline!(p3a, [0.5]; color = "#B4502C", ls = :dash, lw = 1.2, label = "½")
hline!(p3a, [1.0]; color = "#7A3117", ls = :dot,  lw = 1.2, label = "1")
for Tv in sort(unique(c.T for c in cases))
    sub = [c for c in cases if c.T == Tv && c.s > 0]      # N/ω = 0 has no abscissa
    isempty(sub) && continue
    sl = [fit_at(c, SKIP)[1] for c in sub]
    rr = [abs(fit_at(c, SKIP)[2]) for c in sub]
    # Marker size carries |r|, so a weak fit is visibly a small point rather than
    # sitting on the curve with the same authority as a tight one.
    plot!(p3a, [c.s for c in sub], sl; lw = 2, marker = :circle,
          ms = 3 .+ 7 .* rr, label = @sprintf("T = %g m", Tv))
end

# The collapse. y = K_T/(√TKE·l) = l_eff/l_geom = min(1, c·Fr):
#   Fr ≫ 1 (weak stratification) → y → 1, flat      → K_T ~ √TKE·l, slope ½
#   Fr ≪ 1 (strong)              → y → c·Fr, slope 1 → K_T ~ TKE/N,  slope 1
# Undefined at N = 0, so the control is absent here by construction.
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

# --- the table ---------------------------------------------------------------
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
