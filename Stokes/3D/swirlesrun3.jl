#!/usr/bin/env julia
# =============================================================================
# swirlesrun3.jl — the Γ collapse test for the strongly stratified cases.
#
# THE HYPOTHESIS
#   At large N the mixing length is buoyancy-limited, l_eff = c·√TKE/N, so
#
#       K_T = √TKE · l_eff = Γ · TKE / N        with Γ a PURE NUMBER.
#
#   Rearranged, that is a falsifiable statement about a single quantity:
#
#       Γ ≡ K_T · N / TKE        must be CONSTANT
#
#   — constant in time within a case, and the SAME constant across every case,
#   whatever N, whatever the background profile, whatever the interface depth.
#   This is a far stronger test than fitting one exponent per case: eight
#   exponents near 1 can each be explained away, but sixteen cases landing on one
#   value of Γ cannot.
#
#   The figure shows the per-case distribution of Γ; the table adds the sharper
#   scalar. Since log Γ = log K_T + log N − log TKE,
#
#       d(log Γ)/d(log TKE)  =  (panel-d slope) − 1
#
#   exactly, so the dlnΓ/dlnE column IS the deviation of the exponent from 1,
#   read as a residual rather than off a fitted line.
#
# TWO DATA FORMATS, AND WHY THE DISTINCTION MATTERS
#   new     outputs/*/mixing_*.jld2 from MixedLayerDiffusivity.jl. FULL flux,
#           F_b = ⟨w′b′⟩ + F_sgs.
#   legacy  the archived */output_*/​*_profiles.jld2 sweeps. F_sgs WAS NEVER
#           WRITTEN, so K_T there is RESOLVED FLUX ONLY and biased LOW, hence Γ
#           is biased low too. A systematic offset between the two groups is
#           therefore expected as an artefact and must NOT be read as physics.
#           Legacy cases are drawn as open markers and flagged in the table.
#           Their loader is the twin of panelD_legacy.jl:load_case — keep in sync.
#
# USAGE   GKSwstype=100 julia --project=. swirlesrun3.jl
# ENV
#   SQRT_RI_MIN   keep cases with N/ω ≥ this          (default 5)
#   SKIP_PERIODS  tidal periods discarded             (default 3)
#   LEGACY_DIRS   colon-separated archive folders, "" to disable
#   OUT_ROOT / FIG_DIR / RESULT_SUFFIX
# =============================================================================

using JLD2, Plots, Printf, Statistics, Dates
using Oceananigans: FieldTimeSeries, interior, znodes

get!(ENV, "GKSwstype", "100")

const HERE   = @__DIR__
const ω      = 1e-4
const T_tide = 2π / ω
const GRAD_FLOOR = 0.05
const SKIP   = parse(Float64, get(ENV, "SKIP_PERIODS", "3"))
const SMIN   = parse(Float64, get(ENV, "SQRT_RI_MIN", "5"))
const outroot = joinpath(HERE, get(ENV, "OUT_ROOT", "outputs"))
const figdir  = get(ENV, "FIG_DIR", joinpath(HERE, "figures"))
const logdir  = get(ENV, "LOG_DIR", joinpath(HERE, "logs"))
const SUFFIX  = get(ENV, "RESULT_SUFFIX", "")
const LEGACY  = filter(!isempty, split(get(ENV, "LEGACY_DIRS",
    joinpath(HERE, "-Varying L_strat fraction of Lz") * ":" *
    joinpath(HERE, "-Varying L_strat")), ':'))

mkpath(figdir); mkpath(logdir)
default(fontfamily = "sans-serif", framestyle = :box, grid = true, gridalpha = 0.15,
        tickfontsize = 7, guidefontsize = 8, legendfontsize = 6, titlefontsize = 9)

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
include(joinpath(@__DIR__, "mixed_layer_height.jl"))
h_def_of(f) = try jldopen(io -> haskey(io, "h_def") ? io["h_def"] : "crossing", f, "r")
              catch; "crossing" end
interp_at(z, fv, z₀) = isnan(z₀) ? NaN : begin
    i = searchsortedlast(z, z₀)
    i < 1 && return fv[1]
    i >= length(z) && return fv[end]
    fv[i] + (fv[i+1] - fv[i]) * (z₀ - z[i]) / (z[i+1] - z[i])
end

# --- new format --------------------------------------------------------------
function load_new()
    isdir(outroot) || return []
    best = Dict{String,String}()
    for d in readdir(outroot; join = true), f in (isdir(d) ? readdir(d; join = true) : String[])
        startswith(basename(f), "mixing_") && endswith(f, ".jld2") || continue
        h_def_of(f) == H_DEF || continue
        tag = try jldopen(io -> io["tag"], f, "r") catch; continue end
        (!haskey(best, tag) || mtime(f) > mtime(best[tag])) && (best[tag] = f)
    end
    out = []
    for (tag, f) in best
        d = jldopen(f, "r") do io
            (; tag, s = io["n_over_omega"], times = io["times"],
               K = io["K_at_h"], TKE = io["TKE_at_h"])
        end
        length(d.times) == length(d.K) || continue          # phase-binned: no time axis
        push!(out, (; d..., legacy = false, label = tag))
    end
    return out
end

# --- legacy format -----------------------------------------------------------
# Only Ri is needed from the folder name: Γ uses N = √Ri·ω, and h/the mask use
# N²_ref = Ri·ω². The domain height never enters, so one loader serves every
# archived sweep regardless of its Lz or background shape.
function load_legacy(root)
    isdir(root) || (@warn "LEGACY_DIRS: no such folder — skipping" root; return [])
    out = []
    for d in sort(readdir(root; join = true))
        isdir(d) && startswith(basename(d), "output_") || continue
        m = match(r"_Ri(\d+)$", basename(d))
        m === nothing && continue
        Ri = parse(Float64, m[1]); Ri > 0 || continue
        fs = filter(f -> endswith(f, "_profiles.jld2"), readdir(d; join = true))
        isempty(fs) && continue
        ts = Dict(v => FieldTimeSeries(first(fs), v)
                  for v in ("U","V","B","uu","vv","ww","wb"))
        times = ts["B"].times; nt = length(times)
        zc = collect(znodes(ts["B"])); zf = collect(znodes(ts["wb"]))
        grab(v) = (a = zeros(length(interior(ts[v][1])), nt);
                   for n in 1:nt; a[:, n] .= vec(interior(ts[v][n])); end; a)
        U, V, B    = grab("U"), grab("V"), grab("B")
        uu, vv, ww = grab("uu"), grab("vv"), grab("ww")
        wb         = grab("wb")
        # ww = Average(w^2) with w on Faces lands on FACES — interpolate, do not
        # truncate (see panelD_legacy.jl header note 3).
        ww_c = [0.5 * (ww[k, n] + ww[min(k + 1, size(ww, 1)), n])
                for k in 1:length(zc), n in 1:nt]
        TKE = 0.5 .* ((uu .- U .^ 2) .+ (vv .- V .^ 2) .+ ww_c)
        N²_ref = Ri * ω^2
        G = zeros(length(zf), nt)                    # offline ∂B/∂z onto Faces
        for n in 1:nt, k in 2:length(zc)
            G[k, n] = (B[k, n] - B[k-1, n]) / (zc[k] - zc[k-1])
        end
        G[1, :] .= G[2, :]
        # Legacy runs have no F_sgs, so under H_DEF=flux the flux is the resolved
        # part alone — the same caveat their K_T already carries.
        h = [mixed_layer_height(zf, view(G, :, n), N²_ref; zmax = min(40.0, zf[end]),
                                F = view(wb, :, n)) for n in 1:nt]
        fl = GRAD_FLOOR * N²_ref
        K  = [(g = interp_at(zf, view(G, :, n), h[n]);
               (isnan(g) || abs(g) < fl) ? NaN :
                   -interp_at(zf, view(wb, :, n), h[n]) / g) for n in 1:nt]
        Kh = [interp_at(zc, view(TKE, :, n), h[n]) for n in 1:nt]
        lbl = replace(basename(d), "output_" => "")
        push!(out, (; tag = lbl, s = sqrt(Ri), times, K, TKE = Kh,
                      legacy = true, label = lbl))
    end
    return out
end

# =============================================================================
cases = vcat(load_new(), [c for r in LEGACY for c in load_legacy(String(r))])
cases = [c for c in cases if c.s >= SMIN]
sort!(cases, by = c -> (c.s, c.label))
isempty(cases) && (@warn "no cases with N/ω ≥ $SMIN"; exit(0))
@printf("%d case(s) with N/ω ≥ %g  (%d new, %d legacy)\n", length(cases), SMIN,
        count(!, (c.legacy for c in cases)), count(c -> c.legacy, cases))

# Γ = K_T·N/TKE, one value per retained sample.
function gamma_of(c)
    m = c.times .>= SKIP * T_tide
    N = c.s * ω
    K, E, t = c.K[m], c.TKE[m], c.times[m] ./ T_tide
    Γ = K .* N ./ E
    g = @. isfinite(Γ) && Γ > 0 && isfinite(E) && E > 0
    return (Γ = Γ[g], TKE = E[g], t = t[g], ndrop = count(!, g), ntot = length(Γ))
end
G = Dict(c.label => gamma_of(c) for c in cases)
pooled = reduce(vcat, (G[c.label].Γ for c in cases))
Γmed = median(pooled)

mk(c) = c.legacy ? :circle : :diamond
const PAL = cgrad(:viridis)
col(i) = PAL[(i - 1) / max(length(cases) - 1, 1)]

# The collapse, as one figure: median dot, IQR bar, 10-90 whisker, one column per
# case, ordered by N/ω. If K_T = Γ·TKE/N with Γ a pure number, every column sits
# at the same height whatever N, whatever the background scale. A column that
# steps away from the dashed pooled median is a case the closure does not cover.
#
# The within-case whisker is turbulent intermittency, NOT uncertainty on Γ: the
# median of ~800 samples is far better determined than the IQR suggests. What is
# being read here is the alignment of the DOTS.
pb = plot(legend = :topright, yscale = :log10, ylabel = "Γ = K_T·N/TKE",
          xlabel = "case (ordered by N/ω)", xrotation = 60,
          size = (max(760, 62 * length(cases)), 620),
          left_margin = 8Plots.mm, bottom_margin = 16Plots.mm,
          title = @sprintf("Γ = K_T·N/TKE across %d cases — level dots ⇒ one constant (skip %g periods)",
                           length(cases), SKIP))
for (i, c) in enumerate(cases)
    q = quantile(G[c.label].Γ, [0.10, 0.25, 0.50, 0.75, 0.90])
    plot!(pb, [i, i], [q[1], q[5]]; color = col(i), lw = 1.2, label = "")
    plot!(pb, [i, i], [q[2], q[4]]; color = col(i), lw = 5, label = "")
    scatter!(pb, [i], [q[3]]; color = :white, ms = 5, msw = 1.8,
             markerstrokecolor = col(i), marker = mk(c), label = "")
end
# Labelled on the legend rather than annotated in place: an annotation near the
# line lands on top of the first case's marker.
hline!(pb, [Γmed]; color = :black, lw = 2, ls = :dash,
       label = @sprintf("pooled median Γ = %.3f", Γmed))
xticks!(pb, 1:length(cases), [c.label for c in cases])

f1 = joinpath(figdir, "gamma_collapse$SUFFIX.png")
savefig(pb, f1)

# --- table -------------------------------------------------------------------
open(joinpath(logdir, "gamma_collapse$SUFFIX.log"), "a") do io
    for out in (io, stdout)
        println(out, "\n", "="^104)
        @printf(out, "==== %s  swirlesrun3.jl  Γ = K_T·N/TKE   SKIP_PERIODS=%g  N/ω ≥ %g\n",
                Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), SKIP, SMIN)
        println(out, "="^104)
        @printf(out, "%-18s %6s %-9s | %8s %8s %8s %6s | %9s %6s | %6s\n",
                "case", "N/ω", "flux", "Γ p25", "Γ med", "Γ p75", "IQR/m",
                "dlnΓ/dlnE", "r", "n")
        for c in cases
            g = G[c.label]
            q = quantile(g.Γ, [0.25, 0.50, 0.75])
            sl, r, n = loglog_slope(g.TKE, g.Γ)
            @printf(out, "%-18s %6.1f %-9s | %8.4f %8.4f %8.4f %6.2f | %+9.2f %6.2f | %6d\n",
                    c.label, c.s, c.legacy ? "resolved" : "full",
                    q[1], q[2], q[3], (q[3] - q[1]) / q[2], sl, r, n)
        end
        println(out, "-"^104)
        @printf(out, "pooled median Γ = %.4f  (p25 %.4f, p75 %.4f, n = %d)\n",
                Γmed, quantile(pooled, 0.25), quantile(pooled, 0.75), length(pooled))
        legn = [c for c in cases if c.legacy]; newn = [c for c in cases if !c.legacy]
        for (nm, sub) in (("full flux", newn), ("resolved only", legn))
            isempty(sub) && continue
            ms = [median(G[c.label].Γ) for c in sub]
            @printf(out, "  %-14s median Γ = %.4f   spread across cases %.4f – %.4f (%d cases)\n",
                    nm, median(ms), minimum(ms), maximum(ms), length(sub))
        end
        println(out, "dlnΓ/dlnE is EXACTLY (panel-d slope − 1): 0 ⇒ K_T ∝ TKE/N.")
        println(out, "Legacy cases lack F_sgs, so their K_T and Γ are biased LOW —")
        println(out, "an offset between the two flux groups is an artefact, not physics.")
        @printf(out, "wrote %s\n", relpath(f1, HERE))
        println(out, "="^104)
    end
end
