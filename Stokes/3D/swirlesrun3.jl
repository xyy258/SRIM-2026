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
#   Panel (a) is the sharp version. Since log Γ = log K_T + log N − log TKE,
#
#       d(log Γ)/d(log TKE)  =  (panel-d slope) − 1
#
#   exactly. So a FLAT cloud in (a) is the exponent being 1, shown as a residual
#   rather than as a fitted line — a tilt you would have to squint at in panel (d)
#   is immediately visible here as a sloping cloud.
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
const logdir  = joinpath(HERE, "logs")
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
first_crossing(z, fv, lev) = begin
    i = findfirst(k -> isfinite(fv[k]) && fv[k] >= lev, eachindex(fv))
    (i === nothing || i == 1) ? NaN :
        z[i-1] + (z[i] - z[i-1]) * (lev - fv[i-1]) / (fv[i] - fv[i-1])
end
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
        h = [first_crossing(zf, view(G, :, n) ./ N²_ref, 0.1) for n in 1:nt]
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
al(c) = c.legacy ? 0.20 : 0.45
const PAL = cgrad(:viridis)
col(i) = PAL[(i - 1) / max(length(cases) - 1, 1)]

# (a) THE test: Γ against TKE. Flat ⇒ K_T ∝ TKE/N exactly.
pa = plot(xscale = :log10, yscale = :log10, legend = false,
          xlabel = "TKE at z = h (m² s⁻²)", ylabel = "Γ = K_T·N/TKE",
          title = "(a) Γ vs TKE — flat ⇒ exponent exactly 1")
for (i, c) in enumerate(cases)
    g = G[c.label]
    scatter!(pa, g.TKE, g.Γ; ms = 1.3, msw = 0, alpha = al(c), color = col(i))
end
hline!(pa, [Γmed]; color = :black, lw = 2, ls = :dash)
annotate!(pa, xlims(pa)[1] * 1.6, Γmed * 1.9,
          text(@sprintf("pooled median Γ = %.3f", Γmed), 7, :left))

# (b) per-case spread: median dot, IQR bar, 10-90 whisker. Overlap ⇒ collapse.
pb = plot(legend = false, yscale = :log10, ylabel = "Γ = K_T·N/TKE",
          xlabel = "case (ordered by N/ω)", xrotation = 60,
          title = "(b) per-case distribution — overlap ⇒ one constant")
for (i, c) in enumerate(cases)
    q = quantile(G[c.label].Γ, [0.10, 0.25, 0.50, 0.75, 0.90])
    plot!(pb, [i, i], [q[1], q[5]]; color = col(i), lw = 1.2)
    plot!(pb, [i, i], [q[2], q[4]]; color = col(i), lw = 5)
    scatter!(pb, [i], [q[3]]; color = :white, ms = 4, msw = 1.6,
             markerstrokecolor = col(i), marker = mk(c))
end
hline!(pb, [Γmed]; color = :black, lw = 2, ls = :dash)
xticks!(pb, 1:length(cases), [c.label for c in cases])

# (c) residual dependence on N. Case medians only — the within-case spread is
# panel (b)'s job, and drawing it here buries the very thing being tested.
# Jittered in x because there are only two distinct N in this dataset.
pc = plot(xscale = :log10, yscale = :log10, legend = :bottomleft,
          xlabel = "N/ω", ylabel = "median Γ per case",
          title = "(c) is Γ independent of N and of the background scale?")
for (i, c) in enumerate(cases)
    jit = c.s * (1 + 0.05 * (2 * ((i - 1) % 8) / 7 - 1))
    scatter!(pc, [jit], [median(G[c.label].Γ)]; color = col(i), ms = 6,
             marker = mk(c), msw = 0.8, label = "")
end
for sv in sort(unique(c.s for c in cases))
    sub = [c for c in cases if c.s == sv]
    mm  = median([median(G[c.label].Γ) for c in sub])
    plot!(pc, [sv * 0.93, sv * 1.07], [mm, mm]; color = :black, lw = 3, label = "")
    annotate!(pc, sv * 1.10, mm, text(@sprintf("%.3f", mm), 7, :left))
end
hline!(pc, [Γmed]; color = :black, lw = 1.5, ls = :dash, label = "pooled median")
scatter!(pc, [NaN], [NaN]; marker = :circle, color = :grey, label = "resolved only (legacy)")
any(!c.legacy for c in cases) &&
    scatter!(pc, [NaN], [NaN]; marker = :diamond, color = :grey, label = "full flux (new)")

# (d) stationarity: Γ must not drift through the run.
pd = plot(yscale = :log10, legend = false, xlabel = "t / T_tide",
          ylabel = "Γ", title = "(d) Γ vs time — drift ⇒ not yet equilibrated")
for (i, c) in enumerate(cases)
    g = G[c.label]
    isempty(g.t) && continue
    nb = 8                                    # bin, else the clouds are unreadable
    edges = range(minimum(g.t), maximum(g.t); length = nb + 1)
    xs, ys = Float64[], Float64[]
    for j in 1:nb
        sel = (g.t .>= edges[j]) .& (g.t .< edges[j+1])
        count(sel) >= 5 || continue
        push!(xs, 0.5 * (edges[j] + edges[j+1])); push!(ys, median(g.Γ[sel]))
    end
    plot!(pd, xs, ys; color = col(i), lw = 1.0, alpha = 0.55, marker = mk(c), ms = 2)
end
# The pooled median per time bin: the individual traces are intermittent, and
# what matters is whether their CENTRE moves.
let alt = reduce(vcat, (G[c.label].t for c in cases)),
    alg = reduce(vcat, (G[c.label].Γ for c in cases)),
    edges = range(minimum(alt), maximum(alt); length = 9)
    xs, ys = Float64[], Float64[]
    for j in 1:8
        sel = (alt .>= edges[j]) .& (alt .< edges[j+1])
        count(sel) >= 5 || continue
        push!(xs, 0.5 * (edges[j] + edges[j+1])); push!(ys, median(alg[sel]))
    end
    plot!(pd, xs, ys; color = :black, lw = 3, marker = :square, ms = 4)
end
hline!(pd, [Γmed]; color = :black, lw = 1.5, ls = :dash)

f1 = joinpath(figdir, "gamma_collapse$SUFFIX.png")
savefig(plot(pa, pb, pc, pd; layout = (2, 2), size = (1250, 950),
             left_margin = 8Plots.mm, bottom_margin = 12Plots.mm), f1)

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
