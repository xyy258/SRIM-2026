# l against √TKE/N at z = h, T = 10 m, Stokes and Ekman on one axis.
#
# This is Stokes/3D/plot_l_vs_qN_T10.jl with the Ekman column added. Both sides
# use the same mixing length and the same abscissa,
#
#     l = K_T(z=h) / √TKE(z=h)        x = √TKE(z=h) / N
#
# the same mixed-layer height (crossing at 0.1 N²_ref), and the same N, because
# the Stokes tidal frequency ω and the Ekman Coriolis parameter f are both
# 1e-4 s⁻¹, so N/ω and N/f label identical N.
#
# ---------------- The two sides are now like for like ----------------
# They were not, until ekmanrun.jl re-ran the Ekman column writing the subgrid
# buoyancy flux. Both K_T are now built from the whole flux,
#
#     K_T = −(⟨w'b'⟩ + F_sgs) / ⟨∂b/∂z⟩
#
# which matters: the subgrid share at z = h runs from 0.03 to 0.59 across the
# Stokes cases and from 0.04 to 0.56 across the Ekman ones. Reading the old
# resolved-only Ekman points against the full Stokes curve understated l by
# about a factor of two at the strongly stratified end.
#
# If only the old Data/ekman_lengthscales_T10.jld2 is present this script falls
# back to it and says so, and in that case the grey open squares — the Stokes
# medians with the subgrid part removed — are the curve to read the crosses
# against. With the moments file they are not drawn, because nothing needs them.
#
# ---------------- One caveat that survives ----------------
# h(t) is still creeping upward at the end of the record for N/f <= 5, and over
# the same window l itself is still FALLING there, by 10-21 %. Those points are
# therefore upper bounds on the converged l, not lower bounds, and the direction
# matters: the low-N/f Ekman medians sit above the Stokes curve, and they are
# still moving towards it. Whether the excess survives to equilibrium cannot be
# decided from this record. They are drawn hollow, and both drifts are printed
# below.
#
# For the same reason no independent Ekman fit is drawn. Only N/f = 25 and 50
# have settled, and a two-parameter saturating curve through two points is not a
# fit — it pinned to the edge of the search grid when tried.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot_l_vs_qN_T10_combined.jl
#        (run reduce_ekman_moments_T10.jl first — it writes the Ekman side)

using JLD2, Plots, Printf, Statistics

get!(ENV, "GKSwstype", "100")
const HERE   = @__DIR__
const STOKES = "/home/tll46/SRIM-2026/Stokes/3D"
const EKNEW  = joinpath(HERE, "Data", "ekman_lengthscales_T10_moments.jld2")
const EKOLD  = joinpath(HERE, "Data", "ekman_lengthscales_T10.jld2")
const FIGDIR = joinpath(HERE, "figures")
const ω      = 1e-4
const T_tide = 2π / ω
const SKIP   = 3                      # Stokes spin-up, in tidal periods
const SVALS  = [1, 2, 5, 10, 25, 50]
# A case counts as equilibrated if neither h nor l moves by more than this
# across the averaging window. 5 % is set against the 7.5 % rms of the Stokes
# fit itself: drift smaller than the scatter of the reference curve cannot be
# distinguished from it.
const DRIFT_TOL = 0.05

# The ramp swirlesrun4.jl uses, so colours mean the same N in every figure.
const RAMP = [(0.0,   ( 27,  78, 143)), (0.301, ( 46, 139,  87)),
              (0.699, (200, 150,  30)), (1.0,   (180,  80,  44)),
              (1.398, (142,  27,  78)), (1.699, ( 75,  16,  96))]
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
med(v) = (w = filter(isfinite, v); isempty(w) ? NaN : median(w))

interp_at(z, fv, z₀) = isnan(z₀) ? NaN : begin
    i = searchsortedlast(z, z₀)
    i < 1 && return fv[1]
    i >= length(z) && return fv[end]
    fv[i] + (fv[i+1] - fv[i]) * (z₀ - z[i]) / (z[i+1] - z[i])
end

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))

# ---------------- Stokes ----------------
# K_sgs is stored as a full (z, t) profile, so the resolved-only diffusivity at
# z = h has to be interpolated onto h the same way K_at_h was.
stokes = []
for s in SVALS
    tag = "P4_T10_sqrtRi$s"
    f   = joinpath(STOKES, "outputs", tag, "mixing_$(tag)_hcross.jld2")
    isfile(f) || (say("missing $f — skipped"); continue)
    d = jldopen(f, "r") do io
        (t = io["times"], Kh = io["K_at_h"], E = io["TKE_at_h"], h = io["h"],
         zf = io["z_face"], Ksgs = io["K_sgs"], frac = io["checks"]["K_sgs_over_K_T"])
    end
    Ksgs_h = [interp_at(d.zf, view(d.Ksgs, :, n), d.h[n]) for n in eachindex(d.h)]
    m  = d.t .>= SKIP * T_tide
    q  = sqrt.(max.(d.E[m], 0))
    l  = d.Kh[m] ./ q                          # full K_T
    lr = (d.Kh[m] .- Ksgs_h[m]) ./ q           # resolved-only K_T
    x  = q ./ (s * ω)
    g  = @. isfinite(l) && isfinite(x) && l > 0 && x > 0
    gr = @. isfinite(lr) && isfinite(x) && lr > 0 && x > 0
    push!(stokes, (s = s, x = x[g], l = l[g], xm = med(x[g]), lm = med(l[g]),
                   xr = med(x[gr]), lr = med(lr[gr]), frac = d.frac))
end
isempty(stokes) && error("no Stokes T = 10 mixing files under $STOKES/outputs")

# ---------------- Ekman ----------------
EKFILE = isfile(EKNEW) ? EKNEW : EKOLD
FULL_K = EKFILE == EKNEW
ekman = []
if isfile(EKFILE)
    say("Ekman side: $(basename(EKFILE))")
    jldopen(EKFILE, "r") do io
        say("  flux = " * (haskey(io, "flux") ? io["flux"] : "unrecorded"))
        for r in io["ratios"]
            g = @sprintf("r=%.1f", r)
            N = io["$g/N"]; E = io["$g/TKE_at_h"]; Kh = io["$g/K_at_h"]
            q = sqrt.(max.(E, 0))
            l = Kh ./ q
            x = q ./ N
            k = @. isfinite(l) && isfinite(x) && l > 0 && x > 0
            any(k) || continue
            drift = haskey(io, "$g/h_drift")   ? io["$g/h_drift"]   : NaN
            share = haskey(io, "$g/sgs_share") ? io["$g/sgs_share"] : NaN
            # Whether l has settled matters more than whether h has, and the
            # two can disagree, so the trend in l across the window is measured
            # here directly rather than inferred from h.
            lf = l[k]; nq = max(1, length(lf) ÷ 4)
            lq1 = med(lf[1:nq]); lq4 = med(lf[end-nq+1:end])
            ld = (lq4 - lq1) / lq1
            push!(ekman, (r = r, x = x[k], l = lf, xm = med(x[k]), lm = med(lf),
                          lq1 = lq1, lq4 = lq4,
                          drift = drift, share = share, ldrift = ld,
                          settled = !((isfinite(drift) && abs(drift) > DRIFT_TOL) ||
                                      (isfinite(ld) && abs(ld) > DRIFT_TOL))))
        end
    end
else
    say("WARNING: no Ekman reduction found — run reduce_ekman_moments_T10.jl. Stokes only.")
end
FULL_K || say("WARNING: Ekman K_T is resolved-only — read the crosses against the grey squares.")
say("")

# ---------------- fits ----------------
function fit_sat(cs)
    best = (Inf, 0.0, 0.0)
    for L in 0.30:0.002:1.60, x0 in 0.05:0.005:4.0
        sse = 0.0
        for c in cs
            p = L * (1 - exp(-c.xm / x0))
            p > 0 || (sse = Inf; break)
            sse += (log(c.lm) - log(p))^2
        end
        sse < best[1] && (best = (sse, L, x0))
    end
    return best
end
sse, L∞, x0 = fit_sat(stokes)
say(@sprintf("Stokes fit (unchanged): L∞ = %.3f m, x₀ = %.3f m, rms %.1f %% in l",
             L∞, x0, 100 * sqrt(sse / length(stokes))))

# An Ekman fit is only attempted if enough cases have equilibrated. Two settled
# points cannot constrain a two-parameter saturating curve — that attempt pinned
# L∞ to the edge of the search grid — so the bar is four.
ek_fit = filter(c -> c.settled, ekman)
if length(ek_fit) >= 4
    esse, eL, ex0 = fit_sat(ek_fit)
    say(@sprintf("Ekman fit (%d settled cases): L∞ = %.3f m, x₀ = %.3f m, rms %.1f %% in l",
                 length(ek_fit), eL, ex0, 100 * sqrt(esse / length(ek_fit))))
else
    eL = ex0 = NaN
    say(@sprintf("Ekman fit: not attempted — only %d of %d cases have equilibrated, too few to constrain L∞ and x₀",
                 length(ek_fit), length(ekman)))
end

# How far each Ekman median sits from the Stokes curve.
if !isempty(ekman)
    say("")
    say("Ekman medians against the Stokes fit  (ratio > 1 means Ekman mixes more)")
    for c in ekman
        pred = L∞ * (1 - exp(-c.xm / x0))
        say(@sprintf("  N/f = %-5g  l = %.4f m   Stokes fit %.4f m   ratio %5.2f   h drift %+5.1f %%   l drift %+6.1f %%%s",
                     c.r, c.lm, pred, c.lm / pred, 100c.drift, 100c.ldrift,
                     c.settled ? "" : "   (upper bound — l still falling)"))
    end
end

say("")
say("case medians — x = √TKE/N (m), l = K_T/√TKE (m)")
say("  Stokes                                    Ekman")
say("  N/ω      x       l    l(resolved)  K_sgs/K_T |  N/f      x       l   sgs share")
for i in 1:max(length(stokes), length(ekman))
    a = i <= length(stokes) ? stokes[i] : nothing
    b = i <= length(ekman)  ? ekman[i]  : nothing
    sa = a === nothing ? " "^45 :
         @sprintf("  %-5g %7.4f %7.4f %10.4f %10.3f", a.s, a.xm, a.lm, a.lr, a.frac)
    sb = b === nothing ? "" : @sprintf(" | %-5g %8.4f %7.4f %8.2f", b.r, b.xm, b.lm, b.share)
    say(sa * sb)
end

# ---------------- the figure ----------------
# Two ways of drawing the same reduction, selected by STYLE:
#
#   cloud       every retained time sample as a faint point, medians on top.
#               Shows the shape of each case's distribution, including that the
#               clouds are trajectories rather than scatter — the excursions are
#               the forcing cycle, not noise.
#   errorbars   medians only, with the interquartile range in both x and l.
#               The same information reduced to what the fit is actually made
#               against, and readable when the clouds overlap.
#
# Neither is a summary of the other: the bars are quartiles of a strongly
# autocorrelated time series, so they describe the range the case visits over a
# forcing cycle, not the uncertainty in its median.
const STYLE = get(ENV, "STYLE", "both")
STYLE in ("cloud", "errorbars", "both") ||
    error("STYLE must be cloud, errorbars or both — got \"$STYLE\"")

qlo(v) = quantile(filter(isfinite, v), 0.25)
qhi(v) = quantile(filter(isfinite, v), 0.75)

function draw(style)
    ally = sort(reduce(vcat, ([c.l for c in stokes]..., [c.l for c in ekman]...)))
    if style == "cloud"
        ylo = ally[max(1, round(Int, 0.005 * length(ally)))] / 1.5
        yhi = ally[round(Int, 0.999 * length(ally))] * 1.5
    else
        # The bars stop at the quartiles, so the axis can close in on them.
        qs  = reduce(vcat, ([qlo(c.l) for c in stokes], [qlo(c.l) for c in ekman]))
        qh  = reduce(vcat, ([qhi(c.l) for c in stokes], [qhi(c.l) for c in ekman]))
        ylo = minimum(qs) / 1.6
        yhi = maximum(qh) * 1.6
    end

    p = plot(xscale = :log10, yscale = :log10, legend = :bottomright, ylims = (ylo, yhi),
             xlabel = "√TKE / N   (buoyancy scale, m)", ylabel = "l = K_T/√TKE   (m)",
             title = "T = 10 m:  l against √TKE/N at z = h  —  Stokes (tidal) and Ekman",
             size = (980, 720), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
             legendfontsize = 7, foreground_color_legend = nothing)

    if style == "cloud"
        for c in stokes
            scatter!(p, c.x, c.l; ms = 1.6, msw = 0, alpha = 0.40,
                     color = ramp_colour(c.s), label = "")
        end
        for c in ekman
            scatter!(p, c.x, c.l; ms = 2.6, msw = 0.6, alpha = 0.55, marker = :xcross,
                     color = ramp_colour(c.r), msc = ramp_colour(c.r), label = "")
        end
    end

    # Reference lines, identical in both styles.
    xs = reduce(vcat, ([c.x for c in stokes]..., [c.x for c in ekman]...))
    lo, hi = minimum(xs), maximum(xs)
    if style == "errorbars"
        lo = minimum(vcat([qlo(c.x) for c in stokes], [qlo(c.x) for c in ekman]))
        hi = maximum(vcat([qhi(c.x) for c in stokes], [qhi(c.x) for c in ekman]))
    end
    plot!(p, [lo, hi], [lo, hi]; color = :black, lw = 1.2, ls = :dash,
          label = "l = √TKE/N  (1:1)")
    slo = minimum(c.xm for c in stokes); shi = maximum(c.xm for c in stokes)
    xin = exp.(range(log(slo), log(shi); length = 300))
    plot!(p, xin, L∞ .* (1 .- exp.(-xin ./ x0)); color = :black, lw = 2.5,
          label = @sprintf("Stokes fit: l = L∞(1 − e^(−x/x₀)), L∞ = %.2f m, x₀ = %.2f m", L∞, x0))
    xou = exp.(range(log(shi), log(hi); length = 300))
    hi > shi && plot!(p, xou, L∞ .* (1 .- exp.(-xou ./ x0)); color = :black, lw = 1.4,
          ls = :dashdot, label = "Stokes fit, extrapolated past the fitted range")
    hline!(p, [L∞]; color = :black, lw = 1, ls = :dot,
           label = @sprintf("Stokes plateau L∞ = %.2f m", L∞))

    if isfinite(eL)
        elo = minimum(c.xm for c in ek_fit); ehi = maximum(c.xm for c in ek_fit)
        xe = exp.(range(log(elo), log(ehi); length = 300))
        plot!(p, xe, eL .* (1 .- exp.(-xe ./ ex0)); color = :grey30, lw = 2.5, ls = :dash,
              label = @sprintf("Ekman fit (settled cases): L∞ = %.2f m, x₀ = %.2f m", eL, ex0))
    end

    # Only meaningful while the Ekman side lacks its subgrid flux.
    if !FULL_K
        scatter!(p, [c.xr for c in stokes], [c.lr for c in stokes];
                 ms = 7, msw = 1.5, marker = :square, mc = :white, msc = :grey40,
                 label = "Stokes medians, resolved K_T only (Ekman-comparable)")
    end

    set = filter(c -> c.settled, ekman); uns = filter(c -> !c.settled, ekman)

    if style == "errorbars"
        # Bars carry N by colour, so the marker only has to say which flow it is
        # and, for Ekman, whether the case has equilibrated.
        bars(cs, key) = (
            [c.xm for c in cs],
            [c.lm for c in cs],
            ([c.xm - qlo(c.x) for c in cs], [qhi(c.x) - c.xm for c in cs]),
            ([c.lm - qlo(c.l) for c in cs], [qhi(c.l) - c.lm for c in cs]),
            [ramp_colour(getfield(c, key)) for c in cs])
        for (cs, key, mk, ms, lab) in ((stokes, :s, :circle, 7, "Stokes case medians (●)"),
                                       (ekman,  :r, :xcross, 8, "Ekman case medians (✕)"))
            isempty(cs) && continue
            X, Y, XE, YE, C = bars(cs, key)
            # The legend entry is a neutral grey key drawn off-plot, so that the
            # symbol in the legend reads as the shape it is keying and not as
            # one particular case's colour. The real markers carry no label.
            scatter!(p, [NaN], [NaN]; marker = mk, ms = ms, msw = 1.6,
                     mc = :grey70, msc = :black, label = lab)
            # One series per case, so each bar takes its own colour.
            for i in eachindex(X)
                scatter!(p, [X[i]], [Y[i]];
                         xerror = ([XE[1][i]], [XE[2][i]]),
                         yerror = ([YE[1][i]], [YE[2][i]]),
                         marker = mk, ms = ms, msw = 1.6, mc = C[i], msc = :black,
                         linecolor = C[i], lw = 1.6, label = "")
            end
        end
        # The four Ekman cases whose l has not stopped moving. A downward marker
        # tucked just under the lower whisker, because the drift is
        # one-directional and small enough on this axis that a second bar would
        # not be readable. Kept close so it reads as an annotation on that bar
        # rather than as a data point of its own.
        scatter!(p, [NaN], [NaN]; ms = 5, msw = 0, marker = :dtriangle, color = :grey25,
                 label = isempty(uns) ? "" : "▼ Ekman, not equilibrated — l still falling")
        for c in uns
            scatter!(p, [c.xm], [qlo(c.l) / 1.10]; ms = 5, msw = 0, marker = :dtriangle,
                     color = :grey25, label = "")
        end
    else
        # Medians last, so they sit on top of their clouds.
        scatter!(p, [c.xm for c in stokes], [c.lm for c in stokes];
                 ms = 7, msw = 1.5, mc = :white, msc = :black,
                 label = "Stokes case medians")
        isempty(set) || scatter!(p, [c.xm for c in set], [c.lm for c in set];
                 ms = 8, msw = 2.0, marker = :xcross, msc = :black, mc = :black,
                 label = "Ekman case medians (✕), equilibrated")
        # For the cases that have not equilibrated, the bar spans the median of l
        # over the first quarter of the averaging window to the median over the
        # last: the range the value actually moved through, drawn not asserted.
        for (i, c) in enumerate(uns)
            plot!(p, [c.xm, c.xm], [c.lq1, c.lq4]; color = :black, lw = 2.5,
                  label = i == 1 ? "not equilibrated: range l moved through, ▼ = latest" : "")
            scatter!(p, [c.xm], [c.lq4]; ms = 5, msw = 0, marker = :dtriangle,
                     color = :black, label = "")
        end
        isempty(uns) || scatter!(p, [c.xm for c in uns], [c.lm for c in uns];
                 ms = 8, msw = 2.0, marker = :xcross, msc = :black, mc = :black, label = "")
    end

    # One colour key, drawn as invisible series so the N labels appear once.
    for sv in SVALS
        scatter!(p, [NaN], [NaN]; ms = 5, msw = 0, color = ramp_colour(sv),
                 label = @sprintf("N/ω = N/f = %g", sv))
    end

    mkpath(FIGDIR)
    f = joinpath(FIGDIR, style == "cloud" ? "l_vs_q_over_N_ath_T10_combined.png" :
                                            "l_vs_q_over_N_ath_T10_combined_errorbars.png")
    savefig(p, f)
    say("wrote $f")
end

say("")
if !isempty(ekman)
    say("interquartile ranges — the bars in the errorbar figure")
    say("  case          x: q25    med    q75  |  l: q25    med    q75")
    for c in stokes
        say(@sprintf("  Stokes %-5g %7.4f %6.4f %6.4f  | %7.4f %6.4f %6.4f",
                     c.s, qlo(c.x), c.xm, qhi(c.x), qlo(c.l), c.lm, qhi(c.l)))
    end
    for c in ekman
        say(@sprintf("  Ekman  %-5g %7.4f %6.4f %6.4f  | %7.4f %6.4f %6.4f",
                     c.r, qlo(c.x), c.xm, qhi(c.x), qlo(c.l), c.lm, qhi(c.l)))
    end
    say("")
end

for st in (STYLE == "both" ? ("cloud", "errorbars") : (STYLE,))
    draw(st)
end

mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "plot_l_vs_qN_T10_combined.log"), "w") do io
    foreach(l -> println(io, l), logl)
end
