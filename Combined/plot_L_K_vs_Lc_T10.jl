# The one-figure summary: L_K against the combined scale, and which of the two
# scales is setting it where.
#
# plot_Lc_candidates_T10.jl tried seven combinations of L_N and L_s and settled
# on the equal-weight harmonic
#
#     1/L_c = 1/L_N + 1/L_s          L_N = √TKE/N,  L_s = √TKE/S
#
# as the one to prefer: it ties min(L_N, L_s) to within noise, is smooth rather
# than kinked, and has no free parameter to justify. This script draws that one
# candidate only, and next to it the thing the candidate figure could not show —
# how the balance between the two scales shifts with stratification.
#
# ---------------- Why the second panel is the weights ----------------
# Written as 1/L_c = 1/L_N + 1/L_s, the two terms are resistances in series and
# their shares add to one:
#
#     w_N = L_c/L_N       w_s = L_c/L_s       w_N + w_s = 1
#
# w_s is the fraction of 1/L_c contributed by the shear. w_s → 1 is
# shear-limited, w_s → 0 is stratification-limited, and w_s = 1/2 is where the
# two scales are equal (L_s = L_N). That is exactly the regime question, on an
# axis that is bounded and needs no log scale, and it is the same number that
# colours the markers in the left panel — so a point's colour there says which
# regime it came from.
#
# The weight is not a fitted quantity. It is arithmetic on L_N and L_s, so the
# right panel is data, not interpretation.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot_L_K_vs_Lc_T10.jl
#        (run plot_shear_scales_T10.jl first — it writes the cache)

using JLD2, Plots, Printf, Statistics

get!(ENV, "GKSwstype", "100")
const HERE   = @__DIR__
const CACHE  = joinpath(HERE, "Data", "shear_scales_T10.jld2")
const FIGDIR = joinpath(HERE, "figures")
const C_STOK = "#1b3a6b"
const C_EKMA = "#8e1b4e"

fin(v) = filter(isfinite, v)
med(v) = (w = fin(v); isempty(w) ? NaN : median(w))
qlo(v) = (w = fin(v); isempty(w) ? NaN : quantile(w, 0.25))
qhi(v) = (w = fin(v); isempty(w) ? NaN : quantile(w, 0.75))

# Plain numbers on a log axis; 10^0.25 is unreadable over one decade.
function logticks(lo, hi)
    a = floor(Int, log10(lo)); b = ceil(Int, log10(hi))
    vals = Float64[]
    for e in a:b, m in (1, 2, 5)
        v = m * 10.0^e
        lo / 1.05 <= v <= hi * 1.05 && push!(vals, v)
    end
    labs = [v >= 1 ? (v == round(v) ? string(Int(round(v))) : string(v)) :
                     rstrip(rstrip(@sprintf("%.4f", v), '0'), '.') for v in vals]
    return (vals, labs)
end

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))

# ---------------- the cached per-sample scales ----------------
isfile(CACHE) || error("$CACHE not found — run plot_shear_scales_T10.jl first")
stokes, ekman = [], []
jldopen(CACHE, "r") do io
    for (nm, dst) in (("stokes", stokes), ("ekman", ekman))
        for r in io["$nm/ratios"]
            g = @sprintf("%s/r=%.1f", nm, r)
            L_N = io["$g/L_N"]; L_s = io["$g/L_s"]
            # Formed per sample, then reduced. Medians of a ratio and the ratio
            # of medians are not the same thing for series this correlated.
            L_c = 1 ./ (1 ./ L_N .+ 1 ./ L_s)
            push!(dst, (r = r, L_K = io["$g/L_K"], L_N = L_N, L_s = L_s, L_c = L_c,
                        w_N = L_c ./ L_N, w_s = L_c ./ L_s))
        end
    end
end
BOTH = vcat(stokes, ekman)
say(@sprintf("%d Stokes and %d Ekman cases from %s", length(stokes), length(ekman),
             basename(CACHE)))

# ---------------- the two fits drawn ----------------
# b = 1 forced: the proportionality a mixing-length closure would want. A is the
# geometric mean of L_K/L_c and the rms is the spread of that ratio in log space.
function propfit(cs)
    lr = fin([log(med(c.L_K)) - log(med(c.L_c)) for c in cs])
    a = mean(lr)
    return (A = exp(a), rms = 100 * sqrt(mean((lr .- a) .^ 2)),
            f = t -> exp(a) * t)
end
# Free exponent, as the check on the above.
function powfit(cs)
    lx = [log(med(c.L_c)) for c in cs]; ly = [log(med(c.L_K)) for c in cs]
    mx = mean(lx); my = mean(ly)
    b = sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx) .^ 2)
    a = my - b * mx
    res = ly .- (a .+ b .* lx)
    return (A = exp(a), b = b, rms = 100 * sqrt(mean(res .^ 2)))
end
# The house saturating form, brute-forced on a log grid so there is no initial
# guess to get wrong, and reporting whether the best point ran out of grid.
function satfit(cs)
    x = [med(c.L_c) for c in cs]; y = [med(c.L_K) for c in cs]
    ygrid = exp.(range(log(0.3minimum(y)), log(40maximum(y)); length = 260))
    xgrid = exp.(range(log(0.02minimum(x)), log(150maximum(x)); length = 260))
    best = (Inf, 0.0, 0.0)
    for Y in ygrid, x0 in xgrid
        sse = 0.0
        for i in eachindex(x)
            q = Y * (1 - exp(-x[i] / x0))
            q > 0 || (sse = Inf; break)
            sse += (log(y[i]) - log(q))^2
        end
        sse < best[1] && (best = (sse, Y, x0))
    end
    Y, x0 = best[2], best[3]
    pin = Y <= ygrid[2] || Y >= ygrid[end-1] || x0 <= xgrid[2] || x0 >= xgrid[end-1]
    return (L = Y, x0 = x0, rms = 100 * sqrt(best[1] / length(x)), pinned = pin,
            f = t -> Y * (1 - exp(-t / x0)))
end

PR = propfit(BOTH); PW = powfit(BOTH); SA = satfit(BOTH)
say("")
say("L_c = 1/(1/L_N + 1/L_s), on all 13 cases")
say(@sprintf("  proportionality  L_K = %.3f L_c                 rms %.1f %%", PR.A, PR.rms))
say(@sprintf("  power law        L_K = %.3f L_c^%.2f            rms %.1f %%", PW.A, PW.b, PW.rms))
say(@sprintf("  saturating       L_K = %.3f(1 − e^(−L_c/%.3f))  rms %.1f %%%s",
             SA.L, SA.x0, SA.rms, SA.pinned ? "   << PINNED" : ""))
for (nm, cs) in (("Stokes", stokes), ("Ekman ", ekman))
    s = satfit(cs)
    say(@sprintf("  %s alone, saturating:                       rms %.1f %%%s",
                 nm, s.rms, s.pinned ? "   << PINNED" : ""))
end

say("")
say("the shear's share of 1/L_c, w_s = L_c/L_s.  w_s > 1/2 means the shear scale")
say("is the smaller of the two and the mixing is shear-limited")
say("  flow    r       L_N (m)   L_s (m)   L_c (m)   L_K (m)     w_N     w_s")
for (nm, cs) in (("Stokes", stokes), ("Ekman ", ekman)), c in cs
    say(@sprintf("  %s %-5g %9.4f %9.4f %9.4f %9.4f %7.3f %7.3f%s",
                 nm, c.r, med(c.L_N), med(c.L_s), med(c.L_c), med(c.L_K),
                 med(c.w_N), med(c.w_s), med(c.w_s) > 0.5 ? "   << shear-limited" : ""))
end

# ---------------- the figure ----------------
mkpath(FIGDIR)
const CMAP  = cgrad(:RdBu, rev = true)      # blue = stratification, red = shear
const CLIMS = (0.0, 0.6)

# ---- left: the collapse ----
pa = plot(xscale = :log10, yscale = :log10,
          xlabel = "L_c = 1/(1/L_N + 1/L_s)   (m)",
          ylabel = "L_K = K_T/√TKE   (m)",
          title = "one curve through both flows",
          legend = :bottomright, legendfontsize = 6,
          foreground_color_legend = nothing)

xs = [med(c.L_c) for c in BOTH]
xx = exp.(range(log(minimum(xs) / 1.7), log(maximum(xs) * 1.7); length = 300))
plot!(pa, xx, PR.f.(xx); color = :grey55, lw = 1.4, ls = :dot,
      label = @sprintf("L_K = %.3f L_c   (proportional, rms %.0f %%)", PR.A, PR.rms))
SA.pinned || plot!(pa, xx, SA.f.(xx); color = :black, lw = 2.6,
      label = @sprintf("L_K = %.2f(1 − e^(−L_c/%.2f))   rms %.1f %%", SA.L, SA.x0, SA.rms))

# Quartile bars first, without markers, so the coloured markers sit on top.
for cs in (stokes, ekman), c in cs
    x = med(c.L_c); y = med(c.L_K)
    scatter!(pa, [x], [y]; xerror = ([x - qlo(c.L_c)], [qhi(c.L_c) - x]),
             yerror = ([y - qlo(c.L_K)], [qhi(c.L_K) - y]),
             markershape = :none, markeralpha = 0, linecolor = :grey60, lw = 1.3,
             label = "")
end
for (i, (cs, mk, ms, fl)) in enumerate(((stokes, :circle, 8, "Stokes (tidal)"),
                                        (ekman, :diamond, 9, "Ekman (rotating)")))
    scatter!(pa, [med(c.L_c) for c in cs], [med(c.L_K) for c in cs];
             marker_z = [med(c.w_s) for c in cs], c = CMAP, clims = CLIMS,
             marker = mk, ms = ms, msw = 1.5, msc = :black,
             colorbar = (i == 2), colorbar_titlefontsize = 7,
             colorbar_title = "w_s = the shear's share of 1/L_c",
             label = fl)
end
let v = [med(c.L_K) for c in BOTH]
    plot!(pa; xticks = logticks(minimum(xs) / 1.7, maximum(xs) * 1.7),
              yticks = logticks(minimum(v) / 1.7, maximum(v) * 1.7))
end
# The r of each case, so a colour can be read back to a stratification.
for c in BOTH
    annotate!(pa, med(c.L_c), med(c.L_K) / 1.45,
              text(@sprintf("%g", c.r), 5, :grey30, :center))
end

# ---- right: which scale is doing the work ----
# w_N = 1 − w_s, so only w_s is drawn: two lines and a threshold, not four lines
# carrying the same information twice.
pb = plot(xscale = :log10, xlabel = "r = N/ω = N/f",
          ylabel = "w_s = L_c/L_s,  the shear's share of 1/L_c",
          ylims = (0, 0.75), title = "which scale limits the mixing",
          legend = :topright, legendfontsize = 6,
          foreground_color_legend = nothing)
hline!(pb, [0.5]; color = :black, lw = 1.2, ls = :dash,
       label = "L_s = L_N   (the two scales equal)")
for (cs, mk, ms, lc, fl) in ((stokes, :circle, 8, C_STOK, "Stokes (tidal)"),
                             (ekman, :diamond, 9, C_EKMA, "Ekman (rotating)"))
    o = sortperm([c.r for c in cs]); cso = cs[o]
    x = [c.r for c in cso]; y = [med(c.w_s) for c in cso]
    plot!(pb, x, y; color = lc, lw = 1.8, label = fl)
    scatter!(pb, x, y; marker = mk, ms = ms, msw = 1.5,
             marker_z = y, c = CMAP, clims = CLIMS, msc = lc,
             colorbar = false, label = "")
end
let rs = sort(unique(vcat([c.r for c in BOTH])))
    plot!(pb; xticks = (rs, [@sprintf("%g", r) for r in rs]))
end
annotate!(pb, 3.0, 0.545, text("shear-limited above", 6, :grey30, :left))
annotate!(pb, 3.0, 0.455, text("stratification-limited below", 6, :grey30, :left))

f = plot(pa, pb; layout = (1, 2), size = (1250, 560),
         plot_title = "T = 10 m, z = h:  L_K against the combined scale, and where each scale limits it",
         left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, top_margin = 3Plots.mm,
         right_margin = 4Plots.mm)
o = joinpath(FIGDIR, "L_K_vs_Lc_T10.png")
savefig(f, o)
say("")
say("wrote $o")

mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "plot_L_K_vs_Lc_T10.log"), "w") do io
    foreach(l -> println(io, l), logl)
end
