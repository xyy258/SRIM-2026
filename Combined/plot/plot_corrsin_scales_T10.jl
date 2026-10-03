# The L_N/L_s study of plot_shear_scales_T10.jl and plot_Lcomb_candidates_T10.jl,
# redone with the Corrsin scale L_C = (eps/S^3)^(1/2) in place of L_s = sqrt(TKE)/S.
# Writes three figures: the two scales against r, the candidate combinations, and
# the best candidate on its own.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot/plot_corrsin_scales_T10.jl

using JLD2, Plots, Printf, Statistics, LaTeXStrings

get!(ENV, "GKSwstype", "100")
default(dpi = 600, fontfamily = "DejaVu Sans")

const HERE   = dirname(@__DIR__)        # scripts live one level down
include(joinpath(HERE, "sweep.jl"))
const FIGDIR = joinpath(HERE, "figures")
const SFILE  = joinpath(HERE, "Data", "cache", "corrsin_T10.jld2")
const EFILE  = joinpath(HERE, "Data", "cache", "ekman_lengthscales_T10_moments.jld2")
const C_STOK = "#1b3a6b"
const C_EKMA = "#8e1b4e"
const EPS_MIN = 0.5      # a case needs this fraction of samples with eps > 0

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))
fin(v) = filter(isfinite, v)
med(v) = (w = fin(v); isempty(w) ? NaN : median(w))
qlo(v) = (w = fin(v); isempty(w) ? NaN : quantile(w, 0.25))
qhi(v) = (w = fin(v); isempty(w) ? NaN : quantile(w, 0.75))

# One case: per-sample L_K, L_N, L_s, L_C at z = h, from K, TKE, S, eps.
function case(r, N, K, E, S, eps)
    q  = sqrt.(max.(E, 0))
    ok = @. isfinite(K) && K > 0 && isfinite(q) && q > 0 && isfinite(S) && S > 0
    Lc = [ok[i] && isfinite(eps[i]) && eps[i] > 0 ? sqrt(eps[i] / S[i]^3) : NaN
          for i in eachindex(q)]
    frac = count(i -> isfinite(eps[i]) && eps[i] > 0, eachindex(eps)) / length(eps)  # as plot_l_vs_corrsin_T10.jl
    (r = r, N = N, epsok = frac, reliable = frac >= EPS_MIN,
     L_K = [ok[i] ? K[i] / q[i] : NaN for i in eachindex(q)],
     L_N = [ok[i] ? q[i] / N     : NaN for i in eachindex(q)],
     L_s = [ok[i] ? q[i] / S[i]  : NaN for i in eachindex(q)],
     L_C = Lc)
end

S_cases, E_cases = [], []
jldopen(SFILE, "r") do io
    for r in io["ratios"]
        g = @sprintf("r=%.1f", r)
        push!(S_cases, case(r, io["$g/N"], io["$g/K"], io["$g/E"], io["$g/S"], io["$g/eps"]))
    end
end
jldopen(EFILE, "r") do io
    for r in io["ratios"]
        g = @sprintf("r=%.1f", r)
        push!(E_cases, case(r, io["$g/N"], io["$g/K_at_h"], io["$g/TKE_at_h"],
                            io["$g/S_at_h"], io["$g/eps_at_h"]))
    end
end
BOTH = vcat(S_cases, E_cases)

say("per-sample scales at z = h.  L_C = (eps/S^3)^(1/2) exists only where eps > 0")
say("  flow    r      L_N (m)   L_s (m)   L_C (m)   L_K (m)   eps>0 frac")
for (nm, cs) in (("Stokes", S_cases), ("Ekman ", E_cases)), c in cs
    say(@sprintf("  %s %-5g %9.4f %9.4f %9.4f %9.4f %9.2f %s", nm, c.r,
                 med(c.L_N), med(c.L_s), med(c.L_C), med(c.L_K), c.epsok,
                 c.reliable ? "" : "  << eps unusable"))
end

# Candidate combinations of L_N and L_C, matching plot_Lcomb_candidates_T10.jl.
harm(a, b, β) = 1 ./ (1 ./ a .+ β ./ b)
pnorm(a, b, p) = (a .^ (-p) .+ b .^ (-p)) .^ (-1 / p)
geom(a, b, α) = a .^ α .* b .^ (1 - α)

function tune(f, grid)
    best = (Inf, grid[1])
    for g in grid
        x = [med(f(c, g)) for c in BOTH if c.reliable]
        y = [med(c.L_K)   for c in BOTH if c.reliable]
        v = log.(y ./ x); s = sqrt(mean((v .- mean(v)) .^ 2))
        s < best[1] && (best = (s, g))
    end
    return best[2]
end
βb = tune((c, β) -> harm(c.L_N, c.L_C, β), 0.1:0.02:5.0)
pb = tune((c, p) -> pnorm(c.L_N, c.L_C, p), 0.2:0.02:4.0)
αb = tune((c, α) -> geom(c.L_N, c.L_C, α), 0.0:0.01:1.0)

CANDS = [("L_N alone",          c -> c.L_N,                       L"L_N"),
         ("L_C alone",          c -> c.L_C,                       L"L_C"),
         ("min(L_N, L_C)",      c -> min.(c.L_N, c.L_C),          L"\min(L_N, L_C)"),
         ("harmonic, equal",    c -> harm(c.L_N, c.L_C, 1.0),     L"1/L = 1/L_N + 1/L_C"),
         ("harmonic, weighted", c -> harm(c.L_N, c.L_C, βb),
          latexstring(@sprintf("1/L = 1/L_N + %.2f/L_C", βb))),
         ("p-norm",             c -> pnorm(c.L_N, c.L_C, pb),
          latexstring(@sprintf("(L_N^{-%.2f} + L_C^{-%.2f})^{-1/%.2f}", pb, pb, pb))),
         ("geometric",          c -> geom(c.L_N, c.L_C, αb),
          latexstring(@sprintf("L_N^{%.2f} L_C^{%.2f}", αb, 1 - αb)))]

# Fits on the case medians, in log y. Same three forms as the L_s study.
propfit(x, y) = (A = exp(mean(log.(y ./ x))),
                 rms = 100 * sqrt(mean((log.(y ./ x) .- mean(log.(y ./ x))) .^ 2)))
function powfit(x, y)
    lx = log.(x); ly = log.(y); mx = mean(lx); my = mean(ly)
    b = sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx) .^ 2); a = my - b * mx
    (A = exp(a), b = b, rms = 100 * sqrt(mean((ly .- (a .+ b .* lx)) .^ 2)))
end
function satfit(x, y)
    yg = exp.(range(log(0.2minimum(y)), log(60maximum(y)); length = 320))
    xg = exp.(range(log(0.02minimum(x)), log(200maximum(x)); length = 320))
    best = (Inf, 0.0, 0.0)
    for Y in yg, x0 in xg
        s = 0.0; ok = true
        for i in eachindex(x)
            p = Y * (1 - exp(-x[i] / x0))
            p > 0 || (ok = false; break)
            s += (log(y[i]) - log(p))^2
        end
        ok && s < best[1] && (best = (s, Y, x0))
    end
    Y, x0 = best[2], best[3]
    pin = Y <= yg[2] || Y >= yg[end-1] || x0 <= xg[2] || x0 >= xg[end-1]
    (Y = Y, x0 = x0, rms = 100 * sqrt(best[1] / length(x)), pinned = pin)
end

use(cs) = [c for c in cs if c.reliable]
xy(f, cs) = ([med(f(c)) for c in use(cs)], [med(c.L_K) for c in use(cs)])

say("")
say(@sprintf("fits on the case medians of the %d reliable cases of %d", length(use(BOTH)), length(BOTH)))
say("  candidate                  A   prop rms: both  Stokes  Ekman |  power b   rms |  saturating: both Stokes Ekman")
RES = Dict{String,Any}()
for (nm, f, _) in CANDS
    xb, yb = xy(f, BOTH); xs, ys = xy(f, S_cases); xe, ye = xy(f, E_cases)
    pr = propfit(xb, yb); pw = powfit(xb, yb)
    sb = satfit(xb, yb); ss = satfit(xs, ys); se = satfit(xe, ye)
    RES[nm] = (prop = pr, pow = pw, sat = sb)
    say(@sprintf("  %-20s %7.3f %8.1f %% %7.1f %% %7.1f %% | %6.2f %6.1f %% | %7.1f %% %7.1f %% %7.1f %%%s",
                 nm, pr.A, pr.rms, propfit(xs, ys).rms, propfit(xe, ye).rms,
                 pw.b, pw.rms, sb.rms, ss.rms, se.rms, sb.pinned ? "  << PINNED" : ""))
end
BEST = argmin(nm -> RES[nm].sat.pinned ? Inf : RES[nm].sat.rms, [c[1] for c in CANDS])
say("")
say(@sprintf("best saturating curve on both flows: %s, rms %.1f %%", BEST, RES[BEST].sat.rms))

mkpath(FIGDIR)
function logticks(lo, hi)
    a = floor(Int, log10(lo)); b = ceil(Int, log10(hi)); v = Float64[]
    for e in a:b, m in (1, 2, 5)
        x = m * 10.0^e; lo / 1.05 <= x <= hi * 1.05 && push!(v, x)
    end
    (v, [x >= 1 ? (x == round(x) ? string(Int(round(x))) : string(x)) :
         rstrip(rstrip(@sprintf("%.4f", x), '0'), '.') for x in v])
end

# ---- figure 1: L_N and L_C against r ----
# Same convention as L_N_L_s_vs_r_T10.png: colour = flow, style = scale, marker
# colour = r. L_C markers are hollow where the eps estimate is unusable, and the
# dashed line skips those cases.
p1 = plot(xscale = :log10, yscale = :log10, xlabel = L"r = N/\omega = N/f",
          ylabel = L"\mathrm{length\ scale\ at}\ z = h \ \ (\mathrm{m})",
          title = L"L_N\ \mathrm{and}\ L_C", legend = :topright,
          legendfontsize = 6, foreground_color_legend = nothing)
for (cs, mk, ms, lc, fl) in ((S_cases, :circle, 7, C_STOK, "Stokes"),
                             (E_cases, :xcross, 8, C_EKMA, "Ekman"))
    xs = [c.r for c in cs]; o = sortperm(xs)
    plot!(p1, xs[o], [med(c.L_N) for c in cs][o]; color = lc, lw = 1.8,
          label = latexstring("\\mathrm{$fl}\\!: \\ L_N = \\sqrt{\\mathrm{TKE}}/N"))
    scatter!(p1, xs, [med(c.L_N) for c in cs]; marker = mk, ms = ms, msw = 1.6,
             mc = [ramp_colour(r) for r in xs], msc = lc, label = "")
    rel = [c for c in cs if c.reliable]; xr = [c.r for c in rel]; or = sortperm(xr)
    plot!(p1, xr[or], [med(c.L_C) for c in rel][or]; color = lc, lw = 1.8, ls = :dash,
          label = latexstring("\\mathrm{$fl}\\!: \\ L_C = (\\varepsilon/S^3)^{1/2}"))
    scatter!(p1, xs, [med(c.L_C) for c in cs]; marker = mk, ms = ms, msw = 1.6,
             mc = [c.reliable ? ramp_colour(c.r) : "#ffffff" for c in cs],
             msc = lc, label = "")
end
let v = fin(vcat([med(c.L_N) for c in BOTH], [med(c.L_C) for c in BOTH]))
    plot!(p1; yticks = logticks(minimum(v), maximum(v)))
end
for sv in SVALS
    scatter!(p1, [NaN], [NaN]; ms = 5, msw = 0, color = ramp_colour(sv),
             label = @sprintf("N/ω = N/f = %g", sv))
end

p2 = plot(xscale = :log10, yscale = :log10, xlabel = L"r = N/\omega = N/f",
          ylabel = L"L_C/L_N", title = L"\mathrm{which\ scale\ is\ the\ smaller}",
          legend = :bottomright, legendfontsize = 6, foreground_color_legend = nothing)
hline!(p2, [1.0]; color = :black, lw = 1.4, ls = :dash,
       label = L"L_C = L_N \ \ (\mathrm{below\ this\ the\ Corrsin\ scale\ is\ the\ smaller})")
for (cs, mk, ms, lc, fl) in ((S_cases, :circle, 7, C_STOK, "Stokes"),
                             (E_cases, :xcross, 8, C_EKMA, "Ekman"))
    rel = [c for c in cs if c.reliable]; xr = [c.r for c in rel]; or = sortperm(xr)
    plot!(p2, xr[or], [med(c.L_C) / med(c.L_N) for c in rel][or]; color = lc, lw = 1.8,
          label = latexstring("\\mathrm{$fl}"))
    xs = [c.r for c in cs]
    scatter!(p2, xs, [med(c.L_C) / med(c.L_N) for c in cs]; marker = mk, ms = ms, msw = 1.6,
             mc = [c.reliable ? ramp_colour(c.r) : "#ffffff" for c in cs],
             msc = lc, label = "")
end
scatter!(p2, [NaN], [NaN]; ms = 7, msw = 1.6, mc = "#ffffff", msc = :grey40,
         label = L"\mathrm{hollow}\!:\ \varepsilon\ \mathrm{unusable}")
let v = fin([med(c.L_C) / med(c.L_N) for c in BOTH])
    plot!(p2; yticks = logticks(min(0.7, minimum(v)), maximum(v)))
end
f1 = plot(p1, p2; layout = (1, 2), size = (1180, 560),
          plot_title = L"T = 10\,\mathrm{m},\ z = h:\ \ \mathrm{the\ stratification\ and\ Corrsin\ length\ scales\ against}\ r",
          left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, top_margin = 3Plots.mm)
o1 = joinpath(FIGDIR, "L_N_L_C_vs_r_T10.png"); savefig(f1, o1); say("\nwrote $o1")

# ---- figure 2: the candidates ----
function candpanel(nm, f, tex)
    p = plot(xscale = :log10, yscale = :log10, xlabel = L"L_{\mathrm{comb}}\ \ (\mathrm{m})",
             ylabel = L"L_K = K_T/\sqrt{\mathrm{TKE}} \ \ (\mathrm{m})", title = tex,
             legend = false, titlefontsize = 9)
    xb, yb = xy(f, BOTH)
    pr = RES[nm].prop; sa = RES[nm].sat
    xx = exp.(range(log(minimum(xb) / 1.6), log(maximum(xb) * 1.6); length = 200))
    plot!(p, xx, pr.A .* xx; color = :black, lw = 2.0, label = "")
    sa.pinned || plot!(p, xx, sa.Y .* (1 .- exp.(-xx ./ sa.x0)); color = "#c46a1f",
                       lw = 1.8, ls = :dash, label = "")
    for (cs, mk, ms) in ((S_cases, :circle, 7), (E_cases, :xcross, 8))
        for c in cs
            x = med(f(c)); isfinite(x) || continue
            scatter!(p, [x], [med(c.L_K)]; marker = mk, ms = ms, msw = 1.4,
                     msc = c.reliable ? :black : :grey70,
                     mc = c.reliable ? ramp_colour(c.r) : :white, label = "")
        end
    end
    annotate!(p, minimum(xb) / 1.4, maximum(yb) * 1.3,
              text(latexstring(@sprintf("A = %.3f, \\ b\\!=\\!1\\!: %.0f\\,\\%%, \\ \\mathrm{sat}\\!: %s",
                   pr.A, pr.rms, sa.pinned ? "\\mathrm{pinned}" : @sprintf("%.0f\\,\\%%", sa.rms))),
                   7, :grey25, :left))
    plot!(p; xticks = logticks(minimum(xb) / 1.6, maximum(xb) * 1.6),
             yticks = logticks(minimum(yb) / 1.6, maximum(yb) * 1.6))
    return p
end
panels = [candpanel(nm, f, tex) for (nm, f, tex) in CANDS]
pk = plot(framestyle = :none, legend = :left, legendfontsize = 7,
          foreground_color_legend = nothing)
scatter!(pk, [NaN], [NaN]; ms = 7, msw = 1.4, mc = :grey70, msc = :black, label = L"\mathrm{Stokes\ (tidal)}")
scatter!(pk, [NaN], [NaN]; marker = :xcross, ms = 8, msw = 1.4, msc = :black, label = L"\mathrm{Ekman\ (rotating)}")
scatter!(pk, [NaN], [NaN]; ms = 7, msw = 1.4, mc = :white, msc = :grey70, label = L"\mathrm{hollow}\!:\ \varepsilon\ \mathrm{unusable,\ not\ fitted}")
plot!(pk, [NaN], [NaN]; color = :black, lw = 2.0, label = L"L_K = A\,L_{\mathrm{comb}}")
plot!(pk, [NaN], [NaN]; color = "#c46a1f", lw = 1.8, ls = :dash, label = L"L_K = L_\infty(1 - e^{-L_{\mathrm{comb}}/x_0})")
for sv in SVALS
    scatter!(pk, [NaN], [NaN]; ms = 5, msw = 0, color = ramp_colour(sv),
             label = latexstring(@sprintf("N/\\omega = N/f = %g", sv)))
end
push!(panels, pk)
f2 = plot(panels...; layout = (2, 4), size = (1500, 760),
          plot_title = L"T = 10\,\mathrm{m},\ z = h:\ \ L_K\ \mathrm{against\ combinations\ of}\ L_N\ \mathrm{and\ the\ Corrsin\ scale}\ L_C",
          left_margin = 5Plots.mm, bottom_margin = 5Plots.mm, top_margin = 2Plots.mm)
o2 = joinpath(FIGDIR, "L_K_vs_Lcomb_corrsin_T10.png"); savefig(f2, o2); say("wrote $o2")

# ---- figure 3: the best candidate on its own ----
fbest = CANDS[findfirst(c -> c[1] == BEST, CANDS)][2]
texbest = CANDS[findfirst(c -> c[1] == BEST, CANDS)][3]
sa = RES[BEST].sat; pr = RES[BEST].prop
xb, yb = xy(fbest, BOTH)
p3 = plot(xscale = :log10, yscale = :log10,
          xlabel = latexstring("L_{\\mathrm{comb}} = " * texbest.s[2:end-1] * "\\ \\ (\\mathrm{m})"),
          ylabel = L"L_K = K_T/\sqrt{\mathrm{TKE}} \ \ (\mathrm{m})",
          title = L"\mathrm{one\ curve\ through\ both\ flows}",
          legend = :bottomright, legendfontsize = 7, foreground_color_legend = nothing)
xx = exp.(range(log(minimum(xb) / 1.8), log(maximum(xb) * 1.8); length = 300))
plot!(p3, xx, pr.A .* xx; color = :grey55, lw = 1.4, ls = :dot,
      label = latexstring(@sprintf("L_K = %.3f\\,L_{\\mathrm{comb}} \\ (\\mathrm{rms}\\ %.0f\\,\\%%)", pr.A, pr.rms)))
sa.pinned || plot!(p3, xx, sa.Y .* (1 .- exp.(-xx ./ sa.x0)); color = :black, lw = 2.4,
      label = latexstring(@sprintf("L_K = %.3f(1 - e^{-L_{\\mathrm{comb}}/%.3f}), \\ \\mathrm{rms}\\ %.1f\\,\\%%", sa.Y, sa.x0, sa.rms)))
for (cs, mk, ms, lc, fl) in ((S_cases, :circle, 9, C_STOK, "Stokes (tidal)"),
                             (E_cases, :diamond, 9, C_EKMA, "Ekman (rotating)"))
    for c in cs
        x = med(fbest(c)); isfinite(x) || continue
        plot!(p3, [qlo(fbest(c)), qhi(fbest(c))], [med(c.L_K), med(c.L_K)];
              color = :grey70, lw = 1.0, label = "")
        plot!(p3, [x, x], [qlo(c.L_K), qhi(c.L_K)]; color = :grey70, lw = 1.0, label = "")
        scatter!(p3, [x], [med(c.L_K)]; marker = mk, ms = ms, msw = 1.5,
                 msc = c.reliable ? lc : :grey70,
                 mc = c.reliable ? ramp_colour(c.r) : :white, label = "")
    end
    scatter!(p3, [NaN], [NaN]; marker = mk, ms = ms, msw = 1.5, mc = :grey70, msc = lc,
             label = latexstring("\\mathrm{$(replace(fl, " " => "\\ "))}"))
end
scatter!(p3, [NaN], [NaN]; ms = 8, msw = 1.5, mc = :white, msc = :grey70,
         label = L"\mathrm{hollow}\!:\ \varepsilon\ \mathrm{unusable,\ not\ fitted}")
plot!(p3; xticks = logticks(minimum(xb) / 1.8, maximum(xb) * 1.8),
          yticks = logticks(minimum(yb) / 1.8, maximum(yb) * 1.8))
f3 = plot(p3; size = (900, 780),
          plot_title = L"T = 10\,\mathrm{m},\ z = h:\ \ L_K\ \mathrm{against\ the\ best\ combination\ of}\ L_N\ \mathrm{and}\ L_C",
          left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, top_margin = 3Plots.mm)
o3 = joinpath(FIGDIR, "L_K_vs_Lbest_corrsin_T10.png"); savefig(f3, o3); say("wrote $o3")

mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "plot_corrsin_scales_T10.log"), "w") do io
    println(io, join(logl, "\n"))
end
