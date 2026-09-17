# The Ekman counterpart of plot_KTstar_Ri_stokes_T10.jl: K_T* = K_T N/TKE against
# sqrt(Ri) = N/S at z = h on the background N, same harmonic form
# K_T* = A sqrt(Ri)/(1 + sqrt(Ri)).
#
# It is drawn knowing it does not work. S ~ N^1.018 at z = h in this column, so
# Ri is pinned near 10 and six of the eight cases pile up between Ri = 8 and 13
# — there is almost no x-range to fit across. The rough-bed r = 25 run
# (c_D x4, LOG.txt 2026-09-17) is plotted alongside as an open marker: it lands
# on top of the smooth r = 25 point, which is the same statement again.
# The Stokes fit is drawn for reference; it is NOT fitted to these points.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot/plot_KTstar_Ri_ekman_T10.jl

using JLD2, Plots, Printf, Statistics, LaTeXStrings

get!(ENV, "GKSwstype", "100")
default(dpi = 600, fontfamily = "DejaVu Sans")

const HERE   = dirname(@__DIR__)        # scripts live one level down
include(joinpath(HERE, "sweep.jl"))
const FIGDIR = joinpath(HERE, "figures")
const EKFILE = joinpath(HERE, "Data", "cache", "ekman_lengthscales_T10_moments.jld2")
const ROUGH  = joinpath(HERE, "Data", "cache", "ekman_lengthscales_rough_T10.jld2")
const STOKES_A = 0.412                 # from plot_KTstar_Ri_stokes_T10.jl
const C_EKMA = "#8e1b4e"

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))
fin(v) = filter(isfinite, v)
med(v) = (w = fin(v); isempty(w) ? NaN : median(w))
qlo(v) = (w = fin(v); isempty(w) ? NaN : quantile(w, 0.25))
qhi(v) = (w = fin(v); isempty(w) ? NaN : quantile(w, 0.75))

function grab(file, r)
    jldopen(file, "r") do io
        g = @sprintf("r=%.1f", r)
        N, K, E, S = io["$g/N"], io["$g/K_at_h"], io["$g/TKE_at_h"], io["$g/S_at_h"]
        ok = @. isfinite(K) && K > 0 && isfinite(E) && E > 0 && isfinite(S) && S > 0
        (r = r, K = [ok[i] ? K[i] * N / E[i] : NaN for i in eachindex(K)],
                x = [ok[i] ? N / S[i]        : NaN for i in eachindex(K)])
    end
end

cases = [grab(EKFILE, r) for r in RATIOS]
rough = grab(ROUGH, 25.0)

function fit(cs)
    x = [med(c.x) for c in cs]; y = [med(c.K) for c in cs]
    k = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
    lr = log.(y[k] ./ (x[k] ./ (1 .+ x[k])))
    (A = exp(mean(lr)), rms = 100 * sqrt(mean((lr .- mean(lr)) .^ 2)), n = count(k))
end
f = fit(cases)
say(@sprintf("Ekman, z = h, background N:  K_T* = %.4f sqrt(Ri)/(1+sqrt(Ri))  rms %.1f %%  on %d cases",
             f.A, f.rms, f.n))
say(@sprintf("  (Stokes, same construction: A = %.3f, rms 14.7 %%)", STOKES_A))
say("")
say("  r        sqrt(Ri)      Ri        K_T*      fit       ratio")
for c in vcat(cases, rough)
    x = med(c.x); y = med(c.K); p = f.A * x / (1 + x)
    say(@sprintf("  %-7s %10.4f %9.3f %9.4f %9.4f %9.3f",
                 c === rough ? "25 rgh" : @sprintf("%g", c.r), x, x^2, y, p, y / p))
end
let xs = fin([med(c.x)^2 for c in cases])
    say(@sprintf("\n  Ri spans %.3f to %.2f  (x%.0f) — and %d of %d cases sit between 8 and 13",
                 minimum(xs), maximum(xs), maximum(xs) / minimum(xs),
                 count(v -> 8 <= v <= 13, xs), length(xs)))
end

mkpath(FIGDIR)
function logticks(lo, hi)
    a = floor(Int, log10(lo)); b = ceil(Int, log10(hi)); v = Float64[]
    for e in a:b, m in (1, 2, 5)
        x = m * 10.0^e; lo / 1.05 <= x <= hi * 1.05 && push!(v, x)
    end
    (v, [x >= 1 ? (x == round(x) ? string(Int(round(x))) : string(x)) :
         rstrip(rstrip(@sprintf("%.4f", x), '0'), '.') for x in v])
end

xs = [med(c.x) for c in cases]; ys = [med(c.K) for c in cases]
lo, hi = minimum(xs) / 2.5, maximum(xs) * 2.5
ylo, yhi = minimum(ys) / 2.5, maximum(ys) * 2.2
p = plot(xscale = :log10, yscale = :log10,
         xlabel = L"\sqrt{Ri} = N/S \ \ (N = r\,f,\ \mathrm{background})",
         ylabel = L"K_T^{*} = K_T N/\mathrm{TKE}",
         legend = :bottomright, legendfontsize = 6, foreground_color_legend = nothing)
xx = exp.(range(log(lo), log(hi); length = 300))
plot!(p, xx, STOKES_A .* xx ./ (1 .+ xx); color = :grey55, lw = 1.6, ls = :dot,
      label = latexstring(@sprintf("\\mathrm{Stokes\\ fit\\ for\\ reference}\\!: \\ A = %.3f", STOKES_A)))
plot!(p, xx, f.A .* xx ./ (1 .+ xx); color = :black, lw = 2.4,
      label = latexstring(@sprintf("K_T^{*} = %.3f\\,\\sqrt{Ri}/(1+\\sqrt{Ri}), \\ \\mathrm{rms}\\ %.1f\\,\\%%", f.A, f.rms)))
# The pile-up is the point: shade where six of the eight cases sit.
vspan!(p, [sqrt(8.0), sqrt(13.0)]; color = "#8e1b4e", alpha = 0.07, label = "")
for (i, c) in enumerate(cases)
    plot!(p, [qlo(c.x), qhi(c.x)], [ys[i], ys[i]]; color = :grey70, lw = 1.0, label = "")
    plot!(p, [xs[i], xs[i]], [qlo(c.K), qhi(c.K)]; color = :grey70, lw = 1.0, label = "")
    scatter!(p, [xs[i]], [ys[i]]; ms = 9, msw = 1.5, mc = ramp_colour(c.r), msc = :black, label = "")
end
scatter!(p, [med(rough.x)], [med(rough.K)]; marker = :diamond, ms = 11, msw = 1.8,
         mc = "#ffffff", msc = "#c46a1f", label = L"r = 25,\ \mathrm{rough\ bed}\ (c_D \times 4)")
for sv in RATIOS
    scatter!(p, [NaN], [NaN]; ms = 5, msw = 0, color = ramp_colour(sv), label = @sprintf("N/f = %g", sv))
end
plot!(p; xticks = logticks(lo, hi), xlims = (lo, hi),
         yticks = logticks(ylo, yhi), ylims = (ylo, yhi))
annotate!(p, lo * 1.12, yhi / 1.06,
          text(@sprintf("Six of the eight cases (r = 1 to 50) sit in the shaded band,\nRi = 8 to 13, because S ~ N^1.018 here: the layer equilibrates\nto a marginal Ri at its top, so r does not control Ri.\nThe rough-bed run (u_* x1.18) lands on the smooth r = 25 point.\nOnly r = 0.2 and 0.5 reach below — there is no curve to fit."),
               6, :grey30, :left, :top))
fig = plot(p; size = (920, 800),
           plot_title = L"T = 10\,\mathrm{m},\ z = h,\ \mathrm{Ekman}:\ \ K_T^{*} = A\sqrt{Ri}/(1+\sqrt{Ri})",
           bottom_margin = 6Plots.mm, left_margin = 6Plots.mm, top_margin = 3Plots.mm)
o = joinpath(FIGDIR, "KTstar_vs_Ri_ekman_T10.png"); savefig(fig, o); say("\nwrote $o")

mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "plot_KTstar_Ri_ekman_T10.log"), "w") do io
    println(io, join(logl, "\n"))
end
