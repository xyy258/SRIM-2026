# The two l-vs-Corrsin figures redone against the shear scale L_s = sqrt(TKE)/S.
# Same construction, same conventions: cloud + case medians, and medians with
# interquartile bars.
#
# Everything comes from Data/shear_scales_T10.jld2, which already holds
# per-sample L_K and L_s at z = h for all 16 cases. Unlike L_C there is no eps
# to fail, so every case is usable and there are no hollow markers.
#
# The fit grids are wider than the Corrsin script's: L_s is a larger scale
# (0.63 to 13.7 m against 0.05 to 4.5), so the saturating knee has further to go.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot/plot_l_vs_Ls_T10.jl
# ENV    STYLE  cloud | errorbars | both (default both)

using JLD2, Plots, Printf, Statistics, LaTeXStrings

get!(ENV, "GKSwstype", "100")
default(dpi = 600, fontfamily = "DejaVu Sans")

const HERE   = dirname(@__DIR__)        # scripts live one level down
include(joinpath(HERE, "sweep.jl"))
const CACHE  = joinpath(HERE, "Data", "cache", "shear_scales_T10.jld2")
const FIGDIR = joinpath(HERE, "figures")
const STYLE  = get(ENV, "STYLE", "both")
const C_STOK = "#1b3a6b"
const C_EKMA = "#8e1b4e"

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))
fin(v) = filter(isfinite, v)
med(v) = (w = fin(v); isempty(w) ? NaN : median(w))
qlo(v) = (w = fin(v); isempty(w) ? NaN : quantile(w, 0.25))
qhi(v) = (w = fin(v); isempty(w) ? NaN : quantile(w, 0.75))

isfile(CACHE) || error("$CACHE not found — run plot_shear_scales_T10.jl first")
S, E = [], []
jldopen(CACHE, "r") do io
    for (fl, dst) in (("stokes", S), ("ekman", E))
        for r in io["$fl/ratios"]
            g = @sprintf("%s/r=%.1f", fl, r)
            l = io["$g/L_K"]; x = io["$g/L_s"]
            k = @. isfinite(l) && l > 0 && isfinite(x) && x > 0
            push!(dst, (r = r, l = l[k], x = x[k], lm = med(l[k]), xm = med(x[k]), n = count(k)))
        end
    end
end
say(@sprintf("%d Stokes and %d Ekman cases from %s", length(S), length(E), basename(CACHE)))

const L_GRID  = 0.05:0.005:8.00
const X0_GRID = 0.02:0.02:60.0
function fit_sat(cs)
    length(cs) >= 4 || return nothing
    best = (Inf, 0.0, 0.0)
    for L in L_GRID, x0 in X0_GRID
        sse = 0.0
        for c in cs
            p = L * (1 - exp(-c.xm / x0))
            p > 0 || (sse = Inf; break)
            sse += (log(c.lm) - log(p))^2
        end
        sse < best[1] && (best = (sse, L, x0))
    end
    # Pinned on a grid edge is not a fit. Nor is an x0 far above the data: the
    # saturating form then has no knee inside the range and is a straight line
    # with two redundant parameters — better rms, no meaning. Same reasoning as
    # plot_shear_scales_T10.jl uses against tau_s.
    xmax = maximum(c.xm for c in cs)
    pinned = best[2] in (first(L_GRID), last(L_GRID)) ||
             best[3] in (first(X0_GRID), last(X0_GRID)) ||
             best[3] >= 2 * xmax
    (L = best[2], x0 = best[3], pinned = pinned,
     rms = 100 * sqrt(best[1] / length(cs)), n = length(cs))
end
function fit_pow(cs)
    lx = [log(c.xm) for c in cs]; ly = [log(c.lm) for c in cs]
    mx = mean(lx); my = mean(ly)
    b = sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx) .^ 2)
    a = my - b * mx
    (A = exp(a), b = b, rms = 100 * sqrt(mean((ly .- (a .+ b .* lx)) .^ 2)), n = length(cs))
end

F_S = fit_sat(S); F_E = fit_sat(E); F_A = fit_sat(vcat(S, E))
P_S = fit_pow(S); P_E = fit_pow(E); P_A = fit_pow(vcat(S, E))
say("")
say(@sprintf("fits of l against L_s, on the case medians of all %d cases", length(S) + length(E)))
say("  set       saturating  l = L∞(1 − e^(−x/x₀))            power law  l = A x^b")
for (nm, fs, fp) in (("Stokes", F_S, P_S), ("Ekman ", F_E, P_E), ("both  ", F_A, P_A))
    say(@sprintf("  %s  L∞ = %5.2f m, x₀ = %6.2f m, rms %5.1f %%%s  A = %.3f, b = %5.2f, rms %5.1f %%",
                 nm, fs.L, fs.x0, fs.rms, fs.pinned ? " << no knee " : "          ", fp.A, fp.b, fp.rms))
end
say("")
say("medians at z = h")
say("  flow    r        L_s (m)      l (m)     l/L_s   samples")
for (nm, cs) in (("Stokes", S), ("Ekman ", E)), c in cs
    say(@sprintf("  %s %-5g %11.4f %10.4f %9.4f %8d", nm, c.r, c.xm, c.lm, c.lm / c.xm, c.n))
end

mkpath(FIGDIR)
function draw(style)
    ally = sort(reduce(vcat, ([c.l for c in S]..., [c.l for c in E]...)))
    ylo, yhi = if style == "cloud"
        ally[max(1, round(Int, 0.005 * length(ally)))] / 1.5, ally[round(Int, 0.999 * length(ally))] * 1.5
    else
        minimum(vcat([qlo(c.l) for c in S], [qlo(c.l) for c in E])) / 1.6,
        maximum(vcat([qhi(c.l) for c in S], [qhi(c.l) for c in E])) * 1.6
    end

    p = plot(xscale = :log10, yscale = :log10, legend = :bottomright, ylims = (ylo, yhi),
             xlabel = L"L_s = \sqrt{\mathrm{TKE}}/S \ \ (\mathrm{shear\ scale,\ m})",
             ylabel = L"\ell = K_T/\sqrt{\mathrm{TKE}} \ \ (\mathrm{m})",
             title = L"T = 10\,\mathrm{m}:\ \ \ell\ \mathrm{against\ the\ shear\ scale\ at}\ z = h\ \ (\mathrm{Stokes\ and\ Ekman})",
             size = (980, 720), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
             legendfontsize = 7, foreground_color_legend = nothing)

    if style == "cloud"
        for c in S
            scatter!(p, c.x, c.l; ms = 1.6, msw = 0, alpha = 0.40, color = ramp_colour(c.r), label = "")
        end
        for c in E
            scatter!(p, c.x, c.l; ms = 2.6, msw = 0.6, alpha = 0.55, marker = :xcross,
                     color = ramp_colour(c.r), msc = ramp_colour(c.r), label = "")
        end
    end

    xs = reduce(vcat, ([c.x for c in S]..., [c.x for c in E]...))
    lo, hi = minimum(xs), maximum(xs)
    style == "errorbars" && ((lo, hi) = (minimum(vcat([qlo(c.x) for c in S], [qlo(c.x) for c in E])),
                                         maximum(vcat([qhi(c.x) for c in S], [qhi(c.x) for c in E]))))
    plot!(p, [lo, hi], [lo, hi]; color = :black, lw = 1.2, ls = :dash, label = L"\ell = L_s \ \ (1{:}1)")

    function fitline(fs, fp, cs, col, ls, lw, nm)
        isempty(cs) && return
        a = minimum(c.xm for c in cs); b = maximum(c.xm for c in cs)
        xx = exp.(range(log(a), log(b); length = 300))
        if fs !== nothing && !fs.pinned && fs.rms <= fp.rms
            plot!(p, xx, fs.L .* (1 .- exp.(-xx ./ fs.x0)); color = col, ls = ls, lw = lw,
                  label = latexstring(@sprintf("%s\\!: \\ \\ell = %.2f(1 - e^{-L_s/%.2f}), \\ \\mathrm{rms}\\ %.0f\\,\\%%",
                                               nm, fs.L, fs.x0, fs.rms)))
        else
            plot!(p, xx, fp.A .* xx .^ fp.b; color = col, ls = ls, lw = lw,
                  label = latexstring(@sprintf("%s\\!: \\ \\ell = %.3f\\,L_s^{%.2f}, \\ \\mathrm{rms}\\ %.0f\\,\\%%",
                                               nm, fp.A, fp.b, fp.rms)))
        end
    end
    fitline(F_S, P_S, S, C_STOK, :dash, 2.0, "\\mathrm{Stokes\\ fit}")
    fitline(F_E, P_E, E, C_EKMA, :dash, 2.0, "\\mathrm{Ekman\\ fit}")
    fitline(F_A, P_A, vcat(S, E), :black, :solid, 2.6, "\\mathrm{overall\\ fit}")

    for (cs, mk, ms, nm, sym) in ((S, :circle, 8, "Stokes", "N/\\omega"),
                                  (E, :diamond, 9, "Ekman", "N/f"))
        for c in cs
            lab = latexstring(@sprintf("\\mathrm{%s}\\ \\ %s = %g", nm, sym, c.r))
            if style == "cloud"
                scatter!(p, [c.xm], [c.lm]; marker = mk, ms = ms, msw = 1.6,
                         color = ramp_colour(c.r), msc = ramp_colour(c.r), label = lab)
            else
                scatter!(p, [c.xm], [c.lm];
                         xerror = ([c.xm - qlo(c.x)], [qhi(c.x) - c.xm]),
                         yerror = ([c.lm - qlo(c.l)], [qhi(c.l) - c.lm]),
                         marker = mk, ms = ms, msw = 1.6, color = ramp_colour(c.r),
                         msc = ramp_colour(c.r), linecolor = ramp_colour(c.r), lw = 1.6, label = lab)
            end
        end
    end

    f = joinpath(FIGDIR, style == "cloud" ? "l_vs_shear_ath_T10_combined.png" :
                                            "l_vs_shear_ath_T10_combined_errorbars.png")
    savefig(p, f)
    say("wrote $f")
end

say("")
for st in (STYLE == "both" ? ("cloud", "errorbars") : (STYLE,))
    draw(st)
end

mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "plot_l_vs_Ls_T10.log"), "w") do io
    foreach(l -> println(io, l), logl)
end
