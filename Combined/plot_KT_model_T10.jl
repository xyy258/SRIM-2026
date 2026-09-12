# Stage 3: the delta model for K_T.
#
#     K_T = B1 delta sqrt(TKE) (B2 + exp(B3 L_harm/delta))
#
# Divide by sqrt(TKE) and the model is a statement about the mixing length,
#
#     L_K/delta = C1 (1 - exp(-C2 L_harm/delta))       C1 = |B1 B2|, C2 = |B3|
#
# which is the saturating form already fitted in plot_L_K_vs_Lharm_T10.jl, with
# ONE new claim: that the plateau and the knee both scale with delta rather than
# being fitted per flow. (Written as (B2 + exp(B3 x)) the bracket only gives a
# saturating curve with B1 < 0, B2 = -1, B3 < 0; the form above is the same
# thing with the signs resolved.)
#
# So the test is sharp and needs no new fit machinery: does L_K/delta against
# L_harm/delta collapse BOTH flows onto one curve, better than L_K against
# L_harm did unscaled? The unscaled number to beat is 10.4 % on all 13 cases.
#
# ---------------- What the Stokes column can and cannot say ----------------
# The largest Stokes L_harm is 0.23 x0 of the Ekman knee, so exp(-L_harm/x0)
# stays above 0.80 across the whole tidal sweep: Stokes never reaches the bend
# and cannot constrain C2. It constrains C1 C2 — the initial slope — and that
# is worth knowing on its own, because the Ekman initial slope L_inf/x0 = 0.423
# and the Stokes proportionality 0.449 already agree to 6 %.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot_KT_model_T10.jl
#        (run plot_shear_scales_T10.jl, reduce_profiles_T10.jl, plot_delta_T10.jl first)

using JLD2, Plots, Printf, Statistics, LaTeXStrings

get!(ENV, "GKSwstype", "100")
default(dpi = 600, fontfamily = "DejaVu Sans")

const HERE   = @__DIR__
const SCALES = joinpath(HERE, "Data", "shear_scales_T10.jld2")
const DFILE  = joinpath(HERE, "Data", "delta_T10.jld2")
const PFILE  = joinpath(HERE, "Data", "profiles_T10.jld2")
const FIGDIR = joinpath(HERE, "figures")
const SVALS  = [1, 2, 5, 10, 25, 50]
const C_STOK = "#1b3a6b"
const C_EKMA = "#8e1b4e"

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
fin(v) = filter(isfinite, v)
med(v) = (w = fin(v); isempty(w) ? NaN : median(w))
qlo(v) = (w = fin(v); isempty(w) ? NaN : quantile(w, 0.25))
qhi(v) = (w = fin(v); isempty(w) ? NaN : quantile(w, 0.75))

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

# Three ways of building delta, because they answer different questions.
#   published  the constants as given, 0.4 u_*/omega and 1.3 u_*/f. This is the
#              model as proposed — and note the two constants are NOT mutually
#              calibrated: they put h/delta at 2.62 for Stokes and 0.66 for
#              Ekman, so the two flows' delta differ by a factor of five for a
#              reason that has nothing to do with the physics.
#   common     the same formula with ONE constant, c = 1 for both flows. If the
#              thickness idea is right but the published constants are not, this
#              is where it shows.
#   h          the measured mixed-layer height, the baseline. delta is flat
#              against h to 1.6 % and 4.4 % (stage 1), so this is `common`
#              rescaled and is here to make that explicit.
const P_S, P_E = 0.040, 0.155           # stage-1 exponents
δs = Dict{Tuple{String,Float64},NamedTuple}()
jldopen(PFILE, "r") do io
    for fl in ("stokes", "ekman"), r in io["$fl/ratios"]
        g = @sprintf("%s/r=%.1f", fl, r)
        us = io["$g/us"]; N = io["$g/N"]; h = io["$g/h"]
        p  = fl == "stokes" ? P_S : P_E
        c0 = fl == "stokes" ? 0.4 : 1.3
        base = us / 1e-4 * (1 + (N / 1e-4)^2)^(-p)
        δs[(fl, r)] = (published = c0 * base, common = base, h = h)
    end
end
S, E = [], []
jldopen(SCALES, "r") do io
    for (fl, dst) in (("stokes", S), ("ekman", E))
        for r in io["$fl/ratios"]
            g = @sprintf("%s/r=%.1f", fl, r)
            L_K = io["$g/L_K"]; L_N = io["$g/L_N"]; L_s = io["$g/L_s"]
            L_h = 1 ./ (1 ./ L_N .+ 1 ./ L_s)
            N = (fl == "stokes" ? r : r) * 1e-4
            # sqrt(TKE) = L_N N, so K_T = L_K sqrt(TKE) without a second read.
            push!(dst, (flow = fl, r = r, N = N, δ = δs[(fl, r)].published,
                        δall = δs[(fl, r)],
                        L_K = L_K, L_h = L_h, K = L_K .* L_N .* N))
        end
    end
end
BOTH = vcat(S, E)
say(@sprintf("%d Stokes and %d Ekman cases", length(S), length(E)))

# ---------------- the fit ----------------
# Brute force in log y on the case medians, as everywhere else here, with the
# grid-edge check that has caught two silent failures already.
function satfit(x, y)
    yg = exp.(range(log(0.2minimum(y)), log(60maximum(y)); length = 320))
    xg = exp.(range(log(0.02minimum(x)), log(200maximum(x)); length = 320))
    best = (Inf, 0.0, 0.0)
    for Y in yg, x0 in xg
        s = 0.0; ok = true
        for i in eachindex(x)
            q = Y * (1 - exp(-x[i] / x0))
            q > 0 || (ok = false; break)
            s += (log(y[i]) - log(q))^2
        end
        ok && s < best[1] && (best = (s, Y, x0))
    end
    Y, x0 = best[2], best[3]
    pin = Y <= yg[2] || Y >= yg[end-1] || x0 <= xg[2] || x0 >= xg[end-1]
    return (C1 = Y, C2 = 1 / x0, x0 = x0, rms = 100 * sqrt(best[1] / length(x)),
            pinned = pin, n = length(x))
end

xs(cs) = [med(c.L_h) / c.δ for c in cs]
ys(cs) = [med(c.L_K) / c.δ for c in cs]
say("")
say("scaled by delta:  L_K/delta = C1 (1 - exp(-C2 L_harm/delta))")
say("  set       C1        C2      initial slope C1 C2    rms")
FITS = Dict{Symbol,Any}()
for (k, cs) in ((:stokes, S), (:ekman, E), (:both, BOTH))
    f = satfit(xs(cs), ys(cs)); FITS[k] = f
    say(@sprintf("  %-8s %.4f %9.3f %16.3f %10.1f %%%s", string(k), f.C1, f.C2,
                 f.C1 * f.C2, f.rms, f.pinned ? "   << PINNED, not a fit" : ""))
end
say("")
say("for comparison, the same form UNSCALED, L_K = L_inf(1 - exp(-L_harm/x0)):")
say("  both flows 10.4 %, Stokes alone 9.1 %, Ekman alone 8.9 %   (plot_L_K_vs_Lharm_T10.jl)")

# Does the choice of delta rescue it? The unscaled fit is the null: if no delta
# beats 10.4 % on both flows, dividing by a thickness is not buying anything.
say("")
say("which delta, if any, collapses the two flows?  (rms on all 13 cases)")
say("  delta                                  both    Stokes   Ekman")
for key in (:published, :common, :h)
    xk(cs) = [med(c.L_h) / getfield(c.δall, key) for c in cs]
    yk(cs) = [med(c.L_K) / getfield(c.δall, key) for c in cs]
    fb = satfit(xk(BOTH), yk(BOTH)); fs = satfit(xk(S), yk(S)); fe = satfit(xk(E), yk(E))
    say(@sprintf("  %-36s %6.1f %% %7.1f %% %7.1f %%%s", string(key),
                 fb.rms, fs.rms, fe.rms, fb.pinned ? "   << PINNED" : ""))
end
say("  unscaled (no delta at all)             10.4 %     9.1 %     8.9 %")

# ---------------- the figure ----------------
mkpath(FIGDIR)
pa = plot(xscale = :log10, yscale = :log10,
          xlabel = L"L_{\mathrm{harm}}/\delta", ylabel = L"L_K/\delta",
          title = L"\mathrm{does\ scaling\ by}\ \delta\ \mathrm{collapse\ the\ two\ flows?}",
          legend = :bottomright, legendfontsize = 6, foreground_color_legend = nothing)
allx = vcat(xs(S), xs(E))
xx = exp.(range(log(minimum(allx) / 1.8), log(maximum(allx) * 1.8); length = 300))
let f = FITS[:both]
    f.pinned || plot!(pa, xx, f.C1 .* (1 .- exp.(-f.C2 .* xx)); color = :black, lw = 2.6,
        label = latexstring(@sprintf("L_K/\\delta = %.3f(1 - e^{-%.2f L_{\\mathrm{harm}}/\\delta}), \\ \\mathrm{rms}\\ %.1f\\,\\%%",
                                     f.C1, f.C2, f.rms)))
end
for (cs, mk, ms, lc, fl) in ((S, :circle, 8, C_STOK, "Stokes\\ (tidal)"),
                             (E, :diamond, 9, C_EKMA, "Ekman\\ (rotating)"))
    for c in cs
        x = med(c.L_h) / c.δ; y = med(c.L_K) / c.δ
        scatter!(pa, [x], [y];
                 xerror = ([x - qlo(c.L_h) / c.δ], [qhi(c.L_h) / c.δ - x]),
                 yerror = ([y - qlo(c.L_K) / c.δ], [qhi(c.L_K) / c.δ - y]),
                 marker = mk, ms = ms, msw = 1.5, mc = ramp_colour(c.r), msc = lc,
                 linecolor = :grey60, lw = 1.2, label = "")
    end
    scatter!(pa, [NaN], [NaN]; marker = mk, ms = ms, msw = 1.5, mc = :grey70,
             msc = lc, label = latexstring("\\mathrm{$fl}"))
end
let v = vcat(ys(S), ys(E))
    plot!(pa; xticks = logticks(minimum(allx) / 1.8, maximum(allx) * 1.8),
              yticks = logticks(minimum(v) / 1.8, maximum(v) * 1.8))
end
for sv in SVALS
    scatter!(pa, [NaN], [NaN]; ms = 5, msw = 0, color = ramp_colour(sv),
             label = latexstring(@sprintf("N/\\omega = N/f = %g", sv)))
end

# ---- the best case for delta, so the verdict is not blamed on the constants ----
# delta = h is the most favourable thickness available: stage 1 showed delta is
# flat against h within each flow, so this is every delta variant at its best,
# with the arbitrary published constants removed. If the idea worked anywhere it
# would work here.
pb = plot(xscale = :log10, yscale = :log10,
          xlabel = L"L_{\mathrm{harm}}/h", ylabel = L"L_K/h",
          title = L"\mathrm{the\ most\ favourable\ thickness},\ \delta = h",
          legend = :bottomright, legendfontsize = 6, foreground_color_legend = nothing)
xh(cs) = [med(c.L_h) / c.δall.h for c in cs]
yh(cs) = [med(c.L_K) / c.δall.h for c in cs]
fh = satfit(xh(BOTH), yh(BOTH))
allxh = vcat(xh(S), xh(E))
xxh = exp.(range(log(minimum(allxh) / 1.8), log(maximum(allxh) * 1.8); length = 300))
fh.pinned || plot!(pb, xxh, fh.C1 .* (1 .- exp.(-fh.C2 .* xxh)); color = :black, lw = 2.6,
    label = latexstring(@sprintf("L_K/h = %.3f(1 - e^{-%.2f L_{\\mathrm{harm}}/h}), \\ \\mathrm{rms}\\ %.1f\\,\\%%",
                                 fh.C1, fh.C2, fh.rms)))
for (cs, mk, ms, lc, fl) in ((S, :circle, 8, C_STOK, "Stokes\\ (tidal)"),
                             (E, :diamond, 9, C_EKMA, "Ekman\\ (rotating)"))
    scatter!(pb, xh(cs), yh(cs); marker = mk, ms = ms, msw = 1.5,
             mc = [ramp_colour(c.r) for c in cs], msc = lc,
             label = latexstring("\\mathrm{$fl}"))
end
let v = vcat(yh(S), yh(E))
    plot!(pb; xticks = logticks(minimum(allxh) / 1.8, maximum(allxh) * 1.8),
              yticks = logticks(minimum(v) / 1.8, maximum(v) * 1.8))
end
annotate!(pb, minimum(allxh) / 1.5, maximum(vcat(yh(S), yh(E))) * 1.4,
          text(latexstring(@sprintf("\\mathrm{unscaled\\ fit\\ on}\\ L_{\\mathrm{harm}}\\!:\\ \\mathrm{rms}\\ 10.4\\,\\%%")),
               7, :grey25, :left))
say(@sprintf("\ndelta = h, both flows: rms %.1f %%  (unscaled 10.4 %%)", fh.rms))

fig = plot(pa, pb; layout = (1, 2), size = (1300, 580),
           plot_title = L"T = 10\,\mathrm{m},\ z = h:\ \ \mathrm{does}\ K_T = B_1 \delta \sqrt{\mathrm{TKE}}\,(B_2 + e^{B_3 L_{\mathrm{harm}}/\delta})\ \mathrm{beat\ the\ unscaled\ fit?}",
           left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, top_margin = 3Plots.mm)
o = joinpath(FIGDIR, "KT_delta_model_T10.png")
savefig(fig, o)
say("wrote $o")
mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "plot_KT_model_T10.log"), "w") do io
    foreach(l -> println(io, l), logl)
end
