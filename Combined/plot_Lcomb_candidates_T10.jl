# Which combination of the stratification and shear length scales collapses the
# mixing length?
#
# From plot_shear_scales_T10.jl: L_K = K_T/√TKE is close to proportional to
# L_N = √TKE/N but with slope 0.82 rather than 1, and L_s = √TKE/S alone does
# much worse. The question here is whether some weighted combination L_comb of the
# two gives L_K = A·L_comb — a straight proportionality with a single constant,
# which is what a mixing-length closure would want.
#
# ---------------- What is being fitted, and why b = 1 ----------------
# For each candidate the target is the ONE-parameter fit
#
#     L_K = A · L_comb            (b = 1 forced)
#
# not the two-parameter power law L_K = A·L_comb^b. Forcing b = 1 is the whole
# point: any monotone scale can be made to fit with a free exponent, so a good
# rms with b free proves nothing. A candidate earns its keep only if L_K is
# proportional to it. The free-b slope is reported alongside as a check — a
# candidate whose free slope is already near 1 is one whose b = 1 fit is honest.
#
# Free parameters inside a candidate are chosen to minimise the b = 1 rms on
# BOTH flows together, since a scale that needs a different weight for each flow
# has not unified anything.
#
# ---------------- The candidates ----------------
#   L_N, L_s                      each alone, as the baselines
#   min(L_N, L_s)                 the crude limiter: whichever is smaller
#   harmonic, equal weights       1/L_harm = 1/L_N + 1/L_s
#   harmonic, weighted            1/L_β    = 1/L_N + β/L_s          β free
#   p-norm                        L_comb = (L_N^−p + L_s^−p)^(−1/p) p free
#                                 p = 1 is the harmonic, p → ∞ is the min
#   geometric                     L_comb = L_N^α · L_s^(1−α)        α free
#
# All are formed per time sample and then reduced to a case median, never the
# other way round.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot_Lc_candidates_T10.jl
#        (run plot_shear_scales_T10.jl first — it writes the cache)

using JLD2, Plots, Printf, Statistics, LaTeXStrings

get!(ENV, "GKSwstype", "100")
# Publication defaults: 600 dpi, and one font for every figure in this folder.
# LaTeX labels are rendered by GR's own mathtext, so `fontfamily` sets the
# surrounding text and the maths follows the TeX shapes either way.
default(dpi = 600, fontfamily = "DejaVu Sans")
const HERE   = @__DIR__
const CACHE  = joinpath(HERE, "Data", "shear_scales_T10.jld2")
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
med(v) = (w = filter(isfinite, v); isempty(w) ? NaN : median(w))

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))

# ---------------- the cached scales ----------------
isfile(CACHE) || error("$CACHE not found — run plot_shear_scales_T10.jl first")
stokes, ekman = [], []
jldopen(CACHE, "r") do io
    for (nm, dst) in (("stokes", stokes), ("ekman", ekman))
        for r in io["$nm/ratios"]
            g = @sprintf("%s/r=%.1f", nm, r)
            push!(dst, (r = r, L_K = io["$g/L_K"], L_N = io["$g/L_N"], L_s = io["$g/L_s"]))
        end
    end
end
BOTH = vcat(stokes, ekman)
say(@sprintf("%d Stokes and %d Ekman cases from %s", length(stokes), length(ekman),
             basename(CACHE)))

# ---------------- the fits ----------------
# b = 1 forced: the only freedom is the constant, so A is the geometric mean of
# L_K/L_comb and the rms is the spread of that ratio in log space.
function propfit(cs, sc)
    lr = [log(med(c.L_K)) - log(med(sc(c))) for c in cs]
    keep = filter(isfinite, lr)
    isempty(keep) && return (A = NaN, rms = Inf)
    a = mean(keep)
    return (A = exp(a), rms = 100 * sqrt(mean((keep .- a) .^ 2)))
end

# b free, as the honesty check on the above.
function powfit(cs, sc)
    lx = [log(med(sc(c))) for c in cs]; ly = [log(med(c.L_K)) for c in cs]
    k = @. isfinite(lx) & isfinite(ly)
    lx, ly = lx[k], ly[k]
    mx = mean(lx); my = mean(ly)
    b = sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx) .^ 2)
    a = my - b * mx
    res = ly .- (a .+ b .* lx)
    return (A = exp(a), b = b, rms = 100 * sqrt(mean(res .^ 2)))
end

# The third fit: the saturating house form on the combined scale. Several
# candidates have a natural slope near 0.9 rather than 1, and no choice of
# weight can fix that — a proportionality cannot bend. If the curvature is real
# then the right question is whether L_comb straightens the two flows onto ONE
# saturating curve, which a proportionality fit would never reveal.
function satfit(cs, sc)
    x = [med(sc(c)) for c in cs]; y = [med(c.L_K) for c in cs]
    k = @. isfinite(x) & isfinite(y) & (x > 0) & (y > 0)
    x, y = x[k], y[k]
    ygrid = exp.(range(log(0.3minimum(y)), log(40maximum(y)); length = 220))
    xgrid = exp.(range(log(0.02minimum(x)), log(150maximum(x)); length = 220))
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
    pin = best[2] <= ygrid[2] || best[2] >= ygrid[end-1] ||
          best[3] <= xgrid[2] || best[3] >= xgrid[end-1]
    return (L = best[2], x0 = best[3], rms = 100 * sqrt(best[1] / length(x)), pinned = pin)
end

# A candidate is a name, a scale-builder, and optionally a parameter to search.
harmonic(β) = c -> 1 ./ (1 ./ c.L_N .+ β ./ c.L_s)
pnorm(p)    = c -> (c.L_N .^ (-p) .+ c.L_s .^ (-p)) .^ (-1 / p)
geometric(α) = c -> c.L_N .^ α .* c.L_s .^ (1 - α)

# `name` is for the log, `tex` for the panel title, `pname`/`ptex` likewise for
# the free parameter. The generic combined scale is L_comb; the equal-weight
# harmonic is L_harm and the weighted one L_β, so that L_C stays reserved for
# the Corrsin scale in plot_l_vs_corrsin_T10.jl.
struct Cand
    name::String
    tex::String              # LaTeX body, no delimiters
    build::Function          # parameter -> (case -> vector)
    grid                     # parameter values, or nothing
    pname::String
    ptex::String
end
CANDS = [
    Cand("L_N alone", "L_N\\ \\mathrm{alone}",
         _ -> (c -> c.L_N),              nothing, "", ""),
    Cand("L_s alone", "L_s\\ \\mathrm{alone}",
         _ -> (c -> c.L_s),              nothing, "", ""),
    Cand("min(L_N, L_s)", "L_{\\mathrm{comb}} = \\min(L_N, L_s)",
         _ -> (c -> min.(c.L_N, c.L_s)), nothing, "", ""),
    Cand("harmonic, equal", "1/L_{\\mathrm{harm}} = 1/L_N + 1/L_s",
         _ -> harmonic(1.0),             nothing, "", ""),
    Cand("harmonic, weighted", "1/L_\\beta = 1/L_N + \\beta/L_s",
         harmonic,   0.0:0.02:8.0, "β", "\\beta"),
    Cand("p-norm", "L_{\\mathrm{comb}} = (L_N^{-p} + L_s^{-p})^{-1/p}",
         pnorm,      0.1:0.02:8.0, "p", "p"),
    Cand("geometric", "L_{\\mathrm{comb}} = L_N^{\\alpha} L_s^{1-\\alpha}",
         geometric, -0.5:0.01:1.5, "α", "\\alpha"),
]

# The free parameter is chosen on both flows together: a scale needing a
# different weight per flow has unified nothing.
function resolve(cd)
    cd.grid === nothing && return (par = NaN, sc = cd.build(nothing))
    best = (Inf, first(cd.grid))
    for v in cd.grid
        r = propfit(BOTH, cd.build(v)).rms
        r < best[1] && (best = (r, v))
    end
    pin = best[2] <= first(cd.grid) + step(cd.grid) / 2 ||
          best[2] >= last(cd.grid) - step(cd.grid) / 2
    pin && say("  WARNING: $(cd.name) chose $(cd.pname) = $(best[2]) at the edge of its search range")
    return (par = best[2], sc = cd.build(best[2]))
end

say("")
say("L_K against each candidate scale.  b = 1 is the one-parameter proportionality")
say("L_K = A·L_comb; b free is the same fit with the exponent released, as a check.")
say("")
say("  candidate              param      b=1 rms:  both  Stokes  Ekman   | b free: slope   rms | saturating on L_comb: both  Stokes  Ekman")
results = []
for cd in CANDS
    rv = resolve(cd)
    fb = propfit(BOTH, rv.sc); fs = propfit(stokes, rv.sc); fe = propfit(ekman, rv.sc)
    pw = powfit(BOTH, rv.sc)
    sb = satfit(BOTH, rv.sc); ss = satfit(stokes, rv.sc); se = satfit(ekman, rv.sc)
    push!(results, (cd = cd, par = rv.par, sc = rv.sc, A = fb.A,
                    rms = fb.rms, rms_s = fs.rms, rms_e = fe.rms, b = pw.b, prms = pw.rms,
                    sat = sb, sat_s = ss, sat_e = se))
    say(@sprintf("  %-22s %-10s %8.1f %% %6.1f %% %6.1f %% | %6.2f  %6.1f %% | %11.1f %% %6.1f %% %6.1f %%%s",
                 cd.name,
                 cd.grid === nothing ? "—" : @sprintf("%s = %.2f", cd.pname, rv.par),
                 fb.rms, fs.rms, fe.rms, pw.b, pw.rms, sb.rms, ss.rms, se.rms,
                 sb.pinned ? "  (sat pinned)" : ""))
end

best = results[argmin([r.rms for r in results])]
bsat = results[argmin([r.sat.rms for r in results])]
say("")
say(@sprintf("best proportionality on both flows: %s%s, L_K = %.3f · L_comb, rms %.1f %% (slope %.2f with b free)",
             best.cd.name, isnan(best.par) ? "" : @sprintf(" (%s = %.2f)", best.cd.pname, best.par),
             best.A, best.rms, best.b))
say(@sprintf("best saturating curve on both flows: %s%s, L_K = %.3f(1 − e^(−L_comb/%.3f)), rms %.1f %%",
             bsat.cd.name, isnan(bsat.par) ? "" : @sprintf(" (%s = %.2f)", bsat.cd.pname, bsat.par),
             bsat.sat.L, bsat.sat.x0, bsat.sat.rms))
say(@sprintf("  for comparison, the saturating curve on L_N alone: rms %.1f %% on both flows",
             results[1].sat.rms))

# ---------------- the figure ----------------
mkpath(FIGDIR)
function panel(rz)
    p = plot(xscale = :log10, yscale = :log10, legend = false,
             xlabel = L"L_{\mathrm{comb}} \ \ (\mathrm{m})",
             ylabel = L"L_K = K_T/\sqrt{\mathrm{TKE}} \ \ (\mathrm{m})",
             titlefontsize = 8, guidefontsize = 7, tickfontsize = 6,
             title = latexstring(@sprintf("%s%s", rz.cd.tex,
                        isnan(rz.par) ? "" : @sprintf(",\\ \\ %s = %.2f", rz.cd.ptex, rz.par))))
    xs = vcat([med(rz.sc(c)) for c in BOTH]...)
    lo, hi = minimum(xs) / 1.6, maximum(xs) * 1.6
    xx = exp.(range(log(lo), log(hi); length = 200))
    # The title carries the scale, so the three rms numbers go inside the axes
    # rather than into a second title line — GR renders a LaTeX label as one
    # line of maths and has nowhere to put a break.
    ys = [med(c.L_K) for c in BOTH]
    annotate!(p, lo * 1.25, maximum(ys) * 1.15,
              text(latexstring(@sprintf("b{=}1\\!:\\ %.0f\\,\\%%, \\ \\ b\\ \\mathrm{free}\\!:\\ %.2f, \\ \\ \\mathrm{sat}\\!:\\ %.0f\\,\\%%",
                                        rz.rms, rz.b, rz.sat.rms)), 6, :grey25, :left))
    plot!(p, xx, rz.A .* xx; color = :black, lw = 2.0)
    rz.sat.pinned || plot!(p, xx, rz.sat.L .* (1 .- exp.(-xx ./ rz.sat.x0));
                           color = "#c46a1f", lw = 1.8, ls = :dash)
    for (cs, mk, ms, lc) in ((stokes, :circle, 6, C_STOK), (ekman, :xcross, 7, C_EKMA))
        for c in cs
            scatter!(p, [med(rz.sc(c))], [med(c.L_K)]; marker = mk, ms = ms, msw = 1.4,
                     mc = ramp_colour(c.r), msc = lc)
        end
    end
    return p
end
panels = [panel(r) for r in results]
# One legend panel, so the seven stay uncluttered.
pk = plot(framestyle = :none, legend = :left, legendfontsize = 7,
          foreground_color_legend = nothing)
scatter!(pk, [NaN], [NaN]; marker = :circle, ms = 6, msw = 1.4, mc = :grey70,
         msc = C_STOK, label = L"\mathrm{Stokes\ (tidal)}")
scatter!(pk, [NaN], [NaN]; marker = :xcross, ms = 7, msw = 1.4, mc = :grey70,
         msc = C_EKMA, label = L"\mathrm{Ekman\ (rotating)}")
plot!(pk, [NaN], [NaN]; color = :black, lw = 2.0,
      label = L"L_K = A\,L_{\mathrm{comb}} \ \ (\mathrm{proportionality})")
plot!(pk, [NaN], [NaN]; color = "#c46a1f", lw = 1.8, ls = :dash,
      label = L"L_K = L_\infty(1 - e^{-L_{\mathrm{comb}}/x_0}) \ \ (\mathrm{saturating})")
for sv in SVALS
    scatter!(pk, [NaN], [NaN]; ms = 5, msw = 0, color = ramp_colour(sv),
             label = latexstring(@sprintf("N/\\omega = N/f = %g", sv)))
end
push!(panels, pk)

f = plot(panels...; layout = (2, 4), size = (1500, 760),
         plot_title = L"T = 10\,\mathrm{m},\ z = h:\ \ L_K\ \mathrm{against\ candidate\ combinations\ of}\ L_N\ \mathrm{and}\ L_s",
         left_margin = 5Plots.mm, bottom_margin = 5Plots.mm, top_margin = 2Plots.mm)
o = joinpath(FIGDIR, "L_K_vs_Lcomb_candidates_T10.png")
savefig(f, o)
say("")
say("wrote $o")

mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "plot_Lcomb_candidates_T10.log"), "w") do io
    foreach(l -> println(io, l), logl)
end
