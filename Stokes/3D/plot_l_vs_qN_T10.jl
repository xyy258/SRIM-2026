# l vs √TKE/N for the T = 10 m column alone, with the saturating exponential
# fitted to the per-case medians drawn over it.
#
# The full-column figure (l_vs_q_over_N_ath.png) mixes T = 5 and T = 10, which
# hides the curvature: at fixed N/ω the two T sit at different √TKE/N. Dropping
# T = 5 leaves one monotone sequence, which is what the fit is made against.
#
# l = L∞ (1 − exp(−x/x0)),  x = √TKE/N, fitted by least squares in log l to the
# six case medians. Note only N/ω = 1 and 2 lie in the bending part of the
# curve — the rest are in the linear regime, where the fit pins the ratio
# L∞/x0 and not the two separately. The plateau is therefore indicative.
#
# USAGE  GKSwstype=100 julia --project=. plot_l_vs_qN_T10.jl

using JLD2, Plots, Printf, Statistics

get!(ENV, "GKSwstype", "100")
const HERE   = @__DIR__
const ω, ν   = 1e-4, 1e-6
const δs     = sqrt(2ν / ω)
const T_tide = 2π / ω
const SKIP   = 3
const FIGDIR = joinpath(HERE, "h_defs", "crossing_0p1", "figures")
const SVALS  = [1, 2, 5, 10, 25, 50]

# The same ramp swirlesrun4.jl uses, so the colours match the other figures.
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

med(v) = median(filter(isfinite, v))

cases = []
for s in SVALS
    tag = "P4_T10_sqrtRi$s"
    f   = joinpath(HERE, "outputs", tag, "mixing_$(tag)_hcross.jld2")
    isfile(f) || (@warn "missing $f — skipped"; continue)
    d = jldopen(io -> (t = io["times"], Kh = io["K_at_h"], E = io["TKE_at_h"]), f, "r")
    m = d.t .>= SKIP * T_tide
    q = sqrt.(max.(d.E[m], 0))
    l = d.Kh[m] ./ q                     # l = K_T(at h)/√TKE
    x = q ./ (s * ω)                     # √TKE/N
    g = @. isfinite(l) && isfinite(x) && l > 0 && x > 0
    push!(cases, (s = s, x = x[g], l = l[g], xm = med(x[g]), lm = med(l[g])))
end
isempty(cases) && error("no T = 10 mixing files under outputs/")

# --- the fit ---------------------------------------------------------------
# Grid search rather than an optimiser: two parameters, and the coarse grid is
# already finer than the spread of the medians it is fitted to.
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
sse, L∞, x0 = fit_sat(cases)
@printf("L∞ = %.3f m (%.2f δ_s)   x0 = %.3f m (%.2f δ_s)   rms %.1f %% in l\n",
        L∞, L∞ / δs, x0, x0 / δs, 100 * sqrt(sse / length(cases)))
for c in cases
    p = L∞ * (1 - exp(-c.xm / x0))
    @printf("  N/ω=%-3d  median x = %7.4f  l = %7.4f  fit = %7.4f  (%+5.1f %%)\n",
            c.s, c.xm, c.lm, p, 100 * (p / c.lm - 1))
end

# --- the figure ------------------------------------------------------------
# A handful of samples per case dip three decades below the cloud, which on a
# log axis leaves most of the panel empty. Clip the view (not the data) to the
# bulk of the points so the curvature is legible.
ally = sort(reduce(vcat, (c.l for c in cases)))
ylo  = ally[max(1, round(Int, 0.005 * length(ally)))] / 1.5
yhi  = ally[round(Int, 0.999 * length(ally))] * 1.5

p = plot(xscale = :log10, yscale = :log10, legend = :topleft, ylims = (ylo, yhi),
         xlabel = "√TKE / N   (buoyancy scale, m)", ylabel = "l = K_T/√TKE   (m)",
         title = "T = 10 m only:  l against √TKE/N,  K_T at z = h",
         size = (820, 640), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
for c in cases
    scatter!(p, c.x, c.l; ms = 1.6, msw = 0, alpha = 0.45,
             color = ramp_colour(c.s), label = @sprintf("N/ω = %g", c.s))
end
xs = reduce(vcat, (c.x for c in cases))
lo, hi = minimum(xs), maximum(xs)
plot!(p, [lo, hi], [lo, hi]; color = :black, lw = 1.2, ls = :dash, label = "l = √TKE/N  (1:1)")

xf = exp.(range(log(lo), log(hi); length = 400))
plot!(p, xf, L∞ .* (1 .- exp.(-xf ./ x0)); color = :black, lw = 2.5,
      label = @sprintf("l = L∞(1 − e^(−x/x₀)),  L∞ = %.2f m, x₀ = %.2f m", L∞, x0))
# The medians are what the curve was actually fitted to, so they are marked.
scatter!(p, [c.xm for c in cases], [c.lm for c in cases];
         ms = 6, msw = 1.5, mc = :white, msc = :black, label = "case medians (fitted)")
hline!(p, [L∞]; color = :black, lw = 1, ls = :dot,
       label = @sprintf("plateau L∞ = %.2f m = %.1f δ_s", L∞, L∞ / δs))

mkpath(FIGDIR)
f = joinpath(FIGDIR, "l_vs_q_over_N_ath_T10.png")
savefig(p, f)
println("wrote $f")
