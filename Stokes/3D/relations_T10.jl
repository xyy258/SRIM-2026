#!/usr/bin/env julia
# Two explicit relations for the T = 10 m column, each as one labelled figure.
#
#   (1) TKE(z) = TKE₀ exp(−z/Lₑ)          — the log-TKE profile, fitted per case
#   (2) K_T    = L∞ √TKE (1 − exp(−√TKE/(N x₀)))
#                                          — K_T at z = h from TKE at z = h and N
#
# (1) repeats swirlesrun6.jl's fit (same TKE definition, same tide/20 boxcar,
# same window z ∈ [0.2 m, T]) over the wider N/ω column, and reproduces its
# archived numbers for N/ω = 0, 1, 2, 10 exactly. The point of redoing it here is
# the label: each line carries its own fitted equation.
#
# (2) is not a new fit. L∞ and x₀ are the saturating-exponential parameters from
# plot_l_vs_qN_T10.jl, fitted there to six case medians of l = K_T/√TKE against
# √TKE/N. Multiplying that by √TKE turns a mixing length into a diffusivity, so
# the curve drawn over the scatter here is a prediction, not a regression on the
# points it is drawn through — which is why its residual is worth quoting.
#
# The two asymptotes are the reason a single power law K_T ~ TKE^p cannot work:
#   √TKE/N ≫ x₀ (weak)    K_T → L∞ √TKE            slope ½ in log-log
#   √TKE/N ≪ x₀ (strong)  K_T → TKE/(N x₀)         slope 1, prefactor set by N
# so the measured per-case slope must climb from ½ to 1 as N/ω rises, and it does.
#
# USAGE  GKSwstype=100 julia --project=. relations_T10.jl
# ENV    L_INF (0.598)  X0 (1.475)  SKIP_TKE (1)  SKIP_KT (3)  ZFIT_MIN (0.2)

using Oceananigans, JLD2, Plots, Printf, Statistics

get!(ENV, "GKSwstype", "100")
const HERE   = @__DIR__
const ω      = 1e-4
const T_tide = 2π / ω
const T      = 10.0
const SVALS  = [0, 1, 2, 5, 10, 25, 50]
const L∞     = parse(Float64, get(ENV, "L_INF", "0.598"))
const X0     = parse(Float64, get(ENV, "X0",    "1.475"))
const SKIP_TKE = parse(Float64, get(ENV, "SKIP_TKE",  "1"))
const SKIP_KT  = parse(Float64, get(ENV, "SKIP_KT",   "3"))
const ZFIT_MIN = parse(Float64, get(ENV, "ZFIT_MIN",  "0.2"))
const FIGDIR = joinpath(HERE, "h_defs", "crossing_0p1", "figures")

default(fontfamily = "sans-serif", framestyle = :box, grid = true, gridalpha = 0.15,
        tickfontsize = 8, guidefontsize = 10, legendfontsize = 8, titlefontsize = 10)

# The ramp swirlesrun4.jl uses, so colours mean the same N/ω in every figure.
const RAMP = [(0.0,   ( 27,  78, 143)), (0.301, ( 46, 139,  87)),
              (0.699, (200, 150,  30)), (1.0,   (180,  80,  44)),
              (1.398, (142,  27,  78)), (1.699, ( 75,  16,  96))]
function clr(s)
    s > 0 || return "#8C8C8C"
    x = clamp(log10(s), RAMP[1][1], RAMP[end][1])
    for i in 1:length(RAMP)-1
        (x0, c0), (x1, c1) = RAMP[i], RAMP[i+1]
        x <= x1 || continue
        f = x1 == x0 ? 0.0 : (x - x0) / (x1 - x0)
        ch(k) = clamp(round(Int, c0[k] + f * (c1[k] - c0[k])), 0, 255)
        return "#" * join(string(ch(k), base = 16, pad = 2) for k in 1:3)
    end
    return "#000000"
end

tag_of(s) = "P4_T10_sqrtRi$s"

function lsq(x, y)
    k = findall(i -> isfinite(x[i]) && isfinite(y[i]), eachindex(x))
    xx, yy = x[k], y[k]
    x̄, ȳ = mean(xx), mean(yy)
    b = sum((xx .- x̄) .* (yy .- ȳ)) / sum((xx .- x̄) .^ 2)
    a = ȳ - b * x̄
    r = yy .- (a .+ b .* xx)
    return (; a, b, rms = sqrt(mean(r .^ 2)), maxres = maximum(abs.(r)),
              R2 = 1 - sum(r .^ 2) / sum((yy .- ȳ) .^ 2), n = length(k))
end

# ---------------------------------------------------------------- figure 1 --
function boxcar(a::AbstractVector, nh)
    nh <= 0 && return collect(float.(a))
    n = length(a); out = similar(float.(collect(a)))
    for i in 1:n
        lo, hi = max(1, i - nh), min(n, i + nh)
        g = filter(isfinite, @view a[lo:hi])
        out[i] = isempty(g) ? NaN : mean(g)
    end
    return out
end
boxcar(A::AbstractMatrix, nh) =
    reduce(vcat, (reshape(boxcar(view(A, k, :), nh), 1, :) for k in 1:size(A, 1)))

function tke_profile(s)
    f = joinpath(HERE, "outputs", tag_of(s), "TidalBL3D_$(tag_of(s))_moments.jld2")
    isfile(f) || (@warn "missing $f — N/ω=$s skipped"; return nothing)
    ts = Dict(v => FieldTimeSeries(f, v) for v in ("U", "V", "uu", "vv", "ww"))
    times = ts["U"].times; nt = length(times)
    z = collect(znodes(ts["U"]))
    grab(v) = (a = zeros(length(interior(ts[v][1])), nt);
               for n in 1:nt; a[:, n] .= vec(interior(ts[v][n])); end; a)
    U, V = grab("U"), grab("V")
    uu, vv, ww = grab("uu"), grab("vv"), grab("ww")
    raw = 0.5 .* ((uu .- U .^ 2) .+ (vv .- V .^ 2) .+ ww)     # decompose, then smooth
    dt  = times[2] - times[1]
    TKE = boxcar(raw, max(0, round(Int, (T_tide / 20) / (2dt))))
    keep = times .>= SKIP_TKE * T_tide
    # Geometric time mean: the profile oscillates in log space, so the line the
    # eye follows through the movie is the mean of the logarithm.
    mlog = [mean(filter(isfinite, log10.(max.(TKE[k, keep], eps())))) for k in eachindex(z)]
    kf   = findall(k -> ZFIT_MIN <= z[k] <= T, eachindex(z))
    fit  = lsq(z[kf], mlog[kf])
    return (; s, z, mlog, fit, Le = -1 / (fit.b * log(10)), TKE0 = 10^fit.a)
end

function fig_tke(R)
    zmax = T + 10
    kp   = r -> findall(k -> r.z[k] <= zmax && isfinite(r.mlog[k]), eachindex(r.z))
    # Scale the axis to the fit window, not to the whole plotted column: above
    # the pycnocline TKE falls several more decades, and letting that set the
    # range squashes the straight part this figure is about into a corner.
    win  = [r.mlog[k] for r in R for k in kp(r) if r.z[k] <= T]
    vals = [r.mlog[k] for r in R for k in kp(r)]
    lo   = floor(Int, minimum(win)) - 1
    hi   = ceil(Int, maximum(vals))

    # One line through every case at once: all seven profiles are pooled inside
    # the fit window and a single straight line is fitted to the lot. Per-case
    # lines are not drawn, so what the dashed line misses is visible directly.
    zall = reduce(vcat, [r.z[kp(r)][r.z[kp(r)] .<= T] for r in R])
    yall = reduce(vcat, [r.mlog[kp(r)][r.z[kp(r)] .<= T] for r in R])
    g    = lsq(zall, yall)
    Le, TKE0 = -1 / (g.b * log(10)), 10^g.a

    p = plot(xlabel = "TKE  (m² s⁻²)", ylabel = "z  (m)", xscale = :log10,
             xlims = (10.0^lo, 10.0^hi), xticks = 10.0 .^ (lo:hi), ylims = (0, zmax),
             legend = :topright, legendtitle = "N/ω",
             size = (780, 660), left_margin = 6Plots.mm, bottom_margin = 6Plots.mm,
             title = @sprintf("T = 10 m:  time-mean TKE, and one line fitted to all seven cases over z ∈ [%.1f m, %g m]",
                              ZFIT_MIN, T))
    for r in R
        k = kp(r)
        plot!(p, 10.0 .^ r.mlog[k], r.z[k]; lw = 2, color = clr(r.s),
              label = @sprintf("%g", r.s))
    end
    zl = range(ZFIT_MIN, T, length = 2)
    plot!(p, 10.0 .^ (g.a .+ g.b .* zl), zl; lw = 3, ls = :dash, color = :black,
          label = "single fit")
    hline!(p, [T]; color = :black, ls = :dot, lw = 1.2, label = "z = T")

    # The equation, on the figure rather than in the caption.
    eqn = @sprintf("TKE(z) = %.2f × 10⁻⁶ exp(−z / %.2f m)   m² s⁻²", TKE0 * 1e6, Le)
    qual = @sprintf("all 7 cases pooled:  R² = %.4f,  rms %.3f decades (%.0f %%),  max %.2f dec",
                    g.R2, g.rms, 100 * (10^g.rms - 1), g.maxres)
    annotate!(p, 10.0^(lo + 0.06), 0.28zmax, text(eqn, :left, 11, "sans-serif"))
    annotate!(p, 10.0^(lo + 0.06), 0.22zmax, text(qual, :left, 8, "sans-serif", RGB(0.3,0.3,0.3)))

    f = joinpath(FIGDIR, "tke_logfit_T10.png")
    savefig(p, f)
    @printf("  single fit to all 7: TKE₀ = %.3g, Lₑ = %.3f m, R² = %.5f, rms %.4f dec\n",
            TKE0, Le, g.R2, g.rms)
    return f
end

# ---------------------------------------------------------------- figure 2 --
function kt_samples(s)
    f = joinpath(HERE, "outputs", tag_of(s), "mixing_$(tag_of(s))_hcross.jld2")
    isfile(f) || (@warn "missing $f — N/ω=$s skipped"; return nothing)
    d = jldopen(io -> (t = io["times"], K = io["K_at_h"], E = io["TKE_at_h"]), f, "r")
    m = @. (d.t >= SKIP_KT * T_tide) & isfinite(d.K) & isfinite(d.E) & (d.K > 0) & (d.E > 0)
    return (; s, E = d.E[m], K = d.K[m])
end

# The relation itself. N = 0 is the weak limit, K_T = L∞√TKE.
model(E, s) = (q = sqrt(E); s == 0 ? L∞ * q : L∞ * q * (1 - exp(-q / (s * ω * X0))))

# The N-dependent relation: one straight line in log(K_T) against log(TKE/N),
# fitted to every stratified case at once. TKE/N rather than TKE and N as free
# powers because the two exponents come out equal and opposite when both are
# free (0.825 and -0.820), so the pair collapses to a single combination.
function fit_global(C)
    D  = [c for c in C if c.s > 0]
    y  = reduce(vcat, (log10.(c.K) for c in D))
    u  = reduce(vcat, (log10.(c.E ./ (c.s * ω)) for c in D))
    f  = lsq(u, y)
    return (; C0 = 10^f.a, p = f.b, rms = f.rms, R2 = f.R2, n = f.n)
end

function fig_kt(C)
    G = fit_global(C)
    ky   = sort(reduce(vcat, (c.K for c in C)))
    ylo  = ky[max(1, round(Int, 0.005 * length(ky)))] / 1.4
    yhi  = ky[round(Int, 0.999 * length(ky))] * 1.4

    p1 = plot(xscale = :log10, yscale = :log10, legend = false, ylims = (ylo, yhi),
              xlabel = "TKE at z = h  (m² s⁻²)", ylabel = "K_T at z = h  (m² s⁻¹)",
              left_margin = 6Plots.mm, bottom_margin = 6Plots.mm,
              title = "T = 10 m:  K_T against TKE at z = h,  one power law per case")
    rows = Any[]
    for c in C
        scatter!(p1, c.E, c.K; ms = 1.6, msw = 0, alpha = 0.4, color = clr(c.s))
        f = lsq(log10.(c.E), log10.(c.K))
        e = exp10.(range(log10(minimum(c.E)), log10(maximum(c.E)); length = 100))
        plot!(p1, e, (10^f.a) .* e .^ f.b; lw = 2.5, color = clr(c.s))
        push!(rows, (s = c.s, A = 10^f.a, p = f.b, rms = f.rms))
    end

    # The table, as its own axis-free panel so the rows can be laid out by hand.
    p2 = plot(framestyle = :none, legend = false, xlims = (0, 1), ylims = (0, 1))
    annotate!(p2, 0.02, 0.97, text("N/ω      line of best fit", :left, 9, "sans-serif"))
    for (i, r) in enumerate(rows)
        yr = 0.92 - 0.055 * i
        annotate!(p2, 0.02, yr, text(@sprintf("%-4g", r.s), :left, 9, "sans-serif", clr(r.s)))
        annotate!(p2, 0.16, yr, text(@sprintf("K_T = %.2g · TKE^%.2f", r.A, r.p),
                                     :left, 9, "sans-serif", clr(r.s)))
    end
    yb = 0.92 - 0.055 * (length(rows) + 1.4)
    annotate!(p2, 0.02, yb, text("all stratified cases at once:", :left, 9, "sans-serif"))
    annotate!(p2, 0.02, yb - 0.055,
              text(@sprintf("K_T = %.3g (TKE/N)^%.2f", G.C0, G.p), :left, 11, "sans-serif"))
    annotate!(p2, 0.02, yb - 0.105,
              text(@sprintf("rms %.3f dec (×%.2f), R² = %.2f", G.rms, 10^G.rms, G.R2),
                   :left, 8, "sans-serif", RGB(0.3, 0.3, 0.3)))
    annotate!(p2, 0.02, yb - 0.185,
              text("N/ω = 0 is excluded from that fit:\nTKE/N is undefined without\nstratification.",
                   :left, 8, "sans-serif", RGB(0.3, 0.3, 0.3)))

    f = joinpath(FIGDIR, "K_T_vs_TKE_T10.png")
    savefig(plot(p1, p2, layout = grid(1, 2, widths = [0.70, 0.30]), size = (1020, 640)), f)
    @printf("  all stratified cases: K_T = %.4g (TKE/N)^%.3f, rms %.3f dec, R² = %.4f, n = %d\n",
            G.C0, G.p, G.rms, G.R2, G.n)
    return f
end

# ---------------------------------------------------------------- drive -----
mkpath(FIGDIR)

R = filter(!isnothing, [tke_profile(s) for s in SVALS])
println("\n(1)  TKE(z) = TKE₀ exp(−z/Lₑ)   fit window z ∈ [$ZFIT_MIN, $T] m")
@printf("%-5s %9s %11s %9s %9s %9s\n", "N/ω", "Lₑ (m)", "TKE₀", "R²", "rms_dec", "max_dec")
for r in R
    @printf("%-5g %9.3f %11.3g %9.5f %9.4f %9.4f\n",
            r.s, r.Le, r.TKE0, r.fit.R2, r.fit.rms, r.fit.maxres)
end
flat = [r for r in R if r.s <= 10]
@printf("  N/ω ≤ 10: Lₑ = %.2f ± %.2f m, TKE₀ = %.2f ± %.2f ×10⁻⁶ m² s⁻²\n",
        mean(x.Le for x in flat), std(x.Le for x in flat),
        1e6mean(x.TKE0 for x in flat), 1e6std(x.TKE0 for x in flat))
println("wrote ", fig_tke(R))

C = filter(!isnothing, [kt_samples(s) for s in SVALS])
println("\n(2)  K_T = L∞ √TKE (1 − exp(−√TKE/(N x₀)))   residual in decades of log₁₀")
@printf("%-5s %10s %10s %10s %8s\n", "N/ω", "rms_model", "bias", "rms_powerlaw", "slope p")
pooled = Float64[]
for c in C
    r = log10.(model.(c.E, c.s) ./ c.K); append!(pooled, r)
    f = lsq(log10.(c.E), log10.(c.K))            # the best per-case power law
    @printf("%-5g %10.3f %+10.3f %10.3f %8.2f\n",
            c.s, sqrt(mean(r .^ 2)), mean(r), f.rms, f.b)
end
@printf("  pooled: rms %.3f dec, bias %+.3f, n = %d, 2 parameters for all %d cases\n",
        sqrt(mean(pooled .^ 2)), mean(pooled), length(pooled), length(C))
println("wrote ", fig_kt(C))
