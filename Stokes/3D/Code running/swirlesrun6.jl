#!/usr/bin/env julia
# The log-TKE profile is a straight line: how straight, and what does the tide do
# to it?
#
# On the log axis of swirlesrun5.jl's movies the profiles stay nearly straight
# while they rise and fall. A straight line on log-x against linear-z is
#
#     TKE(z) = TKE₀ exp(−z / Lₑ)                                            (1)
#
# so what the eye sees is an exponential decay with one length per case. This
# script measures Lₑ, scores the fit, and shows what the cycle does to it.
#
# Three outputs:
#
#   figures/tke_logfit.png              the time-mean profiles and their fits
#   figures/tke_logfit_error.png        how far the fit misses, in decades
#   outputs/tke_animations/animation_TKEdev_T<T>.mp4
#                                       the perturbation about the time mean,
#                                       folded onto one tidal cycle
#
# plus figures/tke_logfit_summary.csv and logs/tke_logfit.log for the numbers
# that do not need a picture.
#
# The time average is the geometric one, ⟨log₁₀TKE⟩_t, because the movie
# oscillates in log space, so the line the eye follows through a wobbling curve
# is the mean of the logarithm. The arithmetic-mean fit is reported in the log as
# a check; it leans on the peak-flow profiles and comes out slightly steeper.
#
# The fit window is z from ZFIT_MIN to T, below the pycnocline. ZFIT_MIN excludes
# the near-wall peak, which is not part of the decay (1): TKE turns over there
# because it must vanish at the bed. The log also fits the whole plotted column
# and a two-segment line, which locates the knee a single line is penalised for.
#
# Errors are quoted in decades rather than as R². A residual of 0.01 decades
# means the line is within 2.3 % of the mean profile at that height, which stays
# true under a change of variable in a way that R² does not.
#
# The animation shows log₁₀TKE(z,t) − ⟨log₁₀TKE⟩_t, the perturbation about the
# time average at each height. It is taken about the time average rather than
# about the fitted line, so that what moves is separated from what the fit
# already gets wrong; the two add. The cycles are folded onto one, which is fair
# because the response is phase-locked at twice the tidal frequency, and the ±1σ
# ribbon across cycles says how fair.
#
# The input is *_moments.jld2, and the TKE definition, boxcar and analysis window
# are taken unchanged from swirlesrun5.jl, so these numbers describe the movies
# rather than a similar quantity computed a second way.
#
# USAGE
#   GKSwstype=100 julia --project=. swirlesrun6.jl
#
# ENV
#   T_VALUES      "5 10"     N_OVER_OMEGA  "0 1 2 10"
#   SKIP_PERIODS  1          discard the restart transient
#   SMOOTH_WINDOW tide20     boxcar: tide20 | tide | none
#   ZFIT_MIN      0.2        bottom of the fit window (m)
#   ZFIT_MAX      T          top of the fit window (m)
#   ZMAX          T + 10     top of the plotted column
#   NPHASE        100        frames in the composite cycle
#   FPS           25         → 4 s per loop at the default NPHASE
#   PREVIEW       0          1 → one PNG per T instead of the movie
#   OUT_ROOT outputs   FIG_DIR figures   ANIM_DIR outputs/tke_animations

using Oceananigans, JLD2, Plots, Printf, Statistics, Dates

get!(ENV, "GKSwstype", "100")

const HERE   = @__DIR__
const ω      = 1e-4
const U₀     = 0.04
const T_tide = 2π / ω

parse_list(key, default) = parse.(Float64, split(get(ENV, key, default)))
const T_values = parse_list("T_VALUES", "5 10")
const n_over_ω = parse_list("N_OVER_OMEGA", get(ENV, "SQRT_RI", "0 1 2 10"))

const SKIP     = parse(Float64, get(ENV, "SKIP_PERIODS", "1"))
const SMOOTH   = get(ENV, "SMOOTH_WINDOW", "tide20")
const ZFIT_MIN = parse(Float64, get(ENV, "ZFIT_MIN", "0.2"))
const NPHASE   = parse(Int,     get(ENV, "NPHASE", "100"))
const FPS      = parse(Int,     get(ENV, "FPS", "25"))
const PREVIEW  = get(ENV, "PREVIEW", "0") == "1"

const smooth_window = SMOOTH == "tide20" ? T_tide / 20 :
                      SMOOTH == "tide"   ? T_tide      :
                      SMOOTH == "none"   ? 0.0         :
                      error("SMOOTH_WINDOW must be tide20, tide or none — got \"$SMOOTH\"")

num_lbl(x) = isinteger(x) ? string(Int(x)) : replace(string(x), "." => "p")
tag_of(T, s) = "P4_T" * num_lbl(T) * "_sqrtRi" * num_lbl(s)
const outroot = get(ENV, "OUT_ROOT", "outputs")
moments_file(tag) = joinpath(HERE, outroot, tag, "TidalBL3D_" * tag * "_moments.jld2")

const figdir  = get(ENV, "FIG_DIR", joinpath(HERE, "figures"))
const animdir = get(ENV, "ANIM_DIR", joinpath(HERE, outroot, "tke_animations"))
const logdir  = joinpath(HERE, "logs")
mkpath(figdir); mkpath(animdir); mkpath(logdir)

default(fontfamily = "sans-serif", framestyle = :box, grid = true, gridalpha = 0.15,
        tickfontsize = 8, guidefontsize = 10, legendfontsize = 8, titlefontsize = 10)

const CLR = Dict(0.0 => "#8C8C8C", 1.0 => "#1B4E8F", 2.0 => "#2E8B57", 10.0 => "#B4502C")
clr(s) = get(CLR, s, "#000000")

zfit_max(T) = parse(Float64, get(ENV, "ZFIT_MAX", string(T)))
zmax_of(T)  = parse(Float64, get(ENV, "ZMAX",     string(T + 10)))

# ---------------- Loading, identical to swirlesrun5.jl ----------------
function boxcar(a::AbstractVector, nh)
    nh <= 0 && return collect(float.(a))
    n = length(a); out = similar(float.(collect(a)))
    for i in 1:n
        lo, hi = max(1, i - nh), min(n, i + nh)
        good = filter(isfinite, @view a[lo:hi])
        out[i] = isempty(good) ? NaN : mean(good)
    end
    return out
end
boxcar(A::AbstractMatrix, nh) =
    reduce(vcat, (reshape(boxcar(view(A, k, :), nh), 1, :) for k in 1:size(A, 1)))

function load_tke(T, s)
    tag = tag_of(T, s)
    fname = moments_file(tag)
    isfile(fname) || (@warn "Missing $fname — $tag skipped"; return nothing)

    ts = Dict(v => FieldTimeSeries(fname, v) for v in ("U", "V", "uu", "vv", "ww"))
    times = ts["U"].times; nt = length(times)
    nt < 4 && (@warn "$tag has only $nt samples — skipping"; return nothing)

    zc = collect(znodes(ts["U"]))
    grab(v) = (a = zeros(length(interior(ts[v][1])), nt);
               for n in 1:nt; a[:, n] .= vec(interior(ts[v][n])); end; a)
    U, V = grab("U"), grab("V")
    uu, vv, ww = grab("uu"), grab("vv"), grab("ww")

    TKE_raw = 0.5 .* ((uu .- U .^ 2) .+ (vv .- V .^ 2) .+ ww)   # decompose first
    dt = times[2] - times[1]
    nh = max(0, round(Int, smooth_window / (2dt)))
    TKE = boxcar(TKE_raw, nh)                                    # then smooth

    keep = times .>= SKIP * T_tide
    @printf("  %-18s %5d samples, %.2f periods, %d kept past SKIP=%g, boxcar ±%d\n",
            tag, nt, times[end] / T_tide, sum(keep), SKIP, nh)
    return (; tag, T, s, times = times[keep], z = zc, TKE = TKE[:, keep])
end

# ---------------- Fitting ----------------
function fitline(z, y)
    k = findall(i -> isfinite(z[i]) && isfinite(y[i]), eachindex(z))
    length(k) < 3 && return nothing
    zz, yy = z[k], y[k]
    z̄, ȳ = mean(zz), mean(yy)
    Szz = sum((zz .- z̄) .^ 2)
    b = sum((zz .- z̄) .* (yy .- ȳ)) / Szz
    a = ȳ - b * z̄
    r = yy .- (a .+ b .* zz)
    rms = sqrt(mean(r .^ 2))
    # The standard slope error. Neighbouring cells are not independent samples,
    # so this understates the uncertainty; it is quoted to show that the slope is
    # not ambiguous, not as a confidence interval.
    se_b = rms * sqrt(length(k) / (length(k) - 2)) / sqrt(Szz)
    return (; a, b, rms, se_b, n = length(k),
              R2 = 1 - sum(r .^ 2) / sum((yy .- ȳ) .^ 2), maxres = maximum(abs.(r)))
end

Le_of(b) = -1 / (b * log(10))                    # decades per m → e-folding m

# Where does the single line fail? A continuous two-segment fit, with the
# breakpoint found by searching every position. This is a measurement of where
# the knee is rather than a model of anything.
function broken_stick(z, y; margin = 8)
    n = length(z); n < 2margin + 4 && return nothing
    best = nothing
    for j in (margin + 1):(n - margin)
        f1 = fitline(view(z, 1:j), view(y, 1:j))
        f2 = fitline(view(z, j:n), view(y, j:n))
        (f1 === nothing || f2 === nothing) && continue
        sse = f1.rms^2 * f1.n + f2.rms^2 * f2.n
        (best === nothing || sse < best.sse) &&
            (best = (; sse, zbreak = z[j], f1, f2, rms = sqrt(sse / (f1.n + f2.n))))
    end
    return best
end

# Mean plus the first and second tidal harmonics. The phase is quoted as the
# fraction of a period by which the harmonic's maximum lags t = 0, which is slack
# water with U∞ rising.
function harmonic_fit(t, y)
    k = findall(isfinite, y); length(k) < 8 && return nothing
    tt, yy = t[k], y[k]
    X = hcat(ones(length(tt)), cos.(ω .* tt), sin.(ω .* tt),
             cos.(2ω .* tt), sin.(2ω .* tt))
    c = X \ yy
    res = yy .- X * c
    lag(cc, ss, m) = mod(atan(ss, cc) / (2π * m), 1 / m)
    return (; A1 = hypot(c[2], c[3]), A2 = hypot(c[4], c[5]),
              lag2 = lag(c[4], c[5], 2),
              frac_explained = 1 - sum(res .^ 2) / sum((yy .- mean(yy)) .^ 2))
end

# ---------------- Analyse one case ----------------
function analyse(c)
    T = c.T
    zmax, zfmax = zmax_of(T), zfit_max(T)
    L = log10.(replace(x -> x > 0 ? x : NaN, c.TKE))     # NaN where relaminarized
    nanmean(v) = (g = filter(isfinite, v); isempty(g) ? NaN : mean(g))

    mlog  = [nanmean(view(L, k, :)) for k in axes(L, 1)]                          # geometric
    malog = [log10(max(nanmean(view(c.TKE, k, :)), 1e-300)) for k in axes(L, 1)]  # arithmetic

    kfit  = findall(z -> ZFIT_MIN <= z <= zfmax, c.z)
    kfull = findall(z -> ZFIT_MIN <= z <= zmax,  c.z)
    kplot = findall(z -> z <= zmax, c.z)

    fit   = fitline(c.z[kfit],  mlog[kfit])
    full  = fitline(c.z[kfull], mlog[kfull])
    arith = fitline(c.z[kfit],  malog[kfit])
    brk   = broken_stick(c.z[kfull], mlog[kfull])

    # --- what the cycle does to the line, over the fit window ---
    # With ζ = z − z̄ the residual about the time-mean line splits into three
    # orthogonal parts,
    #     r(z,t) = A(t) + B(t)ζ + s(z,t),   ⟨r²⟩_z = A² + B²⟨ζ²⟩ + ⟨s²⟩,
    # a rigid shift of the whole profile (A), a pivot about the middle of the
    # window (B), and a genuine loss of straightness (s). The three add up
    # exactly.
    z_w = c.z[kfit]; ζ = z_w .- mean(z_w); Σζ2 = mean(ζ .^ 2)
    line = fit.a .+ fit.b .* z_w
    Δz = z_w[end] - z_w[1]

    nt = size(L, 2)
    A = fill(NaN, nt); B = fill(NaN, nt); Srms = fill(NaN, nt); Rrms = fill(NaN, nt)
    for n in 1:nt
        r = view(L, kfit, n) .- line
        all(isfinite, r) || continue
        A[n] = mean(r)
        B[n] = sum(ζ .* r) / sum(ζ .^ 2)
        Srms[n] = sqrt(mean((r .- (A[n] .+ B[n] .* ζ)) .^ 2))
        Rrms[n] = sqrt(mean(r .^ 2))
    end
    # The tilt cannot be reported as Lₑ(t): B is comparable to b itself, so the
    # instantaneous slope crosses zero near slack water, where the profile is
    # momentarily flat, and −1/(b ln10) is infinite there. The drop across the
    # window stays finite and is what the movie shows.
    D_t = .-(fit.b .+ B) .* Δz
    D̄   = -fit.b * Δz

    hA = harmonic_fit(c.times, A)

    # Why the line pivots: the response at twice the tidal frequency, height by
    # height. If the top of the window feels the tide later than the bed does,
    # the profile cannot shift rigidly and must tilt instead.
    lagz = fill(NaN, length(c.z))
    for k in kplot
        h = harmonic_fit(c.times, view(L, k, :))
        h === nothing || (lagz[k] = h.lag2)
    end
    for k in kplot[2:end]         # unwrap, since the lag is only known mod ½
        (isfinite(lagz[k]) && isfinite(lagz[k-1])) || continue
        while lagz[k] - lagz[k-1] < -0.25; lagz[k] += 0.5; end
        while lagz[k] - lagz[k-1] >  0.25; lagz[k] -= 0.5; end
    end
    flag = fitline(c.z[kfit], lagz[kfit])
    c_up = flag === nothing || flag.b == 0 ? NaN : 1 / (flag.b * T_tide)   # m s⁻¹
    span2ω = 2 * (lagz[kfit[end]] - lagz[kfit[1]])   # 2ω cycles across the window

    # --- the perturbation about the time mean, folded onto one cycle ---------
    φ = mod.(c.times ./ T_tide, 1.0)
    edges = range(0, 1, length = NPHASE + 1)
    bin = clamp.(searchsortedlast.(Ref(edges), φ), 1, NPHASE)
    dev  = fill(NaN, length(c.z), NPHASE)      # ⟨log₁₀TKE − ⟨log₁₀TKE⟩_t⟩ per bin
    devsd = fill(NaN, length(c.z), NPHASE)
    for i in 1:NPHASE
        cols = findall(bin .== i)
        isempty(cols) && continue
        for k in kplot
            g = filter(isfinite, view(L, k, cols)) .- mlog[k]
            isempty(g) && continue
            dev[k, i] = mean(g)
            length(g) > 1 && (devsd[k, i] = std(g))
        end
    end
    comp = [(φ = (edges[i] + edges[i+1]) / 2,
             A = nanmean(A[bin .== i]), D = nanmean(D_t[bin .== i]),
             R = nanmean(Rrms[bin .== i]), n = count(bin .== i)) for i in 1:NPHASE]

    return (; c.tag, c.T, c.s, c.times, c.z, mlog, kfit, kplot, zmax, zfmax, Δz,
              fit, full, arith, brk, A, D_t, D̄, Srms, Rrms, hA, comp, dev, devsd,
              lagz, c_up, span2ω, flag,
              rmsA = sqrt(nanmean(A .^ 2)), rmsTilt = sqrt(nanmean(B .^ 2) * Σζ2),
              rmsS = sqrt(nanmean(Srms .^ 2)), rmsR = sqrt(nanmean(Rrms .^ 2)))
end

# ---------------- Figure 1: the time averages and their straight lines ------
function fig_mean(analysed)
    ps = Any[]
    for T in T_values
        R = get(analysed, T, Any[]); isempty(R) && continue
        # Snapped to whole decades with one labelled tick each: left to itself
        # GR labels every second decade and the reader counts gridlines.
        vals = [v for r in R for (k, v) in zip(r.kplot, r.mlog[r.kplot]) if isfinite(v)]
        lo, hi = floor(Int, minimum(vals)), ceil(Int, maximum(vals))
        p = plot(xlabel = "TKE  (m² s⁻²)", ylabel = "z  (m)", xscale = :log10,
                 xlims = (10.0^lo, 10.0^hi), xticks = 10.0 .^ (lo:hi),
                 ylims = (0, R[1].zmax), legend = :topright, legendtitle = "N/ω",
                 title = @sprintf("T = %g m", T))
        for r in R
            plot!(p, 10.0 .^ r.mlog, r.z; lw = 2, color = clr(r.s),
                  label = @sprintf("%g   Lₑ = %.2f m", r.s, Le_of(r.fit.b)))
            zl = range(ZFIT_MIN, r.zfmax, length = 2)
            plot!(p, 10.0 .^ (r.fit.a .+ r.fit.b .* zl), zl;
                  lw = 1.3, ls = :dash, color = :black, label = "")
        end
        hline!(p, [T]; color = :black, ls = :dot, lw = 1.2, label = "")
        push!(ps, p)
    end
    out = joinpath(figdir, "tke_logfit.png")
    savefig(plot(ps..., layout = (1, length(ps)), size = (560 * length(ps), 640),
                 plot_title = "Time-mean TKE, ⟨log₁₀TKE⟩_t, with TKE₀exp(−z/Lₑ) fitted over " *
                              @sprintf("z ∈ [%.1f m, T]  (dashed)", ZFIT_MIN),
                 plot_titlefontsize = 11,
                 left_margin = 7Plots.mm, bottom_margin = 7Plots.mm), out)
    return out
end

# ---------------- Figure 2: the error ----------------
function fig_error(analysed)
    ps = Any[]
    for T in T_values
        R = get(analysed, T, Any[]); isempty(R) && continue
        # The axis is set by the residual inside the window, which is what this
        # figure is for. Outside it the line is an extrapolation and misses by
        # whole decades, so those curves leave the frame — which is itself the
        # message, and the mean figure shows where.
        xin = maximum(abs(r.mlog[k] - (r.fit.a + r.fit.b * r.z[k]))
                      for r in R for k in r.kfit)
        p = plot(xlabel = "⟨log₁₀TKE⟩_t − fitted line  (decades)", ylabel = "z  (m)",
                 ylims = (0, R[1].zmax), xlims = (-1.8xin, 1.8xin),
                 legend = :topright, legendtitle = "N/ω,  rms over the window",
                 title = @sprintf("T = %g m", T))
        hspan!(p, [ZFIT_MIN, R[1].zfmax]; color = "#CCCCCC", alpha = 0.25, label = "")
        vline!(p, [0]; color = :black, lw = 0.8, label = "")
        for r in R
            plot!(p, r.mlog .- (r.fit.a .+ r.fit.b .* r.z), r.z; lw = 2, color = clr(r.s),
                  label = @sprintf("%g   %.3f dec  (%.1f%%)", r.s, r.fit.rms,
                                   100 * (10^r.fit.rms - 1)))
        end
        hline!(p, [T]; color = :black, ls = :dot, lw = 1.2, label = "")
        push!(ps, p)
    end
    out = joinpath(figdir, "tke_logfit_error.png")
    savefig(plot(ps..., layout = (1, length(ps)), size = (560 * length(ps), 640),
                 plot_title = "Error of the straight line. Shaded = fit window; " *
                              "outside it the line is an extrapolation.",
                 plot_titlefontsize = 11,
                 left_margin = 7Plots.mm, bottom_margin = 7Plots.mm), out)
    return out
end

# ---------------- Figure 3: the perturbation, as one folded cycle ----------
function anim_dev(T, R)
    isempty(R) && return nothing
    zmax = R[1].zmax

    # One x-axis for every frame and every case: a perturbation that doubles
    # must look twice as wide, which it cannot on an axis that rescales.
    xm = 0.0
    for r in R, i in 1:NPHASE, k in r.kplot
        isfinite(r.dev[k, i]) && (xm = max(xm, abs(r.dev[k, i])))
    end
    xm = 1.1xm       # the ±1σ ribbon may clip at the extremes, the curve does not

    function draw(i)
        φ = R[1].comp[i].φ
        p1 = plot(xlabel = "log₁₀TKE(z,t) − ⟨log₁₀TKE⟩_t   (decades)", ylabel = "z  (m)",
                  xlims = (-xm, xm), ylims = (0, zmax),
                  legend = :topright, legendtitle = "N/ω",
                  title = @sprintf("Perturbation about the time mean, T = %g m   |   ωt/2π = %.2f   |   U∞/U₀ = %+.2f",
                                   T, φ, sin(2π * φ)))
        vline!(p1, [0]; color = :black, lw = 1.0, label = "")
        for r in R
            plot!(p1, r.dev[r.kplot, i], r.z[r.kplot];
                  ribbon = r.devsd[r.kplot, i], fillalpha = 0.15,
                  lw = 2, color = clr(r.s), label = @sprintf("%g", r.s))
        end
        hline!(p1, [T]; color = :black, ls = :dash, lw = 1.2, label = "")
        annotate!(p1, -0.96xm, T + 0.03zmax, text("z = T", 8, :left, :bottom, :black))
        # Guide lines at factors of 2, which are easier to read than decades.
        for f in (0.5, 2.0)
            d = log10(f)
            abs(d) < xm && vline!(p1, [d]; color = "#DDDDDD", ls = :dot, lw = 1.0, label = "")
        end

        p2 = plot(0:0.005:1, sin.(2π .* (0:0.005:1)); color = :black, lw = 1.2, label = "",
                  xlabel = "phase  ωt/2π", ylabel = "U∞/U₀", xlims = (0, 1),
                  ylims = (-1.15, 1.15), yticks = [-1, 0, 1])
        hline!(p2, [0]; color = :grey, lw = 0.8, label = "")
        scatter!(p2, [φ], [sin(2π * φ)]; ms = 6, msw = 0, color = "#B4502C", label = "")

        plot(p1, p2, layout = grid(2, 1, heights = [0.82, 0.18]),
             size = (900, 760), left_margin = 6Plots.mm, bottom_margin = 5Plots.mm)
    end

    if PREVIEW
        out = joinpath(figdir, "preview_TKEdev_T" * num_lbl(T) * ".png")
        savefig(draw(cld(NPHASE, 4)), out)          # a quarter through, at peak flow
        @info "PREVIEW=1 — wrote $(relpath(out, HERE)) and skipped the movie"
        return out
    end

    anim = @animate for i in 1:NPHASE
        draw(i)
    end
    out = joinpath(animdir, "animation_TKEdev_T" * num_lbl(T) * ".mp4")
    mp4(anim, out, fps = FPS)
    @info "wrote $(relpath(out, HERE))"
    return out
end

# ---------------- Drive ----------------
println("Loading moments for T ∈ $(T_values), N/ω ∈ $(n_over_ω)")
analysed = Dict{Float64,Vector{Any}}()
for T in T_values
    rs = Any[]
    for s in n_over_ω
        c = load_tke(T, s)
        c === nothing || push!(rs, analyse(c))
    end
    analysed[T] = rs
end

outs = String[]
push!(outs, fig_mean(analysed), fig_error(analysed))
for T in T_values
    o = anim_dev(T, get(analysed, T, Any[]))
    o === nothing || push!(outs, o)
end

csv = joinpath(figdir, "tke_logfit_summary.csv")
open(csv, "w") do io
    println(io, "tag,T,n_over_omega,zfit_min,zfit_max,Le_m,se_Le_m,R2,rms_dec,max_dec,",
                "Le_full_m,R2_full,rms_full_dec,zbreak_m,Le_below_m,Le_above_m,rms_break_dec,",
                "Le_arith_m,rms_total_dec,rms_level_dec,rms_tilt_dec,rms_shape_dec,",
                "A_p2p_dec,A_amp2_dec,A_lag2_Ttide,A_var_explained,",
                "drop_mean_dec,drop_min_dec,drop_max_dec,c_up_mm_per_s,window_2omega_cycles")
    for T in T_values, r in get(analysed, T, Any[])
        f, fu, br = r.fit, r.full, r.brk
        Ag = filter(isfinite, r.A); Dg = filter(isfinite, r.D_t)
        @printf(io, "%s,%g,%g,%.3f,%.3f,%.4f,%.4f,%.5f,%.4f,%.4f,%.4f,%.5f,%.4f,",
                r.tag, r.T, r.s, ZFIT_MIN, r.zfmax, Le_of(f.b),
                f.se_b * abs(Le_of(f.b) / f.b), f.R2, f.rms, f.maxres,
                Le_of(fu.b), fu.R2, fu.rms)
        br === nothing ? print(io, "NaN,NaN,NaN,NaN,") :
            @printf(io, "%.3f,%.4f,%.4f,%.4f,", br.zbreak, Le_of(br.f1.b), Le_of(br.f2.b), br.rms)
        @printf(io, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f\n",
                Le_of(r.arith.b), r.rmsR, r.rmsA, r.rmsTilt, r.rmsS,
                maximum(Ag) - minimum(Ag), r.hA.A2, r.hA.lag2, r.hA.frac_explained,
                r.D̄, minimum(Dg), maximum(Dg), 1000r.c_up, r.span2ω)
    end
end

open(joinpath(logdir, "tke_logfit.log"), "a") do io
    for out in (io, stdout)
        println(out, "\n", "="^96)
        @printf(out, "==== %s  swirlesrun6.jl  SKIP=%g SMOOTH=%s ZFIT_MIN=%.2f NPHASE=%d\n",
                Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), SKIP, SMOOTH, ZFIT_MIN, NPHASE)
        println(out, "="^96)

        for T in T_values
            R = get(analysed, T, Any[]); isempty(R) && continue
            @printf(out, "\nT = %g m   fit window z ∈ [%.2f, %.2f] m   column z ≤ %.1f m   %d samples\n",
                    T, ZFIT_MIN, R[1].zfmax, R[1].zmax, length(R[1].times))

            println(out, "\n  (1) THE FIT.  TKE = TKE₀exp(−z/Lₑ) on ⟨log₁₀TKE⟩_t.  rms and max are DECADES:")
            println(out, "      0.01 dec = 2.3%, 0.05 dec = 12%, 0.1 dec = 26% error in TKE.")
            println(out, "         fit window                          |  whole column        |  two lines instead")
            println(out, "  N/ω    Lₑ (m)         R²       rms    max  |  Lₑ (m)  R²     rms  |  knee z  Lₑ below  Lₑ above  rms")
            for r in R
                f, fu, br = r.fit, r.full, r.brk
                @printf(out, "  %-5g  %5.3f ± %.3f  %.5f  %.4f %.3f |  %5.3f  %.4f %.3f |",
                        r.s, Le_of(f.b), f.se_b * abs(Le_of(f.b) / f.b), f.R2, f.rms, f.maxres,
                        Le_of(fu.b), fu.R2, fu.rms)
                br === nothing ? println(out, "   —") :
                    @printf(out, "  %5.2f   %5.3f     %5.3f     %.3f\n",
                            br.zbreak, Le_of(br.f1.b), Le_of(br.f2.b), br.rms)
            end
            @printf(out, "      The ± is the least-squares slope error and is a lower bound: residuals are\n")
            @printf(out, "      correlated in z. Geometric vs arithmetic time mean, Lₑ: ")
            for r in R; @printf(out, "%g→%.2f/%.2f ", r.s, Le_of(r.fit.b), Le_of(r.arith.b)); end
            println(out, "m")

            println(out, "\n  (2) THE PERTURBATION.  r(z,t) = A(t) + B(t)ζ + s(z,t) about the time-mean line,")
            println(out, "      ⟨r²⟩_z = A² + B²⟨ζ²⟩ + ⟨s²⟩ exactly — a slide, a pivot, and lost straightness.")
            println(out, "      rms over time (dec)                    |  level A(t)            |  drop across window (dec)")
            println(out, "  N/ω    total   level   tilt   shape  shape%|  peak-peak ×TKE span  |  mean    min     max")
            for r in R
                Ag = filter(isfinite, r.A); Dg = filter(isfinite, r.D_t)
                p2p = maximum(Ag) - minimum(Ag)
                @printf(out, "  %-5g  %.4f  %.4f  %.4f  %.4f  %4.0f%% |  %.3f dec  ×%5.2f     |  %+.3f  %+.3f  %+.3f\n",
                        r.s, r.rmsR, r.rmsA, r.rmsTilt, r.rmsS, 100r.rmsS / r.rmsR,
                        p2p, 10^p2p, r.D̄, minimum(Dg), maximum(Dg))
            end
            @printf(out, "      A(t) is almost pure 2ω — the bed stress ~|U∞|U∞ works twice a cycle. A₂ = ")
            for r in R; @printf(out, "%.3f ", r.hA.A2); end
            @printf(out, "dec,\n      explaining %.0f–%.0f%% of its variance, peaking %+.3f T_tide (%+.1f h) after peak flow.\n",
                    100minimum(r.hA.frac_explained for r in R),
                    100maximum(r.hA.frac_explained for r in R),
                    R[1].hA.lag2 - 0.25, (R[1].hA.lag2 - 0.25) * T_tide / 3600)
            @printf(out, "      The pivot is a phase lag with height: the 2ω signal climbs at %.2f mm/s\n",
                    1000 * mean(r.c_up for r in R))
            @printf(out, "      (%.0f m per cycle), so the window spans %.2f cycles of 2ω. Below ½ a cycle the\n",
                    mean(r.c_up for r in R) * T_tide, mean(r.span2ω for r in R))
            println(out, "      tilt absorbs it; at a full cycle it cannot, and the shape term takes over.")
            println(out, "      Lₑ(t) is deliberately not quoted: B is comparable to b, so the instantaneous")
            println(out, "      slope crosses zero near slack and −1/(b ln10) has a pole there.")

            println(out, "\n  (3) BY PHASE  (cycles folded; slack at 0 and 0.5, peak flow at 0.25 and 0.75)")
            print(out, "  ωt/2π  |")
            for r in R; @printf(out, " N/ω=%-4g  A     ×TKE   drop   rms |", r.s); end
            println(out)
            for i in 1:max(1, cld(NPHASE, 12)):NPHASE
                @printf(out, "  %.3f  |", R[1].comp[i].φ)
                for r in R
                    x = r.comp[i]
                    @printf(out, " %+.3f  ×%4.2f  %+.3f  %.3f |", x.A, 10^x.A, x.D, x.R)
                end
                println(out)
            end
        end

        println(out, "\noutputs: ", join((relpath(o, HERE) for o in outs), ", "))
        println(out, "         ", relpath(csv, HERE))
        println(out, "="^96)
    end
end
