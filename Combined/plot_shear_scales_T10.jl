# Is the mixing length set by the stratification, by the mean shear, or by both?
#
# The previous figure gave l = K_T/√TKE against L_N = √TKE/N and found a
# saturating curve for each flow separately. That says the stratification
# controls l once L_N is small, but it says nothing about what sets the plateau
# — and the natural candidate for the other limit is the mean shear.
#
# Two length scales built from the same TKE at the same height z = h:
#
#     L_N = √TKE / N           N = r·ω (Stokes) or r·f (Ekman), the background
#     L_s = √TKE / S           S = |∂⟨u_h⟩/∂z| = √((∂U/∂z)² + (∂V/∂z)²)
#
# and the mixing length itself, L_K = K_T/√TKE. Dividing each by √TKE turns all
# three into times, which is the dimensionally honest way to plot them:
#
#     τ_K = K_T/TKE            τ_N = 1/N            τ_s = 1/S
#
# ---------------- The two figures ----------------
#   tau_K_vs_timescales_T10.png   τ_K against τ_N and against τ_s, one panel
#                                 each. If τ_K follows τ_N in one panel and not
#                                 the other, that names the limiting process.
#   L_N_L_s_vs_r_T10.png          L_N and L_s against r = N/ω = N/f, and their
#                                 ratio. Where L_s/L_N crosses 1 is where the
#                                 shear scale stops being the smaller of the two.
#
# No fit lines: this is for looking at, before choosing a functional form.
#
# ---------------- Where the shear comes from ----------------
# Neither pipeline stored it. Both sides do store the plane-averaged U and V —
# the Ekman side since ekmanrun.jl, the Stokes side in
# TidalBL3D_*_moments.jld2 — so S is differenced from those here, on the faces,
# with the same time boxcar (one twentieth of a forcing period) that the rest of
# each reduction uses. The Ekman S comes ready-made out of
# reduce_ekman_moments_T10.jl; the Stokes S is built in this script because the
# Stokes mixing files predate the question.
#
# Both components are kept. In the Ekman case the spiral turns the mean flow
# with height so ∂V/∂z is not small, and in the Stokes case the tidal ellipse
# does the same.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot_shear_scales_T10.jl

using Oceananigans, JLD2, Plots, Printf, Statistics

# Plots labels a log axis as 10^0.25 unless told otherwise, which is unreadable
# for a quantity that only spans one decade. Decades get 10^n, everything else
# gets the plain number.
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

get!(ENV, "GKSwstype", "100")
const HERE   = @__DIR__
const STOKES = "/home/tll46/SRIM-2026/Stokes/3D"
const EKFILE = joinpath(HERE, "Data", "ekman_lengthscales_T10_moments.jld2")
const FIGDIR = joinpath(HERE, "figures")
const ω      = 1e-4
const T_tide = 2π / ω
const SKIP   = 3                      # Stokes spin-up, in tidal periods
const SVALS  = [1, 2, 5, 10, 25, 50]

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

interp_at(z, fv, z₀) = isnan(z₀) ? NaN : begin
    i = searchsortedlast(z, z₀)
    i < 1 && return fv[1]
    i >= length(z) && return fv[end]
    fv[i] + (fv[i+1] - fv[i]) * (z₀ - z[i]) / (z[i+1] - z[i])
end

function boxcar(a::AbstractVector, nh)
    nh <= 0 && return collect(float.(a))
    n = length(a); out = similar(float.(collect(a)))
    for i in 1:n
        w = @view a[max(1, i - nh):min(n, i + nh)]
        g = fin(w)
        out[i] = isempty(g) ? NaN : mean(g)
    end
    return out
end
boxcar(A::AbstractMatrix, nh) =
    reduce(vcat, (reshape(boxcar(view(A, k, :), nh), 1, :) for k in 1:size(A, 1)))

function shear_on_faces(U, V, zc, zf)
    S = similar(float(U), length(zf))
    @inbounds for k in 2:length(zc)
        dz = zc[k] - zc[k-1]
        S[k] = hypot((U[k] - U[k-1]) / dz, (V[k] - V[k-1]) / dz)
    end
    S[1] = S[2]; S[end] = S[end-1]
    return S
end

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))

# ---------------- Stokes ----------------
stokes = []
for s in SVALS
    tag = "P4_T10_sqrtRi$s"
    mix = joinpath(STOKES, "outputs", tag, "mixing_$(tag)_hcross.jld2")
    mom = joinpath(STOKES, "outputs", tag, "TidalBL3D_$(tag)_moments.jld2")
    (isfile(mix) && isfile(mom)) || (say("missing files for $tag — skipped"); continue)

    d = jldopen(mix, "r") do io
        (t = io["times"], Kh = io["K_at_h"], E = io["TKE_at_h"], h = io["h"])
    end
    Us = FieldTimeSeries(mom, "U"; backend = OnDisk())
    Vs = FieldTimeSeries(mom, "V"; backend = OnDisk())
    length(Us.times) == length(d.t) ||
        error("$tag: moments and mixing files have different time bases")
    grid = Us.grid
    zc = Array(znodes(grid, Center())); zf = Array(znodes(grid, Face()))

    Sp = Array{Float64}(undef, length(zf), length(d.t))
    for n in eachindex(d.t)
        U = Float64.(Array(interior(Us[n], 1, 1, :)))
        V = Float64.(Array(interior(Vs[n], 1, 1, :)))
        Sp[:, n] = shear_on_faces(U, V, zc, zf)
    end
    dt = d.t[2] - d.t[1]
    Sp = boxcar(Sp, max(0, round(Int, (T_tide / 20) / (2dt))))
    Sh = [interp_at(zf, view(Sp, :, n), d.h[n]) for n in eachindex(d.h)]

    m = d.t .>= SKIP * T_tide
    push!(stokes, (r = float(s), N = s * ω, K = d.Kh[m], E = d.E[m], S = Sh[m]))
    say(@sprintf("Stokes N/ω = %-4g  med S at h = %.4e s⁻¹", s, med(Sh[m])))
end
isempty(stokes) && error("no Stokes T = 10 cases found under $STOKES/outputs")

# ---------------- Ekman ----------------
ekman = []
isfile(EKFILE) || error("$EKFILE not found — run reduce_ekman_moments_T10.jl first")
jldopen(EKFILE, "r") do io
    haskey(io, "r=50.0/S_at_h") ||
        error("$EKFILE has no S_at_h — re-run reduce_ekman_moments_T10.jl")
    for r in io["ratios"]
        g = @sprintf("r=%.1f", r)
        push!(ekman, (r = r, N = io["$g/N"], K = io["$g/K_at_h"],
                      E = io["$g/TKE_at_h"], S = io["$g/S_at_h"]))
        say(@sprintf("Ekman  N/f = %-4g  med S at h = %.4e s⁻¹", r, med(io["$g/S_at_h"])))
    end
end

# ---------------- the derived scales ----------------
# Formed per time sample, then reduced. The other order would divide medians by
# medians, which is not the same thing for quantities this correlated.
function scales(c)
    q   = sqrt.(max.(c.E, 0))
    τ_K = c.K ./ c.E                    # = L_K/√TKE, a time
    τ_N = fill(1 / c.N, length(q))      # N is the background value, constant
    τ_s = 1 ./ c.S
    L_K = c.K ./ q
    L_N = q ./ c.N
    L_s = q ./ c.S
    # Two quite different reasons a sample can drop out, and they must be
    # counted separately. K_T <= 0 is counter-gradient flux, a physical state of
    # the flow that the log axes cannot show — it is common in the oscillating
    # Stokes case near flow reversal. S <= 0 is the mean shear passing through
    # zero, which is a property of where z = h sits, not of the mixing.
    badK = @. !(isfinite(τ_K) && τ_K > 0)
    badS = @. !(isfinite(τ_s) && τ_s > 0)
    ok   = @. !badK & !badS
    return (r = c.r, τ_K = τ_K[ok], τ_N = τ_N[ok], τ_s = τ_s[ok],
            L_K = L_K[ok], L_N = L_N[ok], L_s = L_s[ok], n = count(ok),
            ntot = length(q), nbadK = count(badK), nbadS = count(badS))
end
S = [scales(c) for c in stokes]
E = [scales(c) for c in ekman]

say("")
say("retained samples, and why the rest went")
say("  flow    r      kept      K_T <= 0   S <= 0    (both conditions can hit one sample)")
for (nm, cs) in (("Stokes", S), ("Ekman ", E)), c in cs
    say(@sprintf("  %s %-5g %5d/%-5d %8d %9d%s", nm, c.r, c.n, c.ntot, c.nbadK, c.nbadS,
                 c.nbadK > 0.15 * c.ntot ? "   << counter-gradient a good part of the cycle" : ""))
end

say("")
say("medians at z = h.  L_K = K_T/√TKE,  L_N = √TKE/N,  L_s = √TKE/S")
say("  flow    r      L_K (m)   L_N (m)   L_s (m)   L_s/L_N   τ_K (s)   τ_N (s)   τ_s (s)")
for (nm, cs) in (("Stokes", S), ("Ekman ", E)), c in cs
    say(@sprintf("  %s %-5g %9.4f %9.4f %9.4f %9.3f %9.1f %9.1f %9.1f",
                 nm, c.r, med(c.L_K), med(c.L_N), med(c.L_s),
                 med(c.L_s) / med(c.L_N), med(c.τ_K), med(c.τ_N), med(c.τ_s)))
end

# ---------------- does a combined scale do better? ----------------
# Not a fit — just the question the harmonic combination is meant to answer,
# asked with c_N = c_s = 1 so nothing is tuned. The number reported is the
# scatter of log(L_K) about a straight line in log space, which is what a
# one-parameter proportionality would leave behind. Lower is better.
function collapse(cs, key)
    x = [log(med(getfield(c, key))) for c in cs]
    y = [log(med(c.L_K)) for c in cs]
    n = length(x); mx = mean(x); my = mean(y)
    b = sum((x .- mx) .* (y .- my)) / sum((x .- mx) .^ 2)
    a = my - b * mx
    res = y .- (a .+ b .* x)
    return (slope = b, rms = 100 * sqrt(sum(res .^ 2) / n))
end
harm(c) = (Lc = 1 ./ (1 ./ c.L_N .+ 1 ./ c.L_s);
           (r = c.r, L_K = c.L_K, L_N = c.L_N, L_s = c.L_s, L_c = Lc))
say("")
say("how well does each scale line up with L_K, in log space?  (power law, no free shape)")
say("  set                 vs L_N            vs L_s            vs L_c = 1/(1/L_N + 1/L_s)")
for (nm, cs) in (("Stokes", S), ("Ekman ", E), ("both  ", vcat(S, E)))
    hs = [harm(c) for c in cs]
    a = collapse(cs, :L_N); b = collapse(cs, :L_s); c = collapse(hs, :L_c)
    say(@sprintf("  %s  slope %5.2f rms %4.1f %%   slope %5.2f rms %4.1f %%   slope %5.2f rms %4.1f %%",
                 nm, a.slope, a.rms, b.slope, b.rms, c.slope, c.rms))
end

# ---------------- figures ----------------
mkpath(FIGDIR)

# Case medians with the interquartile range, the convention the other figure
# settled on. Every panel is one series per case so each keeps its own colour.
function bars!(p, cs, mk, ms)
    for c in cs
        scatter!(p, [c.x], [c.y];
                 xerror = ([c.x - c.xlo], [c.xhi - c.x]),
                 yerror = ([c.y - c.ylo], [c.yhi - c.y]),
                 marker = mk, ms = ms, msw = 1.6, mc = ramp_colour(c.r),
                 msc = :black, linecolor = ramp_colour(c.r), lw = 1.6, label = "")
    end
end
pt(c, xf, yf) = (r = c.r, x = med(xf(c)), xlo = qlo(xf(c)), xhi = qhi(xf(c)),
                 y = med(yf(c)), ylo = qlo(yf(c)), yhi = qhi(yf(c)))

function keys!(p)
    scatter!(p, [NaN], [NaN]; marker = :circle, ms = 7, msw = 1.6, mc = :grey70,
             msc = :black, label = "Stokes (tidal)")
    scatter!(p, [NaN], [NaN]; marker = :xcross, ms = 8, msw = 1.6, mc = :grey70,
             msc = :black, label = "Ekman (rotating)")
    for sv in SVALS
        scatter!(p, [NaN], [NaN]; ms = 5, msw = 0, color = ramp_colour(sv),
                 label = @sprintf("N/ω = N/f = %g", sv))
    end
end

# ---- figure 1: the mixing time against the two candidate times ----
function panel(xf, xlab, ttl; oneone = true)
    ps = [pt(c, xf, c -> c.τ_K) for c in S]
    pe = [pt(c, xf, c -> c.τ_K) for c in E]
    p = plot(xscale = :log10, yscale = :log10, xlabel = xlab,
             ylabel = "τ_K = K_T / TKE   (s)", title = ttl,
             legend = :topleft, legendfontsize = 6,
             foreground_color_legend = nothing)
    if oneone
        v = vcat([q.x for q in ps], [q.x for q in pe],
                 [q.y for q in ps], [q.y for q in pe])
        lo, hi = minimum(v), maximum(v)
        plot!(p, [lo, hi], [lo, hi]; color = :black, lw = 1.0, ls = :dash,
              label = "1:1")
    end
    bars!(p, ps, :circle, 7); bars!(p, pe, :xcross, 8)
    return p
end
p1 = panel(c -> c.τ_N, "τ_N = 1/N   (s)", "against the stratification time")
p2 = panel(c -> c.τ_s, "τ_s = 1/S   (s)", "against the shear time")
keys!(p2)
f1 = plot(p1, p2; layout = (1, 2), size = (1180, 560),
          plot_title = "T = 10 m, z = h:  the mixing time K_T/TKE against 1/N and 1/S",
          left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, top_margin = 3Plots.mm)
o1 = joinpath(FIGDIR, "tau_K_vs_timescales_T10.png")
savefig(f1, o1)

# ---- figure 2: the two length scales against the stratification ----
# Line colour says which flow, line style says which scale, marker colour keeps
# the N ramp so the two figures can be read against each other. Without the
# first of those the Stokes and Ekman curves join into one and the eye reads a
# single trend across a discontinuity that is not there.
const C_STOK = "#1b3a6b"
const C_EKMA = "#8e1b4e"

p3 = plot(xscale = :log10, yscale = :log10, xlabel = "r = N/ω = N/f",
          ylabel = "length scale at z = h   (m)", title = "L_N and L_s",
          legend = :bottomleft, legendfontsize = 6,
          foreground_color_legend = nothing)
for (cs, mk, ms, lc, fl) in ((S, :circle, 7, C_STOK, "Stokes"),
                             (E, :xcross, 8, C_EKMA, "Ekman"))
    for (key, ls, nm) in ((:L_N, :solid, "L_N = √TKE/N"), (:L_s, :dash, "L_s = √TKE/S"))
        xs = [c.r for c in cs]; ys = [med(getfield(c, key)) for c in cs]
        o = sortperm(xs)
        plot!(p3, xs[o], ys[o]; color = lc, ls = ls, lw = 1.8,
              label = "$fl  $nm")
        scatter!(p3, xs, ys; marker = mk, ms = ms, msw = 1.6,
                 mc = [ramp_colour(r) for r in xs], msc = lc, label = "")
    end
end
let v = vcat([med(c.L_N) for c in S], [med(c.L_s) for c in S],
             [med(c.L_N) for c in E], [med(c.L_s) for c in E])
    plot!(p3; yticks = logticks(minimum(v), maximum(v)))
end
for sv in SVALS
    scatter!(p3, [NaN], [NaN]; ms = 5, msw = 0, color = ramp_colour(sv),
             label = @sprintf("N/ω = N/f = %g", sv))
end

p4 = plot(xscale = :log10, yscale = :log10, xlabel = "r = N/ω = N/f",
          ylabel = "L_s / L_N  =  N / S", title = "which scale is the smaller",
          legend = :topleft, legendfontsize = 6,
          foreground_color_legend = nothing)
hline!(p4, [1.0]; color = :black, lw = 1.4, ls = :dash,
       label = "L_s = L_N   (below this the shear scale is the smaller)")
for (cs, mk, ms, lc, fl) in ((S, :circle, 7, C_STOK, "Stokes"),
                             (E, :xcross, 8, C_EKMA, "Ekman"))
    xs = [c.r for c in cs]
    ys = [med(c.L_s) / med(c.L_N) for c in cs]
    o = sortperm(xs)
    plot!(p4, xs[o], ys[o]; color = lc, lw = 1.8, label = fl)
    scatter!(p4, xs, ys; marker = mk, ms = ms, msw = 1.6,
             mc = [ramp_colour(r) for r in xs], msc = lc, label = "")
end
let v = vcat([med(c.L_s) / med(c.L_N) for c in S], [med(c.L_s) / med(c.L_N) for c in E])
    plot!(p4; yticks = logticks(min(0.7, minimum(v)), maximum(v)))
end

f2 = plot(p3, p4; layout = (1, 2), size = (1180, 560),
          plot_title = "T = 10 m, z = h:  the stratification and shear length scales against r",
          left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, top_margin = 3Plots.mm)
o2 = joinpath(FIGDIR, "L_N_L_s_vs_r_T10.png")
savefig(f2, o2)

say("")
say("wrote $o1")
say("wrote $o2")

mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "plot_shear_scales_T10.log"), "w") do io
    foreach(l -> println(io, l), logl)
end
