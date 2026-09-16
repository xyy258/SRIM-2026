# Nondimensional form of the harmonic law, Stokes column only.
#
#   1/L_K = 1/L_N + 1/L_s   =>   TKE/K_T = (N + S)/A   =>   K_T* = A sqrt(Ri)/(1 + sqrt(Ri))
#
# with K_T* = K_T N/TKE = tau_K/tau_N and sqrt(Ri) = N/S.
#
# Evaluated at z = h on the background N = r*omega.
#
# WHY NOT A LOCAL GRADIENT Ri: h is defined as the height where db/dz crosses
# 0.1 N2_ref, so db/dz at z = h is identically 0.1 N2_bg (measured: 0.100000,
# min = max = median, every sample of every case) and a local Ri there is just
# 0.1 x the background one. A fixed height gives a free gradient but no single
# height sits at the same place in the flow for every case — median h runs 8.6
# to 11.3 m — so it fits worse (26.3 % at the best height, z = 10 m) and was
# dropped. The scan and the reasons are in LOG.txt, 2026-09-16.

# USAGE  cd Combined && GKSwstype=100 julia --project=. plot_KTstar_Ri_stokes_T10.jl

using Oceananigans, JLD2, Plots, Printf, Statistics, LaTeXStrings

get!(ENV, "GKSwstype", "100")
default(dpi = 600, fontfamily = "DejaVu Sans")

const HERE   = @__DIR__
include(joinpath(HERE, "sweep.jl"))
const FIGDIR = joinpath(HERE, "figures")
const ω      = 1e-4
const T_tide = 2π / ω
const SKIP   = 3                       # spin-up, tidal periods

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))
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

function boxcar(A::AbstractMatrix, nh)
    nh <= 0 && return float.(A)
    out = similar(float.(A))
    for k in axes(A, 1), i in axes(A, 2)
        g = fin(@view A[k, max(1, i - nh):min(size(A, 2), i + nh)])
        out[k, i] = isempty(g) ? NaN : mean(g)
    end
    return out
end

function shear_on_faces(U, V, zc, zf)
    S = similar(float(U), length(zf))
    @inbounds for k in 2:length(zc)
        dz = zc[k] - zc[k-1]
        S[k] = hypot((U[k] - U[k-1]) / dz, (V[k] - V[k-1]) / dz)
    end
    S[1] = S[2]; S[end] = S[end-1]
    return S
end

# K_T* and sqrt(Ri) from series already taken at one height.
function pair(K, E, S, N)
    ok = @. isfinite(K) && K > 0 && isfinite(E) && E > 0 && isfinite(S) && S > 0 &&
            isfinite(N) && N > 0
    (K = [ok[i] ? K[i] * N[i] / E[i] : NaN for i in eachindex(K)],
     x = [ok[i] ? N[i] / S[i]        : NaN for i in eachindex(K)],
     frac = count(ok) / length(ok))
end

cases = []
for s in SVALS
    c = stokes_case(s)
    c === nothing && (say("missing files for $(sqrtRi_tag(s)) — skipped"); continue)
    d = jldopen(c.mix, "r") do io
        (t = io["times"], h = io["h"], Kh = io["K_at_h"], Eh = io["TKE_at_h"])
    end
    Us = FieldTimeSeries(c.mom, "U"; backend = OnDisk())
    Vs = FieldTimeSeries(c.mom, "V"; backend = OnDisk())
    grid = Us.grid
    zc = Array(znodes(grid, Center())); zf = Array(znodes(grid, Face()))
    Sp = Array{Float64}(undef, length(zf), length(d.t))
    for n in eachindex(d.t)
        Sp[:, n] = shear_on_faces(Float64.(Array(interior(Us[n], 1, 1, :))),
                                  Float64.(Array(interior(Vs[n], 1, 1, :))), zc, zf)
    end
    dt = d.t[2] - d.t[1]
    Sp = boxcar(Sp, max(0, round(Int, (T_tide / 20) / (2dt))))

    m  = d.t .>= SKIP * T_tide
    Sh = [interp_at(zf, view(Sp, :, n), d.h[n]) for n in eachindex(d.h)][m]
    bg = pair(d.Kh[m], d.Eh[m], Sh, fill(s * ω, count(m)))
    push!(cases, (r = float(s), bg = bg))
    say(@sprintf("N/ω = %-4g  %.0f %% of samples usable", s, 100 * bg.frac))
end
isempty(cases) && error("no Stokes cases found")

# K_T* = A sqrt(Ri)/(1 + sqrt(Ri)), one free constant, fitted in log on the case medians.
function fit(cs, key)
    x = [med(getfield(c, key).x) for c in cs]; y = [med(getfield(c, key).K) for c in cs]
    keep = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
    x, y = x[keep], y[keep]
    lr = log.(y ./ (x ./ (1 .+ x)))
    (A = exp(mean(lr)), rms = 100 * sqrt(mean((lr .- mean(lr)) .^ 2)), n = length(x))
end

for (key, nm) in ((:bg, "z = h, background N = r·ω"),)
    f = fit(cases, key)
    say("")
    say(@sprintf("%s:  K_T* = %.4f sqrt(Ri)/(1 + sqrt(Ri)),  rms %.1f %% on %d cases", nm, f.A, f.rms, f.n))
    say("  r        sqrt(Ri)      Ri        K_T*      fit       ratio")
    for c in cases
        x = med(getfield(c, key).x); y = med(getfield(c, key).K); p = f.A * x / (1 + x)
        say(@sprintf("  %-5g %11.4f %10.4f %9.4f %9.4f %9.3f", c.r, x, x^2, y, p, y / p))
    end
    let ri = fin([med(getfield(c, key).x)^2 for c in cases])
        say(@sprintf("  Ri spans %.4g to %.4g  (x%.0f)", minimum(ri), maximum(ri), maximum(ri) / minimum(ri)))
    end
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

# Each guide is drawn only where it is actually the asymptote, so neither runs
# across the empty corners the legend and the caption need.
function panel(key, ttl, xlab, note)
    f = fit(cases, key)
    xs = [med(getfield(c, key).x) for c in cases]; ys = [med(getfield(c, key).K) for c in cases]
    lo, hi = minimum(fin(xs)) / 2.5, maximum(fin(xs)) * 2.5
    ylo, yhi = minimum(fin(ys)) / 2.5, maximum(fin(ys)) * 2.2
    p = plot(xscale = :log10, yscale = :log10, xlabel = xlab,
             ylabel = L"K_T^{*} = K_T N/\mathrm{TKE}", title = ttl,
             legend = :bottomright, legendfontsize = 6, foreground_color_legend = nothing)
    xx = exp.(range(log(lo), log(hi); length = 300))
    xlow = xx[xx .<= 2.0]; xhigh = xx[xx .>= 0.7]
    plot!(p, xlow, f.A .* xlow; color = :grey55, lw = 1.2, ls = :dash,
          label = L"Ri \to 0\!: \ K_T^{*} \to A\sqrt{Ri} \ \ (K_T \to \mathrm{TKE}/S)")
    plot!(p, xhigh, fill(f.A, length(xhigh)); color = :grey55, lw = 1.2, ls = :dot,
          label = latexstring(@sprintf("Ri \\to \\infty\\!: \\ K_T^{*} \\to %.3f \\ \\ (K_T \\to \\mathrm{TKE}/N)", f.A)))
    plot!(p, xx, f.A .* xx ./ (1 .+ xx); color = :black, lw = 2.4,
          label = latexstring(@sprintf("K_T^{*} = %.3f\\,\\sqrt{Ri}/(1+\\sqrt{Ri}), \\ \\mathrm{rms}\\ %.1f\\,\\%%", f.A, f.rms)))
    for (i, c) in enumerate(cases)
        (isfinite(xs[i]) && isfinite(ys[i])) || continue
        plot!(p, [qlo(getfield(c, key).x), qhi(getfield(c, key).x)], [ys[i], ys[i]];
              color = :grey70, lw = 1.0, label = "")
        plot!(p, [xs[i], xs[i]], [qlo(getfield(c, key).K), qhi(getfield(c, key).K)];
              color = :grey70, lw = 1.0, label = "")
        scatter!(p, [xs[i]], [ys[i]]; ms = 9, msw = 1.5, mc = ramp_colour(c.r), msc = :black, label = "")
    end
    plot!(p; xticks = logticks(lo, hi), xlims = (lo, hi),
             yticks = logticks(ylo, yhi), ylims = (ylo, yhi))
    annotate!(p, lo * 1.15, yhi / 1.08, text(note, 6, :grey30, :left, :top))
    return p
end

p = panel(:bg, "", L"\sqrt{Ri} = N/S \ \ (N = r\,\omega,\ \mathrm{background})",
          "both axes carry N, so K_T* ~ N and Ri ~ N² share it:\nsome correlation is built in, as in L_K vs L_harm")
for sv in SVALS
    scatter!(p, [NaN], [NaN]; ms = 5, msw = 0, color = ramp_colour(sv), label = @sprintf("N/ω = %g", sv))
end
f = plot(p; size = (900, 780),
         plot_title = L"T = 10\,\mathrm{m},\ z = h,\ \mathrm{Stokes\ only}:\ \ K_T^{*} = A\sqrt{Ri}/(1+\sqrt{Ri})",
         bottom_margin = 6Plots.mm, left_margin = 6Plots.mm, top_margin = 3Plots.mm)
o = joinpath(FIGDIR, "KTstar_vs_Ri_stokes_T10.png"); savefig(f, o); say("\nwrote $o")

mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "plot_KTstar_Ri_stokes_T10.log"), "w") do io
    println(io, join(logl, "\n"))
end
