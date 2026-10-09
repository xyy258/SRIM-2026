# Ekman-only version of plot_KTstar_Ri_zscan_T10.jl: local K_T* against local
# sqrt(Ri), scanned over z/h, fitted with K_T* = A sqrt(Ri)/(1 + sqrt(Ri)).
# Two figures: every r, and the same with r >= DROP left out.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot/plot_KTstar_Ri_zscan_ekman_T10.jl

using Oceananigans, JLD2, Plots, Printf, Statistics, LaTeXStrings

get!(ENV, "GKSwstype", "100")
default(dpi = 600, fontfamily = "DejaVu Sans", guidefontsize = 16, tickfontsize = 12,
        legendfontsize = 10, plot_titlefontsize = 18)

const HERE   = dirname(@__DIR__)
include(joinpath(HERE, "sweep.jl"))
const FIGDIR = joinpath(HERE, "figures")
const f₀     = 1e-4
const T_f    = 2π / f₀
const WINDOW = 4                          # inertial periods
const FRACS  = [0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]   # z/h, as the both-flows scan
const KEEP   = 0.5
const FLOOR  = 0.05
const DROP   = 25                         # r >= DROP left out of the second figure
const C_EKMA = "#8e1b4e"

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))
fin(v) = filter(isfinite, v)
med(v) = (w = fin(v); isempty(w) ? NaN : median(w))

interp_at(z, fv, z₀) = begin
    i = searchsortedlast(z, z₀)
    i < 1 && return fv[1]
    i >= length(z) && return fv[end]
    fv[i] + (fv[i+1] - fv[i]) * (z₀ - z[i]) / (z[i+1] - z[i])
end
c2f(fc, zc, zf) = [interp_at(zc, fc, clamp(z, zc[1], zc[end])) for z in zf]
f2c(ff, zf, zc) = [interp_at(zf, ff, clamp(z, zf[1], zf[end])) for z in zc]

function boxcar(A, nh)
    out = similar(float.(A))
    for k in axes(A, 1), i in axes(A, 2)
        g = fin(@view A[k, max(1, i - nh):min(size(A, 2), i + nh)])
        out[k, i] = isempty(g) ? NaN : mean(g)
    end
    out
end

function shear(U, V, zc, zf)
    S = zeros(length(zf))
    for k in 2:length(zc)
        dz = zc[k] - zc[k-1]
        S[k] = hypot((U[k] - U[k-1]) / dz, (V[k] - V[k-1]) / dz)
    end
    S[1] = S[2]; S[end] = S[end-1]
    S
end

function at_z(zf, zc, K, E, S, G, N2, z0)
    n = size(K, 2)
    k = [interp_at(zf, view(K, :, i), z0) for i in 1:n]
    e = [interp_at(zc, view(E, :, i), z0) for i in 1:n]
    s = [interp_at(zf, view(S, :, i), z0) for i in 1:n]
    g = [interp_at(zf, view(G, :, i), z0) for i in 1:n]
    ok = @. isfinite(k) && k > 0 && e > 0 && s > 0 && g > FLOOR * N2
    (K = med([ok[i] ? k[i] * sqrt(g[i]) / e[i] : NaN for i in 1:n]),
     x = med([ok[i] ? sqrt(g[i]) / s[i]        : NaN for i in 1:n]),
     frac = count(ok) / n)
end

hfile = joinpath(HERE, "Data", "cache", "ekman_lengthscales_T10_moments.jld2")
pts = []
for r in RATIOS
    e = ekman_case(r)
    e === nothing && continue
    file = joinpath(e.dir, "Moments.jld2")
    F = Dict(v => FieldTimeSeries(file, v; backend = OnDisk()) for v in
             ("U", "V", "W", "B", "dBdz", "uu", "vv", "ww", "wb", "F_sgs"))
    g = F["U"].grid; zc = Array(znodes(g, Center())); zf = Array(znodes(g, Face()))
    tv = F["U"].times; sel = findall(t -> t >= tv[end] - WINDOW * T_f, tv)
    N2 = (r * f₀)^2
    col(v, n) = Float64.(Array(interior(F[v][n], 1, 1, :)))
    E = zeros(length(zc), length(sel)); Fb = zeros(length(zf), length(sel)); G = similar(Fb); S = similar(Fb)
    for (i, n) in enumerate(sel)
        U, V, W, B = col("U", n), col("V", n), col("W", n), col("B", n)
        E[:, i] = 0.5 .* ((col("uu", n) .- U .^ 2) .+ (col("vv", n) .- V .^ 2) .+
                          (col("ww", n) .- f2c(W, zf, zc) .^ 2))
        Fb[:, i] = col("wb", n) .- W .* c2f(B, zc, zf) .+ col("F_sgs", n)
        G[:, i] = col("dBdz", n)
        S[:, i] = shear(U, V, zc, zf)
    end
    nh = max(0, round(Int, (T_f / 20) / (2 * (tv[sel[2]] - tv[sel[1]]))))
    E, Fb, G, S = boxcar.((E, Fb, G, S), nh)
    K = [G[k, n] > FLOOR * N2 ? -Fb[k, n] / G[k, n] : NaN for k in axes(G, 1), n in axes(G, 2)]
    hm = med(jldopen(io -> io[@sprintf("r=%.1f/h", r)], hfile, "r"))
    for φ in FRACS
        v = at_z(zf, zc, K, E, S, G, N2, φ * hm)
        v.frac >= KEEP && isfinite(v.x) && isfinite(v.K) &&
            push!(pts, (r = r, φ = φ, x = v.x, K = v.K, frac = v.frac))
    end
    say(@sprintf("r = %-4g  h = %5.2f m  kept %d of %d heights", r, hm,
                 count(p -> p.r == r, pts), length(FRACS)))
end

function fitA(ps)
    lr = [log(p.K / (p.x / (1 + p.x))) for p in ps]
    (A = exp(mean(lr)), rms = 100 * sqrt(mean((lr .- mean(lr)) .^ 2)), n = length(lr))
end

kept = [p for p in pts if p.r < DROP]
fa, fk = fitA(pts), fitA(kept)
say("")
for (nm, ps, f) in (("all r   ", pts, fa), (@sprintf("r < %-4g", DROP), kept, fk))
    say(@sprintf("%s K_T* = %.4f sqrt(Ri)/(1+sqrt(Ri))  rms %5.1f %%  n = %3d  Ri %.3g to %.3g",
                 nm, f.A, f.rms, f.n, minimum(p -> p.x, ps)^2, maximum(p -> p.x, ps)^2))
end
say("\n  per-r median of K_T*/fit, against the all-r fit and the r < $DROP fit")
for r in RATIOS
    ps = [p for p in pts if p.r == r]
    isempty(ps) && continue
    q(f) = median([p.K / (f.A * p.x / (1 + p.x)) for p in ps])
    say(@sprintf("  r = %-4g  %6.2f  %6.2f", r, q(fa), q(fk)))
end
say("\n   r     z/h    sqrt(Ri)      K_T*    usable")
for p in pts
    say(@sprintf("  %-5g %5.2f %10.3f %9.4f %7.0f %%", p.r, p.φ, p.x, p.K, 100 * p.frac))
end

function logticks(lo, hi)
    v = [m * 10.0^e for e in floor(Int, log10(lo)):ceil(Int, log10(hi)) for m in (1, 2, 5)]
    v = filter(x -> lo / 1.05 <= x <= hi * 1.05, v)
    (v, [x >= 1 ? string(round(Int, x)) : rstrip(rstrip(@sprintf("%.4f", x), '0'), '.') for x in v])
end

# Same axes on both figures so they can be laid side by side.
xlo, xhi = minimum(p -> p.x, pts) / 2.5, maximum(p -> p.x, pts) * 2.5
ylo, yhi = minimum(p -> p.K, pts) / 2.5, maximum(p -> p.K, pts) * 2.5
xx = exp.(range(log(xlo), log(xhi); length = 300))

function figure(ps, f, title, out; stats = true)
    p = plot(xscale = :log10, yscale = :log10,
             xlabel = L"\sqrt{Ri} = \sqrt{\langle \partial b/\partial z \rangle}/S \ \ (\mathrm{local})",
             ylabel = L"K_T^{*} = K_T N/\mathrm{TKE} \ \ (\mathrm{local})",
             legend = :bottomright, foreground_color_legend = nothing)
    plot!(p, xx, f.A .* xx ./ (1 .+ xx); color = :black, lw = 2.4,
          label = latexstring(@sprintf("K_T^{*} = %.3f\\,\\sqrt{Ri}/(1+\\sqrt{Ri})", f.A) *
                              (stats ? @sprintf(", \\ \\mathrm{rms}\\ %.0f\\,\\%%, \\ n = %d", f.rms, f.n) : "")))
    for q in ps
        scatter!(p, [q.x], [q.K]; marker = :diamond, ms = 8, msw = 1.3,
                 mc = ramp_colour(q.r), msc = C_EKMA, label = "")
    end
    for r in unique(q.r for q in ps)
        scatter!(p, [NaN], [NaN]; marker = :diamond, ms = 6, msw = 0, color = ramp_colour(r),
                 label = @sprintf("N/f = %g", r))
    end
    plot!(p; xticks = logticks(xlo, xhi), xlims = (xlo, xhi),
             yticks = logticks(ylo, yhi), ylims = (ylo, yhi))
    savefig(plot(p; size = (900, 760), plot_title = title,
                 bottom_margin = 6Plots.mm, left_margin = 6Plots.mm, top_margin = 3Plots.mm), out)
    say("wrote $out")
end

mkpath(FIGDIR)
say("")
figure(pts, fa, L"T = 10\,\mathrm{m},\ \mathrm{Ekman}:\ \ K_T^{*}\ \mathrm{against\ local}\ Ri,\ z/h = 0.6\ \mathrm{to}\ 3",
       joinpath(FIGDIR, "KTstar_vs_Ri_zscan_ekman_T10.png"))
figure(kept, fk, latexstring(@sprintf("T = 10\\,\\mathrm{m},\\ \\mathrm{Ekman},\\ N/f < %g:\\ \\ K_T^{*}\\ \\mathrm{against\\ local}\\ Ri", DROP)),
       joinpath(FIGDIR, "KTstar_vs_Ri_zscan_ekman_lowr_T10.png"); stats = false)

mkpath(joinpath(HERE, "logs"))
write(joinpath(HERE, "logs", "plot_KTstar_Ri_zscan_ekman_T10.log"), join(logl, "\n") * "\n")
