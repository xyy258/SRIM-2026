# K_T* = K_T N/TKE against Ri, sampled at many HEIGHTS rather than one per case,
# with everything local: N = sqrt(<db/dz>) at z, S at z, K_T and TKE at z.
#
# Why: at z = h the Ekman column spans Ri = 0.44 to 12.8 only — S ~ N^1.018
# there, so Ri is pinned near 10 (LOG.txt 2026-09-16). Scanning z frees it.
# Both flows are done the same way so the plateaus can be compared.
#
# Heights are fractions of each case's median h, so the scan is at the same
# place in the flow for every case. A (case, z) point is kept only if at least
# KEEP of its samples clear the 0.05 N2_ref gradient mask, since K_T = -F_b/G
# does not exist below it — which is exactly why the well-mixed interior, where
# Ri is genuinely small, cannot be reached.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot_KTstar_Ri_zscan_T10.jl

using Oceananigans, JLD2, Plots, Printf, Statistics, LaTeXStrings

get!(ENV, "GKSwstype", "100")
default(dpi = 600, fontfamily = "DejaVu Sans")

const HERE   = @__DIR__
include(joinpath(HERE, "sweep.jl"))
const FIGDIR = joinpath(HERE, "figures")
const ω      = 1e-4
const T_tide = 2π / ω
const T_f    = 2π / ω
const SKIP   = 3                          # Stokes spin-up, tidal periods
const WINDOW = 4                          # Ekman window, inertial periods
const FRACS  = [0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]   # z/h
const KEEP   = 0.5                        # min fraction of usable samples
const FLOOR  = 0.05                       # gradient mask, fraction of N2_ref
const C_STOK = "#1b3a6b"
const C_EKMA = "#8e1b4e"

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))
fin(v) = filter(isfinite, v)
med(v) = (w = fin(v); isempty(w) ? NaN : median(w))

interp_at(z, fv, z₀) = isnan(z₀) ? NaN : begin
    i = searchsortedlast(z, z₀)
    i < 1 && return fv[1]
    i >= length(z) && return fv[end]
    fv[i] + (fv[i+1] - fv[i]) * (z₀ - z[i]) / (z[i+1] - z[i])
end
c2f(fc, zc, zf) = [interp_at(zc, fc, clamp(z, zc[1], zc[end])) for z in zf]
f2c(ff, zf, zc) = [interp_at(zf, ff, clamp(z, zf[1], zf[end])) for z in zc]

function boxcar(A::AbstractMatrix, nh)
    nh <= 0 && return float.(A)
    out = similar(float.(A))
    for k in axes(A, 1), i in axes(A, 2)
        g = fin(@view A[k, max(1, i - nh):min(size(A, 2), i + nh)])
        out[k, i] = isempty(g) ? NaN : mean(g)
    end
    return out
end

function shear(U, V, zc, zf)
    S = zeros(Float64, length(zf))
    @inbounds for k in 2:length(zc)
        dz = zc[k] - zc[k-1]
        S[k] = hypot((U[k] - U[k-1]) / dz, (V[k] - V[k-1]) / dz)
    end
    S[1] = S[2]; S[end] = S[end-1]
    return S
end

# One (case, z): local K_T*, sqrt(Ri), and the usable fraction.
function at_z(zf, zc, K, E, S, G, N2ref, z0)
    n = size(K, 2)
    k = [interp_at(zf, view(K, :, i), z0) for i in 1:n]
    e = [interp_at(zc, view(E, :, i), z0) for i in 1:n]
    s = [interp_at(zf, view(S, :, i), z0) for i in 1:n]
    g = [interp_at(zf, view(G, :, i), z0) for i in 1:n]
    ok = @. isfinite(k) && k > 0 && isfinite(e) && e > 0 && isfinite(s) && s > 0 &&
            isfinite(g) && g > FLOOR * N2ref
    nl = [ok[i] ? sqrt(g[i]) : NaN for i in 1:n]
    (K = med([ok[i] ? k[i] * nl[i] / e[i] : NaN for i in 1:n]),
     x = med([ok[i] ? nl[i] / s[i]        : NaN for i in 1:n]),
     frac = count(ok) / n)
end

pts = []   # (flow, r, zfrac, x, K)
for s in SVALS
    c = stokes_case(s)
    c === nothing && continue
    d = jldopen(c.mix, "r") do io
        (t = io["times"], h = io["h"], zf = io["z_face"], zc = io["z_center"],
         K = io["K_T"], E = io["TKE"], G = io["dBdz"], N2 = io["N2_ref"])
    end
    Us = FieldTimeSeries(c.mom, "U"; backend = OnDisk())
    Vs = FieldTimeSeries(c.mom, "V"; backend = OnDisk())
    g = Us.grid; zc = Array(znodes(g, Center())); zf = Array(znodes(g, Face()))
    Sp = Array{Float64}(undef, length(zf), length(d.t))
    for n in eachindex(d.t)
        Sp[:, n] = shear(Float64.(Array(interior(Us[n], 1, 1, :))),
                         Float64.(Array(interior(Vs[n], 1, 1, :))), zc, zf)
    end
    dt = d.t[2] - d.t[1]
    Sp = boxcar(Sp, max(0, round(Int, (T_tide / 20) / (2dt))))
    m  = d.t .>= SKIP * T_tide
    hm = med(d.h[m])
    for φ in FRACS
        v = at_z(d.zf, d.zc, d.K[:, m], d.E[:, m], Sp[:, m], d.G[:, m], d.N2, φ * hm)
        v.frac >= KEEP && isfinite(v.x) && isfinite(v.K) &&
            push!(pts, (flow = "stokes", r = float(s), φ = φ, x = v.x, K = v.K, frac = v.frac))
    end
    say(@sprintf("Stokes r = %-4g  h = %5.2f m  kept %d of %d heights", s, hm,
                 count(p -> p.flow == "stokes" && p.r == s, pts), length(FRACS)))
end

for r in RATIOS
    e = ekman_case(r)
    e === nothing && continue
    file = joinpath(e.dir, "Moments.jld2")
    F = Dict(v => FieldTimeSeries(file, v; backend = OnDisk()) for v in
             ("U", "V", "W", "B", "dBdz", "uu", "vv", "ww", "wb", "F_sgs", "kappa_sgs"))
    g = F["U"].grid; zc = Array(znodes(g, Center())); zf = Array(znodes(g, Face()))
    tv = F["U"].times; sel = findall(t -> t >= tv[end] - WINDOW * T_f, tv)
    nt = length(sel); N2 = (r * ω)^2
    col(v, n) = Float64.(Array(interior(F[v][n], 1, 1, :)))
    Er = Array{Float64}(undef, length(zc), nt); Fr = similar(Er, length(zf), nt)
    Gr = similar(Fr); Sr = similar(Fr)
    for (i, n) in enumerate(sel)
        U, V, W, B = col("U", n), col("V", n), col("W", n), col("B", n)
        Wc = f2c(W, zf, zc)
        Er[:, i] = 0.5 .* ((col("uu", n) .- U .^ 2) .+ (col("vv", n) .- V .^ 2) .+
                           (col("ww", n) .- Wc .^ 2))
        Fr[:, i] = (col("wb", n) .- W .* c2f(B, zc, zf)) .+ col("F_sgs", n)
        Gr[:, i] = col("dBdz", n)
        Sr[:, i] = shear(U, V, zc, zf)
    end
    dt = tv[sel[2]] - tv[sel[1]]; nh = max(0, round(Int, (T_f / 20) / (2dt)))
    E = boxcar(Er, nh); Fb = boxcar(Fr, nh); G = boxcar(Gr, nh); S = boxcar(Sr, nh)
    K = [G[k, n] > FLOOR * N2 ? -Fb[k, n] / G[k, n] : NaN for k in axes(G, 1), n in axes(G, 2)]
    hm = med([interp_at(zf, view(G, :, n), NaN) for n in 1:0])   # placeholder, h below
    hm = med(jldopen(joinpath(HERE, "Data", "ekman_lengthscales_T10_moments.jld2"), "r") do io
                 io[@sprintf("r=%.1f/h", r)] end)
    for φ in FRACS
        v = at_z(zf, zc, K, E, S, G, N2, φ * hm)
        v.frac >= KEEP && isfinite(v.x) && isfinite(v.K) &&
            push!(pts, (flow = "ekman", r = r, φ = φ, x = v.x, K = v.K, frac = v.frac))
    end
    say(@sprintf("Ekman  r = %-4g  h = %5.2f m  kept %d of %d heights", r, hm,
                 count(p -> p.flow == "ekman" && p.r == r, pts), length(FRACS)))
end

fitA(ps) = isempty(ps) ? (A = NaN, rms = NaN, n = 0) : begin
    lr = [log(p.K / (p.x / (1 + p.x))) for p in ps]
    (A = exp(mean(lr)), rms = 100 * sqrt(mean((lr .- mean(lr)) .^ 2)), n = length(lr))
end
S_pts = [p for p in pts if p.flow == "stokes"]; E_pts = [p for p in pts if p.flow == "ekman"]
say("")
for (nm, ps) in (("both  ", pts), ("Stokes", S_pts), ("Ekman ", E_pts))
    f = fitA(ps)
    isempty(ps) && continue
    say(@sprintf("%s  K_T* = %.4f sqrt(Ri)/(1+sqrt(Ri))  rms %5.1f %%  n = %3d  Ri %.4g to %.4g",
                 nm, f.A, f.rms, f.n, minimum(p -> p.x, ps)^2, maximum(p -> p.x, ps)^2))
end
say("")
say("  flow    r     z/h    sqrt(Ri)      K_T*    usable")
for p in pts
    say(@sprintf("  %-7s %-5g %5.2f %10.3f %9.4f %7.0f %%", p.flow, p.r, p.φ, p.x, p.K, 100 * p.frac))
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

fb = fitA(pts); fs = fitA(S_pts); fe = fitA(E_pts)
xs = [p.x for p in pts]; ys = [p.K for p in pts]
lo, hi = minimum(xs) / 2.5, maximum(xs) * 2.5
p = plot(xscale = :log10, yscale = :log10,
         xlabel = L"\sqrt{Ri} = \sqrt{\langle \partial b/\partial z \rangle}/S \ \ (\mathrm{local})",
         ylabel = L"K_T^{*} = K_T N/\mathrm{TKE} \ \ (\mathrm{local})",
         legend = :bottomright, legendfontsize = 6, foreground_color_legend = nothing)
xx = exp.(range(log(lo), log(hi); length = 300))
for (f, col, ls, lw, nm) in ((fs, C_STOK, :dash, 1.8, "Stokes"), (fe, C_EKMA, :dash, 1.8, "Ekman"),
                             (fb, :black, :solid, 2.4, "both"))
    isfinite(f.A) || continue
    plot!(p, xx, f.A .* xx ./ (1 .+ xx); color = col, ls = ls, lw = lw,
          label = latexstring(@sprintf("\\mathrm{%s}\\!: \\ A = %.3f, \\ \\mathrm{rms}\\ %.0f\\,\\%%, \\ n = %d", nm, f.A, f.rms, f.n)))
end
for (ps, mk, ms, lc) in ((S_pts, :circle, 7, C_STOK), (E_pts, :diamond, 7, C_EKMA))
    for q in ps
        scatter!(p, [q.x], [q.K]; marker = mk, ms = ms, msw = 1.3,
                 mc = ramp_colour(q.r), msc = lc, label = "")
    end
end
scatter!(p, [NaN], [NaN]; marker = :circle,  ms = 7, msw = 1.3, mc = :grey70, msc = C_STOK, label = L"\mathrm{Stokes\ (tidal)}")
scatter!(p, [NaN], [NaN]; marker = :diamond, ms = 7, msw = 1.3, mc = :grey70, msc = C_EKMA, label = L"\mathrm{Ekman\ (rotating)}")
for sv in SVALS
    scatter!(p, [NaN], [NaN]; ms = 5, msw = 0, color = ramp_colour(sv), label = @sprintf("N/ω = N/f = %g", sv))
end
plot!(p; xticks = logticks(lo, hi), xlims = (lo, hi),
         yticks = logticks(minimum(ys) / 2.5, maximum(ys) * 2.5),
         ylims = (minimum(ys) / 2.5, maximum(ys) * 2.5))
annotate!(p, lo * 1.15, maximum(ys) * 2.2,
          text("one point per (case, height), z/h = $(FRACS[1]) to $(FRACS[end]).\nOnly where db/dz clears the $(FLOOR) N²_ref mask in >= $(Int(100KEEP)) % of samples —\nthe well-mixed interior, where Ri is genuinely small, has no gradient\nto measure K_T against and cannot appear here.",
               6, :grey30, :left, :top))
f = plot(p; size = (980, 820),
         plot_title = L"T = 10\,\mathrm{m}:\ \ K_T^{*}\ \mathrm{against\ local}\ Ri,\ \mathrm{scanned\ in\ height}",
         bottom_margin = 6Plots.mm, left_margin = 6Plots.mm, top_margin = 3Plots.mm)
o = joinpath(FIGDIR, "KTstar_vs_Ri_zscan_T10.png"); savefig(f, o); say("\nwrote $o")

mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "plot_KTstar_Ri_zscan_T10.log"), "w") do io
    println(io, join(logl, "\n"))
end
