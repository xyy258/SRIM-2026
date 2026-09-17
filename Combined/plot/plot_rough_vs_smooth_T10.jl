# Is Ri at the Ekman layer top self-regulated, or only coincidentally pinned?
#
# One pair: r = 25 at the swept roughness z0 = 0.0016 m against the same case at
# z0 = 0.0137 m, which is c_D x4 and so u_* x2 at identical U_inf, f and N. If Ri
# at z = h is set by the flow rather than by the forcing, it should not move.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot/plot_rough_vs_smooth_T10.jl

using Oceananigans, JLD2, Plots, Printf, Statistics, LaTeXStrings

get!(ENV, "GKSwstype", "100")
default(dpi = 600, fontfamily = "DejaVu Sans")

const HERE   = dirname(@__DIR__)        # scripts live one level down
const FIGDIR = joinpath(HERE, "figures")
const f₀     = 1e-4
const T_f    = 2π / f₀
const WINDOW = 4
const R      = 25.0
const N      = R * f₀
const SMOOTH = (dir = joinpath(HERE, "Data", "raw", "Ekman_moments", "4", "r=25.0, T=10.0"),
                red = joinpath(HERE, "Data", "cache", "ekman_lengthscales_T10_moments.jld2"),
                z0 = 0.0016, cD = 0.00908, lab = "smooth")
const ROUGH  = (dir = joinpath(HERE, "Data", "raw", "rough_ekman", "r=25.0, T=10.0"),
                red = joinpath(HERE, "Data", "cache", "ekman_lengthscales_rough_T10.jld2"),
                z0 = 0.0137, cD = 0.03621, lab = "rough")

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

function load(c)
    file = joinpath(c.dir, "Moments.jld2")
    F = Dict(v => FieldTimeSeries(file, v; backend = OnDisk()) for v in
             ("U", "V", "W", "uw", "vw", "kappa_sgs"))
    g = F["U"].grid; zc = Array(znodes(g, Center())); zf = Array(znodes(g, Face()))
    tv = F["U"].times; sel = findall(t -> t >= tv[end] - WINDOW * T_f, tv)
    nt = length(sel)
    col(v, n) = Float64.(Array(interior(F[v][n], 1, 1, :)))
    Sr = Array{Float64}(undef, length(zf), nt); Tr = similar(Sr)
    for (i, n) in enumerate(sel)
        U, V, W = col("U", n), col("V", n), col("W", n)
        dU = zeros(length(zf)); dV = zeros(length(zf))
        @inbounds for k in 2:length(zc)
            dz = zc[k] - zc[k-1]
            dU[k] = (U[k] - U[k-1]) / dz; dV[k] = (V[k] - V[k-1]) / dz
        end
        dU[1] = dU[2]; dU[end] = dU[end-1]; dV[1] = dV[2]; dV[end] = dV[end-1]
        Sr[:, i] = hypot.(dU, dV)
        ks = col("kappa_sgs", n)
        Uf = c2f(U, zc, zf); Vf = c2f(V, zc, zf)
        Tr[:, i] = hypot.((col("uw", n) .- Uf .* W) .- ks .* dU,
                          (col("vw", n) .- Vf .* W) .- ks .* dV)
    end
    dt = tv[sel[2]] - tv[sel[1]]; nh = max(0, round(Int, (T_f / 20) / (2dt)))
    S = boxcar(Sr, nh); TAU = boxcar(Tr, nh)
    d = jldopen(c.red, "r") do io
        g = @sprintf("r=%.1f", R)
        (h = io["$g/h"], K = io["$g/K_at_h"], E = io["$g/TKE_at_h"], S = io["$g/S_at_h"])
    end
    # u_*^2 = largest total stress below the layer top, per sample — as reduce_profiles_T10.jl
    us = [sqrt(maximum(fin(view(TAU, 1:max(2, searchsortedlast(zf, d.h[i])), i)))) for i in 1:nt]
    (zf = zf, S = S, TAU = TAU, h = med(d.h), us = med(us),
     Sh = med(d.S), K = med(d.K), E = med(d.E),
     Kst = med(d.K .* N ./ d.E), sRi = med(N ./ d.S),
     prof = [med(view(S, k, :)) for k in axes(S, 1)])
end

A = load(SMOOTH); B = load(ROUGH)
say(@sprintf("Ekman r = %.0f, T = 10 m: does raising u_* move Ri at z = h?", R))
say("")
say("  quantity                       smooth        rough      change")
row(nm, a, b) = say(@sprintf("  %-28s %11.4e %11.4e %9.2f x", nm, a, b, b / a))
row("z0 imposed (m)", SMOOTH.z0, ROUGH.z0)
row("c_D", SMOOTH.cD, ROUGH.cD)
row("u_* measured (m/s)", A.us, B.us)
row("h (m)", A.h, B.h)
row("S at h (1/s)", A.Sh, B.Sh)
row("TKE at h (m2/s2)", A.E, B.E)
row("K_T at h (m2/s)", A.K, B.K)
say("")
row("sqrt(Ri) = N/S at h", A.sRi, B.sRi)
row("K_T* = K_T N/TKE at h", A.Kst, B.Kst)
say("")
say(@sprintf("u_*/f  : %.2f m -> %.2f m  (x%.2f);   h/(u_*/f) : %.3f -> %.3f",
             A.us / f₀, B.us / f₀, (B.us / f₀) / (A.us / f₀), A.h / (A.us / f₀), B.h / (B.us / f₀)))

mkpath(FIGDIR)
# Above the layer S dies and sqrt(Ri) runs to 1e9, which would squash everything
# of interest against the left edge. Cut the axis where the layer ends.
p = plot(xscale = :log10, xlabel = L"\sqrt{Ri}(z) = N/S(z)", ylabel = L"z \ \ (\mathrm{m})",
         ylims = (0, 22), xlims = (0.4, 40), legend = :topright, legendfontsize = 7,
         foreground_color_legend = nothing)
vline!(p, [med([A.sRi, B.sRi])]; color = :grey60, lw = 1.2, ls = :dot, label = "")
for (c, s, col, lab) in ((A, SMOOTH, "#1b3a6b", "smooth"), (B, ROUGH, "#c46a1f", "rough"))
    x = N ./ c.prof
    k = findall(i -> isfinite(x[i]) && x[i] > 0 && c.zf[i] > 0.3, eachindex(x))
    plot!(p, x[k], c.zf[k]; color = col, lw = 2.2,
          label = latexstring(@sprintf("\\mathrm{%s}\\!: \\ z_0 = %.4f\\,\\mathrm{m}, \\ c_D = %.4f", lab, s.z0, s.cD)))
    scatter!(p, [c.sRi], [c.h]; ms = 11, msw = 1.8, mc = col, msc = :black, marker = :diamond,
             label = latexstring(@sprintf("z = h = %.2f\\,\\mathrm{m}, \\ \\sqrt{Ri} = %.2f", c.h, c.sRi)))
end
annotate!(p, 0.45, 5.5,
          text(@sprintf("c_D x%.1f raised u_* x%.2f and h x%.2f,\nbut sqrt(Ri) at z = h moved only %+.0f %%.\nThe layer thickened instead of shearing harder.",
                        ROUGH.cD / SMOOTH.cD, B.us / A.us, B.h / A.h,
                        100 * (B.sRi / A.sRi - 1)), 7, :grey25, :left, :bottom))
f = plot(p; size = (900, 780),
         plot_title = L"\mathrm{Ekman}\ r = 25:\ \ \mathrm{a\ 4\times\ rougher\ bed\ does\ not\ move}\ Ri\ \mathrm{at\ the\ layer\ top}",
         bottom_margin = 6Plots.mm, left_margin = 6Plots.mm, top_margin = 3Plots.mm)
o = joinpath(FIGDIR, "rough_vs_smooth_Ri_T10.png"); savefig(f, o); say("\nwrote $o")

mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "plot_rough_vs_smooth_T10.log"), "w") do io
    println(io, join(logl, "\n"))
end
