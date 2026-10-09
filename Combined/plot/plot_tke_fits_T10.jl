# TKE profiles with one exponential fit per flow, all r pooled:
# TKE/u_*^2 = A1 exp(-A2 z/delta), fitted from each TKE peak to z = h. Height is
# scaled by delta, not h, so TKE(h) = A1 u_*^2 exp(-A2 h/delta) depends on h.
# Stokes: all r (the KTstar_vs_Ri_stokes set). Ekman: r < 25 (the
# KTstar_vs_Ri_zscan_ekman_lowr set).
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot/plot_tke_fits_T10.jl
#        (needs Data/cache/profiles_T10.jld2 and delta_T10.jld2)

using JLD2, Plots, Printf, Statistics, LaTeXStrings

get!(ENV, "GKSwstype", "100")
default(dpi = 600, fontfamily = "DejaVu Sans", guidefontsize = 16, tickfontsize = 12,
        legendfontsize = 10, plot_titlefontsize = 18, titlefontsize = 16)

const HERE   = dirname(@__DIR__)
include(joinpath(HERE, "sweep.jl"))
const CACHE  = joinpath(HERE, "Data", "cache", "profiles_T10.jld2")
const DFILE  = joinpath(HERE, "Data", "cache", "delta_T10.jld2")
const DROP   = 25                       # Ekman r >= DROP left out, as in the lowr figure
const ZMAX   = 1.5                      # shown up to ZMAX times the largest h/delta

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))

cases(fl, keep) = jldopen(CACHE, "r") do io
    [(r = r, zc = io[@sprintf("%s/r=%.1f/zc", fl, r)], E = io[@sprintf("%s/r=%.1f/TKE", fl, r)],
      h = io[@sprintf("%s/r=%.1f/h", fl, r)], us = io[@sprintf("%s/r=%.1f/us", fl, r)],
      δ = jldopen(d -> d[@sprintf("%s/r=%.1f/best", fl, r)], DFILE, "r"))
      for r in io["$fl/ratios"] if keep(r)]
end
S = cases("stokes", r -> true)
E = cases("ekman",  r -> r < DROP)

# Fit range per case: the TKE peak up to z = h.
function range_of(c)
    ok  = findall(k -> isfinite(c.E[k]) && c.E[k] > 0 && c.zc[k] > 0, eachindex(c.zc))
    kpk = ok[argmax(c.E[ok])]
    [k for k in ok if k >= kpk && c.zc[k] <= c.h]
end

# Least squares on y against x; returns intercept, slope, rms of the log residual.
function linfit(x, y)
    b = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)) .^ 2)
    a = mean(y) - b * mean(x)
    (a = a, b = b, rms = 100 * sqrt(mean((y .- a .- b .* x) .^ 2)))
end

# One fit per flow, all r pooled: TKE/u_*^2 = A1 exp(-A2 z/delta).
function panel(cs, fl, sym, c0, w, pp)
    ztop = ZMAX * maximum(c.h / c.δ for c in cs)
    p = plot(xscale = :log10, xlabel = L"\mathrm{TKE}/u_*^2", ylabel = L"z/\delta",
             ylims = (0, ztop), legend = :topright, foreground_color_legend = nothing)
    X, Y = Float64[], Float64[]
    say("\n$fl, per case on its own:  r   delta (m)   h/delta     A1      A2      rms")
    for c in cs
        idx = range_of(c); x = c.zc[idx] ./ c.δ; y = log.(c.E[idx] ./ c.us^2)
        f = linfit(x, y); append!(X, x); append!(Y, y)
        say(@sprintf("  %-5g %9.2f %9.3f %7.2f %7.3f %6.1f %%", c.r, c.δ, c.h / c.δ, exp(f.a), -f.b, f.rms))
        k = findall(j -> c.zc[j] > 0 && c.E[j] > 0 && c.zc[j] / c.δ <= ztop, eachindex(c.zc))
        plot!(p, c.E[k] ./ c.us^2, c.zc[k] ./ c.δ; color = ramp_colour(c.r), lw = 2,
              label = @sprintf("%s = %g", sym, c.r))
    end
    f = linfit(X, Y); A, B = exp(f.a), -f.b
    say(@sprintf("%s, all r pooled:  TKE/u_*^2 = %.3f exp(-%.3f z/delta)   rms %.1f %%   n = %d", fl,
                 A, B, f.rms, length(X)))
    zz = range(minimum(X), maximum(X); length = 100)
    plot!(p, A .* exp.(-B .* zz), zz; color = :black, lw = 3,
          label = latexstring(@sprintf("\\mathrm{TKE}/u_*^2 = %.2f\\,e^{-%.2f\\,z/\\delta}", A, B)))
    us = [c.us for c in cs]; e = floor(Int, log10(median(us)))
    say(@sprintf("%s: u_* = %.3e m/s median, %.3e to %.3e", fl, median(us), minimum(us), maximum(us)))
    plot!(p, [NaN], [NaN]; color = :white, lw = 0,
          label = latexstring(@sprintf("u_* = %.2f \\times 10^{%d}\\ \\mathrm{m\\,s^{-1}}", median(us) / 10.0^e, e)))
    plot!(p, [NaN], [NaN]; color = :white, lw = 0,
          label = latexstring(@sprintf("\\delta = %.1f\\,u_*/%s\\,(1+N^2/%s^2)^{-%.3f}", c0, w, w, pp)))
end

pS, pE = jldopen(io -> (io["p_stokes"], io["p_ekman"]), DFILE, "r")
for (cs, fl, sym, tag, c0, w, pp) in ((S, "Stokes", "N/ω", "stokes", 0.4, "\\omega", pS),
                                      (E, "Ekman,\\ N/f < $DROP", "N/f", "ekman_lowr", 1.3, "f", pE))
    fig = plot(panel(cs, fl, sym, c0, w, pp); size = (800, 760),
               title = latexstring("T = 10\\,\\mathrm{m},\\ \\mathrm{$fl}:\\ \\ \\mathrm{TKE\\ profiles\\ and\\ fits}"),
               left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, top_margin = 3Plots.mm)
    o = joinpath(HERE, "figures", "tke_fits_$(tag)_T10.png"); savefig(fig, o); say("wrote $o")
end

mkpath(joinpath(HERE, "logs"))
write(joinpath(HERE, "logs", "plot_tke_fits_T10.log"), join(logl, "\n") * "\n")
