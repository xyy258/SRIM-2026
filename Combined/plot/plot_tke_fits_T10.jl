# TKE profiles with one exponential fit per flow, all r pooled:
# TKE/u_*^2 = a exp(-b z/h), fitted from each TKE peak to z = h. Stokes: all r (the
# KTstar_vs_Ri_stokes set). Ekman: r < 25 (the KTstar_vs_Ri_zscan_ekman_lowr set).
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot/plot_tke_fits_T10.jl
#        (needs Data/cache/profiles_T10.jld2)

using JLD2, Plots, Printf, Statistics, LaTeXStrings

get!(ENV, "GKSwstype", "100")
default(dpi = 600, fontfamily = "DejaVu Sans", guidefontsize = 16, tickfontsize = 12,
        legendfontsize = 10, plot_titlefontsize = 18, titlefontsize = 16)

const HERE   = dirname(@__DIR__)
include(joinpath(HERE, "sweep.jl"))
const CACHE  = joinpath(HERE, "Data", "cache", "profiles_T10.jld2")
const DROP   = 25                       # Ekman r >= DROP left out, as in the lowr figure
const ZMAX   = 1.5                      # z/h shown

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))

cases(fl, keep) = jldopen(CACHE, "r") do io
    [(r = r, zc = io[@sprintf("%s/r=%.1f/zc", fl, r)], E = io[@sprintf("%s/r=%.1f/TKE", fl, r)],
      h = io[@sprintf("%s/r=%.1f/h", fl, r)], us = io[@sprintf("%s/r=%.1f/us", fl, r)])
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

# One fit per flow, all r pooled: TKE/u_*^2 = a exp(-b z/h).
function panel(cs, fl, sym)
    p = plot(xscale = :log10, xlabel = L"\mathrm{TKE}/u_*^2", ylabel = L"z/h",
             ylims = (0, ZMAX), legend = :topright, foreground_color_legend = nothing)
    X, Y = Float64[], Float64[]
    say("\n$fl, per case on its own:  r     a      b      rms")
    for c in cs
        idx = range_of(c); x = c.zc[idx] ./ c.h; y = log.(c.E[idx] ./ c.us^2)
        f = linfit(x, y); append!(X, x); append!(Y, y)
        say(@sprintf("  %-5g %7.2f %7.2f %6.1f %%", c.r, exp(f.a), -f.b, f.rms))
        k = findall(j -> c.zc[j] > 0 && c.E[j] > 0 && c.zc[j] / c.h <= ZMAX, eachindex(c.zc))
        plot!(p, c.E[k] ./ c.us^2, c.zc[k] ./ c.h; color = ramp_colour(c.r), lw = 2,
              label = @sprintf("%s = %g", sym, c.r))
    end
    f = linfit(X, Y); A, B = exp(f.a), -f.b
    say(@sprintf("%s, all r pooled:  TKE/u_*^2 = %.3f exp(-%.3f z/h)   rms %.1f %%   n = %d", fl,
                 A, B, f.rms, length(X)))
    zz = range(minimum(X), 1.0; length = 100)
    hline!(p, [1.0]; color = :grey50, ls = :dot, lw = 1, label = L"z = h")
    plot!(p, A .* exp.(-B .* zz), zz; color = :black, lw = 3,
          label = latexstring(@sprintf("\\mathrm{TKE}/u_*^2 = %.2f\\,e^{-%.2f\\,z/h}", A, B)))
end

for (cs, fl, sym, tag) in ((S, "Stokes", "N/ω", "stokes"), (E, "Ekman,\\ N/f < $DROP", "N/f", "ekman_lowr"))
    fig = plot(panel(cs, fl, sym); size = (800, 760),
               title = latexstring("T = 10\\,\\mathrm{m},\\ \\mathrm{$fl}:\\ \\ \\mathrm{TKE\\ profiles\\ and\\ fits}"),
               left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, top_margin = 3Plots.mm)
    o = joinpath(HERE, "figures", "tke_fits_$(tag)_T10.png"); savefig(fig, o); say("wrote $o")
end

mkpath(joinpath(HERE, "logs"))
write(joinpath(HERE, "logs", "plot_tke_fits_T10.log"), join(logl, "\n") * "\n")
