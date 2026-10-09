# TKE profiles with their exponential fits, TKE(z) = A1 exp(-A2 z/delta), fitted
# from the TKE peak to z = h as in plot_tke_profiles_T10.jl. Stokes: all r (the
# KTstar_vs_Ri_stokes set). Ekman: r < 25 (the KTstar_vs_Ri_zscan_ekman_lowr set).
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
const ZMAX   = 1.5                      # z/h shown

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))

δ = Dict((fl, r) => jldopen(io -> io[@sprintf("%s/r=%.1f/best", fl, r)], DFILE, "r")
         for fl in ("stokes", "ekman") for r in (fl == "stokes" ? SVALS : RATIOS))
cases(fl, keep) = jldopen(CACHE, "r") do io
    [(r = r, zc = io[@sprintf("%s/r=%.1f/zc", fl, r)], E = io[@sprintf("%s/r=%.1f/TKE", fl, r)],
      h = io[@sprintf("%s/r=%.1f/h", fl, r)], us = io[@sprintf("%s/r=%.1f/us", fl, r)],
      δ = δ[(fl, float(r))]) for r in io["$fl/ratios"] if keep(r)]
end
S = cases("stokes", r -> true)
E = cases("ekman",  r -> r < DROP)

# Least squares on log TKE against z/delta, from the TKE peak to z = h.
function expfit(c)
    ok  = findall(k -> isfinite(c.E[k]) && c.E[k] > 0 && c.zc[k] > 0, eachindex(c.zc))
    kpk = ok[argmax(c.E[ok])]
    idx = [k for k in ok if k >= kpk && c.zc[k] <= c.h]
    x = c.zc[idx] ./ c.δ; y = log.(c.E[idx])
    b = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)) .^ 2)
    a = mean(y) - b * mean(x)
    (A1 = exp(a), A2 = -b, rms = 100 * sqrt(mean((y .- a .- b .* x) .^ 2)),
     z = c.zc[idx])
end

function panel(cs, fl, sym)
    p = plot(xscale = :log10, xlabel = L"\mathrm{TKE}/u_*^2", ylabel = L"z/h",
             ylims = (0, ZMAX),
             legend = :topright, foreground_color_legend = nothing)
    say("\n$fl:  r     A1/u_*^2     A2    A2 h/delta    rms")
    for c in cs
        f = expfit(c)
        k = findall(j -> c.zc[j] > 0 && c.E[j] > 0 && c.zc[j] / c.h <= ZMAX, eachindex(c.zc))
        plot!(p, c.E[k] ./ c.us^2, c.zc[k] ./ c.h; color = ramp_colour(c.r), lw = 2,
              label = @sprintf("%s = %g", sym, c.r))
        plot!(p, f.A1 .* exp.(-f.A2 .* f.z ./ c.δ) ./ c.us^2, f.z ./ c.h;
              color = :black, ls = :dash, lw = 1.2, label = "")
        say(@sprintf("  %-5g %8.2f %9.3f %9.3f %8.1f %%", c.r, f.A1 / c.us^2, f.A2,
                     f.A2 * c.h / c.δ, f.rms))
    end
    hline!(p, [1.0]; color = :grey50, ls = :dot, lw = 1, label = L"z = h")
    plot!(p, [NaN], [NaN]; color = :black, ls = :dash, lw = 1.2,
          label = L"A_1 e^{-A_2 z/\delta}\ \mathrm{fit}")
end

for (cs, fl, sym, tag) in ((S, "Stokes", "N/ω", "stokes"), (E, "Ekman,\\ N/f < $DROP", "N/f", "ekman_lowr"))
    fig = plot(panel(cs, fl, sym); size = (800, 760),
               title = latexstring("T = 10\\,\\mathrm{m},\\ \\mathrm{$fl}:\\ \\ \\mathrm{TKE\\ profiles\\ and\\ fits}"),
               left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, top_margin = 3Plots.mm)
    o = joinpath(HERE, "figures", "tke_fits_$(tag)_T10.png"); savefig(fig, o); say("wrote $o")
end

mkpath(joinpath(HERE, "logs"))
write(joinpath(HERE, "logs", "plot_tke_fits_T10.log"), join(logl, "\n") * "\n")
