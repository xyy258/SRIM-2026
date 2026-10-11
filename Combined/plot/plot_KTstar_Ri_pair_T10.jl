# Stokes and Ekman K_T* against sqrt(Ri) (z = h, background N) side by side on
# one A4-landscape figure for the slides. Runs the two single-flow scripts, each
# in its own module so their globals do not clash, and lays out their panels.
# Also rewrites the two single figures and their logs.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot/plot_KTstar_Ri_pair_T10.jl

module S; include(joinpath(@__DIR__, "plot_KTstar_Ri_stokes_T10.jl")); end
module E; include(joinpath(@__DIR__, "plot_KTstar_Ri_ekman_T10.jl")); end

using Plots, LaTeXStrings

o = joinpath(dirname(@__DIR__), "figures", "KTstar_vs_Ri_pair_T10.png")
# Smaller legends than the single figures so they fit the empty corners of a half-width panel.
pan(p, t) = plot!(p; title = t, titlefontsize = 20, legendfontsize = 12, background_color_legend = nothing,
                  xlabel = L"\sqrt{Ri} = N/S")
savefig(plot(pan(S.p, "Stokes"), pan(E.p, "Ekman"); layout = (1, 2), size = (1169, 827),
             plot_title = L"T = 10\,\mathrm{m},\ z = h,\ N\ \mathrm{background}", left_margin = 6Plots.mm,
             bottom_margin = 6Plots.mm, right_margin = 4Plots.mm), o)
println("wrote $o")
