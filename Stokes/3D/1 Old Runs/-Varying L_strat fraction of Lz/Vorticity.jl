using Oceananigans, JLD2, Plots, Printf, Statistics

# Spanwise-averaged spanwise vorticity ⟨ω_y⟩_y(x, z) of the tidal boundary layer,
# from the x–z cross-section field written by Tidal3D.jl (writer 1b):
#   julia --project=.. Vorticity.jl Ri500
#
# ω_y = ∂u/∂z − ∂w/∂x, averaged over the spanwise (y) direction to give a 2D
# cross-section in the x–z plane. Because the flow is statistically homogeneous
# in x and y, the spanwise average mostly retains the phase-organized mean shear
# (and its large-scale modulation), with the incoherent turbulent fluctuations
# averaged down — a clean view of the coherent vorticity structure over the cycle.
#
# Produces figures/Vorticity_<case>.png: four phases of the last full period.
# Scaled by U₀/δ_s (the laminar wall-shear scale). Depth axis in metres.

include(joinpath(@__DIR__, "case_params.jl"))

const zmax_m  = 10.0            # metres shown (test section; sponge excluded)
const ω_scale = U₀ / δ          # ≈ 0.28 s⁻¹ here (U₀ = 0.04, δ_s ≈ 0.141)

vortfile = filename * "_vortxz.jld2"
isfile(vortfile) || error("Missing $vortfile — run the simulation first")

Ω = FieldTimeSeries(vortfile, "omega_y")
times = Ω.times
xf = Array(xnodes(Ω))          # streamwise nodes (m)
zf = Array(znodes(Ω))          # vertical nodes (m); ω_y lives on z-faces
kz = findall(z -> z <= zmax_m, zf)

# Colour limit from the upper quartile of per-frame 99.5th percentiles over a
# sample of frames: robust to the single most violent frame and to slack water.
sample = 1:max(1, length(times) ÷ 40):length(times)
frame_hi(n) = quantile(abs.(vec(interior(Ω[n])[:, 1, kz])), 0.995) / ω_scale
ωlim = max(quantile([frame_hi(n) for n in sample], 0.75), 1.0)
@info @sprintf("%s: |⟨ω_y⟩| colour limit = %.2f U₀/δ_s (from %d sampled frames)",
               casetag, ωlim, length(sample))

default(fontfamily = "sans-serif", framestyle = :box, grid = false,
        tickfontsize = 8, guidefontsize = 9, titlefontsize = 10)

figdir = joinpath(@__DIR__, "figures")
mkpath(figdir)

# Diverging map, symmetric limits so ⟨ω_y⟩ = 0 is always the neutral colour.
function panel(n; kwargs...)
    ωy = interior(Ω[n])[:, 1, kz] ./ ω_scale
    heatmap(xf, zf[kz], ωy';
            color = :balance, clims = (-ωlim, ωlim),
            xlims = (0, Lx), ylims = (0, zmax_m), kwargs...)
end

# Four phases of the last full period.
t0 = max(0.0, (floor(times[end] / T_tide) - 1) * T_tide)
panels = []
for ϕ in (0.0, 0.25, 0.5, 0.75)
    n = argmin(abs.(times .- (t0 + ϕ * T_tide)))
    U∞ = U₀ * sin(ω * times[n])
    push!(panels, panel(n;
        xlabel = ϕ >= 0.5 ? "x (m)" : "",
        ylabel = iseven(round(Int, 4ϕ)) ? "z (m)" : "",
        title = @sprintf("φ = %d°,  U∞/U₀ = %+.2f", round(Int, 360ϕ), U∞ / U₀),
        colorbar_title = "  ⟨ω_y⟩ δ_s / U₀"))
end

Ri_label = passive_scalar ? "0 (passive scalar)" : case[3:end]
fig = plot(panels...; layout = (2, 2), size = (1100, 620),
           leftmargin = 4Plots.mm, bottommargin = 4Plots.mm,
           plot_title = @sprintf("%s — spanwise-averaged vorticity ⟨ω_y⟩ δ_s/U₀ over one cycle (ωt = %.0f–%.0f)",
                                 casetag, ω * t0, ω * (t0 + T_tide)),
           plot_titlefontsize = 12)
savefig(fig, joinpath(figdir, "Vorticity_$casetag.png"))
@info "Saved figures/Vorticity_$casetag.png"
