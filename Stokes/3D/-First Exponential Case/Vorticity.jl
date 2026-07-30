using Oceananigans, JLD2, Plots, Printf, Statistics

# Spanwise vorticity of the tidal boundary layer, from the x–z slices saved by
# Tidal3D.jl:
#   julia --project=.. Vorticity.jl Ri500
#
# Produces, in figures/:
#   Vorticity_<case>.png   four phases of the last complete tidal period
#   Vorticity_<case>.mp4   the whole run
#
# ω_y = ∂u/∂z − ∂w/∂x, scaled by U₀/δ_s. The laminar Stokes layer has
# ∂u/∂z ~ U₀/δ_s at the wall, so O(1) is the laminar wall shear and the
# small-scale patches well above it are the turbulent structures. The colour
# limit is set from the 99.5th percentile of |ω_y| over the run rather than a
# guessed constant, so the same script works across cases.
#
# Staggering: u sits at (Face, Center, Center) and w at (Center, Center, Face),
# so both ∂u/∂z and ∂w/∂x land at (Face, Center, Face) — no interpolation is
# needed, and ω_y is defined on the interior z-faces zf[2:end-1].

include(joinpath(@__DIR__, "case_params.jl"))

const zmax_δ = 25.0
const zmax   = zmax_δ * δ
const ω_scale = U₀ / δ          # ≈ 0.126 s⁻¹ here

xzfile = filename * ".jld2"
isfile(xzfile) || error("Missing $xzfile — run the simulation first")

u_ic = FieldTimeSeries(xzfile, "u", iterations = 0)
grid = u_ic.grid
xf = Array(xnodes(grid, Face()))
zf = Array(znodes(grid, Face()))
zc = Array(znodes(grid, Center()))
Δx = Lx / length(xf)

zω = zf[2:end-1]                       # ω_y lives on the interior z-faces
kω = findall(z -> z <= zmax, zω)

file_xz = jldopen(xzfile)
iterations = sort(parse.(Int, keys(file_xz["timeseries/t"])))
times = [file_xz["timeseries/t/$i"] for i in iterations]

# ω_y[i, k] = (u[i,k+1] − u[i,k])/Δz − (w[i,k+1] − w[i−1,k+1])/Δx, at zf[k+1],
# with the streamwise difference wrapping periodically.
function vorticity(iter)
    u = file_xz["timeseries/u/$iter"][:, 1, :]      # (Nx, Nz)   at z centres
    w = file_xz["timeseries/w/$iter"][:, 1, :]      # (Nx, Nz+1) at z faces
    dudz = (u[:, 2:end] .- u[:, 1:end-1]) ./ reshape(diff(zc), 1, :)
    wi   = w[:, 2:end-1]                            # interior faces only
    dwdx = (wi .- circshift(wi, (1, 0))) ./ Δx
    return dudz .- dwdx
end

# Colour limit from a sample of frames spread over the run. The median of the
# per-frame 99.5th percentiles, taken at the 75th percentile across frames: the
# maximum would be set by the single most violent burst and wash out every other
# frame, while the median would be set by the quiescent slack-water phases and
# saturate the turbulent ones. The upper quartile represents the active phases.
sample = iterations[1:max(1, length(iterations) ÷ 40):end]
ωlim = quantile([quantile(abs.(vec(vorticity(i)[:, kω])), 0.995) for i in sample], 0.75) / ω_scale
ωlim = max(ωlim, 1.0)
@info @sprintf("%s: |ω_y| colour limit = %.2f U₀/δ_s (from %d sampled frames)",
               case, ωlim, length(sample))

default(fontfamily = "sans-serif", framestyle = :box, grid = false,
        tickfontsize = 8, guidefontsize = 9, titlefontsize = 10)

figdir = joinpath(@__DIR__, "figures")
mkpath(figdir)

# Vorticity is signed about zero, so a diverging map: two hues, neutral
# midpoint, limits symmetric so that ω_y = 0 is always the neutral colour.
function frame(iter; kwargs...)
    ωy = vorticity(iter)[:, kω] ./ ω_scale
    heatmap(xf ./ δ, zω[kω] ./ δ, ωy';
            color = :balance, clims = (-ωlim, ωlim),
            xlims = (0, Lx / δ), ylims = (0, zmax_δ), kwargs...)
end

# ---------------- still figure: four phases of the last full period ----------
t0 = max(0.0, (floor(times[end] / T_tide) - 1) * T_tide)
panels = []
for ϕ in (0.0, 0.25, 0.5, 0.75)
    n = argmin(abs.(times .- (t0 + ϕ * T_tide)))
    U∞ = U₀ * sin(ω * times[n])
    push!(panels, frame(iterations[n];
        xlabel = ϕ >= 0.5 ? "x / δ_s" : "",
        ylabel = iseven(round(Int, 4ϕ)) ? "z / δ_s" : "",
        title = @sprintf("φ = %d°,  U∞/U₀ = %+.2f", round(Int, 360ϕ), U∞ / U₀),
        colorbar_title = "  ω_y δ_s / U₀"))
end

Ri_label = passive_scalar ? "0 (passive scalar)" : case[3:end]
fig = plot(panels...; layout = (2, 2), size = (1100, 620),
           leftmargin = 4Plots.mm, bottommargin = 4Plots.mm,
           plot_title = @sprintf("Ri = %s — spanwise vorticity ω_y δ_s/U₀ over one tidal cycle (ωt = %.0f–%.0f)",
                                 Ri_label, ω * t0, ω * (t0 + T_tide)),
           plot_titlefontsize = 12)
savefig(fig, joinpath(figdir, "Vorticity_$case.png"))
@info "Saved figures/Vorticity_$case.png"

# ---------------- animation over the whole run ----------------
stride = max(1, length(iterations) ÷ 600)      # cap the frame count
anim_iters = iterations[1:stride:end]
@info "Animating $(length(anim_iters)) frames..."

anim = @animate for (i, iter) in enumerate(anim_iters)
    i % 100 == 0 && @info "  frame $i / $(length(anim_iters))"
    n = findfirst(==(iter), iterations)
    U∞ = U₀ * sin(ω * times[n])
    frame(iter; xlabel = "x / δ_s", ylabel = "z / δ_s",
          colorbar_title = "  ω_y δ_s / U₀",
          size = (1000, 400), rightmargin = 3Plots.mm,
          title = @sprintf("Ri = %s   ωt = %6.2f   (%.2f T)   U∞/U₀ = %+.2f",
                           Ri_label, ω * times[n], times[n] / T_tide, U∞ / U₀))
end

mp4(anim, joinpath(figdir, "Vorticity_$case.mp4"), fps = 12)
close(file_xz)
@info "Saved figures/Vorticity_$case.mp4"
