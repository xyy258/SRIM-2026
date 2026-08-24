using Oceananigans, JLD2, Plots, Printf, Statistics

# Animation of the plane-averaged vorticity profiles of the tidal boundary layer,
# built from the mean profiles Tidal3D.jl writes to *_profiles.jld2:
#   julia --project=.. VorticityProfileAnim.jl Ri500
#
# In a horizontally periodic domain the plane average removes the x and y
# derivatives, so the plane-averaged vorticity components are exactly the
# vertical shears of the mean horizontal velocity:
#     ⟨ω_x⟩ = ⟨∂w/∂y − ∂v/∂z⟩ = −dV/dz
#     ⟨ω_y⟩ = ⟨∂u/∂z − ∂w/∂x⟩ = +dU/dz
# Without rotation the mean spanwise flow is nearly zero, so ⟨ω_x⟩ is nearly zero
# and ⟨ω_y⟩ carries the oscillating Stokes shear. Both are plotted.
#
# Horizontal axis: vorticity normalised by the forcing frequency, ⟨ω⟩/ω.
# Vertical axis:   height z in metres. One frame per saved profile.

include(joinpath(@__DIR__, "case_params.jl"))

const zmax_m = 10.0                        # metres shown (near-wall region)

pfile = filename * "_profiles.jld2"
isfile(pfile) || error("Missing $pfile — run the simulation first")

Ufield = FieldTimeSeries(pfile, "U")
Vfield = FieldTimeSeries(pfile, "V")
times  = Ufield.times

zc     = Array(znodes(Ufield))             # cell centres (m); U, V live at Center
zf_int = (zc[1:end-1] .+ zc[2:end]) ./ 2   # midpoints where dU/dz, dV/dz land
Δzc    = diff(zc)
kz     = findall(z -> z <= zmax_m, zf_int)

# ⟨ω_x⟩/ω = −(dV/dz)/ω and ⟨ω_y⟩/ω = +(dU/dz)/ω, at the midpoints zf_int.
function omega_profiles(n)
    U = vec(interior(Ufield[n]))
    V = vec(interior(Vfield[n]))
    ωy =  (diff(U) ./ Δzc) ./ ω
    ωx = -(diff(V) ./ Δzc) ./ ω
    return ωx, ωy
end

# Keep the animation short by taking at most about 300 frames.
stride = max(1, length(times) ÷ 300)
frames = 1:stride:length(times)

# Symmetric x-limits, from a percentile over every frame shown.
allvals = Float64[]
for n in frames
    ωx, ωy = omega_profiles(n)
    append!(allvals, abs.(ωx[kz]))
    append!(allvals, abs.(ωy[kz]))
end
xlim = max(quantile(allvals, 0.995), 1e-6)

default(fontfamily = "sans-serif", framestyle = :box, grid = false,
        tickfontsize = 8, guidefontsize = 10, legendfontsize = 9, titlefontsize = 10)

figdir = joinpath(@__DIR__, "figures"); mkpath(figdir)

@info @sprintf("%s: plane-averaged vorticity animation — %d frames (stride %d), |⟨ω⟩/ω| ≤ %.2f",
               casetag, length(frames), stride, xlim)

anim = @animate for n in frames
    ωx, ωy = omega_profiles(n)
    U∞ = U₀ * sin(ω * times[n])
    plot(ωy[kz], zf_int[kz]; label = "⟨ω_y⟩/ω  =  (dU/dz)/ω",
         color = :crimson, lw = 2,
         xlims = (-xlim, xlim), ylims = (0, zmax_m),
         xlabel = "⟨ω⟩ / ω", ylabel = "z (m)",
         legend = :topright, size = (620, 720),
         title = @sprintf("%s   ωt = %.2f (%.2f periods),  U∞/U₀ = %+.2f",
                          casetag, ω * times[n], times[n] / T_tide, U∞ / U₀))
    plot!(ωx[kz], zf_int[kz]; label = "⟨ω_x⟩/ω  = -(dV/dz)/ω",
          color = :royalblue, lw = 2)
    vline!([0]; color = :gray, ls = :dash, label = "")
end

mp4(anim, joinpath(outdir, "vorticity_profile_" * casetag * ".mp4"), fps = 15)
@info "Saved vorticity_profile_$casetag.mp4 in $outdir/"
