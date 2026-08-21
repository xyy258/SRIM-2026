using Oceananigans, JLD2, Plots, Printf

# Analysis of the horizontally averaged profiles saved by Tidal2D.jl.
# Produces:
#   1. Heatmap of ∂b/∂z normalized by N², over the bottom few metres
#   2. Mixed-layer depth against time, next to the pure-diffusion prediction
#   3. Stratification profiles at the end of each tidal period
#   4. Mean velocity profiles against the laminar Stokes solution

# ---- parameters, which must match the simulation ----
ω  = 1.4075235e-4
N² = 1e-7
ν  = 1.109e-5
κ  = ν
U₀ = 0.05
δ  = sqrt(2ν / ω)
T_tide = 2π / ω

fname = "TidalBoundaryLayer2D_profiles.jld2"

B_ts = FieldTimeSeries(fname, "B")     # mean buoyancy on cell centers
U_ts = FieldTimeSeries(fname, "U")     # mean velocity on cell centers

times = B_ts.times
Nt    = length(times)

zc = znodes(B_ts)      # centers, length Nz

# The saved `dbdz` field is deliberately not used here. ∂z(B) evaluates its top
# and bottom boundary faces from the halo of the averaged field, which the
# no-flux boundary condition on buoyancy does not set, and that produced a
# spurious bright line at z = 0. The gradient is instead reconstructed by
# differencing B between adjacent centers, which uses interior data only.
Bmean = zeros(length(zc), Nt)
Umean = zeros(length(zc), Nt)
for n in 1:Nt
    Bmean[:, n] .= vec(interior(B_ts[n]))
    Umean[:, n] .= vec(interior(U_ts[n]))
end

zg = 0.5 .* (zc[1:end-1] .+ zc[2:end])         # midpoints, length Nz-1
G  = diff(Bmean, dims = 1) ./ diff(zc)         # ∂b/∂z at midpoints, Nz-1 × Nt

# ---- 1. Normalized gradient heatmap, zoomed to the bottom ----
zmax = 3.0                       # everything happens within a few Stokes layers
ks   = findall(z -> z <= zmax, zg)

heatmap(times ./ T_tide, zg[ks], G[ks, :] ./ N²;
        clims = (0, 2),          # 1 = unmixed, 0 = mixed, >1 = sharpened pycnocline
        color = :thermal,
        xlabel = "t / tidal period",
        ylabel = "z (m)",
        title = "∂b/∂z / N² (bottom $(zmax) m)",
        colorbar_title = "∂b/∂z / N²")
savefig("buoyancy_gradient_normalized.png")

# ---- 2. Mixed-layer depth against time ----
# Two measures:
#  (a) Threshold depth: the highest continuous height from the wall at which the
#      stratification is below half the background. It is easy to interpret but
#      spikes when a weakly stratified column momentarily joins the wall to an
#      overturn higher up, which is an artefact rather than real deepening.
#  (b) Integral mixing thickness, ∫ (1 − ∂b/∂z / N²)₊ dz, the height-integrated
#      fraction of the stratification that has been removed, clamped to [0,1] so
#      that restratified points do not contribute negatively. Being an integral
#      it is smooth, and it is the more trustworthy of the two.
function threshold_depth(g, z; threshold = 0.5N²)
    k = 1
    while k <= length(z) && g[k] < threshold
        k += 1
    end
    return k == 1 ? 0.0 : z[k-1]
end

function mixing_thickness(g, z)
    deficit = clamp.(1 .- g ./ N², 0, 1)              # 1 = fully mixed, 0 = untouched
    dz = diff(z)
    body = sum(0.5 .* (deficit[1:end-1] .+ deficit[2:end]) .* dz)  # trapezoid
    wall = deficit[1] * z[1]                          # wall → first midpoint
    return wall + body
end

mld_thr = [threshold_depth(G[:, n], zg)   for n in 1:Nt]
mld_int = [mixing_thickness(G[:, n], zg)  for n in 1:Nt]

plot(times ./ T_tide, mld_int;
     lw = 2, label = "mixing thickness ∫(1−∂b/∂z/N²) dz",
     xlabel = "t / tidal period", ylabel = "mixed-layer thickness (m)")
plot!(times ./ T_tide, mld_thr;
      lw = 1, alpha = 0.5, label = "threshold depth (∂b/∂z < ½N²)")
plot!(times ./ T_tide, sqrt.(2κ .* times);
      lw = 2, ls = :dash, label = "pure diffusion √(2κt)")
hline!([δ]; ls = :dot, label = "Stokes layer δ")
savefig("mixed_layer_depth.png")


# ---- 3. Stratification profiles at the end of each tidal period ----
plt = plot(xlabel = "∂b/∂z / N²", ylabel = "z (m)", ylims = (0, zmax),
           xlims = (0, 1.5),   # 1 = unmixed background, below 1 mixed; capped to stay readable
           title = "Stratification profiles", legend = :bottomright)
for p in 0:floor(Int, times[end] / T_tide)
    n = argmin(abs.(times .- p * T_tide))
    plot!(plt, G[ks, n] ./ N², zg[ks]; lw = 2,
          label = @sprintf("t = %d periods", p))
end
savefig("stratification_profiles.png")

# ---- 4. Mean velocity against the laminar Stokes solution ----
# Laminar solution for an oscillating free stream U₀ sin(ωt) over a wall:
#   u(z, t) = U₀ [ sin(ωt) − e^(−z/δ) sin(ωt − z/δ) ]
# If the simulated profile matches this the flow is laminar; a fuller, more
# slab-like profile with a thin sharp wall layer means it is turbulent.
u_laminar(z, t) = U₀ * (sin(ω * t) - exp(-z / δ) * sin(ω * t - z / δ))

phases = (0.25, 0.5, 0.75, 1.0)  # fractions of the final tidal period
plt2 = plot(xlabel = "u (m/s)", ylabel = "z (m)", ylims = (0, zmax),
            title = "Mean velocity: simulation (solid) vs laminar (dashed)",
            legend = :bottomright)
t0 = (floor(times[end] / T_tide) - 1) * T_tide   # start of final full period
for (i, ϕ) in enumerate(phases)
    t = t0 + ϕ * T_tide
    n = argmin(abs.(times .- t))
    kc = findall(z -> z <= zmax, zc)
    plot!(plt2, Umean[kc, n], zc[kc]; lw = 2, c = i,
          label = @sprintf("phase %.2f T", ϕ))
    plot!(plt2, u_laminar.(zc[kc], times[n]), zc[kc]; lw = 1.5, ls = :dash,
          c = i, label = "")
end
savefig("velocity_vs_stokes.png")

@info "Saved: buoyancy_gradient_normalized.png, mixed_layer_depth.png, stratification_profiles.png, velocity_vs_stokes.png"