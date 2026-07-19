using Oceananigans, JLD2, Plots, Printf

# Analysis of horizontally averaged profiles saved by TidalBoundaryLayer.jl
# Produces:
#   1. Heatmap of ∂b/∂z normalized by N² (bottom few metres, fixed color scale)
#   2. Mixed-layer depth vs time, compared with the pure-diffusion prediction
#   3. Stratification profiles at the end of each tidal period
#   4. Mean velocity profiles vs the laminar Stokes solution

# ---- parameters (must match the simulation) ----
ω  = 1.4075235e-4
N² = 1e-7
ν  = 1.109e-5
κ  = ν
U₀ = 0.05
δ  = sqrt(2ν / ω)
T_tide = 2π / ω

fname = "Stokes/TidalBoundaryLayer3D_profiles.jld2"

B_ts = FieldTimeSeries(fname, "B")     # mean buoyancy on cell CENTERS
U_ts = FieldTimeSeries(fname, "U")     # mean velocity on cell centers

times = B_ts.times
Nt    = length(times)

zc = znodes(B_ts)      # centers, length Nz

# IMPORTANT: we do NOT use the saved `dbdz` field here.
# `dbdz = ∂z(B)` evaluates its bottom/top *boundary faces* using the halo of the
# averaged field, which is NOT set by the buoyancy no-flux BC — that produced the
# spurious bright line at z = 0 (where the physical no-flux value is exactly 0).
# Instead we reconstruct the gradient by finite-differencing the mean buoyancy B
# between adjacent centers. This uses only interior data: no halos, no BC, no
# boundary faces, so the artifact cannot appear.
Bmean = zeros(length(zc), Nt)
Umean = zeros(length(zc), Nt)
for n in 1:Nt
    Bmean[:, n] .= vec(interior(B_ts[n]))
    Umean[:, n] .= vec(interior(U_ts[n]))
end

zg = 0.5 .* (zc[1:end-1] .+ zc[2:end])         # midpoints, length Nz-1
G  = diff(Bmean, dims = 1) ./ diff(zc)         # ∂b/∂z at midpoints, Nz-1 × Nt

# ---- 1. Normalized gradient heatmap, zoomed to the bottom ----
zmax = 3.0                       # the action is within a few Stokes layers
ks   = findall(z -> z <= zmax, zg)

heatmap(times ./ T_tide, zg[ks], G[ks, :] ./ N²;
        clims = (0, 2),          # 1 = unmixed, 0 = mixed, >1 = sharpened pycnocline
        color = :thermal,
        xlabel = "t / tidal period",
        ylabel = "z (m)",
        title = "∂b/∂z / N² (bottom $(zmax) m)",
        colorbar_title = "∂b/∂z / N²")
savefig("Stokes/buoyancy_gradient_normalized3D.png")

# ---- 2. Mixed-layer depth vs time ----
# Two metrics:
#  (a) THRESHOLD depth: highest contiguous height from the wall where the
#      stratification is below half background. Intuitive, but spikes when a
#      low-gradient column momentarily connects the wall to a burst/overturn
#      high up — those spikes are a metric artifact, not real deepening.
#  (b) INTEGRAL "mixing thickness": ∫ (1 − ∂b/∂z / N²)₊ dz, i.e. the height-
#      integrated fraction of stratification that has been removed (clamped to
#      [0,1] so restratified pixels with ∂b/∂z > N² don't contribute negatively).
#      Being an integral, it is smooth and is the more trustworthy measure.
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
savefig("Stokes/mixed_layer_depth3D.png")


# ---- 3. Stratification profiles at the end of each tidal period ----
plt = plot(xlabel = "∂b/∂z / N²", ylabel = "z (m)", ylims = (0, zmax),
           xlims = (0, 1.5),   # 1 = unmixed background; <1 mixed; cap so it stays readable
           title = "Stratification profiles", legend = :bottomright)
for p in 0:floor(Int, times[end] / T_tide)
    n = argmin(abs.(times .- p * T_tide))
    plot!(plt, G[ks, n] ./ N², zg[ks]; lw = 2,
          label = @sprintf("t = %d periods", p))
end
savefig("Stokes/stratification_profiles3D.png")

# ---- 4. Mean velocity vs laminar Stokes solution ----
# Laminar solution for an oscillating free stream U₀ sin(ωt) over a wall:
#   u(z, t) = U₀ [ sin(ωt) − e^(−z/δ) sin(ωt − z/δ) ]
# If the simulated profile matches this, the flow is laminar; a fuller,
# more slab-like profile with a thin sharp wall layer indicates turbulence.
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
savefig("Stokes/velocity_vs_stokes3D.png")

@info "Saved: buoyancy_gradient_normalized3D.png, mixed_layer_depth3D.png, stratification_profile3D.png, velocity_vs_stokes3D.png in Stokes folder"