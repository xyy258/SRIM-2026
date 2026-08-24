using Oceananigans, Plots, Printf

ω  = 1.4075235e-4
N² = 1e-7
ν  = 1.109e-5
δ  = sqrt(2ν / ω)
T_tide = 2π / ω

filename = "TidalBoundaryLayer3D"

# ================================================================
# 1. Mean stratification profile:  ∂b/∂z / N²  vs  z
#    mixed layer -> 0,  background -> 1,  the kink between the two
#    is the top of the mixed layer.
# ================================================================
B_ts  = FieldTimeSeries(filename * "_profiles.jld2", "B")
zc    = znodes(B_ts)                 # cell centers, length Nz
times = B_ts.times

# Average B over the final tidal period to smooth turbulent wiggles.
last_period = findall(t -> t >= times[end] - T_tide, times)
Bmean = zeros(length(zc))
for n in last_period
    Bmean .+= vec(interior(B_ts[n]))
end
Bmean ./= length(last_period)

# Gradient by finite-differencing between adjacent centers.
# This uses only interior data (no halos / no BC faces), so no
# spurious value at the wall. Result lives at the midpoints zg.
zg = 0.5 .* (zc[1:end-1] .+ zc[2:end])          # length Nz-1
G  = diff(Bmean) ./ diff(zc)                    # ∂b/∂z at midpoints

zmax = 15.0                                       # raise if the kink is higher
ks   = findall(z -> z <= zmax, zg)

plot(G[ks] ./ N², zg[ks];
     lw = 2, label = "simulation (final-period mean)",
     xlabel = "∂b/∂z / N²", ylabel = "z (m)",
     xlims  = (0, 1.5),        # 0 = fully mixed, 1 = unmixed background
     ylims  = (0, zmax),
     title  = "Mean stratification profile", legend = :bottomright)
vline!([1]; ls = :dash, label = "background (∂b/∂z = N²)")
savefig("stratification_profile3D.png")

# Report where the profile crosses halfway (a simple mixed-layer depth).
kmix = findfirst(g -> g >= 0.5N², G)
mld  = kmix === nothing ? NaN : zg[kmix]
@info @sprintf("Mixed-layer depth (∂b/∂z crosses ½N²) ≈ %.3f m  (δ = %.3f m)", mld, δ)

# ================================================================
# 2. Bottom buoyancy b(x, t), sequential palette
# ================================================================
b_xz = FieldTimeSeries(filename * ".jld2", "b")
x    = xnodes(b_xz)
tt   = b_xz.times
Nt   = length(tt)

b_bottom = zeros(length(x), Nt)
for n in 1:Nt
    b_bottom[:, n] = interior(b_xz[n])[:, 1, 1]
end

heatmap(x, tt ./ T_tide, b_bottom';
        xlabel = "x (m)", ylabel = "t / tidal period",
        title  = "bottom buoyancy  b(x, t)",
        color  = :thermal)          # sequential: values are all one sign
savefig("bottom_buoyancy_xt3D.png")

@info "Saved: stratification_profile3D.png, bottom_buoyancy_xt3D.png"