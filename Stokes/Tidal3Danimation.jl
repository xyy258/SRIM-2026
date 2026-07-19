using Oceananigans, JLD2, Plots, Printf

# Animation of x-z slices saved by TidalBoundaryLayer.jl
# Changes from the original:
#   - plots buoyancy *perturbation* b' = b − N²z (the background ramp otherwise
#     hides the overturns you're looking for)
#   - fixed color limits across all frames, so colors are comparable in time
#   - adds a u panel to see the oscillating shear and turbulent bursts
#   - optional zoom on the bottom few metres, where all the action is
#   - colorbars scaled so labels aren't all "0.000"

# ---- parameters (must match the simulation) ----
ω  = 1.4075235e-4
N² = 1e-7
U₀ = 0.05
T_tide = 2π / ω

filename = "TidalBoundaryLayer3D"

zoom_height = 30.0    # set to Lz to see the whole domain
stride      = 1      # plot every `stride`-th saved frame (increase to speed up)

# Load one snapshot just to get the grid/coordinates
u_ic = FieldTimeSeries(filename * ".jld2", "u", iterations = 0)
b_ic = FieldTimeSeries(filename * ".jld2", "b", iterations = 0)
c_ic = FieldTimeSeries(filename * ".jld2", "c", iterations = 0)

xu, ~, zu = nodes(u_ic)
xb, ~, zb = nodes(b_ic)
xc, ~, zc = nodes(c_ic)

Lx = maximum(xu)

file_xz = jldopen(filename * ".jld2")
iterations = parse.(Int, keys(file_xz["timeseries/t"]))
iterations = iterations[1:stride:end]

# Fixed color limits
ulim  = 1.2 * U₀
bplim = 2N²           # b' scale: N² × (a couple of metres of displacement)

t_save   = zeros(length(iterations))
b_bottom = zeros(length(xb), length(iterations))

@info "Making an animation from $(length(iterations)) frames..."

anim = @animate for (i, iter) in enumerate(iterations)
    i % 100 == 0 && @info "Frame $i / $(length(iterations))"

    u_xz = file_xz["timeseries/u/$iter"][:, 1, :]
    b_xz = file_xz["timeseries/b/$iter"][:, 1, :]
    c_xz = file_xz["timeseries/c/$iter"][:, 1, :]
    t    = file_xz["timeseries/t/$iter"]

    t_save[i] = t
    b_bottom[:, i] = b_xz[:, 1]          # buoyancy in the lowest grid cell

    # Buoyancy perturbation: subtract the background ramp N²z
    bp_xz = b_xz .- reshape(N² .* zb, 1, :)

    u_plot = heatmap(xu, zu, u_xz'; color = :balance, clims = (-ulim, ulim),
                     ylims = (0, zoom_height), xlims = (0, Lx),
                     ylabel = "z")
    b_plot = heatmap(xb, zb, bp_xz'; color = :balance, clims = (-bplim, bplim),
                     ylims = (0, zoom_height), xlims = (0, Lx),
                     ylabel = "z")
    c_plot = heatmap(xc, zc, c_xz'; color = :thermal, clims = (0, 1),
                     ylims = (0, zoom_height), xlims = (0, Lx),
                     xlabel = "x", ylabel = "z")

    ttl = @sprintf("t = %.2f tidal periods", t / T_tide)
    plot(u_plot, b_plot, c_plot, layout = (3, 1), size = (1000, 900),
         title = [string("u, ", ttl) "b' = b − N²z" "dye c"])

    iter == iterations[end] && close(file_xz)
end

mp4(anim, "TidalBoundaryLayer3D.mp4", fps = 12)

# ---- Bottom buoyancy Hovmöller diagram ----
# Scale by 1e6 so the colorbar labels are readable.
heatmap(xb, t_save ./ T_tide, 1e6 .* b_bottom';
        xlabel = "x", ylabel = "t / tidal period",
        title = "buoyancy in lowest grid cell (×10⁻⁶)",
        color = :thermal)
savefig("bottom_buoyancy3D.png")