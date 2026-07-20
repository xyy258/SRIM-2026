using Oceananigans, JLD2, Plots, Printf

# Animation of x-z slices saved by Tidal3D.jl for one case:
#   julia --project=. Tidal3Danimation.jl Ri500
# Outputs go to output_<case>/ with the case in every filename.
#
# Panels: u (oscillating shear + bursts), then buoyancy perturbation
# b' = b − N²z for stratified cases or w for Ri0 (where b ≡ 0), then dye.

include(joinpath(@__DIR__, "case_params.jl"))

zoom_height = 30.0    # set lower to zoom on the bottom
stride      = 1       # plot every `stride`-th saved frame

# Load one snapshot just to get the grid/coordinates
u_ic = FieldTimeSeries(filename * ".jld2", "u", iterations = 0)
b_ic = FieldTimeSeries(filename * ".jld2", "b", iterations = 0)

xu, ~, zu = nodes(u_ic)
xb, ~, zb = nodes(b_ic)

file_xz = jldopen(filename * ".jld2")
iterations = parse.(Int, keys(file_xz["timeseries/t"]))
iterations = iterations[1:stride:end]

# Fixed color limits across frames so colors are comparable in time
ulim  = 1.2 * U₀
wlim  = 0.2 * U₀
bplim = N² > 0 ? 2N² : 1.0     # b' scale (unused when N² = 0)

t_save   = zeros(length(iterations))


@info "Making an animation from $(length(iterations)) frames..."

anim = @animate for (i, iter) in enumerate(iterations)
    i % 100 == 0 && @info "Frame $i / $(length(iterations))"

    u_xz = file_xz["timeseries/u/$iter"][:, 1, :]
    w_xz = file_xz["timeseries/w/$iter"][:, 1, :]
    b_xz = file_xz["timeseries/b/$iter"][:, 1, :]
    c_xz = file_xz["timeseries/c/$iter"][:, 1, :]
    t    = file_xz["timeseries/t/$iter"]

    t_save[i] = t

    u_plot = heatmap(xu, zu, u_xz'; color = :balance, clims = (-ulim, ulim),
                     ylims = (0, zoom_height), xlims = (0, Lx),
                     ylabel = "z")

    if N² > 0
        # Buoyancy perturbation: subtract the background ramp N²z
        bp_xz = b_xz .- reshape(N² .* zb, 1, :)
        mid_plot = heatmap(xb, zb, bp_xz'; color = :balance,
                           clims = (-bplim, bplim),
                           ylims = (0, zoom_height), xlims = (0, Lx),
                           ylabel = "z")
        mid_title = "b' = b − N²z"
    else
        mid_plot = heatmap(xu, zb, w_xz[:, 1:length(zb)]'; color = :balance,
                           clims = (-wlim, wlim),
                           ylims = (0, zoom_height), xlims = (0, Lx),
                           ylabel = "z")
        mid_title = "w"
    end

    c_plot = heatmap(xu, zb, c_xz'; color = :thermal, clims = (0, 1),
                     ylims = (0, zoom_height), xlims = (0, Lx),
                     xlabel = "x", ylabel = "z")

    ttl = @sprintf("%s,  t = %.2f tidal periods", case, t / T_tide)
    plot(u_plot, mid_plot, c_plot, layout = (3, 1), size = (1000, 900),
         title = [string("u,  ", ttl) mid_title "dye c"])

    iter == iterations[end] && close(file_xz)
end

mp4(anim, joinpath(outdir, "animation_" * case * ".mp4"), fps = 12)

@info "Saved animation for $case in $outdir/"
