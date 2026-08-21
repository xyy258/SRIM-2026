using Oceananigans, JLD2, Plots, Printf

# Animation of x-z slices saved by Tidal3D.jl for one case:
#   julia --project=. Tidal3Danimation.jl Ri500
# Outputs go to output_<case>/ with the case in every filename.
#
# Panels: u (oscillating shear + bursts), then the thermal perturbation
# b' = b − N²_ref z, then dye. Ri0 now carries b as a passive scalar with the
# same background gradient, so all three cases show the same three panels.

include(joinpath(@__DIR__, "case_params.jl"))

stride      = 1       # plot every `stride`-th saved frame

# Load one snapshot just to get the grid/coordinates
u_ic = FieldTimeSeries(filename * ".jld2", "u", iterations = 0)
b_ic = FieldTimeSeries(filename * ".jld2", "b", iterations = 0)

xu, ~, zu = nodes(u_ic)
xb, ~, zb = nodes(b_ic)

file_xz = jldopen(filename * ".jld2")
iterations = sort(parse.(Int, keys(file_xz["timeseries/t"])))
iterations = iterations[1:stride:end]

# Fixed color limits across frames so colors are comparable in time
ulim  = 1.2 * U₀
bp_scale = N²_ref * δ           # normalize b' by this so the colorbar isn't a
                                # physically tiny number (~1e-6) that rounds to
                                # 0.00000 under Plots' default tick formatting
bplim = 2                      # b' scale: a 2 δ_s displacement of the background

t_save   = zeros(length(iterations))


@info "Making an animation from $(length(iterations)) frames..."

anim = @animate for (i, iter) in enumerate(iterations)
    i % 100 == 0 && @info "Frame $i / $(length(iterations))"

    u_xz = file_xz["timeseries/u/$iter"][:, 1, :]
    b_xz = file_xz["timeseries/b/$iter"][:, 1, :]
    c_xz = file_xz["timeseries/c/$iter"][:, 1, :]
    t    = file_xz["timeseries/t/$iter"]

    t_save[i] = t

    u_plot = heatmap(xu, zu, u_xz'; color = :balance, clims = (-ulim, ulim),
                     ylims = (0, Lz), xlims = (0, Lx),
                     ylabel = "z")

    # Thermal perturbation: subtract the background profile, then normalize by
    # N²_ref δ so the colorbar shows an O(1) number instead of a value like
    # ~1e-6 that displays as 0.00000 under Plots' default tick formatting.
    # OLD (uniform background): bp_xz = b_xz .- reshape(N²_ref .* zb, 1, :)
    bp_xz = (b_xz .- reshape(b_background.(zb), 1, :)) ./ bp_scale
    mid_plot = heatmap(xb, zb, bp_xz'; color = :balance,
                       clims = (-bplim, bplim),
                       ylims = (0, Lz), xlims = (0, Lx),
                       ylabel = "z", colorbar_title = "  b' / (N²_ref δ)")
    mid_title = passive_scalar ? "b' (passive scalar)" : "b' = b − b_bg(z)"

    c_plot = heatmap(xu, zb, c_xz'; color = :thermal, clims = (0, 1),
                     ylims = (0, Lz), xlims = (0, Lx),
                     xlabel = "x", ylabel = "z")

    ttl = @sprintf("%s,  t = %.2f tidal periods", case, t / T_tide)
    plot(u_plot, mid_plot, c_plot, layout = (3, 1), size = (1000, 900),
         title = [string("u,  ", ttl) mid_title "dye c"])

    iter == iterations[end] && close(file_xz)
end

mp4(anim, joinpath(outdir, "animation_" * case * ".mp4"), fps = 12)

@info "Saved animation for $case in $outdir/"
