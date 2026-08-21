using Oceananigans, JLD2, Plots, Printf

# Animation of the x-z slices saved by Tidal3D.jl for one case:
#   PROFILE=4 T_STRAT=10 julia --project=. Tidal3Danimation.jl sqrtRi2
#
# PROFILE and T_STRAT must match the run being animated: they choose the output
# folder and rebuild the background profile subtracted below.
#
# Two panels: u, showing the oscillating shear and the turbulent bursts, and the
# thermal perturbation b' = b − b_bg(z). At Ri = 0 the buoyancy is a passive
# scalar with the same background gradient, so every case shows both panels.
#
# The plotted depth range is 0 to Lz_test, which follows the pycnocline height T
# (case_params.jl). Override it with LZ_TEST=<metres>.

include(joinpath(@__DIR__, "case_params.jl"))

# Tidal3D.jl writes 200 frames per tidal period, so a 4-period run gives 800.
# STRIDE thins them out for a quicker animation.
stride = parse(Int, get(ENV, "STRIDE", "1"))

# Load one snapshot just to get the grid/coordinates
u_ic = FieldTimeSeries(filename * ".jld2", "u", iterations = 0)
b_ic = FieldTimeSeries(filename * ".jld2", "b", iterations = 0)

xu, ~, zu = nodes(u_ic)
xb, ~, zb = nodes(b_ic)

file_xz = jldopen(filename * ".jld2")
iterations = sort(parse.(Int, keys(file_xz["timeseries/t"])))
iterations = iterations[1:stride:end]

# Fixed color limits across frames so colors are comparable in time.
ulim  = 1.2 * U₀

# b' is normalized by N²_ref·δ, so the colorbar reads as a vertical displacement
# of the background in Stokes thicknesses rather than as a raw value like 1e-6,
# which Plots would render as 0.00000.
bp_scale = N²_ref * δ

# The limit corresponds to a displacement of about 2.8 m, since these boundary
# layers mix over metres. Override it with BPLIM.
bplim = parse(Float64, get(ENV, "BPLIM", "20"))

t_save   = zeros(length(iterations))


@info "Making an animation from $(length(iterations)) frames..."

anim = @animate for (i, iter) in enumerate(iterations)
    i % 100 == 0 && @info "Frame $i / $(length(iterations))"

    u_xz = file_xz["timeseries/u/$iter"][:, 1, :]
    b_xz = file_xz["timeseries/b/$iter"][:, 1, :]
    t    = file_xz["timeseries/t/$iter"]

    t_save[i] = t

    u_plot = heatmap(xu, zu, u_xz'; color = :balance, clims = (-ulim, ulim),
                     ylims = (0, Lz_test), xlims = (0, Lx),
                     ylabel = "z")

    # Thermal perturbation: subtract the background profile, then normalize by
    # N²_ref δ so the colorbar shows a number of order 1.
    bp_xz = (b_xz .- reshape(b_background.(zb), 1, :)) ./ bp_scale
    mid_plot = heatmap(xb, zb, bp_xz'; color = :balance,
                       clims = (-bplim, bplim),
                       ylims = (0, Lz_test), xlims = (0, Lx),
                       ylabel = "z", colorbar_title = "  b' / (N²_ref δ)")
    mid_title = passive_scalar ? "b' (passive scalar)" : "b' = b − b_bg(z)"


    # casetag rather than case, so the frame titles carry the profile and T and
    # the different T values of one √Ri can be told apart.
    ttl = @sprintf("%s,  t = %.2f tidal periods", casetag, t / T_tide)
    plot(u_plot, mid_plot, layout = (2, 1), size = (1000, 700),
         title = [string("u,  ", ttl) mid_title])

    iter == iterations[end] && close(file_xz)
end

mp4(anim, joinpath(outdir, "animation_" * casetag * ".mp4"), fps = 12)

@info "Saved animation for $casetag in $outdir/"
