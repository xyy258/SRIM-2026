using Oceananigans, JLD2, Plots, Printf

# Animation of the x-z slices saved by Tidal3D.jl for one case:
#   PROFILE=4 T_STRAT=10 julia --project=. Tidal3Danimation.jl sqrtRi2
#
# PROFILE and T_STRAT MUST match the run being animated: they select the output
# directory through `casetag`, and b_background(z) — the profile subtracted below
# — is rebuilt from them. Getting them wrong reads the wrong case or subtracts
# the wrong background.
#
# Panels: u (oscillating shear + bursts), then the thermal perturbation
# b' = b − b_bg(z). Ri0 carries b as a passive scalar with the same background
# gradient, so all cases show the same two panels.
#
# The plotted depth range is 0 – Lz_test, which tracks the pycnocline height T
# (case_params.jl); override with LZ_TEST=<metres>.

include(joinpath(@__DIR__, "case_params.jl"))

# Tidal3D.jl writes 200 frames per tidal period, so a 4-period run is 800 frames.
# STRIDE thins that for a quicker/lighter animation.
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
# of the background measured in Stokes thicknesses (b' = N² ζ ⇒ b'/(N²δ) = ζ/δ),
# rather than a raw value like 1e-6 that Plots renders as 0.00000.
bp_scale = N²_ref * δ

# The old limit of 2 (a 2 δ_s = 0.28 m displacement) was set for the shallow
# exponential runs and saturates immediately here: these boundary layers mix over
# metres, i.e. tens of δ_s. 20 ≈ a 2.8 m displacement. Override with BPLIM.
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
    # N²_ref δ so the colorbar shows an O(1) number instead of a value like
    # ~1e-6 that displays as 0.00000 under Plots' default tick formatting.
    bp_xz = (b_xz .- reshape(b_background.(zb), 1, :)) ./ bp_scale
    mid_plot = heatmap(xb, zb, bp_xz'; color = :balance,
                       clims = (-bplim, bplim),
                       ylims = (0, Lz_test), xlims = (0, Lx),
                       ylabel = "z", colorbar_title = "  b' / (N²_ref δ)")
    mid_title = passive_scalar ? "b' (passive scalar)" : "b' = b − b_bg(z)"


    # casetag, not case: the tag carries the profile and T, so the four T values
    # of a given √Ri are distinguishable in the frame titles.
    ttl = @sprintf("%s,  t = %.2f tidal periods", casetag, t / T_tide)
    plot(u_plot, mid_plot, layout = (2, 1), size = (1000, 700),
         title = [string("u,  ", ttl) mid_title])

    iter == iterations[end] && close(file_xz)
end

mp4(anim, joinpath(outdir, "animation_" * casetag * ".mp4"), fps = 12)

@info "Saved animation for $casetag in $outdir/"
