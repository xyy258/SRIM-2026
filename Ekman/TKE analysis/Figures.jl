# The existing Ekman figures, as a separate step.
#
# Ekman3D.jl used to end with `include("Ekman_anim.jl"); include("Ekman_plot.jl")`.
# In this folder those are opt-in (PLOTS=1) and this script is the other way to
# get them, so that:
#   * a failed animation cannot cost the simulation that produced the data, and
#   * the figures can be redrawn from data already on disk without re-running
#     anything, which is minutes rather than an hour.
#
# Ekman_plot.jl, Ekman_anim.jl and TKE.jl are copied into this folder UNCHANGED.
# They find their data through Filename_plot.jl / Filename_anim.jl, which are the
# only files that were edited — to anchor `root` and `save_folder` here instead of
# in "Ekman/3D Simulation".
#
#     julia --project=<repo root> "Ekman/TKE analysis/Figures.jl"
#
# TKE.jl is included last and is the interesting comparison: it computes TKE the
# old way, from the (:, 1, :) x–z slice with a one-plane fluctuation average,
# while MixedLayerDiffusivity.jl uses the full plane averages Moments.jl writes.
# The two should agree to within the sampling noise of a single slice; if they do
# not, something is wrong with one of them.

ENV["GKSwstype"] = "100"
using Printf

include(joinpath(@__DIR__, "Parameters.jl"))

const WHICH = get(ENV, "FIGURES", "all")

if WHICH in ("all", "anim")
    @info "Ekman_anim.jl — buoyancy, velocity and vorticity animations"
    include(joinpath(@__DIR__, "Ekman_anim.jl"))
end
if WHICH in ("all", "plot")
    @info "Ekman_plot.jl — averaged profiles"
    include(joinpath(@__DIR__, "Ekman_plot.jl"))
end
if WHICH in ("all", "tke")
    @info "TKE.jl — the slice-based TKE profile and animation, for comparison"
    include(joinpath(@__DIR__, "TKE.jl"))
end
