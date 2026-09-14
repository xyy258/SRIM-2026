ENV["GKSwstype"] = "100"

using Oceananigans, Plots, Printf, JLD2
using Plots.PlotMeasures
using Statistics

# High-DPI plot formatting with full Unicode glyph support
default(dpi = 600, fontfamily = "DejaVu Sans")

# Import parameters
include("Parameters.jl")
include("Filename_plot.jl")

# ======================================= #
##  Turbulent Kinetic Energy (TKE) Plot  ##
# ======================================= #

u_series = FieldTimeSeries(root * "Velocity.jld2", "u")
v_series = FieldTimeSeries(root * "Velocity.jld2", "v")
w_series = FieldTimeSeries(root * "Velocity.jld2", "w")

u_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "u_avg")
v_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "v_avg")
w_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "w_avg")

# Get vertical center grid points
zC = znodes(u_series.grid, Center())
center_w = a -> 0.5 .* (a[:, :, 1:end-1] .+ a[:, :, 2:end])
center_w_profile = a -> 0.5 .* (a[1:end-1] .+ a[2:end])

# Identify time indices over the last 5 inertial periods
T_f = 2π / f₀
n_periods = 5
t_end = u_series.times[end]
t_indices = findall(t -> t >= t_end - n_periods * T_f, u_series.times)

# Pre-allocate matrix to store vertical profiles: (z_levels x time_steps)
tke_profiles = Matrix{Float64}(undef, length(zC), length(t_indices))

# Compute TKE profiles across selected timesteps
for (i, n) in enumerate(t_indices)
    u = Array(interior(u_series[n], :, :, :))
    v = Array(interior(v_series[n], :, :, :))
    w = center_w(Array(interior(w_series[n], :, :, :)))

    u_mean = Array(interior(u_avg_series[n], 1, 1, :))
    v_mean = Array(interior(v_avg_series[n], 1, 1, :))
    w_mean = center_w_profile(Array(interior(w_avg_series[n], 1, 1, :)))

    # Calculate velocity fluctuations
    u_prime = u .- reshape(u_mean, (1, 1, :))
    v_prime = v .- reshape(v_mean, (1, 1, :))
    w_prime = w .- reshape(w_mean, (1, 1, :))

    # Calculate TKE = 0.5 * (u'² + v'² + w'²)
    tke_inst = 0.5 .* (u_prime.^2 .+ v_prime.^2 .+ w_prime.^2)
    tke_profiles[:, i] = vec(mean(tke_inst, dims=(1, 2)))
end

# Extract time-averaged profile and overall maximum for axis scaling
tke_profile_avg = vec(mean(tke_profiles, dims=2))
tke_max = maximum(tke_profiles)

@info "Making time-averaged TKE plot..."

plot(tke_profile_avg, zC,
    linewidth = 2,
    xlabel    = "TKE [m²/s²]",
    ylabel    = "Depth z [m]",
    xlims     = (0, tke_max * 1.05),
    ylims     = (0, Lz),
    size      = (800, 500),
    margin    = 25px,
    legend    = false,
    title     = @sprintf("Time-Averaged TKE Profile (r = N/f = %.1f)", r)
)

mkpath(save_folder * "TKE")
savefig(save_folder * @sprintf("TKE/r = %.1f.png", r))

# --- Animation ---
@info "Creating TKE animation..."

anim = @animate for (i, n) in enumerate(t_indices)
    t = u_series.times[n]

    plot(tke_profiles[:, i], zC,
        linewidth = 2,
        xlabel    = "TKE [m²/s²]",
        ylabel    = "Depth z [m]",
        size      = (800, 500),
        margin    = 25px,
        legend    = false,
        xlims     = (0, tke_max * 1.1),
        ylims     = (0, Lz),
        title     = @sprintf("TKE Profile (r = %.1f, t = %.1f s)", r, t)
    )
end

@info "Saving TKE animation..."

mp4(anim, save_folder * @sprintf("TKE/r = %.1f.mp4", r), fps=60)