ENV["GKSwstype"] = "100"

using Oceananigans, Plots, Printf, JLD2
using Plots.PlotMeasures
using Statistics

# Import parameters
include("Parameters.jl")
include("Filename_plot.jl")

# ======================================= #
##   Turbulent Kinetic Energy (TKE) Plot ##
# ======================================= #

u_series = FieldTimeSeries(root * "Velocity.jld2", "u")
v_series = FieldTimeSeries(root * "Velocity.jld2", "v")

u_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "u_avg")
v_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "v_avg")

# Get vertical center grid points
zC = znodes(u_series.grid, Center())

# Identify time indices over the last 2 inertial periods
T_f = 2π / f₀
n_periods = 2
t_end = u_series.times[end]
t_indices = findall(t -> t >= t_end - n_periods * T_f, u_series.times)

# Calculate maximum TKE for consistent x-axis scaling
tke_max = 0.0
for n in t_indices
    global tke_max
    u = Array(interior(u_series[n], :, :, :))
    v = Array(interior(v_series[n], :, :, :))
    u_mean = Array(interior(u_avg_series[n], 1, 1, :))
    v_mean = Array(interior(v_avg_series[n], 1, 1, :))
    tke = 0.5 .* ((u .- reshape(u_mean, (1, 1, :))).^2 .+ (v .- reshape(v_mean, (1, 1, :))).^2)
    tke_max = max(tke_max, maximum(mean(tke, dims=(1, 2))))
end

@info "Creating TKE animation..."

# Create animation
anim = @animate for n in t_indices
    # Fetch interior data (Array(...) ensures GPU->CPU safety)
    u = Array(interior(u_series[n], :, :, :))
    v = Array(interior(v_series[n], :, :, :))

    u_mean = Array(interior(u_avg_series[n], 1, 1, :))
    v_mean = Array(interior(v_avg_series[n], 1, 1, :))

    # Calculate velocity fluctuations
    u_prime = u .- reshape(u_mean, (1, 1, :))
    v_prime = v .- reshape(v_mean, (1, 1, :))

    # Calculate horizontal TKE = 0.5 * (u'² + v'²) and average over (x, y)
    tke_inst = 0.5 .* (u_prime.^2 .+ v_prime.^2)

    # vec() converts the (1, 1, Nz) mean output to a 1D vector of length Nz
    tke_profile = vec(mean(tke_inst, dims=(1, 2)))

    plot(tke_profile, zC,
        linewidth = 2,
        xlabel    = "TKE [m²/s²]",
        ylabel    = "Depth z [m]",
        size      = (800, 500),
        margin    = 25px,
        legend    = false,
        xlims     = (0, tke_max * 1.1),
        title     = @sprintf("TKE Profile (r = N/f = %.1f)", r)
    )
end

@info "Plotting TKE animation..."

mkpath(save_folder * "TKE")
mp4(anim, save_folder * @sprintf("TKE/r = %.1f.mp4", r), fps=60)