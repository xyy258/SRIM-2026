using Oceananigans, Plots, Printf, JLD2, Statistics
using Plots.PlotMeasures

include("Parameters.jl")
include("Filename_plot.jl")

# Load data
w_series = FieldTimeSeries(root * "Velocity.jld2", "w")
b_series = FieldTimeSeries(root * "Buoyancy.jld2", "b")

w_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "w_avg")

db_dz_series = FieldTimeSeries(root * "Avg_grad_b.jld2", "db_dz")

zC = znodes(u_series.grid, Center())
zF = znodes(u_series.grid, Face())

# Time indices over last 2 inertial periods
T_f = 2π / f₀
n_periods = 5
t_end = u_series.times[end]
t_indices = findall(t -> t >= t_end - n_periods * T_f, u_series.times)

# Initialize
wb_profile = zeros(length(zC))
db_dz_profile = zeros(length(zC))

# Calculate w'b' and ∂b/∂z profiles
for n in t_indices
    w = Array(interior(w_series[n], :, :, :))
    b = Array(interior(b_series[n], :, :, :))

    w_mean = Array(interior(w_avg_series[n], 1, 1, :))

    # Get matching z dimension
    nz = min(size(w, 3), size(b, 3), length(w_mean))

    # Fluctuations
    w_prime = w[:, :, 1:nz] .- reshape(w_mean[1:nz], (1, 1, nz))
    b_prime = b[:, :, 1:nz] .- reshape(mean(b[:, :, 1:nz], dims=(1,2)), (1, 1, nz))

    # Buoyancy flux
    wb_inst = w_prime .* b_prime
    wb_profile[1:nz] .+= vec(mean(wb_inst, dims=(1, 2)))

    # Buoyancy gradient
    db_dz_data = Array(interior(db_dz_series[n], 1, 1, :))[1:nz]
    db_dz_profile[1:nz] .+= db_dz_data
end

# Average over time
wb_profile ./= length(t_indices)
db_dz_profile ./= length(t_indices)

# Find interface: maximum ∂b/∂z
idx_interface = argmax(db_dz_profile)
z_interface = zC[idx_interface]

# Average w'b' over small region around interface (±2 grid points)
dz_avg = 2  # number of grid points to average
idx_range = max(1, idx_interface - dz_avg):min(length(zC), idx_interface + dz_avg)
wb_interface = mean(wb_profile[idx_range])
db_dz_interface = mean(db_dz_profile[idx_range])

# Turbulent diffusivity: K_t = -w'b' / (∂b/∂z)
K_t_interface = -wb_interface / db_dz_interface

@info "Turbulent Diffusivity at Interface"
@printf("Interface location: z = %.4f m\n", z_interface)
@printf("∂b/∂z at interface: %.6e s⁻²\n", db_dz_interface)
@printf("w'b' at interface: %.6e m/s²\n", wb_interface)
@printf("K_t at interface:  %.6e m²/s\n\n", K_t_interface)

# Plot profiles
p1 = plot(db_dz_profile, zC,
    linewidth = 2,
    xlabel    = "∂b/∂z [s⁻²]",
    ylabel    = "z [m]",
    label     = "",
    margin    = 10px)
vline!([db_dz_interface], color=:red, linestyle=:dash, label="interface")

p2 = plot(wb_profile, zC,
    linewidth = 2,
    xlabel    = "w'b' [m/s²]",
    ylabel    = "z [m]",
    label     = "",
    margin    = 10px)
vline!([wb_interface], color=:red, linestyle=:dash, label="interface")

p3 = plot(-wb_profile ./ (db_dz_profile .+ 1e-10), zC,
    linewidth = 2,
    xlabel    = "K_t [m²/s]",
    ylabel    = "z [m]",
    label     = "",
    margin    = 10px)
vline!([K_t_interface], color=:red, linestyle=:dash, label="interface")

plot_all = plot(p1, p2, p3, layout=(1,3), size=(1000, 400), margin=10Plots.px)

mkpath(save_folder * "Diffusivity")
savefig(plot_all, save_folder * @sprintf("Diffusivity/r=%.1f_interface.png", r))

# Save summary
open(save_folder * @sprintf("Diffusivity/r=%.1f_summary.txt", r), "w") do f
    write(f, @sprintf("K_t at interface (z=%.4f m): %.6e m²/s\n", z_interface, K_t_interface))
    write(f, @sprintf("∂b/∂z at interface: %.6e s⁻²\n", db_dz_interface))
    write(f, @sprintf("w'b' at interface: %.6e m/s²\n", wb_interface))
    write(f, @sprintf("r = N/f: %.1f\n", r))
end

@info "Results saved to $save_folder/Diffusivity/"