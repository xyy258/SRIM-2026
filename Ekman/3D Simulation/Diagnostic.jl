using Oceananigans, Plots, Printf, JLD2, Statistics

global T = 5
global r = 0.5

include("Parameters.jl")
include("Filename_plot.jl")

# Load data
w_series = FieldTimeSeries(root * "Velocity.jld2", "w")
b_series = FieldTimeSeries(root * "Buoyancy.jld2", "b")

w_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "w_avg")

db_dz_series = FieldTimeSeries(root * "Avg_grad_b.jld2", "db_dz")

zC = znodes(w_series.grid, Center())

# Time indices
T_f = 2π / f₀
n_periods = 5
t_end = w_series.times[end]
t_indices = findall(t -> t >= t_end - n_periods * T_f, w_series.times)

# Initialize
wb_profile = zeros(length(zC))
db_dz_profile = zeros(length(zC))
tke_profile = zeros(length(zC))

# Calculate profiles
for n in t_indices
    w = Array(interior(w_series[n], :, :, :))
    b = Array(interior(b_series[n], :, :, :))

    w_mean = Array(interior(w_avg_series[n], 1, 1, :))

    nz = min(size(w, 3), size(b, 3), length(w_mean))

    w_prime = w[:, :, 1:nz] .- reshape(w_mean[1:nz], (1, 1, nz))
    b_prime = b[:, :, 1:nz] .- reshape(mean(b[:, :, 1:nz], dims=(1,2)), (1, 1, nz))

    wb_inst = w_prime .* b_prime
    wb_profile[1:nz] .+= vec(mean(wb_inst, dims=(1, 2)))

    tke_inst = 0.5 .* (w_prime.^2)  # Just w contribution for now
    tke_profile[1:nz] .+= vec(mean(tke_inst, dims=(1, 2)))

    db_dz_data = Array(interior(db_dz_series[n], 1, 1, :))[1:nz]
    db_dz_profile[1:nz] .+= db_dz_data
end

wb_profile ./= length(t_indices)
db_dz_profile ./= length(t_indices)
tke_profile ./= length(t_indices)

# Different interface detection methods
println("\n=== INTERFACE DETECTION METHODS ===\n")

# Method 1: Maximum ∂b/∂z (current)
idx_1 = argmax(db_dz_profile)
@printf("Method 1 - Max ∂b/∂z: z = %.4f m (index %d)\n", zC[idx_1], idx_1)

# Method 2: Maximum |w'b'|
idx_2 = argmax(abs.(wb_profile))
@printf("Method 2 - Max |w'b'|: z = %.4f m (index %d)\n", zC[idx_2], idx_2)

# Method 3: TKE drops to 10% of peak
tke_peak = maximum(tke_profile)
idx_3_candidates = findall(tke_profile .>= 0.1 * tke_peak)
idx_3 = maximum(idx_3_candidates)  # deepest point where TKE > 10% peak
@printf("Method 3 - TKE > 10%% peak: z = %.4f m (index %d)\n", zC[idx_3], idx_3)

# Method 4: TKE drops to 5% of peak
idx_4_candidates = findall(tke_profile .>= 0.05 * tke_peak)
idx_4 = maximum(idx_4_candidates)
@printf("Method 4 - TKE > 5%% peak: z = %.4f m (index %d)\n", zC[idx_4], idx_4)

# Method 5: Richardson number Ri > 0.25 (critical value)
shear_profile = zeros(length(zC))
for i in 2:length(zC)-1
    # Estimate shear from TKE gradient (rough approximation)
    shear_profile[i] = 1e-6  # placeholder - would need velocity data
end
# This requires more data, so skip for now

# Create comparison plot
fig = plot(layout=(2,2), size=(1000, 800))

plot!(fig[1], db_dz_profile, zC, linewidth=2, label="∂b/∂z", xlabel="Buoyancy gradient [s⁻²]")
vline!(fig[1], [db_dz_profile[idx_1]], color=:red, linestyle=:dash, label="Method 1", legend=:bottomright)

plot!(fig[2], abs.(wb_profile), zC, linewidth=2, label="|w'b'|", xlabel="Buoyancy flux [m/s²]")
vline!(fig[2], [abs(wb_profile[idx_2])], color=:red, linestyle=:dash, label="Method 2", legend=:bottomright)

plot!(fig[3], tke_profile, zC, linewidth=2, label="TKE", xlabel="TKE [m²/s²]")
hline!(fig[3], [zC[idx_3]], color=:red, linestyle=:dash, label="Method 3 (10%)", legend=:bottomright)
hline!(fig[3], [zC[idx_4]], color=:orange, linestyle=:dash, label="Method 4 (5%)")

plot!(fig[4], -wb_profile ./ (db_dz_profile .+ 1e-10), zC, linewidth=2, label="K_t", xlabel="K_t [m²/s]")
vline!(fig[4], [-wb_profile[idx_1] / db_dz_profile[idx_1]], color=:red, linestyle=:dash, label="Method 1", legend=:bottomright)
vline!(fig[4], [-wb_profile[idx_2] / db_dz_profile[idx_2]], color=:blue, linestyle=:dash, label="Method 2")

savefig(fig, save_folder * @sprintf("Diffusivity/r=%.1f_interface_diagnostic.png", r))

println("\n=== RECOMMENDED ===")
println("Method 3 or 4 (TKE-based) usually best for identifying turbulent layer height")
println("Method 2 (max flux) good for where mixing is strongest")