using Oceananigans, Plots, Printf, JLD2, Statistics

include("Parameters.jl")
include("Filename_plot.jl")

# Load data
u_series = FieldTimeSeries(root * "Velocity.jld2", "u")
v_series = FieldTimeSeries(root * "Velocity.jld2", "v")
w_series = FieldTimeSeries(root * "Velocity.jld2", "w")
b_series = FieldTimeSeries(root * "Buoyancy.jld2", "b")

u_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "u_avg")
v_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "v_avg")
w_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "w_avg")

db_dz_series = FieldTimeSeries(root * "Avg_grad_b.jld2", "db_dz")

zC = znodes(u_series.grid, Center())
center_w = a -> 0.5 .* (a[:, :, 1:end-1] .+ a[:, :, 2:end])
center_w_profile = a -> 0.5 .* (a[1:end-1] .+ a[2:end])

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
    u = Array(interior(u_series[n], :, :, :))
    v = Array(interior(v_series[n], :, :, :))
    w = center_w(Array(interior(w_series[n], :, :, :)))
    b = Array(interior(b_series[n], :, :, :))

    u_mean = Array(interior(u_avg_series[n], 1, 1, :))
    v_mean = Array(interior(v_avg_series[n], 1, 1, :))
    w_mean = center_w_profile(Array(interior(w_avg_series[n], 1, 1, :)))

    nz = min(size(u, 3), size(b, 3), length(u_mean))

    u_prime = u[:, :, 1:nz] .- reshape(u_mean[1:nz], (1, 1, nz))
    v_prime = v[:, :, 1:nz] .- reshape(v_mean[1:nz], (1, 1, nz))
    w_prime = w[:, :, 1:nz] .- reshape(w_mean[1:nz], (1, 1, nz))
    b_prime = b[:, :, 1:nz] .- reshape(mean(b[:, :, 1:nz], dims=(1,2)), (1, 1, nz))

    wb_inst = w_prime .* b_prime
    wb_profile[1:nz] .+= vec(mean(wb_inst, dims=(1, 2)))

    tke_inst = 0.5 .* (u_prime.^2 .+ v_prime.^2 .+ w_prime.^2)
    tke_profile[1:nz] .+= vec(mean(tke_inst, dims=(1, 2)))

    db_dz_data = Array(interior(db_dz_series[n], 1, 1, :))[1:nz]
    db_dz_profile[1:nz] .+= db_dz_data
end

wb_profile ./= length(t_indices)
db_dz_profile ./= length(t_indices)
tke_profile ./= length(t_indices)

# Different interface detection methods
println("\n=== INTERFACE DETECTION METHODS ===\n")

# Method 1: Maximum ∂b/∂z
idx_1 = argmax(db_dz_profile)
@printf("Method 1 - Max ∂b/∂z: z = %.4f m (index %d)\n", zC[idx_1], idx_1)

# Method 2: Maximum |w'b'|
idx_2 = argmax(abs.(wb_profile))
@printf("Method 2 - Max |w'b'|: z = %.4f m (index %d)\n", zC[idx_2], idx_2)

# Method 3: TKE drops to 10% of peak
tke_peak = maximum(tke_profile)
idx_3_candidates = findall(tke_profile .>= 0.1 * tke_peak)
idx_3 = maximum(idx_3_candidates)
@printf("Method 3 - TKE > 10%% peak: z = %.4f m (index %d)\n", zC[idx_3], idx_3)

# Method 4: TKE drops to 1% of peak
idx_4_candidates = findall(tke_profile .>= 0.02 * tke_peak)
idx_4 = maximum(idx_4_candidates)
@printf("Method 4 - TKE > 1%% peak: z = %.4f m (index %d)\n", zC[idx_4], idx_4)

# Create comparison plot
fig = plot(layout=(2,2), size=(1000, 800))

plot!(fig[1], db_dz_profile, zC, linewidth=2, label="∂b/∂z", xlabel="Buoyancy gradient [s⁻²]", ylabel="Height z [m]")
hline!(fig[1], [zC[idx_1]], color=:red, linestyle=:dash, label="Method 1", legend=:bottomright)

plot!(fig[2], abs.(wb_profile), zC, linewidth=2, label="|w'b'|", xlabel="Buoyancy flux [m²/s³]", ylabel="Height z [m]")
hline!(fig[2], [zC[idx_2]], color=:red, linestyle=:dash, label="Method 2", legend=:bottomright)

plot!(fig[3], tke_profile, zC, linewidth=2, label="TKE", xlabel="TKE [m²/s²]", ylabel="Height z [m]")
hline!(fig[3], [zC[idx_3]], color=:red, linestyle=:dash, label="Method 3 (10%)", legend=:bottomright)
hline!(fig[3], [zC[idx_4]], color=:orange, linestyle=:dash, label="Method 4 (1%)")

plot!(fig[4], -wb_profile ./ (db_dz_profile .+ 1e-10), zC, linewidth=2, label="K_t", xlabel="K_t [m²/s]", ylabel="Height z [m]")
hline!(fig[4], [zC[idx_1]], color=:red, linestyle=:dash, label="Method 1", legend=:bottomright)
hline!(fig[4], [zC[idx_2]], color=:blue, linestyle=:dash, label="Method 2")

savefig(fig, save_folder * @sprintf("Diffusivity/r=%.1f_interface_diagnostic.png", r))

println("\n=== RECOMMENDED ===")
println("Method 3 or 4 (TKE-based) usually best for identifying turbulent layer height")
println("Method 2 (max flux) good for where mixing is strongest")