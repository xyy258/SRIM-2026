using Oceananigans, Plots, Printf, JLD2, Statistics

include("Parameters.jl")
include("Filename_plot.jl")

w_series = FieldTimeSeries(root * "Velocity.jld2", "w")
b_series = FieldTimeSeries(root * "Buoyancy.jld2", "b")
w_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "w_avg")
db_dz_series = FieldTimeSeries(root * "Avg_grad_b.jld2", "db_dz")

zC = znodes(w_series.grid, Center())
t_end = w_series.times[end]
t_indices = findall(t -> t >= t_end - 2π/f₀ * 2, w_series.times)

local wb = zeros(length(zC))
local db_dz = zeros(length(zC))
local tke = zeros(length(zC))

for n in t_indices
    w = Array(interior(w_series[n], :, :, :))
    b = Array(interior(b_series[n], :, :, :))
    w_mean = Array(interior(w_avg_series[n], 1, 1, :))

    nz = min(size(w, 3), size(b, 3), length(w_mean))

    w_prime = w[:, :, 1:nz] .- reshape(w_mean[1:nz], (1, 1, nz))
    b_prime = b[:, :, 1:nz] .- reshape(mean(b[:, :, 1:nz], dims=(1,2)), (1, 1, nz))

    wb[1:nz] .+= vec(mean(w_prime .* b_prime, dims=(1, 2)))
    tke[1:nz] .+= vec(mean(0.5 .* w_prime.^2, dims=(1, 2)))
    db_dz[1:nz] .+= Array(interior(db_dz_series[n], 1, 1, :))[1:nz]
end

wb ./= length(t_indices)
db_dz ./= length(t_indices)
tke ./= length(t_indices)

# Smooth gradient and find interface (max ∂b/∂z)
db_dz_smooth = [mean(db_dz[max(1,i-3):min(length(zC),i+3)]) for i in eachindex(zC)]
idx = argmax(db_dz_smooth)
z = zC[idx]

# Average ±2 points around interface
r_idx = max(1, idx-2):min(length(zC), idx+2)
wb_int = mean(wb[r_idx])
db_dz_int = mean(db_dz[r_idx])
K_t = -wb_int / db_dz_int

@printf("z = %.4f m, ∂b/∂z = %.2e, w'b' = %.2e, K_t = %.2e\n", z, db_dz_int, wb_int, K_t)

p1 = plot(db_dz_smooth, zC, linewidth=2, xlabel="∂b/∂z", ylabel="z")
hline!([z], color=:red, linestyle=:dash)
p2 = plot(wb, zC, linewidth=2, xlabel="w'b'", ylabel="z")
hline!([z], color=:red, linestyle=:dash)
p3 = plot(tke, zC, linewidth=2, xlabel="TKE", ylabel="z")
hline!([z], color=:red, linestyle=:dash)

mkpath(save_folder * "Diffusivity")
savefig(plot(p1, p2, p3, layout=(1,3), size=(1000,400)),
    save_folder * @sprintf("Diffusivity/r=%.1f_interface.png", r))