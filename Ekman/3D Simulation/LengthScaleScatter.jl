ENV["GKSwstype"] = "100"

using Oceananigans, Plots, Printf, JLD2, Statistics
using Plots.PlotMeasures

# Define parameter ranges for sweep
ratios   = [0.5, 1, 2, 5, 10, 25, 50]
values   = [5, 10, 15, 20, 30, 40, 50]
profiles = [4]

# --- Reference Style Color Ramp ---
const RAMP = [(log10(0.5),  ( 27,  78, 143)),
              (log10(1.0),  ( 46, 139,  87)),
              (log10(2.0),  (200, 150,  30)),
              (log10(5.0),  (180,  80,  44)),
              (log10(10.0), ( 75,  16,  96))]

function ramp_colour(s)
    x = clamp(log10(s), RAMP[1][1], RAMP[end][1])
    for i in 1:length(RAMP)-1
        (x0, c0), (x1, c1) = RAMP[i], RAMP[i+1]
        x <= x1 || continue
        f = x1 == x0 ? 0.0 : (x - x0) / (x1 - x0)
        chan(k) = clamp(round(Int, c0[k] + f * (c1[k] - c0[k])), 0, 255)
        return "#" * join(string(chan(k), base = 16, pad = 2) for k in 1:3)
    end
    return "#000000"
end

med(v) = median(filter(isfinite, v))

# --- Sampling Parameters ---
const avg_len = 0.1         # Vertical physical window centered at interface peak [m]
const t_step  = 4

# ========================================================================== #
## Interfacial Length Scale Scatter Plot Across Sampled Timesteps (Log-Log) ##
# ========================================================================== #

for p in profiles
    global profile = p
    for value in values
        global T = value
        @info @sprintf("Generating l_κ vs l_N plot for T = %d...", T)

        plt = plot(size = (850, 550), margin = 25px)

        case_medians_x = Float64[]
        case_medians_y = Float64[]
        all_l_N        = Float64[]
        all_l_kappa    = Float64[]

        for (idx, ratio) in enumerate(ratios)
            global r = ratio

            include("Parameters.jl")
            include("Filename_plot.jl")

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

            # Downsample timesteps over the last 4 inertial periods
            T_f = 2π / f₀
            n_periods = 4
            t_indices = findall(t -> t >= u_series.times[end] - n_periods * T_f, u_series.times)[1:t_step:end]

            l_N_time = Float64[]
            l_kappa_time = Float64[]

            for n in t_indices
                u = Array(interior(u_series[n], :, :, :))
                v = Array(interior(v_series[n], :, :, :))
                w = center_w(Array(interior(w_series[n], :, :, :)))
                b = Array(interior(b_series[n], :, :, :))

                u_mean = Array(interior(u_avg_series[n], 1, 1, :))
                v_mean = Array(interior(v_avg_series[n], 1, 1, :))
                w_mean = center_w_profile(Array(interior(w_avg_series[n], 1, 1, :)))

                nz = min(size(w, 3), size(b, 3), length(w_mean))

                u_prime = u[:, :, 1:nz] .- reshape(u_mean[1:nz], (1, 1, nz))
                v_prime = v[:, :, 1:nz] .- reshape(v_mean[1:nz], (1, 1, nz))
                w_prime = w[:, :, 1:nz] .- reshape(w_mean[1:nz], (1, 1, nz))
                b_prime = b[:, :, 1:nz] .- reshape(mean(b[:, :, 1:nz], dims=(1,2)), (1, 1, nz))

                tke_inst   = vec(mean(0.5 .* (u_prime.^2 .+ v_prime.^2 .+ w_prime.^2), dims=(1, 2)))
                wb_inst    = vec(mean(w_prime .* b_prime, dims=(1, 2)))
                db_dz_inst = Array(interior(db_dz_series[n], 1, 1, :))[1:nz]

                # Identify interface peak depth and sampling window
                idx_peak = argmax(db_dz_inst)
                z_int    = zC[idx_peak]
                rng      = findall(abs.(zC[1:nz] .- z_int) .<= avg_len / 2)
                isempty(rng) && (rng = [idx_peak])

                tke_int = mean(tke_inst[rng])
                K_t_int = -mean(wb_inst[rng]) / (mean(db_dz_inst[rng]) + 1e-10)

                if K_t_int > 0 && tke_int > 1e-8
                    push!(l_N_time, sqrt(max(tke_int, 0)) / N)
                    push!(l_kappa_time, K_t_int / sqrt(max(tke_int, 1e-12)))
                end
            end

            @info @sprintf("Plotting %d timesteps for r = %.1f...", length(l_N_time), r)

            scatter!(plt, l_N_time, l_kappa_time,
                color             = ramp_colour(r),
                markersize        = 2.5,
                markerstrokewidth = 0,
                markeralpha       = 0.45,
                label             = @sprintf("r = %.1f", r)
            )

            if !isempty(l_N_time)
                push!(case_medians_x, med(l_N_time))
                push!(case_medians_y, med(l_kappa_time))
                append!(all_l_N, l_N_time)
                append!(all_l_kappa, l_kappa_time)
            end
        end

        # --- Saturating Exponential Fit & Reference Lines ---
        if length(case_medians_x) >= 2
            lo, hi = minimum(all_l_N), maximum(all_l_N)
            plot!(plt, [lo, hi], [lo, hi], color = :black, linewidth = 1.2, linestyle = :dash, label = "l_κ = l_N  (1:1)")

            # Grid Search Fit minimizing log least-squares on per-case medians
            best_sse = Inf
            L_fit, x0_fit = 0.5, 1.0
            for L in range(0.3 * maximum(case_medians_y), 1.8 * maximum(case_medians_y), length=150)
                for x0 in range(0.1 * minimum(case_medians_x), 2.5 * maximum(case_medians_x), length=150)
                    pred = L .* (1.0 .- exp.(-case_medians_x ./ x0))
                    all(pred .> 0) || continue
                    sse = sum((log.(case_medians_y) .- log.(pred)).^2)
                    if sse < best_sse
                        best_sse = sse
                        L_fit, x0_fit = L, x0
                    end
                end
            end

            xf = exp10.(range(log10(lo), log10(hi), length=300))
            yf = L_fit .* (1.0 .- exp.(-xf ./ x0_fit))
            plot!(plt, xf, yf, color = :black, linewidth = 2.5,
                  label = @sprintf("l_κ = L_∞(1 − e^{-l_N/x_0}), L_∞ = %.2f m, x_0 = %.2f m", L_fit, x0_fit))

            scatter!(plt, case_medians_x, case_medians_y,
                markersize        = 6,
                markerstrokewidth = 1.5,
                markercolor       = :white,
                markerstrokecolor = :black,
                label             = "case medians (fitted)"
            )
            hline!(plt, [L_fit], color = :black, linewidth = 1.0, linestyle = :dot, label = @sprintf("plateau L_∞ = %.2f m", L_fit))
        end

        # --- Dynamic Viewport Clipping (0.5th to 99.9th percentiles) ---
        if !isempty(all_l_kappa)
            sorted_y = sort(all_l_kappa)
            ylo = sorted_y[max(1, round(Int, 0.005 * length(sorted_y)))] / 1.5
            yhi = sorted_y[round(Int, 0.999 * length(sorted_y))] * 1.5
            plot!(plt, ylims = (ylo, yhi))
        end

        plot!(plt,
            xscale    = :log10,
            yscale    = :log10,
            xlabel    = "l_N = √TKE / N   (buoyancy scale, m)",
            ylabel    = "l_κ = κₜ / √TKE   (m)",
            minorgrid = true,
            legend    = :bottomright,
            title     = @sprintf("l_κ vs l_N (T = %d)", T)
        )

        mkpath(save_folder)
        savefig(plt, "l_k_l_N.png")
    end
end