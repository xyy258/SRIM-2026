ENV["GKSwstype"] = "100"

using Oceananigans, Plots, Printf, JLD2
using Plots.PlotMeasures
using Statistics

ratios   = [0.5, 1, 2, 5]
values   = [5, 10, 15, 20, 30, 40, 50]
profiles = [4]

# --- Sampling Parameters ---
const avg_len = 0.5             # Vertical physical window centered at interface peak [m]
const t_step  = 5

# ======================================================================== #
## Interfacial Length Scale Scatter Plot Across Sampled Timesteps (Log-Log) ##
# ======================================================================== #

for p in profiles
    global profile = p
    for value in values
        plt  = plot(size = (800, 500))
        plt2 = plot(size = (800, 500))   # l_κ vs mixed layer height

        global T = value
        @info("Plot for T=$T...")

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

            # Downsampled snapshots over the last 5 inertial periods (T_f)
            T_f = 2π / f₀
            n_periods = 4
            t_indices = findall(t -> t >= u_series.times[end] - 4 * T_f, u_series.times)[1:t_step:end]

            l_N_time = Float64[]
            l_kappa_time = Float64[]
            h_time = Float64[]   # mixed layer height (interface depth) per snapshot

            for n in t_indices
                u = Array(interior(u_series[n], :, :, :))
                v = Array(interior(v_series[n], :, :, :))
                w = center_w(Array(interior(w_series[n], :, :, :)))
                b = Array(interior(b_series[n], :, :, :))

                u_mean = Array(interior(u_avg_series[n], 1, 1, :))
                v_mean = Array(interior(v_avg_series[n], 1, 1, :))
                w_mean = center_w_profile(Array(interior(w_avg_series[n], 1, 1, :)))

                nz = min(size(w, 3), size(b, 3), length(w_mean))

                # Horizontal anomaly fluctuations
                u_prime = u[:, :, 1:nz] .- reshape(u_mean[1:nz], (1, 1, nz))
                v_prime = v[:, :, 1:nz] .- reshape(v_mean[1:nz], (1, 1, nz))
                w_prime = w[:, :, 1:nz] .- reshape(w_mean[1:nz], (1, 1, nz))
                b_prime = b[:, :, 1:nz] .- reshape(mean(b[:, :, 1:nz], dims=(1,2)), (1, 1, nz))

                tke_inst   = vec(mean(0.5 .* (u_prime.^2 .+ v_prime.^2 .+ w_prime.^2), dims=(1, 2)))
                wb_inst    = vec(mean(w_prime .* b_prime, dims=(1, 2)))
                db_dz_inst = Array(interior(db_dz_series[n], 1, 1, :))[1:nz]

                # Identify interface peak depth and sampling region
                idx_peak = argmax(db_dz_inst)
                z_int    = zC[idx_peak]
                rng      = findall(abs.(zC[1:nz] .- z_int) .<= avg_len / 2)
                isempty(rng) && (rng = [idx_peak])

                tke_int = mean(tke_inst[rng])
                K_t_int = -mean(wb_inst[rng]) / (mean(db_dz_inst[rng]) + 1e-10)

                # Filter valid mixing events for log-scale plotting
                if K_t_int > 0 && tke_int > 1e-8
                    push!(l_N_time, sqrt(max(tke_int, 0)) / N)
                    push!(l_kappa_time, K_t_int / sqrt(max(tke_int, 1e-12)))
                    push!(h_time, z_int)
                end
            end

            @info @sprintf("Plotting %d timesteps for r = %.1f...", length(l_N_time), r)

            scatter!(plt, l_N_time, l_kappa_time,
                color             = idx,
                markersize        = 3.0,
                markerstrokewidth = 0,
                markeralpha       = 0.6,
                label             = @sprintf("r = %.1f", r)
            )

            # --- Line of Best Fit with Correlation Coefficient ---
            if length(l_N_time) >= 2
                x_log, y_log = log10.(l_N_time), log10.(l_kappa_time)
                m, c = [x_log ones(length(x_log))] \ y_log
                R = cor(x_log, y_log)

                x_fit = range(minimum(l_N_time), maximum(l_N_time), length=100)
                y_fit = 10 .^ (m .* log10.(x_fit) .+ c)

                fit_lbl = isnan(R) ? @sprintf("l_κ = %.3g l_N^{%.3g}", 10^c, m) :
                                     @sprintf("l_κ = %.3g l_N^{%.3g} (R = %.2f)", 10^c, m, R)

                plot!(plt, x_fit, y_fit,
                    linestyle = :dash,
                    linewidth = 1.5,
                    color     = idx,
                    label     = fit_lbl
                )
            end

            # --- l_κ vs mixed layer height (tests K_t ~ h*sqrt(TKE)) ---
            scatter!(plt2, h_time, l_kappa_time,
                color             = idx,
                markersize        = 3.0,
                markerstrokewidth = 0,
                markeralpha       = 0.6,
                label             = @sprintf("r = %.1f", r)
            )

            if length(h_time) >= 2
                x_log, y_log = log10.(h_time), log10.(l_kappa_time)
                m, c = [x_log ones(length(x_log))] \ y_log
                R = cor(x_log, y_log)

                x_fit = range(minimum(h_time), maximum(h_time), length=100)
                y_fit = 10 .^ (m .* log10.(x_fit) .+ c)

                fit_lbl = isnan(R) ? @sprintf("l_κ = %.3g h^{%.3g}", 10^c, m) :
                                     @sprintf("l_κ = %.3g h^{%.3g} (R = %.2f)", 10^c, m, R)

                plot!(plt2, x_fit, y_fit,
                    linestyle = :dash,
                    linewidth = 1.5,
                    color     = idx,
                    label     = fit_lbl
                )
            end
        end

        plot!(plt,
            xscale    = :log10,
            yscale    = :log10,
            xlabel    = "l_N = √TKE / N [m]",
            ylabel    = "l_κ = κₜ / √TKE [m]",
            minorgrid = true,
            legend    = :bottomright,
            title     = @sprintf("l_κ vs l_N (T = %d)", T),
            margin    = 25px
        )

        mkpath(save_folder)
        savefig(plt, save_folder * "l_k_l_N.png")

        plot!(plt2,
            xscale    = :log10,
            yscale    = :log10,
            xlabel    = "h = mixed layer height [m]",
            ylabel    = "l_κ = κₜ / √TKE [m]",
            minorgrid = true,
            legend    = :bottomright,
            title     = @sprintf("l_κ vs h (T = %d)", T),
            margin    = 25px
        )

        savefig(plt2, save_folder * "l_k_h.png")
    end
end