ENV["GKSwstype"] = "100"

using Oceananigans, Plots, Printf, JLD2
using Plots.PlotMeasures
using Statistics

ratios = [0.5, 1, 2, 5]
values = [5,10,15,20,30,40,50]                   # set fixed T value
profiles = [4]

const dz_near = 15              # grid points above and below the interface peak

# ======================================================================== #
##  Near-Interfacial Length Scale Scatter Plot: l_κ vs l_N (Log-Log)       ##
# ======================================================================== #

for p in profiles
    global profile = p
    for value in values
        plt = plot(size = (800, 500))

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

            T_f = 2π / f₀
            t_indices = findall(t -> t >= u_series.times[end] - 5 * T_f, u_series.times)

            tke_profile_avg = zeros(length(zC))
            wb_profile = zeros(length(zC))
            db_dz_profile = zeros(length(zC))

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

                tke_profile_avg[1:nz] .+= vec(mean(0.5 .* (u_prime.^2 .+ v_prime.^2 .+ w_prime.^2), dims=(1, 2)))
                wb_profile[1:nz]      .+= vec(mean(w_prime .* b_prime, dims=(1, 2)))
                db_dz_profile[1:nz]   .+= Array(interior(db_dz_series[n], 1, 1, :))[1:nz]
            end

            tke_profile_avg ./= length(t_indices)
            wb_profile ./= length(t_indices)
            db_dz_profile ./= length(t_indices)

            # --- Define interface region ---
            idx_int = argmax(db_dz_profile)
            rng = max(1, idx_int - dz_near):min(length(zC), idx_int + dz_near)

            K_t_near = -wb_profile[rng] ./ (db_dz_profile[rng] .+ 1e-10)
            l_N_near = sqrt.(max.(tke_profile_avg[rng], 0)) ./ N
            l_kappa_near = K_t_near ./ sqrt.(max.(tke_profile_avg[rng], 1e-12))

            valid = (K_t_near .> 0) .& (tke_profile_avg[rng] .> 1e-8)

            @info @sprintf("Plotting interface region for r = %.1f...", r)

            scatter!(plt, l_N_near[valid], l_kappa_near[valid],
                color      = idx,
                markersize = 3.5,
                label      = @sprintf("r = %.1f", r)
            )

            # --- Line of Best Fit (Linear regression in log10 space) ---
            if count(valid) >= 2
                x_sub = l_N_near[valid]
                y_sub = l_kappa_near[valid]

                x_log = log10.(x_sub)
                y_log = log10.(y_sub)

                # Fit log10(l_κ) = m * log10(l_N) + c
                A = [x_log ones(length(x_log))]
                m, c = A \ y_log
                C = 10^c  # Coefficient for power law l_κ = C * l_N^m

                # Create smooth range across domain for fit line
                x_fit = range(minimum(x_sub), maximum(x_sub), length=100)
                y_fit = 10 .^ (m .* log10.(x_fit) .+ c)

                # Equation label with 3 significant figures
                fit_label = @sprintf("Fit r=%.1f: l_κ = %.3g l_N^{%.3g}", r, C, m)

                plot!(plt, x_fit, y_fit,
                    linestyle = :dash,
                    linewidth = 1.5,
                    color     = idx,
                    label     = fit_label
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
            title     = "Near-Interfacial Length Scale Log-Log Scatter Plot for T=$T",
            margin    = 25px
        )

        mkpath(save_folder)
        savefig(plt, save_folder * "Near_Interfacial_Length_Scale_LogLog_Scatter_T=$T.png")
    end
end