ENV["GKSwstype"] = "100"

using Oceananigans, Plots, Printf, JLD2, Statistics
using Plots.PlotMeasures

# Define parameter ranges for sweep
ratios   = [0.5, 1, 2, 5, 10, 25, 50]
values   = [5, 10, 20]
profiles = [4]

# =============================================== #
##  Mixed Layer Height Evolution h(t) Over Time  ##
# =============================================== #

for p in profiles
    global profile = p
    for value in values
        global T = value

        # Initialize combined plot for all ratios r at fixed T
        plt_h_combined = plot(
            xlabel    = "Time [Inertial Periods]",
            ylabel    = "Layer Height h(z) [m]",
            size      = (900, 500),
            margin    = 25px,
            legend    = :outertopright,
            title     = @sprintf("Layer Height Evolution Comparison (T = %d)", T)
        )

        for (idx, ratio) in enumerate(ratios)
            global r = ratio

            # Load case-specific parameters and file paths dynamically
            include("Parameters.jl")
            include("Filename_plot.jl")

            @info @sprintf("Processing Case: profile = %d, T = %d, r = %.1f", profile, T, r)

            # Load FieldTimeSeries (only reading db_dz and grid definitions needed for h(t))
            u_series     = FieldTimeSeries(root * "Velocity.jld2", "u")
            db_dz_series = FieldTimeSeries(root * "Avg_grad_b.jld2", "db_dz")

            zC = znodes(u_series.grid, Center())

            T_f_calc = 2π / f₀
            t_all = u_series.times
            t_nondim = t_all ./ T_f_calc
            N_times = length(t_all)

            # Preallocate vector for layer height
            h_time = Vector{Float64}(undef, N_times)

            for (n, t) in enumerate(t_all)
                n % 50 == 0 && @info @sprintf("Step %d / %d", n, N_times)

                db_dz_inst = Array(interior(db_dz_series[n], 1, 1, :))
                nz = length(db_dz_inst)

                # Vertically smooth ∂b/∂z across depth z
                db_dz_smooth = [mean(db_dz_inst[max(1, i-2):min(nz, i+2)]) for i in 1:nz]

                # Depth index where ∂b/∂z is maximized
                idx_peak = argmax(db_dz_smooth)

                h_time[n] = zC[idx_peak]
            end

            # Add current ratio line to the combined plot
            plot!(plt_h_combined, t_nondim, h_time,
                linewidth = 2.0,
                color     = idx,
                label     = @sprintf("r = %.1f", r)
            )

            # Statistics printed to console
            n_periods = 4
            t_end = t_all[end]
            t_indices = findall(t -> t >= t_end - n_periods * T_f_calc, t_all)
            h_final = h_time[t_indices]

            @printf("\n=== r = %.1f, T = %d | LAYER HEIGHT (Last %.1f Inertial Periods) ===\n", r, T, n_periods)
            @printf("  Mean: %.4f m  |  Std: %.4f m  |  Min: %.4f m  |  Max: %.4f m\n\n",
                mean(h_final), std(h_final), minimum(h_final), maximum(h_final))
        end

        # Save combined layer height plot directly into save_folder
        mkpath(save_folder)
        savefig(plt_h_combined, joinpath(save_folder, @sprintf("Layer Height Plot T=%d.png", T)))
    end
end