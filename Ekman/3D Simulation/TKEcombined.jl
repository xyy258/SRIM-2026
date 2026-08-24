ENV["GKSwstype"] = "100"

using Oceananigans, Plots, Printf, JLD2
using Plots.PlotMeasures
using Statistics
using CUDA

ratios = [0, 0.5, 1, 2, 5]
values = [5,10,15,20,30,40,50]
profiles = [4]

# ======================================= #
##  Turbulent Kinetic Energy (TKE) Plot  ##
# ======================================= #

for p in profiles
    global profile = p
    for value in values
        plt = plot(size  = (800, 500)
        )

        global T = value
        @info("Plot for T=$T...")
        for (idx, ratio) in enumerate(ratios)
            global r = ratio

            include("Parameters.jl")
            include("Filename_plot.jl")

            u_series = FieldTimeSeries(root * "Velocity.jld2", "u")
            v_series = FieldTimeSeries(root * "Velocity.jld2", "v")
            w_series = FieldTimeSeries(root * "Velocity.jld2", "w")

            u_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "u_avg")
            v_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "v_avg")
            w_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "w_avg")

            zC = znodes(u_series.grid, Center())
            center_w = a -> 0.5 .* (a[:, :, 1:end-1] .+ a[:, :, 2:end])
            center_w_profile = a -> 0.5 .* (a[1:end-1] .+ a[2:end])

            T_f = 2π / f₀
            n_periods = 5
            t_end = u_series.times[end]
            t_indices = findall(t -> t >= t_end - n_periods * T_f, u_series.times)

            tke_profile_avg = zeros(length(zC))

            for n in t_indices
                u = Array(interior(u_series[n], :, :, :))
                v = Array(interior(v_series[n], :, :, :))
                w = center_w(Array(interior(w_series[n], :, :, :)))

                u_mean = Array(interior(u_avg_series[n], 1, 1, :))
                v_mean = Array(interior(v_avg_series[n], 1, 1, :))
                w_mean = center_w_profile(Array(interior(w_avg_series[n], 1, 1, :)))

                u_prime = u .- reshape(u_mean, (1, 1, :))
                v_prime = v .- reshape(v_mean, (1, 1, :))
                w_prime = w .- reshape(w_mean, (1, 1, :))

                tke_inst = 0.5 .* (u_prime.^2 .+ v_prime.^2 .+ w_prime.^2)
                tke_profile = vec(mean(tke_inst, dims=(1, 2)))
                tke_profile_avg .+= tke_profile
            end

            tke_profile_avg ./= length(t_indices)
            tke_norm = tke_profile_avg / U∞^2

            @info "Making time-averaged TKE plot for r = $r..."

            # Swapped zC and tke_norm, moved log scale to yaxis
            plot!(plt, zC, tke_norm,
                yaxis     = :log,
                linewidth = 2,
                color     = idx,
                label     = @sprintf("r = %.1f", r)
            )

            # Fit points where z <= 0.25*Lz, then extrapolate full line across all zC
            mask = zC .<= 0.25 * Lz
            if count(mask) >= 2
                x_sub = tke_norm[mask]
                z_sub = zC[mask]

                # Fit log10(TKE/U∞²) = m * z + c
                A = [z_sub ones(length(z_sub))]
                m, c = A \ log10.(x_sub)

                # Evaluate over full domain zC
                x_full = 10 .^ (m .* zC .+ c)

                # Swapped zC and x_full
                plot!(plt, zC, x_full,
                    linestyle = :dash,
                    linewidth = 1.5,
                    color     = idx,
                    label     = @sprintf("Fit r = %.1f (grad = %.2e)", r, m)
                )
            end
        end

        plot!(plt,
            xlabel    = "Depth z [m]",
            ylabel    = "TKE/U∞^2",
            xlims     = (0, Lz),
            minorgrid = true,
            legend    = :bottomleft,
            title     = "TKE (averaged over 5 periods) log plot against depth for T=$T",
            margin    = 25px
        )

        mkpath(save_folder)
        savefig(plt, save_folder * "TKE Plots T=$T.png")
    end
end