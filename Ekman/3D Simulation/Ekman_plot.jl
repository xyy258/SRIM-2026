ENV["GKSwstype"] = "100"

using Oceananigans, JLD2, Plots, Printf, LaTeXStrings
using Plots.PlotMeasures

# Set global defaults for high-resolution 600 DPI static publication plots
default(dpi = 600, fontfamily = "Computer Modern")

# Import parameters
include("Parameters.jl")

# Range of profiles and stratification ratios
profiles = 0:4
r_vals   = [0.5, 1, 2, 5, 10, 25, 50]

# Secondary parameter sets matching simulation configurations
T_vals  = [5, 10, 20]
Lᴰ_vals = [5, 10, 20]

for profile in profiles
    for r_input in r_vals
        # Standardize r value for Filename_plot.jl scoping
        r     = (isnothing(r_input) || r_input == 0) ? 0.0 : Float64(r_input)
        r_val = r

        # Determine parameter variations based on profile index
        sub_params = if profile in (1, 4)
            [(T=t, Lᴰ=1.0) for t in T_vals]
        elseif profile in (2, 3)
            [(T=1.0, Lᴰ=l) for l in Lᴰ_vals]
        else
            [(T=1.0, Lᴰ=1.0)]
        end

        for params in sub_params
            T  = params.T
            Lᴰ = params.Lᴰ

            # Dynamically sets 'save_folder' and 'root', and creates directory[cite: 2]
            include("Filename_plot.jl")

            @info "=========================================="
            @info "Processing Profile: $profile | r = $r_val"
            @info "Root: $root"
            @info "Save folder: $save_folder"
            @info "=========================================="


            #  ======================================================  #
            ## Plot of average buoyancy gradient with depth over time ##
            #  ======================================================  #

            filename = root * "Avg_grad_b"
            if isfile(filename * ".jld2")
                db_dz_timeseries = FieldTimeSeries(filename * ".jld2", "db_dz")

                xb, yb, zb = nodes(db_dz_timeseries)
                t_save = db_dz_timeseries.times

                const nz = length(zb)
                const nt = length(t_save)
                gradient_data = zeros(nz, nt)

                for t_idx in 1:nt
                    gradient_data[:, t_idx] = vec(interior(db_dz_timeseries[t_idx], 1, 1, 1:nz))
                end

                zbconcat = zb[findall(<(Lz), zb)]
                Nzconcat = length(zbconcat)

                bscale = (r_val == 0.0) ? 1.0 : N²

                @info "Plot of average buoyancy gradient heatmap with depth over time..."

                heatmap(t_save * f₀, zbconcat, gradient_data[1:Nzconcat, :] / bscale,
                        xlabel = L"t f_0",
                        ylabel = "Height z",
                        title  = @sprintf("(∂b/∂z)/N² for profile %d (r = %.1f)", profile, r_val),
                        size   = (1000, 400),
                        margin = 25px,
                        color  = :thermal)

                mkpath(save_folder * "Buoyancy gradient plot")
                savefig(save_folder * @sprintf("Buoyancy gradient plot/r = %.1f.png", r_val))
            else
                @warn "Skipping buoyancy gradient plot: file $(filename).jld2 not found."
            end


            #  =======================================  #
            ##  Horizontally averaged buoyancy profile ##
            #  =======================================  #

            filename = root * "Avg_b"
            if isfile(filename * ".jld2")
                b_avg_timeseries = FieldTimeSeries(filename * ".jld2", "b")

                zb = znodes(b_avg_timeseries.grid, Center())

                b_initial = vec(interior(b_avg_timeseries[1], 1, 1, :))
                b_final   = vec(interior(b_avg_timeseries[end], 1, 1, :))

                z_mask = findall(<(Lz), zb)

                b_plot_final   = b_final[z_mask]
                b_plot_initial = b_initial[z_mask]
                z_plot         = zb[z_mask]

                bscale = (r_val == 0.0) ? 1.0 : N²

                @info "Plot of average buoyancy profile..."

                plot(b_plot_initial / bscale, z_plot,
                     xlabel    = "b/N²",
                     ylabel    = "Height z",
                     title     = @sprintf("<b> profile %d for r = %.1f", profile, r_val),
                     linewidth = 2,
                     label     = "Initial",
                     linestyle = :dash,
                     legend    = :bottomright,
                     size      = (800, 400),
                     margin    = 25px)

                plot!(b_plot_final / bscale, z_plot,
                      linewidth = 2,
                      label     = "Final")

                mkpath(save_folder * "Averaged buoyancy profile")
                savefig(save_folder * @sprintf("Averaged buoyancy profile/r = %.1f.png", r_val))
            else
                @warn "Skipping averaged buoyancy profile plot: file $(filename).jld2 not found."
            end


            #  ===============================================  #
            ## Horizontally averaged buoyancy gradient profile ##
            #  ===============================================  #

            filename = root * "Avg_grad_b"
            if isfile(filename * ".jld2")
                db_dz_avg_timeseries = FieldTimeSeries(filename * ".jld2", "db_dz")

                zb = znodes(db_dz_avg_timeseries.grid, Center())

                db_dz_initial = vec(interior(db_dz_avg_timeseries[1], 1, 1, :))
                db_dz_final   = vec(interior(db_dz_avg_timeseries[end], 1, 1, :))

                z_mask = findall(<(Lz), zb)

                db_dz_plot_initial = db_dz_initial[z_mask]
                db_dz_plot_final   = db_dz_final[z_mask]
                z_plot             = zb[z_mask]

                bscale = (r_val == 0.0) ? 1.0 : N²

                @info "Plot of average buoyancy gradient profile..."

                plot(db_dz_plot_initial / bscale, z_plot,
                     xlabel    = "(∂b/∂z)/N²",
                     ylabel    = "Height z",
                     title     = @sprintf("∂<b>/∂z profile %d for r = %.1f", profile, r_val),
                     linewidth = 2,
                     label     = "Initial",
                     linestyle = :dash,
                     legend    = :bottomright,
                     size      = (800, 600),
                     margin    = 25px)

                plot!(db_dz_plot_final / bscale, z_plot,
                      linewidth = 2,
                      label     = "Final")

                mkpath(save_folder * "Averaged buoyancy gradient profile")
                savefig(save_folder * @sprintf("Averaged buoyancy gradient profile/r = %.1f.png", r_val))
            else
                @warn "Skipping averaged buoyancy gradient profile plot: file $(filename).jld2 not found."
            end


            #  ==================  #
            ##   Hodograph plot   ##
            #  ==================  #

            vel_file = root * "Avg_vel.jld2"
            if isfile(vel_file)
                u_series = FieldTimeSeries(vel_file, "u_avg")
                v_series = FieldTimeSeries(vel_file, "v_avg")

                xu, yu, zu = nodes(u_series)
                zC = znodes(u_series.grid, Center())

                T_f = 2π / f₀
                n_periods = 5
                t_end = u_series.times[end]
                t_indices = findall(t -> t >= t_end - n_periods * T_f, u_series.times)

                u_profile = vec(sum([interior(u_series[n], 1, 1, :) for n in t_indices]) ./ length(t_indices))
                v_profile = vec(sum([interior(v_series[n], 1, 1, :) for n in t_indices]) ./ length(t_indices))

                slice   = 1:length(zC)
                u_slice = u_profile[slice]
                v_slice = v_profile[slice]
                z_slice = zC[slice]

                @info "Plot of hodograph..."

                plot(u_slice / U∞, v_slice / U∞,
                     linewidth      = 2,
                     line_z         = z_slice,
                     color          = :viridis,
                     marker         = :circle,
                     markersize     = 2,
                     marker_z       = z_slice,
                     xlabel         = L"\langle u \rangle / U_\infty",
                     ylabel         = L"\langle v \rangle / U_\infty",
                     colorbar_title = "Height z",
                     colorbar       = true,
                     size           = (1000, 500),
                     margin         = 25px,
                     legend         = false,
                     title          = @sprintf("Ekman Hodograph (profile %d, r = %.1f)", profile, r_val)
                )

                mkpath(save_folder * "Hodograph")
                savefig(save_folder * @sprintf("Hodograph/r = %.1f.png", r_val))
            else
                @warn "Skipping hodograph plot: file $vel_file not found."
            end

        end
    end
end