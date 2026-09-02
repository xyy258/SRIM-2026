# Set headless rendering environment for Plots.jl
ENV["GKSwstype"] = "100"

using Oceananigans, JLD2, Plots, Printf, LaTeXStrings
using Plots.PlotMeasures

# High-DPI plot formatting with full Unicode glyph support
default(dpi = 600, fontfamily = "DejaVu Sans")

# Import parameters and filename configurations
include("Parameters.jl")
include("Filename_anim.jl")

# Define parameter sweep configurations
profiles_sweep = [4]
r_vals         = [0, 0.5, 1, 2, 5, 10, 25, 50]
T_vals         = [5, 10, 20]
Lᴰ_vals        = [5, 10, 20]

# Main execution loop across parameter combinations
for p in profiles_sweep
    global profile = p # Expose profile to global scope for included scripts

    for r_input in r_vals
        global r = Float64(r_input)

        # Select sub-parameter combinations based on profile type
        sub_params = profile in (1, 4) ? [(T=t, Lᴰ=1.0) for t in T_vals] :
                     profile in (2, 3) ? [(T=1.0, Lᴰ=l) for l in Lᴰ_vals] : [(T=1.0, Lᴰ=1.0)]

        for params in sub_params
            global T  = params.T
            global Lᴰ = params.Lᴰ

            # Load domain parameters and plot path settings
            include("Parameters.jl")
            include("Filename_plot.jl")

            # Dynamically format parameter string for logs and titles
            param_str = if profile in (1, 4)
                @sprintf("r = %.1f, T = %.1f", r, T)
            elseif profile in (2, 3)
                @sprintf("r = %.1f, L_D = %.1f", r, Lᴰ)
            else
                @sprintf("r = %.1f", r)
            end

            @info "=========================================="
            @info "Processing Plots for Profile $profile ($param_str)"
            @info "Save folder: $save_folder"
            @info "=========================================="

            bscale = (r == 0.0) ? 1.0 : N²

            # Dynamic LaTeX label strings for r = 0 vs r > 0
            grad_title_tex = (r == 0.0) ? L"\partial b / \partial z" : L"(\partial b / \partial z) / N^2"
            b_avg_tex      = (r == 0.0) ? L"\langle b \rangle" : L"\langle b \rangle / N^2"
            db_dz_avg_tex  = (r == 0.0) ? L"\partial \langle b \rangle / \partial z" : L"(\partial \langle b \rangle / \partial z) / N^2"

            # ---------------------------------------------------- #
            # 1. Average Buoyancy Gradient Heatmap (Depth vs Time) #
            # ---------------------------------------------------- #
            @info "Plot of average buoyancy gradient heatmap with depth over time..."
            db_dz_series = FieldTimeSeries(root * "Avg_grad_b.jld2", "db_dz")
            xb, yb, zb   = nodes(db_dz_series)
            t_save       = db_dz_series.times

            # Assemble time series data into a 2D depth-time matrix
            grad_data = reduce(hcat, [vec(interior(db_dz_series[i], 1, 1, :)) for i in 1:length(t_save)])
            z_mask    = findall(<(Lz), zb)

            heatmap(t_save * f₀, zb[z_mask], grad_data[z_mask, :] / bscale,
                    color  = :thermal,
                    xlabel = L"t f",
                    ylabel = L"Height $z$",
                    title  = string(grad_title_tex, " (", param_str, ")"),
                    size   = (1000, 400),
                    margin = 25px)

            mkpath(save_folder * "Buoyancy gradient plot")
            savefig(save_folder * @sprintf("Buoyancy gradient plot/r = %.1f.png", r))

            # ----------------------------------- #
            # 2. Averaged Buoyancy Line Profile   #
            # ----------------------------------- #
            @info "Plot of average buoyancy profile..."
            b_avg_series = FieldTimeSeries(root * "Avg_b.jld2", "b")
            zb = znodes(b_avg_series.grid, Center())
            z_mask = findall(<(Lz), zb)

            # Plot initial vs final averaged buoyancy vertical profile
            plot(vec(interior(b_avg_series[1], 1, 1, z_mask)) / bscale, zb[z_mask],
                 xlabel    = b_avg_tex,
                 ylabel    = L"Height $z$",
                 title     = string(L"\langle b \rangle", " (", param_str, ")"),
                 linewidth = 2,
                 linestyle = :dash,
                 label     = "Initial",
                 legend    = :bottomright,
                 size      = (800, 400),
                 margin    = 25px)

            plot!(vec(interior(b_avg_series[end], 1, 1, z_mask)) / bscale, zb[z_mask],
                  linewidth = 2,
                  label     = "Final")

            mkpath(save_folder * "Averaged buoyancy profile")
            savefig(save_folder * @sprintf("Averaged buoyancy profile/r = %.1f.png", r))

            # -------------------------------------------- #
            # 3. Averaged Buoyancy Gradient Line Profile  #
            # -------------------------------------------- #
            @info "Plot of average buoyancy gradient profile..."
            # Plot initial vs final averaged buoyancy gradient profile
            plot(vec(interior(db_dz_series[1], 1, 1, z_mask)) / bscale, zb[z_mask],
                 xlabel    = db_dz_avg_tex,
                 ylabel    = L"Height $z$",
                 title     = string(L"\partial \langle b \rangle / \partial z", " (", param_str, ")"),
                 linewidth = 2,
                 linestyle = :dash,
                 label     = "Initial",
                 legend    = :bottomright,
                 size      = (800, 600),
                 margin    = 25px)

            plot!(vec(interior(db_dz_series[end], 1, 1, z_mask)) / bscale, zb[z_mask],
                  linewidth = 2,
                  label     = "Final")

            mkpath(save_folder * "Averaged buoyancy gradient profile")
            savefig(save_folder * @sprintf("Averaged buoyancy gradient profile/r = %.1f.png", r))

            # ----------------- #
            # 4. Hodograph Plot #
            # ----------------- #
            @info "Plot of hodograph..."
            u_series = FieldTimeSeries(root * "Avg_vel.jld2", "u_avg")
            v_series = FieldTimeSeries(root * "Avg_vel.jld2", "v_avg")
            zC = znodes(u_series.grid, Center())

            # Filter indices based on depth boundary layer height Lz
            z_mask_vel = findall(<(Lz), zC)

            # Time-average over the last 5 inertial periods
            t_indices = findall(t -> t >= u_series.times[end] - 5 * (2π / f₀), u_series.times)
            u_prof = sum([vec(interior(u_series[n], 1, 1, :)) for n in t_indices]) ./ length(t_indices)
            v_prof = sum([vec(interior(v_series[n], 1, 1, :)) for n in t_indices]) ./ length(t_indices)

            # Apply depth slice mask
            u_slice = u_prof[z_mask_vel]
            v_slice = v_prof[z_mask_vel]
            z_slice = zC[z_mask_vel]

            # Plot u-velocity vs v-velocity colored by vertical height z
            plot(u_slice / U∞, v_slice / U∞,
                 linewidth      = 2,
                 line_z         = z_slice,
                 color          = :viridis,
                 marker         = :circle,
                 markersize     = 2,
                 marker_z       = z_slice,
                 xlabel         = L"\langle u \rangle / U_\infty",
                 ylabel         = L"\langle v \rangle / U_\infty",
                 colorbar_title = L"Height $z$",
                 title          = string("Ekman Hodograph (", param_str, ")"),
                 size           = (1000, 500),
                 margin         = 25px,
                 legend         = false)

            mkpath(save_folder * "Hodograph")
            savefig(save_folder * @sprintf("Hodograph/r = %.1f.png", r))
        end
    end
end