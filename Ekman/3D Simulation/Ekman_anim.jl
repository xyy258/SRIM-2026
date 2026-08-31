ENV["GKSwstype"] = "100"

using Oceananigans, JLD2, Plots, Printf, LaTeXStrings
using Plots.PlotMeasures

# Set global defaults for clear LaTeX rendering and heatmaps without 600 DPI lag
default(dpi = 200, fontfamily = "Computer Modern")

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
        # Standardize r value for Filename_anim.jl scoping
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

            # Dynamically sets 'save_folder' and 'root', and creates directory[cite: 1]
            include("Filename_anim.jl")

            @info "=========================================="
            @info "Processing Profile: $profile | r = $r_val"
            @info "Root: $root"
            @info "Save folder: $save_folder"
            @info "=========================================="


            # ===================  #
            ## Buoyancy animation ##
            # ===================  #

            buoyancy_file = root * "Buoyancy.jld2"
            if isfile(buoyancy_file)
                b_series = FieldTimeSeries(buoyancy_file, "b")

                xb, yb, zb = nodes(b_series)
                times = b_series.times

                @info "Making animation of buoyancy heatmaps..."

                # Masking for (0, Lz)
                Lzmask  = zb[findall(x -> x < Lz, zb)]
                NLzmask = length(Lzmask)

                # Masking for height from bottom of domain
                z_mask = findall(x -> x < 60, zb)
                zbmask = zb[z_mask]
                Nzmask = length(zbmask)

                bscale = (r_val == 0.0) ? 1.0 : N²

                # Load initial buoyancy profile
                b_initial = interior(b_series[1], :, 1, 1:Nzmask)

                # Fixing colour limits for buoyancy difference plot
                clim_abs = maximum(1:5:length(times)) do i
                    b_frame = interior(b_series[i], :, 1, 1:Nzmask)
                    maximum(abs, (b_frame .- b_initial) ./ bscale)
                end
                clim_val = max(1e-6, clim_abs * 1.05)

                anim = @animate for i in 1:length(times)
                    t = times[i]
                    if i % 100 == 0
                        @info "Drawing frame $i / $(length(times))..."
                    end

                    b_xz     = interior(b_series[i], :, 1, 1:NLzmask)
                    b_xz_sub = interior(b_series[i], :, 1, 1:Nzmask)

                    b_max = maximum(b_xz ./ bscale)
                    b_xz_plot = heatmap(xb, Lzmask, (b_xz ./ bscale)';
                        color  = :thermal,
                        clims  = (0, max(1e-6, 1.05 * b_max)),
                        xlabel = L"x", ylabel = L"z",
                        xlims  = (0, Lx), ylims = (0, Lz))

                    b_diff_xz_plot = heatmap(xb, zbmask, ((b_xz_sub .- b_initial) ./ bscale)';
                        color  = :coolwarm,
                        clims  = (-clim_val, clim_val),
                        xlabel = L"x", ylabel = L"z",
                        xlims  = (0, Lx), ylims = (0, zbmask[end]))

                    t_round = round(Int, t)
                    b_title      = L"b/N^2\text{ at } t = %$(t_round),\ N/f = %$(r_val)"
                    b_diff_title = L"(b - b_i)/N^2\text{ at } t = %$(t_round),\ N/f = %$(r_val)"

                    plot(b_xz_plot, b_diff_xz_plot,
                        layout = (2, 1),
                        size   = (1000, 550),
                        title  = [b_title b_diff_title],
                        margin = 25px)
                end

                mkpath(save_folder * "Buoyancy")
                mp4(anim, save_folder * @sprintf("Buoyancy/r = %.1f.mp4", r_val), fps = 30)
            else
                @warn "Skipping buoyancy animation: file $buoyancy_file not found."
            end


            #  ==========================  #
            ## Average velocity animation ##
            #  ==========================  #

            vel_file = root * "Avg_vel.jld2"
            if isfile(vel_file)
                u_avg_series = FieldTimeSeries(vel_file, "u_avg")
                v_avg_series = FieldTimeSeries(vel_file, "v_avg")

                times = u_avg_series.times
                zu    = znodes(u_avg_series[1])

                ylimits = (0, Lz)

                @info "Making animation of plane-averaged velocity profiles..."

                anim = @animate for i in 1:length(times)
                    t = times[i]
                    if i % 100 == 0
                        @info "Drawing frame $i / $(length(times)) at sim time t = $(round(t, digits=1))..."
                    end

                    u_prof = vec(interior(u_avg_series[i], 1, 1, :))
                    v_prof = vec(interior(v_avg_series[i], 1, 1, :))

                    p1 = plot((u_prof .- U∞) / u_star, zu,
                              linewidth = 3,
                              color     = :navy,
                              xlabel    = L"(\langle u \rangle - U_\infty)/u_*",
                              ylabel    = L"z",
                              xlims     = (-10, 7.5),
                              ylims     = ylimits,
                              grid      = true,
                              margin    = 25px,
                              legend    = false)

                    p2 = plot(v_prof / u_star, zu,
                              linewidth = 3,
                              color     = :crimson,
                              xlabel    = L"\langle v \rangle/u_*",
                              ylabel    = L"z",
                              xlims     = (-10, 7.5),
                              ylims     = ylimits,
                              grid      = true,
                              margin    = 25px,
                              legend    = false)

                    title_str = L"\text{Velocity profiles }(N/f = %$(r_val))\text{ | } t = %$(round(t, digits=1))"

                    plot(p1, p2,
                        layout     = (2, 1),
                        size       = (1000, 600),
                        margin     = 25px,
                        plot_title = title_str)
                end

                mkpath(save_folder * "Velocity")
                mp4(anim, save_folder * @sprintf("Velocity/r = %.1f.mp4", r_val), fps = 60)
            else
                @warn "Skipping velocity animation: file $vel_file not found."
            end


            # ============================= #
            ## Average vorticity animation ##
            # ============================= #

            vort_file = root * "Avg_vort.jld2"
            if isfile(vort_file)
                @info "Loading vorticity time series..."

                ωx_avg_series = FieldTimeSeries(vort_file, "ωx_avg")
                ωy_avg_series = FieldTimeSeries(vort_file, "ωy_avg")

                vort_times = ωx_avg_series.times

                zx = znodes(ωx_avg_series[1])
                zy = znodes(ωy_avg_series[1])

                ylimits = (0, Lz)

                @info "Making animation of plane-averaged vorticity profiles..."

                anim_vort = @animate for i in 1:length(vort_times)
                    t = vort_times[i]
                    if i % 100 == 0
                        @info "Drawing vorticity frame $i / $(length(vort_times)) at sim time t = $(round(t, digits=1))..."
                    end

                    ωx_prof = vec(interior(ωx_avg_series[i], 1, 1, :))
                    ωy_prof = vec(interior(ωy_avg_series[i], 1, 1, :))

                    p_x = plot(ωx_prof / f₀, zx,
                               linewidth = 2,
                               color     = :crimson,
                               xlabel    = L"\langle \omega_x \rangle / f_0",
                               ylabel    = L"z",
                               xlims     = (-100, 100),
                               ylims     = ylimits,
                               grid      = true,
                               legend    = false)

                    p_y = plot(ωy_prof / f₀, zy,
                               linewidth = 2,
                               color     = :teal,
                               xlabel    = L"\langle \omega_y \rangle / f_0",
                               ylabel    = L"z",
                               xlims     = (-50, 200),
                               ylims     = ylimits,
                               grid      = true,
                               legend    = false)

                    vort_title = L"\text{Plane-Averaged Vorticity Profiles }(N/f = %$(r_val))\text{ | } t = %$(round(t, digits=1))"

                    plot(p_x, p_y,
                        layout     = (1, 2),
                        size       = (1000, 600),
                        margin     = 25px,
                        plot_title = vort_title
                    )
                end

                mkpath(save_folder * "Vorticity")
                mp4(anim_vort, save_folder * @sprintf("Vorticity/r = %.1f.mp4", r_val), fps = 60)
            else
                @warn "Skipping vorticity animation: file $vort_file not found."
            end

        end
    end
end