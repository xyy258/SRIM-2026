# 1. Force headless rendering and use Cairo PNG backend for GR
ENV["GKSwstype"] = "100"
ENV["GKS_USE_CAIRO_PNG"] = "true"

using Oceananigans, JLD2, Plots, Printf, LaTeXStrings, Statistics
using Plots.PlotMeasures

# 2. Use a standard font family with full Unicode/math glyph support
default(dpi = 120, fontfamily = "DejaVu Sans")

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

            # Load domain parameters and animation path settings
            include("Parameters.jl")
            include("Filename_anim.jl")

            # Dynamically format parameter string for logs and titles
            param_str = if profile in (1, 4)
                @sprintf("r = %.1f, T = %.1f", r, T)
            elseif profile in (2, 3)
                @sprintf("r = %.1f, L_D = %.1f", r, Lᴰ)
            else
                @sprintf("r = %.1f", r)
            end

            @info "=========================================="
            @info "Processing Animations for Profile $profile ($param_str)"
            @info "Save folder: $save_folder"
            @info "=========================================="

            # --------------------- #
            # 1. Buoyancy Animation #
            # --------------------- #
            @info "Loading buoyancy time series..."
            b_series = FieldTimeSeries(root * "Buoyancy.jld2", "b")
            xb, yb, zb = nodes(b_series)
            times  = b_series.times
            bscale = (r == 0.0) ? 1.0 : N²

            # Mask depth grid below specific limits for visualization
            Lzmask = zb[zb .< Lz]
            z_mask = findall(<(60), zb)
            zbmask = zb[z_mask]

            # Initial state and color bar limit calculations
            b_initial = interior(b_series[1], :, 1, z_mask)
            clim_val  = max(1e-6, 1.05 * maximum(i -> maximum(abs, (interior(b_series[i], :, 1, z_mask) .- b_initial) ./ bscale), 1:5:length(times)))

            @info "Making animation of buoyancy heatmaps..."
            total_frames_b = length(times)

            # Generate animation frames for buoyancy field and perturbation
            anim_b = @animate for i in 1:total_frames_b
                if i % 200 == 0 || i == total_frames_b
                    @info "  Drawing buoyancy frame $i / $total_frames_b..."
                end

                b_xz     = interior(b_series[i], :, 1, 1:length(Lzmask))
                b_xz_sub = interior(b_series[i], :, 1, z_mask)

                p1 = heatmap(xb, Lzmask, (b_xz ./ bscale)';
                             color  = :thermal,
                             clims  = (0, max(1e-6, 1.05 * maximum(b_xz ./ bscale))),
                             xlabel = L"x",
                             ylabel = L"z",
                             xlims  = (0, Lx),
                             ylims  = (0, Lz))

                p2 = heatmap(xb, zbmask, ((b_xz_sub .- b_initial) ./ bscale)';
                             color  = :coolwarm,
                             clims  = (-clim_val, clim_val),
                             xlabel = L"x",
                             ylabel = L"z",
                             xlims  = (0, Lx),
                             ylims  = (0, zbmask[end]))

                t_round = round(Int, times[i])
                plot(p1, p2,
                     layout = (2, 1),
                     size   = (1000, 550),
                     margin = 25px,
                     title  = [L"$\frac{b}{N^2}$ at $t = %$t_round$",
                               L"$\frac{b - b_i}{N^2}$ at $t = %$t_round$"])
            end

            # Save buoyancy animation output
            mkpath(save_folder * "Buoyancy")
            mp4(anim_b, save_folder * @sprintf("Buoyancy/r = %.1f.mp4", r), fps = 30)

            # ----------------------------- #
            # 2. Average Velocity Animation #
            # ----------------------------- #
            @info "Loading velocity time series..."
            u_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "u_avg")
            v_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "v_avg")
            zu = znodes(u_avg_series[1])

            @info "Making animation of plane-averaged velocity profiles..."
            total_frames_v = length(u_avg_series.times)

            anim_v = @animate for i in 1:total_frames_v
                t = u_avg_series.times[i]
                t_val = round(t, digits=1)
                if i % 200 == 0 || i == total_frames_v
                    @info "   Drawing velocity frame $i / $total_frames_v at sim time t = $t_val..."
                end

                u_prof = vec(interior(u_avg_series[i], 1, 1, :))
                v_prof = vec(interior(v_avg_series[i], 1, 1, :))

                p1 = plot(u_prof, zu,
                        color  = :navy,
                        xlabel = L"\langle u \rangle",
                        ylabel = L"z",
                        ylims  = (0, Lz),
                        legend = false)

                p2 = plot(v_prof, zu,
                        color  = :crimson,
                        xlabel = L"\langle v \rangle",
                        ylabel = L"z",
                        ylims  = (0, Lz),
                        legend = false)

                plot(p1, p2,
                    layout     = (2, 1),
                    size       = (1000, 600),
                    margin     = 25px,
                    plot_title = L"Velocity Profiles (%$param_str) | $t = %$t_val$")
            end

            # Save velocity animation output
            mkpath(save_folder * "Velocity")
            mp4(anim_v, save_folder * @sprintf("Velocity/r = %.1f.mp4", r), fps = 60)

            # ------------------------------ #
            # 3. Average Vorticity Animation #
            # ------------------------------ #
            @info "Loading vorticity time series..."
            ωx_avg_series = FieldTimeSeries(root * "Avg_vort.jld2", "ωx_avg")
            ωy_avg_series = FieldTimeSeries(root * "Avg_vort.jld2", "ωy_avg")
            zx = znodes(ωx_avg_series[1])
            zy = znodes(ωy_avg_series[1])

            @info "Making animation of plane-averaged vorticity profiles..."
            total_frames_w = length(ωx_avg_series.times)

            # Generate animation frames for mean vorticity profiles
            anim_w = @animate for i in 1:total_frames_w
                t = ωx_avg_series.times[i]
                t_val = round(t, digits=1)
                if i % 200 == 0 || i == total_frames_w
                    @info "  Drawing vorticity frame $i / $total_frames_w at sim time t = $t_val..."
                end

                p1 = plot(vec(interior(ωx_avg_series[i], 1, 1, :)) / f₀, zx,
                          color  = :crimson,
                          xlabel = L"\langle \omega_x \rangle / f_0",
                          ylabel = L"z",
                          xlims  = (-100, 100),
                          ylims  = (0, Lz),
                          legend = false)

                p2 = plot(vec(interior(ωy_avg_series[i], 1, 1, :)) / f₀, zy,
                          color  = :teal,
                          xlabel = L"\langle \omega_y \rangle / f_0",
                          ylabel = L"z",
                          xlims  = (-50, 200),
                          ylims  = (0, Lz),
                          legend = false)

                plot(p1, p2,
                     layout     = (1, 2),
                     size       = (1000, 600),
                     margin     = 25px,
                     plot_title = L"Vorticity Profiles (%$param_str) | $t = %$t_val$")
            end

            # Save vorticity animation output
            mkpath(save_folder * "Vorticity")
            mp4(anim_w, save_folder * @sprintf("Vorticity/r = %.1f.mp4", r), fps = 60)
        end
    end
end