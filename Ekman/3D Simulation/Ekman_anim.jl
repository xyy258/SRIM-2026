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
profiles = [4]
r_vals   = [0, 0.5, 1, 2, 5, 10, 25, 50]
T_vals   = [5, 10, 20]
Lᴰ_vals  = [5, 10, 20]


# Create grid for fit_log_layer function (uses same refinement as main simulation)
refinement = 1.8 # controls spacing near surface (higher means finer spaced)
stretching = 10 # controls rate of stretching at bottom
h(k) = (Nz + 1 - k) / Nz
ζ(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))
z_faces(k) = - H * (ζ(k) * Σ(k) - 1)

grid = RectilinearGrid(CPU();
    topology = (Periodic, Periodic, Bounded),
    size     = (Nx, Ny, Nz),
    x = (0, Lx),
    y = (0, Ly),
    z = z_faces)

# Function to fit logarithmic profile and extract friction velocity
function fit_log_layer(grid, u_avg, v_avg; κ=0.41, n_points=5)
    # Extract vertical center points near the wall
    z = Array(znodes(grid, Center()))[1:n_points]

    # Horizontal mean speed U = sqrt(u_avg² + v_avg²)
    U = @. sqrt(u_avg[1:n_points]^2 + v_avg[1:n_points]^2)

    # Design matrix: U = A * ln(z) + B
    X = [log.(z) ones(n_points)]

    # Least-squares regression
    coeff = X \ U
    A, B = coeff[1], coeff[2]

    # Recover u* and z0
    u_star_fit = A * κ
    z0_fit     = exp(-B / A)

    # R² score
    U_pred = X * coeff
    SS_res = sum((U .- U_pred).^2)
    SS_tot = sum((U .- mean(U)).^2)
    r2     = 1.0 - (SS_res / SS_tot)

    return u_star_fit, z0_fit, r2
end


# Main execution loop across parameter combinations
for p in profiles
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

            # Calculate u_star from final velocity state
            vel_file_path = joinpath(root, "Avg_vel.jld2")
            vel_file  = jldopen(vel_file_path, "r")
            time_keys = keys(vel_file["timeseries/t"])
            last_iter = parse(Int, time_keys[end])

            u_avg_data = vel_file["timeseries/u_avg/$last_iter"][1, 1, :]
            v_avg_data = vel_file["timeseries/v_avg/$last_iter"][1, 1, :]
            close(vel_file)

            n_points_fit = 5
            u_star, z₀_fit, r2 = fit_log_layer(grid, u_avg_data, v_avg_data; κ=κ, n_points=n_points_fit)
            @info "Calculated u* = $(u_star) from fitted log layer profile"

            # --------------------- #
            # 1. Buoyancy Animation #
            # --------------------- #
            @info "Loading buoyancy time series..."
            b_series = FieldTimeSeries(root * "Buoyancy.jld2", "b")
            xb, yb, zb = nodes(b_series)
            times  = b_series.times
            bscale = (r == 0.0) ? 1.0 : N²

            # Define LaTeX title strings based on whether buoyancy is scaled by N²
            b_tex      = (r == 0.0) ? L"b" : L"b/N^2"
            b_pert_tex = (r == 0.0) ? L"b - b_i" : L"(b - b_i)/N^2"

            # Grid depth masks
            idx_full = findall(<(Lz), zb)
            idx_sub  = findall(<(60), zb)
            Lzmask   = zb[idx_full]
            zbmask   = zb[idx_sub]

            # Pre-extract and pre-scale data to RAM (eliminates repeated per-frame math & slicing)
            b_full = interior(b_series, :, 1, idx_full, :) ./ bscale
            b_sub  = interior(b_series, :, 1, idx_sub, :)  ./ bscale

            # Pre-calculate perturbations and static color limits
            b_pert   = b_sub .- b_sub[:, :, 1]
            clim_val = max(1e-6, 1.05 * maximum(abs, b_pert))

            @info "Making animation of buoyancy heatmaps..."
            frame_indices_b = 1:2:length(times)
            total_draws_b   = length(frame_indices_b)

            anim_b = @animate for (idx, i) in enumerate(frame_indices_b)
                if idx % 100 == 0 || idx == total_draws_b
                    @info "  Drawing buoyancy frame $idx / $total_draws_b..."
                end

                b_xz    = b_full[:, :, i]
                max_b   = max(1e-6, 1.05 * maximum(b_xz))
                t_round = round(Int, times[i])

                p1 = heatmap(xb, Lzmask, b_xz',
                            title  = string(b_tex, L" | $t = %$t_round$"),
                            color  = :thermal, clims = (0, max_b),
                            xlabel = L"x", ylabel = L"z", xlims = (0, Lx), ylims = (0, Lz))

                p2 = heatmap(xb, zbmask, b_pert[:, :, i]',
                            title  = string(b_pert_tex, L" | $t = %$t_round$"),
                            color  = :coolwarm, clims = (-clim_val, clim_val),
                            xlabel = L"x", ylabel = L"z", xlims = (0, Lx), ylims = (0, zbmask[end]))

                plot(p1, p2, layout = (2, 1), size = (1000, 550), margin = 25px)
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

            @info "Making animation of plane-averaged velocity profiles normalized by friction velocity..."
            total_frames_v = length(u_avg_series.times)

            # Pre-extract 1D velocity profiles into 2D (z, t) arrays
            u_data = interior(u_avg_series, 1, 1, :, :)
            v_data = interior(v_avg_series, 1, 1, :, :)

            anim_v = @animate for i in 1:total_frames_v
                t = u_avg_series.times[i]
                t_val = round(t, digits=1)
                if i % 200 == 0 || i == total_frames_v
                    @info "   Drawing velocity frame $i / $total_frames_v at sim time t = $t_val..."
                end

                u_prof = u_data[:, i]
                v_prof = v_data[:, i]

                p1 = plot((u_prof .- U∞)/u_star, zu,
                        color  = :navy,
                        xlabel = L"(\langle u \rangle - U_\infty)/u_*",
                        ylabel = L"z",
                        xlims  = (-5, 2),
                        ylims  = (0, Lz),
                        legend = false)

                p2 = plot(v_prof/u_star, zu,
                        color  = :crimson,
                        xlabel = L"\langle v \rangle/u_*",
                        ylabel = L"z",
                        xlims  = (-5, 2),
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

            # Pre-extract 1D vorticity profiles into 2D (z, t) arrays divided by f₀
            ωx_data = interior(ωx_avg_series, 1, 1, :, :) ./ f₀
            ωy_data = interior(ωy_avg_series, 1, 1, :, :) ./ f₀

            # Generate animation frames for mean vorticity profiles
            anim_w = @animate for i in 1:total_frames_w
                t = ωx_avg_series.times[i]
                t_val = round(t, digits=1)
                if i % 200 == 0 || i == total_frames_w
                    @info "  Drawing vorticity frame $i / $total_frames_w at sim time t = $t_val..."
                end

                p1 = plot(ωx_data[:, i], zx,
                          color  = :crimson,
                          xlabel = L"\langle \omega_x \rangle / f",
                          ylabel = L"z",
                          xlims  = (-100, 100),
                          ylims  = (0, Lz),
                          legend = false)

                p2 = plot(ωy_data[:, i], zy,
                          color  = :teal,
                          xlabel = L"\langle \omega_y \rangle / f",
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