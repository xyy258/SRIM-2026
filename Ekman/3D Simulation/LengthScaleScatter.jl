ENV["GKSwstype"] = "100"

using Oceananigans, Plots, Printf, JLD2, Statistics, LaTeXStrings
using Plots.PlotMeasures

# ==============================================================================
# Simulation & Parameter Setup
# ==============================================================================

# Sweep parameters
ratios   = [0.5, 1, 2, 5, 10, 25, 50]
values   = [5, 10, 20]
profiles = [4]

# Sampling parameters
const avg_len = 0.1  # Vertical physical window centered at interface peak [m]
const t_step  = 4

# Extended color ramp up to r = 50.0 to make r=10, r=25, and r=50 visually distinct
const RAMP = [
    (log10(0.5),  (27,  78, 143)),  # Navy Blue
    (log10(1.0),  (46, 139,  87)),  # Sea Green
    (log10(2.0),  (200, 150, 30)),  # Ochre / Gold
    (log10(5.0),  (180,  80, 44)),  # Burnt Orange
    (log10(10.0), (110,  25, 120)), # Purple
    (log10(25.0), (205,  45, 115)), # Vivid Magenta / Rose
    (log10(50.0), (90,   15,  40))  # Deep Crimson / Maroon
]

# ==============================================================================
# Helper Functions
# ==============================================================================

# Compute median ignoring non-finite values
med(v) = median(filter(isfinite, v))

# Cell-centering functions for staggered grid w-velocity components
center_w(a)    = 0.5 .* (a[:, :, 1:end-1] .+ a[:, :, 2:end])
center_w_1d(a) = 0.5 .* (a[1:end-1] .+ a[2:end])

# Generate hex color code interpolated across the defined RAMP array
function ramp_colour(s)
    x = clamp(log10(s), RAMP[1][1], RAMP[end][1])

    for i in 1:(length(RAMP) - 1)
        (x0, c0), (x1, c1) = RAMP[i], RAMP[i+1]
        x <= x1 || continue

        f = (x - x0) / (x1 - x0)
        rgb = @. clamp(round(Int, c0 + f * (c1 - c0)), 0, 255)
        return "#" * bytes2hex(UInt8.(rgb))
    end

    return "#000000"
end

# Central difference gradient calculation along the vertical axis (z)
function deriv_z(f, z)
    nz = length(f)
    df = similar(f)

    # Boundary points (forward / backward difference)
    df[1]   = (f[2] - f[1]) / (z[2] - z[1])
    df[end] = (f[end] - f[end-1]) / (z[end] - z[end-1])

    # Interior points (central difference)
    df[2:end-1] = (f[3:end] .- f[1:end-2]) ./ (z[3:end] .- z[1:end-2])

    return df
end

# Robust saturation curve fit: L_infty * (1 - exp(-x / x0))
function fit_saturation(x, y)
    best_loss = Inf
    best_L    = median(y)
    best_x0   = median(x)

    # Physically constrained parameter ranges
    ymax = maximum(y)
    ymin = minimum(y)
    xmax = maximum(x)
    xmin = minimum(x)

    # L_infty bounded close to maximum observed mixing scale
    L_range  = exp10.(range(log10(0.5 * ymax), log10(3.0 * ymax), length=150))
    x0_range = exp10.(range(log10(0.05 * xmin), log10(10.0 * xmax), length=150))

    for L in L_range
        for x0 in x0_range
            pred = @. L * (1.0 - exp(-x / x0))
            any(pred .<= 0) && continue

            # Weighted log SSE + slight penalty to break degenerate L_infty/x0 linear slope ties
            sse  = sum((log.(y) .- log.(pred)).^2)
            loss = sse + 1e-4 * log(L / ymax)

            if loss < best_loss
                best_loss = loss
                best_L    = L
                best_x0   = x0
            end
        end
    end

    return best_L, best_x0
end

# ==============================================================================
# Main Loop: Interfacial Length Scale Scatter Plots Across Sampled Timesteps
# ==============================================================================

for p in profiles
    for value in values
        global profile = p
        global T       = value

        @info @sprintf("Generating length scale plots for T = %d...", T)

        # Subplots with expanded padding around borders to prevent label clipping
        plt1 = plot(
            left_margin   = 50px,
            bottom_margin = 40px,
            top_margin    = 35px,
            right_margin  = 35px
        )
        plt2 = plot(
            left_margin   = 50px,
            bottom_margin = 40px,
            top_margin    = 35px,
            right_margin  = 35px
        )

        # Storage for overall case medians and pooled time-series values
        case_medians_x   = Float64[]
        case_medians_x_s = Float64[]
        case_medians_y   = Float64[]

        all_l_N     = Float64[]
        all_l_S     = Float64[]
        all_l_kappa = Float64[]

        # Process each ratio sweep case
        for ratio in ratios
            global r = ratio

            include("Parameters.jl")
            include("Filename_plot.jl")

            # Load 3D field time series data
            u_series = FieldTimeSeries(root * "Velocity.jld2", "u")
            v_series = FieldTimeSeries(root * "Velocity.jld2", "v")
            w_series = FieldTimeSeries(root * "Velocity.jld2", "w")
            b_series = FieldTimeSeries(root * "Buoyancy.jld2", "b")

            # Load 1D horizontally-averaged profile time series data
            u_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "u_avg")
            v_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "v_avg")
            w_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "w_avg")
            db_dz_series = FieldTimeSeries(root * "Avg_grad_b.jld2", "db_dz")

            # Extract cell-center vertical coordinates
            zC = znodes(u_series.grid, Center())

            # Select downsampled timesteps from the last 4 inertial periods
            T_f       = 2π / f₀
            n_periods = 4
            t_min     = u_series.times[end] - n_periods * T_f
            t_indices = findall(t -> t >= t_min, u_series.times)[1:t_step:end]

            # Storage for current case time-series length scales
            l_N_time     = Float64[]
            l_S_time     = Float64[]
            l_kappa_time = Float64[]

            # Loop over selected time indices
            for n in t_indices
                # Instantaneous 3D field arrays
                u = Array(interior(u_series[n], :, :, :))
                v = Array(interior(v_series[n], :, :, :))
                w = center_w(Array(interior(w_series[n], :, :, :)))
                b = Array(interior(b_series[n], :, :, :))

                # 1D mean profiles
                u_mean = Array(interior(u_avg_series[n], 1, 1, :))
                v_mean = Array(interior(v_avg_series[n], 1, 1, :))
                w_mean = center_w_1d(Array(interior(w_avg_series[n], 1, 1, :)))

                # Align vertical grid dimension
                nz    = min(size(w, 3), size(b, 3), length(w_mean))
                z_sub = zC[1:nz]

                # Calculate turbulent fluctuations (u' = u - u_mean)
                u_prime = u[:, :, 1:nz] .- reshape(u_mean[1:nz], 1, 1, nz)
                v_prime = v[:, :, 1:nz] .- reshape(v_mean[1:nz], 1, 1, nz)
                w_prime = w[:, :, 1:nz] .- reshape(w_mean[1:nz], 1, 1, nz)
                b_prime = b[:, :, 1:nz] .- reshape(mean(b[:, :, 1:nz], dims=(1, 2)), 1, 1, nz)

                # Instantaneous vertical profile metrics
                tke_inst   = vec(mean(0.5 .* (u_prime.^2 .+ v_prime.^2 .+ w_prime.^2), dims=(1, 2)))
                wb_inst    = vec(mean(w_prime .* b_prime, dims=(1, 2)))
                db_dz_inst = Array(interior(db_dz_series[n], 1, 1, :))[1:nz]
                S_inst     = sqrt.(deriv_z(u_mean[1:nz], z_sub).^2 .+ deriv_z(v_mean[1:nz], z_sub).^2)

                # Locate interface peak depth and construct vertical sampling window
                idx_peak = argmax(db_dz_inst)
                rng      = findall(abs.(z_sub .- z_sub[idx_peak]) .<= (avg_len / 2))
                if isempty(rng)
                    rng = [idx_peak]
                end

                # Interfacial window averages
                tke_int = mean(tke_inst[rng])
                K_t_int = -mean(wb_inst[rng]) / (mean(db_dz_inst[rng]) + 1e-10)
                S_int   = mean(S_inst[rng])

                # Filter valid points and compute characteristic length scales
                if K_t_int > 0 && tke_int > 1e-8 && S_int > 1e-8
                    q = sqrt(tke_int)
                    push!(l_N_time, q / N)
                    push!(l_S_time, q / S_int)
                    push!(l_kappa_time, K_t_int / max(q, 1e-6))
                end
            end

            @info @sprintf("Plotting %d timesteps for r = %.1f...", length(l_N_time), r)

            # Keyword options for individual timestep scatter points
            sc_kwargs = (
                color             = ramp_colour(r),
                markersize        = 2.5,
                markerstrokewidth = 0,
                markeralpha       = 0.45,
                label             = @sprintf("r = %.1f", r)
            )

            # Scatter instantaneous timesteps on subplots
            scatter!(plt1, l_N_time, l_kappa_time; sc_kwargs...)
            scatter!(plt2, l_S_time, l_kappa_time; sc_kwargs...)

            # Accumulate case statistics for overall trend analysis
            if !isempty(l_N_time)
                push!(case_medians_x, med(l_N_time))
                push!(case_medians_x_s, med(l_S_time))
                push!(case_medians_y, med(l_kappa_time))

                append!(all_l_N, l_N_time)
                append!(all_l_S, l_S_time)
                append!(all_l_kappa, l_kappa_time)
            end
        end

        # --- Fits for Subplot 1 (l_N) ---
        if length(case_medians_x) >= 2
            lo, hi = minimum(all_l_N), maximum(all_l_N)

            # 1:1 Reference Line
            plot!(
                plt1, [lo, hi], [lo, hi],
                color = :black, linewidth = 1.2, linestyle = :dash,
                label = L"l_\kappa = l_N" * " (1:1)"
            )

            # Fit saturation curve across case medians (or change to all_l_N, all_l_kappa if desired)
            L_fit, x0_fit = fit_saturation(case_medians_x, case_medians_y)
            @info @sprintf("Optimal Fit Parameters: L_infty = %.4f m, x0 = %.4f m", L_fit, x0_fit)

            xf = exp10.(range(log10(lo), log10(hi), length=300))
            yf = L_fit .* (1.0 .- exp.(-xf ./ x0_fit))

            plot!(
                plt1, xf, yf,
                color = :black, linewidth = 2.5,
                label = latexstring(@sprintf("l_\\kappa = L_\\infty (1 - e^{-l_N/x_0}), \\ L_\\infty = %.2f\\text{ m}, \\ x_0 = %.2f\\text{ m}", L_fit, x0_fit))
            )

            hline!(
                plt1, [L_fit],
                color = :black, linewidth = 1.0, linestyle = :dot,
                label = "plateau " * latexstring(@sprintf("L_\\infty = %.2f", L_fit)) * " m"
            )
        end

        # --- Reference Lines for Subplot 2 (l_s) ---
        if length(case_medians_x_s) >= 2
            lo_s, hi_s = minimum(all_l_S), maximum(all_l_S)

            # 1:1 Reference Line
            plot!(
                plt2, [lo_s, hi_s], [lo_s, hi_s],
                color = :black, linewidth = 1.2, linestyle = :dash,
                label = L"l_\kappa = l_s" * " (1:1)"
            )
        end

        # Overlay aggregated case medians on both subplots
        med_kwargs = (
            markersize        = 6,
            markerstrokewidth = 1.5,
            markercolor       = :white,
            markerstrokecolor = :black,
            label             = "case medians"
        )

        scatter!(plt1, case_medians_x, case_medians_y; med_kwargs...)
        scatter!(plt2, case_medians_x_s, case_medians_y; med_kwargs...)

        # --- Dynamic Viewport Range ---
        if !isempty(all_l_kappa)
            sorted_y = sort(all_l_kappa)
            ylo = sorted_y[max(1, round(Int, 0.005 * length(sorted_y)))] / 1.5
            yhi = sorted_y[round(Int, 0.999 * length(sorted_y))] * 1.5

            plot!(plt1, ylims = (ylo, yhi))
            plot!(plt2, ylims = (ylo, yhi))
        end

        # --- Subplot Formatting ---
        plot!(
            plt1,
            xscale    = :log10,
            yscale    = :log10,
            xlabel    = L"l_N = \sqrt{\mathrm{TKE}} / N" * " (m)",
            ylabel    = L"l_\kappa = K_t / \sqrt{\mathrm{TKE}}" * " (m)",
            minorgrid = true,
            legend    = :bottomright,
            title     = L"l_\kappa" * " vs " * L"l_N"
        )

        plot!(
            plt2,
            xscale    = :log10,
            yscale    = :log10,
            xlabel    = L"l_s = \sqrt{\mathrm{TKE}} / |\partial \bar{\mathbf{u}} / \partial z|" * " (m)",
            ylabel    = L"l_\kappa = K_t / \sqrt{\mathrm{TKE}}" * " (m)",
            minorgrid = true,
            legend    = :bottomright,
            title     = L"l_\kappa" * " vs " * L"l_s"
        )

        # --- Combine Subplots & Save Output (300 DPI) ---
        combined_plt = plot(
            plt1, plt2,
            layout = (1, 2),
            size   = (1700, 650),
            dpi    = 300
        )

        mkpath(save_folder)
        savefig(combined_plt, joinpath(save_folder, "Lengthscales.png"))
    end
end