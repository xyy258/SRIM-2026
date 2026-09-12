ENV["GKSwstype"] = "100"

using Oceananigans, Plots, Printf, JLD2, Statistics, LaTeXStrings
using Plots.PlotMeasures

# High-DPI plot formatting with full Unicode glyph support
default(dpi = 600, fontfamily = "DejaVu Sans")

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
# Helper Functions: Adaptive Thresholding
# ==============================================================================

function compute_percentile_threshold(data::Vector; low_pct::Real = 2.5, high_pct::Real = 97.5)
    """
    Filter data to percentile range [low_pct, high_pct].
    Returns: (lower_bound, upper_bound, n_valid, n_excluded)
    """
    finite_data = filter(isfinite, data)
    isempty(finite_data) && return (NaN, NaN, 0, 0)

    lower = quantile(finite_data, low_pct / 100)
    upper = quantile(finite_data, high_pct / 100)

    n_valid = count(lower .<= finite_data .<= upper)
    n_excluded = length(finite_data) - n_valid

    return (lower, upper, n_valid, n_excluded)
end

function apply_adaptive_filter(l_N, ratio_metric; low_pct::Real = 2.5, high_pct::Real = 97.5)
    """
    Apply adaptive filtering to length scale and TKE/(N*kappa) ratio metric:
    1. Remove non-finite values
    2. Use 2.5% - 97.5% percentile thresholding on ratio metric
    
    Returns: (filtered_l_N, filtered_ratio_metric, filter_mask, stats)
    """
    n_total = length(ratio_metric)

    # Start with finite values
    mask_finite = isfinite.(l_N) .& isfinite.(ratio_metric)

    # Percentile-based filtering on ratio metric
    finite_ratio = filter(isfinite, ratio_metric)
    if !isempty(finite_ratio)
        r_lo, r_hi, _, _ = compute_percentile_threshold(ratio_metric; low_pct=low_pct, high_pct=high_pct)
        mask_ratio = (r_lo .<= ratio_metric .<= r_hi)
    else
        mask_ratio = trues(n_total)
    end

    # Combine masks
    final_mask = mask_finite .& mask_ratio

    return (
        l_N[final_mask],
        ratio_metric[final_mask],
        final_mask,
        (n_before = n_total, n_after = sum(final_mask), n_removed = n_total - sum(final_mask))
    )
end

# ==============================================================================
# Original Helper Functions
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

# ==============================================================================
# Main Loop: TKE / (N * kappa) Scatter Plots Across Sampled Timesteps
# ==============================================================================

for p in profiles
    for value in values
        global profile = p
        global T       = value

        @info @sprintf("Generating TKE / (N * kappa) plots for T = %d...", T)

        # Single plot canvas setup with margin padding
        plt = plot(
            left_margin   = 50px,
            bottom_margin = 40px,
            top_margin    = 35px,
            right_margin  = 35px
        )

        # Storage for overall case medians and pooled values
        case_medians_x = Float64[]
        case_medians_y = Float64[]

        all_l_N          = Float64[]
        all_ratio_metric = Float64[]

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

            # Storage for current case time-series (before filtering)
            l_N_time          = Float64[]
            ratio_metric_time = Float64[]

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

                # Locate layer height where TKE falls to 1% of peak TKE
                tke_peak       = maximum(tke_inst)
                tke_candidates = findall(tke_inst .>= 0.01 * tke_peak)

                # Select the highest vertical index meeting the threshold
                idx_peak = isempty(tke_candidates) ? argmax(tke_inst) : maximum(tke_candidates)

                # Construct sampling window around this layer height
                rng = findall(abs.(z_sub .- z_sub[idx_peak]) .<= (avg_len / 2))
                if isempty(rng)
                    rng = [idx_peak]
                end

                # Interfacial window averages
                tke_int = mean(tke_inst[rng])
                K_t_int = -mean(wb_inst[rng]) / (mean(db_dz_inst[rng]) + 1e-10)

                # Compute TKE / (N * K_t) ratio metric and length scale l_N
                if K_t_int > 0 && tke_int > 1e-12 && N > 0
                    push!(l_N_time, sqrt(tke_int) / N)
                    push!(ratio_metric_time, tke_int / (N * K_t_int))
                end
            end

            @info @sprintf("Case r = %.1f: collected %d raw timesteps", r, length(l_N_time))

            # ADAPTIVE FILTERING: Apply to current case data
            if !isempty(l_N_time)
                l_N_filt, ratio_filt, mask_filt, filter_stats = apply_adaptive_filter(
                    l_N_time, ratio_metric_time; low_pct=2.5, high_pct=97.5
                )

                @info @sprintf(
                    "  → Adaptive filter: %d → %d points (removed %d outliers, %.1f%% retained)",
                    filter_stats.n_before, filter_stats.n_after, filter_stats.n_removed,
                    100.0 * filter_stats.n_after / filter_stats.n_before
                )

                # Keyword options for individual timestep scatter points
                sc_kwargs = (
                    color             = ramp_colour(r),
                    markersize        = 2.5,
                    markerstrokewidth = 0,
                    markeralpha       = 0.45,
                    label             = @sprintf("r = %.1f", r)
                )

                # Scatter FILTERED timesteps
                scatter!(plt, l_N_filt, ratio_filt; sc_kwargs...)

                # Accumulate case statistics for overall trend analysis (from filtered data)
                if !isempty(l_N_filt)
                    push!(case_medians_x, med(l_N_filt))
                    push!(case_medians_y, med(ratio_filt))

                    append!(all_l_N, l_N_filt)
                    append!(all_ratio_metric, ratio_filt)
                end
            else
                @warn @sprintf("Case r = %.1f: no valid data collected", r)
            end
        end

        # --- Second-pass filtering: Remove outliers from aggregated case medians ---
        if length(case_medians_y) >= 3
            @info "Applying second-pass filter to case medians..."
            med_lo, med_hi, n_med_valid, n_med_out = compute_percentile_threshold(case_medians_y; low_pct=2.5, high_pct=97.5)
            mask_med = (med_lo .<= case_medians_y .<= med_hi)

            @info @sprintf(
                "  → Case median filter: %d → %d medians retained (removed %d outliers)",
                length(case_medians_y), sum(mask_med), n_med_out
            )

            case_medians_x = case_medians_x[mask_med]
            case_medians_y = case_medians_y[mask_med]
        end

        # Overlay aggregated case medians
        med_kwargs = (
            markersize        = 6,
            markerstrokewidth = 1.5,
            markercolor       = :white,
            markerstrokecolor = :black,
            label             = "case medians"
        )

        scatter!(plt, case_medians_x, case_medians_y; med_kwargs...)

        # --- Dynamic Viewport Range ---
        if !isempty(all_ratio_metric)
            sorted_y = sort(all_ratio_metric)
            ylo = sorted_y[max(1, round(Int, 0.005 * length(sorted_y)))] / 1.5
            yhi = sorted_y[round(Int, 0.999 * length(sorted_y))] * 1.5

            plot!(plt, ylims = (ylo, yhi))
        end

        # --- Plot Formatting ---
        plot!(
            plt,
            xscale    = :log10,
            yscale    = :log10,
            xlabel    = L"l_N = \sqrt{\mathrm{TKE}} / N" * " (m)",
            ylabel    = L"\mathrm{TKE} / (N K_t)",
            minorgrid = true,
            legend    = :bottomright,
            title     = L"\mathrm{TKE} / (N K_t)" * " vs " * L"l_N",
            size      = (950, 650),
            dpi       = 300
        )

        # --- Save Output ---
        mkpath(save_folder)
        savefig(plt, joinpath(save_folder, "TKE_over_NKt.png"))
    end
end