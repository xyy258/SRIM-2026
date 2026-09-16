ENV["GKSwstype"] = "100"

using Oceananigans, Plots, Printf, JLD2, Statistics, LaTeXStrings
using Plots.PlotMeasures

# High-DPI plot formatting with full Unicode glyph support
default(dpi = 600, fontfamily = "DejaVu Sans")

# ==============================================================================
# Simulation & Parameter Setup
# ==============================================================================

ratios   = [0.5, 1, 2, 5, 10, 25, 50]
values   = [5, 10, 20]
profiles = [4]

const avg_len = 0.1  # Vertical physical window centered at interface peak [m]
const t_step  = 4

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

function compute_percentile_threshold(data::Vector; low_pct::Real = 2.5, high_pct::Real = 97.5)
    finite_data = filter(isfinite, data)
    isempty(finite_data) && return (NaN, NaN, 0, 0)

    lower = quantile(finite_data, low_pct / 100)
    upper = quantile(finite_data, high_pct / 100)

    n_valid = count(lower .<= finite_data .<= upper)
    n_excluded = length(finite_data) - n_valid

    return (lower, upper, n_valid, n_excluded)
end

function apply_adaptive_filter(ratio_metric, ri_metric; low_pct::Real = 2.5, high_pct::Real = 97.5)
    n_total = length(ratio_metric)

    mask_finite = isfinite.(ratio_metric) .& isfinite.(ri_metric) .& (ri_metric .> 0)

    if count(mask_finite) > 0
        r_lo, r_hi, _, _ = compute_percentile_threshold(ratio_metric[mask_finite]; low_pct=low_pct, high_pct=high_pct)
        ri_lo, ri_hi, _, _ = compute_percentile_threshold(ri_metric[mask_finite]; low_pct=low_pct, high_pct=high_pct)

        mask_ratio = (r_lo .<= ratio_metric .<= r_hi) .& (ri_lo .<= ri_metric .<= ri_hi)
    else
        mask_ratio = trues(n_total)
    end

    final_mask = mask_finite .& mask_ratio

    return (
        ratio_metric[final_mask],
        ri_metric[final_mask],
        final_mask
    )
end

med(v) = median(filter(isfinite, v))

center_w(a)    = 0.5 .* (a[:, :, 1:end-1] .+ a[:, :, 2:end])
center_w_1d(a) = 0.5 .* (a[1:end-1] .+ a[2:end])

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

# ==============================================================================
# Main Loop
# ==============================================================================

for p in profiles
    for value in values
        global profile = p
        global T       = value

        @info @sprintf("Generating TKE / (N * K_t) vs Ri plots for T = %d...", T)

        plt = plot(
            left_margin   = 50px,
            bottom_margin = 40px,
            top_margin    = 35px,
            right_margin  = 35px
        )

        all_ratio_metric = Float64[]
        all_ri_metric    = Float64[]

        # Dummy plot entry so "case medians" only shows once in the legend
        scatter!(
            plt, [NaN], [NaN];
            markersize        = 6,
            markerstrokewidth = 1.5,
            markercolor       = :gray,
            markerstrokecolor = :black,
            label             = "Medians"
        )

        for ratio in ratios
            global r = ratio

            include("Parameters.jl")
            include("Filename_plot.jl")

            u_series = FieldTimeSeries(root * "Velocity.jld2", "u"; boundary_conditions = nothing)
            v_series = FieldTimeSeries(root * "Velocity.jld2", "v"; boundary_conditions = nothing)
            w_series = FieldTimeSeries(root * "Velocity.jld2", "w"; boundary_conditions = nothing)
            b_series = FieldTimeSeries(root * "Buoyancy.jld2", "b"; boundary_conditions = nothing)

            u_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "u_avg"; boundary_conditions = nothing)
            v_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "v_avg"; boundary_conditions = nothing)
            w_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "w_avg"; boundary_conditions = nothing)
            db_dz_series = FieldTimeSeries(root * "Avg_grad_b.jld2", "db_dz"; boundary_conditions = nothing)

            zC = znodes(u_series.grid, Center())
            dz = zC[2] - zC[1]

            T_f       = 2π / f₀
            n_periods = 4
            t_min     = u_series.times[end] - n_periods * T_f
            t_indices = findall(t -> t >= t_min, u_series.times)[1:t_step:end]

            ratio_metric_time = Float64[]
            ri_metric_time    = Float64[]

            for n in t_indices
                u = Array(interior(u_series[n], :, :, :))
                v = Array(interior(v_series[n], :, :, :))
                w = center_w(Array(interior(w_series[n], :, :, :)))
                b = Array(interior(b_series[n], :, :, :))

                u_mean = Array(interior(u_avg_series[n], 1, 1, :))
                v_mean = Array(interior(v_avg_series[n], 1, 1, :))
                w_mean = center_w_1d(Array(interior(w_avg_series[n], 1, 1, :)))

                nz    = min(size(w, 3), size(b, 3), length(w_mean))
                z_sub = zC[1:nz]

                u_prime = u[:, :, 1:nz] .- reshape(u_mean[1:nz], 1, 1, nz)
                v_prime = v[:, :, 1:nz] .- reshape(v_mean[1:nz], 1, 1, nz)
                w_prime = w[:, :, 1:nz] .- reshape(w_mean[1:nz], 1, 1, nz)
                b_prime = b[:, :, 1:nz] .- reshape(mean(b[:, :, 1:nz], dims=(1, 2)), 1, 1, nz)

                tke_inst   = vec(mean(0.5 .* (u_prime.^2 .+ v_prime.^2 .+ w_prime.^2), dims=(1, 2)))
                wb_inst    = vec(mean(w_prime .* b_prime, dims=(1, 2)))
                db_dz_inst = Array(interior(db_dz_series[n], 1, 1, :))[1:nz]

                du_dz = diff(u_mean[1:nz]) ./ dz
                dv_dz = diff(v_mean[1:nz]) ./ dz
                shear_sq_1d = du_dz.^2 .+ dv_dz.^2
                shear_sq_inst = [shear_sq_1d; shear_sq_1d[end]]

                tke_peak       = maximum(tke_inst)
                tke_candidates = findall(tke_inst .>= 0.01 * tke_peak)
                idx_peak       = isempty(tke_candidates) ? argmax(tke_inst) : maximum(tke_candidates)

                rng = findall(abs.(z_sub .- z_sub[idx_peak]) .<= (avg_len / 2))
                if isempty(rng)
                    rng = [idx_peak]
                end

                tke_int      = mean(tke_inst[rng])
                db_dz_int    = mean(db_dz_inst[rng])
                K_t_int      = -mean(wb_inst[rng]) / (db_dz_int + 1e-12)
                shear_sq_int = mean(shear_sq_inst[rng])

                ri_int = (db_dz_int / (shear_sq_int + 1e-12))^2

                if K_t_int > 0 && tke_int > 1e-12 && N > 0
                    push!(ratio_metric_time, tke_int / (N * K_t_int))
                    push!(ri_metric_time, ri_int)
                end
            end

            if !isempty(ratio_metric_time)
                ratio_filt, ri_filt, mask_filt = apply_adaptive_filter(
                    ratio_metric_time, ri_metric_time; low_pct=2.5, high_pct=97.5
                )

                c_curr = ramp_colour(r)

                # Plot scatter data points for current r
                sc_kwargs = (
                    color             = c_curr,
                    markersize        = 2.5,
                    markerstrokewidth = 0,
                    markeralpha       = 0.45,
                    label             = @sprintf("r = %.1f", r)
                )

                scatter!(plt, ri_filt, ratio_filt; sc_kwargs...)

                if !isempty(ratio_filt)
                    med_x = med(ri_filt)
                    med_y = med(ratio_filt)

                    # Plot individual median point matching current data colour
                    scatter!(
                        plt, [med_x], [med_y];
                        markersize        = 6,
                        markerstrokewidth = 1.2,
                        markercolor       = c_curr,
                        markerstrokecolor = :black,
                        label             = false
                    )

                    append!(all_ratio_metric, ratio_filt)
                    append!(all_ri_metric, ri_filt)
                end
            end
        end

        # Dynamic Viewport Ranges
        if !isempty(all_ratio_metric) && !isempty(all_ri_metric)
            sorted_y  = sort(all_ratio_metric)
            sorted_ri = sort(all_ri_metric)

            ylo  = sorted_y[max(1, round(Int, 0.005 * length(sorted_y)))] / 1.5
            yhi  = sorted_y[round(Int, 0.999 * length(sorted_y))] * 1.5
            xlo  = sorted_ri[max(1, round(Int, 0.005 * length(sorted_ri)))] / 1.5
            xhi  = sorted_ri[round(Int, 0.999 * length(sorted_ri))] * 1.5

            plot!(plt, xlims = (xlo, xhi), ylims = (ylo, yhi))
        end

        # Plot Formatting
        plot!(
            plt,
            xscale    = :log10,
            yscale    = :log10,
            xlabel    = L"\mathrm{Ri}=N^2/|\partial{\mathbf{u}}/\partial{z}|^2",
            ylabel    = L"\mathrm{TKE} / (N K_t)",
            minorgrid = true,
            legend    = :topright,
            title     = L"\mathrm{TKE} / (N K_t)" * " vs " * L"\mathrm{Ri}",
            size      = (950, 650),
            dpi       = 300
        )

        # Save Output
        mkpath(save_folder)
        savefig(plt, joinpath(save_folder, "TKE_over_NKt_vs_Ri.png"))
    end
end