ENV["GKSwstype"] = "100"

using Oceananigans, Plots, Printf, JLD2, Statistics, LaTeXStrings
using Plots.PlotMeasures

# Define parameter ranges for sweep
ratios   = [0.5, 1, 2, 5, 10, 25, 50]
values   = [5, 10, 20]
profiles = [4]

# --- Reference Style Color Ramp ---
const RAMP = [(log10(0.5),  ( 27,  78, 143)),
              (log10(1.0),  ( 46, 139,  87)),
              (log10(2.0),  (200, 150,  30)),
              (log10(5.0),  (180,  80,  44)),
              (log10(10.0), ( 75,  16,  96))]

function ramp_colour(s)
    x = clamp(log10(s), RAMP[1][1], RAMP[end][1])
    for i in 1:length(RAMP)-1
        (x0, c0), (x1, c1) = RAMP[i], RAMP[i+1]
        x <= x1 || continue
        f = x1 == x0 ? 0.0 : (x - x0) / (x1 - x0)
        chan(k) = clamp(round(Int, c0[k] + f * (c1[k] - c0[k])), 0, 255)
        return "#" * join(string(chan(k), base = 16, pad = 2) for k in 1:3)
    end
    return "#000000"
end

med(v) = median(filter(isfinite, v))

# --- Sampling Parameters ---
const avg_len = 0.1         # Vertical physical window centered at interface peak [m]
const t_step  = 4

# ========================================================================== #
## Interfacial Length Scale Scatter Plot Across Sampled Timesteps (Log-Log) ##
# ========================================================================== #

for p in profiles
    global profile = p
    for value in values
        global T = value
        @info @sprintf("Generating length scale plots for T = %d...", T)

        # Create two subplots: plt1 for buoyancy scale (l_N) and plt2 for shear scale (L_s)
        plt1 = plot(margin = 15px)
        plt2 = plot(margin = 15px)

        case_medians_x   = Float64[]
        case_medians_x_s = Float64[]
        case_medians_y   = Float64[]

        all_l_N     = Float64[]
        all_l_S     = Float64[]
        all_l_kappa = Float64[]

        for (idx, ratio) in enumerate(ratios)
            global r = ratio

            include("Parameters.jl")
            include("Filename_plot.jl")

            u_series = FieldTimeSeries(root * "Velocity.jld2", "u")
            v_series = FieldTimeSeries(root * "Velocity.jld2", "v")
            w_series = FieldTimeSeries(root * "Velocity.jld2", "w")
            b_series = FieldTimeSeries(root * "Buoyancy.jld2", "b")

            u_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "u_avg")
            v_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "v_avg")
            w_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "w_avg")
            db_dz_series = FieldTimeSeries(root * "Avg_grad_b.jld2", "db_dz")

            zC = znodes(u_series.grid, Center())
            center_w = a -> 0.5 .* (a[:, :, 1:end-1] .+ a[:, :, 2:end])
            center_w_profile = a -> 0.5 .* (a[1:end-1] .+ a[2:end])

            # Downsample timesteps over the last 4 inertial periods
            T_f = 2π / f₀
            n_periods = 4
            t_indices = findall(t -> t >= u_series.times[end] - n_periods * T_f, u_series.times)[1:t_step:end]

            l_N_time     = Float64[]
            l_S_time     = Float64[]
            l_kappa_time = Float64[]

            for n in t_indices
                u = Array(interior(u_series[n], :, :, :))
                v = Array(interior(v_series[n], :, :, :))
                w = center_w(Array(interior(w_series[n], :, :, :)))
                b = Array(interior(b_series[n], :, :, :))

                u_mean = Array(interior(u_avg_series[n], 1, 1, :))
                v_mean = Array(interior(v_avg_series[n], 1, 1, :))
                w_mean = center_w_profile(Array(interior(w_avg_series[n], 1, 1, :)))

                nz = min(size(w, 3), size(b, 3), length(w_mean))

                u_prime = u[:, :, 1:nz] .- reshape(u_mean[1:nz], (1, 1, nz))
                v_prime = v[:, :, 1:nz] .- reshape(v_mean[1:nz], (1, 1, nz))
                w_prime = w[:, :, 1:nz] .- reshape(w_mean[1:nz], (1, 1, nz))
                b_prime = b[:, :, 1:nz] .- reshape(mean(b[:, :, 1:nz], dims=(1,2)), (1, 1, nz))

                tke_inst   = vec(mean(0.5 .* (u_prime.^2 .+ v_prime.^2 .+ w_prime.^2), dims=(1, 2)))
                wb_inst    = vec(mean(w_prime .* b_prime, dims=(1, 2)))
                db_dz_inst = Array(interior(db_dz_series[n], 1, 1, :))[1:nz]

                # Mean velocity shear calculation: S = sqrt((du/dz)^2 + (dv/dz)^2)
                du_dz = [ (u_mean[2] - u_mean[1]) / (zC[2] - zC[1]);
                          (u_mean[3:nz] .- u_mean[1:nz-2]) ./ (zC[3:nz] .- zC[1:nz-2]);
                          (u_mean[nz] - u_mean[nz-1]) / (zC[nz] - zC[nz-1]) ]
                dv_dz = [ (v_mean[2] - v_mean[1]) / (zC[2] - zC[1]);
                          (v_mean[3:nz] .- v_mean[1:nz-2]) ./ (zC[3:nz] .- zC[1:nz-2]);
                          (v_mean[nz] - v_mean[nz-1]) / (zC[nz] - zC[nz-1]) ]
                S_inst = sqrt.(du_dz.^2 .+ dv_dz.^2)

                # Identify interface peak depth and sampling window
                idx_peak = argmax(db_dz_inst)
                z_int    = zC[idx_peak]
                rng      = findall(abs.(zC[1:nz] .- z_int) .<= avg_len / 2)
                isempty(rng) && (rng = [idx_peak])

                tke_int = mean(tke_inst[rng])
                K_t_int = -mean(wb_inst[rng]) / (mean(db_dz_inst[rng]) + 1e-10)
                S_int   = mean(S_inst[rng])

                if K_t_int > 0 && tke_int > 1e-8 && S_int > 1e-8
                    push!(l_N_time, sqrt(max(tke_int, 0)) / N)
                    push!(l_S_time, sqrt(max(tke_int, 0)) / S_int)
                    push!(l_kappa_time, K_t_int / sqrt(max(tke_int, 1e-12)))
                end
            end

            @info @sprintf("Plotting %d timesteps for r = %.1f...", length(l_N_time), r)

            scatter!(plt1, l_N_time, l_kappa_time,
                color             = ramp_colour(r),
                markersize        = 2.5,
                markerstrokewidth = 0,
                markeralpha       = 0.45,
                label             = @sprintf("r = %.1f", r)
            )

            scatter!(plt2, l_S_time, l_kappa_time,
                color             = ramp_colour(r),
                markersize        = 2.5,
                markerstrokewidth = 0,
                markeralpha       = 0.45,
                label             = @sprintf("r = %.1f", r)
            )

            if !isempty(l_N_time)
                push!(case_medians_x, med(l_N_time))
                push!(case_medians_x_s, med(l_S_time))
                push!(case_medians_y, med(l_kappa_time))
                append!(all_l_N, l_N_time)
                append!(all_l_S, l_S_time)
                append!(all_l_kappa, l_kappa_time)
            end
        end

        # --- Fits and Reference Lines for l_N ---
        if length(case_medians_x) >= 2
            lo, hi = minimum(all_l_N), maximum(all_l_N)
            plot!(plt1, [lo, hi], [lo, hi], color = :black, linewidth = 1.2, linestyle = :dash, label = L"l_\kappa = l_N" * "  (1:1)")

            best_sse = Inf
            L_fit, x0_fit = 0.5, 1.0
            for L in range(0.3 * maximum(case_medians_y), 1.8 * maximum(case_medians_y), length=150)
                for x0 in range(0.1 * minimum(case_medians_x), 2.5 * maximum(case_medians_x), length=150)
                    pred = L .* (1.0 .- exp.(-case_medians_x ./ x0))
                    all(pred .> 0) || continue
                    sse = sum((log.(case_medians_y) .- log.(pred)).^2)
                    if sse < best_sse
                        best_sse = sse
                        L_fit, x0_fit = L, x0
                    end
                end
            end

            xf = exp10.(range(log10(lo), log10(hi), length=300))
            yf = L_fit .* (1.0 .- exp.(-xf ./ x0_fit))
            plot!(plt1, xf, yf, color = :black, linewidth = 2.5,
                  label = latexstring(@sprintf("l_\\kappa = L_\\infty (1 - e^{-l_N/x_0}), \\ L_\\infty = %.2f", L_fit)) * " m")

            scatter!(plt1, case_medians_x, case_medians_y,
                markersize        = 6,
                markerstrokewidth = 1.5,
                markercolor       = :white,
                markerstrokecolor = :black,
                label             = "case medians"
            )
            hline!(plt1, [L_fit], color = :black, linewidth = 1.0, linestyle = :dot, label = "plateau " * latexstring(@sprintf("L_\\infty = %.2f", L_fit)) * " m")
        end

        # --- Reference 1:1 Line and Medians for L_s ---
        if length(case_medians_x_s) >= 2
            lo_s, hi_s = minimum(all_l_S), maximum(all_l_S)
            plot!(plt2, [lo_s, hi_s], [lo_s, hi_s], color = :black, linewidth = 1.2, linestyle = :dash, label = L"l_\kappa = L_s" * "  (1:1)")
            scatter!(plt2, case_medians_x_s, case_medians_y,
                markersize        = 6,
                markerstrokewidth = 1.5,
                markercolor       = :white,
                markerstrokecolor = :black,
                label             = "case medians"
            )
        end

        # --- Dynamic Viewport Clipping ---
        if !isempty(all_l_kappa)
            sorted_y = sort(all_l_kappa)
            ylo = sorted_y[max(1, round(Int, 0.005 * length(sorted_y)))] / 1.5
            yhi = sorted_y[round(Int, 0.999 * length(sorted_y))] * 1.5
            plot!(plt1, ylims = (ylo, yhi))
            plot!(plt2, ylims = (ylo, yhi))
        end

        # Format Subplot 1 (l_N)
        plot!(plt1,
            xscale    = :log10,
            yscale    = :log10,
            xlabel    = L"l_N = \sqrt{\mathrm{TKE}} / N" * "  (m)",
            ylabel    = L"l_\kappa = K_t / \sqrt{\mathrm{TKE}}" * "  (m)",
            minorgrid = true,
            legend    = :bottomright,
            title     = L"l_\kappa \text{ vs } l_N"
        )

        # Format Subplot 2 (L_s)
        plot!(plt2,
            xscale    = :log10,
            yscale    = :log10,
            xlabel    = L"L_s = \sqrt{\mathrm{TKE}} / \left|\partial \bar{\mathbf{u}} / \partial z\right|" * "  (m)",
            ylabel    = L"l_\kappa = K_t / \sqrt{\mathrm{TKE}}" * "  (m)",
            minorgrid = true,
            legend    = :bottomright,
            title     = L"l_\kappa \text{ vs } L_s"
        )

        # Combine both subplots side-by-side into a single figure
        combined_plt = plot(plt1, plt2, layout = (1, 2), size = (1600, 550), dpi = 600)

        mkpath(save_folder)
        savefig(combined_plt, joinpath(save_folder, "Lengthscales.png"))
    end
end