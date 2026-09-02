ENV["GKSwstype"] = "100"

using Oceananigans, Plots, Printf, JLD2, LaTeXStrings
using Plots.PlotMeasures
using Statistics
using CUDA

# High-DPI plot formatting with full Unicode glyph support
default(dpi = 600, fontfamily = "DejaVu Sans")

n_periods = 5

ratios = [0, 0.5, 1, 2, 5, 10]
values = [5,10,20]
profiles = [4]

# Extended color ramp up to r = 50.0 to make r=10, r=25, and r=50 visually distinct
const RAMP = [
    (log10(0.5),  (27,  78, 143)),  # Navy Blue
    (log10(1.0),  (46, 139,  87)),  # Sea Green
    (log10(2.0),  (200, 150, 30)),  # Ochre / Gold
    (log10(5.0),  (180,  80, 44)),  # Burnt Orange
    (log10(10.0), (110,  25, 120)), # Purple
    (log10(25.0), (205,  45, 115)), # Vivid Magenta / Rose
]

# Generate hex color code interpolated across the defined RAMP array
function ramp_colour(s)
    # Explicitly assign off-black to r = 0 so it doesn't clamp to log10(0.5) Navy Blue
    if s <= 0
        return "#1A1A1A"
    end

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

# Fit logarithmic profile near wall to extract u*
function fit_log_layer(grid, u_avg, v_avg; κ=0.41, n_points=5)
    z = Array(znodes(grid, Center()))[1:n_points]
    U = @. sqrt(u_avg[1:n_points]^2 + v_avg[1:n_points]^2)
    X = [log.(z) ones(n_points)]
    coeff = X \ U
    A = coeff[1]
    return A * κ
end

# ======================================= #
##  Turbulent Kinetic Energy (TKE) Plot  ##
# ======================================= #

for p in profiles
    global profile = p
    for value in values
        plt = plot(size = (800, 500))

        global T = value
        @info("Plot for T=$T...")
        for (idx, ratio) in enumerate(ratios)
            global r = ratio

            include("Parameters.jl")
            include("Filename_plot.jl")

            # Inertial period calculated after parameters are loaded
            T_f = 2π / f₀

            u_series = FieldTimeSeries(root * "Velocity.jld2", "u")
            v_series = FieldTimeSeries(root * "Velocity.jld2", "v")
            w_series = FieldTimeSeries(root * "Velocity.jld2", "w")

            u_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "u_avg")
            v_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "v_avg")
            w_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "w_avg")

            zC = znodes(u_series.grid, Center())
            center_w = a -> 0.5 .* (a[:, :, 1:end-1] .+ a[:, :, 2:end])
            center_w_profile = a -> 0.5 .* (a[1:end-1] .+ a[2:end])

            t_end = u_series.times[end]
            t_indices = findall(t -> t >= t_end - n_periods * T_f, u_series.times)

            tke_profile_avg = zeros(length(zC))
            u_avg_time_mean = zeros(length(zC))
            v_avg_time_mean = zeros(length(zC))

            for n in t_indices
                u = Array(interior(u_series[n], :, :, :))
                v = Array(interior(v_series[n], :, :, :))
                w = center_w(Array(interior(w_series[n], :, :, :)))

                u_mean = Array(interior(u_avg_series[n], 1, 1, :))
                v_mean = Array(interior(v_avg_series[n], 1, 1, :))
                w_mean = center_w_profile(Array(interior(w_avg_series[n], 1, 1, :)))

                u_avg_time_mean .+= u_mean
                v_avg_time_mean .+= v_mean

                u_prime = u .- reshape(u_mean, (1, 1, :))
                v_prime = v .- reshape(v_mean, (1, 1, :))
                w_prime = w .- reshape(w_mean, (1, 1, :))

                tke_inst = 0.5 .* (u_prime.^2 .+ v_prime.^2 .+ w_prime.^2)
                tke_profile = vec(mean(tke_inst, dims=(1, 2)))
                tke_profile_avg .+= tke_profile
            end

            tke_profile_avg ./= length(t_indices)
            u_avg_time_mean ./= length(t_indices)
            v_avg_time_mean ./= length(t_indices)

            tke_norm = tke_profile_avg / U∞^2

            # Compute u_star and normalization lengthscale h₀
            κ_val = @isdefined(κ) ? κ : 0.41
            u_star_fit = fit_log_layer(u_series.grid, u_avg_time_mean, v_avg_time_mean; κ=κ_val)
            h₀ = 0.4 * u_star_fit / f₀
            z_norm = zC ./ h₀

            @info "Making time-averaged TKE plot for r = $r..."

            # Plotted against normalized height z_norm
            plot!(plt, z_norm, tke_norm,
                yaxis     = :log,
                linewidth = 2,
                color     = ramp_colour(r),
                label     = @sprintf("r = %.1f", r)
            )

            # Fit points where z <= h₀, then extrapolate across full domain
            mask = zC .<= h₀
            if count(mask) >= 2
                x_sub = tke_norm[mask]
                z_sub = z_norm[mask]

                # Fit log10(TKE/U∞²) = m * (z/h₀) + c
                A = [z_sub ones(length(z_sub))]
                m, c = A \ log10.(x_sub)

                # Evaluate over full domain z_norm
                x_full = 10 .^ (m .* z_norm .+ c)

                plot!(plt, z_norm, x_full,
                    linestyle = :dash,
                    linewidth = 1.5,
                    color     = ramp_colour(r),
                    label     = @sprintf("Fit r = %.1f (grad = %.2e)", r, m)
                )
            end
        end

        plot!(plt,
            xlabel    = "Normalized Height z / h₀",
            ylabel    = "TKE/U∞^2",
            minorgrid = true,
            legend    = :bottomleft,
            title     = "TKE (averaged over $n_periods periods) against " * L"$z/h_0$" * " for T=$T",
            margin    = 25px
        )

        mkpath(save_folder)
        savefig(plt, save_folder * "TKE Plots T=$T.png")
    end
end