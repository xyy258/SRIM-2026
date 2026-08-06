ENV["GKSwstype"] = "100"

using Oceananigans, JLD2, Plots, Printf
using Plots.PlotMeasures

# Import parameters
include("Parameters.jl")
# Sets the following:
# - save_folder: where combined plots are written
include("Plot_filename.jl")

# Choose the list of r = N/f values to compare.
# Keep the same value of T or L for each profile.
r_values = [5.0, 10.0, 31.6, 75.0]
# colours = [:navy, :crimson, :forestgreen, :darkorange]
colours = ["#A8CBEC", "#6BA3DE", "#3C7CC4", "#1F559B", "#0B3164"]
linestyles = [:solid, :dash, :dot, :dashdot]

if profile == 0
    profile_label = "linear"
elseif profile == 1
    profile_label = @sprintf("nonlinear, T = %.2f", T)
elseif profile == 2
    profile_label = @sprintf("exponential, L = %.1fLz", Lᴰ)
elseif profile == 3
    profile_label = @sprintf("linear + exponential decay, L = %.1fLz", Lᴰ)
elseif profile == 4
    profile_label = @sprintf("softplus, T = %.2f", T)
else
    profile_label = @sprintf("profile %d, T = %.2f", profile, T)
end


function load_profile_data(r_val)
    root = root_path(r_val)

    b_file = FieldTimeSeries(root * "Avg_b.jld2", "b")
    db_dz_file = FieldTimeSeries(root * "Avg_grad_b.jld2", "db_dz")

    zb = znodes(b_file.grid, Center())
    b_profile = vec(interior(b_file[end], 1, 1, :))
    db_dz_profile = vec(interior(db_dz_file[end], 1, 1, :))

    return zb, b_profile, db_dz_profile
end

@info "Creating combined buoyancy comparison plots for $profile_label..."

p1 = plot(
    xlabel = "b / N²",
    ylabel = "Height z",
    title = @sprintf("Horizontally averaged buoyancy profiles | %s", profile_label),
    legend = :bottomright,
    size = (900, 500),
    margin = 25px,
    grid = true
)

p2 = plot(
    xlabel = "(∂b/∂z) / N²",
    ylabel = "Height z",
    title = @sprintf("Horizontally averaged buoyancy gradient profiles | %s", profile_label),
    legend = :bottomright,
    size = (900, 500),
    margin = 25px,
    grid = true
)

for (i, r_val) in enumerate(r_values)
    zb, b_profile, db_dz_profile = load_profile_data(r_val)
    label = @sprintf("N/f = %.1f", r_val)

    plot!(p1, b_profile ./ N²,
          zb,
          label = label,
          linewidth = 2,
          color = colours[mod1(i, length(colours))],
          linestyle = linestyles[mod1(i, length(linestyles))])

    plot!(p2, db_dz_profile ./ N²,
          zb,
          label = label,
          linewidth = 2,
          color = colours[mod1(i, length(colours))],
          linestyle = linestyles[mod1(i, length(linestyles))])
end

combined_plot = plot(p1, p2,
                     layout = (1, 2),
                     size = (1800, 600),
                     margin = 25px,
                     title = @sprintf("Combined averaged buoyancy and gradient profiles for %s", profile_label))

savefig(combined_plot, save_folder * @sprintf("Combined buoyancy profiles r comparison %s.png", replace(profile_label, ' ' => '_')))

@info "Saved combined plots to $save_folder"
