using Oceananigans, JLD2, NCDatasets, Plots, Printf

## Plot of average buoyancy gradient with depth over time ##

# Set the filename
filename = "Ekman/Data/Average buoyancy gradient"

db_dz_timeseries = FieldTimeSeries(filename * ".jld2", "db_dz")

# 2. Extract the grid nodes (zb will contain the vertical grid levels)
xb, yb, zb = nodes(db_dz_timeseries)

## Open the file to extract the time array
file_xz = jldopen(filename * ".jld2")
iterations = parse.(Int, keys(file_xz["timeseries/t"]))

# Extract the actual simulation times
t_save = [file_xz["timeseries/t/$i"] for i in iterations]
close(file_xz)

# 3. Extract the data slice into a 2D matrix [Nz, Nt]
Nz = length(zb)
Nt = length(iterations)
gradient_data = zeros(Nz, Nt)

for (t_idx, iter) in enumerate(iterations)
    gradient_data[:, t_idx] = db_dz_timeseries[t_idx].data[1, 1, 1:Nz]
end

zbconcat = zb[findall(<(0.4*δ),zb)]
Nzconcat = length(zbconcat)

heatmap(t_save*f₀, zbconcat/δ, gradient_data[1:Nzconcat, :]/N²,
        xlabel="tf",
        ylabel="Height z/δ",
        title=@sprintf("(∂b/∂z)/N² for N/f = %.1f",r),
        color=:thermal) # :thermal is great for highlighting intensifying gradients
savefig(@sprintf("Ekman/3D Simulation/Buoyancy gradient plot r = %.1f.png",r))


## Horizontally averaged buoyancy profile

filename = "Ekman/Data/Average buoyancy"
b_avg_timeseries = FieldTimeSeries(filename * ".jld2", "b")

# Extract grid coordinates using znodes
zb = znodes(b_avg_timeseries.grid, Center())

# Get initial and final profiles
b_initial = vec(interior(b_avg_timeseries[1], 1, 1, :))    # First saved time step
b_final = vec(interior(b_avg_timeseries[end], 1, 1, :))    # Last time step

# Normalize depth
z_normalized = zb / δ

# Create mask for the boundary layer region
z_mask = findall(<(0.4*δ), zb)
b_plot_final = b_final[z_mask]
b_plot_initial = b_initial[z_mask]
z_plot = z_normalized[z_mask]

# Plot
plot(b_plot_initial/N², z_plot,
     xlabel = "b/N²",
     ylabel = "Height z/δ",
     title = @sprintf("Horizontally Averaged Buoyancy Profile N/f = %.1f", r),
     linewidth = 2,
     label = "Initial",
     linestyle = :dash,
     legend = :bottomright)

plot!(b_plot_final/N², z_plot,
      linewidth = 2,
      label = "Final")

savefig(@sprintf("Ekman/3D Simulation/Averaged buoyancy profile r = %.1f.png",r))


## Hodograph plot ##

u_series = FieldTimeSeries("Ekman/Data/Average velocity.jld2", "u_avg")
v_series = FieldTimeSeries("Ekman/Data/Average velocity.jld2", "v_avg")

xu, yu, zu = nodes(u_series)
zC = znodes(u_series.grid, Center())

u_profile = vec(interior(u_series[end], 1, 1, :))
v_profile = vec(interior(v_series[end], 1, 1, :))

slice = 1:length(zC)
u_slice = u_profile[slice]
v_slice = v_profile[slice]
z_slice = zC[slice] / δ

plot(u_slice/U∞, v_slice/U∞,
    linewidth = 2,
    line_z = z_slice,          # Colors the line based on depth z
    color = :viridis,          # Colormap for the line/markers
    marker = :circle,
    markersize = 3,            # Smaller markers (default is usually 4 or 5)
    marker_z = z_slice,        # Colors the markers based on depth z
    xlabel = "<u>/U∞",
    ylabel = "<v>/U∞",
    colorbar_title = "Height z/δ", # Adds a label to the colorbar
    colorbar = true,
    size = (800,600),
    legend = false,
    title = @sprintf("Ekman Hodograph r = N/f = %.1f",r)
)
savefig(@sprintf("Ekman/3D Simulation/Hodograph r = %.1f.png",r))