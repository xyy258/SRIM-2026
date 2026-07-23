using Oceananigans, JLD2, NCDatasets, Plots, Printf
using Plots.PlotMeasures

# Import parameters
include("Parameters.jl")

## ====================================================== ##
## Plot of average buoyancy gradient with depth over time ##
## ====================================================== ##

# Set the filename
filename = @sprintf("Ekman/Data/Ekman r=%.1f average buoyancy gradient",r)

db_dz_timeseries = FieldTimeSeries(filename * ".jld2", "db_dz")

# Extract the grid nodes (zb will contain the vertical grid levels)
xb, yb, zb = nodes(db_dz_timeseries)

## Open the file to extract the time array
file_xz    = jldopen(filename * ".jld2")
iterations = parse.(Int, keys(file_xz["timeseries/t"]))

# Extract the actual simulation times
t_save = [file_xz["timeseries/t/$i"] for i in iterations]
close(file_xz)

# Extract the data slice into a 2D matrix [Nz, Nt]
Nz = length(zb)
Nt = length(iterations)
gradient_data = zeros(Nz, Nt)

for (t_idx, iter) in enumerate(iterations)
    gradient_data[:, t_idx] = db_dz_timeseries[t_idx].data[1, 1, 1:Nz]
end

# Reduce range of z
zbconcat = zb[findall(<(0.5*δ),zb)]
Nzconcat = length(zbconcat)

heatmap(t_save*f₀, zbconcat/δ, gradient_data[1:Nzconcat, :]/N²,
        xlabel = "tf",
        ylabel = "Height z/δ",
        title  = @sprintf("(∂b/∂z)/N² for N/f = %.1f",r),
        size   = (1000,400),
        margin = 10px,
        color  = :thermal) # :thermal is great for highlighting intensifying gradients
savefig(@sprintf("Ekman/3D Simulation/Plots/Buoyancy gradient plot r = %.1f.png",r))

## ======================================= ##
##  Horizontally averaged buoyancy profile ##
## ======================================= ##

filename = @sprintf("Ekman/Data/Ekman r=%.1f average buoyancy",r)
b_avg_timeseries = FieldTimeSeries(filename * ".jld2", "b")

# Extract grid coordinates using znodes
zb = znodes(b_avg_timeseries.grid, Center())

# Get initial and final profiles
b_initial = vec(interior(b_avg_timeseries[1], 1, 1, :))    # First saved time step
b_final   = vec(interior(b_avg_timeseries[end], 1, 1, :))    # Last time step

# Normalize depth
z_normalized = zb / δ

# Create mask for the boundary layer region
z_mask = findall(<(0.5*δ), zb)
# Otherwise, use the following for full domain plot
# z_mask = 1:length(zb)

b_plot_final   = b_final[z_mask]
b_plot_initial = b_initial[z_mask]
z_plot = z_normalized[z_mask]


# Plot
plot(b_plot_initial/N², z_plot,
     xlabel     = "b/N²",
     ylabel     = "Height z/δ",
     title      = @sprintf("<b> profile for N/f = %.1f", r),
     linewidth  = 2,
     label      = "Initial",
     linestyle  = :dash,
     legend     = :bottomright,
     size       = (800,400),
     margin     = 10px)

plot!(b_plot_final/N², z_plot,
      linewidth = 2,
      label     = "Final")

savefig(@sprintf("Ekman/3D Simulation/Plots/Averaged buoyancy profile r = %.1f.png",r))

## =============================================== ##
## Horizontally averaged buoyancy gradient profile ##
## =============================================== ##

filename = @sprintf("Ekman/Data/Ekman r=%.1f average buoyancy gradient",r)
db_dz_avg_timeseries = FieldTimeSeries(filename * ".jld2", "db_dz")

# Extract grid coordinates using znodes
zb = znodes(db_dz_timeseries.grid, Center())

# Get initial and final profiles
db_dz_initial = vec(interior(db_dz_avg_timeseries[1], 1, 1, :))    # First saved time step
db_dz_final   = vec(interior(db_dz_avg_timeseries[end], 1, 1, :))    # Last time step

# Normalize depth
z_normalized = zb / δ

# Create mask for the boundary layer region
z_mask = findall(<(0.5*δ), zb)
# Otherwise, use the following for full domain plot
# z_mask = 1:length(zb)

db_dz_plot_initial = db_dz_initial[z_mask]
db_dz_plot_final   = db_dz_final[z_mask]
z_plot = z_normalized[z_mask]


# Plot
plot(db_dz_plot_initial/N², z_plot,
     xlabel    = "∂b/∂z/N²",
     ylabel    = "Height z/δ",
     title     = @sprintf("∂<b>/∂z Profile for N/f = %.1f", r),
     linewidth = 2,
     label     = "Initial",
     linestyle = :dash,
     legend    = :bottomright,
     size      = (800,400),
     margin    = 10px)

plot!(db_dz_plot_final/N², z_plot,
      linewidth = 2,
      label = "Final")

savefig(@sprintf("Ekman/3D Simulation/Plots/Averaged buoyancy gradient profile r = %.1f.png",r))

## ================== ##
##   Hodograph plot   ##
## ================== ##

u_series = FieldTimeSeries(@sprintf("Ekman/Data/Ekman r=%.1f average velocity",r), "u_avg")
v_series = FieldTimeSeries(@sprintf("Ekman/Data/Ekman r=%.1f average velocity",r), "v_avg")

xu, yu, zu = nodes(u_series)
zC = znodes(u_series.grid, Center())

u_profile = vec(interior(u_series[end], 1, 1, :))
v_profile = vec(interior(v_series[end], 1, 1, :))

slice = 1:length(zC)
u_slice = u_profile[slice]
v_slice = v_profile[slice]
z_slice = zC[slice] / δ

plot(u_slice/U∞, v_slice/U∞,
    linewidth      = 2,
    line_z         = z_slice,       # Colour line based on z
    color          = :viridis,      # Colour for the line/markers
    marker         = :circle,
    markersize     = 2,             # Smaller marker
    marker_z       = z_slice,       # Colours markers based on z
    xlabel         = "<u>/U∞",
    ylabel         = "<v>/U∞",
    colorbar_title = "Height z/δ",  # Adds a label to colour bar
    colorbar       = true,
    size           = (1000,500),
    margin         = 10px,
    legend         = false,
    title          = @sprintf("Ekman Hodograph r = N/f = %.1f",r)
)
savefig(@sprintf("Ekman/3D Simulation/Plots/Hodograph r = %.1f.png",r))