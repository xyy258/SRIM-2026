ENV["GKSwstype"] = "100"

using Oceananigans, JLD2, Plots, Printf
using Plots.PlotMeasures

# Import parameters
include("Parameters.jl")
# Sets the following:
# Root data file name:    "root"
# Folder to be saved in:  "save_folder"
include("Filename_plot.jl")


#  ======================================================  #
## Plot of average buoyancy gradient with depth over time ##
#  ======================================================  #

# Set the filename
filename = root * "Avg_grad_b"

db_dz_timeseries = FieldTimeSeries(filename * ".jld2", "db_dz")

# Extract the grid nodes (zb will contain the vertical grid levels)
xb, yb, zb = nodes(db_dz_timeseries)

## Open the file to extract the time array
file_xz    = jldopen(filename * ".jld2")
iterations = parse.(Int, keys(file_xz["timeseries/t"]))

# Extract the actual simulation times
t_save = [file_xz["timeseries/t/$i"] for i in iterations]
close(file_xz)

# Extract the data slice into a 2D matrix [nz, nt]
const nz = length(zb)
const nt = length(iterations)
gradient_data = zeros(nz,nt)

for (t_idx, iter) in enumerate(iterations)
    gradient_data[:, t_idx] = db_dz_timeseries[t_idx].data[1, 1, 1:nz]
end

# Reduce range of z
zbconcat = zb[findall(<(Lz),zb)]
Nzconcat = length(zbconcat)

bscale = (r==0 || isnothing(r)) ? 1 : N²

@info "Plot of average buoyancy gradient heatmap with depth over time..."

if (r==0 || isnothing(r)) == true
    plot_title = @sprintf("(∂b/∂z) for N/f = %.1f",r)
else
    plot_title = @sprintf("(∂b/∂z)/N² for N/f = %.1f",r)
end

heatmap(t_save*f₀, zbconcat, gradient_data[1:Nzconcat, :]/bscale,
        xlabel = "tf",
        ylabel = "Height z",
        title  = plot_title,
        size   = (1000,400),
        margin = 25px,
        color  = :thermal) # :thermal is great for highlighting intensifying gradients
mkpath(save_folder*"Buoyancy gradient plot")
savefig(save_folder*@sprintf("Buoyancy gradient plot/r = %.1f.png",r))


#  =======================================  #
##  Horizontally averaged buoyancy profile ##
#  =======================================  #

filename = root * "Avg_b"
b_avg_timeseries = FieldTimeSeries(filename * ".jld2", "b")

# Extract grid coordinates using znodes
zb = znodes(b_avg_timeseries.grid, Center())

# Get initial and final profiles
b_initial = vec(interior(b_avg_timeseries[1], 1, 1, :))    # First saved time step
b_final   = vec(interior(b_avg_timeseries[end], 1, 1, :))    # Last time step

# Create mask for the boundary layer region
z_mask = findall(<(Lz),zb)
# Otherwise, use the following for full domain plot
# z_mask = 1:length(zb)

b_plot_final   = b_final[z_mask]
b_plot_initial = b_initial[z_mask]
z_plot = zb[z_mask]

@info "Plot of average buoyancy profile..."

if (r==0 || isnothing(r)) == true
    Xlabel = "b"
else
    Xlabel = "b/N²"
end

# Plot
plot(b_plot_initial/bscale, z_plot,
     xlabel     = Xlabel,
     ylabel     = "Height z",
     title      = @sprintf("<b> profile for N/f = %.1f", r),
     linewidth  = 2,
     label      = "Initial",
     linestyle  = :dash,
     legend     = :bottomright,
     size       = (800,400),
     margin     = 25px)

plot!(b_plot_final/bscale, z_plot,
      linewidth = 2,
      label     = "Final")
mkpath(save_folder*"Averaged buoyancy profile")
savefig(save_folder*@sprintf("Averaged buoyancy profile/r = %.1f.png",r))

#  ===============================================  #
## Horizontally averaged buoyancy gradient profile ##
#  ===============================================  #

filename = root * "Avg_grad_b"
db_dz_avg_timeseries = FieldTimeSeries(filename * ".jld2", "db_dz")

# Extract grid coordinates using znodes
zb = znodes(db_dz_timeseries.grid, Center())

# Get initial and final profiles
db_dz_initial = vec(interior(db_dz_avg_timeseries[1], 1, 1, :))    # First saved time step
db_dz_final   = vec(interior(db_dz_avg_timeseries[end], 1, 1, :))    # Last time step

# Create mask for the boundary layer region
z_mask = findall(<(Lz),zb)
# Otherwise, use the following for full domain plot
# z_mask = 1:length(zb)

db_dz_plot_initial = db_dz_initial[z_mask]
db_dz_plot_final   = db_dz_final[z_mask]
z_plot = zb[z_mask]

@info "Plot of average buoyancy gradient profile..."

if (r==0 || isnothing(r)) == true
    Xlabel = "∂b/∂z"
else
    Xlabel = "(∂b/∂z)/N²"
end

# Plot
plot(db_dz_plot_initial/bscale, z_plot,
     xlabel    = "(∂b/∂z)/N²",
     ylabel    = "Height z",
     title     = @sprintf("∂<b>/∂z Profile for N/f = %.1f", r),
     linewidth = 2,
     label     = "Initial",
     linestyle = :dash,
     legend    = :bottomright,
     size      = (800,600),
     margin    = 25px)

plot!(db_dz_plot_final/bscale, z_plot,
      linewidth = 2,
      label = "Final")

mkpath(save_folder*"Averaged buoyancy gradient profile")
savefig(save_folder*@sprintf("Averaged buoyancy gradient profile/r = %.1f.png",r))


#  ==================  #
##   Hodograph plot   ##
#  ==================  #

u_series = FieldTimeSeries(root * "Avg_vel.jld2", "u_avg")
v_series = FieldTimeSeries(root * "Avg_vel.jld2", "v_avg")

xu, yu, zu = nodes(u_series)
zC = znodes(u_series.grid, Center())

# Time averaged average velocity profiles over two inertial periods
T_f = 2π / f₀  # Inertial period
n_periods = 5
t_end = u_series.times[end]
t_indices = findall(t -> t >= t_end - n_periods*T_f, u_series.times)

u_profile = vec(sum([interior(u_series[n], 1, 1, :) for n in t_indices])./length(t_indices))
v_profile = vec(sum([interior(v_series[n], 1, 1, :) for n in t_indices])./length(t_indices))

# Looking at a slice of domain
slice = 1:length(zC)
u_slice = u_profile[slice]
v_slice = v_profile[slice]
z_slice = zC[slice]

@info "Plot of hodograph..."

plot(u_slice/U∞, v_slice/U∞,
    linewidth      = 2,
    line_z         = z_slice,       # Colour line based on z
    color          = :viridis,      # Colour for the line/markers
    marker         = :circle,
    markersize     = 2,             # Smaller marker
    marker_z       = z_slice,       # Colours markers based on z
    xlabel         = "<u>/U∞",
    ylabel         = "<v>/U∞",
    colorbar_title = "Height z",    # Adds a label to colour bar
    colorbar       = true,
    size           = (1000,500),
    margin         = 25px,
    legend         = false,
    title          = @sprintf("Ekman Hodograph r = N/f = %.1f",r)
)
mkpath(save_folder*"Hodograph")
savefig(save_folder*@sprintf("Hodograph/r = %.1f.png",r))