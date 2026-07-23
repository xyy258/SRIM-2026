using Oceananigans, JLD2, Plots, Printf
using Plots.PlotMeasures # using units for borders
# using CairoMakie
# using NCDatasets

# Import parameters
include("Parameters.jl")

##============================ ##
## Buoyancy gradient animation ##
##============================ ##

# Set the filename (without the extension)
filename = @sprintf("Ekman/Data/Ekman r=%.1f",r)

# Read in the first iteration.  We do this to load the grid
# filename * ".jld2" concatenates the extension to the end of the filename
b_ic = FieldTimeSeries(filename * "_b.jld2", "b")

## Load in coordinate arrays
## We do this separately for each variable since Oceananigans uses a staggered grid
xb, yb, zb = nodes(b_ic)

## Now, open the file with our data
file_vel = jldopen(filename * "_velocity.jld2")
file_b   = jldopen(filename * "_b.jld2")

## Extract a vector of iterations
iterations = parse.(Int, keys(file_vel["timeseries/t"]))

@info "Making an animation from saved data..."

t_save = zeros(length(iterations))

# Masking for (0,Lz)
Lzmask  = zb[findall(x -> x < Lz, zb)]
NLzmask = length(Lzmask)

# Masking for certain height from bottom of domain
zbmask = zb[findall(x -> x < 0.5*δ, zb)]
Nzmask = length(zbmask)

# Fixing colour limits for buoyancy change
clim_abs = maximum(
    maximum(abs, (file_b["timeseries/b/$iter"][:, 1, 1:Nzmask]' .- N² * zbmask) / N²)
    for iter in iterations
)

# Here, we loop over all iterations
anim = @animate for (i, iter) in enumerate(iterations)

    @info "Drawing frame $i / $(length(iterations))..."
    b_xz = file_b["timeseries/b/$iter"][:, 1, 1:NLzmask];

    t = file_vel["timeseries/t/$iter"];
    t_save[i] = t # save the time

    b_xz_plot = heatmap(xb, Lzmask/δ, b_xz'/N²;
        color = :thermal,
        clims=(0, 1.05).*maximum(abs, b_xz'/N²),
        xlabel = "x", ylabel = "z/δ",
        xlims = (0, Lx), ylims = (0,Lz/δ)); # Shows entire height of domain

    b_diff_xz_plot = heatmap(xb, zbmask/δ, (b_xz[:,1:Nzmask]' .- N²*zbmask)/N²;
        color = :coolwarm,
        clims = (-clim_abs,clim_abs).*1.05,
        xlabel = "x", ylabel = "z/δ",
        xlims = (0, Lx), ylims = (0,zbmask[end]/δ)); # Shows lower part of domain near the rigid boundary

    b_title = @sprintf("b/N² at t = %s, N/f = %.1f", round(t), r);
    b_diff_title = @sprintf("(b-N²z)/N² at t = %s, N/f = %.1f", round(t), r);

# Combine the sub-plots into a single figure
    plot(b_xz_plot, b_diff_xz_plot,
        layout = (2, 1),
        size = (1000, 500),
        title = [b_title b_diff_title],
        margin = 10px)

    if iter == iterations[end]
        close(file_vel)
        close(file_b)
    end
end

# Save the animation to a file
mp4(anim, @sprintf("Ekman/3D Simulation/Animations/Ekman Plot r = %.1f.mp4", r), fps = 20) # hide

## ========================== ##
## Average velocity animation ##
## ========================== ##

# Import parameters
include("Parameters.jl")

# Set the filename path prefix
filename = @sprintf("Ekman/Data/Ekman r=%.1f", r)

# Load FieldTimeSeries directly
u_avg_series = FieldTimeSeries(filename * " average velocity.jld2", "u_avg")

# Extract simulation times and interior vertical nodes (strips halos, matching length 180)
times = u_avg_series.times
zu    = znodes(u_avg_series[1])

@info "Making animation of plane-averaged velocity profiles..."

# Generate the animation
anim = @animate for i in 1:length(times)

    t = times[i]
    @info "Drawing frame $i / $(length(times)) at sim time t = $(round(t, digits=1))..."

    # Extract 1D interior velocity vectors (guaranteed length 180)
    u_prof = vec(interior(u_avg_series[i], 1, 1, :))

    # Plot u_avg profile normalized by U∞
    p = plot(u_prof / U∞, zu / δ,
             linewidth = 3,
             color     = :navy,
             xlabel    = "<u>/U∞",
             ylabel    = "Height z / δ",
             xlims     = (-0.5, 1.5),
             ylims     = (0, Lz / δ),
             legend    = :bottomright,
             grid      = true,
             size      = (750, 500),
             margin    = 10px)

    # Dynamic title
    title!(p, @sprintf("Plane-Averaged Velocity Profile (N/f = %.1f) | t = %.1f", r, t))
end

mp4(anim, @sprintf("Ekman/3D Simulation/Animations/Ekman Velocity Plot r = %.1f.mp4", r), fps = 30)


## ==============================
## Average vorticity animation ##
## ==============================

# Set the filename path prefix
filename = @sprintf("Ekman/Data/Ekman r=%.1f", r)

@info "Loading vorticity time series..."
vort_file = filename * " average vorticity.jld2"

ωx_avg_series = FieldTimeSeries(vort_file, "ωx_avg")
ωy_avg_series = FieldTimeSeries(vort_file, "ωy_avg")
ωz_avg_series = FieldTimeSeries(vort_file, "ωz_avg")

# Extract simulation times
vort_times = ωx_avg_series.times

# Extract interior z-nodes separately to account for grid staggering
zx = znodes(ωx_avg_series[1])
zy = znodes(ωy_avg_series[1])
zz = znodes(ωz_avg_series[1])

@info "Making animation of plane-averaged vorticity profiles..."

# Generate the animation
anim_vort = @animate for i in 1:length(vort_times)

    t = vort_times[i]
    @info "Drawing vorticity frame $i / $(length(vort_times)) at sim time t = $(round(t, digits=1))..."

    # Extract 1D interior vorticity vectors (stripping halo cells)
    ωx_prof = vec(interior(ωx_avg_series[i], 1, 1, :))
    ωy_prof = vec(interior(ωy_avg_series[i], 1, 1, :))
    ωz_prof = vec(interior(ωz_avg_series[i], 1, 1, :))

    # Panel 1: ωx profile
    p_x = plot(ωx_prof / f₀, zx / δ,
               linewidth = 2,
               color     = :crimson,
               xlabel    = "<ωx> / f₀",
               ylabel    = "Height z / δ",
               ylims     = (0, Lz / δ),
               grid      = true)

    # Panel 2: ωy profile
    p_y = plot(ωy_prof / f₀, zy / δ,
               linewidth = 2,
               color     = :teal,
               xlabel    = "<ωy> / f₀",
               ylabel    = "Height z / δ",
               ylims     = (0, Lz / δ),
               grid      = true)

    # # Panel 3: ωz profile
    # p_z = plot(ωz_prof / f₀, zz / δ,
    #            linewidth = 2,
    #            color     = :darkorange,
    #            xlabel    = "<ωz> / f₀",
    #            ylabel    = "Height z / δ",
    #            ylims     = (0, Lz / δ),
    #            grid      = true)

    # Combine into side-by-side stacked layout
    plot(p_x, p_y,
         layout     = (1, 2),
         size       = (1000, 600),
         margin     = 10px,
         plot_title = @sprintf("Plane-Averaged Vorticity Profiles (N/f = %.1f) | t = %.1f", r, t))
end

mp4(anim_vort, @sprintf("Ekman/3D Simulation/Animations/Ekman Vorticity Plot r = %.1f.mp4", r), fps = 30)