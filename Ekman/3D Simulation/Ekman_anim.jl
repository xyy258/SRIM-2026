ENV["GKSwstype"] = "100"

using Oceananigans, JLD2, Plots, Printf
using Plots.PlotMeasures # using units for borders
# using NCDatasets

# Import parameters
include("Parameters.jl")
# Sets the following:
# Root data file name:    "root"
# Folder to be saved in:  "save_folder"
include("Filename_anim.jl")

# ===================  #
## Buoyancy animation ##
# ===================  #

# Read in the first iteration.  We do this to load the grid
# filename * ".jld2" concatenates the extension to the end of the filename
b_ic = FieldTimeSeries(root * "Buoyancy.jld2", "b")

## Load in coordinate arrays
## We do this separately for each variable since Oceananigans uses a staggered grid
xb, yb, zb = nodes(b_ic)

## Now, open the file with our data
file_vel = jldopen(root * "Velocity.jld2")
file_b   = jldopen(root * "Buoyancy.jld2")

## Extract a vector of iterations
iterations = parse.(Int, keys(file_vel["timeseries/t"]))
t_save = zeros(length(iterations))

@info "Making animation of buoyancy heatmaps..."

# Masking for (0,Lz)
Lzmask  = zb[findall(x -> x < Lz, zb)]
NLzmask = length(Lzmask)

# Masking for certain height from bottom of domain
z_mask = findall(x -> x < 60,zb)
zbmask = zb[z_mask]
Nzmask = length(zbmask)

bscale = (r==0 || isnothing(r)) ? 1 : N²

# Load initial buoyancy profile
b_initial = file_b["timeseries/b/$(iterations[1])"][:, 1, 1:NLzmask]

# Fixing colour limits for buoyancy difference plot
clim_abs = maximum(
    maximum(abs, (file_b["timeseries/b/$iter"][:, 1, 1:Nzmask] .- b_initial[:,1:Nzmask])' / bscale)
    for iter in iterations
)

anim = @animate for (i, iter) in enumerate(iterations)
    if i % 200 == 0
    @info "Drawing frame $i / $(length(iterations))..."
    end

    b_xz = file_b["timeseries/b/$iter"][:, 1, 1:NLzmask];

    t = file_vel["timeseries/t/$iter"];
    t_save[i] = t # save the time

    b_xz_plot = heatmap(xb, Lzmask, b_xz'/bscale;
        color = :thermal,
        clims = (0, 1.05.*maximum(b_xz'/bscale)),
        xlabel = "x", ylabel = "z",
        xlims = (0, Lx), ylims = (0,Lz)); # Shows entire height of domain

    b_diff_xz_plot = heatmap(xb, zbmask, (b_xz[:,1:Nzmask] .- b_initial[:,1:Nzmask])'/bscale;
        color = :coolwarm,
        clims = (-clim_abs,clim_abs).*1.05,
        xlabel = "x", ylabel = "z",
        xlims = (0, Lx), ylims = (0,zbmask[end]));

    if (r==0 || isnothing(r)) == true
        b_title = @sprintf("b at t = %s, N/f = %.1f", round(t), r);
        b_diff_title = @sprintf("(b-bᵢ) at t = %s, N/f = %.1f", round(t), r);
    else
        b_title = @sprintf("b/N² at t = %s, N/f = %.1f", round(t), r);
        b_diff_title = @sprintf("(b-bᵢ)/N² at t = %s, N/f = %.1f", round(t), r);
    end

# Combine the sub-plots into a single figure
    plot(b_xz_plot, b_diff_xz_plot,
        layout = (2, 1),
        size = (1000, 550),
        title = [b_title b_diff_title],
        margin = 25px)
end
close(file_vel)
close(file_b)

# Save the animation to a file
mkpath(save_folder*"Buoyancy")
mp4(anim, save_folder*@sprintf("Buoyancy/r = %.1f.mp4", r), fps = 30) # hide

#  ==========================  #
## Average velocity animation ##
#  ==========================  #

# Load FieldTimeSeries directly
u_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "u_avg")
v_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "v_avg")

# Extract simulation times and interior vertical nodes (strips halos, matching length 180)
times = u_avg_series.times
zu    = znodes(u_avg_series[1])

ylimits = (0,Lz)

@info "Making animation of plane-averaged velocity profiles..."

anim = @animate for i in 1:length(times)

    t = times[i]
    if i % 100 == 0
        @info "Drawing frame $i / $(length(times)) at sim time t = $(round(t, digits=1))..."
    end

    # Extract 1D interior velocity vectors
    u_prof = vec(interior(u_avg_series[i], 1, 1, :))
    v_prof = vec(interior(v_avg_series[i], 1, 1, :))

    # Plot u_avg profile normalized by U*
    p1 = plot( (u_prof.-U∞)/u_star, zu,
             linewidth = 3,
             color     = :navy,
             xlabel    = "(<u>-U∞)/u*",
             ylabel    = "Height z",
             xlims     = (-10, 7.5),
             ylims     = ylimits,
             grid      = true,
             margin    = 25px,
             legend    = false)

    # Plot u_avg profile normalized by U*
    p2 = plot( v_prof/u_star, zu,
            linewidth = 3,
            color     = :crimson,
            xlabel    = "<v>/u*",
            ylabel    = "Height z",
            xlims     = (-10, 7.5),
            ylims     = ylimits,
            grid      = true,
            margin    = 25px,
            legend    = false)

    plot(p1,p2,
        layout     = (2,1),
        size       = (1000,600),
        margin     = 25px,
        plot_title = @sprintf("Velocity profiles (N/f = %.1f) | t = %.1f", r, t))
end

mkpath(save_folder*"Velocity")
mp4(anim, save_folder*@sprintf("Velocity/r = %.1f.mp4", r), fps = 60)


# ============================= #
## Average vorticity animation ##
# ============================= #

@info "Loading vorticity time series..."
vort_file = root * "Avg_vort.jld2"

ωx_avg_series = FieldTimeSeries(vort_file, "ωx_avg")
ωy_avg_series = FieldTimeSeries(vort_file, "ωy_avg")
ωz_avg_series = FieldTimeSeries(vort_file, "ωz_avg")

# Extract simulation times
vort_times = ωx_avg_series.times

# Extract interior z-nodes separately to account for grid staggering
zx = znodes(ωx_avg_series[1])
zy = znodes(ωy_avg_series[1])
zz = znodes(ωz_avg_series[1])

ylimits = (0,Lz)

@info "Making animation of plane-averaged vorticity profiles..."

anim_vort = @animate for i in 1:length(vort_times)

    t = vort_times[i]
    if i % 100 == 0
        @info "Drawing vorticity frame $i / $(length(vort_times)) at sim time t = $(round(t, digits=1))..."
    end

    # Extract 1D interior vorticity vectors (stripping halo cells)
    ωx_prof = vec(interior(ωx_avg_series[i], 1, 1, :))
    ωy_prof = vec(interior(ωy_avg_series[i], 1, 1, :))
    ωz_prof = vec(interior(ωz_avg_series[i], 1, 1, :))

    # Panel 1: ωx profile
    p_x = plot(ωx_prof / f₀, zx,
               linewidth = 2,
               color     = :crimson,
               xlabel    = "<ωx> / f₀",
               ylabel    = "Height z",
               xlims     = (-100, 100),
               ylims     = ylimits,
               grid      = true,
               legend    = false)

    # Panel 2: ωy profile
    p_y = plot(ωy_prof / f₀, zy,
               linewidth = 2,
               color     = :teal,
               xlabel    = "<ωy> / f₀",
               ylabel    = "Height z",
               xlims     = (-50, 200),
               ylims     = ylimits,
               grid      = true,
               legend    = false)

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
         margin     = 25px,
         plot_title = @sprintf("Plane-Averaged Vorticity Profiles (N/f = %.1f) | t = %.1f", r, t))
end

mkpath(save_folder*"Vorticity")
mp4(anim_vort, save_folder*@sprintf("Vorticity/r = %.1f.mp4", r), fps = 60)