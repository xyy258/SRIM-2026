using Oceananigans, JLD2, Plots, Printf
# using NCDatasets

# Import parameters
if isempty(r)
    include("Parameters.jl")
end

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

zbconcat = zb[findall(x -> x < 0.5*δ, zb)]
Nzconcat = length(zbconcat)

# Here, we loop over all iterations
anim = @animate for (i, iter) in enumerate(iterations)

    @info "Drawing frame $i from iteration $iter..."
    b_xz = file_b["timeseries/b/$iter"][:, 1, :];

    t = file_vel["timeseries/t/$iter"];
    t_save[i] = t # save the time

    b_xz_plot = heatmap(xb, zb/δ, b_xz'/N²;
        color = :thermal, clims=(0, 1.05).*maximum(abs, b_xz'/N²),
        xlabel = "x", ylabel = "z/δ",
        xlims = (0, Lx), ylims = (0,Lz/δ)); # Shows entire height of domain

    diff = (b_xz[:,1:Nzconcat]' .- N²*zbconcat)/N²
    clim_max = maximum(diff)
    clim_min = minimum(diff)

    b_diff_xz_plot = heatmap(xb, zbconcat/δ, (b_xz[:,1:Nzconcat]' .- N²*zbconcat)/N²;
        color = :coolwarm, clims = (clim_min,clim_max).*1.4,
        xlabel = "x", ylabel = "z/δ",
        xlims = (0, Lx), ylims = (0,zbconcat[end]/δ)); # Shows lower part of domain near the rigid boundary

    b_title = @sprintf("b/N² at t = %s, N/f = %.1f", round(t), r);
    b_diff_title = @sprintf("(b-N²z)/N² at t = %s, N/f = %.1f", round(t), r);

# Combine the sub-plots into a single figure
    plot(b_xz_plot, b_diff_xz_plot, layout = (2, 1), size = (1000, 500), title = [b_title b_diff_title])

    if iter == iterations[end]
        close(file_vel)
        close(file_b)
    end
end

# Save the animation to a file
mp4(anim, @sprintf("Ekman/3D Simulation/Ekman Plot r = %.1f.mp4", r), fps = 15) # hide