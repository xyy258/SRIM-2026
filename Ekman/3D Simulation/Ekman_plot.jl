using Oceananigans, JLD2, Plots, Printf
# using NCDatasets

# Set the filename (without the extension)
filename = "Data/Ekman"

# Read in the first iteration.  We do this to load the grid
# filename * ".jld2" concatenates the extension to the end of the filename
b_ic = FieldTimeSeries(filename * "_b_c.jld2", "b")

## Load in coordinate arrays
## We do this separately for each variable since Oceananigans uses a staggered grid
xb, yb, zb = nodes(b_ic)

## Now, open the file with our data
file_vel = jldopen(filename * "_velocity.jld2")
file_b   = jldopen(filename * "_b_c.jld2")

## Extract a vector of iterations
iterations = parse.(Int, keys(file_vel["timeseries/t"]))

@info "Making an animation from saved data..."

t_save = zeros(length(iterations))
b_bottom = zeros(length(xb), length(iterations))

# Here, we loop over all iterations
anim = @animate for (i, iter) in enumerate(iterations)

    @info "Drawing frame $i from iteration $iter..."
    b_xz = file_b["timeseries/b/$iter"][:, 1, :];

    t = file_vel["timeseries/t/$iter"];
    t_save[i] = t # save the time

    zbconcat = zb[findall(x -> x < 5, zb)]
    Nzconcat = length(zbconcat)

    b_xz_plot = heatmap(xb, zbconcat/δ, b_xz[:, 1:Nzconcat]'/N²;
        color = :thermal, xlabel = "x", ylabel = "z/δ",
        xlims = (0, Lx), ylims = (0,zbconcat[end]));
    b_diff_xz_plot = heatmap(xb, zbconcat/δ, b_xz[:, 1:Nzconcat]'/N² .- reshape(zbconcat, Nzconcat, 1);
        color = :thermal, xlabel = "x", ylabel = "z/δ",
        xlims = (0, Lx), ylims = (0,zbconcat[end]));

    b_title = @sprintf("b, t = %s", round(t));
    b_diff_title = @sprintf("b, t = %s", round(t));

# Combine the sub-plots into a single figure
    plot(b_xz_plot, b_diff_xz_plot, layout = (2, 1), size = (1000, 400), title = [b_title b_diff_title])

    if iter == iterations[end]
        close(file_vel)
        close(file_b)
    end
end

# Save the animation to a file
mp4(anim, "Ekman/3D Simulation/Ekman Plot.mp4", fps = 20) # hide