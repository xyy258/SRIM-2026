using Oceananigans, JLD2, NCDatasets, Plots, Printf

# Set the filename (without the extension)
filename = "Data/Ekman_2D"

# Read in the first iteration.  We do this to load the grid
# filename * ".jld2" concatenates the extension to the end of the filename
b_ic = FieldTimeSeries(filename * "_b_c.jld2", "b", iterations=0)

## Load in coordinate arrays
## We do this separately for each variable since Oceananigans uses a staggered grid
xb, yb, zb = nodes(b_ic)

## Now, open the file with our data
file_xz_velocity = jldopen(filename * "_velocity.jld2")
file_xz_b_c      = jldopen(filename * "_b_c.jld2")

## Extract a vector of iterations
iterations = parse.(Int, keys(file_xz_velocity["timeseries/t"]))

@info "Making an animation from saved data..."

t_save = zeros(length(iterations))

# Here, we loop over all iterations
anim = @animate for (i, iter) in enumerate(iterations)

    # Only print a progress message every 10 frames
    if i % 5 == 0
        @info "Drawing frame $i from iteration $iter..."
    end

    b_xz = file_xz_b_c["timeseries/b/$iter"][:, 1, :];

    t = file_xz_velocity["timeseries/t/$iter"];
    t_save[i] = t # save the time

    zbconcat = zb[findall(x -> x < 5, zb)]
    Nzconcat = length(zbconcat)

    b_xz_plot = heatmap(xb, zb, b_xz[1:Nzconcat]'/N²; color=:thermal, xlabel="x", ylabel="z",
        xlims=(0, Lx), ylims=(0, zb[Nzconcat]));
    b_diff_xz_plot = heatmap(xb, zb, b_xz[1:Nzconcat]'/N²-zb[1:Nzconcat]; color=:thermal, xlabel="x", ylabel="z",
        xlims=(0, Lx), ylims=(0, zb[Nzconcat]));

    b_title = @sprintf("b/N², t = %s", round(t));
    b_diff_title = @sprintf("(b-N²z)/N², t = %s", round(t));

    # Combine the sub-plots into a single figure
        plot(b_xz_plot, b_diff_xz_plot, layout=(2, 1), size=(1000, 500), title=[b_title b_diff_title])

    if iter == iterations[end]
        close(file_xz_velocity)
        close(file_xz_b_c)
    end
end

# Save the animation to a file
mp4(anim, "Ekman/2D Simulation/Ekman Plot 2D.mp4", fps=15) # hide