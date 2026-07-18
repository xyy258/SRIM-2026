using Oceananigans, JLD2, NCDatasets, Plots, Printf

# Set the filename (without the extension)
filename = "Data/Ekman"

# Read in the first iteration.  We do this to load the grid
# filename * ".jld2" concatenates the extension to the end of the filename
u_ic = FieldTimeSeries(filename * "_velocity.jld2", "u")
v_ic = FieldTimeSeries(filename * "_velocity.jld2", "v")
w_ic = FieldTimeSeries(filename * "_velocity.jld2", "w")
b_ic = FieldTimeSeries(filename * "_b_c.jld2", "b")
c_ic = FieldTimeSeries(filename * "_b_c.jld2", "c")

## Load in coordinate arrays
## We do this separately for each variable since Oceananigans uses a staggered grid
xu, yu, zu = nodes(u_ic)
xv, yv, zv = nodes(v_ic)
xw, yw, zw = nodes(w_ic)
xb, yb, zb = nodes(b_ic)
xc, yc, zc = nodes(c_ic)

## Now, open the file with our data
file_xz_velocity = jldopen(filename * "_velocity.jld2")
file_xz_b_c      = jldopen(filename * "_b_c.jld2")

## Extract a vector of iterations
iterations = parse.(Int, keys(file_xz_velocity["timeseries/t"]))

@info "Making an animation from saved data..."

t_save = zeros(length(iterations))
b_bottom = zeros(length(xb), length(iterations))

# Here, we loop over all iterations
anim = @animate for (i, iter) in enumerate(iterations)

    @info "Drawing frame $i from iteration $iter..."

    u_xz = file_xz_velocity["timeseries/u/$iter"][:, 1, :];
    v_xz = file_xz_velocity["timeseries/v/$iter"][:, 1, :];
    w_xz = file_xz_velocity["timeseries/w/$iter"][:, 1, :];
    b_xz = file_xz_b_c["timeseries/b/$iter"][:, 1, :];
    c_xz = file_xz_b_c["timeseries/c/$iter"][:, 1, :];

# If you want an x-y slice, you can get it this way:
    # b_xy = file_xy["timeseries/b/$iter"][:, :, 1];

    t = file_xz_velocity["timeseries/t/$iter"];

    # Save some variables to plot at the end
    b_bottom[:,i] = b_xz[:, 1]; # This is the buoyancy along the bottom wall
    t_save[i] = t # save the time

    u_xz_plot = heatmap(xu, zu, u_xz'; color = :balance, xlabel = "x", ylabel = "z",
                        xlims = (0, Lx), ylims = (0,Lz));
    v_xz_plot = heatmap(xv, zv, v_xz'; color = :balance, xlabel = "x", ylabel = "z",
                        xlims = (0, Lx), ylims = (0,Lz));
    w_xz_plot = heatmap(xw, zw, w_xz'; color = :balance, xlabel = "x", ylabel = "z",
                        xlims = (0, Lx), ylims = (0,Lz));
    b_xz_plot = heatmap(xb, zb, b_xz'; color = :thermal, xlabel = "x", ylabel = "z",
                        xlims = (0, Lx), ylims = (0,Lz));
    c_xz_plot = heatmap(xc, zc, c_xz'; color = :thermal, xlabel = "x", ylabel = "z",
                        xlims = (0, Lx), ylims = (0,Lz ));

    u_title = @sprintf("u, t = %s", round(t));
    v_title = @sprintf("v, t = %s", round(t));
    w_title = @sprintf("w, t = %s", round(t));
    b_title = @sprintf("b, t = %s", round(t));
    c_title = @sprintf("c, t = %s", round(t));

# Combine the sub-plots into a single figure
    plot(b_xz_plot, c_xz_plot, layout = (2, 1), size = (1000, 400), title = [b_title c_title])

    if iter == iterations[end]
        close(file_xz_velocity)
        close(file_xz_b_c)
    end
end

# Save the animation to a file
mp4(anim, "Ekman/3D Simulation/Ekman Plot.mp4", fps = 20) # hide