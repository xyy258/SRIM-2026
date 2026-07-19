using Oceananigans, JLD2, NCDatasets, Plots, Printf

# Set the new filename
filename = "Average buoyancy gradient"

# Load the FieldTimeSeries for the gradient
# The key "db_dz" matches what we named it in the JLD2Writer tuple above
db_dz_timeseries = FieldTimeSeries(filename * ".jld2", "db_dz")

# Extract the grid nodes (zb will contain the vertical grid levels)
xb, yb, zb = nodes(db_dz_timeseries)

## Open the file to extract the time array
file_xz = jldopen(filename * ".jld2")
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

zbconcat = zb[findall(<(10),zb)]
Nzconcat = length(zbconcat)

heatmap(t_save, zbconcat, gradient_data[1:Nzconcat,:],
    xlabel="Time (seconds)",
    ylabel="Depth z (m)",
    title="Horizontally-Averaged Buoyancy Gradient 2D",
    color=:thermal) # :thermal is great for highlighting intensifying gradients
savefig("Buoyancy gradient plot 2D.png")