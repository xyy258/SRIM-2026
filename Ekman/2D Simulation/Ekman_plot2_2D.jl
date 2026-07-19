using Oceananigans, JLD2, NCDatasets, Plots, Printf

# Set the new filename
filename = "Ekman/Data/Average b gradient 2D"

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

zbconcat = zb[findall(<(5),zb)]
Nzconcat = length(zbconcat)

heatmap(t_save*f₀, zbconcat/δ, gradient_data[1:Nzconcat,:]/N²,
        xlabel="tf",
        ylabel="Height z/δ",
        title="2D (∂b/∂z)/N²",
        color=:thermal) # :thermal is great for highlighting intensifying gradients
savefig("Ekman/2D Simulation/Buoyancy gradient plot 2D.png")