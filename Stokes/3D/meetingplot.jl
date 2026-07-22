using Oceananigans, Plots, Printf

## Plot of horizontally-averaged buoyancy gradient ∂B/∂z with depth over time ##

# Select the case and pull in its physical parameters (ω, N², δ, T_tide,
# filename, outdir, ...). Default to Ri500 if run without an argument.
isempty(ARGS) && push!(ARGS, "Ri500")
include("case_params.jl")

# The simulation never saves db_dz directly; it saves the horizontally
# averaged buoyancy B(z, t) in the profiles file. We differentiate that.
profiles_file = filename * "_profiles.jld2"

B_ts = FieldTimeSeries(profiles_file, "B")

# Vertical grid nodes (B is a z-center field) and simulation times.
_, _, zb = nodes(B_ts)
t_save = B_ts.times

Nz = length(zb)
Nt = length(t_save)

# Centered finite difference of a profile onto its own grid points,
# handling the non-uniform vertical spacing.
function ddz(prof, z)
    d = similar(prof)
    d[1]   = (prof[2]   - prof[1])     / (z[2]   - z[1])
    d[end] = (prof[end] - prof[end-1]) / (z[end] - z[end-1])
    for k in 2:length(z)-1
        d[k] = (prof[k+1] - prof[k-1]) / (z[k+1] - z[k-1])
    end
    return d
end

# Assemble ∂B/∂z into a [Nz, Nt] matrix.
gradient_data = zeros(Nz, Nt)
for n in 1:Nt
    B_prof = interior(B_ts[n], 1, 1, :)
    gradient_data[:, n] = ddz(B_prof, zb)
end

# Zoom into the near-wall region where the gradient structure lives.
z_top = 5δ
kmask = findall(<(z_top), zb)
zbzoom = zb[kmask]

heatmap(t_save * ω, zbzoom / δ, gradient_data[kmask, :] / N²,
        xlabel = "ωt",
        ylabel = "Height z/δ",
        title  = @sprintf("(∂b/∂z)/N² for Ri = %.0f", Ri),
        color  = :thermal) # :thermal highlights intensifying gradients

savefig(joinpath(outdir, @sprintf("buoyancy_gradient_Ri%.0f.png", Ri)))
