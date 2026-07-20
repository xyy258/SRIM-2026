using Oceananigans, JLD2, Plots, Printf

# Faithful reproduction of Figure 4 of Gayen, Sarkar & Taylor (2010):
# time-depth heatmap of the plane-averaged buoyancy (their "temperature")
# gradient, normalized by the background value, for Ri = 500 (a) and
# Ri = 2500 (b), with white contours at the 0.3 and 0.5 levels that they
# use to mark the mixed layer / thermocline.
#
# Their axes: z/δ_s ∈ [0, 40], ωt ∈ [0, 25] (≈ 4 tidal periods — matches the
# length of our Ri500/Ri2500 runs). Uses the profile data already saved by
# Tidal3D.jl; does not rerun any simulation.

const ω = 1.4075235e-4
const ν = 1.109e-5
const δ = sqrt(2ν / ω)

cases = [("Ri500", 500.0 * ω^2), ("Ri2500", 2500.0 * ω^2)]

zmax_δ = 40.0    # matches the paper's z/δ_s axis range

panels = []
for (case, N²) in cases
    fname = joinpath("output_" * case, "TidalBL3D_" * case * "_profiles.jld2")
    B_ts  = FieldTimeSeries(fname, "B")
    times = B_ts.times
    zc    = znodes(B_ts)
    Nt    = length(times)

    Bmean = zeros(length(zc), Nt)
    for n in 1:Nt
        Bmean[:, n] .= vec(interior(B_ts[n]))
    end

    zg = 0.5 .* (zc[1:end-1] .+ zc[2:end])       # midpoints
    G  = diff(Bmean, dims = 1) ./ diff(zc)        # ∂b/∂z at midpoints

    ks = findall(z -> z / δ <= zmax_δ, zg)
    Gn = G[ks, :] ./ N²                           # normalized gradient
    zδ = zg[ks] ./ δ
    ωt = times .* ω

    plt = heatmap(ωt, zδ, Gn;
                  clims = (0, 2), color = :thermal,
                  xlabel = "ωt", ylabel = "z / δ_s",
                  title  = "$case: ∂b/∂z / N²  (reproducing Fig. 4)",
                  colorbar_title = "∂b/∂z / N²")
    contour!(plt, ωt, zδ, Gn; levels = [0.3, 0.5],
             color = :white, linewidth = 2, colorbar = false)
    push!(panels, plt)
end

plot(panels...; layout = (2, 1), size = (900, 900))
savefig("Figure4_reproduction.png")
@info "Saved Figure4_reproduction.png (Ri500 top, Ri2500 bottom — cf. paper Fig. 4a,b)"
