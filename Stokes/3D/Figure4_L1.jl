using Oceananigans, JLD2, Plots, Printf

# Figure 4, FOCUSED on the reran L = 1·Lz case only: two panels (Ri = 500, 2500)
# on the new taller 150 δ_s domain. Time–depth heatmap of the plane-averaged
# buoyancy gradient ∂⟨b⟩/∂z normalized by N∞² = Ri·ω², over the FULL physical
# domain (0 → Lz = 150 δ_s; the 20 δ_s top sponge is excluded — its buoyancy is
# relaxed to the background so its gradient is imposed, not physical). Tightened
# color limits so the near-wall mixed layer and its evolution actually show.
#
# Reads only *_profiles.jld2 written by Tidal3D.jl.  Run: julia --project=. Figure4_L1.jl

const ω  = 1e-4
const f  = ω
const ν  = 1.0e-6
const T_tide = 2π / ω

const δs        = sqrt(2ν / ω)
const Lz_new    = 150 * δs           # NEW physical domain height ≈ 21.21 m
const Ri_values = [(500, "Ri500"), (2500, "Ri2500")]
const zmax_m    = Lz_new             # FULL physical domain (sponge above Lz excluded)
const clim_hi   = 0.8                # tightened from 2.0: field peaks ≈ 0.75

const gradient_map = cgrad(["#7A3117", "#B4502C", "#D9855F", "#E9E7E4",
                            "#7FADE0", "#3C7CC4", "#1B4E8F"])

default(fontfamily = "sans-serif", framestyle = :box, grid = false,
        tickfontsize = 8, guidefontsize = 9, titlefontsize = 10)

outdir = joinpath(@__DIR__, "figures")
mkpath(outdir)

# δ = u*/f (peak wall stress over the final period), for the title only.
function delta_ustar(U_ts, zc, times)
    z1 = zc[1]
    uτ = [sqrt(ν * abs(interior(U_ts[n])[1, 1, 1]) / z1) for n in 1:length(times)]
    ustar = maximum(uτ[times .>= (times[end] - T_tide)])
    return ustar / f
end

panels = []
for (Ri, ricase) in Ri_values
    tag   = "L1Lz_$(ricase)"
    fname = joinpath(@__DIR__, "output_" * tag, "TidalBL3D_" * tag * "_profiles.jld2")
    if !isfile(fname)
        @warn "Missing $fname — inserting empty panel for $tag"
        push!(panels, plot(title = "$tag (no data)", framestyle = :box,
                           showaxis = false, grid = false))
        continue
    end

    B_ts  = FieldTimeSeries(fname, "B")
    U_ts  = FieldTimeSeries(fname, "U")
    times = B_ts.times
    zc    = znodes(B_ts)
    Nt    = length(times)

    Bmean = zeros(length(zc), Nt)
    for n in 1:Nt
        Bmean[:, n] .= vec(interior(B_ts[n]))
    end

    zg = 0.5 .* (zc[1:end-1] .+ zc[2:end])       # gradients at midpoints
    G  = diff(Bmean, dims = 1) ./ diff(zc)        # ∂b/∂z
    N²_ref = Ri * ω^2
    δ = delta_ustar(U_ts, zc, times)

    ks = findall(z -> z <= zmax_m, zg)
    Gn = G[ks, :] ./ N²_ref
    zm = zg[ks]
    ωt = times .* ω

    ttl = @sprintf("L=1.0Lz=%.1f m [150δ full domain], Ri=%d   (δ=u*/f=%.2f m)",
                   Lz_new, Ri, δ)
    plt = heatmap(ωt, zm, Gn;
                  clims = (0, clim_hi), color = gradient_map,
                  xlabel = "ωt", ylabel = "z (m)", title = ttl,
                  colorbar_title = "  ∂⟨b⟩/∂z / N²")
    push!(panels, plt)
end

fig = plot(panels...; layout = (1, 2), size = (1300, 460),
           leftmargin = 6Plots.mm, rightmargin = 10Plots.mm,
           bottommargin = 5Plots.mm, topmargin = 3Plots.mm,
           plot_title = "Buoyancy gradient — L=1·Lz on new 150δ domain (full physical depth; z in metres)",
           plot_titlefontsize = 12)
savefig(fig, joinpath(outdir, "Figure4_L1_fulldomain.png"))
@info "Saved figures/Figure4_L1_fulldomain.png"
