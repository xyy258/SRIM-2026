using Oceananigans, JLD2, Plots, Printf

# Figure 4 for the L_strat sweep, depth axis in PHYSICAL METRES (companion to
# Figure4.jl, which uses z/δ with δ = u*/f). Since δ = u*/f ≈ 10 m is comparable
# to the domain, the metres axis is often the more intuitive read of where the
# mixed layer and pycnocline actually sit.
#
# Time–depth heatmap of the plane-averaged buoyancy gradient ∂⟨b⟩/∂z normalized
# by the far-field N∞² = Ri·ω², with mixed-layer contours at 0.3, 0.5.
# Grid of panels: rows = L_strat ∈ {2,4,6,8} m, columns = Ri ∈ {500, 2500}.
#
# Reads only *_profiles.jld2 written by Tidal3D.jl.  Run: julia --project=.. Figure4_metres.jl

const ω  = 1e-4
const f  = ω
const ν  = 1.0e-6
const T_tide = 2π / ω

const L_values  = [2, 4, 6, 8]
const Ri_values = [(500, "Ri500"), (2500, "Ri2500")]
const zmax_m    = 10.0            # metres shown (≈ test section, sponge excluded)

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
for L in L_values, (Ri, ricase) in Ri_values
    tag   = "L$(L)_$(ricase)"
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

    # OLD: clims = (0, 2) shared by every panel. Comparable across cases, but the
    # weakly stratified panels (large L, where N²_bg near the bed is a small
    # fraction of N∞²) sat in the bottom sliver of the range and read as flat.
    # NEW: each panel is scaled to its own [min, max], so structure is visible in
    # every case — at the cost of cross-panel comparability, hence the range is
    # printed in the panel title. Degenerate (constant) fields get a padded range
    # so the colorbar is still well defined.
    gmin, gmax = extrema(filter(isfinite, Gn))
    if gmax - gmin < 1e-12
        pad  = max(abs(gmax), 1.0) * 1e-3
        gmin, gmax = gmin - pad, gmax + pad
    end

    ttl = @sprintf("L=%d m, Ri=%d   (δ=u*/f=%.2f m)  [%.2f, %.2f]",
                   L, Ri, δ, gmin, gmax)
    plt = heatmap(ωt, zm, Gn;
                  clims = (gmin, gmax), color = gradient_map,
                  xlabel = "ωt", ylabel = "z (m)", title = ttl,
                  colorbar_title = "  ∂⟨b⟩/∂z / N²")
    # OLD: mixed-layer contours at 0.3 and 0.5 overlaid on the heatmap. Dropped —
    # with per-panel colour limits they were the only absolute reference left, and
    # they clutter the weakly stratified panels where the field wanders across
    # both levels.
    # contour!(plt, ωt, zm, Gn; levels = [0.3, 0.5],
    #          color = RGB(0.15, 0.15, 0.15), linewidth = 1.0)
    push!(panels, plt)
end

isempty(panels) && error("No profile files found — run the simulations first")

fig = plot(panels...; layout = (length(L_values), length(Ri_values)),
           size = (1150, 320 * length(L_values)),
           leftmargin = 6Plots.mm, rightmargin = 10Plots.mm,
           bottommargin = 4Plots.mm, topmargin = 3Plots.mm,
           plot_title = "Buoyancy gradient — L_strat sweep (exponential background, z in metres)",
           plot_titlefontsize = 12)
savefig(fig, joinpath(outdir, "Figure4_sweep_metres.png"))
@info "Saved figures/Figure4_sweep_metres.png"
