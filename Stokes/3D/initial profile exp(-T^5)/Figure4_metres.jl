using Oceananigans, JLD2, Plots, Printf

# Figure 4 for the T sweep, depth axis in PHYSICAL METRES. Linear background with
# an initial bottom mixed layer of depth T carved in (case_params.jl).
#
# Time–depth heatmap of the plane-averaged buoyancy gradient ∂⟨b⟩/∂z normalized
# by the far-field N∞² = Ri·ω², with mixed-layer contours at 0.3, 0.5.
# Grid of panels: rows = T ∈ {3, 4.5, 6} m, columns = Ri ∈ {500, 2500}.
#
# Reads only *_profiles.jld2 written by Tidal3D.jl.  Run: julia --project=.. Figure4_metres.jl

const ω  = 1e-4
const f  = ω
const ν  = 1.0e-6
const T_tide = 2π / ω

const T_values  = [3.0, 4.5, 6.0]     # initial mixed-layer depth T (metres)
const Ri_values = [(500, "Ri500"), (2500, "Ri2500")]
const zmax_m    = 10.0            # metres shown (≈ test section, sponge excluded)

# Tag builder matching case_params.jl: T=4.5 → "T4p5".
tint(T) = isinteger(T) ? string(Int(T)) : replace(string(T), "." => "p")

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
for T in T_values, (Ri, ricase) in Ri_values
    tag   = "T$(tint(T))_$(ricase)"
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

    ttl = @sprintf("T=%.1f m, Ri=%d   (δ=u*/f=%.2f m)", T, Ri, δ)
    plt = heatmap(ωt, zm, Gn;
                  clims = (0, 2), color = gradient_map,
                  xlabel = "ωt", ylabel = "z (m)", title = ttl,
                  colorbar_title = "  ∂⟨b⟩/∂z / N²")
    contour!(plt, ωt, zm, Gn; levels = [0.3, 0.5],
             color = RGB(0.15, 0.15, 0.15), linewidth = 1.0)
    push!(panels, plt)
end

isempty(panels) && error("No profile files found — run the simulations first")

fig = plot(panels...; layout = (length(T_values), length(Ri_values)),
           size = (1150, 320 * length(T_values)),
           leftmargin = 6Plots.mm, rightmargin = 10Plots.mm,
           bottommargin = 4Plots.mm, topmargin = 3Plots.mm,
           plot_title = "Buoyancy gradient — T sweep (linear background, mixed-layer IC, z in metres)",
           plot_titlefontsize = 12)
savefig(fig, joinpath(outdir, "Figure4_sweep_metres.png"))
@info "Saved figures/Figure4_sweep_metres.png"
