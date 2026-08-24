using Oceananigans, JLD2, Plots, Printf, Statistics

# Figure 4 for the L_strat sweep (deviates from Gayen et al.; see case_params.jl).
# Time–depth heatmap of the plane-averaged buoyancy gradient ∂⟨b⟩/∂z normalized
# by the far-field N∞² = Ri·ω², with the paper's mixed-layer contours at 0.3, 0.5.
#
# Grid of panels: rows = L_strat ∈ {2,4,6,8} m, columns = Ri ∈ {500, 2500}. This
# lays the sweep out so the effect of the stratification scale L (down each
# column) and of Ri (across each row) is read directly.
#
# Depth axis is z/δ with the NEW length scale δ = u*/f (f = ω = 1e-4), u* being
# the peak wall-friction velocity over the final tidal period of each run — the
# colleague's Ekman scaling. δ is computed per case (printed in each title), so
# each panel is normalized by its own friction scale.
#
# Reads only the *_profiles.jld2 already written by Tidal3D.jl. Run:
#   julia --project=.. Figure4.jl

const ω  = 1e-4
const f  = ω                       # ω is equated with the Coriolis parameter
const ν  = 1.0e-6
const δs = sqrt(2ν / ω)            # Stokes thickness (old scale), for reference
const T_tide = 2π / ω

const L_values  = [2, 4, 6, 8]
const Ri_values = [(500, "Ri500"), (2500, "Ri2500")]
# δ = u*/f ≈ 10 m turns out comparable to the domain (Lz ≈ 12.7 m), so the whole
# analysed test section is only z/δ ≲ 1. Cap at 1.0 to show the test section and
# exclude the sponge (which begins at 70 δ_s ≈ 0.99 δ).
const zmax_δ    = 1.0             # z/δ axis range (δ = u*/f)

# Diverging ramp anchored at 1: warm = mixed (gradient below background),
# neutral = undisturbed background, cool = sharpened pycnocline.
const gradient_map = cgrad(["#7A3117", "#B4502C", "#D9855F", "#E9E7E4",
                            "#7FADE0", "#3C7CC4", "#1B4E8F"])

default(fontfamily = "sans-serif", framestyle = :box, grid = false,
        tickfontsize = 8, guidefontsize = 9, titlefontsize = 10)

outdir = joinpath(@__DIR__, "figures")
mkpath(outdir)

# u* = sqrt(ν |∂U/∂z|_wall), peak over the last full period; δ = u*/f.
function delta_ustar(U_ts, zc, times)
    z1 = zc[1]
    uτ = [sqrt(ν * abs(interior(U_ts[n])[1, 1, 1]) / z1) for n in 1:length(times)]
    mask = times .>= (times[end] - T_tide)
    ustar = maximum(uτ[mask])
    return ustar / f, ustar
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

    δ, ustar = delta_ustar(U_ts, zc, times)

    ks = findall(z -> z / δ <= zmax_δ, zg)
    Gn = G[ks, :] ./ N²_ref
    zδ = zg[ks] ./ δ
    ωt = times .* ω

    ttl = @sprintf("L=%d m, Ri=%d   (δ=u*/f=%.2f m)", L, Ri, δ)
    plt = heatmap(ωt, zδ, Gn;
                  clims = (0, 2), color = gradient_map,
                  xlabel = "ωt", ylabel = "z / δ  (δ=u*/f)", title = ttl,
                  colorbar_title = "  ∂⟨b⟩/∂z / N²")
    contour!(plt, ωt, zδ, Gn; levels = [0.3, 0.5],
             color = RGB(0.15, 0.15, 0.15), linewidth = 1.0)
    push!(panels, plt)
end

isempty(panels) && error("No profile files found — run the simulations first")

# 4 rows (L) × 2 columns (Ri); panels were pushed in that order.
fig = plot(panels...; layout = (length(L_values), length(Ri_values)),
           size = (1150, 320 * length(L_values)),
           leftmargin = 6Plots.mm, rightmargin = 10Plots.mm,
           bottommargin = 4Plots.mm, topmargin = 3Plots.mm,
           plot_title = "Buoyancy gradient — L_strat sweep (exponential background, δ = u*/f)",
           plot_titlefontsize = 12)
savefig(fig, joinpath(outdir, "Figure4_sweep.png"))
@info "Saved figures/Figure4_sweep.png"
