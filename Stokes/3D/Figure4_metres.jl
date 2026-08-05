using Oceananigans, JLD2, Plots, Printf

# Figure 4 for the L sweep, depth axis in PHYSICAL METRES. Exponential background
# N²_bg = N∞²[1−exp(−z/L)] with L a fraction of Lz (case_params.jl).
#
# Time–depth heatmap of the plane-averaged buoyancy gradient ∂⟨b⟩/∂z normalized
# by the far-field N∞² = Ri·ω², with mixed-layer contours at 0.3, 0.5.
# Grid of panels: rows = L ∈ {0.2, 0.5, 1.0, 1.5} Lz, columns = Ri ∈ {500, 2500}.
#
# Reads only *_profiles.jld2 written by Tidal3D.jl.  Run: julia --project=.. Figure4_metres.jl

const ω  = 1e-4
const f  = ω
const ν  = 1.0e-6
const T_tide = 2π / ω

const Lz        = 90 * sqrt(2ν / ω)   # OLD domain height ≈ 12.73 m
const Lz_new    = 150 * sqrt(2ν / ω)  # NEW taller domain ≈ 21.21 m
const L_fracs   = [0.2, 0.5, 1.0, 1.5]   # L_strat as a fraction of Lz
const Ri_values = [(500, "Ri500"), (2500, "Ri2500")]
const zmax_m    = 10.0            # metres shown (≈ test section, sponge excluded)

# MIXED SWEEP: only L = 1·Lz was rerun on the new taller domain (150δ + top
# sponge); the other three L values are retained from the earlier 90δ run. So the
# right Lz for L = 1·Lz's label is Lz_new, and the old Lz for the rest.
const new_domain_fracs = [1.0]
Lz_of(fr) = fr in new_domain_fracs ? Lz_new : Lz

# Tag builder matching case_params.jl: L_frac 0.5 → "L0p5Lz", 1.0 → "L1Lz".
flbl(f) = isinteger(f) ? string(Int(f)) : replace(string(f), "." => "p")

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
for f in L_fracs, (Ri, ricase) in Ri_values
    tag   = "L$(flbl(f))Lz_$(ricase)"
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

    # Derive the domain from the data itself (per PANEL, not per L): the new 150δ
    # grid tops out at zc[end] ≈ 23.9 m, the old 90δ grid at ≈ 12.6 m. This is now
    # mixed WITHIN a row — e.g. L=0.5 Ri500 was reran on 150δ while its Ri2500
    # sibling is still the old 90δ run — so a per-panel test is the only honest one.
    is_new = zc[end] > 15.0
    Lz_dom = is_new ? Lz_new : Lz
    dom    = is_new ? " [150δ dom]" : " [90δ dom]"
    ttl = @sprintf("L=%.1fLz=%.1f m%s, Ri=%d   (δ=u*/f=%.2f m)", f, f*Lz_dom, dom, Ri, δ)
    cmin, cmax = extrema(Gn)          # per-panel colorbar: this case's own gradient range
    plt = heatmap(ωt, zm, Gn;
                  clims = (cmin, cmax), color = gradient_map,
                  xlabel = "ωt", ylabel = "z (m)", title = ttl,
                  colorbar_title = "  ∂⟨b⟩/∂z / N²")
    push!(panels, plt)
end
isempty(panels) && error("No profile files found — run the simulations first")

fig = plot(panels...; layout = (length(L_fracs), length(Ri_values)),
           size = (1150, 320 * length(L_fracs)),
           leftmargin = 6Plots.mm, rightmargin = 10Plots.mm,
           bottommargin = 4Plots.mm, topmargin = 3Plots.mm,
           plot_title = "Buoyancy gradient — L sweep (exp. background; per-panel domain tagged [150δ]/[90δ]; z in metres)",
           plot_titlefontsize = 12)
savefig(fig, joinpath(outdir, "Figure4_sweep_metres.png"))
@info "Saved figures/Figure4_sweep_metres.png"
