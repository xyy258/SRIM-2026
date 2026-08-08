using Oceananigans, JLD2, Plots, Printf

# Figure 4 for the SOFTPLUS sweep, depth axis in PHYSICAL METRES.
# Background: N²_bg(z) = N∞²·sigmoid(sharp·(z−T)) — unstratified below the
# pycnocline at z = T, N∞² above, over a transition of width ~1/sharp.
#
# Time–depth heatmap of the plane-averaged buoyancy gradient ∂⟨b⟩/∂z normalized
# by the far-field N∞² = Ri·ω².
# Panel grid: rows = T ∈ {5, 10, 20, 30} m, columns = √Ri ∈ {0, 0.5, 1, 2, 5, 10}.
#
# Reads only *_profiles.jld2 written by Tidal3D.jl.
#   julia --project=. Figure4_metres.jl
# Subsets:  T_VALUES="10 20" SQRT_RI="1 2 5" julia --project=. Figure4_metres.jl

const ω  = 1e-4
const f  = ω
const ν  = 1.0e-6
const T_tide = 2π / ω
const δs = sqrt(2ν / ω)           # Stokes thickness ≈ 0.1414 m

# These must match case_params.jl — the panel titles and the pycnocline marker
# are only meaningful if they describe the profile the run actually used.
const sharp = parse(Float64, get(ENV, "SHARP", "6"))
const Lz    = 50.0

parse_list(key, default) = parse.(Float64, split(get(ENV, key, default)))
const T_values    = parse_list("T_VALUES", "5 10 20 30")
const sqrtRi_vals = parse_list("SQRT_RI",  "0 0.5 1 2 5 10")

# Tag builder matching case_params.jl: T = 5.0 → "5", √Ri = 0.5 → "0p5",
# giving e.g. "P4_T5_sqrtRi0p5".
num_lbl(x) = isinteger(x) ? string(Int(x)) : replace(string(x), "." => "p")
tag_of(T, s) = "P4_T" * num_lbl(T) * "_sqrtRi" * num_lbl(s)

# Depth window per row, mirroring Lz_test in case_params.jl so the figure shows
# the same section the animations do: the pycnocline at z = T plus headroom.
zmax_of(T) = min(Lz, max(70δs, T + 10))

const gradient_map = cgrad(["#7A3117", "#B4502C", "#D9855F", "#E9E7E4",
                            "#7FADE0", "#3C7CC4", "#1B4E8F"])

default(fontfamily = "sans-serif", framestyle = :box, grid = false,
        tickfontsize = 7, guidefontsize = 8, titlefontsize = 9)

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
nfound = 0
for T in T_values, s in sqrtRi_vals
    tag   = tag_of(T, s)
    Ri    = s^2
    fname = joinpath(@__DIR__, "output_" * tag, "TidalBL3D_" * tag * "_profiles.jld2")
    if !isfile(fname)
        @warn "Missing $fname — inserting empty panel for $tag"
        push!(panels, plot(title = "$tag\n(no data)", framestyle = :box,
                           showaxis = false, grid = false))
        continue
    end
    global nfound += 1

    B_ts  = FieldTimeSeries(fname, "B")
    U_ts  = FieldTimeSeries(fname, "U")
    times = B_ts.times
    zc    = znodes(B_ts)

    Bmean = zeros(length(zc), length(times))
    for n in 1:length(times)
        Bmean[:, n] .= vec(interior(B_ts[n]))
    end

    zg = 0.5 .* (zc[1:end-1] .+ zc[2:end])       # gradients at midpoints
    G  = diff(Bmean, dims = 1) ./ diff(zc)        # ∂b/∂z
    δ  = delta_ustar(U_ts, zc, times)

    # Normalize by N²_ref, matching case_params.jl: at √Ri = 0 the buoyancy is
    # zero but b is still advected as a passive scalar carrying the ω²-scaled
    # softplus background, so ω² is the reference there. Dividing by Ri·ω² would
    # be a divide-by-zero and blank the whole √Ri = 0 column.
    N²_ref = Ri > 0 ? Ri * ω^2 : ω^2
    ks = findall(z -> z <= zmax_of(T), zg)
    Gn = G[ks, :] ./ N²_ref
    zm = zg[ks]
    ωt = times .* ω

    # Per-panel colour limits: with N²_bg → 0 below the pycnocline, a shared scale
    # would flatten every panel's near-bed structure into one colour. The range is
    # printed in the title since that costs cross-panel comparability.
    gmin, gmax = extrema(filter(isfinite, Gn))
    if gmax - gmin < 1e-12
        pad = max(abs(gmax), 1.0) * 1e-3
        gmin, gmax = gmin - pad, gmax + pad
    end

    ttl = @sprintf("T=%d m, √Ri=%g (Ri=%g)\nδ=u*/f=%.2f m  [%.2f, %.2f]",
                   Int(T), s, Ri, δ, gmin, gmax)
    plt = heatmap(ωt, zm, Gn;
                  clims = (gmin, gmax), color = gradient_map,
                  xlabel = "ωt", ylabel = "z (m)", title = ttl,
                  colorbar_title = "  ∂⟨b⟩/∂z / N²")
    # Initial pycnocline height, the reference the mixed layer grows towards.
    hline!(plt, [T]; color = RGB(0.15, 0.15, 0.15), linestyle = :dash,
           linewidth = 1.2, label = "")
    push!(panels, plt)
end

nfound == 0 && error("No profile files found — run the simulations first")

fig = plot(panels...; layout = (length(T_values), length(sqrtRi_vals)),
           size = (390 * length(sqrtRi_vals), 330 * length(T_values)),
           leftmargin = 5Plots.mm, rightmargin = 8Plots.mm,
           bottommargin = 4Plots.mm, topmargin = 4Plots.mm,
           plot_title = "Buoyancy gradient — softplus sweep (dashed = initial pycnocline z = T)",
           plot_titlefontsize = 13)
savefig(fig, joinpath(outdir, "Figure4_softplus_sweep.png"))
@info "Saved figures/Figure4_softplus_sweep.png ($nfound of $(length(T_values)*length(sqrtRi_vals)) cases found)"
