using Oceananigans, JLD2, Plots, Printf

# Reproduction of Figure 4 of Gayen, Sarkar & Taylor (2010): time-depth heatmap
# of the plane-averaged buoyancy ("temperature") gradient normalized by the
# background value, with white contours at the 0.3 and 0.5 levels that they use
# to mark the mixed layer / thermocline.
#
# Their axes: z/δ_s ∈ [0, 40], ωt from 0. The paper shows Ri = 500 (a) and
# Ri = 2500 (b); we add Ri = 0 as a third panel, since that case now carries the
# thermal field as a genuine passive scalar (see case_params.jl) and its
# unrestrained mixing is the natural baseline for the other two.
#
# This run uses an exponential background, N²_bg(z) = N∞²[1 − exp(−z/L)] with
# L = 10 δ_s, instead of the paper's uniform one. The normalization is still the
# far-field N∞², which keeps the colour scale directly comparable to the linear
# run in "Centered - Linear/" — but note the consequence: the initial condition
# is no longer flat at 1. It rises from 0 at the seabed to 1 aloft, so the warm
# band near the wall at ωt = 0 is the background profile, not mixing. The
# dashed line marks L, above which the background is within 63 % of N∞².
#
# Uses the profile data already saved by Tidal3D.jl; reruns no simulation.
# Run from this directory:  julia --project=.. Figure4.jl

const ω  = 1.4075235e-4
const ν  = 1.0e-6
const δ  = sqrt(2ν / ω)
const T_tide = 2π / ω
const L_strat_δ = 10.0         # background scale height, in δ_s (case_params.jl)

const zmax_δ = 40.0            # the paper's z/δ_s axis range

# (label, N² used to normalize the gradient). For Ri = 0 the scalar is passive
# and its background gradient is the reference ω² set in case_params.jl.
cases = [("Ri0",    "(a) Ri = 0  (no stratification)",      1.0   * ω^2),
         ("Ri500",  "(b) Ri = 500",                       500.0 * ω^2),
         ("Ri2500", "(c) Ri = 2500",                      2500.0 * ω^2)]

default(fontfamily = "sans-serif", framestyle = :box, grid = false,
        tickfontsize = 9, guidefontsize = 10, titlefontsize = 11)

# ∂b/∂z / N² has a meaningful midpoint — 1 is the undisturbed background, below
# it the gradient has been mixed away and above it the thermocline has been
# sharpened. That is polarity, not magnitude, so it gets a diverging ramp
# anchored at 1: warm for mixed, a light neutral at the background, cool for
# sharpened. A single-hue sequential ramp would put the background halfway up
# the scale and leave the mixed layer and the thermocline looking alike.
const gradient_map = cgrad(["#7A3117", "#B4502C", "#D9855F", "#E9E7E4",
                            "#7FADE0", "#3C7CC4", "#1B4E8F"])

outdir = joinpath(@__DIR__, "figures")
mkpath(outdir)

panels = []
for (case, ttl, N²) in cases
    fname = joinpath(@__DIR__, "output_" * case,
                     "TidalBL3D_" * case * "_profiles.jld2")
    isfile(fname) || (@warn "Missing $fname — skipping $case"; continue)

    B_ts  = FieldTimeSeries(fname, "B")
    times = B_ts.times
    zc    = znodes(B_ts)
    Nt    = length(times)

    Bmean = zeros(length(zc), Nt)
    for n in 1:Nt
        Bmean[:, n] .= vec(interior(B_ts[n]))
    end

    zg = 0.5 .* (zc[1:end-1] .+ zc[2:end])        # gradients live at midpoints
    G  = diff(Bmean, dims = 1) ./ diff(zc)        # ∂b/∂z at midpoints

    ks = findall(z -> z / δ <= zmax_δ, zg)
    Gn = G[ks, :] ./ N²                           # normalized gradient
    zδ = zg[ks] ./ δ
    ωt = times .* ω

    # clims symmetric about 1 so the neutral colour always lands on the
    # background gradient, whatever the case.
    plt = heatmap(ωt, zδ, Gn;
                  clims = (0, 2), color = gradient_map,
                  xlabel = "ωt", ylabel = "z / δ_s", title = ttl,
                  colorbar_title = "  ∂⟨b⟩/∂z / N²")
    # The paper's mixed-layer / thermocline contours. Drawn in ink rather than
    # white so they stay legible over both poles of the diverging ramp.
    # NB: no colorbar = false here — that attribute is per-subplot, so setting
    # it on the contour silently removes the heatmap's colorbar as well.
    contour!(plt, ωt, zδ, Gn; levels = [0.3, 0.5],
             color = RGB(0.15, 0.15, 0.15), linewidth = 1.2)
    # Background scale height: below this the initial profile is itself weakly
    # stratified, so warm colours there are partly inherited, not mixed away.
    hline!(plt, [L_strat_δ]; color = RGB(0.15, 0.15, 0.15), linestyle = :dash,
           linewidth = 1.0, label = "")
    push!(panels, plt)
end

isempty(panels) && error("No profile files found — run the simulations first")

fig = plot(panels...; layout = (length(panels), 1),
           size = (1000, 330 * length(panels)),
           leftmargin = 6Plots.mm, rightmargin = 10Plots.mm,
           bottommargin = 4Plots.mm, topmargin = 3Plots.mm,
           plot_title = "Plane-averaged buoyancy gradient (cf. Gayen et al. 2010, fig. 4)",
           plot_titlefontsize = 12)
savefig(fig, joinpath(outdir, "Figure4_reproduction.png"))
@info "Saved figures/Figure4_reproduction.png"
