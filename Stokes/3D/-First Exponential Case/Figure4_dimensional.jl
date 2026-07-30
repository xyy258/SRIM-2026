using Oceananigans, JLD2, Plots, Printf

# Dimensional version of Figure4.jl — same plane-averaged buoyancy gradient
# heatmap, but with every axis in ocean units instead of the paper's (z/δ_s, ωt):
#
#   height  z  in metres           (the paper's 40 δ_s ends up at 4.77 m)
#   time    t  in hours            (one tidal period = 12.42 h)
#   colour  ∂⟨b⟩/∂z in s⁻²         (per-panel decade, since N∞² spans 0 → 2500 ω²)
#
# Output: figures/Figure4_dimensional.png. Figure4.jl and its normalized axes are
# untouched — that one is the like-for-like comparison against the paper, this one
# is for reading heights and durations off directly.
#
# The colour mapping is identical to Figure4.jl: each panel spans 0 → 2 N∞², so
# the neutral colour still lands on the undisturbed background gradient and the
# panels remain comparable to each other despite the different tick numbers.
#
# Background is the exponential N²_bg(z) = N∞²[1 − exp(−z/L)], L = 10 δ_s = 1.19 m
# (case_params.jl), so the warm band near the seabed at t = 0 is the initial
# condition, not mixing. The dashed line marks L.
#
# Uses the profile data already saved by Tidal3D.jl; reruns no simulation.
# Run from this directory:  julia --project=.. Figure4_dimensional.jl

const ω  = 1.4075235e-4
const ν  = 1.0e-6
const δ  = sqrt(2ν / ω)          # 0.1192 m
const T_tide = 2π / ω            # 44643 s = 12.40 h
const L_strat = 10δ              # 1.192 m

const zmax_m = 40δ               # the paper's 40 δ_s, in metres ≈ 4.77 m

# (label, N∞² used to set the colour range). For Ri = 0 the scalar is passive and
# its background gradient is the reference ω² set in case_params.jl.
cases = [("Ri0",    "(a) Ri = 0  (no stratification)",      1.0 * ω^2),
         ("Ri500",  "(b) Ri = 500",                       500.0 * ω^2),
         ("Ri2500", "(c) Ri = 2500",                     2500.0 * ω^2)]

default(fontfamily = "sans-serif", framestyle = :box, grid = false,
        tickfontsize = 9, guidefontsize = 10, titlefontsize = 11)

const gradient_map = cgrad(["#7A3117", "#B4502C", "#D9855F", "#E9E7E4",
                            "#7FADE0", "#3C7CC4", "#1B4E8F"])

# Superscript digits, so a colourbar can be labelled 10⁻⁸ s⁻² rather than 1e-8.
const supers = Dict('0'=>'⁰','1'=>'¹','2'=>'²','3'=>'³','4'=>'⁴',
                    '5'=>'⁵','6'=>'⁶','7'=>'⁷','8'=>'⁸','9'=>'⁹','-'=>'⁻')
sup(n::Int) = join(get(supers, c, c) for c in string(n))

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

    zg = 0.5 .* (zc[1:end-1] .+ zc[2:end])
    G  = diff(Bmean, dims = 1) ./ diff(zc)        # ∂b/∂z (s⁻²) at midpoints

    ks = findall(z -> z <= zmax_m, zg)
    zm = zg[ks]                                    # metres
    th = times ./ 3600                             # hours

    # One decade per panel, chosen from the case's own N∞², so the ticks read as
    # O(1) numbers instead of 4.95e-6.
    p     = floor(Int, log10(2N²))
    scale = 10.0^p
    Gs    = G[ks, :] ./ scale

    plt = heatmap(th, zm, Gs;
                  clims = (0, 2N² / scale), color = gradient_map,
                  xlabel = "t (hours)", ylabel = "z (m)", title = ttl,
                  ylims = (0, zmax_m),
                  colorbar_title = "  ∂⟨b⟩/∂z  (10$(sup(p)) s⁻²)")
    # Paper's mixed-layer / thermocline markers, at 0.3 and 0.5 of the far-field
    # gradient — dimensional here, so the levels carry N∞².
    contour!(plt, th, zm, Gs; levels = [0.3, 0.5] .* N² ./ scale,
             color = RGB(0.15, 0.15, 0.15), linewidth = 1.2)
    # Background scale height in metres: below this the initial profile is itself
    # weakly stratified, so warm colours there are partly inherited, not mixed.
    hline!(plt, [L_strat]; color = RGB(0.15, 0.15, 0.15), linestyle = :dash,
           linewidth = 1.0, label = "")
    push!(panels, plt)
end

isempty(panels) && error("No profile files found — run the simulations first")

fig = plot(panels...; layout = (length(panels), 1),
           size = (1000, 330 * length(panels)),
           leftmargin = 6Plots.mm, rightmargin = 10Plots.mm,
           bottommargin = 4Plots.mm, topmargin = 3Plots.mm,
           plot_title = @sprintf("Plane-averaged buoyancy gradient, dimensional (δ_s = %.3f m, tidal period = %.2f h)",
                                 δ, T_tide / 3600),
           plot_titlefontsize = 12)
savefig(fig, joinpath(outdir, "Figure4_dimensional.png"))
@info "Saved figures/Figure4_dimensional.png"
