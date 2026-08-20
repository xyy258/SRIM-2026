using Oceananigans, JLD2, Plots, Printf

# Figure 4 for the SOFTPLUS sweep, depth axis in PHYSICAL METRES.
# Background: N²_bg(z) = N∞²·sigmoid(sharp·(z−T)) — unstratified below the
# pycnocline at z = T, N∞² above, over a transition of width ~1/sharp.
#
# Time–depth heatmap of the plane-averaged buoyancy gradient ∂⟨b⟩/∂z normalized
# by the far-field N∞² = Ri·ω². Stratification is labelled by N/ω = √Ri.
#
# Produces, in figures/:
#   Figure4_softplus_T<T>.png   one per T, columns = N/ω          (the main output)
#   Figure4_softplus_sweep.png  all T stacked, rows = T           (overview)
#
# COLORBARS ARE DRAWN BY HAND, as a narrow extra subplot beside each heatmap.
# GR sizes a subplot's built-in colorbar from the FIGURE width, not the subplot
# width, so in a 6-column layout each colorbar claimed ~6x its proper share and
# squeezed every heatmap into an unreadable sliver. Passing colorbar = false and
# building the ribbon as a heatmap in its own grid cell puts the widths back
# under our control. Do not revert to `colorbar_title` / the automatic colorbar
# unless the layout is one or two columns wide.
#
# Reads only *_profiles.jld2 written by Tidal3D.jl, under outputs/<tag>/.
#   julia --project=. Figure4_metres.jl
# Subsets:  T_VALUES="10 20" N_OVER_OMEGA="1 2 5" julia --project=. Figure4_metres.jl
# Taller:   FIG4_ZMAX=30 julia --project=. Figure4_metres.jl
#
# THE DEPTH WINDOW IS A PLOTTING CHOICE ONLY. The profiles files hold the plane
# average over the WHOLE column — the 50 m physical domain plus the 10 m sponge —
# so any window up to Lz can be drawn from data already on disk, without re-running
# a single case. FIG4_ZMAX (metres) overrides the default window for every panel;
# it is clamped to Lz, since the sponge damps the flow and its gradients describe
# the boundary condition rather than the physics.

const ω  = 1e-4
const f  = ω
const ν  = 1.0e-6
const T_tide = 2π / ω
const δs = sqrt(2ν / ω)           # Stokes thickness ≈ 0.1414 m

# These must match case_params.jl — the panel titles and the pycnocline marker
# are only meaningful if they describe the profile the run actually used. The default tracks
# case_params.jl, which changed from 6 to 2 for the K_T / TKE study: at sharp = 6
# the transition width 2ln9/sharp = 0.73 m is thinner than the grid above z ≈ 10 m,
# so the pycnocline began life as a numerical step at large T.
#
# EVERY RUN ARCHIVED UNDER "-Softplus T sweep, no-slip bottom/" WAS MADE AT
# SHARP = 6. Redrawing those figures needs SHARP=6 set explicitly, or the overlay
# will not be the initial condition they actually started from.
const sharp = parse(Float64, get(ENV, "SHARP", "2"))
const Lz    = 50.0

parse_list(key, default) = parse.(Float64, split(get(ENV, key, default)))
const T_values = parse_list("T_VALUES", "5 10 20 30")
# N/ω and √Ri are the same number; SQRT_RI is still honoured so the existing
# sweep driver keeps working unchanged.
const n_over_ω = parse_list("N_OVER_OMEGA", get(ENV, "SQRT_RI", "0 0.5 1 2 5 10"))

# Tag builder matching case_params.jl: T = 5.0 → "5", N/ω = 0.5 → "0p5",
# giving e.g. "P4_T5_sqrtRi0p5". The directory tags keep the sqrtRi spelling —
# they name data already on disk — while every label the reader sees says N/ω.
num_lbl(x) = isinteger(x) ? string(Int(x)) : replace(string(x), "." => "p")
tag_of(T, s) = "P4_T" * num_lbl(T) * "_sqrtRi" * num_lbl(s)
const outroot = get(ENV, "OUT_ROOT", "outputs")
profiles_file(tag) = joinpath(@__DIR__, outroot, tag, "TidalBL3D_" * tag * "_profiles.jld2")

# Depth window per row. Unset, it mirrors Lz_test in case_params.jl so the figure
# shows the same section the animations do: the pycnocline at z = T plus headroom.
# FIG4_ZMAX replaces that with one fixed height for every panel, which is also what
# makes rows comparable when several T are drawn — the default window is T-dependent
# and so gives each row its own vertical scale.
const zmax_override = let v = get(ENV, "FIG4_ZMAX", "")
    isempty(v) ? nothing : min(Lz, parse(Float64, v))
end
zmax_of(T) = zmax_override === nothing ? min(Lz, max(70δs, T + 10)) : zmax_override

# Canvas size in pixels: FIG4_WIDTH is per case COLUMN (heatmap plus its colorbar
# ribbon), FIG4_HEIGHT the whole per-T figure, FIG4_ROW_HEIGHT one row of the
# overview. Together they set the panel ASPECT, which is worth choosing rather than
# inheriting: raising FIG4_ZMAX on an unchanged canvas stretches each panel
# vertically until it reads as a portrait strip, since the data fills whatever box
# it is given. Roughly, panel width ≈ 0.82·FIG4_WIDTH − 75 px of guide and margin,
# panel height ≈ FIG4_HEIGHT − 145 px of title, axis and margins. 640 × 390 at
# FIG4_ZMAX = 30 lands at the same ~1.6:1 the 480 × 370 default did over 0–15 m.
const fig_width   = parse(Int, get(ENV, "FIG4_WIDTH", "480"))
const fig_height  = parse(Int, get(ENV, "FIG4_HEIGHT", "370"))
const row_height  = parse(Int, get(ENV, "FIG4_ROW_HEIGHT", "275"))

# MARGINS ARE SCALED WITH THE CANVAS, and every mm below is a multiple of this.
# GR sizes its glyphs from the canvas as a whole, so a wider figure gets larger
# text while a margin in mm stays the same number of pixels — at 6 × 620 px the
# fixed 12 mm bottom margin no longer cleared the "ωt" label and it was rendered
# off the bottom edge. The reference is the 6 × 480 px canvas the values were
# tuned on. Only the width enters: the labels that overflowed are set in the
# figure's font size, which does not track the height.
const ncol = length(n_over_ω)
const margin_scale = max(1.0, fig_width * ncol / 2880)

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

# ---------------------------------------------------------------------------
# Load every case once. Both the per-T figures and the combined overview are
# built from this cache, so each profiles file is read a single time.
# ---------------------------------------------------------------------------
cases = Dict{Tuple{Float64,Float64},NamedTuple}()
for T in T_values, s in n_over_ω
    tag   = tag_of(T, s)
    fname = profiles_file(tag)
    isfile(fname) || (@warn "Missing $fname — $tag will render as an empty panel"; continue)

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

    # Normalize by N²_ref, matching case_params.jl: at N/ω = 0 the buoyancy is
    # zero but b is still advected as a passive scalar carrying the ω²-scaled
    # softplus background, so ω² is the reference there. Dividing by Ri·ω² would
    # be a divide-by-zero and blank the whole N/ω = 0 column.
    Ri = s^2
    N²_ref = Ri > 0 ? Ri * ω^2 : ω^2
    zmax = zmax_of(T)
    ks = findall(z -> z <= zmax, zg)

    Gn = G[ks, :] ./ N²_ref
    # Per-panel colour limits: with N²_bg → 0 below the pycnocline, a shared scale
    # would flatten every panel's near-bed structure into one colour. The range is
    # printed in the title since that costs cross-panel comparability.
    gmin, gmax = extrema(filter(isfinite, Gn))
    if gmax - gmin < 1e-12
        pad = max(abs(gmax), 1.0) * 1e-3
        gmin, gmax = gmin - pad, gmax + pad
    end

    cases[(T, s)] = (ωt = times .* ω, z = zg[ks], Gn = Gn, zmax = zmax,
                     gmin = gmin, gmax = gmax, δ = delta_ustar(U_ts, zc, times))
end

isempty(cases) && error("No profile files found under $outroot/ — run the simulations first")

# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

# A colorbar as a standalone subplot: a 2-column heatmap of its own value range,
# ticks mirrored to the right so they read as labels on the ribbon.
function cbar_panel(lo, hi)
    yy = collect(range(lo, hi, length = 256))
    heatmap([1, 2], yy, repeat(yy, 1, 2);
            color = gradient_map, clims = (lo, hi), colorbar = false,
            xticks = false, xlims = (0.5, 2.5), ylims = (lo, hi),
            ymirror = true, framestyle = :box, tickfontsize = 6,
            # Margins are set per subplot, not on the outer plot call: a global
            # left margin wide enough for the "z (m)" guide would also inset every
            # ribbon, stranding it half a cell from its own tick labels.
            leftmargin = 0Plots.mm, rightmargin = margin_scale * 1Plots.mm)
end

blank() = plot(framestyle = :none, grid = false, showaxis = false)

# ωt ticks. A fixed 0:5:25 covered a 4-period run exactly and labelled only the
# first half of an 8-period one, so the step is chosen from the run length to
# leave roughly five to seven labels either way.
xtick_step(ωt_max) = ωt_max <= 30 ? 5 : 10

# `first_col` gets the z axis labels; `show_xlabel` marks the bottom row.
function case_panels(T, s; first_col, show_xlabel)
    c = get(cases, (T, s), nothing)
    if c === nothing
        # Two cells per case, so a missing one still has to occupy both or every
        # panel after it shifts into the wrong column.
        return (plot(title = @sprintf("T=%d m, N/ω=%g\n(no data)", Int(T), s),
                     framestyle = :box, showaxis = false, grid = false), blank())
    end

    # One line, so the panels can be short and wide like the earlier Figure 4s.
    # The colour range is not repeated here — the colorbar ticks already show it.
    ttl = @sprintf("T=%d m, N/ω=%g (Ri=%g)  (δ=u*/f=%.2f m)", Int(T), s, s^2, c.δ)
    plt = heatmap(c.ωt, c.z, c.Gn;
                  clims = (c.gmin, c.gmax), color = gradient_map, colorbar = false,
                  xlims = (0, maximum(c.ωt)), ylims = (0, c.zmax),
                  xticks = 0:xtick_step(maximum(c.ωt)):maximum(c.ωt),
                  xlabel = show_xlabel ? "ωt" : "",
                  ylabel = first_col ? "z (m)" : "",
                  yformatter = first_col ? :auto : (_ -> ""),
                  leftmargin = margin_scale * (first_col ? 18Plots.mm : 3Plots.mm),
                  rightmargin = margin_scale * 1Plots.mm,
                  title = ttl)

    # The initial pycnocline, which the figure title has always announced as a
    # dashed line. It matters more the taller the window is: with the panel top far
    # above z = T there is no longer any reading off the axis where the initial
    # interface sat relative to the mixed layer that grew from the bed.
    T < c.zmax && hline!(plt, [T]; color = :black, linestyle = :dash,
                         linewidth = 1, label = "", legend = false)
    return (plt, cbar_panel(c.gmin, c.gmax))
end

# Explicit column widths: heatmap gets ~3.5x the colorbar cell (which also has to
# hold the ribbon's tick labels). ncol is defined with the canvas knobs above,
# since the margin scale needs it.
column_widths(n) = (w = repeat([0.82, 0.18], n); w ./ sum(w))

# ---------------------------------------------------------------------------
# One figure per T
# ---------------------------------------------------------------------------
for T in T_values
    any(haskey(cases, (T, s)) for s in n_over_ω) || continue
    panels = []
    for (si, s) in enumerate(n_over_ω)
        p, cb = case_panels(T, s; first_col = si == 1, show_xlabel = true)
        push!(panels, p, cb)
    end
    # Short and wide, matching the aspect of the earlier Figure 4s. The figure
    # title takes a fixed FRACTION of the height, so on a short figure it has to
    # be given a bigger share explicitly or it collides with the panel titles.
    fig = plot(panels...; layout = grid(1, 2ncol, widths = column_widths(ncol)),
               size = (fig_width * ncol, fig_height),
               bottommargin = margin_scale * 14Plots.mm,
               topmargin = margin_scale * 4Plots.mm,
               plot_title = @sprintf("Buoyancy gradient ∂⟨b⟩/∂z / N²  —  T = %d m  (dashed = initial pycnocline z = T)", Int(T)),
               # ~55 px of title, expressed as the fraction the argument wants: a
               # fixed 0.15 was tuned against the 370 px canvas and would eat a
               # tenth of the panels once FIG4_HEIGHT is raised.
               plot_titlefontsize = 14, plot_titlevspan = min(0.15, 55 / fig_height))
    fname = joinpath(outdir, "Figure4_softplus_T" * num_lbl(T) * ".png")
    savefig(fig, fname)
    @info "Saved figures/" * basename(fname)
end

# ---------------------------------------------------------------------------
# Combined overview: all T stacked
# ---------------------------------------------------------------------------
# SKIP_SWEEP=1 leaves the existing overview alone. It is written unconditionally
# from whatever T_VALUES holds, so a run over a SUBSET of the sweep (one column,
# or the T = 5/10 pair) would otherwise replace the full four-T figure with a
# partial one — the per-T figures above are unaffected either way.
if get(ENV, "SKIP_SWEEP", "0") == "1"
    @info "SKIP_SWEEP=1 — leaving figures/Figure4_softplus_sweep.png untouched"
    exit(0)
end

nrow = length(T_values)
panels = []
for (ti, T) in enumerate(T_values), (si, s) in enumerate(n_over_ω)
    p, cb = case_panels(T, s; first_col = si == 1, show_xlabel = ti == nrow)
    push!(panels, p, cb)
end
fig = plot(panels...; layout = grid(nrow, 2ncol, widths = column_widths(ncol)),
           size = (fig_width * ncol, row_height * nrow),
           bottommargin = margin_scale * 9Plots.mm, topmargin = margin_scale * 4Plots.mm,
           plot_title = "Buoyancy gradient ∂⟨b⟩/∂z / N²  — softplus sweep (dashed = initial pycnocline z = T)",
           plot_titlefontsize = 14, plot_titlevspan = 0.045)
savefig(fig, joinpath(outdir, "Figure4_softplus_sweep.png"))
@info "Saved figures/Figure4_softplus_sweep.png ($(length(cases)) of $(length(T_values)*ncol) cases found)"
