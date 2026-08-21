using Oceananigans, JLD2, Plots, Printf

# Figure 4 for the softplus sweep, with the depth axis in metres.
# Background: N²_bg(z) = N∞²·sigmoid(sharp·(z−T)) — unstratified below the
# pycnocline at z = T, N∞² above it, over a transition of width ~1/sharp.
#
# A time–depth heatmap of the plane-averaged buoyancy gradient ∂⟨b⟩/∂z normalized
# by the far-field N∞² = Ri·ω². The stratification is labelled by N/ω = √Ri.
#
# Produces, in figures/:
#   Figure4_softplus_T<T>.png   one per T, columns = N/ω     (the main output)
#   Figure4_softplus_sweep.png  all T stacked, rows = T      (overview)
#
# The colorbars are drawn by hand, as a narrow extra subplot beside each heatmap.
# GR sizes a built-in colorbar from the figure width rather than the subplot
# width, so in a six-column layout each one takes about six times its share and
# squeezes the heatmaps flat. Keep colorbar = false and the hand-drawn ribbon
# unless the layout is only one or two columns wide.
#
# Reads only the *_profiles.jld2 files Tidal3D.jl writes, under outputs/<tag>/.
#   julia --project=. Figure4_metres.jl
# Subsets:  T_VALUES="10 20" N_OVER_OMEGA="1 2 5" julia --project=. Figure4_metres.jl
# Taller:   FIG4_ZMAX=30 julia --project=. Figure4_metres.jl
#
# The depth window is a plotting choice only: the profiles files hold the whole
# column, so any window up to Lz can be drawn from data already on disk without
# re-running anything. FIG4_ZMAX sets one window for every panel, clamped to Lz
# because the sponge damps the flow and its gradients describe the boundary
# condition rather than the physics.

const ω  = 1e-4
const f  = ω
const ν  = 1.0e-6
const T_tide = 2π / ω
const δs = sqrt(2ν / ω)           # Stokes thickness ≈ 0.1414 m

# These must match case_params.jl: the panel titles and the pycnocline marker
# only mean anything if they describe the profile the run actually used.
# The archived runs under "-Softplus T sweep, no-slip bottom/" were made with
# SHARP = 6, so redrawing those figures needs SHARP=6 set explicitly.
const sharp = parse(Float64, get(ENV, "SHARP", "2"))
const Lz    = 50.0

parse_list(key, default) = parse.(Float64, split(get(ENV, key, default)))
const T_values = parse_list("T_VALUES", "5 10 20 30")
# N/ω and √Ri are the same number; SQRT_RI is still accepted so the sweep driver
# keeps working unchanged.
const n_over_ω = parse_list("N_OVER_OMEGA", get(ENV, "SQRT_RI", "0 0.5 1 2 5 10"))

# Tag builder matching case_params.jl: T = 5.0 → "5", N/ω = 0.5 → "0p5", giving
# e.g. "P4_T5_sqrtRi0p5". The folder names keep the sqrtRi spelling, since they
# name data already on disk, while every label the reader sees says N/ω.
num_lbl(x) = isinteger(x) ? string(Int(x)) : replace(string(x), "." => "p")
tag_of(T, s) = "P4_T" * num_lbl(T) * "_sqrtRi" * num_lbl(s)
const outroot = get(ENV, "OUT_ROOT", "outputs")
profiles_file(tag) = joinpath(@__DIR__, outroot, tag, "TidalBL3D_" * tag * "_profiles.jld2")

# Depth window per row. Left unset it follows Lz_test in case_params.jl, so the
# figure shows the same section as the animations. FIG4_ZMAX replaces that with
# one fixed height for every panel, which is what makes rows comparable when
# several T are drawn: the default window depends on T and so gives each row its
# own vertical scale.
const zmax_override = let v = get(ENV, "FIG4_ZMAX", "")
    isempty(v) ? nothing : min(Lz, parse(Float64, v))
end
zmax_of(T) = zmax_override === nothing ? min(Lz, max(70δs, T + 10)) : zmax_override

# Canvas size in pixels: FIG4_WIDTH is per case column (heatmap plus its colorbar
# ribbon), FIG4_HEIGHT the whole per-T figure, FIG4_ROW_HEIGHT one row of the
# overview. Together they set the aspect ratio of a panel, so raising FIG4_ZMAX
# without changing the canvas stretches each panel into a tall strip. As a guide,
# 640 × 390 at FIG4_ZMAX = 30 gives the same shape as the 480 × 370 default does
# over 0–15 m.
const fig_width   = parse(Int, get(ENV, "FIG4_WIDTH", "480"))
const fig_height  = parse(Int, get(ENV, "FIG4_HEIGHT", "370"))
const row_height  = parse(Int, get(ENV, "FIG4_ROW_HEIGHT", "275"))

# The margins scale with the canvas, and every mm below is a multiple of this.
# GR sizes its text from the canvas as a whole, so a wider figure gets larger
# text while a margin in mm stays the same number of pixels, and a fixed margin
# eventually stops clearing the axis labels. The reference is the 6 × 480 px
# canvas the values were tuned on; only the width matters, since the labels that
# overflow are set in the figure's font size.
const ncol = length(n_over_ω)
const margin_scale = max(1.0, fig_width * ncol / 2880)

const gradient_map = cgrad(["#7A3117", "#B4502C", "#D9855F", "#E9E7E4",
                            "#7FADE0", "#3C7CC4", "#1B4E8F"])

default(fontfamily = "sans-serif", framestyle = :box, grid = false,
        tickfontsize = 7, guidefontsize = 8, titlefontsize = 9)

outdir = joinpath(@__DIR__, "figures")
mkpath(outdir)

# δ = u*/f from the peak wall stress over the final period, for the title only.
function delta_ustar(U_ts, zc, times)
    z1 = zc[1]
    uτ = [sqrt(ν * abs(interior(U_ts[n])[1, 1, 1]) / z1) for n in 1:length(times)]
    ustar = maximum(uτ[times .>= (times[end] - T_tide)])
    return ustar / f
end

# ---------------- Load every case once ----------------
# Both the per-T figures and the combined overview are built from this cache, so
# each profiles file is read only once.
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

    # Normalize by N²_ref, as in case_params.jl: at N/ω = 0 the buoyancy is zero
    # but b is still carried as a passive scalar, so ω² is the reference there.
    # Dividing by Ri·ω² would blank the whole N/ω = 0 column.
    Ri = s^2
    N²_ref = Ri > 0 ? Ri * ω^2 : ω^2
    zmax = zmax_of(T)
    ks = findall(z -> z <= zmax, zg)

    Gn = G[ks, :] ./ N²_ref
    # Colour limits per panel: with the background gradient going to zero below
    # the pycnocline, a shared scale would flatten the near-bed structure into a
    # single colour. The range is printed in the title, since per-panel limits
    # cost comparability between panels.
    gmin, gmax = extrema(filter(isfinite, Gn))
    if gmax - gmin < 1e-12
        pad = max(abs(gmax), 1.0) * 1e-3
        gmin, gmax = gmin - pad, gmax + pad
    end

    cases[(T, s)] = (ωt = times .* ω, z = zg[ks], Gn = Gn, zmax = zmax,
                     gmin = gmin, gmax = gmax, δ = delta_ustar(U_ts, zc, times))
end

isempty(cases) && error("No profile files found under $outroot/ — run the simulations first")

# ---------------- Panel construction ----------------

# A colorbar drawn as its own subplot: a two-column heatmap of its value range,
# with the ticks mirrored to the right so they label the ribbon.
function cbar_panel(lo, hi)
    yy = collect(range(lo, hi, length = 256))
    heatmap([1, 2], yy, repeat(yy, 1, 2);
            color = gradient_map, clims = (lo, hi), colorbar = false,
            xticks = false, xlims = (0.5, 2.5), ylims = (lo, hi),
            ymirror = true, framestyle = :box, tickfontsize = 6,
            # Margins are set per subplot rather than on the outer plot: a
            # global left margin wide enough for the "z (m)" label would also
            # indent every ribbon, leaving it away from its own ticks.
            leftmargin = 0Plots.mm, rightmargin = margin_scale * 1Plots.mm)
end

blank() = plot(framestyle = :none, grid = false, showaxis = false)

# ωt ticks. The step is chosen from the length of the run, so there are roughly
# five to seven labels whatever the run length.
xtick_step(ωt_max) = ωt_max <= 30 ? 5 : 10

# `first_col` carries the z axis labels, `show_xlabel` marks the bottom row.
function case_panels(T, s; first_col, show_xlabel)
    c = get(cases, (T, s), nothing)
    if c === nothing
        # Two cells per case, so a missing case must still fill both or every
        # panel after it shifts into the wrong column.
        return (plot(title = @sprintf("T=%d m, N/ω=%g\n(no data)", Int(T), s),
                     framestyle = :box, showaxis = false, grid = false), blank())
    end

    # One line, so the panels stay short and wide. The colour range is not
    # repeated here, since the colorbar ticks already show it.
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

    # The initial pycnocline, drawn as the dashed line the title refers to. It
    # matters more the taller the window is, since with the panel top far above
    # z = T the axis alone no longer shows where the interface started.
    T < c.zmax && hline!(plt, [T]; color = :black, linestyle = :dash,
                         linewidth = 1, label = "", legend = false)
    return (plt, cbar_panel(c.gmin, c.gmax))
end

# Column widths set explicitly: the heatmap gets about 3.5 times the colorbar
# cell, which also has to hold the ribbon's tick labels. ncol is defined with the
# canvas settings above, since the margin scale needs it.
column_widths(n) = (w = repeat([0.82, 0.18], n); w ./ sum(w))

# ---------------- One figure per T ----------------
for T in T_values
    any(haskey(cases, (T, s)) for s in n_over_ω) || continue
    panels = []
    for (si, s) in enumerate(n_over_ω)
        p, cb = case_panels(T, s; first_col = si == 1, show_xlabel = true)
        push!(panels, p, cb)
    end
    # Short and wide. The figure title takes a fixed fraction of the height, so
    # on a short figure it needs a larger share or it collides with the panel
    # titles.
    fig = plot(panels...; layout = grid(1, 2ncol, widths = column_widths(ncol)),
               size = (fig_width * ncol, fig_height),
               bottommargin = margin_scale * 14Plots.mm,
               topmargin = margin_scale * 4Plots.mm,
               plot_title = @sprintf("Buoyancy gradient ∂⟨b⟩/∂z / N²  —  T = %d m  (dashed = initial pycnocline z = T)", Int(T)),
               # About 55 px of title, given as the fraction the argument
               # wants. A fixed fraction would eat a tenth of the panels once
               # FIG4_HEIGHT is raised.
               plot_titlefontsize = 14, plot_titlevspan = min(0.15, 55 / fig_height))
    fname = joinpath(outdir, "Figure4_softplus_T" * num_lbl(T) * ".png")
    savefig(fig, fname)
    @info "Saved figures/" * basename(fname)
end

# ---------------- Combined overview: all T stacked ----------------
# SKIP_SWEEP=1 leaves the existing overview alone. It is written from whatever
# T_VALUES holds, so a run over a subset of the sweep would otherwise replace the
# full figure with a partial one. The per-T figures above are unaffected.
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
