#!/usr/bin/env julia
# =============================================================================
# swirlesrun5.jl — TKE(z, t) animations, one per T, all four N/ω on each frame.
#
# The colleague's Ekman/TKE analysis/TKE.jl animates ONE case per movie: a single
# TKE profile sweeping up and down its axis. That shows the cycle but not the
# comparison, and the comparison is the whole point of this sweep — how far the
# turbulent layer pushes past the pycnocline is what varies with N/ω, and it
# varies WITHIN a tidal cycle, not just in the period-averaged profile.
#
# So: one movie per T, four curves per frame (N/ω = 0, 1, 2, 10), a shared and
# frame-independent x-axis so the curves may be read against each other and
# against themselves at a different phase, and a phase strip underneath saying
# where in the tidal cycle the frame sits.
#
#   outputs/tke_animations/animation_TKE_T5.mp4
#   outputs/tke_animations/animation_TKE_T10.mp4
#
# The name is `animation_*` and it lives under outputs/ because that is the ONLY
# pattern .gitignore re-includes (see the negation at the bottom of it). A movie
# written anywhere else, or under any other name, is invisible to `git add -A`
# and never reaches the workstation.
#
# ---------------------------------------------------------------------------
# WHERE TKE COMES FROM — and why not from the 3D fields
# ---------------------------------------------------------------------------
# From *_moments.jld2, which Moments.jl already writes every T_tide/200:
#
#     TKE(z,t) = ½[(⟨uu⟩ − U²) + (⟨vv⟩ − V²) + ⟨ww⟩]        Centers
#
# ⟨w⟩_xy ≡ 0 exactly (rigid lid, impermeable bottom, incompressibility), so ⟨ww⟩
# needs no mean subtracted; the two horizontal variances do. This is the same
# expression MixedLayerDiffusivity.jl forms, deliberately — one definition of TKE
# in the project, not two.
#
# The 3D fields would give the same thing at ~1.5 GB per case and 16 snapshots,
# against 1601 samples in a file of a few MB. The sweep cases were run with
# FIELDS3D=0 in any case, so the 3D route does not exist for them.
#
# ORDERING. The mean is subtracted PER SAMPLE, then the result is smoothed in
# time. The other order leaks Var_t(U) — the variance of the tidal mean flow over
# the window — straight into "TKE", ~(U₀ωΔt)²/12, the same order as u*² itself.
# Moments.jl's header has the full argument; the boxcar here is applied to TKE
# after the decomposition, never to ⟨uu⟩ before it.
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   GKSwstype=100 julia --project=. swirlesrun5.jl
#
# ENV
#   T_VALUES        "5 10"        one movie per entry
#   N_OVER_OMEGA    "0 1 2 10"    curves within each movie
#   SKIP_PERIODS    1             discard the restart transient
#   STRIDE          2             use every STRIDE-th sample as a frame
#   FPS             30            → 100 frames per tidal period at the defaults
#   SMOOTH_WINDOW   tide20        boxcar width: tide20 | tide | none
#   ZMAX            T + 10        top of the plotted column (m)
#   XMAX            (auto)        TKE axis top; auto = max over the window
#   XSCALE          linear        linear | log   (log needs XMIN)
#   XMIN            1e-9          only used when XSCALE=log
#   SHARE_XLIM      0             1 → the same TKE axis in BOTH movies
#   PREVIEW         0             1 → one PNG per T instead of the movie, to
#                                 check the axes before spending the render
#   OUT_ROOT        outputs
#   ANIM_DIR        outputs/tke_animations
#   FIG_DIR         figures       where PREVIEW writes
# =============================================================================

using Oceananigans, JLD2, Plots, Printf, Statistics, Dates

get!(ENV, "GKSwstype", "100")

const HERE   = @__DIR__
const ω      = 1e-4
const U₀     = 0.04
const T_tide = 2π / ω

parse_list(key, default) = parse.(Float64, split(get(ENV, key, default)))
const T_values = parse_list("T_VALUES", "5 10")
const n_over_ω = parse_list("N_OVER_OMEGA", get(ENV, "SQRT_RI", "0 1 2 10"))

const SKIP     = parse(Float64, get(ENV, "SKIP_PERIODS", "1"))
const STRIDE   = parse(Int,     get(ENV, "STRIDE", "2"))
const FPS      = parse(Int,     get(ENV, "FPS", "30"))
const SMOOTH   = get(ENV, "SMOOTH_WINDOW", "tide20")
const XSCALE   = get(ENV, "XSCALE", "linear")
const XMIN     = parse(Float64, get(ENV, "XMIN", "1e-9"))
const SHARE_XL = get(ENV, "SHARE_XLIM", "0") == "1"
const PREVIEW  = get(ENV, "PREVIEW", "0") == "1"

const smooth_window = SMOOTH == "tide20" ? T_tide / 20 :
                      SMOOTH == "tide"   ? T_tide      :
                      SMOOTH == "none"   ? 0.0         :
                      error("SMOOTH_WINDOW must be tide20, tide or none — got \"$SMOOTH\"")
XSCALE in ("linear", "log") || error("XSCALE must be linear or log — got \"$XSCALE\"")

# Tag builder matching case_params.jl, identical to MixedLayerDiffusivity.jl and
# Figure4_metres.jl. Do not re-derive it here — if the naming ever changes it
# must change in one place.
num_lbl(x) = isinteger(x) ? string(Int(x)) : replace(string(x), "." => "p")
tag_of(T, s) = "P4_T" * num_lbl(T) * "_sqrtRi" * num_lbl(s)
const outroot = get(ENV, "OUT_ROOT", "outputs")
moments_file(tag) = joinpath(HERE, outroot, tag, "TidalBL3D_" * tag * "_moments.jld2")

const animdir = get(ENV, "ANIM_DIR", joinpath(HERE, outroot, "tke_animations"))
# The preview PNGs go to figures/, not next to the movies: everything under
# outputs/ is gitignored except animation_*.mp4, so a preview written there would
# never reach the workstation.
const figdir  = get(ENV, "FIG_DIR", joinpath(HERE, "figures"))
const logdir  = joinpath(HERE, "logs")
mkpath(animdir); mkpath(figdir); mkpath(logdir)

default(fontfamily = "sans-serif", framestyle = :box, grid = true, gridalpha = 0.15,
        tickfontsize = 8, guidefontsize = 10, legendfontsize = 9, titlefontsize = 11)

# Same palette as swirlesrun4.jl, so a curve keeps its colour from the scatter
# plots to the movies.
const CLR = Dict(0.0 => "#8C8C8C", 1.0 => "#1B4E8F", 2.0 => "#2E8B57", 10.0 => "#B4502C")
clr(s) = get(CLR, s, "#000000")

# "−5" → "⁻⁵", so the axis can be labelled "×10⁻⁵ m² s⁻²" rather than carrying
# five leading zeros on every tick.
const SUPS = Dict('-'=>'⁻', '0'=>'⁰', '1'=>'¹', '2'=>'²', '3'=>'³', '4'=>'⁴',
                  '5'=>'⁵', '6'=>'⁶', '7'=>'⁷', '8'=>'⁸', '9'=>'⁹')
sup(n::Integer) = String([SUPS[c] for c in string(n)])

# Centred boxcar of half-width `nh` samples, shrinking at the edges — copied from
# MixedLayerDiffusivity.jl so both scripts smooth identically.
function boxcar(a::AbstractVector, nh)
    nh <= 0 && return collect(float.(a))
    n = length(a)
    out = similar(float.(collect(a)))
    for i in 1:n
        lo, hi = max(1, i - nh), min(n, i + nh)
        good = filter(isfinite, @view a[lo:hi])
        out[i] = isempty(good) ? NaN : mean(good)
    end
    return out
end
boxcar(A::AbstractMatrix, nh) =
    reduce(vcat, (reshape(boxcar(view(A, k, :), nh), 1, :) for k in 1:size(A, 1)))

# ---------------------------------------------------------------------------
# Load one case → TKE(z, t)
# ---------------------------------------------------------------------------
function load_tke(T, s)
    tag = tag_of(T, s)
    fname = moments_file(tag)
    isfile(fname) || (@warn "Missing $fname — $tag will not appear in the T = $T movie"; return nothing)

    ts = Dict(v => FieldTimeSeries(fname, v) for v in ("U", "V", "uu", "vv", "ww"))
    times = ts["U"].times
    nt = length(times)
    nt < 4 && (@warn "$tag has only $nt samples — skipping"; return nothing)

    zc = collect(znodes(ts["U"]))                  # Centers: U V uu vv ww all live here
    grab(v) = (a = zeros(length(interior(ts[v][1])), nt);
               for n in 1:nt; a[:, n] .= vec(interior(ts[v][n])); end; a)
    U, V = grab("U"), grab("V")
    uu, vv, ww = grab("uu"), grab("vv"), grab("ww")

    # (1) decompose per sample — ⟨w⟩ ≡ 0, so ww needs no mean subtracted
    TKE_raw = 0.5 .* ((uu .- U .^ 2) .+ (vv .- V .^ 2) .+ ww)

    # (2) and only then smooth in time
    dt = times[2] - times[1]
    nh = max(0, round(Int, smooth_window / (2dt)))
    TKE = boxcar(TKE_raw, nh)

    @printf("  %-18s %5d samples, %.2f periods, Nz = %3d, boxcar ±%d samples\n",
            tag, nt, times[end] / T_tide, length(zc), nh)
    return (; tag, T, s, times, z = zc, TKE)
end

# Nearest sample in `times` to t. The cases share a writer cadence (T_tide/200)
# and a start time, so in practice this is the identity — it exists so a case
# that was restarted, or stopped early, still lines up on the common axis rather
# than being drawn a frame or two out of phase with the others.
function nearest_index(times, t)
    i = searchsortedfirst(times, t)
    i <= 1 && return 1
    i > length(times) && return length(times)
    return (t - times[i-1]) <= (times[i] - t) ? i - 1 : i
end

# ---------------------------------------------------------------------------
# One movie
# ---------------------------------------------------------------------------
function animate_T(T, cases, xmax_shared)
    isempty(cases) && (@warn "No cases loaded for T = $T — no movie written"; return nothing)

    zmax = parse(Float64, get(ENV, "ZMAX", string(T + 10)))

    # Common frame axis: the samples of the SHORTEST case, past SKIP, so no frame
    # asks a case for a time it does not have.
    ref = argmin(c -> c.times[end], cases)
    t_end = minimum(c.times[end] for c in cases)
    frame_times = [t for t in ref.times if SKIP * T_tide <= t <= t_end][1:STRIDE:end]
    isempty(frame_times) && (@warn "SKIP_PERIODS = $SKIP leaves no samples for T = $T"; return nothing)

    # Per-case index into its own time axis, once, rather than per frame.
    idx = Dict(c.tag => [nearest_index(c.times, t) for t in frame_times] for c in cases)

    # x-axis fixed across every frame AND every case: a curve that shrinks by half
    # must look half as long, which it cannot do on an axis that rescales with it.
    # Only the plotted column counts towards the maximum — the near-wall peak at
    # z < 0.1 m is an order above anything at the interface and would flatten the
    # part of the profile the movie is about... except it is IN the plotted column,
    # so it does set the scale. Use XMAX to crop to the interface instead.
    auto_xmax = 0.0
    for c in cases
        kz = findall(z -> z <= zmax, c.z)
        for n in idx[c.tag]
            v = @view c.TKE[kz, n]
            g = filter(isfinite, v)
            isempty(g) || (auto_xmax = max(auto_xmax, maximum(g)))
        end
    end
    xmax = xmax_shared !== nothing ? xmax_shared :
           parse(Float64, get(ENV, "XMAX", string(1.05 * auto_xmax)))

    # TKE here runs to ~1e-4 m² s⁻², which on a linear axis prints as five
    # leading zeros on every tick. Pull the leading decade out into the axis
    # label instead, so the ticks read 0, 2, 4 … A log axis needs the real
    # numbers, so it keeps them.
    xfac = (XSCALE == "log" || !(xmax > 0)) ? 1.0 : 10.0^floor(Int, log10(xmax))
    xlab = xfac == 1.0 ? "TKE  (m² s⁻²)" :
           "TKE  (×10" * sup(floor(Int, log10(xmax))) * " m² s⁻²)"
    xlims = XSCALE == "log" ? (XMIN, xmax) : (0.0, xmax / xfac)

    @printf("T = %.0f: %d frames (%.2f–%.2f periods, stride %d), z ≤ %.1f m, TKE ≤ %.3e m²/s²\n",
            T, length(frame_times), frame_times[1] / T_tide, frame_times[end] / T_tide,
            STRIDE, zmax, xmax)

    # The phase strip's own axis, drawn once and reused: the full animated window
    # of U∞ = U₀ sin(ωt), with a dot marking the frame.
    t_line = range(frame_times[1], frame_times[end], length = 600)
    U_line = U₀ .* sin.(ω .* t_line)

    # One frame, as a function, so PREVIEW=1 renders exactly the frame the movie
    # would contain rather than an approximation of it.
    function draw_frame(i)
        t = frame_times[i]

        p1 = plot(xlabel = xlab, ylabel = "z  (m)",
                  xlims = xlims, ylims = (0, zmax),
                  xscale = XSCALE == "log" ? :log10 : :identity,
                  legend = :topright, legendtitle = "N/ω",
                  title = @sprintf("TKE profiles, T = %g m   |   ωt/2π = %.2f   |   U∞/U₀ = %+.2f",
                                   T, t / T_tide, sin(ω * t)))

        # The pycnocline. Below z = T the softplus background is unstratified in
        # every case, so any difference between the curves down there is not
        # stratification acting locally — it is the interface above throttling the
        # layer beneath it. Drawing the line makes that readable. It is labelled by
        # an annotation rather than a legend entry: the legend is the N/ω key, and
        # a dashed line sitting inside it reads as a fifth case.
        hline!(p1, [T]; color = :black, ls = :dash, lw = 1.2, label = "")
        annotate!(p1, xlims[1] + 0.02 * (xlims[2] - xlims[1]), T + 0.03 * zmax,
                  text("z = T", 8, :left, :bottom, :black))

        for c in cases
            plot!(p1, c.TKE[:, idx[c.tag][i]] ./ xfac, c.z;
                  lw = 2, color = clr(c.s), label = @sprintf("%g", c.s))
        end

        p2 = plot(t_line ./ T_tide, U_line ./ U₀;
                  color = :black, lw = 1.2, label = "",
                  xlabel = "t / T_tide", ylabel = "U∞/U₀",
                  ylims = (-1.15, 1.15), yticks = [-1, 0, 1],
                  xlims = (frame_times[1] / T_tide, frame_times[end] / T_tide))
        hline!(p2, [0]; color = :grey, lw = 0.8, label = "")
        scatter!(p2, [t / T_tide], [sin(ω * t)];
                 ms = 6, msw = 0, color = "#B4502C", label = "")

        plot(p1, p2, layout = grid(2, 1, heights = [0.82, 0.18]),
             size = (900, 760), left_margin = 6Plots.mm, bottom_margin = 5Plots.mm)
    end

    # PREVIEW=1 draws the middle frame to a PNG and stops. Rendering the full
    # movie is minutes of GR per T, so check the axes, the legend and the z range
    # on one frame before spending it — especially when XMAX or ZMAX has been set
    # by hand and might have cropped the profile out of the box.
    if PREVIEW
        out = joinpath(figdir, "preview_TKE_T" * num_lbl(T) * ".png")
        savefig(draw_frame(cld(length(frame_times), 2)), out)
        @info "PREVIEW=1 — wrote $(relpath(out, HERE)) and skipped the movie"
        return (; T, out, nframes = 1, xmax, zmax, tags = [c.tag for c in cases])
    end

    anim = @animate for i in eachindex(frame_times)
        i % 100 == 0 && @info @sprintf("T = %.0f: frame %d / %d (%.2f periods)",
                                       T, i, length(frame_times), frame_times[i] / T_tide)
        draw_frame(i)
    end

    out = joinpath(animdir, "animation_TKE_T" * num_lbl(T) * ".mp4")
    mp4(anim, out, fps = FPS)
    @info "wrote $(relpath(out, HERE))"
    return (; T, out, nframes = length(frame_times), xmax, zmax,
            tags = [c.tag for c in cases])
end

# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------
println("Loading moments for T ∈ $(T_values), N/ω ∈ $(n_over_ω)")
loaded = Dict{Float64,Vector{Any}}()
for T in T_values
    cs = Any[]
    for s in n_over_ω
        c = load_tke(T, s)
        c === nothing || push!(cs, c)
    end
    loaded[T] = cs
end

# SHARE_XLIM makes the two movies directly comparable to each other, at the cost
# of whichever T has the weaker turbulence using only part of its axis.
function global_xmax(loaded)
    m = 0.0
    for T in T_values, c in get(loaded, T, Any[])
        zmax = parse(Float64, get(ENV, "ZMAX", string(T + 10)))
        kz = findall(z -> z <= zmax, c.z)
        keep = c.times .>= SKIP * T_tide
        g = filter(isfinite, @view c.TKE[kz, keep])
        isempty(g) || (m = max(m, maximum(g)))
    end
    return 1.05 * m
end

xmax_shared = nothing
if SHARE_XL
    xmax_shared = global_xmax(loaded)
    @info @sprintf("SHARE_XLIM=1 — both movies use TKE ∈ [0, %.3e] m² s⁻²", xmax_shared)
end

results = [r for r in (animate_T(T, get(loaded, T, Any[]), xmax_shared) for T in T_values)
           if r !== nothing]

open(joinpath(logdir, "tke_animations.log"), "a") do io
    for out in (io, stdout)
        println(out, "\n", "="^88)
        @printf(out, "==== %s  swirlesrun5.jl  SKIP_PERIODS=%g STRIDE=%d SMOOTH=%s FPS=%d\n",
                Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), SKIP, STRIDE, SMOOTH, FPS)
        println(out, "="^88)
        for r in results
            @printf(out, "T = %-4g  %4d frames  z ≤ %5.1f m  TKE ≤ %.3e  cases: %s\n    %s\n",
                    r.T, r.nframes, r.zmax, r.xmax, join(r.tags, ", "), relpath(r.out, HERE))
        end
        isempty(results) && println(out, "no movies written — no moments files found")
        println(out, "="^88)
    end
end
