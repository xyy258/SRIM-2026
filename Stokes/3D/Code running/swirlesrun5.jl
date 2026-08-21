#!/usr/bin/env julia
# TKE(z, t) animations, one movie per T, with all four N/ω on each frame.
#
# The Ekman TKE animation shows one case per movie: a single TKE profile sweeping
# up and down its axis. That shows the cycle but not the comparison, and the
# comparison is the point of this sweep — how far the turbulent layer pushes past
# the pycnocline varies with N/ω, and it varies within a tidal cycle rather than
# only in the period-averaged profile.
#
# So each movie has four curves per frame (N/ω = 0, 1, 2, 10), a single x-axis
# shared by every frame and every case, so the curves can be compared with each
# other and with themselves at another phase, and a strip underneath showing
# where in the tidal cycle the frame sits.
#
#   outputs/tke_animations/animation_TKE_T5_log.mp4
#   outputs/tke_animations/animation_TKE_T10_log.mp4
#
# The name must start with `animation_` and live under outputs/, since that is
# the only pattern .gitignore lets back in. A movie written anywhere else never
# reaches the workstation.
#
# ---------------- Where TKE comes from ----------------
# From the *_moments.jld2 files Moments.jl writes every T_tide/200:
#
#     TKE(z,t) = ½[(⟨uu⟩ − U²) + (⟨vv⟩ − V²) + ⟨ww⟩]        at Centers
#
# ⟨w⟩_xy is zero with a rigid lid, an impermeable bottom and incompressibility,
# so ⟨ww⟩ needs no mean subtracted while the two horizontal variances do. This is
# the same expression MixedLayerDiffusivity.jl uses, so that there is one
# definition of TKE in the project rather than two.
#
# The 3D fields would give the same thing from 16 snapshots at about 1.5 GB per
# case, against 1601 samples in a file of a few MB, and the sweep cases were run
# with FIELDS3D=0 in any case.
#
# The mean is subtracted per sample and only then is the result smoothed in time.
# The other order leaks the variance of the tidal mean flow into "TKE" — see the
# header of Moments.jl.
#
# ---------------- Usage ----------------
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
#   XSCALE          log           log | linear
#   DECADES         4             log only: how many decades below XMAX the axis
#                                 floor sits, so a dead cell above the interface
#                                 does not squash it into the bottom pixel
#   XMIN            (auto)        log only: a fixed floor, overriding DECADES
#   SHARE_XLIM      0             1 → the same TKE axis in both movies
#   PREVIEW         0             1 → one PNG per T instead of the movie, to
#                                 check the axes before rendering
#   OUT_ROOT        outputs
#   ANIM_DIR        outputs/tke_animations
#   FIG_DIR         figures       where PREVIEW writes

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
const XSCALE   = get(ENV, "XSCALE", "log")
const DECADES  = parse(Int, get(ENV, "DECADES", "4"))
# Empty by default rather than a number: the floor is derived from XMAX and
# DECADES unless it is set, since no single value suits both T = 5 and T = 10.
const XMIN_ENV = get(ENV, "XMIN", "")
const SHARE_XL = get(ENV, "SHARE_XLIM", "0") == "1"
const PREVIEW  = get(ENV, "PREVIEW", "0") == "1"

const smooth_window = SMOOTH == "tide20" ? T_tide / 20 :
                      SMOOTH == "tide"   ? T_tide      :
                      SMOOTH == "none"   ? 0.0         :
                      error("SMOOTH_WINDOW must be tide20, tide or none — got \"$SMOOTH\"")
XSCALE in ("linear", "log") || error("XSCALE must be linear or log — got \"$XSCALE\"")

# Tag builder matching case_params.jl, MixedLayerDiffusivity.jl and
# Figure4_metres.jl.
num_lbl(x) = isinteger(x) ? string(Int(x)) : replace(string(x), "." => "p")
tag_of(T, s) = "P4_T" * num_lbl(T) * "_sqrtRi" * num_lbl(s)
const outroot = get(ENV, "OUT_ROOT", "outputs")
moments_file(tag) = joinpath(HERE, outroot, tag, "TidalBL3D_" * tag * "_moments.jld2")

const animdir = get(ENV, "ANIM_DIR", joinpath(HERE, outroot, "tke_animations"))
# The preview PNGs go to figures/ rather than next to the movies: everything
# under outputs/ is ignored by git except animation_*.mp4.
const figdir  = get(ENV, "FIG_DIR", joinpath(HERE, "figures"))
const logdir  = joinpath(HERE, "logs")
mkpath(animdir); mkpath(figdir); mkpath(logdir)

default(fontfamily = "sans-serif", framestyle = :box, grid = true, gridalpha = 0.15,
        tickfontsize = 8, guidefontsize = 10, legendfontsize = 9, titlefontsize = 11)

# The same palette as swirlesrun4.jl, so a case keeps its colour from the
# scatter plots to the movies.
const CLR = Dict(0.0 => "#8C8C8C", 1.0 => "#1B4E8F", 2.0 => "#2E8B57", 10.0 => "#B4502C")
clr(s) = get(CLR, s, "#000000")

# Superscript digits, so the axis can be labelled "×10⁻⁵ m² s⁻²" rather than
# carrying five leading zeros on every tick.
const SUPS = Dict('-'=>'⁻', '0'=>'⁰', '1'=>'¹', '2'=>'²', '3'=>'³', '4'=>'⁴',
                  '5'=>'⁵', '6'=>'⁶', '7'=>'⁷', '8'=>'⁸', '9'=>'⁹')
sup(n::Integer) = String([SUPS[c] for c in string(n)])

# TKE is a sum of variances and so cannot be negative in principle, but
# ⟨uu⟩ − U² is a difference of two nearly equal numbers wherever the flow has
# relaminarized, and it does come back slightly negative there. On a log axis
# that is undefined, so those cells are set to NaN and Plots draws a gap, which
# says there is no measurable turbulence rather than pinning a line to the floor.
poslog(v) = XSCALE == "log" ? [x > 0 ? x : NaN for x in v] : v

# Centred boxcar of half-width `nh` samples, narrowing at the edges. Copied from
# MixedLayerDiffusivity.jl so that both scripts smooth identically.
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

# ---------------- Load one case as TKE(z, t) ----------------
function load_tke(T, s)
    tag = tag_of(T, s)
    fname = moments_file(tag)
    isfile(fname) || (@warn "Missing $fname — $tag will not appear in the T = $T movie"; return nothing)

    ts = Dict(v => FieldTimeSeries(fname, v) for v in ("U", "V", "uu", "vv", "ww"))
    times = ts["U"].times
    nt = length(times)
    nt < 4 && (@warn "$tag has only $nt samples — skipping"; return nothing)

    zc = collect(znodes(ts["U"]))                  # Centers: U, V, uu, vv, ww
    grab(v) = (a = zeros(length(interior(ts[v][1])), nt);
               for n in 1:nt; a[:, n] .= vec(interior(ts[v][n])); end; a)
    U, V = grab("U"), grab("V")
    uu, vv, ww = grab("uu"), grab("vv"), grab("ww")

    # (1) decompose each sample; ⟨w⟩ is zero, so ww needs no mean subtracted
    TKE_raw = 0.5 .* ((uu .- U .^ 2) .+ (vv .- V .^ 2) .+ ww)

    # (2) and only then smooth in time
    dt = times[2] - times[1]
    nh = max(0, round(Int, smooth_window / (2dt)))
    TKE = boxcar(TKE_raw, nh)

    @printf("  %-18s %5d samples, %.2f periods, Nz = %3d, boxcar ±%d samples\n",
            tag, nt, times[end] / T_tide, length(zc), nh)
    return (; tag, T, s, times, z = zc, TKE)
end

# Nearest sample in `times` to t. The cases share a writing interval and a start
# time, so this is usually the identity; it exists so that a case which was
# restarted or stopped early still lines up with the others.
function nearest_index(times, t)
    i = searchsortedfirst(times, t)
    i <= 1 && return 1
    i > length(times) && return length(times)
    return (t - times[i-1]) <= (times[i] - t) ? i - 1 : i
end

# ---------------- One movie ----------------
function animate_T(T, cases, xmax_shared)
    isempty(cases) && (@warn "No cases loaded for T = $T — no movie written"; return nothing)

    zmax = parse(Float64, get(ENV, "ZMAX", string(T + 10)))

    # A common frame axis, taken from the shortest case past SKIP, so no frame
    # asks a case for a time it does not have.
    ref = argmin(c -> c.times[end], cases)
    t_end = minimum(c.times[end] for c in cases)
    frame_times = [t for t in ref.times if SKIP * T_tide <= t <= t_end][1:STRIDE:end]
    isempty(frame_times) && (@warn "SKIP_PERIODS = $SKIP leaves no samples for T = $T"; return nothing)

    # The index into each case's own time axis, computed once rather than per
    # frame.
    idx = Dict(c.tag => [nearest_index(c.times, t) for t in frame_times] for c in cases)

    # The x-axis is fixed across every frame and every case: a curve that halves
    # must look half as long, which it cannot on an axis that rescales with it.
    #
    # It is logarithmic by default. The near-wall peak is one to two orders above
    # anything at the interface and lies inside the plotted column, so on a
    # linear axis it sets the scale and presses the interface — the part the
    # movie is about — into the left edge. TKE also decays across the pycnocline
    # by decades, which is what a log axis shows. XSCALE=linear switches back,
    # and XMAX is then needed to crop the near-wall peak out.
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

    # On a linear axis TKE of order 1e-4 prints as five leading zeros on every
    # tick, so the leading decade is moved into the axis label. A log axis needs
    # no such trick.
    xfac = (XSCALE == "log" || !(xmax > 0)) ? 1.0 : 10.0^floor(Int, log10(xmax))
    xlab = xfac == 1.0 ? "TKE  (m² s⁻²)" :
           "TKE  (×10" * sup(floor(Int, log10(xmax))) * " m² s⁻²)"

    # The floor is set from the top of the axis rather than from the data. Above
    # the pycnocline TKE falls to whatever the closure leaves behind, often 1e-12
    # or less, so an axis reaching the smallest positive sample would spend most
    # of its length on dead water. It is placed DECADES below xmax and snapped to
    # a decade boundary so the ticks are round powers of ten.
    xmin = if XSCALE != "log"
        0.0
    elseif !isempty(XMIN_ENV)
        parse(Float64, XMIN_ENV)
    else
        10.0^(floor(Int, log10(xmax)) - DECADES + 1)
    end
    xlims = XSCALE == "log" ? (xmin, xmax) : (0.0, xmax / xfac)

    # One labelled tick per decade. Left to itself GR labels every second decade
    # on a short log axis, which leaves the reader counting gridlines.
    xticks = XSCALE == "log" ?
             10.0 .^ (ceil(Int, log10(xmin)):floor(Int, log10(xmax))) : :auto

    @printf("T = %.0f: %d frames (%.2f–%.2f periods, stride %d), z ≤ %.1f m, TKE ∈ [%.1e, %.3e] m²/s² (%s)\n",
            T, length(frame_times), frame_times[1] / T_tide, frame_times[end] / T_tide,
            STRIDE, zmax, xlims[1] * (XSCALE == "log" ? 1.0 : xfac), xmax, XSCALE)

    # The phase strip, drawn once and reused: the whole animated window of
    # U∞ = U₀ sin(ωt), with a dot marking the current frame.
    t_line = range(frame_times[1], frame_times[end], length = 600)
    U_line = U₀ .* sin.(ω .* t_line)

    # One frame, as a function, so that PREVIEW=1 renders exactly the frame the
    # movie would contain.
    function draw_frame(i)
        t = frame_times[i]

        p1 = plot(xlabel = xlab, ylabel = "z  (m)",
                  xlims = xlims, ylims = (0, zmax), xticks = xticks,
                  xscale = XSCALE == "log" ? :log10 : :identity,
                  legend = :topright, legendtitle = "N/ω",
                  title = @sprintf("TKE profiles, T = %g m   |   ωt/2π = %.2f   |   U∞/U₀ = %+.2f",
                                   T, t / T_tide, sin(ω * t)))

        # The pycnocline. Below z = T the background is unstratified in every
        # case, so a difference between the curves down there is not local
        # stratification but the interface above holding the layer back. The line
        # is labelled by an annotation rather than a legend entry, since the
        # legend is the N/ω key and a dashed line there would read as a fifth
        # case.
        hline!(p1, [T]; color = :black, ls = :dash, lw = 1.2, label = "")
        # Just inside the left edge, offset geometrically rather than linearly,
        # since on a log axis a linear offset lands most of the way across.
        x_ann = XSCALE == "log" ? xlims[1] * (xlims[2] / xlims[1])^0.02 :
                                  xlims[1] + 0.02 * (xlims[2] - xlims[1])
        annotate!(p1, x_ann, T + 0.03 * zmax, text("z = T", 8, :left, :bottom, :black))

        for c in cases
            plot!(p1, poslog(c.TKE[:, idx[c.tag][i]] ./ xfac), c.z;
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

    # PREVIEW=1 draws the middle frame to a PNG and stops. Rendering a full movie
    # takes minutes per T, so check the axes, the legend and the z range on one
    # frame first, especially if XMAX or ZMAX has been set by hand.
    if PREVIEW
        out = joinpath(figdir, "preview_TKE_T" * num_lbl(T) * "_" * XSCALE * ".png")
        savefig(draw_frame(cld(length(frame_times), 2)), out)
        @info "PREVIEW=1 — wrote $(relpath(out, HERE)) and skipped the movie"
        return (; T, out, nframes = 1, xmax, zmax, tags = [c.tag for c in cases])
    end

    anim = @animate for i in eachindex(frame_times)
        i % 100 == 0 && @info @sprintf("T = %.0f: frame %d / %d (%.2f periods)",
                                       T, i, length(frame_times), frame_times[i] / T_tide)
        draw_frame(i)
    end

    # The scale is part of the filename, so a log run does not overwrite the
    # linear movie of the same T; both are worth keeping side by side.
    out = joinpath(animdir, "animation_TKE_T" * num_lbl(T) * "_" * XSCALE * ".mp4")
    mp4(anim, out, fps = FPS)
    @info "wrote $(relpath(out, HERE))"
    return (; T, out, nframes = length(frame_times), xmax, zmax,
            tags = [c.tag for c in cases])
end

# ---------------- Drive ----------------
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

# SHARE_XLIM makes the two movies directly comparable, at the cost of whichever
# T has the weaker turbulence using only part of its axis.
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
        @printf(out, "==== %s  swirlesrun5.jl  XSCALE=%s SKIP_PERIODS=%g STRIDE=%d SMOOTH=%s FPS=%d\n",
                Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), XSCALE, SKIP, STRIDE, SMOOTH, FPS)
        println(out, "="^88)
        for r in results
            @printf(out, "T = %-4g  %4d frames  z ≤ %5.1f m  TKE ≤ %.3e  cases: %s\n    %s\n",
                    r.T, r.nframes, r.zmax, r.xmax, join(r.tags, ", "), relpath(r.out, HERE))
        end
        isempty(results) && println(out, "no movies written — no moments files found")
        println(out, "="^88)
    end
end
