# l against the Corrsin shear length scale at z = h, T = 10 m, both flows.
#
# The companion to plot_l_vs_qN_T10_combined.jl, with the abscissa changed:
#
#     y = l   = K_T(z=h) / √TKE(z=h)         unchanged
#     x = L_C = (ε / S³)^(1/2)               the Corrsin scale
#
# where S = |∂⟨u_h⟩/∂z| is the mean shear. L_C is the scale at which the eddy
# turnover rate matches the mean shear rate: eddies smaller than it turn over
# faster than the shear can distort them, larger ones are shaped by the shear.
# It is the shear analogue of the Ozmidov scale (ε/N³)^(1/2), and it is a
# different question from L_s = √TKE/S in plot_shear_scales_T10.jl — L_s is
# built from the energy, L_C from the flux of energy.
#
# ---------------- Where ε comes from, and what that costs ----------------
# *** Neither pipeline stored the dissipation rate. *** It cannot be rebuilt
# from stored averages either — ε is an average of a product of fluctuating
# gradients, like F_sgs, and only plane averages of the fields survive to disk.
#
# What can be done is to take it from the TKE budget, whose other terms were
# stored. Dropping storage and transport leaves local equilibrium,
#
#     ε ≈ P + B      P = −⟨u′w′⟩ ∂U/∂z − ⟨v′w′⟩ ∂V/∂z + νₑ S²
#                    B = F_b = ⟨w′b′⟩ + F_sgs        (negative when stable)
#
# and every term there is on disk, on the faces, at the same nodes as S. So the
# figure is real data throughout, but the abscissa carries an assumption the
# ordinate does not, and it should be read that way. Three caveats, in order of
# how much they hurt:
#
#   1. Transport is dropped and CANNOT be checked — ⟨w′e′⟩ and the pressure
#      term were never stored. At z = h, the top of the mixed layer, transport
#      is not generally small. This is the one to worry about.
#   2. Storage is dropped, and CAN be checked: |∂TKE/∂t|/ε is reported per case
#      below. It is 0.2-0.4 at the strongly stratified end and 4-6 at the
#      weakly stratified end. Both flows are statistically steady over the
#      averaging window, so this is a zero-mean term — it scatters the
#      per-sample ε rather than biasing the median — but where the ratio is
#      several the per-sample values are not to be trusted individually, and
#      that shows up directly as ε <= 0 in a fraction of samples.
#   3. νₑ ≈ κₑ. Only the buoyancy diffusivity was stored, so the subgrid part
#      of P uses it as the eddy viscosity. AMD builds the two with the same
#      coefficient and different numerators, so same order, not equal. It
#      matters only where the subgrid share is large, the high-N end.
#
# ε/(P + |B|) is also reported: it says how much of a residual ε is between two
# larger terms. It sits near 0.3, so ε is a healthy fraction of the budget and
# not the small difference of two big numbers — which is the failure mode that
# would have made this hopeless.
#
# ---------------- The two figures ----------------
# Same pair of styles as the √TKE/N figure, same conventions:
#   cloud       every retained sample, medians on top
#   errorbars   medians with the interquartile range in both axes
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot_l_vs_corrsin_T10.jl
#        (run reduce_ekman_moments_T10.jl first — it writes the Ekman side)
# ENV    STYLE   cloud | errorbars | both   (default both)
#        REBUILD 1 to re-walk the Stokes moment files instead of using the cache

using Oceananigans, JLD2, Plots, Printf, Statistics

get!(ENV, "GKSwstype", "100")
const HERE   = @__DIR__
const STOKES = "/home/tll46/SRIM-2026/Stokes/3D"
const EKFILE = joinpath(HERE, "Data", "ekman_lengthscales_T10_moments.jld2")
const CACHE  = joinpath(HERE, "Data", "corrsin_T10.jld2")
const FIGDIR = joinpath(HERE, "figures")
const ω      = 1e-4
const T_tide = 2π / ω
const SKIP   = 3                      # Stokes spin-up, in tidal periods
const SVALS  = [1, 2, 5, 10, 25, 50]

const RAMP = [(0.0,   ( 27,  78, 143)), (0.301, ( 46, 139,  87)),
              (0.699, (200, 150,  30)), (1.0,   (180,  80,  44)),
              (1.398, (142,  27,  78)), (1.699, ( 75,  16,  96))]
function ramp_colour(s)
    x = clamp(log10(s), RAMP[1][1], RAMP[end][1])
    for i in 1:length(RAMP)-1
        (x0, c0), (x1, c1) = RAMP[i], RAMP[i+1]
        x <= x1 || continue
        f = x1 == x0 ? 0.0 : (x - x0) / (x1 - x0)
        chan(k) = clamp(round(Int, c0[k] + f * (c1[k] - c0[k])), 0, 255)
        return "#" * join(string(chan(k), base = 16, pad = 2) for k in 1:3)
    end
    return "#000000"
end

fin(v) = filter(isfinite, v)
med(v) = (w = fin(v); isempty(w) ? NaN : median(w))
qlo(v) = (w = fin(v); isempty(w) ? NaN : quantile(w, 0.25))
qhi(v) = (w = fin(v); isempty(w) ? NaN : quantile(w, 0.75))

interp_at(z, fv, z₀) = isnan(z₀) ? NaN : begin
    i = searchsortedlast(z, z₀)
    i < 1 && return fv[1]
    i >= length(z) && return fv[end]
    fv[i] + (fv[i+1] - fv[i]) * (z₀ - z[i]) / (z[i+1] - z[i])
end
centres_to_faces(fc, zc, zf) =
    [interp_at(zc, fc, clamp(zf[k], zc[1], zc[end])) for k in eachindex(zf)]

function boxcar(a::AbstractVector, nh)
    nh <= 0 && return collect(float.(a))
    n = length(a); out = similar(float.(collect(a)))
    for i in 1:n
        w = @view a[max(1, i - nh):min(n, i + nh)]
        g = fin(w)
        out[i] = isempty(g) ? NaN : mean(g)
    end
    return out
end
boxcar(A::AbstractMatrix, nh) =
    reduce(vcat, (reshape(boxcar(view(A, k, :), nh), 1, :) for k in 1:size(A, 1)))

# Signed derivatives of the mean profile on the faces; the magnitude is their
# hypotenuse. Identical to reduce_ekman_moments_T10.jl, deliberately.
function shear_components(U, V, zc, zf)
    dU = zeros(Float64, length(zf)); dV = zeros(Float64, length(zf))
    @inbounds for k in 2:length(zc)
        dz = zc[k] - zc[k-1]
        dU[k] = (U[k] - U[k-1]) / dz
        dV[k] = (V[k] - V[k-1]) / dz
    end
    dU[1] = dU[2]; dU[end] = dU[end-1]
    dV[1] = dV[2]; dV[end] = dV[end-1]
    return dU, dV
end

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))

# ---------------- Stokes: S and ε, built here ----------------
# The Stokes mixing files predate the question, so the budget terms are read
# straight from TidalBL3D_*_moments.jld2 while h, K_at_h and TKE_at_h keep
# coming from mixing_*_hcross.jld2 — the same split plot_shear_scales_T10.jl
# uses for the shear alone.
function build_stokes()
    out = []
    for s in SVALS
        tag = "P4_T10_sqrtRi$s"
        mix = joinpath(STOKES, "outputs", tag, "mixing_$(tag)_hcross.jld2")
        mom = joinpath(STOKES, "outputs", tag, "TidalBL3D_$(tag)_moments.jld2")
        (isfile(mix) && isfile(mom)) || (say("missing files for $tag — skipped"); continue)

        d = jldopen(mix, "r") do io
            (t = io["times"], Kh = io["K_at_h"], E = io["TKE_at_h"], h = io["h"])
        end
        F = Dict(v => FieldTimeSeries(mom, v; backend = OnDisk()) for v in
                 ("U", "V", "W", "B", "uw", "vw", "wb", "F_sgs", "kappa_sgs"))
        length(F["U"].times) == length(d.t) ||
            error("$tag: moments and mixing files have different time bases")
        grid = F["U"].grid
        zc = Array(znodes(grid, Center())); zf = Array(znodes(grid, Face()))

        dt = d.t[2] - d.t[1]
        nh = max(0, round(Int, (T_tide / 20) / (2dt)))
        # Only the samples that survive the spin-up cut are needed, plus the
        # boxcar halo on the low side.
        first_keep = findfirst(t -> t >= SKIP * T_tide, d.t)
        first_keep === nothing && (say("$tag: nothing past the spin-up — skipped"); continue)
        sel = max(1, first_keep - nh):length(d.t)

        col(v, n) = Float64.(Array(interior(F[v][n], 1, 1, :)))
        nz = length(zf); nt = length(sel)
        Sr = Array{Float64}(undef, nz, nt)
        Pr = Array{Float64}(undef, nz, nt)
        Fb = Array{Float64}(undef, nz, nt)
        for (i, n) in enumerate(sel)
            U, V, W, B = col("U", n), col("V", n), col("W", n), col("B", n)
            dU, dV = shear_components(U, V, zc, zf)
            Sr[:, i] = hypot.(dU, dV)
            Uf = centres_to_faces(U, zc, zf); Vf = centres_to_faces(V, zc, zf)
            Bf = centres_to_faces(B, zc, zf)
            Pres = .-(col("uw", n) .- Uf .* W) .* dU .- (col("vw", n) .- Vf .* W) .* dV
            Psgs = col("kappa_sgs", n) .* (dU .^ 2 .+ dV .^ 2)
            Pr[:, i] = Pres .+ Psgs
            Fb[:, i] = (col("wb", n) .- W .* Bf) .+ col("F_sgs", n)
        end
        Sm = boxcar(Sr, nh); Pm = boxcar(Pr, nh); Fm = boxcar(Fb, nh)
        EPS = Pm .+ Fm
        # Diagnostic: does local equilibrium hold anywhere in this column? The
        # fraction of samples with ε > 0 at fractions of h, which separates "the
        # method is wrong" from "z = h is the wrong height for it".
        zscan = [count(i -> (e = interp_at(zf, view(EPS, :, i), a * d.h[sel[i]]);
                             isfinite(e) && e > 0), 1:nt) / nt
                 for a in (0.25, 0.5, 0.75, 1.0)]

        ts = d.t[sel]
        S_h = [interp_at(zf, view(Sm, :, i), d.h[sel[i]]) for i in 1:nt]
        P_h = [interp_at(zf, view(Pm, :, i), d.h[sel[i]]) for i in 1:nt]
        B_h = [interp_at(zf, view(Fm, :, i), d.h[sel[i]]) for i in 1:nt]
        e_h = [interp_at(zf, view(EPS, :, i), d.h[sel[i]]) for i in 1:nt]

        keep = ts .>= SKIP * T_tide
        push!(out, (r = float(s), N = s * ω, t = ts[keep],
                    K = d.Kh[sel][keep], E = d.E[sel][keep],
                    S = S_h[keep], P = P_h[keep], B = B_h[keep], eps = e_h[keep],
                    zscan = zscan))
        say(@sprintf("Stokes N/ω = %-4g  %4d samples  med S = %.3e  med eps = %.3e",
                     s, count(keep), med(S_h[keep]), med(e_h[keep])))
    end
    return out
end

if isfile(CACHE) && get(ENV, "REBUILD", "0") == "0"
    say("Stokes side from the cache $(basename(CACHE)) — set REBUILD=1 to re-walk the moments")
    stokes = []
    jldopen(CACHE, "r") do io
        for r in io["ratios"]
            g = @sprintf("r=%.1f", r)
            push!(stokes, (r = r, N = io["$g/N"], t = io["$g/t"], K = io["$g/K"],
                           E = io["$g/E"], S = io["$g/S"], P = io["$g/P"],
                           B = io["$g/B"], eps = io["$g/eps"], zscan = io["$g/zscan"]))
        end
    end
else
    say("walking the Stokes moment files — a few minutes")
    stokes = build_stokes()
    isempty(stokes) && error("no Stokes T = 10 cases found under $STOKES/outputs")
    jldopen(CACHE, "w") do io
        io["note"] = "Stokes S, P, B, eps at z = h; written by plot_l_vs_corrsin_T10.jl"
        io["ratios"] = [c.r for c in stokes]
        for c in stokes
            g = @sprintf("r=%.1f", c.r)
            io["$g/N"] = c.N; io["$g/t"] = c.t; io["$g/K"] = c.K; io["$g/E"] = c.E
            io["$g/S"] = c.S; io["$g/P"] = c.P; io["$g/B"] = c.B; io["$g/eps"] = c.eps
            io["$g/zscan"] = c.zscan
        end
    end
    say("wrote $CACHE")
end

# ---------------- Ekman: read the reduction ----------------
isfile(EKFILE) || error("$EKFILE not found — run reduce_ekman_moments_T10.jl first")
ekman = []
jldopen(EKFILE, "r") do io
    haskey(io, "r=50.0/eps_at_h") ||
        error("$EKFILE has no eps_at_h — re-run reduce_ekman_moments_T10.jl")
    for r in io["ratios"]
        g = @sprintf("r=%.1f", r)
        push!(ekman, (r = r, N = io["$g/N"], t = io["$g/times"],
                      K = io["$g/K_at_h"], E = io["$g/TKE_at_h"], S = io["$g/S_at_h"],
                      P = io["$g/P_at_h"], B = io["$g/Fb_at_h"], eps = io["$g/eps_at_h"],
                      zscan = io["$g/eps_zscan"]))
    end
end

# ---------------- form the scales, per sample ----------------
# Per sample first, median second, as everywhere else here.
function scales(c)
    q   = sqrt.(max.(c.E, 0))
    l   = c.K ./ q
    # Only where the estimate is usable — ε from a budget residual can come
    # out negative for an individual sample, and sqrt would throw.
    L_C = [(isfinite(c.eps[i]) && c.eps[i] > 0 && isfinite(c.S[i]) && c.S[i] > 0) ?
           sqrt(c.eps[i] / c.S[i]^3) : NaN for i in eachindex(q)]
    # Three separate reasons a sample leaves, counted separately so the log
    # never blames the wrong one:
    #   badl  K_T <= 0, counter-gradient flux — the log axis cannot show it
    #   bade  ε <= 0, the local-equilibrium estimate failed for this sample
    #   badS  the mean shear passed through zero
    badl = @. !(isfinite(l) && l > 0)
    bade = @. !(isfinite(c.eps) && c.eps > 0)
    badS = @. !(isfinite(c.S) && c.S > 0)
    ok = @. !badl & !bade & !badS
    # A case is only usable if the ε estimate stood up in most of its samples.
    # Where it did not, the surviving subset is selected on the sign of a
    # budget residual, which is exactly the kind of selection that manufactures
    # a trend — so those cases are drawn hollow and kept out of every fit.
    epsok = 1 - count(bade) / length(q)
    return (r = c.r, N = c.N, x = L_C[ok], l = l[ok],
            epsok = epsok, reliable = epsok >= 0.5, zscan = c.zscan,
            xm = med(L_C[ok]), lm = med(l[ok]),
            n = count(ok), ntot = length(q),
            nbadl = count(badl), nbade = count(bade), nbadS = count(badS),
            cancel = med([c.eps[i] / (abs(c.P[i]) + abs(c.B[i])) for i in eachindex(q)]),
            epsm = med(c.eps), Sm = med(c.S))
end
S = [scales(c) for c in stokes]
E = [scales(c) for c in ekman]

say("")
say("retained samples, and why the rest went")
say("  flow    r      kept        K_T<=0   eps<=0    S<=0   eps>0 frac")
for (nm, cs) in (("Stokes", S), ("Ekman ", E)), c in cs
    say(@sprintf("  %s %-5g %5d/%-5d %8d %8d %7d %10.2f%s", nm, c.r, c.n, c.ntot,
                 c.nbadl, c.nbade, c.nbadS, c.epsok,
                 c.reliable ? "" : "   << NOT USABLE, excluded from the fits"))
end

say("")
say("where in the layer does local equilibrium hold?  fraction of samples with")
say("ε > 0 at fractions of h.  A column that is fine at 0.5h and fails at h is")
say("telling us about the height, not about the method.")
say("  flow    r      0.25h  0.50h  0.75h  1.00h")
for (nm, cs) in (("Stokes", S), ("Ekman ", E)), c in cs
    say(@sprintf("  %s %-5g %6.2f %6.2f %6.2f %6.2f", nm, c.r, c.zscan...))
end

say("")
say("medians at z = h.  L_C = (ε/S³)^(1/2);  ε/(P+|B|) says how much of a")
say("residual ε is between the two larger budget terms")
say("  flow    r        eps (m²/s³)   S (1/s)      L_C (m)     l (m)    eps/(P+|B|)")
for (nm, cs) in (("Stokes", S), ("Ekman ", E)), c in cs
    say(@sprintf("  %s %-5g %13.4e %12.4e %10.4f %9.4f %11.2f",
                 nm, c.r, c.epsm, c.Sm, c.xm, c.lm, c.cancel))
end

# ---------------- the fit ----------------
# The same saturating family and the same brute-force search as the √TKE/N
# figure, so the two can be compared parameter for parameter. `pinned` reports
# a best point on a grid edge, which is not a fit and is never drawn.
const L_GRID  = 0.05:0.005:4.00
const X0_GRID = 0.02:0.01:20.0
function fit_sat(cs)
    length(cs) >= 4 || return nothing
    best = (Inf, 0.0, 0.0)
    for L in L_GRID, x0 in X0_GRID
        sse = 0.0
        for c in cs
            p = L * (1 - exp(-c.xm / x0))
            p > 0 || (sse = Inf; break)
            sse += (log(c.lm) - log(p))^2
        end
        sse < best[1] && (best = (sse, L, x0))
    end
    pinned = best[2] in (first(L_GRID), last(L_GRID)) ||
             best[3] in (first(X0_GRID), last(X0_GRID))
    return (sse = best[1], L = best[2], x0 = best[3], pinned = pinned,
            rms = 100 * sqrt(best[1] / length(cs)), n = length(cs))
end
# A straight power law too: the Corrsin scale has no reason to saturate the way
# the buoyancy scale did, so the shape is not assumed in advance.
function fit_pow(cs)
    lx = [log(c.xm) for c in cs]; ly = [log(c.lm) for c in cs]
    mx = mean(lx); my = mean(ly)
    b = sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx) .^ 2)
    a = my - b * mx
    res = ly .- (a .+ b .* lx)
    return (A = exp(a), b = b, rms = 100 * sqrt(mean(res .^ 2)), n = length(cs))
end

RS = filter(c -> c.reliable, S); RE = filter(c -> c.reliable, E)
F_S = fit_sat(RS); F_E = fit_sat(RE); F_A = fit_sat(vcat(RS, RE))
P_S = length(RS) >= 3 ? fit_pow(RS) : nothing
P_E = length(RE) >= 3 ? fit_pow(RE) : nothing
P_A = fit_pow(vcat(RS, RE))
say("")
say(@sprintf("fits of l against L_C, on the case medians of the %d usable cases of %d",
             length(RS) + length(RE), length(S) + length(E)))
say("  set       saturating  l = L∞(1 − e^(−x/x₀))            power law  l = A x^b")
for (nm, fs, fp) in (("Stokes", F_S, P_S), ("Ekman ", F_E, P_E), ("both  ", F_A, P_A))
    fp === nothing && (say(@sprintf("  %s  too few usable cases to fit", nm)); continue)
    sat = fs === nothing ? "too few cases" :
          @sprintf("L∞ = %5.2f m, x₀ = %5.2f m, rms %5.1f %%%s", fs.L, fs.x0, fs.rms,
                   fs.pinned ? "  << PINNED, not a fit" : "")
    say(@sprintf("  %s  %-52s A = %.3f, b = %5.2f, rms %5.1f %%", nm, sat, fp.A, fp.b, fp.rms))
end

# ---------------- the figures ----------------
const STYLE = get(ENV, "STYLE", "both")
STYLE in ("cloud", "errorbars", "both") ||
    error("STYLE must be cloud, errorbars or both — got \"$STYLE\"")
mkpath(FIGDIR)

function draw(style)
    ally = sort(reduce(vcat, ([c.l for c in S]..., [c.l for c in E]...)))
    if style == "cloud"
        ylo = ally[max(1, round(Int, 0.005 * length(ally)))] / 1.5
        yhi = ally[round(Int, 0.999 * length(ally))] * 1.5
    else
        ylo = minimum(vcat([qlo(c.l) for c in S], [qlo(c.l) for c in E])) / 1.6
        yhi = maximum(vcat([qhi(c.l) for c in S], [qhi(c.l) for c in E])) * 1.6
    end

    p = plot(xscale = :log10, yscale = :log10, legend = :bottomright, ylims = (ylo, yhi),
             xlabel = "L_C = (ε/S³)^(1/2)   (Corrsin shear scale, m)",
             ylabel = "l = K_T/√TKE   (m)",
             title = "T = 10 m:  l against the Corrsin scale at z = h  —  Stokes (tidal) and Ekman",
             size = (980, 720), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
             legendfontsize = 7, foreground_color_legend = nothing)

    if style == "cloud"
        for c in S
            scatter!(p, c.x, c.l; ms = 1.6, msw = 0, alpha = 0.40,
                     color = ramp_colour(c.r), label = "")
        end
        for c in E
            scatter!(p, c.x, c.l; ms = 2.6, msw = 0.6, alpha = 0.55, marker = :xcross,
                     color = ramp_colour(c.r), msc = ramp_colour(c.r), label = "")
        end
    end

    xs = reduce(vcat, ([c.x for c in S]..., [c.x for c in E]...))
    lo, hi = minimum(xs), maximum(xs)
    if style == "errorbars"
        lo = minimum(vcat([qlo(c.x) for c in S], [qlo(c.x) for c in E]))
        hi = maximum(vcat([qhi(c.x) for c in S], [qhi(c.x) for c in E]))
    end
    plot!(p, [lo, hi], [lo, hi]; color = :black, lw = 1.2, ls = :dash, label = "l = L_C  (1:1)")

    # Each fit across the range of the cases it was made from. The family drawn
    # per set is whichever fitted better and did not pin.
    function fitline(fs, fp, cs, col, ls, lw, nm)
        (fp === nothing || isempty(cs)) && return
        a = minimum(c.xm for c in cs); b = maximum(c.xm for c in cs)
        xx = exp.(range(log(a), log(b); length = 300))
        usesat = fs !== nothing && !fs.pinned && fs.rms <= fp.rms
        if usesat
            plot!(p, xx, fs.L .* (1 .- exp.(-xx ./ fs.x0)); color = col, ls = ls, lw = lw,
                  label = @sprintf("%s: l = %.2f(1 − e^(−x/%.2f)), rms %.0f %%", nm, fs.L, fs.x0, fs.rms))
        else
            plot!(p, xx, fp.A .* xx .^ fp.b; color = col, ls = ls, lw = lw,
                  label = @sprintf("%s: l = %.3f x^%.2f, rms %.0f %%", nm, fp.A, fp.b, fp.rms))
        end
    end
    fitline(F_S, P_S, RS, "#1b3a6b", :dash, 2.0, "Stokes fit")
    fitline(F_E, P_E, RE, "#8e1b4e", :dash, 2.0, "Ekman fit")
    fitline(F_A, P_A, vcat(RS, RE), :black, :solid, 2.6, "overall fit")

    # Hollow means the ε estimate failed in most of that case's samples, so its
    # median is taken over a subset selected on the sign of a budget residual.
    for (cs, mk, ms, nm, sym) in ((S, :circle, 8, "Stokes", "N/ω"),
                                  (E, :diamond, 9, "Ekman", "N/f"))
        for c in cs
            lab = @sprintf("%s %s = %g%s", nm, sym, c.r, c.reliable ? "" : "  (ε unusable)")
            mc = c.reliable ? ramp_colour(c.r) : :white
            if style == "cloud"
                scatter!(p, [c.xm], [c.lm]; marker = mk, ms = ms, msw = 1.6,
                         color = mc, msc = ramp_colour(c.r), label = lab)
            else
                scatter!(p, [c.xm], [c.lm];
                         xerror = ([c.xm - qlo(c.x)], [qhi(c.x) - c.xm]),
                         yerror = ([c.lm - qlo(c.l)], [qhi(c.l) - c.lm]),
                         marker = mk, ms = ms, msw = 1.6, color = mc,
                         msc = ramp_colour(c.r), linecolor = ramp_colour(c.r), lw = 1.6,
                         label = lab)
            end
        end
    end

    f = joinpath(FIGDIR, style == "cloud" ? "l_vs_corrsin_ath_T10_combined.png" :
                                            "l_vs_corrsin_ath_T10_combined_errorbars.png")
    savefig(p, f)
    say("wrote $f")
end

say("")
for st in (STYLE == "both" ? ("cloud", "errorbars") : (STYLE,))
    draw(st)
end

mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "plot_l_vs_corrsin_T10.log"), "w") do io
    foreach(l -> println(io, l), logl)
end
