ENV["GKSwstype"] = "100"        # headless GR — must precede `using Plots`

using Oceananigans, JLD2, Plots, Printf, Statistics

# Turbulent diffusivity of the tidal bottom boundary layer, from the second
# moments Moments.jl writes to *_moments.jld2.
#
# The question this answers: does K_T scale as √TKE · l, or as TKE / N? The two
# differ by the slope of log K_T against log TKE at the mixed-layer top —
# 1/2 for √TKE·l, 1 for TKE/N. That is panel (d).
#
#   julia --project=. MixedLayerDiffusivity.jl
# Subsets (same convention as Figure4_metres.jl / Figure5.jl):
#   T_VALUES="3 5" N_OVER_OMEGA="1 2 5" julia --project=. MixedLayerDiffusivity.jl
#
# Per case it writes
#   outputs/<tag>/mixing_<tag>.jld2   TKE(z,t), K_T(z,t), K_T_bulk, K_T_pe,
#                                     delta_eff, K_sgs/K_T, h, PE, u*
#   figures/K_T_<tag>.png             4 panels (see below)
# and prints a VERIFICATION block. READ THE PANELS IN THIS ORDER — an earlier
# failure invalidates everything after it:
#   1. ⟨w⟩_xy ≈ 1e-18.        If not, stop: every flux below is wrong.
#   2. K_T_bulk ≈ K_T_pe (c). They share no code path — one uses wb and F_sgs,
#      the other only B. Disagreement means the SGS flux is wrong, usually
#      under-counted, which shows as the flux route reading low.
#   3. K_sgs/K_T ≪ 1.         Above ~0.5 the number characterises AMD, not the flow.
#   4. delta_eff steady, ideally tracking h.
#   5. only THEN the panel (d) slope.
#
# ---------------------------------------------------------------------------
# ORDER OF OPERATIONS — this is where the physics is won or lost
# ---------------------------------------------------------------------------
#   (1) DECOMPOSE PER SAMPLE. The plane average IS the Reynolds average here
#       (Periodic×Periodic, horizontally uniform forcing), so u′ = u − ⟨u⟩_xy
#       exactly, and the tidal flow — being horizontally uniform — is removed
#       instantaneously with it. Every mean is subtracted at its OWN sample time.
#   (2) SMOOTH IN TIME AFTERWARDS, for noise reduction only. Doing it the other
#       way round gives  time_avg(⟨uu⟩) − time_avg(U)² = time_avg(TKE) + Var_t(U):
#       the variance of the tidal mean flow over the window leaks in as "TKE",
#       ~(U₀ωΔt)²/12 ≈ 4e-6 m²s⁻² at Δt = 1800 s, the same order as u*² itself.
#   (3) FORM RATIOS LAST. K_T = −F_b/(dB/dz) is built from the already-smoothed
#       F_b and dBdz, never from smoothed ratios: a ratio of two noisy profiles is
#       badly behaved wherever the denominator is small, which is most of the
#       mixed layer.
# Moments.jl guarantees step (1) by writing raw instantaneous moments on a plain
# TimeInterval; see its header for why AveragedTimeInterval would break it.

const ω  = 1e-4
const ν  = 1.0e-6
const Pr = 10
const κ_mol = ν / Pr
const U₀ = 0.04
const T_tide = 2π / ω                  # 62832 s = 17.45 h — NOT the 12.4 h of a
                                       # real M2 tide; ω was set to 1e-4 to match
                                       # the colleague's Coriolis parameter.
const δs = sqrt(2ν / ω)
const Lz = 50.0

parse_list(key, default) = parse.(Float64, split(get(ENV, key, default)))
const T_values = parse_list("T_VALUES", "2 3 5 8")
const n_over_ω = parse_list("N_OVER_OMEGA", get(ENV, "SQRT_RI", "0 1 2 5 10"))

# Tag builder matching case_params.jl, identical to Figure4_metres.jl / Figure5.jl.
num_lbl(x) = isinteger(x) ? string(Int(x)) : replace(string(x), "." => "p")
tag_of(T, s) = "P4_T" * num_lbl(T) * "_sqrtRi" * num_lbl(s)
const outroot = get(ENV, "OUT_ROOT", "outputs")
moments_file(tag) = joinpath(@__DIR__, outroot, tag, "TidalBL3D_" * tag * "_moments.jld2")

# ---------------------------------------------------------------------------
# GRAD_FLOOR — the dominant term in the error bar on K_T
# ---------------------------------------------------------------------------
# K_T = −F_b/(dB/dz) is 0/0 inside the mixed layer, where dB/dz → 0 by definition.
# Cells with |dB/dz| < GRAD_FLOOR · N_ref² are masked out.
#
# THIS THRESHOLD DOMINATES THE ERROR BAR ON EVERY K_T REPORTED HERE. It decides
# how far down into the weakly stratified layer the ratio is trusted, and K_T
# grows without bound as the denominator is allowed to shrink. Sweep it and quote
# the spread rather than a single number:
#     for g in 0.02 0.05 0.10; do
#         GRAD_FLOOR=$g RESULT_SUFFIX=_g$g julia --project=. MixedLayerDiffusivity.jl
#     done
# RESULT_SUFFIX is what keeps such a sweep from writing every pass to the same
# mixing_<tag>.jld2 and leaving only the last one.
const GRAD_FLOOR = parse(Float64, get(ENV, "GRAD_FLOOR", "0.05"))
const RESULT_SUFFIX = get(ENV, "RESULT_SUFFIX", "")

# Minimum |r| for the panel (d) slope to be quotable as an exponent. A
# least-squares slope through a round cloud is still a number, and with n in the
# thousands its formal error bar is small — which is exactly how a null result
# gets written up as a finding. 0.5 (r² = 0.25) is a low bar deliberately: below
# it the power law is not describing the data at all.
const R_MIN = parse(Float64, get(ENV, "R_MIN", "0.5"))

# ---------------------------------------------------------------------------
# Time reduction. Applied only AFTER the per-sample decomposition (step 2 above).
# ---------------------------------------------------------------------------
#   tide20 (default)  boxcar of T_tide/20 ≈ 52 min — kills plane-average noise
#                     while keeping the intra-cycle turbulence burst
#   tide              boxcar of the full T_tide — the slowly evolving envelope,
#                     which is what mixed-layer growth should be read from
#   phase             bin by mod(ωt, 2π), ensembled over periods — phase structure
const SMOOTH = get(ENV, "SMOOTH", "tide20")
const N_PHASE = parse(Int, get(ENV, "N_PHASE", "40"))
# Periods to discard before ensembling / fitting: the restart from a spin-up
# snapshot (and, for these runs, the change from a no-slip to a drag bottom) is a
# transient that is not the physics being measured.
const SKIP_PERIODS = parse(Float64, get(ENV, "SKIP_PERIODS", "1.0"))

const smooth_window = SMOOTH == "tide20" ? T_tide / 20 :
                      SMOOTH == "tide"   ? T_tide      :
                      SMOOTH == "phase"  ? 0.0         :
                      error("SMOOTH must be tide20, tide or phase — got \"$SMOOTH\"")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Lowest height at which fv crosses `level` from below, linearly interpolated.
# COPIED VERBATIM from Figure5.jl so the mixed-layer height h is defined
# identically across the project — do not "improve" one copy alone.
function first_crossing(z, fv, level; zmin = -Inf)
    for i in 1:length(fv)-1
        z[i] < zmin && continue
        if fv[i] < level <= fv[i+1]
            return z[i] + (level - fv[i]) * (z[i+1] - z[i]) / (fv[i+1] - fv[i])
        end
    end
    return NaN
end

interp_at(z, fv, z₀) = isnan(z₀) ? NaN : begin
    i = searchsortedlast(z, z₀)
    i < 1 && return fv[1]
    i >= length(z) && return fv[end]
    fv[i] + (fv[i+1] - fv[i]) * (z₀ - z[i]) / (z[i+1] - z[i])
end

# Trapezoidal integral of f over [0, zmax] on the ACTUAL node spacing.
# The vertical grid is stretched — Δz runs from 0.0086 m at the wall to 0.34 m
# above z ≈ 10 m, a factor of 40 — so sum(f)*dz is not an approximation of this
# integral, it is a different number. The last cell is cut at zmax by linear
# interpolation, and when the nodes start above 0 (Centers do, at z = 0.0043 m)
# the sliver below is closed with `f0`.
function trapz_to(z, f, zmax; f0 = nothing)
    s = 0.0
    zprev, fprev = z[1], f[1]
    if z[1] > 0
        v0 = f0 === nothing ? f[1] : f0
        s += 0.5 * (v0 + f[1]) * z[1]
    end
    for i in 2:length(z)
        z[i] > zmax && break
        isfinite(f[i]) && isfinite(fprev) && (s += 0.5 * (fprev + f[i]) * (z[i] - zprev))
        zprev, fprev = z[i], f[i]
    end
    if zprev < zmax                      # partial final cell
        fend = interp_at(z, f, zmax)
        isfinite(fend) && isfinite(fprev) && (s += 0.5 * (fprev + fend) * (zmax - zprev))
    end
    return s
end

# Centred boxcar of half-width `nh` samples, shrinking at the edges. NOTE a boxcar
# and a centred difference are both linear and shift-invariant, so smoothing PE
# then differencing gives the same K_T_pe as differencing then smoothing.
function boxcar(a::AbstractVector, nh)
    nh <= 0 && return collect(float.(a))
    n = length(a)
    out = similar(float.(collect(a)))
    for i in 1:n
        lo, hi = max(1, i - nh), min(n, i + nh)
        w = @view a[lo:hi]
        good = filter(isfinite, w)
        out[i] = isempty(good) ? NaN : mean(good)
    end
    return out
end
boxcar(A::AbstractMatrix, nh) =
    reduce(vcat, (reshape(boxcar(view(A, k, :), nh), 1, :) for k in 1:size(A, 1)))

# Phase bins of mod(ωt, 2π), ensembled over whole periods past SKIP_PERIODS.
function phase_bins(times)
    φ = mod.(ω .* times, 2π)
    edges = range(0, 2π, length = N_PHASE + 1)
    keep = times .>= SKIP_PERIODS * T_tide
    idx = [findall(i -> keep[i] && edges[j] <= φ[i] < edges[j+1], eachindex(times))
           for j in 1:N_PHASE]
    return collect(edges[1:end-1] .+ step(edges) / 2), idx
end

phase_reduce(a::AbstractVector, idx) =
    [isempty(I) ? NaN : (g = filter(isfinite, a[I]); isempty(g) ? NaN : mean(g)) for I in idx]
phase_reduce(A::AbstractMatrix, idx) =
    reduce(vcat, (reshape(phase_reduce(view(A, k, :), idx), 1, :) for k in 1:size(A, 1)))

# Least-squares slope of log10 y against log10 x, over the finite positive pairs.
function loglog_slope(x, y)
    m = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
    n = count(m)
    n < 8 && return (NaN, NaN, n)
    lx, ly = log10.(x[m]), log10.(y[m])
    sx, sy = mean(lx), mean(ly)
    slope = sum((lx .- sx) .* (ly .- sy)) / sum((lx .- sx) .^ 2)
    r = sum((lx .- sx) .* (ly .- sy)) / sqrt(sum((lx .- sx) .^ 2) * sum((ly .- sy) .^ 2))
    return (slope, r, n)
end

default(fontfamily = "sans-serif", framestyle = :box, grid = true, gridalpha = 0.15,
        tickfontsize = 8, guidefontsize = 9, legendfontsize = 7, titlefontsize = 10)

const figdir = get(ENV, "FIG_DIR", joinpath(@__DIR__, "figures"))
mkpath(figdir)

# ---------------------------------------------------------------------------
# Per-case analysis
# ---------------------------------------------------------------------------
function analyse(T, s)
    tag = tag_of(T, s)
    fname = moments_file(tag)
    isfile(fname) || (@warn "Missing $fname — skipping $tag"; return nothing)

    # N²_ref matches case_params.jl: at N/ω = 0 the buoyancy force is off but b is
    # still advected as a passive scalar carrying the ω²-scaled softplus
    # background, so ω² is the reference there rather than a divide-by-zero.
    Ri = s^2
    N²_ref = Ri > 0 ? Ri * ω^2 : ω^2

    # zref must sit well above the interface but well below the sponge at z = Lz.
    zref = min(T + 15, 40.0)

    # --- load ---------------------------------------------------------------
    ts = Dict(v => FieldTimeSeries(fname, v) for v in
              ("U", "V", "W", "B", "dBdz", "uu", "vv", "ww", "uw", "vw", "wb",
               "kappa_sgs", "F_sgs"))
    times = ts["B"].times
    nt = length(times)
    nt < 4 && (@warn "$tag has only $nt samples — skipping"; return nothing)

    zc = collect(znodes(ts["B"]))          # Centers: U V B uu vv ww
    zf = collect(znodes(ts["wb"]))         # Faces:   W dBdz uw vw wb kappa_sgs F_sgs

    grab(v) = (a = zeros(length(interior(ts[v][1])), nt);
               for n in 1:nt; a[:, n] .= vec(interior(ts[v][n])); end; a)
    U, V, W, B    = grab("U"), grab("V"), grab("W"), grab("B")
    dBdz          = grab("dBdz")
    uu, vv, ww    = grab("uu"), grab("vv"), grab("ww")
    uw, vw, wb    = grab("uw"), grab("vw"), grab("wb")
    kap, F_sgs    = grab("kappa_sgs"), grab("F_sgs")

    # --- (1) INSTANTANEOUS DECOMPOSITION -----------------------------------
    # ⟨w⟩_xy ≡ 0 with a rigid lid, impermeable bottom and incompressibility, so
    # ⟨w′b′⟩ = ⟨wb⟩ and ⟨u′w′⟩ = ⟨uw⟩ with NO mean subtraction. Only the
    # variances need one, and each uses the mean at ITS OWN sample time.
    W_max = maximum(abs, W)
    uu′ = uu .- U .^ 2
    vv′ = vv .- V .^ 2
    ww′ = ww                                    # ⟨w⟩ = 0
    TKE_raw = 0.5 .* (uu′ .+ vv′ .+ ww′)        # Centers
    F_b_raw = wb .+ F_sgs                       # Faces, positive upward, already signed

    # PE = −(1/Lz)∫₀^zref z B dz, integrated to the SAME zref as the flux, so the
    # boundary term z F_b|_zref cancels between the two routes. The 1/Lz is an
    # arbitrary normalisation and cancels again in K_T_pe.
    PE_raw = [-trapz_to(zc, zc .* B[:, n], zref; f0 = 0.0) / Lz for n in 1:nt]

    # Δb to the same zref the integral uses — mismatching them is a silent bias.
    δb_raw = [interp_at(zc, B[:, n], zref) - interp_at(zc, B[:, n], 0.0) for n in 1:nt]

    # u*(t) from the drag law the bottom BC actually applies, τ = cᴰ|u|u, so
    # u*² = cᴰ (U₁² + V₁²) at the first cell centre. This uses the PLANE-AVERAGED
    # U₁ where the BC uses the pointwise one, so it is a diagnostic estimate, not
    # the exact wall stress. It also omits the molecular part ν·U₁/z₁, which on
    # this wall-resolving grid is the LARGER term — so read u* as a relative
    # measure of the tidal forcing over the cycle, not as the wall stress itself.
    #
    # cᴰ MUST MATCH THE RUN. The default mirrors case_params.jl: the log law at
    # the fixed reference height z_drag_ref, NOT at z₁ (see the note there).
    z₁ = zc[1]
    z_drag_ref = parse(Float64, get(ENV, "Z_DRAG_REF", "0.0667"))
    cᴰ = parse(Float64, get(ENV, "CD", string((0.41 / log(max(z_drag_ref, 2*0.0016) / 0.0016))^2)))
    ustar_raw = [sqrt(cᴰ * (U[1, n]^2 + V[1, n]^2)) for n in 1:nt]

    # --- (2) TIME REDUCTION -------------------------------------------------
    # Anonymous functions, not `red_m(A) = ...`: a named definition in each branch
    # of an if would be two method definitions of the same local name.
    local xs, red_m, red_v, xlab
    if SMOOTH == "phase"
        xs, idx = phase_bins(times)
        red_m = A -> phase_reduce(A, idx)
        red_v = a -> phase_reduce(a, idx)
        xlab = "ωt mod 2π (rad)"
    else
        dt = length(times) > 1 ? times[2] - times[1] : 1.0
        nh = max(0, round(Int, smooth_window / (2dt)))
        xs = ω .* times
        red_m = A -> boxcar(A, nh)
        red_v = a -> boxcar(a, nh)
        xlab = "ωt"
    end
    # dPE/dt is taken on the TIME series and only then reduced. In phase mode that
    # ordering is mandatory — a phase bin gathers samples from different periods,
    # so a difference taken across one is meaningless. In boxcar mode it is free:
    # a boxcar and a centred difference are both linear and shift-invariant, so
    # they commute.
    dPEdt = red_v(centred_diff(PE_raw, times))

    TKE   = red_m(TKE_raw)
    F_b   = red_m(F_b_raw)
    F_s   = red_m(F_sgs)
    G     = red_m(dBdz)
    Bs    = red_m(B)
    Us    = red_m(U)
    uws   = red_m(uw)
    kaps  = red_m(kap)
    PE    = red_v(PE_raw)
    δb    = red_v(δb_raw)
    ustar = red_v(ustar_raw)
    nx    = length(xs)

    # --- (3) RATIOS, from the reduced fields --------------------------------
    # Masked where the gradient is below the floor: inside the mixed layer
    # −F_b/(dB/dz) is 0/0. See the GRAD_FLOOR note above — this mask is the
    # dominant uncertainty in every K_T here.
    floor_val = GRAD_FLOOR * N²_ref
    mask = abs.(G) .>= floor_val
    K_T   = [mask[k, n] ? -F_b[k, n] / G[k, n] : NaN for k in axes(G, 1), n in axes(G, 2)]
    K_sgs = [mask[k, n] ? -F_s[k, n] / G[k, n] : NaN for k in axes(G, 1), n in axes(G, 2)]
    K_ratio = K_sgs ./ K_T

    # Mixed-layer height: normalised gradient first reaching 0.1, the same
    # definition (and the same helper) Figure5.jl uses. The gradient here is the
    # model's own ∂z(b) on Faces rather than an offline difference of B on
    # Centers — same definition, better discretisation.
    h = [first_crossing(zf, view(G, :, n) ./ N²_ref, 0.1) for n in 1:nx]

    # --- bulk relation ------------------------------------------------------
    #   (1/Lz)∫₀^zref F_b dz = −(1/Lz)∫₀^zref K_T dB/dz dz ≈ −K_T δb/Lz
    # The approximation holds because dB/dz ≈ 0 below h and the background is
    # uniform above, so the integral is dominated by the sharp step at z = h.
    # PULLING K_T OUT ASSUMES IT IS ROUGHLY CONSTANT ACROSS THE INTERFACE — that,
    # not the shape of the profile, is the actual assumption, and delta_eff below
    # is what tests whether it is safe. The 1/Lz cancels.
    Fint     = [trapz_to(zf, view(F_b, :, n), zref) for n in 1:nx]
    K_T_bulk = -Fint ./ δb                              # δb > 0, F_b < 0 ⇒ K > 0

    # Independent cross-check. K_T_pe comes from B alone; K_T_bulk from wb and
    # F_sgs. They share no code path, so a disagreement points at the flux — and
    # usually at the SGS half of it. This is the most valuable single test here.
    #
    # CORRECTION TO THE BRIEF. Integrating the buoyancy budget ∂B/∂t = −∂F_b/∂z
    # by parts leaves a boundary term the brief drops:
    #
    #     d/dt ∫₀^H z B dz  =  −H·F_b(H) + ∫₀^H F_b dz
    #
    # so with PE = −(1/Lz)∫₀^H zB dz,
    #
    #     K_T_bulk = Lz·(dPE/dt)/Δb  −  H·F_b(H)/Δb  =  K_T_pe + K_bdy
    #
    # The brief's K_T_pe = Lz(dPE/dt)/Δb assumes F_b(zref) = 0. THAT IS NOT TRUE
    # FOR THIS BACKGROUND: the softplus puts N² = N∞² at EVERY level above z = T,
    # so there is background SGS + molecular flux at every candidate zref and no
    # height where F_b vanishes. Measured on the smoke run the dropped term was
    # LARGER than K_T_pe itself, and restoring it brought the two routes to within
    # 2–10 % (the residual being the dPE/dt differencing).
    #
    # Both are kept: K_T_pe is the brief's fully independent version, K_T_pe_full
    # is the exact identity. The correction touches F_b at ONE level, well above
    # the mixed layer, so K_T_pe_full still gets all of its structure from B and
    # the cross-check keeps its diagnostic power.
    K_T_pe   = Lz .* dPEdt ./ δb
    F_zref   = [interp_at(zf, view(F_b, :, n), zref) for n in 1:nx]
    K_bdy    = -zref .* F_zref ./ δb
    K_T_pe_full = K_T_pe .+ K_bdy
    # How much flux is still crossing zref, relative to what the layer below it is
    # carrying. Small means zref is high enough for the brief's approximation; of
    # order 1 means it is not, and the boundary term is doing the work.
    leakage  = abs.(zref .* F_zref) ./ max.(abs.(Fint), eps())

    # (b) "peak = mean" shorthand, TESTED rather than assumed. Replacing the
    # integral by the peak value is only legitimate if the flux profile's
    # effective width is comparable to the depth it is spread over. Measure it:
    kref = findlast(z -> z <= zref, zf)
    Fpk  = [(c = view(F_b, 1:kref, n); g = filter(isfinite, c);
             isempty(g) ? NaN : (abs(minimum(g)) >= abs(maximum(g)) ? minimum(g) : maximum(g)))
            for n in 1:nx]
    δ_eff = Fint ./ Fpk                                 # metres

    # --- panel (d) sampling -------------------------------------------------
    # K_T and TKE at z = h(t): the mixed-layer top is where the two candidate
    # scalings differ, and where the flux actually is.
    K_at_h   = [interp_at(zf, view(K_T, :, n),  h[n]) for n in 1:nx]
    TKE_at_h = [interp_at(zc, view(TKE, :, n),  h[n]) for n in 1:nx]
    fit_keep = SMOOTH == "phase" ? trues(nx) : (times .>= SKIP_PERIODS * T_tide)
    slope, rcorr, nfit = loglog_slope(TKE_at_h[fit_keep], K_at_h[fit_keep])

    return (; tag, T, s, Ri, N²_ref, zref, cᴰ, zc, zf, xs, xlab, times,
              W_max, TKE, K_T, K_sgs, K_ratio, K_T_bulk, K_T_pe, K_T_pe_full,
              K_bdy, leakage, δ_eff, h, PE, δb,
              ustar, Fint, Fpk, F_b, G, Bs, Us, uws, kaps, dPEdt,
              K_at_h, TKE_at_h, fit_keep, slope, rcorr, nfit)
end

# Centred difference on a possibly-uneven time axis; one-sided at the ends.
function centred_diff(a, t)
    n = length(a)
    d = fill(NaN, n)
    for i in 2:n-1
        d[i] = (a[i+1] - a[i-1]) / (t[i+1] - t[i-1])
    end
    n >= 2 && (d[1] = (a[2] - a[1]) / (t[2] - t[1]);
               d[n] = (a[n] - a[n-1]) / (t[n] - t[n-1]))
    return d
end

# ---------------------------------------------------------------------------
# Verification report — the five checks, in the order they must be read
# ---------------------------------------------------------------------------
# Returns (fails, metrics). `metrics` is saved into the results file so the sweep
# driver can gate stage 3 on check 2 without re-deriving it — see
# swirlesrun.jl / run_moments_sweep.sh.
function verify(c)
    fails = String[]
    met = Dict{String,Float64}("w_over_U0" => NaN, "pe_rel_diff" => NaN,
                               "pe_rel_diff_uncorrected" => NaN, "zref_leakage" => NaN,
                               "K_sgs_over_K_T" => NaN, "delta_eff" => NaN,
                               "delta_eff_cv" => NaN, "slope" => NaN,
                               "slope_r" => NaN, "slope_n" => NaN)
    println("\n", "="^78)
    @printf("VERIFICATION  %s   (T = %g m, N/ω = %g, Ri = %g, zref = %.1f m, GRAD_FLOOR = %.3f, SMOOTH = %s)\n",
            c.tag, c.T, c.s, c.Ri, c.zref, GRAD_FLOOR, SMOOTH)
    println("="^78)

    # 1. ⟨w⟩_xy
    r = c.W_max / U₀
    ok1 = r < 1e-10
    met["w_over_U0"] = r
    @printf("  1. max|⟨w⟩_xy| = %.3e m/s  (= %.2e U₀)                     %s\n",
            c.W_max, r, ok1 ? "PASS" : "FAIL")
    ok1 || (push!(fails, "⟨w⟩_xy ≠ 0 — the plane average is not a clean Reynolds average, every flux is wrong");
            println("     STOP HERE: nothing below this line is meaningful."))

    # 2. the two K_T routes. The verdict is taken on the EXACT identity
    #    (K_T_pe_full, boundary term restored); the brief's uncorrected K_T_pe is
    #    printed beside it so the size of the dropped term is visible.
    m = @. isfinite(c.K_T_bulk) & isfinite(c.K_T_pe_full) & c.fit_keep
    if count(m) >= 8
        a = c.K_T_bulk[m]
        scale = max(mean(abs.(a)), eps())
        relof(b) = sqrt(mean((a .- b) .^ 2)) / scale
        rel, bias = relof(c.K_T_pe_full[m]), mean(a .- c.K_T_pe_full[m]) / scale
        met["pe_rel_diff"] = rel
        rel_raw = relof(c.K_T_pe[m])
        met["pe_rel_diff_uncorrected"] = rel_raw
        lk = filter(isfinite, c.leakage[c.fit_keep])
        met["zref_leakage"] = isempty(lk) ? NaN : median(lk)
        ok2 = rel < 0.3
        @printf("  2. K_T_bulk vs K_T_pe_full: rel RMS diff = %5.1f %%, bias = %+5.1f %%   %s\n",
                100rel, 100bias, ok2 ? "PASS" : "FAIL")
        @printf("     (brief's uncorrected K_T_pe: %.1f %%; flux leaking past zref = %.2f × ∫F_b\n",
                100rel_raw, isempty(lk) ? NaN : median(lk))
        @printf("      — %s, so the boundary term %s be dropped)\n",
                isempty(lk) || median(lk) < 0.1 ? "small" : "NOT small",
                isempty(lk) || median(lk) < 0.1 ? "may" : "must NOT")
        ok2 || push!(fails, @sprintf("K_T_bulk and K_T_pe_full differ by %.0f %% — the SGS flux is likely %s",
                                     100rel, bias < 0 ? "under-counted (the flux route reads low)" : "over-counted"))
    else
        println("  2. K_T_bulk vs K_T_pe: too few finite samples to compare        FAIL")
        push!(fails, "K_T_bulk/K_T_pe could not be compared")
    end

    # 3. SGS share at the mixed-layer top
    rat = [interp_at(c.zf, view(c.K_ratio, :, n), c.h[n]) for n in eachindex(c.h)]
    g = filter(isfinite, rat[c.fit_keep])
    if isempty(g)
        println("  3. K_sgs/K_T at z = h: no finite samples                        FAIL")
        push!(fails, "K_sgs/K_T could not be evaluated")
    else
        med = median(g)
        met["K_sgs_over_K_T"] = med
        ok3 = med < 0.5
        @printf("  3. K_sgs/K_T at z = h: median %.2f (90th pct %.2f)              %s\n",
                med, quantile(g, 0.9), ok3 ? "PASS" : "FAIL")
        ok3 || push!(fails, @sprintf("K_sgs/K_T = %.2f — K_T characterises the AMD closure, not the flow", med))
    end

    # 4. shape factor
    de = filter(isfinite, c.δ_eff[c.fit_keep])
    hh = filter(isfinite, c.h[c.fit_keep])
    if length(de) >= 8
        cv = std(de) / abs(mean(de))
        met["delta_eff"] = mean(de); met["delta_eff_cv"] = cv
        ok4 = cv < 0.5
        @printf("  4. delta_eff = %.2f ± %.2f m (CV %.2f); mean h = %.2f m          %s\n",
                mean(de), std(de), cv, isempty(hh) ? NaN : mean(hh), ok4 ? "PASS" : "FAIL")
        @printf("     delta_eff/Lz = %.3f — the 'peak = mean' shorthand needs this ≈ 1; %s\n",
                mean(de) / Lz,
                mean(de) / Lz > 0.5 ? "it is justified here" : "IT IS NOT — use the integral")
        ok4 || push!(fails, @sprintf("delta_eff wanders (CV %.2f) — use the integral, not the peak", cv))
    else
        println("  4. delta_eff: too few finite samples                            FAIL")
    end

    # 5. the answer — with the two things that make a slope UNREADABLE stated
    #    before the number, not after it.
    @printf("  5. panel (d) slope d(log K_T)/d(log TKE) at z = h = %+.2f  (r = %+.2f, n = %d)\n",
            c.slope, c.rcorr, c.nfit)

    # (a) AT N = 0 THE TEST DOES NOT EXIST. K_T ~ TKE/N is undefined there, so the
    #     unstratified case cannot discriminate between the two scalings no matter
    #     how clean the fit is. It is a control — it says the machinery works and
    #     gives the √TKE·l branch with nothing suppressing it — and that is all.
    #     Saying "nearest: √TKE·l" here would be comparing against a hypothesis
    #     that has no value to compare with.
    unstratified = c.Ri == 0
    weak_fit = isfinite(c.rcorr) && abs(c.rcorr) < R_MIN

    if unstratified
        println("     N/ω = 0: K_T ~ TKE/N is UNDEFINED (N = 0), so this case CANNOT")
        println("     discriminate between the two scalings. It is the unstratified control:")
        println("     it shows the machinery works and the √TKE·l branch unsuppressed.")
        println("     The exponent needs the stratified cases.")
    else
        @printf("     1/2 ⇒ K_T ~ √TKE·l ;  1 ⇒ K_T ~ TKE/N.  Nearest: %s\n",
                isnan(c.slope) ? "—" :
                abs(c.slope - 0.5) < abs(c.slope - 1.0) ? "√TKE·l" : "TKE/N")
    end

    # (b) A SLOPE WITHOUT A CORRELATION IS NOT A MEASUREMENT. A least-squares slope
    #     through a round cloud is still a number, and with n in the thousands its
    #     formal error bar is tiny — which is exactly how a null result gets
    #     reported as a finding. r² is the honest summary of how much of the
    #     scatter the power law actually accounts for.
    if weak_fit
        @printf("     WEAK FIT: r = %.2f means the power law explains only %.0f %% of the\n",
                c.rcorr, 100 * c.rcorr^2)
        @printf("     scatter. Do not quote this slope as an exponent. Usual causes: too few\n")
        println("     independent samples (SMOOTH=tide gives the envelope, tide20 does not),")
        println("     or h jittering between samples, so 'at z = h' is a moving target.")
    end

    if !isempty(fails)
        println("  → FAILED CHECKS (read no further down the list than the first):")
        for f in fails; println("      • ", f); end
    elseif unstratified
        println("  → checks 1-4 pass: the measurement chain is sound. Check 5 is not")
        println("    applicable at N = 0 — run the stratified cases for the exponent.")
    elseif weak_fit
        println("  → checks 1-4 pass, but the check-5 fit is too weak to quote a slope.")
    else
        println("  → all checks pass; the slope above can be read as physics.")
    end
    met["slope"]   = c.slope
    met["slope_r"] = c.rcorr
    met["slope_n"] = float(c.nfit)
    return fails, met
end

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
const heat_map = cgrad(["#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
const div_map  = cgrad(["#7A3117", "#B4502C", "#D9855F", "#E9E7E4",
                        "#7FADE0", "#3C7CC4", "#1B4E8F"])

function make_figure(c)
    ztop = min(Lz, c.zref)
    kc = findall(z -> z <= ztop, c.zc)
    kf = findall(z -> z <= ztop, c.zf)

    # (a) TKE with h overlaid
    pa = heatmap(c.xs, c.zc[kc], c.TKE[kc, :] ./ U₀^2;
                 color = heat_map, xlabel = c.xlab, ylabel = "z (m)",
                 title = "(a) TKE / U₀²", colorbar_title = "")
    plot!(pa, c.xs, c.h; color = :black, lw = 2, label = "h")
    hline!(pa, [c.T]; color = :grey40, ls = :dash, lw = 1, label = "z = T")

    # (b) K_T, log-scaled: it spans decades and a linear scale shows one cell.
    Klog = [isfinite(v) && v > 0 ? log10(v) : NaN for v in c.K_T[kf, :]]
    pb = heatmap(c.xs, c.zf[kf], Klog;
                 color = div_map, xlabel = c.xlab, ylabel = "z (m)",
                 title = @sprintf("(b) log₁₀ K_T (m² s⁻¹), masked |dB/dz| < %.2f N²_ref", GRAD_FLOOR))
    plot!(pb, c.xs, c.h; color = :black, lw = 2, label = "h")

    # (c) THE cross-check. Same axes, deliberately: they share no code path.
    #
    # LOG y AXIS, as in (b) and (d): K_T_bulk swings over more than a decade
    # within a tidal cycle and again between stratifications. On a linear axis
    # everything outside the peak is pressed onto zero, so the agreement this
    # panel exists to show is legible only where it matters least.
    #
    # A log axis cannot render a sign change, and the brief's uncorrected
    # K_T_pe is the one series that goes negative — it is no longer drawn.
    # Nothing is lost: check 2 reports its error as a number for every case, and
    # K_T_pe is still computed and still saved to mixing_<tag>.jld2. Its message
    # is quantitative ("dropping the boundary term costs 107 %"), which the
    # verification block states far more precisely than a curve ever did.
    #
    # Non-positive samples cannot be drawn on a log axis. They are dropped, and
    # the count goes in the title rather than being left to infer from a gap.
    onlypos(v) = (isfinite(v) && v > 0) ? v : NaN
    cb, cpe = onlypos.(c.K_T_bulk), onlypos.(c.K_T_pe_full)
    ndrop = count(v -> isfinite(v) && v <= 0, c.K_T_bulk) +
            count(v -> isfinite(v) && v <= 0, c.K_T_pe_full)
    ctitle = "(c) two independent routes to K_T"
    ndrop > 0 && (ctitle *= @sprintf("\n%d of %d samples ≤ 0, not drawable on a log axis",
                                     ndrop, 2 * length(cb)))

    pc = plot(c.xs, cb; yscale = :log10, color = "#1B4E8F", lw = 2,
              label = "K_T_bulk = −∫F_b dz / Δb", xlabel = c.xlab,
              ylabel = "K_T (m² s⁻¹)", title = ctitle,
              titlefontsize = ndrop > 0 ? 8 : 10, legend = :topright)
    plot!(pc, c.xs, cpe; color = "#B4502C", lw = 2, ls = :dash,
          label = "K_T_pe (+ boundary term)")
    # A shared y-axis would hide a disagreement, which is the whole point of the
    # panel; instead the shape factor rides along on a twin axis.
    #
    # Plots.jl draws no legend for a twinx series, so an all-NaN proxy on the main
    # axis carries the label into the main legend — an unlabelled dotted line was
    # unreadable, and a second legend box would land on top of the first. The
    # "(right axis)" is load-bearing: δ_eff is in metres, not m² s⁻¹, and shares
    # none of the left axis's scale.
    plot!(pc, c.xs, fill(NaN, length(c.xs)); color = "#7FADE0", lw = 1.2, ls = :dot,
          label = "δ_eff = ∫F_b dz / F_b|peak  (right axis, m)")
    plot!(twinx(pc), c.xs, c.δ_eff; color = "#7FADE0", lw = 1.2, ls = :dot,
          ylabel = "δ_eff (m)", label = "", legend = false)

    # (d) the answer
    x, y = c.TKE_at_h[c.fit_keep], c.K_at_h[c.fit_keep]
    m = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
    # THE TITLE CARRIES THE CAVEAT, not just the number. A slope through a round
    # cloud is still a slope, and a reader who sees only "slope = 0.17" on a plot
    # will quote it. r² says how much of the scatter the power law accounts for,
    # and at N = 0 the TKE/N hypothesis does not exist to be compared against.
    unstrat = c.Ri == 0
    weak = isfinite(c.rcorr) && abs(c.rcorr) < R_MIN
    ttl = @sprintf("(d) slope = %.2f (r = %.2f, r² = %.2f, n = %d)",
                   c.slope, c.rcorr, c.rcorr^2, c.nfit)
    unstrat && (ttl *= "\nN = 0: CONTROL ONLY — TKE/N undefined, cannot discriminate")
    weak && !unstrat && (ttl *= "\nWEAK FIT — not quotable as an exponent")
    pd = plot(xscale = :log10, yscale = :log10, xlabel = "TKE at z = h (m² s⁻²)",
              ylabel = "K_T at z = h (m² s⁻¹)", title = ttl,
              titlefontsize = unstrat || weak ? 8 : 10)
    if count(m) >= 8
        scatter!(pd, x[m], y[m]; ms = 2.5, mc = "#3C7CC4", msw = 0, label = "", alpha = 0.5)
        xr = [minimum(x[m]), maximum(x[m])]
        x0, y0 = exp10(mean(log10.(x[m]))), exp10(mean(log10.(y[m])))
        plot!(pd, xr, y0 .* (xr ./ x0) .^ c.slope; color = :black, lw = 2,
              label = @sprintf("fit %.2f%s", c.slope, weak ? " (weak)" : ""))
        plot!(pd, xr, y0 .* (xr ./ x0) .^ 0.5; color = "#B4502C", ls = :dash, lw = 1.5,
              label = "½ → √TKE·l")
        # The TKE/N line is omitted at N = 0 rather than drawn and ignored: it is
        # a reference to a hypothesis with no value at this N, and drawing it
        # invites exactly the comparison the case cannot support.
        unstrat || plot!(pd, xr, y0 .* (xr ./ x0) .^ 1.0; color = "#7A3117", ls = :dot,
                         lw = 1.5, label = "1 → TKE/N")
    else
        # No annotate! here: on empty log-scaled axes there is no data range to
        # place it in. The title already carries n.
        plot!(pd, title = "(d) too few unmasked points (n = $(count(m)))")
    end

    fig = plot(pa, pb, pc, pd; layout = (2, 2), size = (1250, 850),
               leftmargin = 6Plots.mm, bottommargin = 6Plots.mm,
               plot_title = @sprintf("%s — T = %g m, N/ω = %g (Ri = %g), zref = %.1f m, GRAD_FLOOR = %.2f, %s smoothing",
                                     c.tag, c.T, c.s, c.Ri, c.zref, GRAD_FLOOR, SMOOTH),
               plot_titlefontsize = 11)
    out = joinpath(figdir, "K_T_$(c.tag)$(RESULT_SUFFIX).png")
    savefig(fig, out)
    return out
end

# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------
all_fails = Dict{String,Vector{String}}()
nfound = 0

for T in T_values, s in n_over_ω
    c = analyse(T, s)
    c === nothing && continue
    global nfound += 1

    fails, met = verify(c)
    isempty(fails) || (all_fails[c.tag] = fails)

    # Results file. Its own "mixing_" prefix, so it cannot collide with anything
    # a simulation writer produces (those all use overwrite_existing = true).
    outdir = joinpath(@__DIR__, outroot, c.tag)
    mkpath(outdir)
    resfile = joinpath(outdir, "mixing_$(c.tag)$(RESULT_SUFFIX).jld2")
    jldsave(resfile;
            tag = c.tag, T = c.T, n_over_omega = c.s, Ri = c.Ri, N2_ref = c.N²_ref,
            zref = c.zref, grad_floor = GRAD_FLOOR, smooth = SMOOTH,
            cD = c.cᴰ, times = c.times, x = c.xs, xlabel = c.xlab,
            z_center = c.zc, z_face = c.zf,
            TKE = c.TKE, K_T = c.K_T, K_sgs = c.K_sgs, K_sgs_over_K_T = c.K_ratio,
            K_T_bulk = c.K_T_bulk, K_T_pe = c.K_T_pe, K_T_pe_full = c.K_T_pe_full,
            K_boundary_term = c.K_bdy, zref_leakage = c.leakage, delta_eff = c.δ_eff,
            h = c.h, PE = c.PE, dPEdt = c.dPEdt, delta_b = c.δb, ustar = c.ustar,
            F_b = c.F_b, dBdz = c.G, kappa_sgs = c.kaps, uw = c.uws,
            F_integral = c.Fint, F_peak = c.Fpk,
            K_at_h = c.K_at_h, TKE_at_h = c.TKE_at_h,
            slope = c.slope, slope_r = c.rcorr, slope_n = c.nfit,
            max_abs_W = c.W_max,
            # Machine-readable copy of the VERIFICATION block, so the sweep driver
            # can gate stage 3 on check 2 without re-deriving it.
            checks = met)

    fig = make_figure(c)
    @printf("  wrote %s\n         %s\n", relpath(resfile, @__DIR__), relpath(fig, @__DIR__))
end

println("\n", "="^78)
if nfound == 0
    @warn "No *_moments.jld2 found under $outroot/ for T ∈ $(T_values), N/ω ∈ $(n_over_ω). " *
          "Run the simulations with MOMENTS=1 (the default) first — the older runs in " *
          "outputs/ predate Moments.jl and have no second moments on disk."
elseif isempty(all_fails)
    @printf("%d case(s) analysed; ALL VERIFICATION CHECKS PASS.\n", nfound)
else
    @printf("%d case(s) analysed; %d with failing checks:\n", nfound, length(all_fails))
    for (tag, fs) in sort(collect(all_fails), by = first)
        println("  ", tag)
        for f in fs; println("      • ", f); end
    end
    println("\nDo not read the panel (d) slope for a case whose check 1, 2 or 3 failed.")
end
println("="^78)
