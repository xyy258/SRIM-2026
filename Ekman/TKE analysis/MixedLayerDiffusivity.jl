ENV["GKSwstype"] = "100"        # headless GR — must precede `using Plots`

using Oceananigans, JLD2, Plots, Printf, Statistics

# Turbulent diffusivity of the stratified Ekman bottom boundary layer, from the
# second moments Moments.jl writes to Moments.jld2.
#
# The question this answers: does K_T scale as √TKE · l, or as TKE / N? The two
# differ by the slope of log K_T against log TKE at the mixed-layer top —
# 1/2 for √TKE·l, 1 for TKE/N. That is panel (d).
#
#     julia --project=<repo root> "Ekman/TKE analysis/MixedLayerDiffusivity.jl"
#
# It writes
#   Data/<case>/mixing_<case>.jld2   TKE(z,t), K_T(z,t), K_T_bulk, K_T_pe,
#                                    delta_eff, K_sgs/K_T, h, PE, u*
#   Plots/K_T_<case>.png             4 panels (see below)
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
# THIS IS A COPY OF Stokes/3D/MixedLayerDiffusivity.jl
# ---------------------------------------------------------------------------
# Everything from the helpers down — the decomposition, the smoothing, the two
# K_T routes, the boundary-term correction, the checks and the figure — is
# UNCHANGED, so the tidal and Ekman answers are produced by the same code and can
# be compared without arguing about method. What differs is confined to the block
# immediately below and to three lines inside `analyse`:
#
#     Stokes                        here
#     ------                        ----
#     ω = 1e-4 (tidal)              f₀ = 1e-4 (Coriolis) — NUMERICALLY IDENTICAL,
#                                   which is why the two studies share windows
#     T_tide = 2π/ω                 T_f = 2π/f₀, the inertial period
#     U₀ = 0.04                     U∞ = 0.04
#     κ = ν/Pr                      κ₀ = ν₀/Pr   (`κ` is 0.41 here — see Moments.jl)
#     Lz = 50                       Lz = 100
#     N²_ref = Ri·ω², ω² at Ri=0    N²_ref = (r f₀)², 1 at r = 0 (matches `scale`
#                                   in Ekman3D.jl, which is 1 not f₀² there)
#     cᴰ from a reference height    cᴰ from the log law at the first cell centre,
#                                   which on THIS grid is genuinely valid
#     a grid of (T, N/ω) cases      one case
#
# The aliases `ω`, `T_tide` and `U₀` are kept as names so the body needs no edit;
# they are bound to the Ekman quantities. Do not "fix" them into f₀/T_f/U∞ — the
# point is that this file and its Stokes twin stay diffable.
#
# ---------------------------------------------------------------------------
# ORDER OF OPERATIONS — this is where the physics is won or lost
# ---------------------------------------------------------------------------
#   (1) DECOMPOSE PER SAMPLE. The plane average IS the Reynolds average here
#       (Periodic×Periodic, horizontally uniform forcing), so u′ = u − ⟨u⟩_xy
#       exactly, and the mean flow — free stream, geostrophic balance and the
#       inertial oscillation — is removed instantaneously with it. Every mean is
#       subtracted at its OWN sample time.
#   (2) SMOOTH IN TIME AFTERWARDS, for noise reduction only. Doing it the other
#       way round gives  time_avg(⟨uu⟩) − time_avg(U)² = time_avg(TKE) + Var_t(U):
#       the variance of the mean flow over the window leaks in as "TKE". Here that
#       mean flow is an inertial oscillation of amplitude comparable to U∞, so the
#       leak is if anything worse than in the tidal case.
#   (3) FORM RATIOS LAST. K_T = −F_b/(dB/dz) is built from the already-smoothed
#       F_b and dBdz, never from smoothed ratios: a ratio of two noisy profiles is
#       badly behaved wherever the denominator is small, which is most of the
#       mixed layer.
# Moments.jl guarantees step (1) by writing raw instantaneous moments on a plain
# TimeInterval; see its header for why AveragedTimeInterval would break it.

# ---------------------------------------------------------------------------
# The case. Must agree with Parameters.jl — read from the environment with the
# same defaults, rather than including it, because Parameters.jl declares consts
# that clash with the aliases below.
# ---------------------------------------------------------------------------
const r  = parse(Float64, get(ENV, "R",       "1"))       # N/f
const Tc = parse(Float64, get(ENV, "T_STRAT", "20"))      # pycnocline height (m)

const f₀ = 1e-4
const ω  = f₀                          # alias: the inertial frequency plays the
                                       # tidal frequency's role in the body below
const ν  = 1.0e-6
const Pr = 10
const κ_mol = ν / Pr                   # = κ₀ in Parameters.jl. `κ` there is 0.41.
const U∞ = 0.04
const U₀ = U∞                          # alias
const T_f = 2π / f₀                    # inertial period, 62832 s = 17.45 h
const T_tide = T_f                     # alias
const Lz = 100.0                       # PHYSICAL domain height; sponge sits above
const z₀_rough = 0.0016                # roughness length (m)
const κ_vk = 0.41                      # von Kármán constant

const casename = get(ENV, "CASE_NAME", @sprintf("r=%.1f, T=%.1f", r, Tc))
const dataroot = get(ENV, "DATA_ROOT", joinpath(@__DIR__, "Data"))
moments_file(name) = joinpath(dataroot, name, "Moments.jld2")

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

# ---------------------------------------------------------------------------
# Time reduction. Applied only AFTER the per-sample decomposition (step 2 above).
# ---------------------------------------------------------------------------
#   tide20 (default)  boxcar of T_f/20 ≈ 52 min — kills plane-average noise while
#                     keeping the turbulence burst within an inertial cycle.
#                     Spelled "inertial20" too; the tidal names are kept so this
#                     file stays diffable against its Stokes twin.
#   tide              boxcar of the full inertial period — the slowly evolving
#                     envelope, which is what boundary-layer growth should be read
#                     from, and the mode to use for the panel (d) slope here
#   phase             bin by mod(f₀t, 2π), ensembled over periods. Meaningful
#                     because the spin-up rings at f, but with only ~4 usable
#                     periods the ensemble is thin — treat it as a cross-check.
const SMOOTH = replace(get(ENV, "SMOOTH", "tide20"), "inertial" => "tide")
const N_PHASE = parse(Int, get(ENV, "N_PHASE", "40"))
# Inertial periods to discard before ensembling / fitting. THIS RUN STARTS FROM
# REST-RELATIVE IMBALANCE, not from a turbulent snapshot the way the Stokes cases
# do: u = U∞ + noise everywhere at t = 0, and the boundary layer has to trip,
# become turbulent and grow. That transient is not the physics being measured, so
# the default here is 2 periods rather than the Stokes 1 — out of the 6.37 the
# run lasts, leaving ~4.4 usable.
const SKIP_PERIODS = parse(Float64, get(ENV, "SKIP_PERIODS", "2.0"))

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

const figdir = get(ENV, "FIG_DIR", joinpath(@__DIR__, "Plots"))
mkpath(figdir)

# ---------------------------------------------------------------------------
# Per-case analysis
# ---------------------------------------------------------------------------
function analyse(T, s)
    tag = casename
    fname = moments_file(tag)
    isfile(fname) || (@warn "Missing $fname — skipping $tag"; return nothing)

    # N²_ref matches Ekman3D.jl's `scale = (r==0) ? 1 : N²`, which is NOT the
    # Stokes convention: there a passive tracer at Ri = 0 carries an ω²-scaled
    # background, here it carries an unscaled one. r = 1 is the case that runs, so
    # this only matters if someone sets R=0.
    Ri = s^2                                  # s = N/f, so Ri = N²/f² as before
    N²_ref = Ri > 0 ? Ri * f₀^2 : 1.0

    # zref must sit well above the interface but well below the sponge at z = Lz.
    # The Ekman domain is 100 m deep against the Stokes 50, so the T + 15 rule has
    # far more headroom here; the cap is raised in proportion.
    zref = min(T + 15, 80.0)

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
    # cᴰ MUST MATCH THE RUN. Ekman3D.jl forms it as (κ_vk/log(z₁/z₀))² at the first
    # cell centre, and unlike the Stokes grid this one does not resolve below the
    # roughness sublayer — z₁ = 0.0667 m = 41.7 z₀ — so the log law is being used
    # inside its range of validity and no reference height is needed. cᴰ ≈ 0.0121.
    z₁ = zc[1]
    cᴰ = parse(Float64, get(ENV, "CD", string((κ_vk / log(z₁ / z₀_rough))^2)))
    ustar_raw = [sqrt(cᴰ * (U[1, n]^2 + V[1, n]^2)) for n in 1:nt]

    # --- (2) TIME REDUCTION -------------------------------------------------
    # Anonymous functions, not `red_m(A) = ...`: a named definition in each branch
    # of an if would be two method definitions of the same local name.
    local xs, red_m, red_v, xlab
    if SMOOTH == "phase"
        xs, idx = phase_bins(times)
        red_m = A -> phase_reduce(A, idx)
        red_v = a -> phase_reduce(a, idx)
        xlab = "f t mod 2π (rad)"
    else
        dt = length(times) > 1 ? times[2] - times[1] : 1.0
        nh = max(0, round(Int, smooth_window / (2dt)))
        xs = ω .* times
        red_m = A -> boxcar(A, nh)
        red_v = a -> boxcar(a, nh)
        xlab = "f t  (inertial periods × 2π)"
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
                               "delta_eff_cv" => NaN, "slope" => NaN)
    println("\n", "="^78)
    @printf("VERIFICATION  %s   (T = %g m, N/f = %g, Ri = %g, zref = %.1f m, GRAD_FLOOR = %.3f, SMOOTH = %s)\n",
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

    # 5. the answer
    @printf("  5. panel (d) slope d(log K_T)/d(log TKE) at z = h = %+.2f  (r = %+.2f, n = %d)\n",
            c.slope, c.rcorr, c.nfit)
    @printf("     1/2 ⇒ K_T ~ √TKE·l ;  1 ⇒ K_T ~ TKE/N.  Nearest: %s\n",
            isnan(c.slope) ? "—" :
            abs(c.slope - 0.5) < abs(c.slope - 1.0) ? "√TKE·l" : "TKE/N")

    if isempty(fails)
        println("  → all checks pass; the slope above can be read as physics.")
    else
        println("  → FAILED CHECKS (read no further down the list than the first):")
        for f in fails; println("      • ", f); end
    end
    met["slope"] = c.slope
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
    pc = plot(c.xs, c.K_T_bulk; color = "#1B4E8F", lw = 2,
              label = "K_T_bulk = −∫F_b dz / Δb", xlabel = c.xlab,
              ylabel = "K_T (m² s⁻¹)", title = "(c) two independent routes to K_T")
    plot!(pc, c.xs, c.K_T_pe_full; color = "#B4502C", lw = 2, ls = :dash,
          label = "K_T_pe (+ boundary term)")
    # The brief's uncorrected version, kept visible: the gap between the two dashed
    # curves IS the zref·F_b(zref) term it drops.
    plot!(pc, c.xs, c.K_T_pe; color = "#D9855F", lw = 1.2, ls = :dashdot,
          label = "K_T_pe = Lz·(dPE/dt)/Δb (uncorrected)")
    hline!(pc, [0.0]; color = :grey60, lw = 0.7, label = "")
    # A second y-axis would hide a disagreement, which is the whole point of the
    # panel; instead the shape factor rides along on a twin axis.
    plot!(twinx(pc), c.xs, c.δ_eff; color = "#7FADE0", lw = 1.2, ls = :dot,
          ylabel = "δ_eff (m)", label = "", legend = false)

    # (d) the answer
    x, y = c.TKE_at_h[c.fit_keep], c.K_at_h[c.fit_keep]
    m = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
    pd = plot(xscale = :log10, yscale = :log10, xlabel = "TKE at z = h (m² s⁻²)",
              ylabel = "K_T at z = h (m² s⁻¹)",
              title = @sprintf("(d) slope = %.2f (r = %.2f, n = %d)", c.slope, c.rcorr, c.nfit))
    if count(m) >= 8
        scatter!(pd, x[m], y[m]; ms = 2.5, mc = "#3C7CC4", msw = 0, label = "", alpha = 0.5)
        xr = [minimum(x[m]), maximum(x[m])]
        x0, y0 = exp10(mean(log10.(x[m]))), exp10(mean(log10.(y[m])))
        plot!(pd, xr, y0 .* (xr ./ x0) .^ c.slope; color = :black, lw = 2,
              label = @sprintf("fit %.2f", c.slope))
        plot!(pd, xr, y0 .* (xr ./ x0) .^ 0.5; color = "#B4502C", ls = :dash, lw = 1.5,
              label = "½ → √TKE·l")
        plot!(pd, xr, y0 .* (xr ./ x0) .^ 1.0; color = "#7A3117", ls = :dot, lw = 1.5,
              label = "1 → TKE/N")
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

# ONE CASE. The Stokes twin loops over a (T, N/ω) grid; this folder runs a single
# r = N/f, so the loop is kept — same code path, same output — over a list of one.
for T in [Tc], s in [r]
    c = analyse(T, s)
    c === nothing && continue
    global nfound += 1

    fails, met = verify(c)
    isempty(fails) || (all_fails[c.tag] = fails)

    # Results file, beside the moments file it was derived from. Its own "mixing_"
    # prefix, so it cannot collide with anything a simulation writer produces
    # (those all use overwrite_existing = true).
    outdir = joinpath(dataroot, c.tag)
    mkpath(outdir)
    resfile = joinpath(outdir, "mixing_$(c.tag)$(RESULT_SUFFIX).jld2")
    jldsave(resfile;
            tag = c.tag, T = c.T, n_over_f = c.s, Ri = c.Ri, N2_ref = c.N²_ref,
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
            checks = met)

    fig = make_figure(c)
    @printf("  wrote %s\n         %s\n", relpath(resfile, @__DIR__), relpath(fig, @__DIR__))
end

println("\n", "="^78)
if nfound == 0
    @warn "No Moments.jld2 found at $(moments_file(casename)). Run Ekman3D.jl with " *
          "MOMENTS=1 (the default) first — the runs under Ekman/Data predate Moments.jl " *
          "and have no second moments on disk."
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
