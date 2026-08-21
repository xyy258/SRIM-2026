ENV["GKSwstype"] = "100"        # headless GR — must precede `using Plots`

using Oceananigans, JLD2, Plots, Printf, Statistics

# Turbulent diffusivity of the tidal bottom boundary layer, computed from the
# second moments Moments.jl writes to *_moments.jld2.
#
# The question: does K_T scale as √TKE · l, or as TKE / N? The two differ in the
# slope of log K_T against log TKE at the mixed-layer top — 1/2 for √TKE·l, 1 for
# TKE/N. That slope is panel (d).
#
#   julia --project=. MixedLayerDiffusivity.jl
# Subsets, as in Figure4_metres.jl and Figure5.jl:
#   T_VALUES="3 5" N_OVER_OMEGA="1 2 5" julia --project=. MixedLayerDiffusivity.jl
#
# For each case it writes
#   outputs/<tag>/mixing_<tag>.jld2   TKE(z,t), K_T(z,t), K_T_bulk, K_T_pe,
#                                     delta_eff, K_sgs/K_T, h, PE, u*
#   figures/K_T_<tag>.png             four panels (see below)
# and prints a verification block. Read the checks in order — a failure early on
# invalidates everything after it:
#   1. ⟨w⟩_xy ≈ 1e-18. If not, stop: every flux below is wrong.
#   2. K_T_bulk ≈ K_T_pe, panel (c). The two share no code — one uses wb and
#      F_sgs, the other only B — so a disagreement means the subgrid flux is
#      wrong, usually under-counted.
#   3. K_sgs/K_T well below 1. Above about 0.5 the number describes the closure
#      rather than the flow.
#   3b. h unambiguous, i.e. the peak of dB/dz is a single interface.
#   4. delta_eff steady, ideally following h.
#   5. only then the panel (d) slope.
#
# ---------------- Order of operations ----------------
#   (1) Decompose each sample on its own. The plane average is the Reynolds
#       average here, so u′ = u − ⟨u⟩_xy exactly and the tidal flow is removed
#       with it. Every mean is subtracted at its own sample time.
#   (2) Smooth in time afterwards, to reduce noise. The other order gives
#       time_avg(⟨uu⟩) − time_avg(U)² = time_avg(TKE) + Var_t(U), so the variance
#       of the tidal mean flow leaks in as "TKE", at the same size as u*² itself.
#   (3) Form ratios last. K_T = −F_b/(dB/dz) is built from the already-smoothed
#       F_b and dBdz, since a ratio of two noisy profiles behaves badly wherever
#       the denominator is small, which is most of the mixed layer.
# Moments.jl guarantees step (1) by writing raw instantaneous moments; see its
# header for why an averaging schedule would break it.

const ω  = 1e-4
const ν  = 1.0e-6
const Pr = 10
const κ_mol = ν / Pr
const U₀ = 0.04
const T_tide = 2π / ω                  # 17.45 h, not the 12.4 h of a real M2
                                       # tide: ω is 1e-4 to match the Coriolis
                                       # parameter of the Ekman case.
const δs = sqrt(2ν / ω)
const Lz = 50.0

parse_list(key, default) = parse.(Float64, split(get(ENV, key, default)))
const T_values = parse_list("T_VALUES", "2 3 5 8")
const n_over_ω = parse_list("N_OVER_OMEGA", get(ENV, "SQRT_RI", "0 1 2 5 10"))

# Tag builder matching case_params.jl, Figure4_metres.jl and Figure5.jl.
num_lbl(x) = isinteger(x) ? string(Int(x)) : replace(string(x), "." => "p")
tag_of(T, s) = "P4_T" * num_lbl(T) * "_sqrtRi" * num_lbl(s)
const outroot = get(ENV, "OUT_ROOT", "outputs")
moments_file(tag) = joinpath(@__DIR__, outroot, tag, "TidalBL3D_" * tag * "_moments.jld2")

# ---------------- GRAD_FLOOR ----------------
# K_T = −F_b/(dB/dz) is 0/0 inside the mixed layer, where dB/dz goes to zero by
# definition, so cells with |dB/dz| below GRAD_FLOOR · N_ref² are masked out.
#
# This threshold is the largest single source of uncertainty in K_T: it decides
# how far into the weakly stratified layer the ratio is trusted, and K_T grows
# without bound as the denominator shrinks. Sweep it and quote the spread rather
# than a single number:
#     for g in 0.02 0.05 0.10; do
#         GRAD_FLOOR=$g RESULT_SUFFIX=_g$g julia --project=. MixedLayerDiffusivity.jl
#     done
# RESULT_SUFFIX keeps each pass of such a sweep in its own file.
const GRAD_FLOOR = parse(Float64, get(ENV, "GRAD_FLOOR", "0.05"))
const RESULT_SUFFIX = get(ENV, "RESULT_SUFFIX", "")
# The saved mixing_*.jld2 may need a suffix of its own: with several definitions
# of h on disk at once, the figures use the same filename in three folders while
# the data files share one folder and must stay distinguishable.
const MIX_SUFFIX = get(ENV, "MIX_SUFFIX", RESULT_SUFFIX)

# Minimum |r| for the panel (d) slope to be quoted as an exponent. A
# least-squares slope through a shapeless cloud is still a number, and with
# thousands of points its formal error bar is small, so the correlation is what
# says whether the power law describes the data at all.
const R_MIN = parse(Float64, get(ENV, "R_MIN", "0.5"))

# ---------------- Time reduction, applied after the decomposition ----------
#   tide20 (default)  boxcar of T_tide/20, about 52 min: removes plane-average
#                     noise while keeping the turbulence burst within a cycle
#   tide              boxcar over a full period: the slowly evolving envelope,
#                     which is what mixed-layer growth should be read from
#   phase             binned by mod(ωt, 2π) and averaged over periods, for the
#                     structure within a cycle
const SMOOTH = get(ENV, "SMOOTH", "tide20")
const N_PHASE = parse(Int, get(ENV, "N_PHASE", "40"))
# Periods to discard before averaging or fitting: the restart from the spin-up
# snapshot is a transient and not the physics being measured.
const SKIP_PERIODS = parse(Float64, get(ENV, "SKIP_PERIODS", "1.0"))

const smooth_window = SMOOTH == "tide20" ? T_tide / 20 :
                      SMOOTH == "tide"   ? T_tide      :
                      SMOOTH == "phase"  ? 0.0         :
                      error("SMOOTH must be tide20, tide or phase — got \"$SMOOTH\"")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The mixed-layer height, and first_crossing with it, live in one file that every
# script includes — see its header for the definition.
include(joinpath(@__DIR__, "mixed_layer_height.jl"))

interp_at(z, fv, z₀) = isnan(z₀) ? NaN : begin
    i = searchsortedlast(z, z₀)
    i < 1 && return fv[1]
    i >= length(z) && return fv[end]
    fv[i] + (fv[i+1] - fv[i]) * (z₀ - z[i]) / (z[i+1] - z[i])
end

# Trapezoidal integral of f over [0, zmax] on the actual node spacing. The
# vertical grid is stretched by a factor of about 40, so sum(f)*dz would be a
# different number rather than an approximation of this. The last cell is cut at
# zmax by interpolation, and where the nodes start above z = 0 the gap below is
# closed with `f0`.
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

# Centred boxcar of half-width `nh` samples, narrowing at the edges. A boxcar and
# a centred difference commute, so smoothing PE and then differencing gives the
# same K_T_pe as differencing and then smoothing.
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

# Phase bins of mod(ωt, 2π), averaged over the whole periods past SKIP_PERIODS.
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

# Least-squares slope of log10 y against log10 x, over the finite positive points.
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

    # N²_ref as in case_params.jl: at N/ω = 0 the buoyancy force is off but b is
    # still carried as a passive scalar, so ω² is the reference there rather
    # than a division by zero.
    Ri = s^2
    N²_ref = Ri > 0 ? Ri * ω^2 : ω^2

    # zref must sit well above the interface and well below the sponge at z = Lz.
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

    # --- (1) decompose each sample -----------------------------------------
    # ⟨w⟩_xy is zero with a rigid lid, an impermeable bottom and
    # incompressibility, so the fluxes need no mean subtracted. Only the
    # variances do, each using the mean at its own sample time.
    W_max = maximum(abs, W)
    uu′ = uu .- U .^ 2
    vv′ = vv .- V .^ 2
    ww′ = ww                                    # ⟨w⟩ = 0
    TKE_raw = 0.5 .* (uu′ .+ vv′ .+ ww′)        # Centers
    F_b_raw = wb .+ F_sgs                       # Faces, positive upward, already signed

    # PE = −(1/Lz)∫₀^zref z B dz, integrated to the same zref as the flux so the
    # boundary term cancels between the two routes. The 1/Lz is an arbitrary
    # normalisation and cancels again in K_T_pe.
    PE_raw = [-trapz_to(zc, zc .* B[:, n], zref; f0 = 0.0) / Lz for n in 1:nt]

    # Δb taken to the same zref as the integral; a mismatch would be a quiet bias.
    δb_raw = [interp_at(zc, B[:, n], zref) - interp_at(zc, B[:, n], 0.0) for n in 1:nt]

    # u*(t) from the drag law the bottom boundary condition applies, τ = cᴰ|u|u,
    # so u*² = cᴰ(U₁² + V₁²) at the first cell centre. This uses the plane-
    # averaged velocity where the boundary condition uses the pointwise one, and
    # it leaves out the molecular part, which is the larger term on this grid, so
    # read u* as a relative measure of the forcing over the cycle rather than as
    # the wall stress itself.
    #
    # cᴰ must match the run. The default follows case_params.jl: the log law at
    # the fixed reference height, not at the first cell centre.
    z₁ = zc[1]
    z_drag_ref = parse(Float64, get(ENV, "Z_DRAG_REF", "0.0667"))
    cᴰ = parse(Float64, get(ENV, "CD", string((0.41 / log(max(z_drag_ref, 2*0.0016) / 0.0016))^2)))
    ustar_raw = [sqrt(cᴰ * (U[1, n]^2 + V[1, n]^2)) for n in 1:nt]

    # --- (2) reduce in time -------------------------------------------------
    # Anonymous functions rather than named ones: a named definition in each
    # branch of the if would be two methods of the same local name.
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
    # dPE/dt is taken on the time series and only then reduced. In phase mode
    # this is necessary, since a phase bin gathers samples from different
    # periods; in boxcar mode it makes no difference, because the two commute.
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

    # --- (3) ratios, from the reduced fields --------------------------------
    # Masked where the gradient is below the floor, since −F_b/(dB/dz) is 0/0
    # inside the mixed layer. See the note on GRAD_FLOOR above.
    floor_val = GRAD_FLOOR * N²_ref
    mask = abs.(G) .>= floor_val
    K_T   = [mask[k, n] ? -F_b[k, n] / G[k, n] : NaN for k in axes(G, 1), n in axes(G, 2)]
    K_sgs = [mask[k, n] ? -F_s[k, n] / G[k, n] : NaN for k in axes(G, 1), n in axes(G, 2)]
    K_ratio = K_sgs ./ K_T

    # Mixed-layer height: where ∂⟨b⟩/∂z peaks, i.e. the middle of the sharpened
    # pycnocline rather than its foot (see mixed_layer_height.jl). The gradient
    # is the model's own ∂z(b) on Faces, not an offline difference.
    #
    # The search stops at zref, above which the sponge and the far field take
    # over and the tallest bump is noise rather than an interface.
    h = [mixed_layer_height(zf, view(G, :, n), N²_ref; zmax = zref,
                            F = view(F_b, :, n)) for n in 1:nx]
    # Whether that peak is a single interface or the tallest of several bumps,
    # counted per sample and reported in the verification block below.
    h_nup = [h_upcrossings(zf, view(G, :, n); zmax = zref,
                           F = view(F_b, :, n)) for n in 1:nx]

    # --- bulk relation ------------------------------------------------------
    #   (1/Lz)∫₀^zref F_b dz = −(1/Lz)∫₀^zref K_T dB/dz dz ≈ −K_T δb/Lz
    # This holds because dB/dz is nearly zero below h and the background is
    # uniform above, so the integral is dominated by the step at z = h. Taking
    # K_T outside the integral assumes it is roughly constant across the
    # interface, which is what delta_eff below tests.
    Fint     = [trapz_to(zf, view(F_b, :, n), zref) for n in 1:nx]
    K_T_bulk = -Fint ./ δb                              # δb > 0, F_b < 0 ⇒ K > 0

    # An independent cross-check: K_T_pe comes from B alone, K_T_bulk from wb and
    # F_sgs. They share no code, so a disagreement points at the flux, usually at
    # its subgrid half. This is the single most useful test here.
    #
    # Integrating the buoyancy budget ∂B/∂t = −∂F_b/∂z by parts leaves a boundary
    # term,
    #
    #     d/dt ∫₀^H z B dz  =  −H·F_b(H) + ∫₀^H F_b dz
    #
    # so with PE = −(1/Lz)∫₀^H zB dz,
    #
    #     K_T_bulk = Lz·(dPE/dt)/Δb  −  H·F_b(H)/Δb  =  K_T_pe + K_bdy
    #
    # The simpler form K_T_pe = Lz(dPE/dt)/Δb assumes F_b(zref) = 0, which is not
    # true here: the softplus background is stratified at every level above z = T,
    # so there is a background flux at any zref. The dropped term turned out to be
    # larger than K_T_pe itself, and restoring it brings the two routes to within
    # 2–10 % of each other.
    #
    # Both are kept. The correction touches F_b at one level, well above the mixed
    # layer, so K_T_pe_full still takes all of its structure from B.
    K_T_pe   = Lz .* dPEdt ./ δb
    F_zref   = [interp_at(zf, view(F_b, :, n), zref) for n in 1:nx]
    K_bdy    = -zref .* F_zref ./ δb
    K_T_pe_full = K_T_pe .+ K_bdy
    # How much flux still crosses zref, relative to what the layer below carries.
    # Small means zref is high enough to drop the boundary term; of order 1 means
    # the boundary term is doing the work.
    leakage  = abs.(zref .* F_zref) ./ max.(abs.(Fint), eps())

    # Replacing the integral by the peak value is only legitimate if the flux
    # profile's effective width is comparable to the depth it spreads over, so
    # measure that width rather than assuming it:
    kref = findlast(z -> z <= zref, zf)
    Fpk  = [(c = view(F_b, 1:kref, n); g = filter(isfinite, c);
             isempty(g) ? NaN : (abs(minimum(g)) >= abs(maximum(g)) ? minimum(g) : maximum(g)))
            for n in 1:nx]
    δ_eff = Fint ./ Fpk                                 # metres

    # --- panel (d) sampling -------------------------------------------------
    # K_T and TKE at z = h(t): the mixed-layer top is where the two candidate
    # scalings differ and where the flux is.
    K_at_h   = [interp_at(zf, view(K_T, :, n),  h[n]) for n in 1:nx]
    TKE_at_h = [interp_at(zc, view(TKE, :, n),  h[n]) for n in 1:nx]
    fit_keep = SMOOTH == "phase" ? trues(nx) : (times .>= SKIP_PERIODS * T_tide)
    slope, rcorr, nfit = loglog_slope(TKE_at_h[fit_keep], K_at_h[fit_keep])

    return (; tag, T, s, Ri, N²_ref, zref, cᴰ, zc, zf, xs, xlab, times,
              W_max, TKE, K_T, K_sgs, K_ratio, K_T_bulk, K_T_pe, K_T_pe_full,
              K_bdy, leakage, δ_eff, h, h_nup, PE, δb,
              ustar, Fint, Fpk, F_b, G, Bs, Us, uws, kaps, dPEdt,
              K_at_h, TKE_at_h, fit_keep, slope, rcorr, nfit)
end

# Centred difference on a possibly uneven time axis, one-sided at the ends.
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

# ---------------- Verification: the five checks, in reading order ----------
# Returns (fails, metrics). The metrics are saved into the results file so the
# sweep driver can gate stage 3 on check 2 without recomputing it.
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

    # 2. the two K_T routes. The verdict uses the exact identity K_T_pe_full,
    #    with the uncorrected K_T_pe printed beside it so the size of the dropped
    #    boundary term is visible.
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

    # 3b. Is h well defined? The peak definition assumes there is one peak. Where
    #     the gradient above the layer is noise rather than an interface, as in
    #     the passive N/ω = 0 case, the maximum jumps between bumps and h is not
    #     a length scale.
    nup = c.h_nup[c.fit_keep]
    frac_amb = isempty(nup) ? NaN : count(>(1), nup) / length(nup)
    hh_all = filter(isfinite, c.h[c.fit_keep])
    met["h_ambiguous_frac"] = frac_amb
    if H_DEF != "crossing"
        ok3b = frac_amb < 0.2
        @printf("  3b. h = %s: ambiguous in %4.0f %% of samples, h = %.2f ± %.2f m  %s\n",
                H_DEF == "flux" ? "peak of −F_b" : "peak of dB/dz", 100frac_amb, isempty(hh_all) ? NaN : mean(hh_all),
                isempty(hh_all) ? NaN : std(hh_all), ok3b ? "PASS" : "FAIL")
        ok3b || push!(fails, @sprintf("h is ambiguous in %.0f %% of samples — the gradient above the layer has several comparable bumps, so the peak is not an interface. Do not use h as a length scale for this case", 100frac_amb))
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

    # 5. the answer, with the two things that can make a slope unreadable stated
    #    before the number rather than after it.
    @printf("  5. panel (d) slope d(log K_T)/d(log TKE) at z = h = %+.2f  (r = %+.2f, n = %d)\n",
            c.slope, c.rcorr, c.nfit)

    # (a) At N = 0 the test does not exist: K_T ~ TKE/N is undefined there, so
    #     the unstratified case cannot distinguish between the two scalings
    #     however clean the fit is. It is a control, showing that the method works
    #     and giving the √TKE·l branch with nothing suppressing it.
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

    # (b) A slope without a correlation is not a measurement: a least-squares
    #     slope through a shapeless cloud is still a number, and with thousands of
    #     points its formal error bar is tiny. r² says how much of the scatter the
    #     power law actually accounts for.
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

    # (b) K_T on a log scale, since it spans decades.
    Klog = [isfinite(v) && v > 0 ? log10(v) : NaN for v in c.K_T[kf, :]]
    pb = heatmap(c.xs, c.zf[kf], Klog;
                 color = div_map, xlabel = c.xlab, ylabel = "z (m)",
                 title = @sprintf("(b) log₁₀ K_T (m² s⁻¹), masked |dB/dz| < %.2f N²_ref", GRAD_FLOOR))
    plot!(pb, c.xs, c.h; color = :black, lw = 2, label = "h")

    # (c) The cross-check, drawn on shared axes even though the two routes share
    # no code.
    #
    # The y axis is logarithmic, as in (b) and (d): K_T_bulk swings over more than
    # a decade within a tidal cycle and again between stratifications, and on a
    # linear axis everything away from the peak would be pressed onto zero.
    #
    # A log axis cannot show a sign change, and the uncorrected K_T_pe is the one
    # series that goes negative, so it is not drawn. It is still computed, saved
    # and reported as a number by check 2.
    #
    # Non-positive samples cannot be drawn either; they are dropped and counted in
    # the title.
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
    # The shape factor is drawn on a second y axis, since δ_eff is in metres and
    # shares none of the left axis's scale. Plots.jl gives a twinx series no
    # legend entry, so an all-NaN series on the main axis carries its label into
    # the main legend, and the label says "(right axis)".
    plot!(pc, c.xs, fill(NaN, length(c.xs)); color = "#7FADE0", lw = 1.2, ls = :dot,
          label = "δ_eff = ∫F_b dz / F_b|peak  (right axis, m)")
    plot!(twinx(pc), c.xs, c.δ_eff; color = "#7FADE0", lw = 1.2, ls = :dot,
          ylabel = "δ_eff (m)", label = "", legend = false)

    # (d) the answer
    x, y = c.TKE_at_h[c.fit_keep], c.K_at_h[c.fit_keep]
    m = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
    # The title carries the caveat as well as the number: a slope through a
    # shapeless cloud is still a slope, and r² says how much of the scatter the
    # power law accounts for. At N = 0 there is no TKE/N value to compare with.
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
        # The TKE/N line is left out at N = 0 rather than drawn and ignored,
        # since it refers to a scaling that has no value at this N.
        unstrat || plot!(pd, xr, y0 .* (xr ./ x0) .^ 1.0; color = "#7A3117", ls = :dot,
                         lw = 1.5, label = "1 → TKE/N")
    else
        # No annotation here: on empty log axes there is no data range to place
        # it in, and the title already carries n.
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

    # Results file, with its own "mixing_" prefix so it cannot collide with
    # anything a simulation writer produces.
    outdir = joinpath(@__DIR__, outroot, c.tag)
    mkpath(outdir)
    resfile = joinpath(outdir, "mixing_$(c.tag)$(MIX_SUFFIX).jld2")
    jldsave(resfile;
            tag = c.tag, T = c.T, n_over_omega = c.s, Ri = c.Ri, N2_ref = c.N²_ref,
            zref = c.zref, grad_floor = GRAD_FLOOR, smooth = SMOOTH,
            cD = c.cᴰ, times = c.times, x = c.xs, xlabel = c.xlab,
            z_center = c.zc, z_face = c.zf,
            TKE = c.TKE, K_T = c.K_T, K_sgs = c.K_sgs, K_sgs_over_K_T = c.K_ratio,
            K_T_bulk = c.K_T_bulk, K_T_pe = c.K_T_pe, K_T_pe_full = c.K_T_pe_full,
            K_boundary_term = c.K_bdy, zref_leakage = c.leakage, delta_eff = c.δ_eff,
            h = c.h, h_def = H_DEF, h_nup = c.h_nup,
            PE = c.PE, dPEdt = c.dPEdt, delta_b = c.δb, ustar = c.ustar,
            F_b = c.F_b, dBdz = c.G, kappa_sgs = c.kaps, uw = c.uws,
            F_integral = c.Fint, F_peak = c.Fpk,
            K_at_h = c.K_at_h, TKE_at_h = c.TKE_at_h,
            slope = c.slope, slope_r = c.rcorr, slope_n = c.nfit,
            max_abs_W = c.W_max,
            # A copy of the verification results, so the sweep driver can gate
            # stage 3 on check 2 without recomputing it.
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
