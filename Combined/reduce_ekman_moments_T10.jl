# Reduce the Ekman T = 10 m softplus runs to the same three time series the
# Stokes pipeline stores, so the two can go on one axis:
#
#     h(t)          mixed-layer height, crossing definition at 0.1 N²_ref
#     K_at_h(t)     turbulent diffusivity at z = h
#     TKE_at_h(t)   turbulent kinetic energy at z = h
#     S_at_h(t)     magnitude of the mean shear at z = h
#     eps_at_h(t)   dissipation rate at z = h, from local equilibrium
#
# from which l = K_at_h/√TKE_at_h and x = √TKE_at_h/N, exactly as in
# Stokes/3D/plot_l_vs_qN_T10.jl.
#
# ---------------- ε, and why it is an estimate ----------------
# Neither pipeline stores the dissipation rate, and it cannot be rebuilt from
# stored averages: ε is an average of a product of fluctuating gradients. What
# it CAN be got from is the TKE budget, whose other terms were stored. Dropping
# storage and transport leaves local equilibrium,
#
#     ε ≈ P + B         P = −⟨u′w′⟩ ∂U/∂z − ⟨v′w′⟩ ∂V/∂z + νₑ S²
#                       B = F_b = ⟨w′b′⟩ + F_sgs      (negative when stable)
#
# every term of which is on disk, on the faces, at the same nodes. Three things
# this assumes, all of them worth remembering when reading a Corrsin scale
# built from it:
#
#   1. Storage is dropped. ∂TKE/∂t is computable from the record and its size
#      relative to ε is reported per case in the table below — it is the direct
#      check on the assumption, and it is the one that bites in the oscillating
#      Stokes case rather than here.
#   2. Transport is dropped. ⟨w′e′⟩ and the pressure term were never stored, so
#      unlike storage this one cannot be checked. At z = h, the top of the
#      mixed layer, transport is not generally small — this is the weakest of
#      the three.
#   3. νₑ ≈ κₑ. Only the buoyancy diffusivity κₑ + κ₀ was written, so the
#      subgrid production uses it in place of the eddy viscosity. AMD computes
#      the two with the same coefficient but different numerators, so they are
#      the same order, not equal. It matters only where the subgrid share is
#      large, which is the high-N end.
#
# ---------------- Why this replaces reduce_ekman_T10.jl ----------------
# The older script read Data/Ekman/4/, the runs "Ekman 3D.jl" produced with its
# diffusivity_fields writer commented out. Two things were wrong there and both
# are fixed here, because ekmanrun.jl re-ran the column writing plane-averaged
# moments directly:
#
#   1. K_T was built from the resolved flux alone. The subgrid flux is not a
#      correction in these runs — F_sgs is comparable to, and at the strongly
#      stratified end larger than, ⟨w'b'⟩. Here
#
#          F_b = (⟨wb⟩ − ⟨w⟩⟨b⟩) + F_sgs        F_sgs = −⟨(κₑ + κ₀)·∂b/∂z⟩
#
#      and F_sgs is the average of the product, formed inside the run where κₑ
#      and ∂b/∂z are still correlated through the local strain. It cannot be
#      reconstructed offline from stored averages.
#
#   2. The horizontal average was over a single y-slice, indices = (:, 1, :).
#      The moments are true plane averages over all 100 x 100 columns, so the
#      time series here are far less noisy and no asymmetry is introduced.
#
# The moments were written on TimeInterval, never AveragedTimeInterval, so the
# raw second moments are instantaneous and Var_t of the mean does not leak into
# TKE when the Reynolds decomposition is done below.
#
# ---------------- What is still worth knowing ----------------
# h(t) is settled by the end for N/f >= 10 but is still creeping upward for
# N/f <= 5 even at 12.73 inertial periods, so those cases give a lower bound on
# l rather than a converged value. The per-case drift over the averaging window
# is reported in the table below and stored as `h_drift`.
#
# Nothing in Ekman/ is read for code — the h definition is the copy in this
# folder, and only the .jld2 data under Combined/Data is touched.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. reduce_ekman_moments_T10.jl
# ENV    WINDOW (4)        how many inertial periods at the end to average over
#        GRAD_FLOOR (0.05) mask cells with ∂b/∂z below this fraction of N²_ref
#        RATIOS            space-separated r values (default 0.5 1 2 5 10 25 50)

using Oceananigans, JLD2, Printf, Statistics, Dates

const HERE    = @__DIR__
const DATA    = joinpath(HERE, "Data", "Ekman_moments", "4")
const OUT     = joinpath(HERE, "Data", "ekman_lengthscales_T10_moments.jld2")
const f₀      = 1e-4                      # Ekman Coriolis parameter
const T_f     = 2π / f₀                   # inertial period, 62832 s
const T_STRAT = 10.0
const WINDOW  = parse(Float64, get(ENV, "WINDOW", "4"))
const RATIOS  = [parse(Float64, s) for s in
                 split(get(ENV, "RATIOS", "0.5 1 2 5 10 25 50"))]

# The Stokes reduction smooths in time with a boxcar of one twentieth of a
# forcing period before forming any ratio (SMOOTH=tide20). f₀ here equals ω
# there, so the same physical window applies unchanged.
const SMOOTH_WINDOW = T_f / 20

# Cells with ∂b/∂z below this fraction of N²_ref are masked, since K = −F/G is
# 0/0 in the well-mixed interior. Same default as the Stokes reduction.
const GRAD_FLOOR = parse(Float64, get(ENV, "GRAD_FLOOR", "0.05"))

ENV["H_DEF"] = "crossing"
ENV["H_LEVEL"] = "0.1"
include(joinpath(HERE, "mixed_layer_height.jl"))

log_lines = String[]
function log(s)
    println(s); flush(stdout); push!(log_lines, s)
end

# ---------------- helpers, matching the Stokes ones ----------------
interp_at(z, fv, z₀) = isnan(z₀) ? NaN : begin
    i = searchsortedlast(z, z₀)
    i < 1 && return fv[1]
    i >= length(z) && return fv[end]
    fv[i] + (fv[i+1] - fv[i]) * (z₀ - z[i]) / (z[i+1] - z[i])
end

function boxcar(a::AbstractVector, nh)
    nh <= 0 && return collect(float.(a))
    n = length(a); out = similar(float.(collect(a)))
    for i in 1:n
        w = @view a[max(1, i - nh):min(n, i + nh)]
        g = filter(isfinite, w)
        out[i] = isempty(g) ? NaN : mean(g)
    end
    return out
end
boxcar(A::AbstractMatrix, nh) =
    reduce(vcat, (reshape(boxcar(view(A, k, :), nh), 1, :) for k in 1:size(A, 1)))

# Linear interpolation of a centre-valued profile onto the face nodes. The grid
# is stretched — Δz runs from 0.24 m at the bottom upward — so this is written
# with the actual node positions rather than a two-point average.
centres_to_faces(fc, zc, zf) =
    [interp_at(zc, fc, clamp(zf[k], zc[1], zc[end])) for k in eachindex(zf)]

# Magnitude of the mean shear, |∂⟨u_h⟩/∂z| = √((∂U/∂z)² + (∂V/∂z)²), on the
# faces so that it sits alongside ∂b/∂z. Both components matter here: the Ekman
# spiral turns the mean flow with height, so ∂V/∂z is not small. A face k lies
# between centres k-1 and k, which is exact for a stretched grid; the two
# boundary faces have no such pair and take their neighbour's value.
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
function shear_on_faces(U, V, zc, zf)
    dU, dV = shear_components(U, V, zc, zf)
    return hypot.(dU, dV)
end
faces_to_centres(ff, zf, zc) =
    [interp_at(zf, ff, clamp(zc[k], zf[1], zf[end])) for k in eachindex(zc)]

# ---------------- one case ----------------
function reduce_case(r)
    root = joinpath(DATA, @sprintf("r=%.1f, T=%.1f", r, T_STRAT))
    file = joinpath(root, "Moments.jld2")
    isfile(file) || (log("  r=$r: no Moments.jld2 — skipped"); return nothing)

    series(v) = FieldTimeSeries(file, v; backend = OnDisk())
    S = Dict(v => series(v) for v in
             ("U", "V", "W", "B", "dBdz", "uu", "vv", "ww", "wb", "F_sgs",
              "uw", "vw", "kappa_sgs"))

    grid = S["B"].grid
    zc = Array(znodes(grid, Center()))       # 400, where U V B uu vv ww live
    zf = Array(znodes(grid, Face()))         # 401, where W dBdz wb F_sgs live
    N²_ref = (r * f₀)^2

    tv  = S["B"].times
    sel = findall(t -> t >= tv[end] - WINDOW * T_f, tv)
    nt  = length(sel)
    nt > 1 || (log("  r=$r: only $nt samples in the window — skipped"); return nothing)

    TKE_raw = Array{Float64}(undef, length(zc), nt)
    Fb_raw  = Array{Float64}(undef, length(zf), nt)
    Fr_raw  = Array{Float64}(undef, length(zf), nt)   # resolved part alone
    dBdz    = Array{Float64}(undef, length(zf), nt)
    S_raw   = Array{Float64}(undef, length(zf), nt)
    Pr_raw  = Array{Float64}(undef, length(zf), nt)   # resolved shear production
    Ps_raw  = Array{Float64}(undef, length(zf), nt)   # subgrid shear production

    col(v, n) = Float64.(Array(interior(S[v][n], 1, 1, :)))

    for (i, n) in enumerate(sel)
        U, V, B = col("U", n), col("V", n), col("B", n)
        uu, vv, ww = col("uu", n), col("vv", n), col("ww", n)
        W = col("W", n)

        # Reynolds decomposition of the plane averages. ⟨ww⟩ was written at
        # centres so it lands on the same grid as ⟨uu⟩ and ⟨vv⟩; ⟨w⟩ is on
        # faces and is interpolated down. It is ~1e-20 m/s here — the check
        # stage reports it — but subtracting it costs nothing and keeps the
        # decomposition exact rather than nearly exact.
        Wc = faces_to_centres(W, zf, zc)
        TKE_raw[:, i] = 0.5 .* ((uu .- U.^2) .+ (vv .- V.^2) .+ (ww .- Wc.^2))

        # Buoyancy flux at the faces, where ∂b/∂z already lives.
        Bf = centres_to_faces(B, zc, zf)
        res = col("wb", n) .- W .* Bf
        Fr_raw[:, i] = res
        Fb_raw[:, i] = res .+ col("F_sgs", n)
        dBdz[:, i]   = col("dBdz", n)

        # Shear production, on the faces where ⟨uw⟩ and ⟨vw⟩ already live.
        # P = −⟨u′w′⟩ ∂U/∂z − ⟨v′w′⟩ ∂V/∂z, with the Reynolds decomposition
        # ⟨u′w′⟩ = ⟨uw⟩ − ⟨u⟩⟨w⟩ done the same way it is for the buoyancy flux.
        # ⟨w⟩ is ~1e-20 m/s so the correction is nominal, but it keeps the
        # decomposition exact rather than nearly exact.
        dU, dV = shear_components(U, V, zc, zf)
        S_raw[:, i] = hypot.(dU, dV)
        Uf = centres_to_faces(U, zc, zf); Vf = centres_to_faces(V, zc, zf)
        Pr_raw[:, i] = .-(col("uw", n) .- Uf .* W) .* dU .-
                        (col("vw", n) .- Vf .* W) .* dV
        # The subgrid part of the same term, νₑ S². Only the buoyancy
        # diffusivity κₑ + κ₀ was stored, so this takes νₑ ≈ κₑ — see the
        # header. It matters where the subgrid share is large, i.e. at high N.
        Ps_raw[:, i] = col("kappa_sgs", n) .* (dU .^ 2 .+ dV .^ 2)
    end

    # --- reduce in time, then form the ratio (never the other way round) ---
    dt = tv[sel[2]] - tv[sel[1]]
    nh = max(0, round(Int, SMOOTH_WINDOW / (2dt)))
    TKE = boxcar(TKE_raw, nh)
    F_b = boxcar(Fb_raw, nh)
    F_r = boxcar(Fr_raw, nh)
    G   = boxcar(dBdz, nh)
    S   = boxcar(S_raw, nh)
    P_r = boxcar(Pr_raw, nh)
    P_s = boxcar(Ps_raw, nh)
    # Local-equilibrium dissipation: ε = P + B, with B = F_b the buoyancy flux
    # (negative under stable stratification, so it is a sink). Storage and
    # transport are dropped; how big the storage term is, is reported below.
    EPS = P_r .+ P_s .+ F_b
    # ∂TKE/∂t at fixed z, central in time, so that the storage term can be both
    # reported and optionally kept. Taking it on the profile and interpolating
    # afterwards keeps it a pure tendency; differencing TKE_at_h(t) instead
    # would fold in the motion of h itself.
    dEdt = fill(NaN, size(TKE))
    tsel = tv[sel]
    @inbounds for k in axes(TKE, 1), n in 2:nt-1
        dEdt[k, n] = (TKE[k, n+1] - TKE[k, n-1]) / (tsel[n+1] - tsel[n-1])
    end

    floorv = GRAD_FLOOR * N²_ref
    mask(F) = [G[k, n] > floorv ? -F[k, n] / G[k, n] : NaN
               for k in axes(G, 1), n in axes(G, 2)]
    K_T = mask(F_b)
    K_r = mask(F_r)

    h        = [mixed_layer_height(zf, view(G, :, n), N²_ref) for n in 1:nt]
    K_at_h   = [interp_at(zf, view(K_T, :, n), h[n]) for n in 1:nt]
    K_res_h  = [interp_at(zf, view(K_r, :, n), h[n]) for n in 1:nt]
    TKE_at_h = [interp_at(zc, view(TKE, :, n), h[n]) for n in 1:nt]
    S_at_h   = [interp_at(zf, view(S, :, n), h[n]) for n in 1:nt]
    P_at_h   = [interp_at(zf, view(P_r, :, n), h[n]) + interp_at(zf, view(P_s, :, n), h[n])
                for n in 1:nt]
    Pr_at_h  = [interp_at(zf, view(P_r, :, n), h[n]) for n in 1:nt]
    eps_at_h = [interp_at(zf, view(EPS, :, n), h[n]) for n in 1:nt]
    Fb_at_h  = [interp_at(zf, view(F_b, :, n), h[n]) for n in 1:nt]
    dEdt_at_h = [interp_at(zc, view(dEdt, :, n), h[n]) for n in 1:nt]

    times = tv[sel]
    fin(v) = (w = filter(isfinite, v); isempty(w) ? NaN : median(w))
    # Drift of h across the averaging window, as a fraction of its median: the
    # cases that have not equilibrated are the ones this is large for.
    q1 = fin(view(h, 1:max(1, nt ÷ 4)))
    q4 = fin(view(h, (nt - max(1, nt ÷ 4) + 1):nt))
    drift = (q4 - q1) / fin(h)
    # How much of K_T the subgrid flux carries, at z = h.
    sgs_share = fin([isfinite(K_at_h[i]) && K_at_h[i] != 0 ?
                     1 - K_res_h[i] / K_at_h[i] : NaN for i in 1:nt])

    # Two checks on the local-equilibrium assumption, both reported per case.
    # `store` is |∂TKE/∂t|/ε — how much of the budget the dropped storage term
    # is. `cancel` is ε/(P + |B|) — how much of a residual ε is between two
    # larger terms; when that is small, ε is the difference of two nearly equal
    # numbers and the estimate is fragile whatever the storage does.
    store  = fin([abs(dEdt_at_h[n]) / abs(eps_at_h[n]) for n in 1:nt
                  if isfinite(dEdt_at_h[n]) && isfinite(eps_at_h[n]) && eps_at_h[n] != 0])
    cancel = fin([eps_at_h[n] / (abs(P_at_h[n]) + abs(Fb_at_h[n])) for n in 1:nt
                  if isfinite(eps_at_h[n]) && isfinite(P_at_h[n]) && isfinite(Fb_at_h[n])])
    eps_neg = count(x -> isfinite(x) && x <= 0, eps_at_h)
    # Where in the layer does local equilibrium actually hold? The fraction of
    # samples with ε > 0, at fractions of h. It is a diagnostic only — nothing
    # downstream evaluates anywhere but z = h — but it says whether a failure
    # at z = h is the method or the height.
    zscan = [count(n -> (e = interp_at(zf, view(EPS, :, n), a * h[n]);
                         isfinite(e) && e > 0), 1:nt) / nt
             for a in (0.25, 0.5, 0.75, 1.0)]

    log(@sprintf("  r=%-5.1f N=%.2e  nt=%4d  med h=%6.3f m  drift %+5.1f %%  med K_at_h=%.3e  sgs share %5.2f  med TKE_at_h=%.3e  med S_at_h=%.3e  med P=%.3e  med eps=%.3e  eps/(P+|B|)=%5.2f  |dTKE/dt|/eps=%6.2f  eps<=0: %3d  eps>0 at (0.25,0.5,0.75,1)h: %.2f %.2f %.2f %.2f  finite l: %d/%d",
                 r, r * f₀, nt, fin(h), 100drift, fin(K_at_h), sgs_share, fin(TKE_at_h),
                 fin(S_at_h), fin(P_at_h), fin(eps_at_h), cancel, store, eps_neg, zscan...,
                 count(i -> isfinite(K_at_h[i]) && isfinite(TKE_at_h[i]) &&
                            TKE_at_h[i] > 0 && K_at_h[i] > 0, 1:nt), nt))
    return (r = r, N = r * f₀, N2_ref = N²_ref, times = times, h = h,
            K_at_h = K_at_h, K_res_at_h = K_res_h, TKE_at_h = TKE_at_h,
            S_at_h = S_at_h, P_at_h = P_at_h, P_res_at_h = Pr_at_h,
            eps_at_h = eps_at_h, Fb_at_h = Fb_at_h, dEdt_at_h = dEdt_at_h,
            eps_store = store, eps_cancel = cancel, eps_neg = eps_neg,
            eps_zscan = zscan,
            h_drift = drift, sgs_share = sgs_share)
end

# ---------------- run ----------------
log("Ekman T = 10 m reduction from the plane-averaged moments — " *
    "h = crossing at $(H_LEVEL) N²_ref, K_T from resolved + subgrid flux,")
log("window = $(WINDOW) inertial periods at the end, grad floor = $(GRAD_FLOOR) N²_ref")
log("started $(Dates.now())")

cases = filter(!isnothing, [reduce_case(r) for r in RATIOS])
isempty(cases) && error("no Ekman T = 10 moment files under $DATA")

jldopen(OUT, "w") do io
    io["ratios"] = [c.r for c in cases]
    io["h_def"] = "crossing"; io["h_level"] = H_LEVEL
    io["grad_floor"] = GRAD_FLOOR; io["window_periods"] = WINDOW
    io["f0"] = f₀; io["T_strat"] = T_STRAT
    io["flux"] = "resolved + subgrid — F_b = (<wb> - <w><b>) + F_sgs"
    io["source"] = "Data/Ekman_moments/4, written by ekmanrun.jl"
    for c in cases
        g = @sprintf("r=%.1f", c.r)
        io["$g/N"] = c.N; io["$g/N2_ref"] = c.N2_ref
        io["$g/times"] = c.times; io["$g/h"] = c.h
        io["$g/K_at_h"] = c.K_at_h; io["$g/K_res_at_h"] = c.K_res_at_h
        io["$g/TKE_at_h"] = c.TKE_at_h; io["$g/S_at_h"] = c.S_at_h
        io["$g/P_at_h"] = c.P_at_h; io["$g/P_res_at_h"] = c.P_res_at_h
        io["$g/eps_at_h"] = c.eps_at_h; io["$g/Fb_at_h"] = c.Fb_at_h
        io["$g/dEdt_at_h"] = c.dEdt_at_h; io["$g/eps_cancel"] = c.eps_cancel
        io["$g/eps_zscan"] = c.eps_zscan
        io["$g/eps_store"] = c.eps_store; io["$g/eps_neg"] = c.eps_neg
        io["$g/h_drift"] = c.h_drift; io["$g/sgs_share"] = c.sgs_share
    end
end
log("wrote $OUT")
log("finished $(Dates.now())")

mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "reduce_ekman_moments_T10.log"), "w") do io
    foreach(l -> println(io, l), log_lines)
end
