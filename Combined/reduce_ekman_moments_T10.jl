# Reduce the Ekman T = 10 m softplus runs to the same three time series the
# Stokes pipeline stores, so the two can go on one axis:
#
#     h(t)          mixed-layer height, crossing definition at 0.1 N²_ref
#     K_at_h(t)     turbulent diffusivity at z = h
#     TKE_at_h(t)   turbulent kinetic energy at z = h
#
# from which l = K_at_h/√TKE_at_h and x = √TKE_at_h/N, exactly as in
# Stokes/3D/plot_l_vs_qN_T10.jl.
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
faces_to_centres(ff, zf, zc) =
    [interp_at(zf, ff, clamp(zc[k], zf[1], zf[end])) for k in eachindex(zc)]

# ---------------- one case ----------------
function reduce_case(r)
    root = joinpath(DATA, @sprintf("r=%.1f, T=%.1f", r, T_STRAT))
    file = joinpath(root, "Moments.jld2")
    isfile(file) || (log("  r=$r: no Moments.jld2 — skipped"); return nothing)

    series(v) = FieldTimeSeries(file, v; backend = OnDisk())
    S = Dict(v => series(v) for v in
             ("U", "V", "W", "B", "dBdz", "uu", "vv", "ww", "wb", "F_sgs"))

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
    end

    # --- reduce in time, then form the ratio (never the other way round) ---
    dt = tv[sel[2]] - tv[sel[1]]
    nh = max(0, round(Int, SMOOTH_WINDOW / (2dt)))
    TKE = boxcar(TKE_raw, nh)
    F_b = boxcar(Fb_raw, nh)
    F_r = boxcar(Fr_raw, nh)
    G   = boxcar(dBdz, nh)

    floorv = GRAD_FLOOR * N²_ref
    mask(F) = [G[k, n] > floorv ? -F[k, n] / G[k, n] : NaN
               for k in axes(G, 1), n in axes(G, 2)]
    K_T = mask(F_b)
    K_r = mask(F_r)

    h        = [mixed_layer_height(zf, view(G, :, n), N²_ref) for n in 1:nt]
    K_at_h   = [interp_at(zf, view(K_T, :, n), h[n]) for n in 1:nt]
    K_res_h  = [interp_at(zf, view(K_r, :, n), h[n]) for n in 1:nt]
    TKE_at_h = [interp_at(zc, view(TKE, :, n), h[n]) for n in 1:nt]

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

    log(@sprintf("  r=%-5.1f N=%.2e  nt=%4d  med h=%6.3f m  drift %+5.1f %%  med K_at_h=%.3e  sgs share %5.2f  med TKE_at_h=%.3e  finite l: %d/%d",
                 r, r * f₀, nt, fin(h), 100drift, fin(K_at_h), sgs_share, fin(TKE_at_h),
                 count(i -> isfinite(K_at_h[i]) && isfinite(TKE_at_h[i]) &&
                            TKE_at_h[i] > 0 && K_at_h[i] > 0, 1:nt), nt))
    return (r = r, N = r * f₀, N2_ref = N²_ref, times = times, h = h,
            K_at_h = K_at_h, K_res_at_h = K_res_h, TKE_at_h = TKE_at_h,
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
        io["$g/TKE_at_h"] = c.TKE_at_h
        io["$g/h_drift"] = c.h_drift; io["$g/sgs_share"] = c.sgs_share
    end
end
log("wrote $OUT")
log("finished $(Dates.now())")

mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "reduce_ekman_moments_T10.log"), "w") do io
    foreach(l -> println(io, l), log_lines)
end
