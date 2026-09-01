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
# ---------------- What this can and cannot reproduce ----------------
# "Ekman 3D.jl" writes no subgrid buoyancy flux — its diffusivity_fields writer
# is commented out — so K_T here is built from the RESOLVED flux alone:
#
#     K_T = −⟨w'b'⟩ / ⟨∂b/∂z⟩          (Stokes uses F_b = ⟨w'b'⟩ + F_sgs)
#
# On the Stokes side the subgrid share at z = h runs from 0.03 at N/ω = 1 to
# 0.59 at N/ω = 50, so this omission is small at weak stratification and a
# factor-of-two underestimate at the strong end. The companion plotting script
# therefore also forms the resolved-only Stokes K_T, and that is the curve the
# Ekman points should be read against.
#
# Second limitation: Velocity.jld2 and Buoyancy.jld2 are written with
# indices = (:, 1, :), a single y-slice, so ⟨·⟩ here is an average over 100 x
# points rather than the full plane. The means subtracted are the true
# horizontal averages from Avg_vel.jld2 and Avg_b.jld2, so the fluctuations are
# unbiased, but they are noisier than the Stokes equivalents.
#
# Nothing in Ekman/ is read for code — the h definition is the copy in this
# folder, and only the .jld2 data under Combined/Data is touched.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. reduce_ekman_T10.jl
# ENV    T_STEP (1)        keep every T_STEP-th velocity snapshot
#        WINDOW (4)        how many inertial periods at the end to keep
#        RATIOS            space-separated r values (default 0.5 1 2 5 10 25 50)

using Oceananigans, JLD2, Printf, Statistics, Dates

const HERE    = @__DIR__
const DATA    = joinpath(HERE, "Data", "Ekman", "4")
const OUT     = joinpath(HERE, "Data", "ekman_lengthscales_T10.jld2")
const f₀      = 1e-4                      # Ekman Coriolis parameter
const T_f     = 2π / f₀                   # inertial period, 62832 s
const T_STRAT = 10.0
const T_STEP  = parse(Int,     get(ENV, "T_STEP", "1"))
const WINDOW  = parse(Float64, get(ENV, "WINDOW", "4"))
const RATIOS  = [parse(Float64, s) for s in
                 split(get(ENV, "RATIOS", "0.5 1 2 5 10 25 50"))]

# The Stokes reduction smooths in time with a boxcar of one twentieth of a
# forcing period before forming any ratio (SMOOTH=tide20). f₀ here equals ω
# there, so the same physical window applies unchanged.
const SMOOTH_WINDOW = T_f / 20

# Cells with |∂b/∂z| below this fraction of N²_ref are masked, since K = −F/G is
# 0/0 in the well-mixed interior. Same default as MixedLayerDiffusivity.jl.
const GRAD_FLOOR = parse(Float64, get(ENV, "GRAD_FLOOR", "0.05"))

# The crossing definition of h, taken from Stokes/3D/mixed_layer_height.jl —
# a copy lives in this folder so nothing outside Combined/ is needed.
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

# Linear interpolation of a centre-valued profile onto the face nodes, so that
# w'b' and ∂b/∂z live on the same grid, as they do in the Stokes reduction.
function centres_to_faces(fc, zc, zf)
    out = similar(fc, length(zf))
    @inbounds for k in eachindex(zf)
        out[k] = interp_at(zc, fc, clamp(zf[k], zc[1], zc[end]))
    end
    return out
end

# ---------------- one case ----------------
function reduce_case(r)
    root = joinpath(DATA, @sprintf("r=%.1f, T=%.1f", r, T_STRAT))
    for f in ("Velocity.jld2", "Buoyancy.jld2", "Avg_vel.jld2", "Avg_b.jld2",
              "Avg_grad_b.jld2")
        isfile(joinpath(root, f)) || (log("  r=$r: missing $f — skipped"); return nothing)
    end

    us = FieldTimeSeries(joinpath(root, "Velocity.jld2"), "u"; backend = OnDisk())
    vs = FieldTimeSeries(joinpath(root, "Velocity.jld2"), "v"; backend = OnDisk())
    ws = FieldTimeSeries(joinpath(root, "Velocity.jld2"), "w"; backend = OnDisk())
    bs = FieldTimeSeries(joinpath(root, "Buoyancy.jld2"), "b"; backend = OnDisk())
    ua = FieldTimeSeries(joinpath(root, "Avg_vel.jld2"), "u_avg"; backend = OnDisk())
    va = FieldTimeSeries(joinpath(root, "Avg_vel.jld2"), "v_avg"; backend = OnDisk())
    wa = FieldTimeSeries(joinpath(root, "Avg_vel.jld2"), "w_avg"; backend = OnDisk())
    ba = FieldTimeSeries(joinpath(root, "Avg_b.jld2"), "b"; backend = OnDisk())
    gb = FieldTimeSeries(joinpath(root, "Avg_grad_b.jld2"), "db_dz"; backend = OnDisk())

    grid = us.grid
    zc = Array(znodes(grid, Center()))          # 500
    zf = Array(znodes(grid, Face()))            # 501
    N²_ref = (r * f₀)^2

    # The velocity/buoyancy slices are on a 200 s cadence and the gradient on a
    # 100 s one, so the gradient is matched by time value, not by index.
    tv = us.times
    t_end = tv[end]
    sel = findall(t -> t >= t_end - WINDOW * T_f, tv)[1:T_STEP:end]
    gi  = [argmin(abs.(gb.times .- tv[n])) for n in sel]

    nt = length(sel)
    TKE_raw = Array{Float64}(undef, length(zc), nt)
    wb_raw  = Array{Float64}(undef, length(zf), nt)
    dBdz    = Array{Float64}(undef, length(zf), nt)

    for (i, n) in enumerate(sel)
        u = Float64.(Array(interior(us[n], :, :, :)))          # (100,1,500)
        v = Float64.(Array(interior(vs[n], :, :, :)))
        w = Float64.(Array(interior(ws[n], :, :, :)))          # (100,1,501) on faces
        b = Float64.(Array(interior(bs[n], :, :, :)))          # (100,1,500)

        um = Float64.(Array(interior(ua[n], 1, 1, :)))
        vm = Float64.(Array(interior(va[n], 1, 1, :)))
        wm = Float64.(Array(interior(wa[n], 1, 1, :)))
        bm = Float64.(Array(interior(ba[n], 1, 1, :)))

        # --- TKE, at centres: w is dropped onto centres first, as in TKE.jl ---
        wcen  = 0.5 .* (w[:, :, 1:end-1] .+ w[:, :, 2:end])
        wmcen = 0.5 .* (wm[1:end-1] .+ wm[2:end])
        up = u .- reshape(um, 1, 1, :)
        vp = v .- reshape(vm, 1, 1, :)
        wp = wcen .- reshape(wmcen, 1, 1, :)
        TKE_raw[:, i] = vec(mean(0.5 .* (up.^2 .+ vp.^2 .+ wp.^2), dims = (1, 2)))

        # --- resolved buoyancy flux, at faces alongside ∂b/∂z ---
        bf  = mapslices(c -> centres_to_faces(c, zc, zf), b; dims = 3)
        bmf = centres_to_faces(bm, zc, zf)
        wpf = w .- reshape(wm, 1, 1, :)
        bpf = bf .- reshape(bmf, 1, 1, :)
        wb_raw[:, i] = vec(mean(wpf .* bpf, dims = (1, 2)))

        dBdz[:, i] = Float64.(Array(interior(gb[gi[i]], 1, 1, :)))
    end

    # --- reduce in time, then form the ratio (never the other way round) ---
    dt = nt > 1 ? (tv[sel[2]] - tv[sel[1]]) : 1.0
    nh = max(0, round(Int, SMOOTH_WINDOW / (2dt)))
    TKE = boxcar(TKE_raw, nh)
    F_b = boxcar(wb_raw, nh)
    G   = boxcar(dBdz, nh)

    floorv = GRAD_FLOOR * N²_ref
    K_T = [G[k, n] > floorv ? -F_b[k, n] / G[k, n] : NaN
           for k in axes(G, 1), n in axes(G, 2)]

    h        = [mixed_layer_height(zf, view(G, :, n), N²_ref) for n in 1:nt]
    K_at_h   = [interp_at(zf, view(K_T, :, n), h[n]) for n in 1:nt]
    TKE_at_h = [interp_at(zc, view(TKE, :, n), h[n]) for n in 1:nt]

    times = tv[sel]
    fin(v) = (w = filter(isfinite, v); isempty(w) ? NaN : median(w))
    log(@sprintf("  r=%-5.1f N=%.2e  nt=%4d  med h=%6.3f m  med K_at_h=%.3e  med TKE_at_h=%.3e  finite l: %d/%d",
                 r, r * f₀, nt, fin(h), fin(K_at_h), fin(TKE_at_h),
                 count(i -> isfinite(K_at_h[i]) && isfinite(TKE_at_h[i]) &&
                            TKE_at_h[i] > 0 && K_at_h[i] > 0, 1:nt), nt))
    return (r = r, N = r * f₀, N2_ref = N²_ref, times = times,
            h = h, K_at_h = K_at_h, TKE_at_h = TKE_at_h)
end

# ---------------- run ----------------
log("Ekman T = 10 m reduction — h = crossing at $(H_LEVEL) N²_ref, " *
    "resolved flux only, window = $(WINDOW) inertial periods, T_STEP = $T_STEP")
log("started $(Dates.now())")

cases = filter(!isnothing, [reduce_case(r) for r in RATIOS])
isempty(cases) && error("no Ekman T = 10 cases reduced")

jldopen(OUT, "w") do io
    io["ratios"] = [c.r for c in cases]
    io["h_def"] = "crossing"; io["h_level"] = H_LEVEL
    io["grad_floor"] = GRAD_FLOOR; io["window_periods"] = WINDOW
    io["f0"] = f₀; io["T_strat"] = T_STRAT
    io["flux"] = "resolved only — Ekman 3D.jl saves no subgrid buoyancy flux"
    for c in cases
        g = @sprintf("r=%.1f", c.r)
        io["$g/N"] = c.N; io["$g/N2_ref"] = c.N2_ref
        io["$g/times"] = c.times; io["$g/h"] = c.h
        io["$g/K_at_h"] = c.K_at_h; io["$g/TKE_at_h"] = c.TKE_at_h
    end
end
log("wrote $OUT")
log("finished $(Dates.now())")

mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "reduce_ekman_T10.log"), "w") do io
    foreach(l -> println(io, l), log_lines)
end
