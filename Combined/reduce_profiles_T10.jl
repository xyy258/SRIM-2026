# Profiles and the friction velocity, for both flows, T = 10 m.
#
# The three stages of the delta model all need things the earlier reductions
# never stored — a TKE profile rather than TKE(z=h), and a stress profile to
# get u_* from. This walks both columns once and caches them, so the three
# plotting scripts are seconds each.
#
# ---------------- What u_* is here ----------------
# Both runs drive the bottom with quadratic bulk drag, so the honest friction
# velocity is the total kinematic momentum flux at the wall. Neither pipeline
# stored the wall stress, but both stored the two pieces that make it:
#
#     tau_x = (<uw> - <u><w>) - nu_e dU/dz          resolved + subgrid
#     tau_y = (<vw> - <v><w>) - nu_e dV/dz
#     u_*^2 = max over 0 < z <= h of |tau|
#
# taking nu_e ~ kappa_sgs as in plot_l_vs_corrsin_T10.jl, since only the
# buoyancy diffusivity was written. The maximum is taken below z = h because
# the stress falls off above the mixed layer, and it is taken per sample and
# then reduced, never the other way round.
#
# Two other estimates are logged for comparison and NOT used:
#   sqrt(c_D)*U_inf     the free-stream value, what the textbook formulae mean
#   sqrt(c_D*|U_1|^2)   the drag law on the mean velocity at the first centre,
#                       which underestimates because it drops the gustiness
#
# For the Stokes column u_* swings through the tidal cycle, so both the median
# and the 90th percentile are stored; the plain tidal thickness 0.4 u_*/omega
# is conventionally written with the peak. Which one is used only moves delta
# by a constant per flow, which B1 absorbs in stage 3 — but it matters for
# reading delta/h against the published constants.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. reduce_profiles_T10.jl
#        ~6 min. Writes Data/profiles_T10.jld2.

using Oceananigans, JLD2, Printf, Statistics

const HERE   = @__DIR__
const STOKES = "/home/tll46/SRIM-2026/Stokes/3D"
const EKDATA = joinpath(HERE, "Data", "Ekman_moments", "4")
const EKRED  = joinpath(HERE, "Data", "ekman_lengthscales_T10_moments.jld2")
const OUT    = joinpath(HERE, "Data", "profiles_T10.jld2")
const ω      = 1e-4                    # = f₀; both flows share it
const T_tide = 2π / ω
const T_f    = 2π / ω
const SKIP   = 3                       # Stokes spin-up, tidal periods
const WINDOW = 4                       # Ekman window, inertial periods
const SVALS  = [1, 2, 5, 10, 25, 50]
const RATIOS = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
const U∞     = 0.04
const κ_vk   = 0.41
const z₀     = 0.0016

fin(v) = filter(isfinite, v)
med(v) = (w = fin(v); isempty(w) ? NaN : median(w))

interp_at(z, fv, z₀) = isnan(z₀) ? NaN : begin
    i = searchsortedlast(z, z₀)
    i < 1 && return fv[1]
    i >= length(z) && return fv[end]
    fv[i] + (fv[i+1] - fv[i]) * (z₀ - z[i]) / (z[i+1] - z[i])
end
centres_to_faces(fc, zc, zf) =
    [interp_at(zc, fc, clamp(zf[k], zc[1], zc[end])) for k in eachindex(zf)]
faces_to_centres(ff, zf, zc) =
    [interp_at(zf, ff, clamp(zc[k], zf[1], zf[end])) for k in eachindex(zc)]

function boxcar(A::AbstractMatrix, nh)
    nh <= 0 && return float.(A)
    out = similar(float.(A))
    for k in axes(A, 1), i in axes(A, 2)
        w = @view A[k, max(1, i - nh):min(size(A, 2), i + nh)]
        g = fin(w)
        out[k, i] = isempty(g) ? NaN : mean(g)
    end
    return out
end

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

# One case, given open series and the sample selection. Returns the time-median
# profiles and the u_* series.
function walk(F, sel, hsel, zc, zf, nh)
    col(v, n) = Float64.(Array(interior(F[v][n], 1, 1, :)))
    nt = length(sel)
    TKE = Array{Float64}(undef, length(zc), nt)
    TAU = Array{Float64}(undef, length(zf), nt)
    U1  = Array{Float64}(undef, nt)
    for (i, n) in enumerate(sel)
        U, V, W = col("U", n), col("V", n), col("W", n)
        uu, vv, ww = col("uu", n), col("vv", n), col("ww", n)
        Wc = faces_to_centres(W, zf, zc)
        TKE[:, i] = 0.5 .* ((uu .- U .^ 2) .+ (vv .- V .^ 2) .+ (ww .- Wc .^ 2))
        dU, dV = shear_components(U, V, zc, zf)
        Uf = centres_to_faces(U, zc, zf); Vf = centres_to_faces(V, zc, zf)
        ks = col("kappa_sgs", n)
        τx = (col("uw", n) .- Uf .* W) .- ks .* dU
        τy = (col("vw", n) .- Vf .* W) .- ks .* dV
        TAU[:, i] = hypot.(τx, τy)
        U1[i] = hypot(U[1], V[1])
    end
    TKE = boxcar(TKE, nh); TAU = boxcar(TAU, nh)
    # u_*^2 = the largest total stress below the mixed-layer top, per sample.
    us = Array{Float64}(undef, nt)
    for i in 1:nt
        hi = hsel[i]
        k = isnan(hi) ? length(zf) : max(2, searchsortedlast(zf, hi))
        us[i] = sqrt(maximum(fin(view(TAU, 1:k, i))))
    end
    return (TKE = [med(view(TKE, k, :)) for k in axes(TKE, 1)],
            TAU = [med(view(TAU, k, :)) for k in axes(TAU, 1)],
            us = us, U1 = med(U1), nt = nt)
end

cases = []

# ---------------- Stokes ----------------
cD_S = (κ_vk / log(0.0667 / z₀))^2      # z_drag_ref from Stokes/3D/case_params.jl
for s in SVALS
    tag = "P4_T10_sqrtRi$s"
    mix = joinpath(STOKES, "outputs", tag, "mixing_$(tag)_hcross.jld2")
    mom = joinpath(STOKES, "outputs", tag, "TidalBL3D_$(tag)_moments.jld2")
    (isfile(mix) && isfile(mom)) || (say("missing files for $tag — skipped"); continue)
    d = jldopen(mix, "r") do io; (t = io["times"], h = io["h"]) end
    F = Dict(v => FieldTimeSeries(mom, v; backend = OnDisk()) for v in
             ("U", "V", "W", "uu", "vv", "ww", "uw", "vw", "kappa_sgs"))
    grid = F["U"].grid
    zc = Array(znodes(grid, Center())); zf = Array(znodes(grid, Face()))
    dt = d.t[2] - d.t[1]; nh = max(0, round(Int, (T_tide / 20) / (2dt)))
    k0 = findfirst(t -> t >= SKIP * T_tide, d.t)
    sel = max(1, k0 - nh):length(d.t)
    w = walk(F, sel, d.h[sel], zc, zf, nh)
    keep = d.t[sel] .>= SKIP * T_tide
    push!(cases, (flow = "stokes", r = float(s), N = s * ω, zc = zc, zf = zf,
                  TKE = w.TKE, TAU = w.TAU, h = med(d.h[sel][keep]),
                  us = med(w.us[keep]), us_p90 = quantile(fin(w.us[keep]), 0.9),
                  us_free = sqrt(cD_S) * U∞, us_drag = sqrt(cD_S) * w.U1, cD = cD_S))
    say(@sprintf("Stokes N/ω = %-4g  h = %6.3f m  u_* med %.4e  p90 %.4e  (free stream %.4e, drag law %.4e)",
                 s, cases[end].h, cases[end].us, cases[end].us_p90,
                 cases[end].us_free, cases[end].us_drag))
end

# ---------------- Ekman ----------------
hE = Dict{Float64,Vector{Float64}}()
jldopen(EKRED, "r") do io
    for r in io["ratios"]; hE[r] = io[@sprintf("r=%.1f/h", r)] end
end
for r in RATIOS
    file = joinpath(EKDATA, @sprintf("r=%.1f, T=10.0", r), "Moments.jld2")
    isfile(file) || (say("missing Moments.jld2 for r=$r — skipped"); continue)
    F = Dict(v => FieldTimeSeries(file, v; backend = OnDisk()) for v in
             ("U", "V", "W", "uu", "vv", "ww", "uw", "vw", "kappa_sgs"))
    grid = F["U"].grid
    zc = Array(znodes(grid, Center())); zf = Array(znodes(grid, Face()))
    tv = F["U"].times
    sel = findall(t -> t >= tv[end] - WINDOW * T_f, tv)
    length(sel) == length(hE[r]) ||
        error("r=$r: $(length(sel)) samples but the reduction stored $(length(hE[r])) heights")
    dt = tv[sel[2]] - tv[sel[1]]; nh = max(0, round(Int, (T_f / 20) / (2dt)))
    cD_E = (κ_vk / log(zc[1] / z₀))^2
    w = walk(F, sel, hE[r], zc, zf, nh)
    push!(cases, (flow = "ekman", r = r, N = r * ω, zc = zc, zf = zf,
                  TKE = w.TKE, TAU = w.TAU, h = med(hE[r]),
                  us = med(w.us), us_p90 = quantile(fin(w.us), 0.9),
                  us_free = sqrt(cD_E) * U∞, us_drag = sqrt(cD_E) * w.U1, cD = cD_E))
    say(@sprintf("Ekman  N/f = %-4g  h = %6.3f m  u_* med %.4e  p90 %.4e  (free stream %.4e, drag law %.4e)",
                 r, cases[end].h, cases[end].us, cases[end].us_p90,
                 cases[end].us_free, cases[end].us_drag))
end

isempty(cases) && error("no cases found")
jldopen(OUT, "w") do io
    io["note"] = "time-median TKE and stress profiles, h and u_*; by reduce_profiles_T10.jl"
    io["flows"] = unique([c.flow for c in cases])
    for fl in unique([c.flow for c in cases])
        io["$fl/ratios"] = [c.r for c in cases if c.flow == fl]
    end
    for c in cases
        g = @sprintf("%s/r=%.1f", c.flow, c.r)
        for k in (:N, :zc, :zf, :TKE, :TAU, :h, :us, :us_p90, :us_free, :us_drag, :cD)
            io["$g/$k"] = getfield(c, k)
        end
    end
end
say("wrote $OUT")
mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "reduce_profiles_T10.log"), "w") do io
    foreach(l -> println(io, l), logl)
end
