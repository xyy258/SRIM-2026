#!/usr/bin/env julia
# =============================================================================
# panelD_legacy.jl — panel (d) for the L_strat-as-a-fraction-of-Lz sweep.
#
# Lives here rather than in Stokes/3D because it only understands THIS folder's
# on-disk format, which differs from the current Moments.jl output in four ways:
#
#   1. F_sgs IS NOT SAVED. F_b = ⟨w′b′⟩ + F_sgs, and only the resolved half is on
#      disk, so every K_T here is RESOLVED-FLUX-ONLY and biased low. In the
#      current runs the SGS share was 3-10 % median and ROSE with stratification;
#      these cases sit at N/ω = 22 and 50, far beyond that calibration, so the
#      bias is expected to be larger and cannot be bounded from these files —
#      kappa_sgs is the variable that would bound it, and it is also absent.
#      If the SGS share grows as the resolved turbulence weakens, K_T is lifted
#      more at low TKE than at high, the true curve is flatter, and the slopes
#      printed here are an UPPER BOUND. Every title says so.
#   2. dBdz is not saved either, so it is differenced offline from B on Centers.
#      The current code uses the model's own ∂z(b) on Faces — same definition,
#      better discretisation.
#   3. ww = Average(w^2) with w on Faces, so it lands on FACES (Nz+1), whereas
#      Moments.jl interpolates to Centers with @at before averaging. It must be
#      interpolated here or TKE is wrong by half a cell. (Tidal3Dprofiles.jl in
#      this folder truncates with [1:length(zc)] instead — that is not the same.)
#   4. W is not saved, so check 1 (max|⟨w⟩|) cannot run. ⟨w⟩ = 0 follows from the
#      rigid lid plus incompressibility and the current runs confirm it at 1e-19.
#
# The moments themselves are sound: Tidal3D.jl writes them on a plain
# TimeInterval(T_tide/200), NOT an AveragedTimeInterval, so no mean-flow variance
# leaks into TKE. That is the one thing that would have made these unusable.
#
# USAGE   GKSwstype=100 julia --project=../  panelD_legacy.jl
# ENV     SKIP_PERIODS (default 3)   SKIP_SWEEP (default "0 1 2 3 4 5")
# =============================================================================

using Oceananigans, Plots, Printf, Statistics

get!(ENV, "GKSwstype", "100")

const HERE   = @__DIR__
const ω      = 1e-4
const T_tide = 2π / ω
const ν      = 1.0e-6
const δ      = sqrt(2ν / ω)
const Lz     = 150δ                    # PHYSICAL domain ≈ 21.213 m (case_params.jl)
const GRAD_FLOOR = 0.05
const SKIP   = parse(Float64, get(ENV, "SKIP_PERIODS", "3"))
const SWEEP  = parse.(Float64, split(get(ENV, "SKIP_SWEEP", "0 1 2 3 4 5")))

default(fontfamily = "sans-serif", framestyle = :box, grid = true, gridalpha = 0.15,
        tickfontsize = 7, guidefontsize = 8, legendfontsize = 6, titlefontsize = 8)

# Verbatim from MixedLayerDiffusivity.jl — see the note there; do not diverge.
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

first_crossing(z, fv, lev) = begin
    i = findfirst(k -> isfinite(fv[k]) && fv[k] >= lev, eachindex(fv))
    (i === nothing || i == 1) ? NaN :
        z[i-1] + (z[i] - z[i-1]) * (lev - fv[i-1]) / (fv[i] - fv[i-1])
end
interp_at(z, fv, z₀) = isnan(z₀) ? NaN : begin
    i = searchsortedlast(z, z₀)
    i < 1 && return fv[1]
    i >= length(z) && return fv[end]
    fv[i] + (fv[i+1] - fv[i]) * (z₀ - z[i]) / (z[i+1] - z[i])
end

function load_case(dir)
    m = match(r"^output_L(.+?)Lz_Ri(\d+)$", basename(dir))
    m === nothing && return nothing
    Lfrac = parse(Float64, replace(m[1], "p" => "."))
    Ri    = parse(Float64, m[2])
    fs = filter(f -> endswith(f, "_profiles.jld2"), readdir(dir; join = true))
    isempty(fs) && return nothing
    f = first(fs)
    ts = Dict(v => FieldTimeSeries(f, v) for v in ("U","V","B","uu","vv","ww","wb"))
    times = ts["B"].times; nt = length(times)
    zc = collect(znodes(ts["B"])); zf = collect(znodes(ts["wb"]))
    grab(v) = (a = zeros(length(interior(ts[v][1])), nt);
               for n in 1:nt; a[:, n] .= vec(interior(ts[v][n])); end; a)
    U, V, B      = grab("U"), grab("V"), grab("B")
    uu, vv, ww   = grab("uu"), grab("vv"), grab("ww")
    wb           = grab("wb")

    # ww is on Faces (see header note 3) — average the two faces bounding each cell.
    ww_c = [0.5 * (ww[k, n] + ww[min(k + 1, size(ww, 1)), n])
            for k in 1:length(zc), n in 1:nt]
    TKE = 0.5 .* ((uu .- U .^ 2) .+ (vv .- V .^ 2) .+ ww_c)

    N²_ref = Ri > 0 ? Ri * ω^2 : ω^2          # passive scalar carries ω² at Ri = 0
    G = zeros(length(zf), nt)                 # offline ∂B/∂z onto Faces
    for n in 1:nt, k in 2:length(zc)
        G[k, n] = (B[k, n] - B[k-1, n]) / (zc[k] - zc[k-1])
    end
    G[1, :] .= G[2, :]

    h = [first_crossing(zf, view(G, :, n) ./ N²_ref, 0.1) for n in 1:nt]
    floor_val = GRAD_FLOOR * N²_ref
    K_at_h = [(g = interp_at(zf, view(G, :, n), h[n]);
               (isnan(g) || abs(g) < floor_val) ? NaN :
                   -interp_at(zf, view(wb, :, n), h[n]) / g) for n in 1:nt]
    TKE_at_h = [interp_at(zc, view(TKE, :, n), h[n]) for n in 1:nt]

    tag = @sprintf("L%gLz_Ri%d", Lfrac, Ri)
    return (; tag, Lfrac, Ri, s = sqrt(Ri), times, K = K_at_h, TKE = TKE_at_h, h)
end

fit_at(c, skip) = (m = c.times .>= skip * T_tide; loglog_slope(c.TKE[m], c.K[m]))

# =============================================================================
dirs  = sort(filter(d -> isdir(d) && startswith(basename(d), "output_"),
                    readdir(HERE; join = true)))
cases = filter(!isnothing, [load_case(d) for d in dirs])
sort!(cases, by = c -> (c.Ri, c.Lfrac))
@printf("loaded %d case(s)\n", length(cases))

const RIS = sort(unique(c.Ri for c in cases))
const LFS = sort(unique(c.Lfrac for c in cases))

function panel(c, skip)
    m = c.times .>= skip * T_tide
    x, y, t = c.TKE[m], c.K[m], c.times[m] ./ T_tide
    g = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
    slope, r, n = loglog_slope(x, y)
    p = plot(xscale = :log10, yscale = :log10, legend = :topleft, colorbar = false,
             xlabel = "TKE at z = h (m² s⁻²)", ylabel = "K_T at z = h (m² s⁻¹)",
             title = @sprintf("%s   N/ω = %.1f\nslope %.2f (r %.2f, n %d)",
                              c.tag, c.s, slope, r, n))
    if count(g) >= 8
        scatter!(p, x[g], y[g]; marker_z = t[g], c = :viridis, ms = 1.6, msw = 0,
                 alpha = 0.55, label = "")
        x0, y0 = 10^mean(log10.(x[g])), 10^mean(log10.(y[g]))
        xr = 10 .^ range(log10(minimum(x[g])), log10(maximum(x[g])); length = 2)
        plot!(p, xr, y0 .* (xr ./ x0) .^ 0.5; color = "#B4502C", lw = 1.4, ls = :dash,
              label = "½ → geometric l")
        plot!(p, xr, y0 .* (xr ./ x0) .^ 1.0; color = "#7A3117", lw = 1.4, ls = :dot,
              label = "1 → TKE/N")
        isfinite(slope) && plot!(p, xr, y0 .* (xr ./ x0) .^ slope; color = :black,
                                 lw = 2, label = @sprintf("fit %.2f", slope))
    end
    return p
end

# Rows Ri, columns L/Lz. Ri = 0 was only run at L = 1 Lz, so the row is ragged and
# the empty slots are drawn blank rather than left to Plots' recycling.
panels = Any[]
for Riv in RIS, Lf in LFS
    i = findfirst(c -> c.Ri == Riv && c.Lfrac == Lf, cases)
    push!(panels, i === nothing ? plot(framestyle = :none, legend = false) :
                                  panel(cases[i], SKIP))
end
fig = plot(panels...; layout = (length(RIS), length(LFS)),
           size = (430 * length(LFS), 390 * length(RIS)),
           plot_title = @sprintf("panel (d), L_strat sweep — RESOLVED FLUX ONLY (no F_sgs): slopes are an upper bound.  First %g periods discarded", SKIP),
           plot_titlefontsize = 10, left_margin = 11Plots.mm, bottom_margin = 7Plots.mm)
f1 = joinpath(HERE, "figures", "panelD_legacy_grid.png")
mkpath(dirname(f1)); savefig(fig, f1)

# Stability: a slope that moves with the cutoff is the transient, not the closure.
p2 = plot(xlabel = "tidal periods discarded", ylabel = "slope  d(log K_T)/d(log TKE)",
          title = "L_strat sweep — panel (d) slope vs spin-up cutoff (resolved flux only)",
          legend = :outerright, size = (950, 520), ylims = (-0.4, 1.8),
          left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
hline!(p2, [0.5]; color = "#B4502C", ls = :dash, lw = 1.2, label = "½  geometric l")
hline!(p2, [1.0]; color = "#7A3117", ls = :dot,  lw = 1.2, label = "1  TKE/N")
const RCLR = Dict(0.0 => "#8C8C8C", 500.0 => "#1B4E8F", 2500.0 => "#B4502C")
const LSTY = Dict(0.2 => :dot, 0.5 => :dashdot, 1.0 => :solid, 1.5 => :dash)
rows = Dict{String,Any}()
for c in cases
    rr = [(sk, fit_at(c, sk)...) for sk in SWEEP]
    rows[c.tag] = rr
    plot!(p2, SWEEP, [x[2] for x in rr]; color = get(RCLR, c.Ri, :black),
          ls = get(LSTY, c.Lfrac, :solid), lw = 2, marker = :circle, ms = 3,
          label = @sprintf("Ri=%d  L=%.1fLz", c.Ri, c.Lfrac))
end
f2 = joinpath(HERE, "figures", "panelD_legacy_stability.png")
savefig(p2, f2)

@printf("\n%-16s %6s %6s %7s | %8s %7s %7s | %s\n",
        "case", "Ri", "N/ω", "L/Lz", "slope", "r", "n", "slope at cutoff " * join(SWEEP, " "))
for c in cases
    sl, r, n = fit_at(c, SKIP)
    @printf("%-16s %6d %6.1f %7.1f | %+8.2f %7.2f %7d | %s\n", c.tag, c.Ri, c.s,
            c.Lfrac, sl, r, n,
            join((@sprintf("%+.2f", x[2]) for x in rows[c.tag]), " "))
end
println("\nRESOLVED FLUX ONLY — F_sgs absent from these files; slopes are an upper bound.")
@printf("wrote %s\n      %s\n", relpath(f1, HERE), relpath(f2, HERE))
