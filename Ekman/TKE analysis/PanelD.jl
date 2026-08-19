#!/usr/bin/env julia
# =============================================================================
# PanelD.jl — panel (d) for the Ekman TKE sweep: K_T against TKE at z = h,
# at fixed pycnocline depth T, one panel per stratification ratio r = N/f.
#
# NEW FILE. Nothing else in this folder is touched.
#
# WHAT HAD TO BE REBUILT, AND WHY IT IS NOT AS CLEAN AS THE STOKES VERSION
# -----------------------------------------------------------------------
# There are no *_moments.jld2 here — Moments.jl was never run on these cases —
# so the second moments are reconstructed from what is on disk:
#
#   Avg_vel.jld2     u_avg, v_avg   TRUE plane means over the full 100×100 plane
#   Avg_b.jld2       b              TRUE plane mean
#   Avg_grad_b.jld2  db_dz          TRUE plane mean of ∂z(b), on Faces
#   Velocity.jld2    u, v, w        ONE x–z SLICE, indices = (:, 1, :)
#   Buoyancy.jld2    b              ONE x–z SLICE
#
# The means are exact. The FLUCTUATION products are not: Ekman3D.jl saves the
# 3D fields as a single j = 1 slice, so ⟨u′u′⟩ and ⟨w′b′⟩ can only be estimated
# by averaging 100 x-points instead of the full 100×100 = 10⁴. That is ~10×
# noisier per sample than a genuine plane average. Time smoothing recovers a
# usable signal but the scatter in panel (d) is inflated relative to Stokes, and
# r there is NOT comparable to the Stokes r values.
#
# Using the TRUE plane mean to form the fluctuation (rather than the slice's own
# x-mean) is deliberate: it removes the leading bias, leaving only the variance
# penalty. ⟨w⟩ = 0 by incompressibility plus the rigid lid, so w′ = w.
#
# RESOLVED FLUX ONLY. There is no SGS flux on disk, so F_b = ⟨w′b′⟩ alone and
# K_T is biased low — same limitation as the archived Stokes sweeps.
#
# STAGGERING. u, v, b are on z-Centers (500); w and db_dz are on z-Faces (501);
# the grid is STRETCHED, so face↔centre transfers are interpolated with the real
# node spacing, not by halving. Averaged files carry halos, sliced files do not,
# hence FieldTimeSeries (which strips halos) for the former and raw JLD2
# streaming for the latter — the sliced files are ~1.6 GB per case and are read
# one snapshot at a time rather than held in memory.
#
# USAGE  GKSwstype=100 julia --project=<a project with Oceananigans+Plots> PanelD.jl
# ENV    T_FIXED (40)  SKIP_PERIODS (2)  R_VALUES ("" = every r found)
# =============================================================================

using JLD2, Oceananigans, Plots, Printf, Statistics

get!(ENV, "GKSwstype", "100")

const HERE  = @__DIR__
const f₀    = 1e-4
const T_f   = 2π / f₀                       # inertial period, 62832 s
const TFIX  = parse(Float64, get(ENV, "T_FIXED", "40"))
const SKIP  = parse(Float64, get(ENV, "SKIP_PERIODS", "2"))
const GRAD_FLOOR = 0.05
const RSEL  = filter(!isempty, split(get(ENV, "R_VALUES", "")))
const figdir = joinpath(HERE, "Plots")
mkpath(figdir)

default(fontfamily = "sans-serif", framestyle = :box, grid = true, gridalpha = 0.15,
        tickfontsize = 7, guidefontsize = 8, legendfontsize = 6, titlefontsize = 8)

function loglog_slope(x, y)                 # verbatim from MixedLayerDiffusivity.jl
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
function boxcar(a, nh)                      # centred, shrinking at the edges
    nh <= 0 && return collect(float.(a))
    n = length(a); o = similar(float.(a))
    for i in 1:n
        lo, hi = max(1, i - nh), min(n, i + nh)
        o[i] = mean(@view a[lo:hi])
    end
    return o
end

sorted_iters(io, v) = sort(parse.(Int, filter(!=("serialized"), keys(io["timeseries/$v"]))))

function analyse(dir, r)
    N²_ref = r > 0 ? (r * f₀)^2 : 1.0       # Ekman3D.jl: scale = (r==0) ? 1 : N²
    ua = FieldTimeSeries(joinpath(dir, "Avg_vel.jld2"), "u_avg")
    va = FieldTimeSeries(joinpath(dir, "Avg_vel.jld2"), "v_avg")
    ba = FieldTimeSeries(joinpath(dir, "Avg_b.jld2"),   "b")
    ga = FieldTimeSeries(joinpath(dir, "Avg_grad_b.jld2"), "db_dz")
    zc, zf = collect(znodes(ba)), collect(znodes(ga))
    tg = ga.times

    velf = jldopen(joinpath(dir, "Velocity.jld2"), "r")
    bf   = jldopen(joinpath(dir, "Buoyancy.jld2"), "r")
    its  = sorted_iters(velf, "u")
    times = [velf["timeseries/t/$k"] for k in its]
    nt, nz = length(its), length(zc)

    # Stretched grid: real interpolation weights, not 0.5.
    wf2c = [(zc[k] - zf[k]) / (zf[k+1] - zf[k]) for k in 1:nz]        # face k,k+1 -> centre k
    wc2f = [k == 1 || k > nz ? 0.0 : (zf[k] - zc[k-1]) / (zc[k] - zc[k-1]) for k in 1:nz+1]

    TKE = zeros(nz, nt); F_b = zeros(nz + 1, nt); G = zeros(nz + 1, nt)
    for (n, k) in enumerate(its)
        u = Float64.(dropdims(velf["timeseries/u/$k"], dims = 2))     # (100, 500)
        v = Float64.(dropdims(velf["timeseries/v/$k"], dims = 2))
        w = Float64.(dropdims(velf["timeseries/w/$k"], dims = 2))     # (100, 501)
        b = Float64.(dropdims(bf["timeseries/b/$k"], dims = 2))
        U = vec(interior(ua[n])); V = vec(interior(va[n])); B = vec(interior(ba[n]))

        up = u .- U'; vp = v .- V'; bp = b .- B'                      # w′ = w since ⟨w⟩ = 0
        ww_f = vec(mean(w .^ 2, dims = 1))                            # Faces
        ww_c = [ww_f[j] + (ww_f[j+1] - ww_f[j]) * wf2c[j] for j in 1:nz]
        TKE[:, n] = 0.5 .* (vec(mean(up .^ 2, dims = 1)) .+
                            vec(mean(vp .^ 2, dims = 1)) .+ ww_c)

        # F_b on Faces, so it shares nodes with dB/dz: b′ interpolated up.
        bpf = similar(w)
        @inbounds for j in 2:nz
            @. bpf[:, j] = bp[:, j-1] + (bp[:, j] - bp[:, j-1]) * wc2f[j]
        end
        bpf[:, 1] .= bp[:, 1]; bpf[:, nz+1] .= bp[:, nz]
        F_b[:, n] = vec(mean(w .* bpf, dims = 1))

        # db_dz is written at twice the cadence of the slices — match by time.
        G[:, n] = vec(interior(ga[argmin(abs.(tg .- times[n]))]))
    end
    close(velf); close(bf)

    # Smooth in time only after the products are formed (ratios last).
    nh = max(1, round(Int, (T_f / 20) / (times[2] - times[1]) / 2))
    for k in 1:nz;     TKE[k, :] = boxcar(view(TKE, k, :), nh); end
    for k in 1:nz+1;   F_b[k, :] = boxcar(view(F_b, k, :), nh);
                       G[k, :]   = boxcar(view(G,   k, :), nh); end

    h  = [first_crossing(zf, view(G, :, n) ./ N²_ref, 0.1) for n in 1:nt]
    fl = GRAD_FLOOR * N²_ref
    K  = [(g = interp_at(zf, view(G, :, n), h[n]);
           (isnan(g) || abs(g) < fl) ? NaN :
               -interp_at(zf, view(F_b, :, n), h[n]) / g) for n in 1:nt]
    E  = [interp_at(zc, view(TKE, :, n), h[n]) for n in 1:nt]
    return (; r, times, K, TKE = E, h)
end

# --- discover ----------------------------------------------------------------
root = joinpath(HERE, "Data", "4")
cases = []
for d in readdir(root; join = true)
    m = match(Regex("^r=([0-9.]+), T=" * @sprintf("%.1f", TFIX) * "\$"), basename(d))
    m === nothing && continue
    rv = parse(Float64, m[1])
    (isempty(RSEL) || string(rv) in RSEL || m[1] in RSEL) || continue
    push!(cases, (dir = d, r = rv))
end
sort!(cases, by = c -> c.r)
isempty(cases) && error("no cases found for T = $TFIX under $root")
@printf("T = %g m; %d case(s): r = %s\n", TFIX, length(cases),
        join((string(c.r) for c in cases), ", "))

results = []
for c in cases
    @printf("  analysing r = %-5g ... ", c.r); flush(stdout)
    t0 = time(); push!(results, analyse(c.dir, c.r))
    @printf("%.1f s\n", time() - t0)
end

# --- panels ------------------------------------------------------------------
function panel(res)
    m = res.times .>= SKIP * T_f
    x, y, t = res.TKE[m], res.K[m], res.times[m] ./ T_f
    g = @. isfinite(x) && isfinite(y) && x > 0 && y > 0
    slope, rr, n = loglog_slope(x, y)
    p = plot(xscale = :log10, yscale = :log10, legend = :topleft, colorbar = false,
             xlabel = "TKE at z = h (m² s⁻²)", ylabel = "K_T at z = h (m² s⁻¹)",
             title = @sprintf("r = N/f = %g   (T = %g m)\nslope %.2f (r %.2f, n %d)",
                              res.r, TFIX, slope, rr, n))
    if count(g) >= 8
        scatter!(p, x[g], y[g]; marker_z = t[g], c = :viridis, ms = 1.7, msw = 0,
                 alpha = 0.55, label = "")
        x0, y0 = 10^mean(log10.(x[g])), 10^mean(log10.(y[g]))
        xr = 10 .^ range(log10(minimum(x[g])), log10(maximum(x[g])); length = 2)
        plot!(p, xr, y0 .* (xr ./ x0) .^ 0.5; color = "#B4502C", lw = 1.4, ls = :dash,
              label = "½ → geometric l")
        plot!(p, xr, y0 .* (xr ./ x0) .^ 1.0; color = "#7A3117", lw = 1.4, ls = :dot,
              label = "1 → TKE/N")
        isfinite(slope) && plot!(p, xr, y0 .* (xr ./ x0) .^ slope; color = :black,
                                 lw = 2, label = @sprintf("fit %.2f", slope))
    else
        annotate!(p, 0.5, 0.5, text("too few samples", 8))
    end
    return p
end

panels = [panel(rs) for rs in results]
ncol = min(3, length(panels)); nrow = cld(length(panels), ncol)
grid_png = joinpath(figdir, @sprintf("panelD_grid_T%g.png", TFIX))
savefig(plot(panels...; layout = (nrow, ncol), size = (430 * ncol, 390 * nrow),
             plot_title = @sprintf("Ekman panel (d), T = %g m — RESOLVED FLUX ONLY, moments from a single x–z slice.  First %g inertial periods discarded", TFIX, SKIP),
             plot_titlefontsize = 9, left_margin = 11Plots.mm, bottom_margin = 7Plots.mm),
        grid_png)

# One standalone figure per r as well, since the panels are small in the grid.
singles = String[]
for (p, rs) in zip(panels, results)
    f = joinpath(figdir, @sprintf("panelD_T%g_r%g.png", TFIX, rs.r))
    savefig(plot(p; size = (620, 540), left_margin = 6Plots.mm,
                 bottom_margin = 6Plots.mm), f)
    push!(singles, f)
end

@printf("\n%-8s %10s %8s %8s | %10s %10s\n", "r=N/f", "slope", "r", "n", "mean h (m)", "med K_T")
for rs in results
    m = rs.times .>= SKIP * T_f
    sl, rr, n = loglog_slope(rs.TKE[m], rs.K[m])
    hh = filter(isfinite, rs.h[m]); kk = filter(v -> isfinite(v) && v > 0, rs.K[m])
    @printf("%-8g %10.2f %8.2f %8d | %10.2f %10.3e\n", rs.r, sl, rr, n,
            isempty(hh) ? NaN : mean(hh), isempty(kk) ? NaN : median(kk))
end
println("\nResolved flux only (no SGS) and moments from one x–z slice — slopes are")
println("indicative; the correlation r is depressed by slice noise, not by physics.")
@printf("wrote %s\n", relpath(grid_png, HERE))
for f in singles; @printf("      %s\n", relpath(f, HERE)); end
