using Oceananigans, JLD2, Plots, Printf

# Analysis of horizontally averaged profiles saved by Tidal3D.jl for one case:
#   julia --project=. Tidal3Dprofiles.jl Ri500
# Produces (all in output_<case>/, labeled with the case):
#   1. Mean velocity profiles vs the laminar Stokes solution
#   2. Friction velocity u_τ over the tidal cycle
#   3. Turbulence intensity + Reynolds stress profiles at peak flow
# and for stratified cases additionally:
#   4. Heatmap of ∂b/∂z normalized by N²
#   5. Mixed-layer depth vs time (threshold + integral metrics)
#   6. Stratification profiles at the end of each tidal period

include(joinpath(@__DIR__, "case_params.jl"))

fname = filename * "_profiles.jld2"

B_ts = FieldTimeSeries(fname, "B")     # mean buoyancy on cell centers
U_ts = FieldTimeSeries(fname, "U")     # mean streamwise velocity

times = B_ts.times
Nt    = length(times)
zc    = znodes(B_ts)                   # centers, length Nz

# Reconstruct gradients by finite-differencing between adjacent centers —
# interior data only (no halos/BC faces), so no spurious wall artifact.
Bmean = zeros(length(zc), Nt)
Umean = zeros(length(zc), Nt)
for n in 1:Nt
    Bmean[:, n] .= vec(interior(B_ts[n]))
    Umean[:, n] .= vec(interior(U_ts[n]))
end

zg = 0.5 .* (zc[1:end-1] .+ zc[2:end])         # midpoints, length Nz-1
G  = diff(Bmean, dims = 1) ./ diff(zc)         # ∂b/∂z at midpoints

zmax = 3.0                       # near-wall zoom for profile plots

# ---- 1. Mean velocity vs laminar Stokes solution ----
# Laminar solution: u(z, t) = U₀ [sin(ωt) − e^(−z/δ) sin(ωt − z/δ)].
# A fuller, more slab-like profile with a thin sharp wall layer than this
# indicates turbulence.
u_laminar(z, t) = U₀ * (sin(ω * t) - exp(-z / δ) * sin(ω * t - z / δ))

phases = (0.25, 0.5, 0.75, 1.0)  # fractions of the final tidal period
plt1 = plot(xlabel = "u (m/s)", ylabel = "z (m)", ylims = (0, zmax),
            title = "$case: mean velocity — simulation (solid) vs laminar (dashed)",
            legend = :bottomright)
t0 = (floor(times[end] / T_tide) - 1) * T_tide   # start of final full period
kc = findall(z -> z <= zmax, zc)
for (i, ϕ) in enumerate(phases)
    t = t0 + ϕ * T_tide
    n = argmin(abs.(times .- t))
    plot!(plt1, Umean[kc, n], zc[kc]; lw = 2, c = i,
          label = @sprintf("phase %.2f T", ϕ))
    plot!(plt1, u_laminar.(zc[kc], times[n]), zc[kc]; lw = 1.5, ls = :dash,
          c = i, label = "")
end
savefig(plt1, joinpath(outdir, "velocity_vs_stokes_" * case * ".png"))

# ---- 2. Friction velocity over the tidal cycle ----
# u_τ = sqrt(ν |∂U/∂z|_wall), wall gradient estimated from the lowest cell
# center (U = 0 at the wall by the no-slip BC).
uτ = sqrt.(ν .* abs.(Umean[1, :]) ./ zc[1])
plt2 = plot(times ./ T_tide, uτ;
            lw = 2, label = "u_τ",
            xlabel = "t / tidal period", ylabel = "u_τ (m/s)",
            title = "$case: friction velocity")
plot!(plt2, times ./ T_tide,
      sqrt.(√2 * ν * U₀ / δ .* abs.(sin.(ω .* times .+ π/4)));
      lw = 1.5, ls = :dash, label = "laminar |u_τ|")
savefig(plt2, joinpath(outdir, "friction_velocity_" * case * ".png"))

# ---- 3. Turbulence statistics at peak flow of the final period ----
# rms values from the saved raw second moments: u'rms = sqrt(⟨u²⟩ − U²) etc.,
# Reynolds stress u'w' ≈ ⟨uw⟩ (mean w ≈ 0).
uu_ts = FieldTimeSeries(fname, "uu")
vv_ts = FieldTimeSeries(fname, "vv")
ww_ts = FieldTimeSeries(fname, "ww")
uw_ts = FieldTimeSeries(fname, "uw")
V_ts  = FieldTimeSeries(fname, "V")

n_pk = argmin(abs.(times .- (t0 + 0.25T_tide)))   # free stream maximum
Upk  = Umean[:, n_pk]
Vpk  = vec(interior(V_ts[n_pk]))
urms = sqrt.(max.(vec(interior(uu_ts[n_pk])) .- Upk.^2, 0))
vrms = sqrt.(max.(vec(interior(vv_ts[n_pk])) .- Vpk.^2, 0))
wrms = sqrt.(max.(vec(interior(ww_ts[n_pk]))[1:length(zc)], 0))
uwpk = vec(interior(uw_ts[n_pk]))[1:length(zc)]

zstat = 6.0
ks6 = findall(z -> z <= zstat, zc)
plt3a = plot(xlabel = "rms velocity / U₀", ylabel = "z (m)",
             title = "$case: turbulence intensities at peak flow",
             legend = :topright, ylims = (0, zstat))
plot!(plt3a, urms[ks6] ./ U₀, zc[ks6]; lw = 2, label = "u′ rms")
plot!(plt3a, vrms[ks6] ./ U₀, zc[ks6]; lw = 2, label = "v′ rms")
plot!(plt3a, wrms[ks6] ./ U₀, zc[ks6]; lw = 2, label = "w′ rms")

plt3b = plot(uwpk[ks6] ./ U₀^2, zc[ks6]; lw = 2, label = "⟨u′w′⟩/U₀²",
             xlabel = "⟨u′w′⟩ / U₀²", ylabel = "z (m)",
             title = "$case: Reynolds stress at peak flow",
             legend = :topright, ylims = (0, zstat))
plot(plt3a, plt3b, layout = (1, 2), size = (1100, 500))
savefig(joinpath(outdir, "turbulence_stats_" * case * ".png"))

# ============ Stratified-case diagnostics ============
if N² > 0
    ks = findall(z -> z <= zmax, zg)

    # ---- 4. Normalized gradient heatmap, zoomed to the bottom ----
    heatmap(times ./ T_tide, zg[ks], G[ks, :] ./ N²;
            clims = (0, 2),   # 1 = unmixed, 0 = mixed, >1 = sharpened pycnocline
            color = :thermal,
            xlabel = "t / tidal period",
            ylabel = "z (m)",
            title = "$case: ∂b/∂z / N² (bottom $(zmax) m)",
            colorbar_title = "∂b/∂z / N²")
    savefig(joinpath(outdir, "buoyancy_gradient_normalized_" * case * ".png"))

    # ---- 5. Mixed-layer depth vs time ----
    #  (a) threshold depth: highest contiguous height from the wall where the
    #      stratification is below half background (can spike on bursts)
    #  (b) integral "mixing thickness": ∫ (1 − ∂b/∂z / N²)₊ dz — smooth and
    #      the more trustworthy measure.
    function threshold_depth(g, z; threshold = 0.5N²)
        k = 1
        while k <= length(z) && g[k] < threshold
            k += 1
        end
        return k == 1 ? 0.0 : z[k-1]
    end

    function mixing_thickness(g, z)
        deficit = clamp.(1 .- g ./ N², 0, 1)
        dz = diff(z)
        body = sum(0.5 .* (deficit[1:end-1] .+ deficit[2:end]) .* dz)
        wall = deficit[1] * z[1]
        return wall + body
    end

    mld_thr = [threshold_depth(G[:, n], zg)  for n in 1:Nt]
    mld_int = [mixing_thickness(G[:, n], zg) for n in 1:Nt]

    plot(times ./ T_tide, mld_int;
         lw = 2, label = "mixing thickness ∫(1−∂b/∂z/N²) dz",
         xlabel = "t / tidal period", ylabel = "mixed-layer thickness (m)",
         title = "$case: mixed-layer growth")
    plot!(times ./ T_tide, mld_thr;
          lw = 1, alpha = 0.5, label = "threshold depth (∂b/∂z < ½N²)")
    plot!(times ./ T_tide, sqrt.(2κ .* times);
          lw = 2, ls = :dash, label = "pure diffusion √(2κt)")
    hline!([δ]; ls = :dot, label = "Stokes layer δ")
    savefig(joinpath(outdir, "mixed_layer_depth_" * case * ".png"))

    # ---- 6. Stratification profiles at the end of each tidal period ----
    plt6 = plot(xlabel = "∂b/∂z / N²", ylabel = "z (m)", ylims = (0, zmax),
                xlims = (0, 1.5),
                title = "$case: stratification profiles", legend = :bottomright)
    for p in 0:floor(Int, times[end] / T_tide)
        n = argmin(abs.(times .- p * T_tide))
        plot!(plt6, G[ks, n] ./ N², zg[ks]; lw = 2,
              label = @sprintf("t = %d periods", p))
    end
    savefig(plt6, joinpath(outdir, "stratification_profiles_" * case * ".png"))
end

@info "Saved profile plots for $case in $outdir/"
