using Oceananigans, JLD2, Plots, Printf, Statistics

# Plane-averaged velocity set against the instantaneous velocity field, so the
# mixed layer can be read off both at once:
#   julia --project=.. MeanVelocity.jl Ri500
#
# Three panels sharing the z/δ_s axis:
#   (a) ⟨u⟩(z, t) over the whole run, with the mixed-layer height h_m(t) and the
#       thermocline marker Ri_g = 0.25 drawn on top — this is the "how does the
#       mixed layer change" view.
#   (b) an instantaneous x–z slice of u at the peak phase of the last complete
#       period, showing the turbulent structure the average is taken over.
#   (c) the profile ⟨u⟩(z) at four phases against the laminar Stokes solution.
#
# (b) and (c) are separate panels on a shared vertical axis rather than a curve
# drawn over the heatmap: overlaying them would silently put two different
# horizontal scales (x position and velocity) on one axis.
#
# Reads only files already written by Tidal3D.jl.

include(joinpath(@__DIR__, "case_params.jl"))

const zmax_δ = 25.0            # boundary layer is ~15 δ_s in the paper
const zmax   = zmax_δ * δ

default(fontfamily = "sans-serif", framestyle = :box, grid = true,
        gridalpha = 0.15, tickfontsize = 9, guidefontsize = 10,
        legendfontsize = 8, titlefontsize = 11)

figdir = joinpath(@__DIR__, "figures")
mkpath(figdir)

# ---------------- profiles ----------------
pfile = filename * "_profiles.jld2"
isfile(pfile) || error("Missing $pfile — run the simulation first")

U_ts = FieldTimeSeries(pfile, "U")
V_ts = FieldTimeSeries(pfile, "V")
B_ts = FieldTimeSeries(pfile, "B")
times = U_ts.times
zc    = znodes(U_ts)
Nt    = length(times)

Umean = reduce(hcat, [vec(interior(U_ts[n])) for n in 1:Nt])
Vmean = reduce(hcat, [vec(interior(V_ts[n])) for n in 1:Nt])
Bmean = reduce(hcat, [vec(interior(B_ts[n])) for n in 1:Nt])

zg = 0.5 .* (zc[1:end-1] .+ zc[2:end])
dz = diff(zc)
G  = diff(Bmean, dims = 1) ./ dz ./ N²_ref        # ∂⟨b⟩/∂z / N², at midpoints

# Lowest height where f crosses `level` from below, linearly interpolated.
function first_crossing(z, f, level; zmin = -Inf)
    for i in 1:length(f)-1
        z[i] < zmin && continue
        if f[i] < level <= f[i+1]
            return z[i] + (level - f[i]) * (z[i+1] - z[i]) / (f[i+1] - f[i])
        end
    end
    return NaN
end

# Mixed-layer height, the paper's ∘ marker: ∂⟨b⟩/∂z / N² = 0.1.
h_m = [first_crossing(zg, G[:, n], 0.1) for n in 1:Nt]

# Thermocline, the paper's △ marker: Ri_g = N²_local / S² = 0.25. Only
# meaningful where buoyancy acts back on the flow.
if passive_scalar
    z_Rig = fill(NaN, Nt)
else
    z_Rig = map(1:Nt) do n
        S² = (diff(Umean[:, n]) ./ dz) .^ 2 .+ (diff(Vmean[:, n]) ./ dz) .^ 2
        first_crossing(zg, (G[:, n] .* N²) ./ max.(S², eps()), 0.25; zmin = δ)
    end
end

kc = findall(z -> z <= zmax, zc)
ωt = times .* ω

# ---- (a) ⟨u⟩(z, t) with the mixed-layer markers ----
# Velocity is a signed quantity oscillating about zero, so this is the one
# diverging map in the figure: two hues, neutral grey at u = 0, symmetric limits.
ulim = 1.15U₀
pa = heatmap(ωt, zc[kc] ./ δ, Umean[kc, :] ./ U₀;
             color = :balance, clims = (-ulim, ulim) ./ U₀,
             xlabel = "ωt", ylabel = "z / δ_s",
             title = "(a) ⟨u⟩ₓᵧ(z, t) / U₀ with mixed-layer height",
             colorbar_title = "  ⟨u⟩ / U₀", rightmargin = 3Plots.mm)
plot!(pa, ωt, h_m ./ δ; color = :black, linewidth = 2.5, label = "h_m  (∂b/∂z / N² = 0.1)",
      legend = :topleft, foreground_color_legend = nothing,
      background_color_legend = RGBA(1, 1, 1, 0.6))
passive_scalar || plot!(pa, ωt, z_Rig ./ δ; color = :black, linewidth = 2,
                        linestyle = :dash, label = "Ri_g = 0.25")

# ---------------- instantaneous slice ----------------
xzfile = filename * ".jld2"
# Start of the last complete period (0 if the run is shorter than two).
t0 = max(0.0, (floor(times[end] / T_tide) - 1) * T_tide)
t_peak = t0 + 0.25T_tide                            # φ = 90°, free-stream maximum

pb = plot(title = "(b) instantaneous u — slice unavailable")
if isfile(xzfile)
    u_ic = FieldTimeSeries(xzfile, "u", iterations = 0)
    xu, ~, zu = nodes(u_ic)

    file_xz = jldopen(xzfile)
    iters = sort(parse.(Int, keys(file_xz["timeseries/t"])))
    ts    = [file_xz["timeseries/t/$i"] for i in iters]
    ipk   = iters[argmin(abs.(ts .- t_peak))]
    u_xz  = file_xz["timeseries/u/$ipk"][:, 1, :]
    t_xz  = file_xz["timeseries/t/$ipk"]
    close(file_xz)

    # Plot the departure from the plane average, u' = u − ⟨u⟩ₓᵧ(z), rather than
    # u itself: at φ = 90° the free stream fills the panel at u ≈ U₀ and
    # saturates the map, hiding exactly the turbulent structure that the mean
    # profile is an average over. u' is what panel (a) averages away.
    n_pk  = argmin(abs.(times .- t_xz))
    up_xz = u_xz .- reshape(Umean[:, n_pk], 1, :)

    ku = findall(z -> z <= zmax, zu)
    uplim = max(quantile(abs.(vec(up_xz[:, ku])), 0.995) / U₀, 0.02)
    pb = heatmap(xu ./ δ, zu[ku] ./ δ, (up_xz[:, ku] ./ U₀)';
                 color = :balance, clims = (-uplim, uplim),
                 xlabel = "x / δ_s", ylabel = "z / δ_s",
                 title = @sprintf("(b) instantaneous u′ = u − ⟨u⟩ₓᵧ(z), at ωt = %.1f (φ = 90°)",
                                  ω * t_xz),
                 colorbar_title = "  u′ / U₀", rightmargin = 3Plots.mm)
    isnan(h_m[n_pk]) || hline!(pb, [h_m[n_pk] / δ]; color = :black, linewidth = 2.5,
                               label = "h_m", legend = :topright,
                               foreground_color_legend = nothing,
                               background_color_legend = RGBA(1, 1, 1, 0.6))
end

# ---- (c) mean profiles at four phases vs the laminar Stokes solution ----
# Laminar: u(z,t) = U₀ [sin(ωt) − e^(−z/δ) sin(ωt − z/δ)]. A fuller, more
# slab-like profile with a thin sharp wall layer is the signature of turbulence.
u_laminar(z, t) = U₀ * (sin(ω * t) - exp(-z / δ) * sin(ω * t - z / δ))

# Four phases of one cycle: identity, not magnitude, so a categorical set —
# assigned in fixed order, distinct in lightness as well as hue.
phase_colors = ["#3C7CC4", "#D1633C", "#4F9A6A", "#8A5FA8"]
phases = (0.0, 0.25, 0.5, 0.75)

pc = plot(xlabel = "⟨u⟩ₓᵧ / U₀", ylabel = "z / δ_s", ylims = (0, zmax_δ),
          title = "(c) mean profile vs laminar Stokes (dashed)",
          legend = :bottomright, foreground_color_legend = nothing,
          background_color_legend = nothing)
for (i, ϕ) in enumerate(phases)
    n = argmin(abs.(times .- (t0 + ϕ * T_tide)))
    col = phase_colors[i]
    plot!(pc, Umean[kc, n] ./ U₀, zc[kc] ./ δ; color = col, linewidth = 2,
          label = @sprintf("φ = %d°", round(Int, 360ϕ)))
    plot!(pc, u_laminar.(zc[kc], times[n]) ./ U₀, zc[kc] ./ δ;
          color = col, linewidth = 1.2, linestyle = :dash, label = "")
    isnan(h_m[n]) || scatter!(pc, [Umean[argmin(abs.(zc .- h_m[n])), n] / U₀],
                              [h_m[n] / δ]; color = col, markershape = :circle,
                              markersize = 6, markerstrokecolor = :white,
                              markerstrokewidth = 1.5, label = "")
end

Ri_label = passive_scalar ? "0 (passive scalar)" : case[3:end]
fig = plot(pa, pb, pc; layout = (3, 1), size = (950, 1250),
           leftmargin = 6Plots.mm, bottommargin = 5Plots.mm,
           plot_title = "Ri = $Ri_label — mean velocity, instantaneous field and mixed layer",
           plot_titlefontsize = 12)
savefig(fig, joinpath(figdir, "MeanVelocity_$case.png"))
@info "Saved figures/MeanVelocity_$case.png"
