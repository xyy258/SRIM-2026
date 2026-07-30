using Oceananigans, JLD2, Plots, Printf

# Reproduction of Figure 5 of Gayen, Sarkar & Taylor (2010): vertical profiles
# of the plane-averaged thermal field — (a) buoyancy and (b) buoyancy gradient —
# against z/δ_s, with one curve per output time.
#
# The paper shows Ri = 500 and Ri = 2500 on shared axes at t = 15 and t = 50.
# Here each Ri gets its own figure with several times, so the mixed-layer growth
# is visible as a family of curves rather than a two-line before/after.
#
# Non-dimensionalization follows the paper's (2.4): time is t = t_d ω, lengths in
# the profile axis are z/δ_s, buoyancy is scaled by a·N² with a = U₀/ω (their
# θ = θ_d / (a dθ/dz|∞), and b = βgθ so b/(a N²) is the same quantity), and the
# gradient by N² so the far field tends to 1.
#
# Markers follow the paper: ∘ is the mixed-layer height h_m where the normalized
# gradient first reaches 0.1; △ is where Ri_g = N²_local/S² first reaches 0.25.
#
# Reads only the profile files already written by Tidal3D.jl. Run from this
# directory:  julia --project=.. Figure5.jl

const ω  = 1.4075235e-4
const ν  = 1.0e-6              # paper §2.4 dimensional values
const U₀ = 0.015
const δ  = sqrt(2ν / ω)        # Stokes thickness ≈ 0.119 m
const a  = U₀ / ω              # tidal excursion scale, sets the buoyancy scale
const T_tide = 2π / ω

const zmax_δ = 50.0            # the paper's z/δ_s axis range
const z_sponge_δ = 70.0        # sponge floor (paper: 70 δ_s – 90 δ_s); the mask
                               # is < 1e-8 anywhere shown here, so nothing is
                               # shaded — kept so the guard survives a domain change

# Ri = 0 carries the thermal field as a passive scalar with background gradient
# ω² (case_params.jl), so it gets a real pair of panels rather than a note.
cases = [("Ri0", 1.0 * ω^2), ("Ri500", 500.0 * ω^2), ("Ri2500", 2500.0 * ω^2)]

# Snapshots taken at whole tidal periods, i.e. all at the same phase (φ = 0,
# U∞ = 0). Sampling at a fixed phase means the differences between curves are
# mixed-layer growth and not the within-cycle modulation. n = 8 lands at
# ωt = 50.3, the paper's late time (their fig. 5 shows t = 15 and t = 50).
n_periods_plot = [1, 2, 4, 6, 8]

# Sequential single-hue ramp: later time = darker. Lightness-ordered ramps stay
# separable under every form of colour-vision deficiency (OKLab L steps here are
# 0.83 → 0.70 → 0.58 → 0.45 → 0.32, gaps of ≈0.13 ≫ the 0.08 target).
ramp = ["#A8CBEC", "#6BA3DE", "#3C7CC4", "#1F559B", "#0B3164"]

# Lowest height at which f crosses `level` from below, linearly interpolated.
function first_crossing(z, f, level; zmin = -Inf)
    for i in 1:length(f)-1
        z[i] < zmin && continue
        if f[i] < level <= f[i+1]
            return z[i] + (level - f[i]) * (z[i+1] - z[i]) / (f[i+1] - f[i])
        end
    end
    return NaN
end

interp_at(z, f, z₀) = isnan(z₀) ? NaN : begin
    i = searchsortedlast(z, z₀)
    i < 1 && return f[1]
    i >= length(z) && return f[end]
    f[i] + (f[i+1] - f[i]) * (z₀ - z[i]) / (z[i+1] - z[i])
end

default(fontfamily = "sans-serif", grid = true, gridalpha = 0.15,
        framestyle = :box, tickfontsize = 9, guidefontsize = 10,
        legendfontsize = 8, titlefontsize = 11)

outdir = joinpath(@__DIR__, "figures")
mkpath(outdir)

for (case, N²) in cases
    fname = joinpath(@__DIR__, "output_" * case,
                     "TidalBL3D_" * case * "_profiles.jld2")
    isfile(fname) || (@warn "Missing $fname — skipping $case"; continue)

    B_ts = FieldTimeSeries(fname, "B")
    U_ts = FieldTimeSeries(fname, "U")
    V_ts = FieldTimeSeries(fname, "V")
    times = B_ts.times
    zc = znodes(B_ts)

    zg = 0.5 .* (zc[1:end-1] .+ zc[2:end])     # gradients live at the midpoints
    dz = diff(zc)
    kc = findall(z -> z / δ <= zmax_δ, zc)
    kg = findall(z -> z / δ <= zmax_δ, zg)

    passive = case == "Ri0"
    bscale = a * N²                        # paper's θ = θ_d / (a dθ/dz|∞)
    gscale = N²

    pa = plot(; ylabel = "z / δ_s", xlabel = "⟨b⟩ₓᵧ / (a N²)",
              title = "(a) buoyancy", ylims = (0, zmax_δ), yticks = 0:5:zmax_δ,
              legend = :bottomright,
              foreground_color_legend = nothing, background_color_legend = nothing)
    pb = plot(; ylabel = "z / δ_s", xlabel = "∂⟨b⟩ₓᵧ/∂z / N²",
              title = "(b) buoyancy gradient", ylims = (0, zmax_δ),
              yticks = 0:5:zmax_δ, legend = :topleft,
              foreground_color_legend = nothing, background_color_legend = nothing)

    # Symbol key, drawn in neutral ink so the marker shape carries the meaning
    # and the series colour keeps carrying the time.
    scatter!(pb, [NaN], [NaN]; markershape = :circle, markersize = 6,
             color = RGB(0.42, 0.42, 0.42), markerstrokecolor = :white,
             markerstrokewidth = 1.5, label = "h_m: ∂b/∂z / N² = 0.1")
    passive || scatter!(pb, [NaN], [NaN]; markershape = :utriangle, markersize = 6,
                        color = RGB(0.42, 0.42, 0.42), markerstrokecolor = :white,
                        markerstrokewidth = 1.5, label = "Ri_g = 0.25")

    if z_sponge_δ < zmax_δ
        for p in (pa, pb)
            # Profiles inside the sponge are relaxed toward the background
            # rather than freely evolving, so they are shaded as unreliable.
            hspan!(p, [z_sponge_δ, zmax_δ]; color = :gray, alpha = 0.12,
                   linewidth = 0, label = "")
        end
    end

    for (j, np) in enumerate(n_periods_plot)
        t_target = np * T_tide
        t_target > times[end] && continue
        n = argmin(abs.(times .- t_target))
        col = ramp[min(j, length(ramp))]

        Bp = vec(interior(B_ts[n]))
        Up = vec(interior(U_ts[n]))
        Vp = vec(interior(V_ts[n]))

        G  = diff(Bp) ./ dz ./ gscale            # ∂b/∂z (normalized), midpoints
        S² = (diff(Up) ./ dz) .^ 2 .+ (diff(Vp) ./ dz) .^ 2
        bn = Bp ./ bscale

        lbl = @sprintf("ωt = %.1f  (%d T)", ω * times[n], np)
        plot!(pa, bn[kc], zc[kc] ./ δ; color = col, linewidth = 2, label = lbl)
        plot!(pb, G[kg], zg[kg] ./ δ; color = col, linewidth = 2, label = "")

        # ∘ mixed-layer height: ∂b/∂z / N² = 0.1
        h_m = first_crossing(zg, G, 0.1)
        # △ Ri_g = 0.25, searched above z = δ_s to skip the near-wall noise
        # Ri_g has no dynamical meaning where buoyancy does not act back on the
        # flow, so the △ marker is drawn for the stratified cases only.
        Rig = (G .* N²) ./ max.(S², eps())
        z_Rig = passive ? NaN : first_crossing(zg, Rig, 0.25; zmin = δ)

        for (z₀, mk) in ((h_m, :circle), (z_Rig, :utriangle))
            isnan(z₀) && continue
            scatter!(pa, [interp_at(zc, bn, z₀)], [z₀ / δ]; color = col,
                     markershape = mk, markersize = 6, markerstrokecolor = :white,
                     markerstrokewidth = 1.5, label = "")
            scatter!(pb, [interp_at(zg, G, z₀)], [z₀ / δ]; color = col,
                     markershape = mk, markersize = 6, markerstrokecolor = :white,
                     markerstrokewidth = 1.5, label = "")
        end
    end

    xlims!(pb, 0, 1.5)

    Ri_label = passive ? "0 (no stratification)" : case[3:end]
    fig = plot(pa, pb; layout = (1, 2), size = (950, 620), leftmargin = 5Plots.mm,
               bottommargin = 5Plots.mm,
               plot_title = "Ri = $Ri_label — plane-averaged thermal field (cf. Gayen et al. 2010, fig. 5)",
               plot_titlefontsize = 12)
    savefig(fig, joinpath(outdir, "Figure5_$case.png"))
    @info "Saved figures/Figure5_$case.png"
end
