using Oceananigans, JLD2, Plots, Printf

# Figure 5 for the L_strat sweep (deviates from Gayen et al.; see case_params.jl).
# Per case (L, Ri): vertical profiles of the plane-averaged (a) buoyancy and
# (b) buoyancy gradient at several whole tidal periods, so the mixed-layer growth
# shows as a family of curves. One PNG per case: figures/Figure5_L<L>_<Ri>.png.
#
# Depth axis is z/δ with δ = u*/f (f = ω = 1e-4), u* = peak wall-friction velocity
# over the final period. Markers follow the paper: ∘ = mixed-layer height h_m
# where the normalized gradient first reaches 0.1; △ = where Ri_g = N²/S² first
# reaches 0.25.
#
# Reads only *_profiles.jld2 written by Tidal3D.jl.  Run:  julia --project=.. Figure5.jl

const ω  = 1e-4
const f  = ω
const ν  = 1.0e-6
const U₀ = 0.04
const a  = U₀ / ω                  # tidal excursion scale, sets the buoyancy scale
const T_tide = 2π / ω

const L_values  = [2, 4, 6, 8]
const Ri_values = [(500, "Ri500"), (2500, "Ri2500")]
# δ = u*/f ≈ 10 m ≈ domain height, so the tidal boundary layer sits in z/δ ≲ 0.5.
# Zoom the axis to 0.8 so the mixed layer and pycnocline are visible rather than
# crushed into a sliver at the bottom.
const zmax_δ    = 0.8              # z/δ axis range (δ = u*/f)
const n_periods_plot = [1, 2, 4, 6, 8]

ramp = ["#A8CBEC", "#6BA3DE", "#3C7CC4", "#1F559B", "#0B3164"]

# Lowest height at which f crosses `level` from below, linearly interpolated.
function first_crossing(z, fv, level; zmin = -Inf)
    for i in 1:length(fv)-1
        z[i] < zmin && continue
        if fv[i] < level <= fv[i+1]
            return z[i] + (level - fv[i]) * (z[i+1] - z[i]) / (fv[i+1] - fv[i])
        end
    end
    return NaN
end

interp_at(z, fv, z₀) = isnan(z₀) ? NaN : begin
    i = searchsortedlast(z, z₀)
    i < 1 && return fv[1]
    i >= length(z) && return fv[end]
    fv[i] + (fv[i+1] - fv[i]) * (z₀ - z[i]) / (z[i+1] - z[i])
end

default(fontfamily = "sans-serif", grid = true, gridalpha = 0.15,
        framestyle = :box, tickfontsize = 9, guidefontsize = 10,
        legendfontsize = 8, titlefontsize = 11)

outdir = joinpath(@__DIR__, "figures")
mkpath(outdir)

for L in L_values, (Ri, ricase) in Ri_values
    tag   = "L$(L)_$(ricase)"
    fname = joinpath(@__DIR__, "output_" * tag, "TidalBL3D_" * tag * "_profiles.jld2")
    isfile(fname) || (@warn "Missing $fname — skipping $tag"; continue)

    B_ts = FieldTimeSeries(fname, "B")
    U_ts = FieldTimeSeries(fname, "U")
    V_ts = FieldTimeSeries(fname, "V")
    times = B_ts.times
    zc = znodes(B_ts)

    zg = 0.5 .* (zc[1:end-1] .+ zc[2:end])
    dz = diff(zc)

    N² = Ri * ω^2
    bscale = a * N²
    gscale = N²

    # δ = u*/f (peak wall stress over the final period)
    z1 = zc[1]
    uτ = [sqrt(ν * abs(interior(U_ts[n])[1, 1, 1]) / z1) for n in 1:length(times)]
    ustar = maximum(uτ[times .>= (times[end] - T_tide)])
    δ = ustar / f

    kc = findall(z -> z / δ <= zmax_δ, zc)
    kg = findall(z -> z / δ <= zmax_δ, zg)

    pa = plot(; ylabel = "z / δ  (δ=u*/f)", xlabel = "⟨b⟩ₓᵧ / (a N²)",
              title = "(a) buoyancy", ylims = (0, zmax_δ),
              legend = :bottomright,
              foreground_color_legend = nothing, background_color_legend = nothing)
    pb = plot(; ylabel = "z / δ  (δ=u*/f)", xlabel = "∂⟨b⟩ₓᵧ/∂z / N²",
              title = "(b) buoyancy gradient", ylims = (0, zmax_δ),
              legend = :topright,
              foreground_color_legend = nothing, background_color_legend = nothing)

    scatter!(pb, [NaN], [NaN]; markershape = :circle, markersize = 6,
             color = RGB(0.42, 0.42, 0.42), markerstrokecolor = :white,
             markerstrokewidth = 1.5, label = "h_m: ∂b/∂z / N² = 0.1")
    scatter!(pb, [NaN], [NaN]; markershape = :utriangle, markersize = 6,
             color = RGB(0.42, 0.42, 0.42), markerstrokecolor = :white,
             markerstrokewidth = 1.5, label = "Ri_g = 0.25")

    for (j, np) in enumerate(n_periods_plot)
        t_target = np * T_tide
        t_target > times[end] && continue
        n = argmin(abs.(times .- t_target))
        col = ramp[min(j, length(ramp))]

        Bp = vec(interior(B_ts[n]))
        Up = vec(interior(U_ts[n]))
        Vp = vec(interior(V_ts[n]))

        G  = diff(Bp) ./ dz ./ gscale
        S² = (diff(Up) ./ dz) .^ 2 .+ (diff(Vp) ./ dz) .^ 2
        bn = Bp ./ bscale

        lbl = @sprintf("ωt = %.1f  (%d T)", ω * times[n], np)
        plot!(pa, bn[kc], zc[kc] ./ δ; color = col, linewidth = 2, label = lbl)
        plot!(pb, G[kg], zg[kg] ./ δ; color = col, linewidth = 2, label = "")

        h_m = first_crossing(zg, G, 0.1)
        Rig = (G .* N²) ./ max.(S², eps())
        z_Rig = first_crossing(zg, Rig, 0.25; zmin = δ)

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

    fig = plot(pa, pb; layout = (1, 2), size = (950, 620), leftmargin = 5Plots.mm,
               bottommargin = 5Plots.mm,
               plot_title = @sprintf("L = %d m, Ri = %d — thermal field (δ = u*/f = %.2f m)", L, Ri, δ),
               plot_titlefontsize = 12)
    savefig(fig, joinpath(outdir, "Figure5_$tag.png"))
    @info "Saved figures/Figure5_$tag.png"
end
