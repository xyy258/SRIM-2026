using Oceananigans, JLD2, Plots, Printf

# Figure 5 for the softplus sweep.
# Background: b_bg(z) = (N∞²/s)·log(1+e^{s(z−T)}), N²_bg(z) = N∞²·sigmoid(s(z−T))
# — unstratified below the pycnocline at z = T, N∞² above it (case_params.jl).
#
# For each case (T, √Ri): vertical profiles of the plane-averaged (a) buoyancy
# and (b) buoyancy gradient at each whole tidal period, so that mixed-layer
# growth appears as a family of curves. One PNG per case, as
# figures/Figure5_P4_T<T>_sqrtRi<s>.png.
#
# Markers follow the paper: ∘ is the mixed-layer height h_m, where the normalized
# gradient peaks, and △ is where Ri_g = N²/S² first reaches 0.25.
#
# Reads only the *_profiles.jld2 files under outputs/<tag>/. The stratification
# is labelled by N/ω = √Ri; the folder names keep the sqrtRi spelling, since they
# name data already on disk.
#   julia --project=. Figure5.jl
# Subsets:  T_VALUES="10 20" N_OVER_OMEGA="1 2 5" julia --project=. Figure5.jl

const ω  = 1e-4
const f  = ω
const ν  = 1.0e-6
const U₀ = 0.04
const δs = sqrt(2ν / ω)            # Stokes thickness — near-wall cutoff for the Ri_g search
const a  = U₀ / ω                  # tidal excursion scale, sets the buoyancy scale
const T_tide = 2π / ω

# These must match case_params.jl: the t = 0 curve is only the true initial
# condition if the profile parameters agree with the run.
# The archived runs under "-Softplus T sweep, no-slip bottom/" were made with
# SHARP = 6, so redrawing those figures needs SHARP=6 set explicitly.
const sharp = parse(Float64, get(ENV, "SHARP", "2"))
const Lz    = 50.0

parse_list(key, default) = parse.(Float64, split(get(ENV, key, default)))
const T_values    = parse_list("T_VALUES", "5 10 20 30")
# N/ω and √Ri are the same number; SQRT_RI is still accepted so the sweep driver
# keeps working unchanged.
const n_over_ω = parse_list("N_OVER_OMEGA", get(ENV, "SQRT_RI", "0 0.5 1 2 5 10"))
const outroot  = get(ENV, "OUT_ROOT", "outputs")
# Whole periods to draw, one curve each. Four at a time, which is what the colour
# ramp below holds; a longer list reuses its last colour. To plot the end of an
# 8-period run instead, pass N_PERIODS_PLOT="5 6 7 8". Periods past the end of a
# run are dropped rather than treated as an error.
const n_periods_plot = parse.(Int, split(get(ENV, "N_PERIODS_PLOT", "1 2 3 4")))

num_lbl(x) = isinteger(x) ? string(Int(x)) : replace(string(x), "." => "p")
tag_of(T, s) = "P4_T" * num_lbl(T) * "_sqrtRi" * num_lbl(s)

# Depth window per case, following Lz_test in case_params.jl.
zmax_of(T) = min(Lz, max(70δs, T + 10))

# Softplus background, normalized by N∞² so that it depends on shape only and not
# on Ri. The same expressions as profile 4 in case_params.jl.
b_bg_over_N²(z, T)  = log(1 + exp(sharp * (z - T))) / sharp     # metres²·s⁻²/N∞² → metres
N²_bg_over_N²(z, T) = 1 / (1 + exp(-sharp * (z - T)))           # dimensionless

ramp = ["#A8CBEC", "#6BA3DE", "#3C7CC4", "#0B3164"]

# The shared definitions of the mixed-layer height and of first_crossing.
include(joinpath(@__DIR__, "mixed_layer_height.jl"))

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

nsaved = 0
for T in T_values, s in n_over_ω
    tag   = tag_of(T, s)
    Ri    = s^2
    fname = joinpath(@__DIR__, outroot, tag, "TidalBL3D_" * tag * "_profiles.jld2")
    isfile(fname) || (@warn "Missing $fname — skipping $tag"; continue)

    B_ts = FieldTimeSeries(fname, "B")
    U_ts = FieldTimeSeries(fname, "U")
    V_ts = FieldTimeSeries(fname, "V")
    times = B_ts.times
    zc = znodes(B_ts)

    zg = 0.5 .* (zc[1:end-1] .+ zc[2:end])
    dz = diff(zc)
    zmax_m = zmax_of(T)

    # N² is the buoyancy actually felt, which is zero at √Ri = 0. N²_ref is what
    # the tracer carries and what the profiles are normalized by, so the √Ri = 0
    # panels show the passive scalar instead of a division by zero.
    N²     = Ri * ω^2
    N²_ref = Ri > 0 ? N² : ω^2
    bscale = a * N²_ref
    gscale = N²_ref

    # δ = u*/f, from the peak wall stress over the final period
    z1 = zc[1]
    uτ = [sqrt(ν * abs(interior(U_ts[n])[1, 1, 1]) / z1) for n in 1:length(times)]
    ustar = maximum(uτ[times .>= (times[end] - T_tide)])
    δ = ustar / f

    kc = findall(z -> z <= zmax_m, zc)
    kg = findall(z -> z <= zmax_m, zg)

    pa = plot(; ylabel = "z (m)", xlabel = "⟨b⟩ₓᵧ / (a N²)",
              title = "(a) buoyancy", ylims = (0, zmax_m),
              legend = :bottomright,
              foreground_color_legend = nothing, background_color_legend = nothing)
    pb = plot(; ylabel = "z (m)", xlabel = "∂⟨b⟩ₓᵧ/∂z / N²",
              title = "(b) buoyancy gradient", ylims = (0, zmax_m),
              legend = :topright,
              foreground_color_legend = nothing, background_color_legend = nothing)

    # The background profile at t = 0, normalized like the data: buoyancy by
    # a·N∞² and gradient by N∞². Drawn first so the curves sit on top of it.
    plot!(pa, b_bg_over_N².(zc[kc], T) ./ a, zc[kc]; color = :black,
          linestyle = :dash, linewidth = 1.5, label = "background (t = 0)")
    plot!(pb, N²_bg_over_N².(zg[kg], T), zg[kg]; color = :black,
          linestyle = :dash, linewidth = 1.5, label = "background (t = 0)")
    hline!(pb, [T]; color = RGB(0.55, 0.15, 0.15), linestyle = :dot,
           linewidth = 1.5, label = "pycnocline z = T")

    scatter!(pb, [NaN], [NaN]; markershape = :circle, markersize = 6,
             color = RGB(0.42, 0.42, 0.42), markerstrokecolor = :white,
             markerstrokewidth = 1.5, label = "h_m: peak ∂b/∂z")
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
        plot!(pa, bn[kc], zc[kc]; color = col, linewidth = 2, label = lbl)
        plot!(pb, G[kg], zg[kg]; color = col, linewidth = 2, label = "")

        h_m = mixed_layer_height(zg, G, 1.0; zmax = min(40.0, zg[end]))   # G is already ÷ N²_ref
        Rig = (G .* N²) ./ max.(S², eps())
        z_Rig = first_crossing(zg, Rig, 0.25; zmin = δs)   # skip near-wall noise

        for (z₀, mk) in ((h_m, :circle), (z_Rig, :utriangle))
            isnan(z₀) && continue
            scatter!(pa, [interp_at(zc, bn, z₀)], [z₀]; color = col,
                     markershape = mk, markersize = 6, markerstrokecolor = :white,
                     markerstrokewidth = 1.5, label = "")
            scatter!(pb, [interp_at(zg, G, z₀)], [z₀]; color = col,
                     markershape = mk, markersize = 6, markerstrokecolor = :white,
                     markerstrokewidth = 1.5, label = "")
        end
    end

    xlims!(pb, 0, 1.5)

    fig = plot(pa, pb; layout = (1, 2), size = (950, 620), leftmargin = 5Plots.mm,
               bottommargin = 5Plots.mm,
               plot_title = @sprintf("T = %d m, N/ω = %g (Ri = %g) — thermal field (δ = u*/f = %.2f m)",
                                     Int(T), s, Ri, δ),
               plot_titlefontsize = 12)
    savefig(fig, joinpath(outdir, "Figure5_$tag.png"))
    @info "Saved figures/Figure5_$tag.png"
    global nsaved += 1
end

@info "Figure 5: saved $nsaved of $(length(T_values)*length(n_over_ω)) cases"
