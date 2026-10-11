# Ri at the layer top against the layer height h, followed through each run.
# Ri = N^2/S^2 with N the BACKGROUND value (r omega or r f) and S = |dU/dz| at
# z = h, as in the z = h K_T* figures. h is the crossing height (db/dz back to
# 0.1 N^2_ref). Running window of WIN forcing periods (tidal for Stokes,
# inertial for Ekman) stepped every STEP of a period: h is the window mean and
# Ri = N^2/<S^2>, the ratio of averages, so the near-zero shear twice per tide
# cannot spike it. Neighbouring points share data. The first period is
# dropped: the flow has not reached h yet.
# Dashed: Ri = N^2/S(h)^2 with S(h) = S0 exp(-h/l), one fit per flow pooled over
# every run and period (S = N/sqrt(Ri)): at a given h, S hardly depends on N.
# Stokes: all r. Ekman: r < 25, the set the growth curves use.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot/plot_Ri_vs_h_T10.jl

using Oceananigans, JLD2, Plots, Printf, Statistics, LaTeXStrings

get!(ENV, "GKSwstype", "100")
default(dpi = 600, fontfamily = "DejaVu Sans", guidefontsize = 22, tickfontsize = 17,
        legendfontsize = 15, titlefontsize = 20)

const HERE  = dirname(@__DIR__)
include(joinpath(HERE, "sweep.jl"))
include(joinpath(HERE, "mixed_layer_height.jl"))
const CACHE = joinpath(HERE, "Data", "cache", "profiles_T10.jld2")
const Ω     = 1e-4
const TP    = 2π / Ω                    # one tidal or inertial period
const DROP  = 25
const WIN   = 2.0                      # window, forcing periods
const STEP  = 0.25                     # step, forcing periods
const A4    = (1169, 827)              # A4 landscape at 100 px/in, for the side-by-side figure

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))

interp_at(z, fv, z₀) = isnan(z₀) ? NaN : begin
    i = searchsortedlast(z, z₀)
    i < 1 && return fv[1]
    i >= length(z) && return fv[end]
    fv[i] + (fv[i+1] - fv[i]) * (z₀ - z[i]) / (z[i+1] - z[i])
end

function shear_on_faces(U, V, zc, zf)
    S = similar(float(U), length(zf))
    @inbounds for k in 2:length(zc)
        dz = zc[k] - zc[k-1]
        S[k] = hypot((U[k] - U[k-1]) / dz, (V[k] - V[k-1]) / dz)
    end
    S[1] = S[2]; S[end] = S[end-1]
    return S
end

col(F, n) = Float64.(Array(interior(F[n], 1, 1, :)))

# Per-period medians of h and Ri from series t, h, S.
function by_period(t, h, S, N)
    out = NamedTuple[]
    for a in TP:STEP * TP:t[end] - WIN * TP
        k = findall(i -> a <= t[i] < a + WIN * TP && isfinite(h[i]) && isfinite(S[i]), eachindex(t))
        length(k) < 3 && continue
        push!(out, (h = mean(h[k]), Ri = N^2 / mean(S[k] .^ 2)))
    end
    out
end

function stokes(s)
    c = stokes_case(s)
    t, h = jldopen(io -> (io["times"], io["h"]), c.mix, "r")
    U = FieldTimeSeries(c.mom, "U"; backend = OnDisk()); V = FieldTimeSeries(c.mom, "V"; backend = OnDisk())
    zc = Array(znodes(U.grid, Center())); zf = Array(znodes(U.grid, Face()))
    S = [interp_at(zf, shear_on_faces(col(U, n), col(V, n), zc, zf), h[n]) for n in eachindex(t)]
    by_period(t, h, S, s * Ω)
end

function ekman(r; every = 4)
    file = joinpath(ekman_case(r).dir, "Moments.jld2")
    U = FieldTimeSeries(file, "U"; backend = OnDisk()); V = FieldTimeSeries(file, "V"; backend = OnDisk())
    G = FieldTimeSeries(file, "dBdz"; backend = OnDisk())
    zc = Array(znodes(U.grid, Center())); zf = Array(znodes(U.grid, Face()))
    ns = 1:every:length(U.times); t = U.times[ns]
    h = [first_crossing(zf, col(G, n) ./ (r * Ω)^2, H_LEVEL; zmin = 0.5) for n in ns]
    S = [interp_at(zf, shear_on_faces(col(U, n), col(V, n), zc, zf), h[i]) for (i, n) in enumerate(ns)]
    by_period(t, h, S, r * Ω)
end

# ln S = a - h/l, least squares; returns S0, l and the rms of ln S.
function sfit(H, S)
    y = log.(S); b = sum((H .- mean(H)) .* (y .- mean(y))) / sum((H .- mean(H)) .^ 2); a = mean(y) - b * mean(H)
    (S0 = exp(a), l = -1 / b, rms = sqrt(mean((y .- a .- b .* H) .^ 2)))
end

ps = []
for (fl, rs, get, sym, nm) in (("stokes", filter(<(DROP), SVALS), stokes, "N/ω", "Stokes"),
                               ("ekman", filter(<(DROP), RATIOS), ekman, "N/f", "Ekman"))
    p = plot(yscale = :log10, xlabel = L"h\ \ (\mathrm{m})", ylabel = L"Ri = N^2/S^2\ \ \mathrm{at}\ z = h",
             legend = :outerright, foreground_color_legend = nothing, minorgrid = true, minorticks = 9,
             title = latexstring("\\mathrm{$nm}:\\ \\ Ri\\ \\mathrm{at\\ the\\ layer\\ top,\\ running\\ mean\\ over\\ 2\\ periods}"))
    say("\n$nm: per-period median h (m) and Ri (background N)")
    qs = [(r, get(r)) for r in rs]
    H = [x.h for (r, q) in qs for x in q]; Sx = [r * Ω / sqrt(x.Ri) for (r, q) in qs for x in q]
    f = sfit(H, Sx)
    say(@sprintf("%s pooled: S(h) = %.3e exp(-h/%.2f m)  rms(ln S) %.2f  n = %d", nm, f.S0, f.l, f.rms, length(H)))
    for (r, q) in qs
        hs = [x.h for x in q]; ri = [x.Ri for x in q]
        say(@sprintf("  %s = %-4g  ", sym, r) * join((@sprintf("(%.1f, %.3g)", a, b) for (a, b) in zip(hs, ri)), " "))
        scatter!(p, hs, ri; mc = ramp_colour(r), ms = 4, msw = 0,
              label = @sprintf("%s = %g", sym, r))
        hh = range(minimum(hs), maximum(hs); length = 20)
        plot!(p, hh, (r * Ω ./ (f.S0 .* exp.(-hh ./ f.l))) .^ 2; color = ramp_colour(r), lw = 1.2, ls = :dash, label = "")
    end
    e = floor(Int, log10(f.S0))
    plot!(p, [NaN], [NaN]; color = :grey40, lw = 1.2, ls = :dash,
          label = latexstring(@sprintf("S = %.2f\\times10^{%d}\\,e^{-h/%.1f}", f.S0 / 10.0^e, e, f.l)))
    lo, hi = extrema(vcat([x.Ri for (r, q) in qs for x in q]))
    plot!(p; yticks = exp10.(floor(Int, log10(lo)):ceil(Int, log10(hi))))
    o = joinpath(HERE, "figures", "Ri_vs_h_$(fl)_T10.png")
    savefig(plot(p; size = (1100, 760), left_margin = 6Plots.mm, bottom_margin = 6Plots.mm), o); say("wrote $o")
    # Stokes: empty decades on top for the legend; Ekman: legend in the empty lower right.
    nm == "Stokes" && plot!(p; ylims = (lo / 1.5, 1e4), yticks = exp10.(floor(Int, log10(lo)):4))
    push!(ps, plot!(p; title = nm, legend = nm == "Stokes" ? :topright : :bottomright, legendfontsize = 12, background_color_legend = nothing))
end
o = joinpath(HERE, "figures", "Ri_vs_h_pair_T10.png")
savefig(plot(ps...; layout = (1, 2), size = A4, left_margin = 6Plots.mm, bottom_margin = 6Plots.mm,
             plot_title = L"Ri\ \mathrm{at\ the\ layer\ top,\ running\ mean\ over\ 2\ periods}"), o); say("wrote $o")

mkpath(joinpath(HERE, "logs"))
write(joinpath(HERE, "logs", "plot_Ri_vs_h_T10.log"), join(logl, "\n") * "\n")
