# Stage 2: is the TKE profile exponential, and does it collapse on delta?
#
#     TKE(z) = A1 exp(-A2 z/delta)
#
# Plotted with log TKE against LINEAR z/delta, so an exponential is a straight
# line and the eye can check the form rather than take it on trust. TKE is
# scaled by u_*^2, which is the only other velocity scale in the problem, so a
# collapse means A1/u_*^2 is universal as well as A2.
#
# delta is the stage-1 winner: u_*/Omega (1 + N^2/Omega^2)^(-p) with p fitted
# per flow, 0.040 for Stokes and 0.155 for Ekman. Those give h/delta flat to
# 1.6 % and 4.4 %, so delta and h are interchangeable here up to a constant —
# delta is used because the model of stage 3 is written in terms of it.
#
# ---------------- The fitting range ----------------
# TKE does not decay from the wall: it rises through the drag layer, peaks, and
# decays above. An exponential can only describe the decaying part, so the fit
# runs from the peak to z = h, and both ends are reported. Fitting from the wall
# would fold the near-wall rise into A2 and make it meaningless.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot_tke_profiles_T10.jl
#        (run reduce_profiles_T10.jl and plot_delta_T10.jl first)

using JLD2, Plots, Printf, Statistics, LaTeXStrings

get!(ENV, "GKSwstype", "100")
default(dpi = 600, fontfamily = "DejaVu Sans")

const HERE   = @__DIR__
const CACHE  = joinpath(HERE, "Data", "profiles_T10.jld2")
const DFILE  = joinpath(HERE, "Data", "delta_T10.jld2")
const FIGDIR = joinpath(HERE, "figures")
const SVALS  = [1, 2, 5, 10, 25, 50]
const C_STOK = "#1b3a6b"
const C_EKMA = "#8e1b4e"

const RAMP = [(0.0,   ( 27,  78, 143)), (0.301, ( 46, 139,  87)),
              (0.699, (200, 150,  30)), (1.0,   (180,  80,  44)),
              (1.398, (142,  27,  78)), (1.699, ( 75,  16,  96))]
function ramp_colour(s)
    x = clamp(log10(s), RAMP[1][1], RAMP[end][1])
    for i in 1:length(RAMP)-1
        (x0, c0), (x1, c1) = RAMP[i], RAMP[i+1]
        x <= x1 || continue
        f = x1 == x0 ? 0.0 : (x - x0) / (x1 - x0)
        chan(k) = clamp(round(Int, c0[k] + f * (c1[k] - c0[k])), 0, 255)
        return "#" * join(string(chan(k), base = 16, pad = 2) for k in 1:3)
    end
    return "#000000"
end

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))

isfile(CACHE) || error("$CACHE not found — run reduce_profiles_T10.jl first")
isfile(DFILE) || error("$DFILE not found — run plot_delta_T10.jl first")
δs = Dict{Tuple{String,Float64},Float64}()
jldopen(DFILE, "r") do io
    for fl in ("stokes", "ekman"), r in io["$fl/ratios"]
        δs[(fl, r)] = io[@sprintf("%s/r=%.1f/best", fl, r)]
    end
end

S, E = [], []
jldopen(CACHE, "r") do io
    for (fl, dst) in (("stokes", S), ("ekman", E))
        for r in io["$fl/ratios"]
            g = @sprintf("%s/r=%.1f", fl, r)
            push!(dst, (flow = fl, r = r, N = io["$g/N"], zc = io["$g/zc"],
                        TKE = io["$g/TKE"], h = io["$g/h"], us = io["$g/us"],
                        δ = δs[(fl, r)]))
        end
    end
end
BOTH = vcat(S, E)
say(@sprintf("%d Stokes and %d Ekman cases", length(S), length(E)))

# ---------------- the fit ----------------
# Straight line of log TKE against z/delta, from the TKE peak to z = h.
function expfit(c)
    z = c.zc; e = c.TKE
    ok = findall(k -> isfinite(e[k]) && e[k] > 0 && z[k] > 0, eachindex(z))
    isempty(ok) && return nothing
    kpk = ok[argmax(e[ok])]
    khi = something(findlast(k -> z[k] <= c.h, ok), ok[end])
    idx = [k for k in ok if k >= kpk && k <= khi]
    length(idx) >= 5 || return nothing
    x = z[idx] ./ c.δ; y = log.(e[idx])
    mx = mean(x); my = mean(y)
    b = sum((x .- mx) .* (y .- my)) / sum((x .- mx) .^ 2)
    a = my - b * mx
    res = y .- (a .+ b .* x)
    return (A1 = exp(a), A2 = -b, rms = 100 * sqrt(mean(res .^ 2)),
            zpk = z[kpk], zhi = z[khi], n = length(idx))
end

fits = Dict{Tuple{String,Float64},Any}()
say("")
say("TKE(z) = A1 exp(-A2 z/delta), fitted from the TKE peak up to z = h")
say("  flow    r      delta (m)   h/delta   z_peak/delta   A1/u_*^2      A2     rms")
for (nm, cs) in (("Stokes", S), ("Ekman ", E)), c in cs
    f = expfit(c)
    f === nothing && (say(@sprintf("  %s %-5g  no usable range", nm, c.r)); continue)
    fits[(c.flow, c.r)] = f
    say(@sprintf("  %s %-5g %10.3f %9.3f %13.4f %10.2f %8.3f %6.1f %%",
                 nm, c.r, c.δ, c.h / c.δ, f.zpk / c.δ, f.A1 / c.us^2, f.A2, f.rms))
end

for (nm, cs) in (("Stokes", S), ("Ekman ", E))
    a2 = [fits[(c.flow, c.r)].A2 for c in cs if haskey(fits, (c.flow, c.r))]
    a1 = [fits[(c.flow, c.r)].A1 / c.us^2 for c in cs if haskey(fits, (c.flow, c.r))]
    a2h = [fits[(c.flow, c.r)].A2 * c.h / c.δ for c in cs if haskey(fits, (c.flow, c.r))]
    say(@sprintf("  %s: A2 = %.3f - %.3f (x%.2f)   A2 h/delta = %.3f - %.3f (x%.2f)   A1/u_*^2 = %.2f - %.2f (x%.2f)",
                 nm, minimum(a2), maximum(a2), maximum(a2) / minimum(a2),
                 minimum(a2h), maximum(a2h), maximum(a2h) / minimum(a2h),
                 minimum(a1), maximum(a1), maximum(a1) / minimum(a1)))
end

# ---------------- the figure ----------------
mkpath(FIGDIR)
# Plotted against z/h, not z/delta. h/delta is constant within a flow so it is
# the same information rescaled, but the published constants 0.4 and 1.3 are
# calibrated quite differently against h (h/delta is 2.62 for Stokes and 0.66
# for Ekman), so z/delta would put the two flows on incomparable axes and hide
# z = h off the edge for Stokes.
pa = plot(yscale = :log10, xlabel = L"z/h", ylabel = L"\mathrm{TKE}/u_*^2",
          title = L"\mathrm{do\ the\ profiles\ collapse?}", xlims = (0, 2.0),
          legend = :topright, legendfontsize = 6, foreground_color_legend = nothing)
for (cs, ls, fl) in ((S, :solid, "Stokes"), (E, :dash, "Ekman"))
    for c in cs
        k = findall(j -> c.zc[j] > 0 && isfinite(c.TKE[j]) && c.TKE[j] > 0 &&
                         c.zc[j] / c.h <= 2.0, eachindex(c.zc))
        plot!(pa, c.zc[k] ./ c.h, c.TKE[k] ./ c.us^2; color = ramp_colour(c.r),
              ls = ls, lw = 1.6, label = "")
    end
end
# h/delta is nearly the same for every case in a flow, so one marker per flow
# says where the fits stop.
for (cs, mk, fl, lc) in ((S, :circle, "Stokes", C_STOK), (E, :diamond, "Ekman", C_EKMA))
    xs = fill(1.0, length(cs))
    ys = [c.TKE[argmin(abs.(c.zc .- c.h))] / c.us^2 for c in cs]
    scatter!(pa, xs, ys; marker = mk, ms = 6, msw = 1.2, mc = :white, msc = lc,
             label = latexstring("z = h,\\ \\mathrm{" * fl * "}"))
end
plot!(pa, [NaN], [NaN]; color = :grey40, ls = :solid, lw = 1.6, label = L"\mathrm{Stokes}")
plot!(pa, [NaN], [NaN]; color = :grey40, ls = :dash,  lw = 1.6, label = L"\mathrm{Ekman}")
for sv in SVALS
    plot!(pa, [NaN], [NaN]; color = ramp_colour(sv), lw = 2.2,
          label = latexstring(@sprintf("N/\\omega = N/f = %g", sv)))
end

# A_2 is defined per delta, and the two flows' delta carry different published
# constants, so A_2 alone is not comparable between them. A_2 h/delta is the
# decay across one mixed-layer depth, which is.
pb = plot(xscale = :log10, xlabel = L"r = N/\omega = N/f",
          ylabel = L"A_2\,h/\delta \ \ \mathrm{(decay\ over\ one\ }h)",
          title = L"\mathrm{is\ the\ decay\ rate\ universal?}", legend = :topleft,
          legendfontsize = 6, foreground_color_legend = nothing, ylims = (0, 5.5))
for (cs, mk, lc, fl) in ((S, :circle, C_STOK, "Stokes\\ (tidal)"),
                         (E, :diamond, C_EKMA, "Ekman\\ (rotating)"))
    xs = [c.r for c in cs if haskey(fits, (c.flow, c.r))]
    ys = [fits[(c.flow, c.r)].A2 * c.h / c.δ for c in cs if haskey(fits, (c.flow, c.r))]
    o = sortperm(xs)
    plot!(pb, xs[o], ys[o]; color = lc, lw = 1.8, label = latexstring("\\mathrm{$fl}"))
    scatter!(pb, xs, ys; marker = mk, ms = 7, msw = 1.3,
             mc = [ramp_colour(r) for r in xs], msc = lc, label = "")
end
let all2 = [fits[(c.flow, c.r)].A2 * c.h / c.δ for c in BOTH if haskey(fits, (c.flow, c.r))]
    hline!(pb, [mean(all2)]; color = :grey40, ls = :dash, lw = 1.3,
           label = latexstring(@sprintf("\\mathrm{mean} = %.2f", mean(all2))))
end

f = plot(pa, pb; layout = (1, 2), size = (1240, 560),
         plot_title = L"T = 10\,\mathrm{m}:\ \ \mathrm{TKE}(z) = A_1 e^{-A_2 z/\delta}",
         left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, top_margin = 3Plots.mm)
o = joinpath(FIGDIR, "tke_profiles_T10.png")
savefig(f, o)
say("")
say("wrote $o")

jldopen(joinpath(HERE, "Data", "tke_fits_T10.jld2"), "w") do io
    io["note"] = "A1, A2 per case; by plot_tke_profiles_T10.jl"
    for ((fl, r), v) in fits
        g = @sprintf("%s/r=%.1f", fl, r)
        io["$g/A1"] = v.A1; io["$g/A2"] = v.A2; io["$g/rms"] = v.rms
    end
end
mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "plot_tke_profiles_T10.log"), "w") do io
    foreach(l -> println(io, l), logl)
end
