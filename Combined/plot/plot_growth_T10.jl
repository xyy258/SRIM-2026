# Mixed-layer growth predicted by the closure:
#   dh/dt = 2 K_T/h,   K_T = K_T* TKE(h)/N,
#   TKE(h) = A1 u_*^2 exp(-A2 h/delta)          (plot_tke_fits_T10.jl, pooled)
#   K_T*   = A sqrt(Ri)/(1 + sqrt(Ri))           (the two K_T* figures)
# S(h) has no model, so each run's measured sqrt(Ri) at h is held fixed as h
# grows. K_T* is then constant and, with x = A2 h/delta, h(0) = 0,
#   t(h) = N delta^2 [(x - 1) e^x + 1] / (2 K_T* A1 u_*^2 A2^2).
# Both flows: r < 25 (N/omega, N/f = 25 and 50 dropped everywhere). Times in years.
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot/plot_growth_T10.jl
#        (needs Data/cache/profiles_T10.jld2 and delta_T10.jld2)

using JLD2, Plots, Printf, LaTeXStrings

get!(ENV, "GKSwstype", "100")
default(dpi = 600, fontfamily = "DejaVu Sans", guidefontsize = 22, tickfontsize = 17,
        legendfontsize = 15, titlefontsize = 20)

const HERE  = dirname(@__DIR__)
include(joinpath(HERE, "sweep.jl"))
const CACHE = joinpath(HERE, "Data", "cache", "profiles_T10.jld2")
const DFILE = joinpath(HERE, "Data", "cache", "delta_T10.jld2")
const HS    = [10, 20, 30, 40, 50, 60]      # heights for the table (m)
const YR    = 365.25 * 86400                # one year (s)
const A4    = (1169, 827)              # A4 landscape at 100 px/in, for the side-by-side figure

# Fitted constants and measured sqrt(Ri) at h, from logs/plot_tke_fits_T10.log,
# logs/plot_KTstar_Ri_stokes_T10.log and logs/plot_KTstar_Ri_ekman_T10.log. Both
# K_T* fits are at z = h on the BACKGROUND N, the same N used in K_T = K_T* TKE/N
# here. (The Ekman z-scan uses the local N, which at z = h is sqrt(0.1) of the
# background by the definition of h, so it must not be mixed with this N.)
# Rerun those scripts and update these if they change.
const FL = (
    (name = "Stokes", key = "stokes", sym = "N/ω", tend = 8 * 2π / 1e-4, A1 = 3.422, A2 = 1.460, A = 0.4025,
     sRi = Dict(0.2 => 0.0655, 0.5 => 0.1805, 1 => 0.6931, 2 => 1.5348, 5 => 3.3078,
                10 => 4.4774)),
    (name = "Ekman", key = "ekman", sym = "N/f", tend = 12.73 * 2π / 1e-4, A1 = 1.718, A2 = 5.680, A = 0.2998,
     sRi = Dict(0.2 => 0.6612, 0.5 => 1.2678, 1 => 2.9051, 2 => 3.5808, 5 => 3.3083, 10 => 3.4607)))

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))

growth(c, f) = h -> (x = f.A2 * h / c.δ;
    c.N * c.δ^2 * ((x - 1) * exp(x) + 1) / (2 * c.Ks * f.A1 * c.us^2 * f.A2^2))

# h reached at time t, by bisection on the monotone t(h).
hat(T, t) = (lo = 0.0; hi = 200.0; for _ in 1:60; m = (lo + hi) / 2; T(m) < t ? (lo = m) : (hi = m); end; lo)

texnum(t) = (e = floor(Int, log10(t)); @sprintf("\$%.1f\\times10^{%d}\$", t / 10.0^e, e))

tex = String[]; ps = []
for f in FL
    cs = jldopen(CACHE, "r") do io
        [(r = r, N = io[@sprintf("%s/r=%.1f/N", f.key, r)], us = io[@sprintf("%s/r=%.1f/us", f.key, r)],
          h = io[@sprintf("%s/r=%.1f/h", f.key, r)],
          δ = jldopen(d -> d[@sprintf("%s/r=%.1f/best", f.key, r)], DFILE, "r"),
          Ks = f.A * f.sRi[r] / (1 + f.sRi[r]))
         for r in io["$(f.key)/ratios"] if haskey(f.sRi, r)]
    end
    say("\n$(f.name): t(h) in years")
    say("  r      N (1/s)   delta (m)   sqrt(Ri)   K_T*    h_sim  h_model at run end (m)  " * join((@sprintf("%9d m", h) for h in HS)))
    p = plot(xscale = :log10, xlabel = L"t\ \ (\mathrm{years})", ylabel = L"h\ \ (\mathrm{m})",
             ylims = (0, 65), xlims = (1e-5, 1e8), yticks = 0:10:60, xticks = exp10.(-4:2:6),
             minorgrid = true, minorticks = 2, legend = :bottomright, foreground_color_legend = nothing,
             title = latexstring("\\mathrm{$(f.name)}:\\ \\ dh/dt = 2K_T/h,\\ \\ \\sqrt{Ri}\\ \\mathrm{fixed\\ at\\ its\\ value\\ at}\\ h"))
    push!(tex, "\\begin{table}[h]\n\\centering\n\\caption{$(f.name): time (years) for the mixed layer to reach height \$h\$, " *
               "from \$dh/dt = 2K_T/h\$ with \$\\sqrt{Ri}\$ held at its measured value at \$h\$.}\n" *
               "\\begin{tabular}{r" * "r"^length(HS) * "}\n\\hline\n\$$(f.sym == "N/ω" ? "N/\\omega" : "N/f")\$ & " *
               join(("\$h = $(h)\$\\,m" for h in HS), " & ") * " \\\\\n\\hline")
    for c in cs
        T = growth(c, f); ts = T.(HS) ./ YR
        say(@sprintf("  %-5g %9.1e %9.2f %10.3f %7.3f %7.2f %9.2f            ", c.r, c.N, c.δ, f.sRi[c.r], c.Ks, c.h, hat(T, f.tend)) *
            join((@sprintf("%11.3e", t) for t in ts)))
        hh = range(0.05, 65; length = 400)
        plot!(p, T.(hh) ./ YR, hh; color = ramp_colour(c.r), lw = 2, label = @sprintf("%s = %g", f.sym, c.r))
        push!(tex, @sprintf("%g & ", c.r) * join(texnum.(ts), " & ") * " \\\\")
    end
    vline!(p, [f.tend / YR]; color = :grey, ls = :dash, lw = 1.5, label = "end of LES run")
    push!(tex, "\\hline\n\\end{tabular}\n\\end{table}\n")
    fig = plot(p; size = (900, 700), left_margin = 6Plots.mm, bottom_margin = 6Plots.mm)
    o = joinpath(HERE, "figures", "growth_$(f.key)_T10.png"); savefig(fig, o); say("wrote $o")
    push!(ps, plot!(p; title = f.name))
end
o = joinpath(HERE, "figures", "growth_pair_T10.png")
savefig(plot(ps...; layout = (1, 2), size = A4, left_margin = 6Plots.mm, bottom_margin = 6Plots.mm,
             plot_title = L"dh/dt = 2K_T/h,\ \ \sqrt{Ri}\ \mathrm{fixed\ at\ its\ value\ at}\ h"), o); say("wrote $o")

mkpath(joinpath(HERE, "tables"))
o = joinpath(HERE, "tables", "growth_times_T10.tex"); write(o, join(tex, "\n")); say("wrote $o")
mkpath(joinpath(HERE, "logs"))
write(joinpath(HERE, "logs", "plot_growth_T10.log"), join(logl, "\n") * "\n")
