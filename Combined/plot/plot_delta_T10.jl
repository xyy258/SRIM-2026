# Stage 1: which boundary-layer thickness delta actually tracks the mixed-layer
# height h?
#
# The delta model of stages 2 and 3 needs a thickness, and every candidate has a
# free leading constant that a later fit would absorb. So the constant is not
# what is being tested here — the SHAPE is. A candidate is good if h/delta is
# FLAT in r; where it sits vertically is the constant's job.
#
# Candidates, all built on the measured u_* from reduce_profiles_T10.jl, with
# Omega = omega for the tidal column and f for the rotating one:
#
#   plain        u_*/Omega                              no stratification at all
#   WM 1/4       u_*/Omega (1 + N^2/Omega^2)^(-1/4)     Weatherly & Martin 1978
#   fitted p     u_*/Omega (1 + N^2/Omega^2)^(-p)       p chosen per flow
#   sqrt(Omega N) u_*/sqrt(Omega N)                     Zilitinkevich / Pollard-
#                                                       Rhines-Thompson family,
#                                                       the large-N limit of WM
#   h            the measured height itself             the honest baseline
#
# The flatness measure reported is the ratio max(h/delta)/min(h/delta) across
# each flow's sweep, and the rms of log(h/delta) about its geometric mean. A
# perfect thickness would give 1.00 and 0 %.
#
# EVERY CASE IS FITTED, INCLUDING THE WEAKLY STRATIFIED ONES. That was checked,
# not assumed. The lowN figures show h(t) climbing to z = T = 10 m and sitting
# there, which looked like grounds to drop r = 0.2 and 0.5 from a free-growth
# law. reduce_profiles_T10.jl therefore reports h_pin, the fraction of samples
# with h inside a +/-5 % band on z = T, and it says the opposite:
#
#     Stokes   r     0.2   0.5    1     2     5    10    25    50
#     h_pin         24 %  26 %  14 %  36 %  45 %  94 %   0 %   0 %
#
# The Stokes layer top is near the pycnocline at EVERY r — r = 10 sits on it for
# 94 % of samples against 24 % for r = 0.2 — because h is 8.6 to 11.3 m right
# across the sweep while T = 10 m. Sitting on the pycnocline is the geometry of
# this configuration, not something the low-N cases do. There is no basis for
# treating the new points differently, so they are fitted with the rest and
# h_pin is left as a logged diagnostic.
#
# (Ekman is nowhere near this: h is 6.2 to 22.6 m and h_pin is 0 % at every r.
# A one-sided h >= T test would have called all eight Ekman cases pinned, which
# is why the test is a band and not a threshold.)
#
# USAGE  cd Combined && GKSwstype=100 julia --project=. plot/plot_delta_T10.jl
#        (run reduce_profiles_T10.jl first)

using JLD2, Plots, Printf, Statistics, LaTeXStrings

get!(ENV, "GKSwstype", "100")
default(dpi = 600, fontfamily = "DejaVu Sans")

const HERE   = dirname(@__DIR__)        # scripts live one level down
const CACHE  = joinpath(HERE, "Data", "cache", "profiles_T10.jld2")
const FIGDIR = joinpath(HERE, "figures")
const Ω      = 1e-4                    # omega for Stokes, f for Ekman: equal here
const C_STOK = "#1b3a6b"
const C_EKMA = "#8e1b4e"

logl = String[]
say(s) = (println(s); flush(stdout); push!(logl, s))

function logticks(lo, hi)
    a = floor(Int, log10(lo)); b = ceil(Int, log10(hi))
    vals = Float64[]
    for e in a:b, m in (1, 2, 5)
        v = m * 10.0^e
        lo / 1.05 <= v <= hi * 1.05 && push!(vals, v)
    end
    labs = [v >= 1 ? (v == round(v) ? string(Int(round(v))) : string(v)) :
                     rstrip(rstrip(@sprintf("%.4f", v), '0'), '.') for v in vals]
    return (vals, labs)
end

isfile(CACHE) || error("$CACHE not found — run reduce_profiles_T10.jl first")
S, E = [], []
jldopen(CACHE, "r") do io
    for (fl, dst) in (("stokes", S), ("ekman", E))
        haskey(io, "$fl/ratios") || continue
        for r in io["$fl/ratios"]
            g = @sprintf("%s/r=%.1f", fl, r)
            hp = haskey(io, "$g/h_pin") ? io["$g/h_pin"] : 0.0
            push!(dst, (r = r, N = io["$g/N"], h = io["$g/h"], us = io["$g/us"],
                        us_p90 = io["$g/us_p90"], h_pin = hp))
        end
    end
end
say(@sprintf("%d Stokes and %d Ekman cases from %s", length(S), length(E), basename(CACHE)))

# h_pin is reported, not acted on — see the header. It is here so the next
# person to suspect the low-N cases of being pycnocline-limited can see at a
# glance that the whole Stokes column is, equally.
say("")
say("h_pin, the fraction of samples with h within 5 % of z = T = 10 m:")
for (nm, cs) in (("Stokes", S), ("Ekman ", E))
    say("  $nm  " * join([@sprintf("r=%-5g %3.0f %%", c.r, 100 * c.h_pin) for c in cs], "  "))
end

# ---------------- the candidates ----------------
# Each is a function of a case returning a length. The leading constants are the
# published ones where they exist, so that h/delta can be read against them, but
# they do not affect the flatness that decides the comparison.
plain(c0)   = c -> c0 * c.us / Ω
wm(c0, p)   = c -> c0 * c.us / Ω * (1 + (c.N / Ω)^2)^(-p)
zil(c0)     = c -> c0 * c.us / sqrt(Ω * c.N)

# The exponent that makes h/delta flattest for a given flow.
function best_p(cs)
    best = (Inf, 0.0)
    for p in 0.0:0.005:0.8
        v = [log(c.h / wm(1.0, p)(c)) for c in cs]
        s = sqrt(mean((v .- mean(v)) .^ 2))
        s < best[1] && (best = (s, p))
    end
    return best[2]
end
pS = best_p(S); pE = best_p(E)
say(@sprintf("\nflattest stratification exponent:  Stokes p = %.3f   Ekman p = %.3f   (Weatherly & Martin use 1/4)",
             pS, pE))

function spread(cs, f)
    v = [c.h / f(c) for c in cs]
    lv = log.(v)
    return (ratio = maximum(v) / minimum(v), rms = 100 * sqrt(mean((lv .- mean(lv)) .^ 2)),
            gm = exp(mean(lv)))
end

CANDS_S = [("plain  0.4u_*/\\omega",              plain(0.4),      L"0.4\,u_*/\omega"),
           ("WM 1/4",                             wm(0.4, 0.25),   L"0.4\,u_*/\omega\,(1+N^2/\omega^2)^{-1/4}"),
           (@sprintf("fitted p = %.3f", pS),      wm(0.4, pS),     latexstring(@sprintf("0.4\\,u_*/\\omega\\,(1+N^2/\\omega^2)^{-%.3f}", pS))),
           ("u_*/sqrt(omega N)",                  zil(1.0),        L"u_*/\sqrt{\omega N}")]
CANDS_E = [("plain  1.3u_*/f",                    plain(1.3),      L"1.3\,u_*/f"),
           ("WM 1/4",                             wm(1.3, 0.25),   L"1.3\,u_*/f\,(1+N^2/f^2)^{-1/4}"),
           (@sprintf("fitted p = %.3f", pE),      wm(1.3, pE),     latexstring(@sprintf("1.3\\,u_*/f\\,(1+N^2/f^2)^{-%.3f}", pE))),
           ("u_*/sqrt(f N)",                      zil(1.0),        L"u_*/\sqrt{f N}")]

for (nm, cs, cd) in (("Stokes", S, CANDS_S), ("Ekman ", E, CANDS_E))
    say("")
    say("$nm:  h/delta across the sweep.  flat is what matters, not the level")
    say("  candidate                   h/delta range        max/min   rms about the mean")
    for (lab, f, _) in cd
        v = [c.h / f(c) for c in cs]
        sp = spread(cs, f)
        say(@sprintf("  %-26s %6.3f - %-6.3f %10.2f %12.1f %%",
                     lab, minimum(v), maximum(v), sp.ratio, sp.rms))
    end
end

say("")
say("h and u_* per case")
say("  flow    r        h (m)     u_* (m/s)   u_*(p90)   h_pin")
for (nm, cs) in (("Stokes", S), ("Ekman ", E)), c in cs
    say(@sprintf("  %s %-5g %9.3f %12.4e %11.4e %7.2f",
                 nm, c.r, c.h, c.us, c.us_p90, c.h_pin))
end

# ---------------- the figure ----------------
mkpath(FIGDIR)
const MARK = [:circle, :diamond, :utriangle, :square]
const COLS = ["#1b3a6b", "#c46a1f", "#2e8b57", "#8e1b4e"]

function panel(cs, cd, ttl, lc; legpos = :bottomleft)
    p = plot(xscale = :log10, yscale = :log10, xlabel = L"r = N/\omega = N/f",
             ylabel = L"h/\delta", title = ttl, legend = legpos,
             legendfontsize = 6, foreground_color_legend = nothing)
    xs = [c.r for c in cs]; o = sortperm(xs)
    allv = Float64[]
    for (i, (lab, f, tex)) in enumerate(cd)
        v = [c.h / f(c) for c in cs]
        append!(allv, v)
        sp = spread(cs, f)
        plot!(p, xs[o], v[o]; color = COLS[i], lw = 1.8, marker = MARK[i], ms = 6,
              msw = 1.2, msc = :black,
              label = latexstring(tex.s[2:end-1] * @sprintf(",\\ \\ \\times%.2f", sp.ratio)))
    end
    # A flat candidate is a horizontal line; the guide is each candidate's own
    # geometric mean, so the eye compares shape and not level.
    # ylims is set from the same lo/hi as the ticks. Letting Plots choose it
    # instead clipped the lowest marker and left a tick label stranded above the
    # panel once the sweep widened to r = 0.2.
    lo, hi = minimum(allv) / 1.3, maximum(allv) * 1.3
    # Pad x as well: both axes are log and Plots fits them tight to the data, so
    # the markers at the two ends of the sweep are drawn half outside the panel.
    plot!(p; yticks = logticks(lo, hi), ylims = (lo, hi),
             xlims = (minimum(xs) / 1.3, maximum(xs) * 1.3))
    return p
end

p1 = panel(S, CANDS_S, L"\mathrm{Stokes\ (tidal)}", C_STOK; legpos = :topleft)
p2 = panel(E, CANDS_E, L"\mathrm{Ekman\ (rotating)}", C_EKMA)
f = plot(p1, p2; layout = (1, 2), size = (1240, 560),
         plot_title = L"T = 10\,\mathrm{m}:\ \ \mathrm{does}\ \delta\ \mathrm{track\ the\ layer\ height}\ h?\ \ (\mathrm{flat\ is\ good;\ the\ legend\ gives\ max/min})",
         left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, top_margin = 3Plots.mm)
o = joinpath(FIGDIR, "delta_vs_h_T10.png")
savefig(f, o)
say("")
say("wrote $o")

jldopen(joinpath(HERE, "Data", "cache", "delta_T10.jld2"), "w") do io
    io["note"] = "chosen delta per case, by plot_delta_T10.jl"
    io["p_stokes"] = pS; io["p_ekman"] = pE
    for (fl, cs, cd) in (("stokes", S, CANDS_S), ("ekman", E, CANDS_E))
        io["$fl/ratios"] = [c.r for c in cs]
        for c in cs, (lab, f, _) in cd
            io[@sprintf("%s/r=%.1f/%s", fl, c.r, replace(lab, " " => "_"))] = f(c)
        end
        for c in cs
            io[@sprintf("%s/r=%.1f/best", fl, c.r)] = (fl == "stokes" ? wm(0.4, pS) : wm(1.3, pE))(c)
            io[@sprintf("%s/r=%.1f/plain", fl, c.r)] = (fl == "stokes" ? plain(0.4) : plain(1.3))(c)
        end
    end
end
mkpath(joinpath(HERE, "logs"))
open(joinpath(HERE, "logs", "plot_delta_T10.log"), "w") do io
    foreach(l -> println(io, l), logl)
end
