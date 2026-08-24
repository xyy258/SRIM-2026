#!/usr/bin/env julia
# One table with the three definitions of h side by side.
#
# Reads the mixing_<tag>_{hcross,hgrad,hflux}.jld2 files written by
# run_h_definitions.sh and prints, for each case and each definition:
#
#   h            where the definition puts the mixed-layer top
#   sd           how much it moves between samples (h is used as a length, so a
#                jittery h is a jittery length, not just a noisy plot)
#   amb          % of samples where that peak is the tallest of several bumps
#   TKE, K_T     what is being sampled there
#   K_sgs/K_T    how much of that K_T comes from the closure rather than the flow
#   slope, r     the panel (d) exponent and how well the power law holds
#
# The exponent is what the study is after: 1/2 means K_T = c√TKE·l, 1 means
# Γ·TKE/N. A definition that puts h where there is no turbulence cannot measure
# it, which is what the r column shows.
#
#   julia --project=. compare_h_definitions.jl        → h_defs/comparison.txt

using JLD2, Printf, Statistics

const HERE   = @__DIR__
const ω      = 1e-4
const T_tide = 2π / ω
const SKIP   = parse(Float64, get(ENV, "SKIP_PERIODS", "3"))
const DEFS   = ["_hcross" => "0.1 crossing", "_hgrad" => "peak dB/dz", "_hflux" => "peak −F_b"]
const outroot = joinpath(HERE, get(ENV, "OUT_ROOT", "outputs"))

function read_case(tag, sfx)
    f = joinpath(outroot, tag, "mixing_$(tag)$(sfx).jld2")
    isfile(f) || return nothing
    jldopen(f, "r") do io
        keep = io["times"] .>= SKIP * T_tide
        h    = io["h"][keep]
        nup  = haskey(io, "h_nup") ? io["h_nup"][keep] : Int[]
        rat  = io["K_sgs_over_K_T"]
        zf   = io["z_face"]
        # K_sgs/K_T interpolated to z = h, from the same sample as the K_T column
        ksg = Float64[]
        for (i, n) in enumerate(findall(keep))
            isfinite(h[i]) || continue
            k = searchsortedlast(zf, h[i])
            (k < 1 || k >= length(zf)) && continue
            v = rat[k, n]
            isfinite(v) && push!(ksg, v)
        end
        med(v) = (g = filter(isfinite, v); isempty(g) ? NaN : median(g))
        (; h = filter(isfinite, h),
           amb = isempty(nup) ? NaN : 100count(>(1), nup) / length(nup),
           TKE = med(io["TKE_at_h"][keep]), K = med(io["K_at_h"][keep]),
           ksgs = med(ksg), slope = io["slope"], r = io["slope_r"], n = io["slope_n"],
           T = io["T"], s = io["n_over_omega"])
    end
end

tags = String[]
for d in sort(readdir(outroot; join = true))
    isdir(d) || continue
    any(f -> startswith(basename(f), "mixing_") && endswith(f, "_hcross.jld2"),
        readdir(d; join = true)) && push!(tags, basename(d))
end
sort!(tags, by = t -> (parse(Float64, match(r"_T(\d+)_", t)[1]),
                       parse(Float64, replace(match(r"sqrtRi(.+)$", t)[1], "p" => "."))))

open(joinpath(HERE, "h_defs", "comparison.txt"), "w") do io
    for out in (io, stdout)
        println(out, "="^108)
        @printf(out, "MIXED-LAYER HEIGHT: THREE DEFINITIONS  (samples past %g periods)\n", SKIP)
        println(out, "="^108)
        @printf(out, "%-16s %-13s %13s %5s %10s %10s %8s %8s %6s\n",
                "case", "definition", "h ± sd (m)", "amb%", "TKE at h", "K_T at h",
                "Ksgs/K", "slope", "r")
        for tag in tags
            first = true
            for (sfx, name) in DEFS
                c = read_case(tag, sfx)
                c === nothing && continue
                @printf(out, "%-16s %-13s %6.2f ±%5.2f %5.0f %10.2e %10.2e %8.2f %+8.2f %6.2f\n",
                        first ? tag : "", name, mean(c.h), std(c.h), c.amb,
                        c.TKE, c.K, c.ksgs, c.slope, c.r)
                first = false
            end
        end
        println(out, "="^108)
        println(out, "amb% is the fraction of samples where the profile being peaked has more than one")
        println(out, "bump reaching half its maximum — 0 means one clean feature. The crossing is not a")
        println(out, "peak at all, so its 0 means 'does not apply' rather than 'unambiguous'.")
        println(out, "slope: 1/2 ⇒ K_T = c·√TKE·l ;  1 ⇒ K_T = Γ·TKE/N.  Read r first — a slope from")
        println(out, "an r of 0.3 is not an exponent, whatever it says.")
        println(out, "="^108)
    end
end
