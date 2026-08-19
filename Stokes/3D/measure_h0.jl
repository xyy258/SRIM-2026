ENV["GKSwstype"] = "100"

using Oceananigans, JLD2, Printf, Statistics

# h0 — the turbulent layer height, measured from a *_moments.jld2 file.
#
# STAGE 0 OF THE MOMENTS SWEEP EXISTS TO PRODUCE THIS NUMBER. The T values of
# stages 1–3 are only meaningful relative to how far the turbulence actually
# reaches: a pycnocline placed above the turbulent layer never gets eroded, so
# nothing happens and the case measures nothing. The brief's provisional
# T ∈ {2, 3, 5, 8} m should be replaced by roughly {0.4, 0.7, 1.2, 2.0} × h0 once
# h0 is known.
#
# DEFINITION: at the phase of the last full period where near-wall TKE peaks, h0
# is the first height ABOVE that peak at which TKE has fallen to TKE_FRAC (default
# 1 %) of it. That is a different quantity from the mixed-layer height h used
# elsewhere (normalised buoyancy gradient reaching 0.1, Figure5.jl's
# first_crossing) — h0 measures where the TURBULENCE reaches, h where the
# STRATIFICATION has been erased. Both are reported so the two can be compared.
#
#   julia --project=. measure_h0.jl outputs/<tag>/TidalBL3D_<tag>_moments.jld2
#   julia --project=. measure_h0.jl <tag>          # resolves under OUT_ROOT

const TKE_FRAC = parse(Float64, get(ENV, "TKE_FRAC", "0.01"))
const ω = 1e-4
const T_tide = 2π / ω
const U₀ = 0.04
const outroot = get(ENV, "OUT_ROOT", "outputs")

isempty(ARGS) && error("usage: julia --project=. measure_h0.jl <moments file | case tag>")
const arg = ARGS[1]
const fname = endswith(arg, ".jld2") ? arg :
              joinpath(@__DIR__, outroot, arg, "TidalBL3D_" * arg * "_moments.jld2")
isfile(fname) || error("no moments file at $fname — run the case with MOMENTS=1 first")

ts(v) = FieldTimeSeries(fname, v)
tsB = ts("B")
times = tsB.times
nt = length(times)
zc = collect(znodes(tsB))
zf = collect(znodes(ts("wb")))

grab(v) = (T = ts(v); a = zeros(length(interior(T[1])), nt);
           for n in 1:nt; a[:, n] .= vec(interior(T[n])); end; a)
U, V = grab("U"), grab("V")
uu, vv, ww = grab("uu"), grab("vv"), grab("ww")
G = grab("dBdz")

# Instantaneous decomposition, per sample — never time-average first. See the
# ordering-trap note in Moments.jl: doing it the other way round folds the
# variance of the tidal mean flow into "TKE".
TKE = 0.5 .* ((uu .- U .^ 2) .+ (vv .- V .^ 2) .+ ww)

# Last full period only: earlier periods still carry the restart transient.
t_end = times[end]
last_period = findall(n -> times[n] >= t_end - T_tide, 1:nt)
length(last_period) < 3 && (last_period = collect(1:nt))

# Peak phase = the sample in that period with the largest NEAR-WALL TKE.
#
# The search is confined to z ≤ H_PEAK. Taking the maximum over the whole column
# instead lets a spurious maximum aloft win — decaying initial noise, radiated
# internal waves, or the sponge — and then h0 is measured downward from the wrong
# level entirely. The height the peak was found at is printed so a bad H_PEAK is
# visible rather than silent.
const H_PEAK = parse(Float64, get(ENV, "H_PEAK", "10.0"))
const k_search = findall(z -> z <= H_PEAK, zc)
isempty(k_search) && error("H_PEAK = $H_PEAK m is below the first grid centre")

peak_of(n) = maximum(view(TKE, k_search, n))
npk = last_period[argmax(peak_of.(last_period))]
prof = TKE[:, npk]
kpk = k_search[argmax(view(prof, k_search))]
TKE_pk = prof[kpk]

# First height above the near-wall peak where TKE has fallen to TKE_FRAC of it.
# Descending crossing, so first_crossing (which looks for ascending ones) does not
# apply; linear interpolation in z, same convention.
function first_drop_below(z, f, level, kstart)
    for i in kstart:length(f)-1
        if f[i] >= level > f[i+1]
            return z[i] + (level - f[i]) * (z[i+1] - z[i]) / (f[i+1] - f[i])
        end
    end
    return NaN
end
# FALLBACK LADDER. A single fixed fraction is brittle: if TKE has not fallen that
# far by the top of the analysed column the answer is NaN, and NaN blocks stages
# 1-3 outright — which turns the whole first submission into a wasted allocation.
# So try the requested fraction, then progressively looser ones, and report which
# one actually produced the number rather than silently substituting it.
const FRAC_LADDER = [TKE_FRAC; filter(>(TKE_FRAC), [0.02, 0.05, 0.10, 0.20])]
frac_used = NaN
h0 = NaN
for f in FRAC_LADDER
    global h0, frac_used
    h0 = first_drop_below(zc, prof, f * TKE_pk, kpk)
    isnan(h0) || (frac_used = f; break)
end

# The mixed-layer height at the same instant, for comparison. N²_ref is recovered
# from the far field rather than passed in, so this script needs no case knobs.
N²_ref = maximum(filter(isfinite, view(G, :, npk)))
function first_crossing(z, fv, level)
    for i in 1:length(fv)-1
        fv[i] < level <= fv[i+1] &&
            return z[i] + (level - fv[i]) * (z[i+1] - z[i]) / (fv[i+1] - fv[i])
    end
    return NaN
end
h_ml = N²_ref > 0 ? first_crossing(zf, view(G, :, npk) ./ N²_ref, 0.1) : NaN

println("\n", "="^72)
@printf("h0 MEASUREMENT — %s\n", basename(fname))
println("="^72)
@printf("  samples %d over %.2f periods; peak phase at sample %d (t = %.2f periods, ωt mod 2π = %.2f)\n",
        nt, times[end] / T_tide, npk, times[npk] / T_tide, mod(ω * times[npk], 2π))
@printf("  near-wall TKE peak = %.4e m² s⁻² (= %.2e U₀²) at z = %.4f m (searched z ≤ %.1f m)\n",
        TKE_pk, TKE_pk / U₀^2, zc[kpk], H_PEAK)
zc[kpk] > 0.5H_PEAK &&
    @warn @sprintf("the TKE peak sits at z = %.2f m, over half of H_PEAK = %.1f m — that is probably not a wall peak. Check the profile before trusting h0.",
                   zc[kpk], H_PEAK)
@printf("  mixed-layer height h (∂B/∂z reaching 0.1 N²_ref) = %.3f m\n", h_ml)
println()
@printf("  >>>  h0 (TKE down to %.0f %% of peak)  =  %.3f m  <<<\n",
        100 * (isnan(frac_used) ? TKE_FRAC : frac_used), h0)
println()
if isnan(h0)
    println("  h0 is NaN: TKE never fell even to $(100*last(FRAC_LADDER)) % of its near-wall")
    println("  peak below the top of the analysed column. The run is either far too short")
    println("  or never became turbulent — check the profile before running stages 1-3.")
    @printf("  Falling back on the mixed-layer height instead: h = %.3f m, i.e.\n", h_ml)
    isnan(h_ml) || @printf("      T_VALUES=\"%.1f %.1f %.1f %.1f\"   (UNVERIFIED — h, not h0)\n",
                           0.4h_ml, 0.7h_ml, 1.2h_ml, 2.0h_ml)
else
    frac_used > TKE_FRAC &&
        @printf("  NOTE: %.0f %% of peak was never reached; this is the %.0f %% level instead, so h0 is an UPPER bound on the 1 %% height.\n",
                100TKE_FRAC, 100frac_used)
    @printf("  RE-PARAMETERISE STAGES 1-3 ON THIS. Suggested T = {0.4, 0.7, 1.2, 2.0} x h0:\n")
    @printf("      T_VALUES=\"%.1f %.1f %.1f %.1f\"\n",
            0.4h0, 0.7h0, 1.2h0, 2.0h0)
    @printf("  (the brief's provisional T = {2, 3, 5, 8} m corresponds to h0 = %.1f m)\n", 5.0 / 1.2)
end
println("="^72)

# Machine-readable, for the sweep driver.
open(joinpath(dirname(fname), "h0.txt"), "w") do io
    @printf(io, "%.6f\n", h0)
end
