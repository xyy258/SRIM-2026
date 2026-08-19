ENV["GKSwstype"] = "100"

using Oceananigans, JLD2, Printf, Statistics

# h0 — the turbulent layer height, measured from a Moments.jld2 file.
#
# A COPY of Stokes/3D/measure_h0.jl with the tidal names bound to the Ekman ones
# (f₀ for ω, the inertial period for T_tide, U∞ for U₀) and the default file
# resolution pointed at this folder's Data/. The measurement itself is unchanged.
#
# WHAT IT IS FOR HERE. The Stokes sweep used h0 to choose T before running a grid
# of cases. This folder runs ONE case at a T that is already fixed at 20 m, so h0
# is not steering anything — it is a diagnostic. It answers the one question that
# decides whether the run measured anything at all: did the turbulent layer
# actually reach the pycnocline at z = T? If h0 comes back well below 20 m the
# buoyancy flux there is negligible, Δb never changes, and K_T is being formed
# from noise. Read it before the VERIFICATION block, not after.
#
# DEFINITION: at the phase of the last full period where near-wall TKE peaks, h0
# is the first height ABOVE that peak at which TKE has fallen to TKE_FRAC (default
# 1 %) of it. That is a different quantity from the mixed-layer height h used
# elsewhere (normalised buoyancy gradient reaching 0.1, Figure5.jl's
# first_crossing) — h0 measures where the TURBULENCE reaches, h where the
# STRATIFICATION has been erased. Both are reported so the two can be compared.
#
#   julia --project=<root> measure_h0.jl                      # the default case
#   julia --project=<root> measure_h0.jl "Data/r=1.0, T=20.0/Moments.jld2"

const TKE_FRAC = parse(Float64, get(ENV, "TKE_FRAC", "0.01"))
const f₀ = 1e-4
const ω = f₀                  # alias: the inertial frequency, so the body below
const T_tide = 2π / f₀        # alias: the inertial period, 62832 s
const U₀ = 0.04               # alias: U∞
const dataroot = get(ENV, "DATA_ROOT", joinpath(@__DIR__, "Data"))
const casename = get(ENV, "CASE_NAME",
                     @sprintf("r=%.1f, T=%.1f",
                              parse(Float64, get(ENV, "R", "1")),
                              parse(Float64, get(ENV, "T_STRAT", "20"))))

const arg = isempty(ARGS) ? "" : ARGS[1]
const fname = isempty(arg)          ? joinpath(dataroot, casename, "Moments.jld2") :
              endswith(arg, ".jld2") ? arg :
                                       joinpath(dataroot, arg, "Moments.jld2")
isfile(fname) || error("no moments file at $fname — run Ekman3D.jl with MOMENTS=1 first")

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

# Last full inertial period only: earlier ones still carry the spin-up transient,
# which here is the whole transition from a laminar start to a turbulent layer.
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
# 30 m, not the Stokes 10: this domain is 100 m deep and the pycnocline under
# study sits at 20 m, so the wall peak has to be searched for over a taller column.
const H_PEAK = parse(Float64, get(ENV, "H_PEAK", "30.0"))
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
    T_case = parse(Float64, get(ENV, "T_STRAT", "20"))
    ratio  = T_case / h0
    @printf("  T / h0 = %.1f / %.1f = %.2f\n", T_case, h0, ratio)
    if ratio > 2
        println("  THE PYCNOCLINE IS ABOVE THE TURBULENT LAYER. The flux at z = T is negligible,")
        println("  Δb never changes, and K_T is being formed from noise. This run measures")
        @printf("  nothing about mixing at the interface; T ≈ %.0f m would.\n", h0)
    elseif ratio < 0.3
        println("  THE PYCNOCLINE IS DEEP INSIDE THE TURBULENT LAYER — wiped out early, so Δb")
        println("  collapses and K_T = −∫F dz/Δb is ill-conditioned. Check panel (c) carefully.")
    else
        println("  T sits inside the usable window (0.3 h0 to 2 h0): the turbulent layer")
        println("  reaches the pycnocline, and K_T at the interface is a real measurement.")
    end
end
println("="^72)

# Machine-readable, for the sweep driver.
open(joinpath(dirname(fname), "h0.txt"), "w") do io
    @printf(io, "%.6f\n", h0)
end
