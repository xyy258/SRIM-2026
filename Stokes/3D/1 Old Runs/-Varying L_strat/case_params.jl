# Shared case parameters for the Gayen, Sarkar & Taylor (2010) reproduction.
# Every script (simulation, animation, profiles, figures) includes this file and
# selects the case from the command line:  julia Tidal3D.jl Ri500
#
# The paper defines its cases by Ri = N∞² / ω² ∈ {0, 500, 2500} at fixed
# Re_s = U₀ δ_s / ν = 1790.  Everything else below is derived from that.
#
# Declared `const` so closures that capture these (forcing, sponge masks)
# compile to fast kernels.

using Printf

const case = isempty(ARGS) ? "Ri0" : ARGS[1]

const Ri_targets = Dict("Ri0" => 0.0, "Ri500" => 500.0, "Ri2500" => 2500.0)
haskey(Ri_targets, case) ||
    error("Unknown case \"$case\" — use one of: Ri0, Ri500, Ri2500")

const Ri = Ri_targets[case]

# ---------------- Physical parameters ----------------
# These are the paper's own dimensional values, quoted verbatim in their §2.4:
# "Take the viscosity of water to be ν = 10⁻⁶ m² s⁻¹, amplitude of current
#  velocity as U₀ = 1.5 cm s⁻¹ and ω = 1.407 × 10⁻⁴ rad s⁻¹ ... The Stokes
#  boundary layer thickness is δ_s = √(2ν/ω) = 0.119 m. The Stokes Reynolds
#  number becomes Re_s = U₀ δ_s / ν = 1788."
#
# OLD: U₀ = 0.05 with ν = 1.109e-5 — a rescaled realization that hit the same
# Re_s = 1790 but with δ_s = 0.397 m, i.e. the paper's flow in different units.
# Dynamically equivalent; replaced so every dimensional number below can be
# checked directly against the paper. NOTE: U₀ = 0.15 (which is what "1.5 cm/s"
# looks like if read as 15 cm/s) would give Re_s = 5370 — a different regime,
# not the one figures 4 and 5 are drawn from.
const ω  = 1e-4        # M2 tidal frequency (s⁻¹) to match coriolis parameter in colleague's case
const U₀ = 0.04               # tidal velocity amplitude (m s⁻¹) = 4 cm s⁻¹
const ν  = 1.0e-6              # molecular viscosity of water (m² s⁻¹)
const Pr = 10                 # molecular Prandtl number 
const κ  = ν / Pr              # molecular diffusivity

const δ      = sqrt(2ν / ω)    # laminar Stokes layer thickness ≈ 
const Re_s   = U₀ * δ / ν      # ≈ 1788
const T_tide = 2π / ω

# Active background stratification. For Ri = 0 the paper still carries a
# temperature field — it is simply passive (no buoyancy feedback) and keeps the
# same background gradient. We reproduce that by giving the b tracer a
# reference gradient and switching the buoyancy term off in Tidal3D.jl, so the
# Ri = 0 thermal panels of figures 4 and 5 are meaningful rather than blank.
const N²     = Ri * ω^2                    # buoyancy frequency² actually felt
const N²_ref = Ri > 0 ? N² : ω^2           # far-field gradient carried by b
const passive_scalar = Ri == 0

# ---------------- Shape of the background stratification ----------------
# OLD: uniform background, N²_bg(z) = N²_ref everywhere, i.e. b_bg(z) = N²_ref z.
# That is the paper's own set-up and is what the "Centered - Linear" run used.
#
# NEW: the stratification vanishes at the seabed and recovers exponentially to
# the paper's far-field value over a scale L_strat:
#
#     N²_bg(z) = N²_ref [1 − exp(−z/L)]
#     b_bg(z)  = N²_ref [z + L (exp(−z/L) − 1)]      (b_bg(0) = 0, b_bg′(0) = 0)
#
# The far field is untouched, so Ri = N∞²/ω² keeps the paper's meaning and every
# quantity normalized by N∞² remains directly comparable to the linear run.
# L = 10 δ_s is comparable to the mixed-layer height h_m ≈ 10–15 δ_s reached in
# the linear runs, so the altered region is where the thermocline actually sits.
#
# Note for the diagnostics: at t = 0 the normalized gradient ∂b_bg/∂z / N∞² is
# already below the paper's h_m threshold of 0.1 for z < 0.105 L ≈ 1.05 δ_s.
# That offset is small against h_m, so h_m keeps the paper's literal definition;
# the integral diagnostics in Tidal3Dprofiles.jl are instead measured against
# N²_bg(z) so they still start from zero.
# OLD: const L_strat = 10δ  (≈ 1.4 m, tied to the Stokes thickness).
# NEW: L_strat is set in METRES from the environment so the sweep {2,4,6,8} m can
# be driven externally (L_STRAT_M). The Stokes thickness δ_s ≈ 0.14 m is deemed
# too small a length scale for this study (matching a colleague's Ekman set-up
# where δ = u*/f ≈ several m); L is therefore given directly in metres.
const L_m     = parse(Float64, get(ENV, "L_STRAT_M", "4.0"))
const L_strat = L_m

@inline N²_background(z) = N²_ref * (1 - exp(-z / L_strat))
@inline b_background(z)  = N²_ref * (z + L_strat * (exp(-z / L_strat) - 1))

# ---------------- Domain ----------------
# Paper §2.4: lx = 50 δ_s, ly = 25 δ_s, test section lz = 70 δ_s with the
# sponge spanning 70 δ_s – 90 δ_s, i.e. 5.95 × 2.98 × 10.73 m here.
const Lx      = 50δ            # streamwise  ≈ 5.960 m
const Ly      = 25δ            # spanwise    ≈ 2.980 m
const Lz_test = 70δ            # top of the analysed test section ≈ 8.344 m
const Lz      = 90δ            # full domain ≈ 10.728 m (sponge above Lz_test)

# ---------------- Run length ----------------
# Target; the overnight driver (run_night.sh) may stop a case earlier at a
# wall-clock deadline. Every half period is a U∞ = 0 snapshot, so cutting a run
# at any half-period boundary still leaves a phase-consistent restart state.
# The paper's figure 5 late time is t = ωt_d = 50, i.e. 7.96 tidal periods, so
# 8 completed periods already covers the range those figures span.
const n_periods = parse(Int, get(ENV, "N_PERIODS", "12"))

# ---------------- Output naming ----------------
# Each run is tagged by both its stratification scale L (metres) and its Ri case,
# e.g. L4_Ri500, so the four-value L_strat sweep does not collide on disk.
const Lint     = isinteger(L_m) ? string(Int(L_m)) : replace(string(L_m), "." => "p")
const casetag  = "L" * Lint * "_" * case
const outdir   = "output_" * casetag
const filename = joinpath(outdir, "TidalBL3D_" * casetag)
mkpath(outdir)

@info @sprintf("Case %s (L = %g m): Ri = %g, N² = %.4g s⁻², δ_s = %.4f m, Re_s = %.0f",
               casetag, L_m, Ri, N², δ, Re_s)
@info @sprintf("Domain %.3f × %.3f × %.3f m = %.0f × %.0f × %.0f δ_s, %d periods%s",
               Lx, Ly, Lz, Lx/δ, Ly/δ, Lz/δ, n_periods,
               passive_scalar ? " (b is a passive scalar)" : "")
@info @sprintf("Background: exponential, N²_bg = N∞²[1−exp(−z/L)], L = %.3f m = %.0f δ_s (N²_bg/N∞² = %.3f at z = δ_s, %.3f at z = 10 δ_s)",
               L_strat, L_strat/δ, N²_background(δ)/N²_ref, N²_background(10δ)/N²_ref)
