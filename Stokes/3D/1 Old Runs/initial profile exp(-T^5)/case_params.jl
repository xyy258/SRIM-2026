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

# ---------------- Background stratification: LINEAR far field ----------------
# This run reverts to the uniform-N² background of the "Centered - Linear" set-up
# (the paper's own): N²_bg(z) = N²_ref everywhere, b_bg(z) = N²_ref z. It is used
# only for the sustained forcing — the top gradient BC and the sponge target —
# NOT for the initial condition, which now carries a pre-formed bottom mixed
# layer (below). The previous exponential background N²_bg = N∞²[1−exp(−z/L)],
# with its parameter L_strat, is retained in "-Varying L_strat/".
@inline N²_background(z) = N²_ref
@inline b_background(z)  = N²_ref * z

# ---------------- Initial buoyancy profile: bottom mixed layer of depth T -------
# The velocities restart from a turbulent snapshot; the buoyancy is initialized
# with a mixed layer already carved into an otherwise-linear stratification:
#
#     bᵢ(z) = N²_ref z [1 − exp(−0.2 (z/T)^5)]
#
# Near the wall the bracket → 0 (well mixed, ∂b/∂z → 0); above z ≈ T it → 1 so
# the profile recovers the linear far field N²_ref z. The gradient overshoots to
# ≈ 2.4 N²_ref just above z = T (a sharp initial pycnocline), then relaxes.
#
# T is the height of the peak departure from linear: with u = z/T the deficit
# N²_ref z exp(−0.2 u⁵) ∝ u e^{−0.2u⁵} is maximized at u = 1, i.e. exactly z = T.
# T is chosen to match the mixed-layer height h_m ≈ 3–6 m reached in the earlier
# exponential runs (see -Varying L_strat/ figure 4); swept {3, 4.5, 6} m here.
# Set in METRES from the environment (T_STRAT_M) so the sweep is driven
# externally, replacing the old L_STRAT_M knob.
const T_m     = parse(Float64, get(ENV, "T_STRAT_M", "4.0"))
const T_strat = T_m

@inline b_initial(z) = N²_ref * z * (1 - exp(-0.2 * (z / T_strat)^5))

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
# Each run is tagged by its mixed-layer depth T (metres) and its Ri case, e.g.
# T4p5_Ri500, so the three-value T sweep does not collide on disk (a fractional
# T like 4.5 becomes "4p5").
const Tint     = isinteger(T_m) ? string(Int(T_m)) : replace(string(T_m), "." => "p")
const casetag  = "T" * Tint * "_" * case
const outdir   = "output_" * casetag
const filename = joinpath(outdir, "TidalBL3D_" * casetag)
mkpath(outdir)

@info @sprintf("Case %s (T = %g m): Ri = %g, N² = %.4g s⁻², δ_s = %.4f m, Re_s = %.0f",
               casetag, T_m, Ri, N², δ, Re_s)
@info @sprintf("Domain %.3f × %.3f × %.3f m = %.0f × %.0f × %.0f δ_s, %d periods%s",
               Lx, Ly, Lz, Lx/δ, Ly/δ, Lz/δ, n_periods,
               passive_scalar ? " (b is a passive scalar)" : "")
@info @sprintf("Background: linear N²_bg = N∞² (uniform). Initial b: mixed layer of depth T = %.2f m = %.1f δ_s carved into N∞²z (bᵢ/(N∞²z) = %.3f at z = δ_s, %.3f at z = T)",
               T_strat, T_strat/δ, b_initial(δ)/(N²_ref*δ), b_initial(T_strat)/(N²_ref*T_strat))
