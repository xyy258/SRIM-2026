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
const ω  = 1.4075235e-4        # M2 tidal frequency (s⁻¹), period ≈ 12.4 h
const U₀ = 0.015               # tidal velocity amplitude (m s⁻¹) = 1.5 cm s⁻¹
const ν  = 1.0e-6              # molecular viscosity of water (m² s⁻¹)
const Pr = 0.7                 # molecular Prandtl number (paper value)
const κ  = ν / Pr              # molecular diffusivity

const δ      = sqrt(2ν / ω)    # laminar Stokes layer thickness ≈ 0.1192 m
const Re_s   = U₀ * δ / ν      # ≈ 1788
const T_tide = 2π / ω

# Active background stratification. For Ri = 0 the paper still carries a
# temperature field — it is simply passive (no buoyancy feedback) and keeps the
# same background gradient. We reproduce that by giving the b tracer a
# reference gradient and switching the buoyancy term off in Tidal3D.jl, so the
# Ri = 0 thermal panels of figures 4 and 5 are meaningful rather than blank.
const N²     = Ri * ω^2                    # buoyancy frequency² actually felt
const N²_ref = Ri > 0 ? N² : ω^2           # background gradient carried by b
const passive_scalar = Ri == 0

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
const outdir   = "output_" * case
const filename = joinpath(outdir, "TidalBL3D_" * case)
mkpath(outdir)

@info @sprintf("Case %s: Ri = %g, N² = %.4g s⁻², δ_s = %.4f m, Re_s = %.0f",
               case, Ri, N², δ, Re_s)
@info @sprintf("Domain %.3f × %.3f × %.3f m = %.0f × %.0f × %.0f δ_s, %d periods%s",
               Lx, Ly, Lz, Lx/δ, Ly/δ, Lz/δ, n_periods,
               passive_scalar ? " (b is a passive scalar)" : "")
