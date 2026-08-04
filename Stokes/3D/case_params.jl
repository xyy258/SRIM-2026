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

const δ      = sqrt(2ν / ω)    # laminar Stokes layer thickness ≈ 0.14142
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

# ---------------- Background stratification: EXPONENTIAL ----------------
# Reverting to the -Varying L_strat configuration. The stratification vanishes at
# the seabed and recovers exponentially to the far-field N∞² over a scale L:
#
#     N²_bg(z) = N∞² [1 − exp(−z/L)]
#     b_bg(z)  = N∞² [z + L (exp(−z/L) − 1)]      (b_bg(0) = 0, b_bg′(0) = 0)
#
# The initial buoyancy IS this profile (Tidal3D.jl: bᵢ = b_background), and the
# sponge target and top-gradient BC relax toward it — the full L_strat set-up,
# not just an initial condition.
#
# L is now set as a FRACTION of the domain height Lz = 90 δ_s ≈ 12.73 m, via
# L_STRAT_LZ, sweeping {0.2, 0.5, 1.0, 1.5} Lz. For L ≳ Lz the far field is only
# partly established inside the domain — at the lid N²_bg/N∞² = 1 − e^{−Lz/L}
# (0.993, 0.865, 0.632, 0.487 for the four fractions) — which is the regime under
# study here. The definitions follow the Domain block below since they need Lz.
const L_frac = parse(Float64, get(ENV, "L_STRAT_LZ", "0.5"))

# ---------------- Domain ----------------
# Paper §2.4: lx = 50 δ_s, ly = 25 δ_s, test section lz = 70 δ_s with the
# sponge spanning 70 δ_s – 90 δ_s, i.e. 5.95 × 2.98 × 10.73 m here.
const Lx      = 50δ            # streamwise  ≈ 5.960 m
const Ly      = 25δ            # spanwise    ≈ 2.980 m
const Lz_test = 70δ            # top of the analysed test section ≈ 8.344 m
const Lz      = 150δ           # PHYSICAL domain ≈ 21.213 m

# OLD: the sponge was a slice taken out of the top of the Lz-tall grid — a
# Gaussian centred on the lid, so the uppermost ~20 δ_s of the physical domain
# was permanently relaxed and the free interior was really only 0 – ~126 δ_s.
#
# NEW: the sponge SITS ON TOP of the physical domain. The grid is built to
# Lz_total = Lz + L_sponge and the damping mask is identically zero at and below
# z = Lz, so all 150 δ_s of physical domain evolve freely. Lz remains the
# physical domain everywhere else in the code: L_strat = L_frac·Lz, the
# background buoyancy profile, and plot limits are all unchanged by this.
const L_sponge = 20δ           # thickness of the added damping layer ≈ 2.828 m
const Lz_total = Lz + L_sponge # full grid height ≈ 24.042 m

# ---------------- Exponential background (needs Lz) ----------------
# L set as a fraction of the domain height (L_frac read above), then the
# exponential background and its buoyancy integral.
const L_strat = L_frac * Lz

@inline N²_background(z) = N²_ref * (1 - exp(-z / L_strat))
@inline b_background(z)  = N²_ref * (z + L_strat * (exp(-z / L_strat) - 1))

# ---------------- Run length ----------------
# Target; the overnight driver (run_night.sh) may stop a case earlier at a
# wall-clock deadline. Every half period is a U∞ = 0 snapshot, so cutting a run
# at any half-period boundary still leaves a phase-consistent restart state.
# The paper's figure 5 late time is t = ωt_d = 50, i.e. 7.96 tidal periods, so
# 8 completed periods already covers the range those figures span.
const n_periods = parse(Int, get(ENV, "N_PERIODS", "12"))

# ---------------- Output naming ----------------
# Each run is tagged by its stratification scale L as a fraction of Lz and its Ri
# case, e.g. L0p5Lz_Ri500, so the four-value sweep does not collide on disk
# (a fractional fraction like 0.5 becomes "0p5", 1.0 becomes "1").
const Lfrac_lbl = isinteger(L_frac) ? string(Int(L_frac)) : replace(string(L_frac), "." => "p")
const casetag  = "L" * Lfrac_lbl * "Lz_" * case
const outdir   = "output_" * casetag
const filename = joinpath(outdir, "TidalBL3D_" * casetag)
mkpath(outdir)

@info @sprintf("Case %s (L = %.3f Lz = %.2f m): Ri = %g, N² = %.4g s⁻², δ_s = %.4f m, Re_s = %.0f",
               casetag, L_frac, L_strat, Ri, N², δ, Re_s)
@info @sprintf("Domain %.3f × %.3f × %.3f m = %.0f × %.0f × %.0f δ_s (physical), + %.1f δ_s sponge on top → grid top %.0f δ_s, %d periods%s",
               Lx, Ly, Lz, Lx/δ, Ly/δ, Lz/δ, L_sponge/δ, Lz_total/δ, n_periods,
               passive_scalar ? " (b is a passive scalar)" : "")
@info @sprintf("Background: exponential, N²_bg = N∞²[1−exp(−z/L)], L = %.2f m = %.1f δ_s (N²_bg/N∞² = %.3f at z = δ_s, %.3f at lid z = Lz)",
               L_strat, L_strat/δ, N²_background(δ)/N²_ref, N²_background(Lz)/N²_ref)
