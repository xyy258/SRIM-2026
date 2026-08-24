# Shared case parameters for Tidal3D.jl
# Every script (simulation, animation, profiles, figures) includes this file and
# selects the case from the command line, with the run knobs in the environment:
#     PROFILE=4 T_STRAT=10 julia --project=. Tidal3D.jl sqrtRi2
#
# IMPORTANT: PROFILE and T_STRAT feed into `casetag`, so every post-processing
# script must be invoked with the SAME values as the run it is reading, or it
# will look in the wrong output directory and reconstruct the wrong background.
#
# Current sweep: profile 4 (softplus), T ∈ {5, 10, 20, 30} m,
# √Ri ∈ {0.5, 1, 2, 5, 10} → 20 runs.
#
# Declared `const` so closures that capture these (forcing, sponge masks)
# compile to fast kernels.
using Printf

const case = isempty(ARGS) ? "Ri0" : ARGS[1]

# Named cases from the Gayen reproduction, kept so old commands still work.
const Ri_targets = Dict("Ri0" => 0.0, "Ri500" => 500.0, "Ri2500" => 2500.0)

# The softplus sweep is parameterised by √Ri ∈ {0.5, 1, 2, 5, 10}, i.e.
# Ri ∈ {0.25, 1, 4, 25, 100} — values the fixed dict above cannot express.
# Accept either "sqrtRi<v>" or "Ri<v>", with 'p' standing in for the decimal
# point so the case name is also a safe directory name:
#     sqrtRi0p5 → Ri = 0.25      Ri0p25 → Ri = 0.25
function parse_Ri(c)
    haskey(Ri_targets, c) && return Ri_targets[c]
    for (pat, f) in ((r"^sqrtRi([0-9]+(?:p[0-9]+)?)$", x -> x^2),
                     (r"^Ri([0-9]+(?:p[0-9]+)?)$",     identity))
        m = match(pat, c)
        m === nothing || return f(parse(Float64, replace(m[1], "p" => ".")))
    end
    error("Unknown case \"$c\" — use Ri0/Ri500/Ri2500, or sqrtRi0p5 / Ri0p25 style")
end

const Ri = parse_Ri(case)

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
const passive_scalar = Ri == 0

# The far-field gradient the b tracer actually carries. For Ri = 0, N² = 0 but b
# is still advected as a passive scalar, so post-processing needs a finite
# reference to normalize by — ω² stands in there. Used by Tidal3Danimation.jl and
# Tidal3Dprofiles.jl; do not delete without updating both.
const N²_ref = Ri > 0 ? N² : ω^2

# Softplus sharpness (m⁻¹), profile 4 only. WARNING: the 10–90 % width of the
# transition in db/dz is 2ln9/sharp = 0.73 m at sharp = 6, while the vertical
# grid coarsens to Δz ≈ 0.43 m above z ≈ 10 m — so for T = 10, 20, 30 the initial
# pycnocline spans under 2 cells and starts life as a numerical step. Only T = 5
# (Δz ≈ 0.13 m, ~5.6 cells) is properly resolved. Lower `sharp` to ~1 for a 4.4 m
# transition (~10 cells) if the T-dependence is meant to be the signal.
const sharp = parse(Float64, get(ENV, "SHARP", "6"))

# Which background buoyancy profile the run uses (the if-chain below). Set from
# the shell alongside the other run knobs, e.g.
#     PROFILE=4 T_STRAT=10 julia --project=. Tidal3D.jl sqrtRi2
# NOTE these must be `const`: b_background and N²_background are evaluated
# inside GPU kernels (the initial condition and the sponge target), so every
# global they capture has to be constant-folded at compile time.
const profile = parse(Int,     get(ENV, "PROFILE",  "4"))
const T       = parse(Float64, get(ENV, "T_STRAT", "10.0"))   # profiles 1 and 4

# ---------------- Exponential-background scale (profiles 2 and 3 only) --------
# Sets L_strat = L_frac · Lz, the e-folding scale over which the stratification
# recovers to the far field. UNUSED by the current softplus sweep — profile 4
# takes its length scale from T instead — but kept as the knob for profiles 2/3.
# The definition of L_strat itself follows the Domain block, since it needs Lz.
const L_frac = parse(Float64, get(ENV, "L_STRAT_LZ", "0.5"))

# ---------------- Domain ----------------
# The paper's §2.4 box is lx = 50 δ_s, ly = 25 δ_s, lz = 70 δ_s. This run is much
# taller in absolute terms (10 × 10 × 40 m = 71 × 71 × 283 δ_s) because the
# softplus pycnocline sits at z = T up to 30 m and needs clear water above it.
const Lx      = 10             # streamwise (m)
const Ly      = 10             # spanwise (m)
const Lz      = 50             # PHYSICAL domain height (m)

# The sponge SITS ON TOP of the physical domain: the grid is built to
# Lz_total = Lz + L_sponge and the damping mask is identically zero at and below
# z = Lz, so the whole physical domain evolves freely. Lz remains the physical
# domain everywhere else in the code.
const L_sponge = 10
const Lz_total = Lz + L_sponge

# Top of the analysed / plotted section. In the paper this was a structural
# boundary (test section 0–70 δ_s, sponge above it); here the sponge lives above
# Lz instead, so this is now purely a diagnostic and plotting height — the cell
# count logged by Tidal3D.jl and the vertical range of Tidal3Danimation.jl.
#
# It therefore has to cover the feature under study. For the T-parameterised
# profiles that is the pycnocline at z = T, so the window tracks T with ~10 m of
# headroom (T = 5 → 15 m, 10 → 20, 20 → 30, 30 → 40) rather than sitting at a
# fixed 70 δ_s = 9.9 m, which would have cropped the pycnocline out of every
# animation with T ≥ 10. LZ_TEST overrides it.
const Lz_test_default = profile in (1, 4) ? min(Lz, max(70δ, T + 10)) : min(Lz, 70δ)
const Lz_test = parse(Float64, get(ENV, "LZ_TEST", string(Lz_test_default)))

# ---------------- Exponential background scale (needs Lz) ----------------
const L_strat = L_frac * Lz

# N²_background MUST stay the exact derivative of b_background in every branch:
# it supplies the top gradient BC (Tidal3D.jl), while b_background supplies the
# initial condition and the sponge target. If they disagree the lid fights the
# interior. Each pair below is written with no auxiliary local bindings so the
# whole expression constant-folds inside the GPU kernel.
#
# These scale with N²_ref, NOT N². The two are equal whenever Ri > 0, but at
# Ri = 0 (√Ri = 0 in the sweep) N² = 0, which would make b_background ≡ 0 — the
# tracer would carry no background at all, every T would give an identical run,
# and the figure 4/5 panels would be blank. N²_ref = ω² there instead, so Ri = 0
# is a genuine passive scalar advected through the same softplus profile, which
# is what the paper's Ri = 0 case is. The buoyancy FORCE is still off (Tidal3D.jl
# passes buoyancy = nothing when passive_scalar), so this adds no dynamics.
if profile == 0         # linear
    @inline b_background(z)  = N²_ref * z
    @inline N²_background(z) = N²_ref * one(z)
    Profile = "linear"
elseif profile == 1     # nonlinear
    # a = 0.2 (z/T)^5, so 5a = (z/T)^5 and b′ = N²_ref[1 − e^{−a}(1 − 5a)].
    @inline b_background(z)  = N²_ref*z*(1-exp(-0.2*(z/T)^5))
    @inline N²_background(z) = N²_ref * (1 - exp(-0.2*(z/T)^5) * (1 - (z/T)^5))
    Profile = "nonlinear"
elseif profile == 2     # exponential with fixed buoyancy difference
    # Normalised so b(0) = 0 and b(Lz) = N²_ref·Lz regardless of L_strat.
    @inline b_background(z)  = N²_ref*Lz*(L_strat*(exp(z/L_strat)-1)-z) /
                               (L_strat*(exp(Lz/L_strat)-1)-Lz)
    @inline N²_background(z) = N²_ref*Lz*(exp(z/L_strat)-1) /
                               (L_strat*(exp(Lz/L_strat)-1)-Lz)
    Profile = "exponential"
elseif profile == 3     # linear with exponential decay
    @inline b_background(z)  = N²_ref*(z+L_strat*(exp(-z/L_strat)-1))
    @inline N²_background(z) = N²_ref * (1 - exp(-z / L_strat))
    Profile = "linear with exponential decay"
elseif profile == 4     # softplus, T is a parameter, taking values {5,10,20,30}
    # b = N²_ref/s · log(1 + e^{s(z−T)}) → b′ = N²_ref·sigmoid(s(z−T)): unstratified
    # below z = T, N²_ref above, over a transition of width ~1/sharp.
    @inline b_background(z)  = N²_ref/sharp*log(1+exp(sharp*(z-T)))
    @inline N²_background(z) = N²_ref/(1+exp(-sharp*(z-T)))
    Profile = "softplus"
else
    error("Unknown profile $profile — set PROFILE to one of 0,1,2,3,4")
end
# Tidal periods simulated. Env-settable so one driver can ask for a short
# unstratified spin-up and long production cases: the sweep drivers have always
# passed N_PERIODS for the spin-up, and until now it was silently ignored.
const n_periods = parse(Int, get(ENV, "N_PERIODS", "8"))

# ---------------- Output naming ----------------
# The tag carries everything that changes the physics, so the 20-run softplus
# sweep cannot collide on disk: e.g. P4_T10_sqrtRi2. Decimal points become "p"
# so the tag is a safe directory name. Note the writers use
# overwrite_existing = true, so a tag collision silently destroys existing data.
num_lbl(x) = isinteger(x) ? string(Int(x)) : replace(string(x), "." => "p")

const shape_lbl = profile in (1, 4) ? "_T" * num_lbl(T) :
                  profile in (2, 3) ? "_L" * num_lbl(L_frac) * "Lz" : ""
# RUN_TAG overrides the whole tag — used to park the one shared spin-up run in
# outputs/spinup*/ rather than under a case name it does not really belong to.
const casetag  = get(ENV, "RUN_TAG", "P" * string(profile) * shape_lbl * "_" * case)
# Every run's data lives under one outputs/ root, one subdirectory per case, so
# the working directory holds scripts rather than a few dozen output_* folders.
const outroot  = get(ENV, "OUT_ROOT", "outputs")
const outdir   = joinpath(outroot, casetag)
const filename = joinpath(outdir, "TidalBL3D_" * casetag)
mkpath(outdir)

# Default source of turbulent velocities for a restart. ONE Ri = 0 spin-up on
# this grid serves every case in the sweep: the velocity field does not depend
# on the buoyancy profile, so T and Ri may vary freely against it. Produce it with
#     RUN_TAG=spinup julia --project=. Tidal3D.jl Ri0
const spinup_default = joinpath(outroot, "spinup", "TidalBL3D_spinup_fields.jld2")

@info @sprintf("Case %s: Ri = %g (√Ri = %g), N² = %.4g s⁻², δ_s = %.4f m, Re_s = %.0f",
               casetag, Ri, sqrt(Ri), N², δ, Re_s)
@info @sprintf("Domain %.3f × %.3f × %.3f m = %.0f × %.0f × %.0f δ_s (physical), + %.1f δ_s sponge on top → grid top %.0f δ_s, %d periods%s",
               Lx, Ly, Lz, Lx/δ, Ly/δ, Lz/δ, L_sponge/δ, Lz_total/δ, n_periods,
               passive_scalar ? " (b is a passive scalar)" : "")
@info @sprintf("Analysed/plotted section: 0 – %.2f m = %.0f δ_s%s",
               Lz_test, Lz_test/δ,
               profile in (1, 4) ? @sprintf(" (pycnocline at T = %.1f m)", T) : "")
# Report the profile that actually ran. Normalizing by N²_ref (not N²) keeps the
# ratios finite and shape-only at Ri = 0, where the passive scalar carries the
# same profile scaled by ω².
nrat(z) = N²_background(z) / N²_ref
@info @sprintf("Background: %s (profile %d)%s — N²_bg/N∞² = %.3f at z = δ_s, %.3f at z = Lz, %.3f at lid; Δb over domain = %.3g m s⁻²",
               Profile, profile,
               profile in (1, 4) ? @sprintf(", T = %.1f m = %.1f δ_s", T, T/δ) :
               profile in (2, 3) ? @sprintf(", L = %.2f m = %.1f δ_s", L_strat, L_strat/δ) : "",
               nrat(δ), nrat(Lz), nrat(Lz_total), b_background(Lz_total) - b_background(0.0))
