# Shared parameters for every script in this folder (simulation, animation,
# profiles, figures). The case is given on the command line and the run options
# come from the environment:
#     PROFILE=4 T_STRAT=10 julia --project=. Tidal3D.jl sqrtRi2
# Post-processing must use the same PROFILE and T_STRAT as the run it reads,
# since these choose the output folder and rebuild the background profile.
# Everything is `const` because GPU kernels capture these values.
using Printf

const case = isempty(ARGS) ? "Ri0" : ARGS[1]

# Named cases from the Gayen reproduction, kept so old commands still work.
const Ri_targets = Dict("Ri0" => 0.0, "Ri500" => 500.0, "Ri2500" => 2500.0)

# A case may also be named by √Ri or by Ri, with 'p' in place of the decimal
# point so the name is a safe directory name:
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
# Dimensional values from Gayen, Sarkar & Taylor (2010) §2.4, with ω lowered to
# 1e-4 so it matches the Coriolis parameter of the Ekman case.
const ω  = 1e-4        # M2 tidal frequency (s⁻¹) to match coriolis parameter in colleague's case
const U₀ = 0.04               # tidal velocity amplitude (m s⁻¹) = 4 cm s⁻¹
const ν  = 1.0e-6              # molecular viscosity of water (m² s⁻¹)
const Pr = 10                 # molecular Prandtl number 
const κ  = ν / Pr              # molecular diffusivity

const δ      = sqrt(2ν / ω)    # laminar Stokes layer thickness ≈ 0.14142
const Re_s   = U₀ * δ / ν      # ≈ 1788
const T_tide = 2π / ω

# ---------------- Bottom drag ----------------
# Quadratic bulk drag, as in Ekman 3D.jl: cᴰ = (κ_vk/log(z/z₀))².
# Note `κ` above is the molecular diffusivity, so the von Kármán constant needs
# a name of its own.
const κ_vk = 0.41              # von Kármán constant
const z₀   = 0.0016            # roughness length (m), from the Ekman case

# The log law only holds above the roughness sublayer. This grid resolves the
# wall, so its first cell centre (z₁ = 0.0043 m = 2.7 z₀) sits below that layer
# and the law cannot be evaluated there. It is used at a fixed reference height
# instead — the first cell centre of the Ekman grid:
const z_drag_ref = parse(Float64, get(ENV, "Z_DRAG_REF", "0.0667"))
const cᴰ_ref = (κ_vk / log(max(z_drag_ref, 2z₀) / z₀))^2   # ≈ 0.0121

# The first cell lies in the viscous sublayer, so the molecular stress is the
# larger term there and the drag law acts as a roughness correction on top of it.
# Both numbers can be overridden from the shell:
#     CD=0.0121  julia --project=. Tidal3D.jl sqrtRi2     # set cᴰ directly
#     Z_DRAG_REF=0.0667 ...                               # move the reference height

# Background stratification. At Ri = 0 the paper still carries a temperature
# field, but a passive one: the tracer keeps its background gradient while the
# buoyancy term is switched off in Tidal3D.jl.
const N²     = Ri * ω^2                    # buoyancy frequency² actually felt
const passive_scalar = Ri == 0

# The far-field gradient the b tracer carries. At Ri = 0 the buoyancy frequency
# is zero, so ω² stands in and post-processing has something finite to
# normalize by. Used by Tidal3Danimation.jl and Tidal3Dprofiles.jl.
const N²_ref = Ri > 0 ? N² : ω^2

# Softplus sharpness (m⁻¹), profile 4 only. The 10–90 % width of the transition
# in db/dz is 2ln9/sharp = 2.20 m at the default, the same physical width at
# every T and wide enough for the grid to resolve it. Keep it fixed across a
# T-sweep, so that T is the only thing varying.
const sharp = parse(Float64, get(ENV, "SHARP", "2"))

# Which background buoyancy profile to use (the if-chain below), and the
# pycnocline height T for profiles 1 and 4. Set from the shell, e.g.
#     PROFILE=4 T_STRAT=10 julia --project=. Tidal3D.jl sqrtRi2
# Both must be `const`: b_background and N²_background run inside GPU kernels.
const profile = parse(Int,     get(ENV, "PROFILE",  "4"))
const T       = parse(Float64, get(ENV, "T_STRAT", "10.0"))   # profiles 1 and 4

# ---------------- Exponential background scale (profiles 2 and 3) ------------
# L_strat = L_frac · Lz is the scale over which the stratification recovers to
# the far field. Unused by the softplus sweep, which takes its length from T.
# L_strat itself is defined after the Domain block, since it needs Lz.
const L_frac = parse(Float64, get(ENV, "L_STRAT_LZ", "0.5"))

# ---------------- Domain ----------------
# Taller than the paper's box (70 δ_s) because the softplus pycnocline sits at
# z = T, up to 30 m, and needs clear water above it.
const Lx      = 10             # streamwise (m)
const Ly      = 10             # spanwise (m)
const Lz      = 50             # PHYSICAL domain height (m)

# The sponge sits on top of the physical domain: the grid is built to
# Lz + L_sponge and the damping is zero at and below z = Lz, so the physical
# domain evolves freely. Lz is the physical domain everywhere else in the code.
const L_sponge = 10
const Lz_total = Lz + L_sponge

# Top of the analysed and plotted section — a diagnostic and plotting height
# only. It tracks the pycnocline at z = T with about 10 m of headroom, so the
# feature under study is always in frame. LZ_TEST overrides it.
const Lz_test_default = profile in (1, 4) ? min(Lz, max(70δ, T + 10)) : min(Lz, 70δ)
const Lz_test = parse(Float64, get(ENV, "LZ_TEST", string(Lz_test_default)))

# ---------------- Exponential background scale (needs Lz) ----------------
const L_strat = L_frac * Lz

# N²_background must be the exact derivative of b_background in every branch:
# b_background sets the initial condition and the sponge target, N²_background
# the top gradient boundary condition, and the lid fights the interior if they
# disagree. Each pair is written without local variables so the expression
# constant-folds inside the GPU kernel.
# Both scale with N²_ref rather than N², so at Ri = 0 the tracer still carries
# the profile as a passive scalar instead of a flat zero.
if profile == 0         # linear
    @inline b_background(z)  = N²_ref * z
    @inline N²_background(z) = N²_ref * one(z)
    Profile = "linear"
elseif profile == 1     # nonlinear
    # b′ is the derivative of N²_ref z (1 − e^{−0.2(z/T)^5}).
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
    # Unstratified below z = T, N²_ref above it, over a transition of
    # width ~1/sharp.
    @inline b_background(z)  = N²_ref/sharp*log(1+exp(sharp*(z-T)))
    @inline N²_background(z) = N²_ref/(1+exp(-sharp*(z-T)))
    Profile = "softplus"
else
    error("Unknown profile $profile — set PROFILE to one of 0,1,2,3,4")
end
# Tidal periods simulated. Set from the environment so one driver can ask for a
# short spin-up and long production runs.
const n_periods = parse(Int, get(ENV, "N_PERIODS", "8"))

# ---------------- Output naming ----------------
# The tag carries everything that changes the physics, e.g. P4_T10_sqrtRi2, so
# the runs of a sweep cannot overwrite each other. Decimal points become "p" so
# the tag is a safe directory name.
num_lbl(x) = isinteger(x) ? string(Int(x)) : replace(string(x), "." => "p")

const shape_lbl = profile in (1, 4) ? "_T" * num_lbl(T) :
                  profile in (2, 3) ? "_L" * num_lbl(L_frac) * "Lz" : ""
# RUN_TAG replaces the whole tag, used to park the shared spin-up run.
const casetag  = get(ENV, "RUN_TAG", "P" * string(profile) * shape_lbl * "_" * case)
# All data lives under one outputs/ folder, one subfolder per case.
const outroot  = get(ENV, "OUT_ROOT", "outputs")
const outdir   = joinpath(outroot, casetag)
const filename = joinpath(outdir, "TidalBL3D_" * casetag)
mkpath(outdir)

# Default spin-up used for restarts. One Ri = 0 spin-up serves every case in a
# sweep, since the velocity field does not depend on the buoyancy profile.
# Produce it with
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
# Report the profile that actually ran, normalized by N²_ref so the ratios stay
# finite at Ri = 0.
nrat(z) = N²_background(z) / N²_ref
@info @sprintf("Background: %s (profile %d)%s — N²_bg/N∞² = %.3f at z = δ_s, %.3f at z = Lz, %.3f at lid; Δb over domain = %.3g m s⁻²",
               Profile, profile,
               profile in (1, 4) ? @sprintf(", T = %.1f m = %.1f δ_s", T, T/δ) :
               profile in (2, 3) ? @sprintf(", L = %.2f m = %.1f δ_s", L_strat, L_strat/δ) : "",
               nrat(δ), nrat(Lz), nrat(Lz_total), b_background(Lz_total) - b_background(0.0))
