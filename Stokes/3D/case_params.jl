# Shared case parameters for the Gayen et al. (2010) reproduction.
# Every script (simulation, animation, profiles) includes this file and
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
const ω  = 1.4075235e-4        # M2 tidal frequency (s⁻¹), period ≈ 12.4 h
const U₀ = 0.15                # tidal velocity amplitude (m s⁻¹)
const N² = Ri * ω^2            # background stratification from Ri = N²/ω²
const ν  = 1.109e-5            # molecular viscosity (m² s⁻¹) → Re_s = 1790
const Pr = 0.7                 # molecular Prandtl number (paper value)
const κ  = ν / Pr              # molecular diffusivity

const δ      = sqrt(2ν / ω)    # laminar Stokes layer thickness ≈ 0.40 m
const T_tide = 2π / ω

# ---------------- Domain ----------------
const Lx = 20.0                # streamwise (m)
const Ly = 10.0                # spanwise ≈ 25 δ_s, matches the paper; halving
                               # from 20 m brings Δy⁺ near their ≈30
const Lz = 30.0                # test section + sponge

# ---------------- Run length ----------------
# Ri=0 runs from rest through transition and serves as the turbulent
# spin-up state for the stratified cases; the paper spins up 15+ cycles
# before switching on stratification, so we match that here. Stratified
# cases restart from the final Ri=0 snapshot and need enough cycles for
# the mixed layer to approach the paper's quasi-steady plateau (their
# figure 4 shows growth still slowly evolving out to ωt ≈ 28, i.e. ~4.5
# periods, but our first 4-period attempt hadn't saturated — 8 gives more
# margin to see whether/where it levels off).
const n_periods = 15

# ---------------- Output naming ----------------
const outdir   = "output_" * case
const filename = joinpath(outdir, "TidalBL3D_" * case)
mkpath(outdir)

@info @sprintf("Case %s: Ri = %g, N² = %.4g s⁻², δ = %.3f m, Re_δ = %.0f, %d periods",
               case, Ri, N², δ, U₀ * δ / ν, n_periods)
