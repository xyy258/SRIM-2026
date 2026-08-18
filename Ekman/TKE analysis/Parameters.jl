# Parameters for the TKE / K_T analysis run.
#
# COPIED FROM "Ekman/3D Simulation/Parameters.jl" and changed in exactly three
# ways, all of them about WHICH CASE runs — no physical parameter, grid size,
# duration or timestep differs from the original:
#
#   1. r, T and profile are pinned here (r = 1, T = 20, softplus) instead of being
#      set by an outer driver via `global`. The original is driven by Overnight.jl,
#      which assigns them before `include`ing the simulation; this folder runs one
#      case, so they are plain values with environment overrides.
#   2. `sharp` is `const`. It has to be: the softplus background is evaluated
#      inside GPU kernels, and a non-const global there compiles to a slow,
#      type-unstable kernel. Its VALUE is unchanged at 6.
#   3. An inertial period T_f = 2π/f₀ is defined, since every schedule and
#      smoothing window in the TKE analysis is expressed in it.
#
# Everything below the "unchanged" line is the original file, value for value.

using Printf

# ---------------- The case ----------------
# r = N/f = 1, softplus background with the step at T = 20 m, sharpness 6.
# Override from the shell if a second case is ever wanted:
#     R=2 T_STRAT=30 julia --project=... Ekman3D.jl
const r       = parse(Float64, get(ENV, "R",       "1"))
const T       = parse(Float64, get(ENV, "T_STRAT", "20"))
const profile = parse(Int,     get(ENV, "PROFILE", "4"))
const sharp   = parse(Float64, get(ENV, "SHARP",   "6"))

# SHARP = 6 IS FINE ON THIS GRID, unlike on the Stokes one. The 10–90 % width of
# the db/dz transition is 2ln9/sharp = 0.73 m, and this grid is almost uniform at
# Δz = 0.135 m from the wall up past z = 30 m, so the pycnocline at z = 20 m spans
# 5.4 cells. The Stokes study had to drop its default from 6 to 2 because its grid
# coarsens to Δz ≈ 0.34 m above 10 m, where 0.73 m is only ~2 cells and the
# pycnocline begins life as a numerical step. That does not happen here — the
# comparison is worth making explicitly, since the two studies otherwise use the
# same background profile and the same analysis.

# ---------------- Unchanged from Ekman/3D Simulation/Parameters.jl ----------
const U∞ = 0.04                 # far stream velocity
const f₀ = 1e-4                 # Coriolis parameter

const Pr = 10                   # Prandtl number
const z₀ = 0.0016               # m (roughness length)

# Dimensions
const Lx, Ly, Lz = 75,75,100
# Grid size
const Nx, Ny, Nz = 100,100,500

# Duration and timestep
const max_Δt   = 7.5            # maximum allowable timestep
const duration = 40e4           # duration of the simulation (s)

# Sponge layer thickness
const S = 20

# Other parameters
const N   = r*f₀                # buoyancy frequency
const N²  = (r*f₀)^2            # squared buoyancy frequency
const κ   = 0.41                # von Karman constant
const ν₀  = 1e-6                # molecular kinematic viscosity
const D   = U∞/f₀               # Rossby lengthscale
const κ₀  = ν₀/Pr               # molecular diffusivity
const Re∞ = U∞*D/ν₀             # Reynolds number

const mask = 1                  # type of masking in sponge layer (0=piecewise, 1=Gaussian)
const H = Lz + S                # domain height, with sponge layer

const kick = 0.01*U∞            # amplitude of random perturbation

# ---------------- Added for the TKE analysis ----------------
# THE INERTIAL PERIOD IS THIS RUN'S CLOCK. The Ekman flow has no tide, but it is
# not steady either: it starts from rest-relative geostrophic imbalance and rings
# at f while the boundary layer grows. 2π/f₀ = 62832 s is therefore the natural
# unit for the output cadence, for the smoothing window, and for how much of the
# start to discard as transient.
#
# It is also NUMERICALLY THE SAME as the Stokes tidal period, because ω there was
# set to 1e-4 to match this f₀ — so the two analyses use identical windows and
# their K_T results are directly comparable.
const T_f = 2π / f₀             # inertial period, 62832 s = 17.45 h
const n_periods_total = duration / T_f      # 6.37

@info @sprintf("Ekman TKE case: r = N/f = %.1f, N² = %.3g s⁻², profile %d (softplus), T = %.1f m, sharp = %.1f",
               r, N², profile, T, sharp)
@info @sprintf("Domain %.0f × %.0f × %.0f m + %.0f m sponge, grid %d × %d × %d; duration %.0f s = %.2f inertial periods (T_f = %.0f s)",
               Lx, Ly, Lz, S, Nx, Ny, Nz, duration, n_periods_total, T_f)
