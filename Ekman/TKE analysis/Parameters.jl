# ==============================================================================
# Simulation Parameters: Ekman TKE / K_T Analysis
#
# Defines physical constants, domain dimensions, grid resolution, and time-stepping.
# Environment variables can override default parameters:
#   R=1 T_STRAT=20 PROFILE=4 SHARP=6 julia --project=... Ekman3D.jl
# ==============================================================================

using Printf

# ------------------------------------------------------------------------------
# Case Configuration Parameters
# ------------------------------------------------------------------------------

# Stratification and background profile settings
const r       = parse(Float64, get(ENV, "R",       "1"))    # N/f ratio
const T       = parse(Float64, get(ENV, "T_STRAT", "20"))   # Pycnocline depth (m)
const profile = parse(Int,     get(ENV, "PROFILE", "4"))    # Background profile index (4 = softplus)

# Declared `const` for GPU kernel performance/type stability.
# At sharp=6, the 10–90% transition is ~0.73 m (~5.4 grid cells at Δz ≈ 0.135 m near z = 20 m).
const sharp   = parse(Float64, get(ENV, "SHARP",   "6"))    # Pycnocline sharpness parameter

# ------------------------------------------------------------------------------
# Physical Parameters & Fluid Properties
# ------------------------------------------------------------------------------

const U∞ = 0.04                # Far-stream boundary velocity (m/s)
const f₀ = 1e-4                # Coriolis parameter (s⁻¹)
const Pr = 10                  # Prandtl number
const z₀ = 0.0016              # Surface roughness length (m)
const κ  = 0.41                # von Kármán constant
const ν₀ = 1e-6                # Molecular kinematic viscosity (m²/s)
const κ₀ = ν₀ / Pr             # Molecular diffusivity (m²/s)

# Derived physical quantities
const N  = r * f₀              # Buoyancy frequency (s⁻¹)
const N² = (r * f₀)^2          # Squared buoyancy frequency (s⁻²)
const D  = U∞ / f₀             # Rossby length scale (m)
const Re∞ = U∞ * D / ν₀        # Reynolds number based on Rossby scale

# ------------------------------------------------------------------------------
# Domain & Grid Setup
# ------------------------------------------------------------------------------

# Domain dimensions (m)
const Lx, Ly, Lz = 75, 75, 100

# Grid dimensions (100 × 100 × 500)
const Nx, Ny, Nz = 100, 100, 500

# Sponge layer configuration
const S    = 20                # Sponge layer thickness (m)
const H    = Lz + S            # Total domain height including sponge (m)
const mask = 1                 # Sponge layer damping profile (0 = piecewise, 1 = Gaussian)

# Perturbation parameters
const kick = 0.01 * U∞         # Amplitude of initial random velocity perturbation (m/s)

# ------------------------------------------------------------------------------
# Time Discretization & Time-Scales
# ------------------------------------------------------------------------------

const max_Δt   = 7.5           # Maximum allowable timestep (s)
const duration = 40e4          # Total simulation duration (s)

# Inertial period T_f = 2π/f₀ serves as the reference clock for output cadences,
# smoothing windows, and transient filtering (matches the Stokes tidal period).
const T_f             = 2π / f₀           # Inertial period (~62,832 s = 17.45 h)
const n_periods_total = duration / T_f    # Total runtime in inertial periods (~6.37)

# ------------------------------------------------------------------------------
# Diagnostic Output
# ------------------------------------------------------------------------------

@info @sprintf("Ekman TKE case: r = N/f = %.1f, N² = %.3g s⁻², profile %d (softplus), T = %.1f m, sharp = %.1f",
               r, N², profile, T, sharp)
@info @sprintf("Domain %.0f × %.0f × %.0f m + %.0f m sponge, grid %d × %d × %d; duration %.0f s = %.2f inertial periods (T_f = %.0f s)",
               Lx, Ly, Lz, S, Nx, Ny, Nz, duration, n_periods_total, T_f)