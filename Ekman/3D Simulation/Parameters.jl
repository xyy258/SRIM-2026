# Input parameters
const U∞ = 0.04                 # far stream velocity
const f₀ = 1e-4                 # Coriolis parameter

if !@isdefined(r) || isnothing(r)
const r   = 75                  # ratio N/f
end
const Re∞ = 4.55e7              # Reynolds number
const Pr  = 10                  # Prandtl number

# Dimensions
const Lx, Ly, Lz = 75,75,60
# Grid size
const Nx, Ny, Nz = 100,100,400

# Duration and timestep
const max_Δt   = 7.5            # maximum allowable timestep
const duration = 18e4           # non-dimensional duration of the simulation

# Sponge layer thickness
const S = 10

# Other parameters
const N² = (r*f₀)^2             # buoyancy frequency
const κ  = 0.41                 # von Karman constant
const ν₀ = 1e-6                 # molecular kinematic viscosity
const D  = U∞/f₀                # Rossby lengthscale
const κ₀ = ν₀/Pr                # molecular diffusivity
u_star   = 0.049*U∞             # friction velocity
z₀       = 0.0016               # m (roughness length)
δ        = u_star/f₀            # boundary layer lengthscale
Re_star  = u_star*δ/ν₀          # frictional Reynolds
Ri_star  = N²/f₀^2              # frictional Richardson

# Coefficient of drag calculated later:
# z₁ = abs(Array(znodes(grid, Center()))[1])
# cᴰ = (κ/log(z₁/z₀))^2

if !@isdefined(profile) || isnothing(profile)
    const profile = 2           # type of initial buoyancy profile
                                # (0=linear, 1=nonlinear, 2=exponential, 3=linear+exp decay, 4=softplus)
end
const mask = 1                  # type of masking in sponge layer (0=piecewise, 1=Gaussian)
const H = Lz + S                # domain height, with sponge layer

if profile == 1
    if !@isdefined(T) || isnothing(T)
        const T = 10            # adjusts when sharp change in buoyancy occurs
    end
elseif profile == 2
    if !@isdefined(Lᴰ) || isnothing(Lᴰ)
        const Lᴰ = Lz           # decay length of exp profile with fixed buoyancy difference
    end
elseif profile == 3
    if !@isdefined(Lᴰ) || isnothing(Lᴰ)
        const Lᴰ = Lz           # decay length of exp profile with fixed top gradient
    end
elseif profile == 4
    if !@isdefined(T) || isnothing(T)
        const T = 10            # change in buoyancy
    end
    sharp = 6                   # changes how sharply buoyancy profile changes
end

const kick = 0.01*U∞            # amplitude of random perturbation