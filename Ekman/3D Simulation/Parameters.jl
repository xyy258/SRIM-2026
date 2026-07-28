# Input parameters
U∞ = 0.0674                 # far stream velocity
f₀ = 1e-4                   # Coriolis parameter

if !@isdefined(r) || isnothing(r)
r   = 75                    # ratio N/f
end
Re∞ = 4.55e7                # Reynolds number
Pr  = 10                    # Prandtl number
z₀  = 0.0016                # m (roughness length)

# Dimensions
Lx, Ly, Lz = 80,80,60
# Grid size
Nx, Ny, Nz = 100,100,300

# Duration and timestep
max_Δt = 7.5 # maximum allowable timestep
duration = 18e4 # The non-dimensional duration of the simulation

# Sponge layer thickness
S = 10

# Other parameters
N²      = (r*f₀)^2          # buoyancy frequency
κ       = 0.41              # von Karman constant
ν₀      = 1e-6              # molecular kinematic viscosity
D       = U∞/f₀             # Rossby lengthscale
κ₀      = ν₀/Pr             # molecular diffusivity
u_star  = 0.049*U∞          # friction velocity
δ       = u_star/f₀         # boundary layer lengthscale
Re_star = u_star*δ/ν₀       # frictional Reynolds
Ri_star = N²/f₀^2           # frictional Richardson

# Coefficient of drag calculated later:
# z₁ = abs(Array(znodes(grid, Center()))[1])
# cᴰ = (κ/log(z₁/z₀))^2

profile = 1                 # type of initial buoyancy profile (0=linear, 1=exponential)
mask = 0                    # type of masking in sponge layer (0=piecewise, 1=Gaussian)
H = Lz + S                  # domain height, with sponge layer

# If using profile == "Exponential"
if profile == "Exponential"
    if !@isdefined(r) || isnothing(r)
        efoldfactor = 1
    end
end
efold = efoldfactor*Lz      # e-folding length for buoyancy

kick = 0.01*U∞              # amplitude of random perturbation