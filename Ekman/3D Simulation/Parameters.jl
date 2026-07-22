# Dimensions
Lx, Ly, Lz = 72.8,72.8,27.3
# Grid size
Nx, Ny, Nz = 100,100,180

# Duration and timestep
max_Δt = 5 # maximum allowable timestep
duration = 18e4 # The non-dimensional duration of the simulation

# Sponge layer thickness
S = 10

# Input parameters
U∞ = 0.0674                 # far stream velocity
f₀ = 1e-4                   # Coriolis parameter

if isnothing(r)
r   = 75                    # ratio N/f
end
Re∞ = 4.55e7                # Reynolds number
Pr  = 10                    # Prandtl number
z₀  = 0.0016                # m (roughness length)


N²      = (r*f₀)^2          # buoyancy frequency
κ       = 0.41              # von Karman constant
ν₀      = 1e-6              # molecular kinematic viscosity
D       = U∞/f₀             # Rossby lengthscale
κ₀      = ν₀/Pr             # molecular diffusivity
u_star  = 0.049*U∞          # friction velocity
cᴰ      = 2(u_star/U∞)^2    # drag coefficient
δ       = u_star/f₀         # boundary layer lengthscale
Re_star = u_star*δ/ν₀       # frictional Reynolds
Ri_star = N²/f₀^2           # frictional Richardson

# cᴰ drag coefficient calculated later in code, if used

kick = U∞ * 0.05             # amplitude of random perturbation