using Oceananigans, Printf
# using CUDA
# using NCDatasets

# Running on GPU or CPU
arch = CPU()
# Command to run file in Julia
# include("Ekman/2D Simulation/Ekman 2D.jl")

# Dimensions
Lx, Lz = 70, 30

# Grid size
Nx, Nz = 64, 256

# Duration and timestep
max_Δt = 7.5 # maximum allowable timestep
duration = 5e4 # The non-dimensional duration of the simulation

# Ratio of N/f (compare with profiles in Taylor & Sarkar 2008)
r = 75
# Coriolis parameter
f₀ = 1e-4
# Buoyancy frequency
N² = (r*f₀)^2

# Creates a grid with near-constant spacing `refinement * Lz / Nz`
# near the bottom:
refinement = 2 # controls spacing near surface (higher means finer spaced)
stretching = 10 # controls rate of stretching at bottom

# "Warped" height coordinate
h(k) = (Nz + 1 - k) / Nz

# Linear near-surface generator
ζ(k) = 1 + (h(k) - 1) / refinement

# Bottom-intensified stretching function
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

# Generating function
z_faces(k) = - Lz * (ζ(k) * Σ(k) - 1)

grid = RectilinearGrid(arch;
                        topology=(Periodic, Flat, Bounded),
                        size=(Nx, Nz),
                        x=(0, Lx),
                        z=z_faces)

# Set the amplitude of the random perturbation (kick)
kick = 0.05

## Boundary conditions
U∞ = 0.0674
z₀ = 0.0016 # m (roughness length)
κ = 0.41  # von Karman constant

z₁ = abs(first(Array(znodes(grid, Center())))) # Closest grid center to the bottom
cᴰ = (κ / log(z₁ / z₀))^2 # drag coefficient

ν₀ = 1e-6 # molecular kinematic viscosity
D = U∞/f₀
Re∞ = U∞*D/ν₀ # Reynolds number
Pr = 10 # Prandtl number
κ₀ = ν₀/Pr # molecular diffusivity
u_star = 0.049*U∞ # friction velocity
δ = u_star/f₀ # boundary layer lengthscale

# Drag boundary condition
drag_bc_u = BulkDrag(coefficient=cᴰ)

# No slip
# drag_bc_u = ValueBoudaryCondition(0)

u_bcs = FieldBoundaryConditions(bottom=drag_bc_u)
b_bcs = FieldBoundaryConditions(top=GradientBoundaryCondition(N²),
                                bottom = GradientBoundaryCondition(0))

## Forcing
v_forcing_fn(x, z, t, p) = p.f * p.s  # to balance for initial geostrophic balance
forcing_params = (s=U∞, f=f₀)
v_forcing = Forcing(v_forcing_fn, parameters=forcing_params)

# Now, define a 'model' where we specify the grid, advection scheme, bcs, and other settings
model = NonhydrostaticModel(grid;
    advection = WENO(order=5),
    timestepper = :RungeKutta3, # Timestep scheme
    tracers = :b,  # Tracers: b is buoyancy, c is a passive tracer (e.g. dye)
    buoyancy = BuoyancyTracer(),
    closure = ScalarDiffusivity(ν=ν₀, κ=κ₀),
    boundary_conditions = (u=u_bcs, b=b_bcs), # specify the boundary conditions that we defiend above
    coriolis = FPlane(f=f₀), # Coriolis with Coriolis parameter f₀
    forcing = (v=v_forcing,) # Forcing due to constant pressure gradient to balance initial velocity U
)

## Initial conditions
uᵢ(x, z) = U∞ + kick * randn()
vᵢ(x, z) = 0
wᵢ(x, z) = kick * randn()
bᵢ(x, z) = N² * z

@info "2D simulation parameters"
@printf("
Dimensions      %.1f m × %.1f m
Grid size       %.1f × %.1f
Square buoyancy frequency:      N² = %.2e,
Coriolis parameter:             f = %.2e,
Ratio:                          r = N/f = %.1f
Molecular kinematic viscosity:  ν = %.2e,
Reynolds number:                Re∞ = %.2e,
Prandtl number:                 Pr = %.1f,
Molecular diffusivity:          κ = %.2e,
Drag coefficient:               cᴰ = %.4f,
Layer lengthscale:              δ = %.2f\n",
Lx, Lz, Nx, Nz, N², f₀, r, ν₀, Re∞, Pr, κ₀, cᴰ, δ)

# Send the initial conditions to the model to initialize the variables
set!(model, u=uᵢ, v=vᵢ, w=wᵢ, b=bᵢ)

# Now, we create a 'simulation' to run the model for a specified length of time
simulation = Simulation(model, Δt=max_Δt, stop_time=duration)

## The `TimeStepWizard`
#
# The TimeStepWizard manages the time-step adaptively, keeping the
# Courant-Freidrichs-Lewy (CFL) number close to `1.0` while ensuring
# the time-step does not increase beyond the maximum allowable value
wizard = TimeStepWizard(cfl=0.85, max_change=1.2, max_Δt=max_Δt)
# A "Callback" pauses the simulation after a specified number of timesteps and calls a function (here the timestep wizard to update the timestep)
# To update the timestep more or less often, change IterationInterval in the next line
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

# ## A progress messenger
# We add a callback that prints out a helpful progress message while the simulation runs.

start_time = time_ns()

progress(sim) = @printf("i: % 6d, sim time: % 5f, wall time: % 10s, Δt: % 5f, CFL: %.2e\n",
    sim.model.clock.iteration,
    sim.model.clock.time,
    prettytime(1e-9 * (time_ns() - start_time)),
    sim.Δt,
    AdvectiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))

## Output

u, v, w = model.velocities # unpack velocity `Field`s
b = model.tracers.b # extract the buoyancy

# Set the name of the output file
filename = "Data/Ekman_2D"

# JLD2 output file
simulation.output_writers[:xz_velocity] =
    JLD2Writer(model, (; u, v, w),
               filename = filename * "_velocity.jld2",
               indices = (:, 1, :),
               schedule = TimeInterval(250),
               overwrite_existing = true,
               with_halos = false)
simulation.output_writers[:xz_b_c] =
    JLD2Writer(model, (; b),
               filename = filename * "_b_c.jld2",
               indices = (:, 1, :),
               schedule = TimeInterval(250),
               overwrite_existing = true,
               with_halos = false)

# NetCDF output file
# simulation.output_writers[:xz_b_c] =
#     NetCDFWriter(model, (; b),
#                filename = filename * "_velocity.nc",
#                indices = (:, 1, :),
#                schedule = TimeInterval(500),
#                overwrite_existing = true,
#                with_halos = false)

# Horizontally-averaged buoyancy
db_dz_avg = Field(Average(∂z(b), dims=(1, 2)))

# JLD2 output file
simulation.output_writers[:avg_db_dz] =
    JLD2Writer(model, (; db_dz=db_dz_avg),
                filename="Data/Average b gradient 2D.jld2",
                schedule=IterationInterval(2),
                overwrite_existing=true)
# NetCDF output file
# simulation.output_writers[:avg_db_dz] =
#     NetCDFWriter(model, (; db_dz=db_dz_avg),
#                 filename="Data/Average b gradient 2D.nc",
#                 schedule=TimeInterval(20),
#                 overwrite_existing=true)

nothing # hide

# Now, run the simulation
run!(simulation)

include("Ekman_plot_2D.jl")
include("Ekman_plot2_2D.jl")