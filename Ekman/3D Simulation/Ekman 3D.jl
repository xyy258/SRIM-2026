using Oceananigans, Printf
# using CUDA
# using NCDatasets

# Running on GPU or CPU
arch = CPU()
# Command to run file in Julia
# include("Ekman/3D Simulation/Ekman 3D.jl")

# Dimensions
Lx, Ly, Lz = 72.8,72.8,27.3

# Grid size
Nx, Ny, Nz = 32,32,64

# Duration and timestep
max_Δt = 5 # maximum allowable timestep
duration = 2.5e4 # The non-dimensional duration of the simulation

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
                        topology = (Periodic, Periodic, Bounded),
                        size = (Nx, Ny, Nz),
                        x = (0, Lx),
                        y = (0, Ly),
                        z = z_faces)

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

# Quadratic drag
drag_bc_u = BulkDrag(coefficient=cᴰ)
drag_bc_v = BulkDrag(coefficient=cᴰ)

# No slip boundary conditions
# drag_bc_u = ValueBoundaryCondition(0)
# drag_bc_v = ValueBoundaryCondition(0)

u_bcs = FieldBoundaryConditions(bottom=drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom=drag_bc_v)
b_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(N²),
                                bottom = GradientBoundaryCondition(0))

## Forcing
v_forcing_fn(x, y, z, t, p) = p.f * p.s  # to balance for initial geostrophic balance
forcing_params = (s=U∞, f=f₀)
v_forcing = Forcing(v_forcing_fn, parameters=forcing_params)

# Now, define a 'model' where we specify the grid, advection scheme, bcs, and other settings
model = NonhydrostaticModel(grid;
            advection = WENO(order=5),
            timestepper = :RungeKutta3, # Timestepping scheme
            tracers = :b,  # Set the name(s) of any tracers: b is buoyancy, c is a passive tracer (e.g. dye)
            buoyancy = BuoyancyTracer(),

            # Closures for LES
            closure = AnisotropicMinimumDissipation(),
            # closure = DynamicSmagorinsky(Pr=Pr),
            # closure = SmagorinskyLilly(Pr=Pr),

            boundary_conditions = (u = u_bcs, v = v_bcs, b=b_bcs), # specify the boundary conditions that we defiend above
            coriolis = FPlane(f=f₀),
            forcing = (v=v_forcing,)
)

## Initial conditions
uᵢ(x,y,z) = U∞ + kick * randn()
vᵢ(x,y,z) = kick * randn()
wᵢ(x,y,z) = kick * randn()
bᵢ(x,y,z) = N² * z

# Send the initial conditions to the model to initialize the variables
set!(model, u = uᵢ, v = vᵢ, w = wᵢ, b = bᵢ)

# Now, we create a 'simulation' to run the model for a specified length of time
simulation = Simulation(model, Δt = 0.8 * max_Δt, stop_time = duration)

## The `TimeStepWizard`
#
# The TimeStepWizard manages the time-step adaptively, keeping the
# Courant-Freidrichs-Lewy (CFL) number close to `1.0` while ensuring
# the time-step does not increase beyond the maximum allowable value
wizard = TimeStepWizard(cfl = 0.8, max_change = 1.25, max_Δt = max_Δt)
# A "Callback" pauses the simulation after a specified number of timesteps and calls a function (here the timestep wizard to update the timestep)
# To update the timestep more or less often, change IterationInterval in the next line
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))

# ## A progress messenger
# We add a callback that prints out a helpful progress message while the simulation runs.

start_time = time_ns()

progress(sim) = @printf("i: % 6d, sim time: % 8f, wall time: % 10s, Δt: % 6f, CFL: %.2e\n",
                        sim.model.clock.iteration,
                        sim.model.clock.time,
                        prettytime(1e-9 * (time_ns() - start_time)),
                        sim.Δt,
                        AdvectiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

# ## Output

u, v, w = model.velocities # unpack velocity `Field`s
b = model.tracers.b # extract the buoyancy

# Set the name of the output file
filename = "Ekman/Data/Ekman"

simulation.output_writers[:xz_velocity] =
    JLD2Writer(model, (; u, v, w),
               filename = filename * "_velocity.jld2",
               indices = (:, 1, :),
               schedule = TimeInterval(150),
               overwrite_existing = true,
               with_halos = false)
simulation.output_writers[:xz_b_c] =
    JLD2Writer(model, (; b),
               filename = filename * "_b_c.jld2",
               indices = (:, 1, :),
               schedule = TimeInterval(150),
               overwrite_existing = true,
               with_halos = false)

# Horizontally-averaged velocities & buoyancy
u_avg = Field(Average(u, dims=(1, 2)))
v_avg = Field(Average(v, dims=(1, 2)))
b_avg = Field(Average(b, dims=(1, 2)))

# Horizontally-averaged buoyancy gradient ∂b/∂z
db_dz_avg = Field(Average(∂z(b), dims=(1, 2)))

simulation.output_writers[:avg_db_dz] =
    JLD2Writer(model, (; db_dz = db_dz_avg),
                filename = "Ekman/Data/Average buoyancy gradient.jld2",
                schedule = IterationInterval(2),
                overwrite_existing = true)
simulation.output_writers[:avg_b] =
    JLD2Writer(model, (; b = b_avg),
                filename = "Ekman/Data/Average buoyancy.jld2",
                schedule = IterationInterval(2),
                overwrite_existing = true)
simulation.output_writers[:avg_velocity] =
    JLD2Writer(model, (; u_avg, v_avg),
                filename = "Ekman/Data/Average velocity.jld2",
                schedule = IterationInterval(2),
                overwrite_existing = true)
# NetCDF output file
# simulation.output_writers[:avg_db_dz] =
#     NetCDFWriter(model, (; db_dz=db_dz_avg),
#                 filename = "Ekman/Data/Average buoyancy gradient.nc",
#                 schedule = IterationInterval(2),
#                 overwrite_existing = true)

nothing # hide

# Now, run the simulation
run!(simulation)

include("Ekman_anim.jl")
include("Ekman_plot.jl")