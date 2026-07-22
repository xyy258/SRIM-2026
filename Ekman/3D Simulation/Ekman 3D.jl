using Oceananigans, Printf
using CUDA
# using NCDatasets

# Running on GPU or CPU
arch = GPU()
# Command to run file in Julia
# include("Ekman/3D Simulation/Ekman 3D.jl")

# Import parameters
include("Parameters.jl")
H = Lz + S # domain height, with sponge layer

# Creates a grid with near-constant spacing `refinement * Lz / Nz`
# near the bottom:
refinement = 1.8 # controls spacing near surface (higher means finer spaced)
stretching = 10 # controls rate of stretching at bottom
# "Warped" height coordinate
h(k) = (Nz + 1 - k) / Nz
# Linear near-surface generator
ζ(k) = 1 + (h(k) - 1) / refinement
# Bottom-intensified stretching function
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

# Generating function
z_faces(k) = - H * (ζ(k) * Σ(k) - 1)

grid = RectilinearGrid(arch;
    topology = (Periodic, Periodic, Bounded),
    size = (Nx, Ny, Nz),
    x = (0, Lx),
    y = (0, Ly),
    z = z_faces)

# # Calculating drag coefficient
z₁ = abs(first(Array(znodes(grid, Center())))) # Closest grid center to the bottom
cᴰ = (κ / log(z₁ / z₀))^2 # drag coefficient

## Boundary conditions
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

## Initial conditions
uᵢ(x,y,z) = U∞ + kick * randn()
vᵢ(x,y,z) = kick * randn()
wᵢ(x,y,z) = kick * randn()
bᵢ(x,y,z) = N² * z

## Forcing
v_forcing_fn(x, y, z, t, p) = p.f * p.s  # to balance for initial geostrophic balance
forcing_params = (s=U∞, f=f₀)
v_forcing = Forcing(v_forcing_fn, parameters=forcing_params)

## Sponge layers
sponge_rate  = 10*r*f₀ # set to 10*(buoyancy frequency)
sponge_mask = GaussianMask{:z}(center=H, width=S)

u_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask,
                      target = U∞)
v_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask)
w_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask)
b_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask,
                      target = LinearTarget{:z}(intercept = 0, gradient = N²))



# Now, define a 'model' where we specify the grid, advection scheme, bcs, and other settings
model = NonhydrostaticModel(grid;
    advection = Centered(order=2),
    timestepper = :RungeKutta3, # Timestepping scheme
    tracers = :b,  # Set the name(s) of any tracers: b is buoyancy, c is a passive tracer (e.g. dye)
    buoyancy = BuoyancyTracer(),

    # Closures for LES
    closure = (ScalarDiffusivity(ν=ν₀,κ=κ₀),AnisotropicMinimumDissipation()),
    # closure = DynamicSmagorinsky(Pr=Pr),
    # closure = SmagorinskyLilly(Pr=Pr),

    boundary_conditions = (u = u_bcs, v = v_bcs, b=b_bcs), # specify the boundary conditions that we defiend above
    coriolis = FPlane(f=f₀),
    forcing = (
    u = u_sponge,
    v = (v_forcing, v_sponge),
    w = w_sponge,
    b = b_sponge)
)

@info "3D simulation parameters"
params_string =

@printf("Dimensions                      %.1f m × %.1f m × %.1f m
Grid size                       %.1f × %.1f × %.1f
Far stream velocity             U∞ = %.4f
Square buoyancy frequency:      N² = %.2e,
Coriolis parameter:             f = %.2e,
Ratio:                          r = N/f = %.1f
Molecular kinematic viscosity:  ν = %.2e,
Reynolds number:                Re∞ = %.2e,
Prandtl number:                 Pr = %.1f,
Molecular diffusivity:          κ = %.2e,
Frictional velocity             u* = %.4f
Drag coefficient:               cᴰ = %.4f,
Layer lengthscale:              δ = %.2f
Frictional Reynolds             Re* = %.2e
Frictional Richardson           Ri* = %.1f\n",
Lx, Ly, Lz, Nx, Ny, Nz, U∞, N², f₀, r, ν₀, Re∞, Pr, κ₀, u_star, cᴰ, δ, Re_star, Ri_star)

open(@sprintf("Ekman/3D Simulation/r=%.1f parameters.txt",r), "w") do file
    write(file, @sprintf("Dimensions                      %.1f m × %.1f m × %.1f m
Grid size                       %.1f × %.1f × %.1f
Far stream velocity             U∞ = %.4f
Square buoyancy frequency:      N² = %.2e,
Coriolis parameter:             f = %.2e,
Ratio:                          r = N/f = %.1f
Molecular kinematic viscosity:  ν = %.2e,
Reynolds number:                Re∞ = %.2e,
Prandtl number:                 Pr = %.1f,
Molecular diffusivity:          κ = %.2e,
Frictional velocity             u* = %.4f
Drag coefficient:               cᴰ = %.4f,
Layer lengthscale:              δ = %.2f
Frictional Reynolds             Re* = %.2e
Frictional Richardson           Ri* = %.1f",
Lx, Ly, Lz, Nx, Ny, Nz, U∞, N², f₀, r, ν₀, Re∞, Pr, κ₀, u_star, cᴰ, δ, Re_star, Ri_star))
end

# Send the initial conditions to the model to initialize the variables
set!(model, u = uᵢ, v = vᵢ, w = wᵢ, b = bᵢ)

# Now, we create a 'simulation' to run the model for a specified length of time
simulation = Simulation(model, Δt = 0.5 * max_Δt, stop_time = duration)

## The `TimeStepWizard`
#
# The TimeStepWizard manages the time-step adaptively, keeping the
# Courant-Freidrichs-Lewy (CFL) number close to `1.0` while ensuring
# the time-step does not increase beyond the maximum allowable value
wizard = TimeStepWizard(cfl = 0.85, max_change = 1.25, max_Δt = max_Δt)
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

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# ## Output

u, v, w = model.velocities # unpack velocity `Field`s
b = model.tracers.b # extract the buoyancy

# Set the name of the output file
filename = @sprintf("Ekman/Data/Ekman r=%.1f",r)

simulation.output_writers[:xz_velocity] =
    JLD2Writer(model, (; u, v, w),
               filename = filename * "_velocity.jld2",
               indices = (:, 1, :),
               schedule = TimeInterval(100),
               overwrite_existing = true,
               with_halos = false)
simulation.output_writers[:xz_b_c] =
    JLD2Writer(model, (; b),
               filename = filename * "_b.jld2",
               indices = (:, 1, :),
               schedule = TimeInterval(100),
               overwrite_existing = true,
               with_halos = false)

# Horizontally-averaged velocities & buoyancy
u_avg = Field(Average(u, dims=(1, 2)))
v_avg = Field(Average(v, dims=(1, 2)))
b_avg = Field(Average(b, dims=(1, 2)))

# Horizontally-averaged buoyancy gradient ∂b/∂z
db_dz_avg = Field(Average(∂z(b), dims=(1, 2)))

# Horizontally-averaged vorticity
ωx_avg = Field(Average(∂y(w)-∂z(v), dims=(1, 2)))
ωy_avg = Field(Average(∂z(u)-∂x(w), dims=(1, 2)))
ωz_avg = Field(Average(∂x(v)-∂y(u), dims=(1, 2)))

simulation.output_writers[:avg_db_dz] =
    JLD2Writer(model, (; db_dz = db_dz_avg),
                filename = filename * " average buoyancy gradient.jld2",
                schedule = IterationInterval(5),
                overwrite_existing = true)
simulation.output_writers[:avg_b] =
    JLD2Writer(model, (; b = b_avg),
                filename = filename * " average buoyancy.jld2",
                schedule = IterationInterval(20),
                overwrite_existing = true)
simulation.output_writers[:avg_velocity] =
    JLD2Writer(model, (; u_avg, v_avg),
                filename = filename * " average velocity.jld2",
                schedule = IterationInterval(20),
                overwrite_existing = true)
simulation.output_writers[:avg_vorticity] =
    JLD2Writer(model, (; ωx_avg, ωy_avg, ωz_avg),
                filename = filename * " average vorticity.jld2",
                schedule = IterationInterval(20),
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