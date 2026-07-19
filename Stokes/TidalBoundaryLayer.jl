using Printf
using Oceananigans
using NCDatasets
using CUDA

arch = CPU()

# Resolved LES, constant viscocity, no slip
# want thickness O(1), reynolds number
# viscocity order 10e-5, stokes length approx 0.4
# Simulation of a tidal (stokes) boundary layer following the paper by Gayen et al (2009)
# Dimensions
Lx, Ly, Lz = 20,20,30

# Grid size
Nx, Nz = 64,256

# Duration and timestep
n_periods = 4
ω = 0.00014075235
max_Δt = 100 # maximum allowable timestep
duration = n_periods * 2π/ω # The non-dimensional duration of the simulation
@info "Simulation time is $duration"

# Buoyancy frequency
N² = 1e-7

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
z_faces(k) = - Lz * (ζ(k) * Σ(k) - 1)

grid = RectilinearGrid(arch;
                        topology = (Periodic, Flat, Bounded),
                        size = (Nx, Nz),
                        x = (0, Lx),
                        z = z_faces)


# Set the amplitude of the random perturbation (kick)
kick = 0.05

# set the boundary conditions
# FluxBoundaryCondition specifies the momentum or buoyancy flux (in this case zero)
# ValueBoundaryCondition specifies the value of the corresponding variable
# top/bottom correspond to the boundaries in the z-direction
# east/west correspond to the boundaries in the x-direction
# north/south correspond to the boundaries in the y-direction (not used for periodic topology)
# by default, Oceananigans imposes no flux and no normal flow boundary conditions in bounded directions

## Boundary conditions
U₀ = 0.05 # 5cm per second
z₀ = 0.01 # m (roughness length)
κ = 0.41  # von Karman constant
z₁ = first(znodes(grid, Center())) # Closest grid center to the bottom
@assert z₁ > z₀
cᴰ = (κ / log(z₁ / z₀))^2 # Drag coefficient
#@inline drag_u(x, t, u, v, p) = - p.c_drag * √(u^2 + v^2) * u
#@inline drag_v(x, t, u, v, p) = - p.c_drag * √(u^2 + v^2) * v
#drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; c_drag = cᴰ))
#drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; c_drag = cᴰ))
# No slip boundary conditions
drag_bc_u = ValueBoundaryCondition(0)
drag_bc_v = ValueBoundaryCondition(0)

u_bcs = FieldBoundaryConditions(bottom=drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom=drag_bc_v)

#w_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(0)) # Free slip top boundary condition
b_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(N²),
                                bottom = FluxBoundaryCondition(0))

#@inline BackgroundVelocity(x, y, z, t, u, v, w, U) = U
#U_field = BackgroundField(U, parameters = U)

#Tidal Forcing
@inline tidal_forcing(x, z, t, p) = p.U₀ * p.ω * cos(p.ω * t)
u_forcing = Forcing(tidal_forcing, parameters = (; U₀, ω))


# Now, define a 'model' where we specify the grid, advection scheme, bcs, and other settings
model = NonhydrostaticModel(grid;
            advection =  WENO(order=5),
            timestepper = :RungeKutta3, # # Timestepping scheme
            tracers = (:b, :c),  # Set the name(s) of any tracers: b is buoyancy, c is a passive tracer (e.g. dye)
            buoyancy = BuoyancyTracer(),
            #closure = ScalarDiffusivity(ν=1.109e-5, κ=1.109e-5),
            closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1.109e-5, κ=1.109e-5),
            # Assume stable scheme for now
            boundary_conditions = (u = u_bcs, v = v_bcs, b=b_bcs), # specify the boundary conditions that we defined above
            coriolis = nothing, # No coriolis for now
            forcing = (; u = u_forcing)
)

## Initial conditions
# Here, we start with a linear function for buoyancy (assume linear stratification) and add a random perturbation to the velocity.
uᵢ(x, z) = kick * randn()
vᵢ(x, z) = kick * randn()
wᵢ(x, z) = kick * randn()
bᵢ(x, z) = N² * z
cᵢ(x, z) = exp(-((x - Lx / 2) / (Lx / 50))^2) # Initialize with a thin tracer (dye) streak in the center of the domain

# Send the initial conditions to the model to initialize the variables
set!(model, u = uᵢ, v = vᵢ, w = wᵢ, b = bᵢ, c = cᵢ)

# Now, we create a 'simulation' to run the model for a specified length of time
simulation = Simulation(model, Δt = 1.0, stop_time = duration)
#simulation = Simulation(model, Δt = max_Δt, stop_time = duration)

### The `TimeStepWizard`
#
# The TimeStepWizard manages the time-step adaptively, keeping the
# Courant-Freidrichs-Lewy (CFL) number close to `1.0` while ensuring
# the time-step does not increase beyond the maximum allowable value
wizard = TimeStepWizard(cfl = 0.85, max_change = 1.1, max_Δt = max_Δt)
# A "Callback" pauses the simulation after a specified number of timesteps and calls a function (here the timestep wizard to update the timestep)
# To update the timestep more or less often, change IterationInterval in the next line
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
#add_callback!(simulation, wizard, IterationInterval(10), name = :wizard)
# ### A progress messenger
# We add a callback that prints out a helpful progress message while the simulation runs.

start_time = time_ns()

progress(sim) = @printf("i: % 6d, sim time: % 8f, wall time: % 10s, Δt: % 6f, CFL: %.2e\n",
                        sim.model.clock.iteration,
                        sim.model.clock.time,
                        prettytime(1e-9 * (time_ns() - start_time)),
                        sim.Δt,
                        AdvectiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))
#add_callback!(simulation, progress, IterationInterval(10), name = :progress)
### Output

u, v, w = model.velocities # unpack velocity `Field`s
b = model.tracers.b # extract the buoyancy
c = model.tracers.c # extract the tracer

# Set the name of the output file
filename = "Stokes/TidalBoundaryLayer"

# simulation.output_writers[:xz_slices] =
#     JLD2Writer(model, (; u, v, w, b, c),
#                 filename = filename * ".jld2",
#                 indices = (:, 1, :),
#                 schedule = TimeInterval(20),
#                 overwrite_existing = true,
#                 with_halos = false)
n_frames = 4000
simulation.output_writers[:xz_slices] =
    JLD2Writer(model, (; u, v, w, b, c),
                filename = filename * ".jld2",
                indices = (:, 1, :),
                schedule = TimeInterval(duration / n_frames),
                overwrite_existing = true,
                with_halos = false)
# simulation.output_writers[:xz_slices] =
#     NetCDFWriter(model, (; u, v, w, b, c),
#                 filename = filename * ".nc",
#                 indices = (:, 1, :),
#                 schedule = TimeInterval(0.2),
#                 overwrite_existing = true,
#                 with_halos = false)

# If you are running in 3D, you could save an xy slice like this:
#simulation.output_writers[:xy_slices] =
#    JLD2Writer(model, (; u, v, w, b),
                    #     filename = filename * "_xy.jld2",
                    #     indices = (:,:,10),
                    #     schedule = TimeInterval(0.1),
                    #     overwrite_existing = true)

nothing # hide
# Horizontally-averaged velocities & buoyancy
u_avg = Field(Average(u, dims=(1, 2)))
v_avg = Field(Average(v, dims=(1, 2)))
b_avg = Field(Average(b, dims=(1, 2)))

# (Horizontally-averaged) buoyancy gradient ∂b/∂z
db_dz_avg = ∂z(b_avg)

# JLD2 output file
simulation.output_writers[:avg_db_dz] =
    JLD2Writer(model, (; db_dz=db_dz_avg),
                filename="Stokes/Average buoyancy gradient.jld2",
                schedule = TimeInterval(duration / n_frames),
                overwrite_existing=true)

run!(simulation)

#include("TidalBoundaryLayer_plot.jl")