# Ekman 3D with second-moment output — the TKE / K_T analysis run.
#
# A COPY of "Ekman/3D Simulation/Ekman 3D.jl". Nothing physical is changed.
# Four things differ, all of them bookkeeping:
#
#   1. Moments.jl is included immediately before run!(simulation), adding ONE
#      output writer that produces the plane-averaged second moments the K_T
#      measurement needs. No existing writer, filename or schedule is touched.
#   2. Output goes to Data/ beside this script instead of the shared Ekman/Data
#      tree, and the parameters .txt to this folder instead of
#      "Ekman/3D Simulation/Parameters/", so this run cannot overwrite anything
#      belonging to the original.
#   3. Paths are anchored to @__DIR__ rather than to the working directory.
#   4. The animation and plotting scripts at the end are opt-in (PLOTS=1), since
#      the batch driver runs them as a separate, restartable step.
#
# Run it directly with
#     julia --project=<repo root> "Ekman/TKE analysis/Ekman3D.jl"
# or through swirlesrun.jl, which is what swirles.sh submits.

using Pkg
Pkg.instantiate()
## Start code

using Oceananigans, Printf, JLD2
using CUDA
using Statistics

# Running on GPU or CPU
arch = GPU()

# Import parameters
include(joinpath(@__DIR__, "Parameters.jl"))

# Smoke test: a few iterations instead of the full duration, to check the whole
# chain (grid, drag, closure, the SGS lookup in Moments.jl, the writers) on a
# workstation GPU before spending a batch allocation on it.
const EKMAN_SMOKE = get(ENV, "EKMAN_SMOKE", "0") == "1"
const SMOKE_ITERS = parse(Int, get(ENV, "SMOKE_ITERS", "20"))

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
    size     = (Nx, Ny, Nz),
    x = (0, Lx),
    y = (0, Ly),
    z = z_faces)

# Calculating drag coefficient
z₁ = abs(Array(znodes(grid, Center()))[1]) # Closest grid center to the bottom
cᴰ = (κ/log(z₁/z₀))^2 # drag coefficient

## Boundary conditions
# Quadratic drag
drag_bc_u = BulkDrag(coefficient=cᴰ)
drag_bc_v = BulkDrag(coefficient=cᴰ)

# No slip boundary conditions
# drag_bc_u = ValueBoundaryCondition(0)
# drag_bc_v = ValueBoundaryCondition(0)

u_bcs = FieldBoundaryConditions(bottom=drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom=drag_bc_v)
b_bcs = FieldBoundaryConditions(
                                # top = GradientBoundaryCondition(N²),
                                bottom = GradientBoundaryCondition(0))

## Initial conditions
uᵢ(x,y,z) = U∞ + kick * randn()
vᵢ(x,y,z) = kick * randn()
wᵢ(x,y,z) = kick * randn()

scale = (r==0 || isnothing(r)) ? 1 : N²
bᵢ = if profile == 0
    Profile = "linear"
    (x, y, z) -> scale * z
elseif profile == 1
    Profile = "nonlinear"
    (x, y, z) -> scale * z * (1 - exp(-0.2 * (z / T)^5))
elseif profile == 2
    Profile = "exponential"
    (x, y, z) -> scale * Lz * (Lᴰ * (exp(z / Lᴰ) - 1) - z) / (Lᴰ * (exp(Lz / Lᴰ) - 1) - Lz)
elseif profile == 3
    Profile = "linear with exponential decay"
    (x, y, z) -> scale * (z + Lᴰ * (exp(-z / Lᴰ) - 1))
elseif profile == 4
    Profile = "softplus"
    (x, y, z) -> (scale / sharp) * log(1 + exp(sharp * (z - T)))
end
@info "Using a(n) $Profile initial buoyancy profile..."

## Forcing
v_forcing_fn(x, y, z, t, p) = p.f * p.s  # to balance for initial geostrophic balance
forcing_params = (s=U∞, f=f₀)
v_forcing = Forcing(v_forcing_fn, parameters=forcing_params)

## Sponge layers
sponge_rate = r*f₀ # set to (buoyancy frequency)

if mask == 0
    sponge_mask = PiecewiseLinearMask{:z}(center=H, width=S)
    Mask = "piecewise linear"
elseif mask == 1
    sponge_mask = GaussianMask{:z}(center=H, width=0.85S)
    Mask = "Gaussian"
end
@info "Using $Mask mask for sponge layer..."

u_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask,
                      target = U∞)
v_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask)
w_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask)

# Create a target field on the GPU grid
b_target = Field{Center, Center, Center}(grid)
# Set field values using bᵢ
set!(b_target, bᵢ)
b_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask,
                    #   target = LinearTarget{:z}(intercept = 0, gradient = N²))
                      target = b_target)

buoyancy_model = !(r == 0 || isnothing(r)) ? BuoyancyTracer() : nothing

# Define our model: specify grid, advection scheme, bcs, etc...
model = NonhydrostaticModel(grid;
    advection   = Centered(order=4),
    timestepper = :RungeKutta3, # Timestepping scheme
    tracers     = :b,  # Set the name(s) of any tracers: b is buoyancy, c is a passive tracer (e.g. dye)
    buoyancy    = buoyancy_model,

    # Closures for LES
    closure = (ScalarDiffusivity(ν=ν₀,κ=κ₀),AnisotropicMinimumDissipation()),
    # closure = (ScalarDiffusivity(ν=ν₀,κ=κ₀),DynamicSmagorinsky(Pr=Pr)),
    # closure = (ScalarDiffusivity(ν=ν₀,κ=κ₀),SmagorinskyLilly(Pr=Pr)),

    boundary_conditions = (u = u_bcs, v = v_bcs, b=b_bcs), # specify the boundary conditions that we defiend above
    coriolis = FPlane(f=f₀),
    forcing = (
    u = u_sponge,
    v = (v_forcing, v_sponge),
    w = w_sponge,
    b = b_sponge)
)

# Send the initial conditions to the model to initialize the variables
set!(model, u = uᵢ, v = vᵢ, w = wᵢ, b = bᵢ)

@info "3D simulation parameters"

@printf(
"Dimensions                      %.1f m × %.1f m × %.1f m
Grid size                       %d × %d × %d
Far stream velocity             U∞  = %.4f
Square buoyancy frequency:      N²  = %.2e,
Coriolis parameter:             f   = %.2e,
Ratio:                          r   = N/f = %.1f\n",
Lx, Ly, Lz, Nx, Ny, Nz, U∞, N², f₀, r)

# Now, we create a 'simulation' to run the model for a specified length of time
simulation = Simulation(model, Δt = 0.1 * max_Δt, stop_time = duration)

## The `TimeStepWizard`
#
# The TimeStepWizard manages the time-step adaptively, keeping the
# Courant-Freidrichs-Lewy (CFL) number close to `1.0` while ensuring
# the time-step does not increase beyond the maximum allowable value
wizard = TimeStepWizard(cfl = 0.9, max_change = 1.2, max_Δt = max_Δt)
# A "Callback" pauses the simulation after a specified number of timesteps and calls a function (here the timestep wizard to update the timestep)
# To update the timestep more or less often, change IterationInterval in the next line
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))

# ## A progress messenger
# We add a callback that prints out a helpful progress message while the simulation runs.

start_time = time_ns()

progress(sim) = @printf("i: % 6d, sim time: % 8f, wall time: % 10s, Δt: % 5f, CFL: %.2e\n",
    sim.model.clock.iteration,
    sim.model.clock.time,
    prettytime(1e-9 * (time_ns() - start_time)),
    sim.Δt,
    AdvectiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# ## Output

u, v, w = model.velocities # unpack velocity `Field`s
b = model.tracers.b # extract the buoyancy

# Set the name of the output file in
# Folder: "data_folder"
# File name: "filename"
# Remember to also update "Filename.jl"
# Data/ beside this script, NOT the shared Ekman/Data tree. Matches the `root`
# that Filename_plot.jl / Filename_anim.jl build, so the plotting scripts find it.
filename = joinpath(@__DIR__, "Data", @sprintf("r=%.1f, T=%.1f", r, T)) * "/"
mkpath(filename)

simulation.output_writers[:velocity] =
    JLD2Writer(model, (; u, v, w),
               filename = filename * "Velocity.jld2",
               indices = (:, 1, :),
               schedule = TimeInterval(200),
               overwrite_existing = true,
               with_halos = false)
simulation.output_writers[:b] =
    JLD2Writer(model, (; b),
               filename = filename * "Buoyancy.jld2",
               indices = (:, 1, :),
               schedule = TimeInterval(200),
               overwrite_existing = true,
               with_halos = false)

# Horizontally-averaged velocities & buoyancy
u_avg = Field(Average(u, dims=(1, 2)))
v_avg = Field(Average(v, dims=(1, 2)))
w_avg = Field(Average(w, dims=(1, 2)))
b_avg = Field(Average(b, dims=(1, 2)))


# Horizontally-averaged buoyancy gradient ∂b/∂z
db_dz_avg = Field(Average(∂z(b), dims=(1,2)))

# Horizontally-averaged vorticity
ωx_avg = Field(Average(∂y(w)-∂z(v), dims=(1, 2)))
ωy_avg = Field(Average(∂z(u)-∂x(w), dims=(1, 2)))
ωz_avg = Field(Average(∂x(v)-∂y(u), dims=(1, 2)))

simulation.output_writers[:avg_db_dz] =
    JLD2Writer(model, (; db_dz = db_dz_avg),
                filename = filename * "Avg_grad_b.jld2",
                schedule = TimeInterval(100),
                overwrite_existing = true)
simulation.output_writers[:avg_b] =
    JLD2Writer(model, (; b = b_avg),
                filename = filename * "Avg_b.jld2",
                schedule = TimeInterval(200),
                overwrite_existing = true)
simulation.output_writers[:avg_velocity] =
    JLD2Writer(model, (; u_avg, v_avg, w_avg),
                filename = filename * "Avg_vel.jld2",
                schedule = TimeInterval(200),
                overwrite_existing = true)
simulation.output_writers[:avg_vorticity] =
    JLD2Writer(model, (; ωx_avg, ωy_avg, ωz_avg),
                filename = filename * "Avg_vort.jld2",
                schedule = TimeInterval(200),
                overwrite_existing = true)

# # Outputting diffusivity fields νₑ, κₑ [to be fixed]
# simulation.output_writers[:diffusivity_fields] =
#     JLD2Writer(model, model.diffusivity_fields,
#                filename = filename * "Diffusivity_fields.jld2",
#                schedule = TimeInterval(200),
#                overwrite_existing = true,
#                with_halos = false)

nothing # hide

# ---------------- Second moments ----------------
# THE WHOLE POINT OF THIS COPY. Included last, so every field it references is
# already bound, and before run! so its writer is registered. It adds the 13
# plane-averaged profiles K_T is measured from and changes nothing else.
if get(ENV, "MOMENTS", "1") == "1"
    include(joinpath(@__DIR__, "Moments.jl"))
else
    @info "MOMENTS=0 — no second-moment output; K_T cannot be measured from this run"
end

if EKMAN_SMOKE
    simulation.stop_iteration = SMOKE_ITERS
    simulation.stop_time = Inf
    @info "Smoke test: stopping after $SMOKE_ITERS iterations"
end

# Now, run the simulation
run!(simulation)

# A smoke run has no meaningful log layer to fit and no animation worth making.
EKMAN_SMOKE && (@info "Smoke test complete — skipping the u* fit and the figures"; exit(0))


## Finding friction velocity, u*, and friction length, z₀
vel_file_path = joinpath(filename, "Avg_vel.jld2")
vel_file  = jldopen(vel_file_path, "r")
time_keys = keys(vel_file["timeseries/t"])
last_iter = parse(Int, time_keys[end])

u_avg_data = vel_file["timeseries/u_avg/$last_iter"][1, 1, :]
v_avg_data = vel_file["timeseries/v_avg/$last_iter"][1, 1, :]
close(vel_file)

# Fit logarithmic profile: U(z) = (u*/κ) * ln(d) - (u*/κ) * ln(z₀)
function fit_log_layer(grid, u_avg, v_avg; κ=0.41, n_points=5)
    # Extract vertical center points near the wall
    z = Array(znodes(grid, Center()))[1:n_points]

    # Horizontal mean speed U = sqrt(u_avg² + v_avg²)
    U = @. sqrt(u_avg[1:n_points]^2 + v_avg[1:n_points]^2)

    # Design matrix: U = A * ln(z) + B
    X = [log.(z) ones(n_points)]

    # Least-squares regression
    coeff = X \ U
    A, B = coeff[1], coeff[2]

    # Recover u* and z0
    u_star_fit = A * κ
    z0_fit     = exp(-B / A)

    # R² score
    U_pred = X * coeff
    SS_res = sum((U .- U_pred).^2)
    SS_tot = sum((U .- mean(U)).^2)
    r2     = 1.0 - (SS_res / SS_tot)

    return u_star_fit, z0_fit, r2
end

const n_points_fit = 5         # number of points near the bottom to fit
u_star_fit, z₀_fit, r2 = fit_log_layer(grid, u_avg_data, v_avg_data; κ=κ, n_points=n_points_fit)

const u_star  = u_star_fit      # friction velocity
δ       = u_star/f₀             # boundary layer lengthscale
Re_star = u_star*δ/ν₀           # frictional Reynolds
Ri_star = N²/f₀^2               # frictional Richardson

# This folder, not "Ekman/3D Simulation/Parameters/" — nothing here writes into
# the folder it was copied from.
param_dir = joinpath(@__DIR__, "Parameters") * "/"
mkpath(param_dir)
open(param_dir*@sprintf("%d r=%.1f T=%.1f parameters.txt",profile,r,T), "w") do file
    write(file, @sprintf(
"Dimensions                      %.1f m × %.1f m × %.1f m
Grid size                       %d × %d × %d
Far stream velocity             U∞  = %.4f
Square buoyancy frequency:      N²  = %.2e,
Coriolis parameter:             f   = %.2e,
Ratio:                          r   = N/f = %.1f
Molecular kinematic viscosity:  ν₀  = %.2e,
Reynolds number:                Re∞ = %.2e,
Prandtl number:                 Pr  = %.1f,
Molecular diffusivity:          κ₀  = %.2e,
Frictional velocity             u*  = %.2e
Drag coefficient:               cᴰ  = %.4f,
Layer lengthscale:              δ   = %.2f
Friction Reynolds               Re* = %.2e
Friction Richardson             Ri* = %.1f",
Lx, Ly, Lz, Nx, Ny, Nz, U∞, N², f₀, r, ν₀, Re∞, Pr, κ₀, u_star, cᴰ, δ, Re_star, Ri_star))
end

# Opt-in: the batch driver runs these as its own step so a failed animation does
# not cost the simulation, and so they can be redrawn without re-running it.
#     PLOTS=1 julia --project=<repo root> "Ekman/TKE analysis/Ekman3D.jl"
if get(ENV, "PLOTS", "0") == "1"
    include(joinpath(@__DIR__, "Ekman_anim.jl"))
    include(joinpath(@__DIR__, "Ekman_plot.jl"))
else
    @info "PLOTS=0 — simulation only; run Figures.jl to draw the animations and plots"
end