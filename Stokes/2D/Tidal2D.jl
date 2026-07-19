using Printf
using Oceananigans

# Tidal (Stokes) boundary layer following Gayen et al. (2009)
# z runs from 0 (bottom wall) to Lz (top), grid refined near the bottom.

# ---------------- Physical & numerical parameters ----------------
Lx, Ly, Lz = 20, 20, 30        # domain size (m); Ly only used if you go 3D
Nx, Ny, Nz = 64, 64, 256       # Ny only used if you go 3D

ω  = 1.4075235e-4              # M2 tidal frequency (s⁻¹), period ≈ 12.4 h
U₀ = 0.05                      # tidal velocity amplitude (m s⁻¹)
N² = 1e-7                      # background stratification (s⁻²)
ν  = 1.109e-5                  # viscosity (m² s⁻¹)
κ  = ν                         # diffusivity

δ      = sqrt(2ν / ω)          # laminar Stokes layer thickness
Re_δ   = U₀ * δ / ν            # Stokes Reynolds number (transition ≈ 500-800)
T_tide = 2π / ω

@info @sprintf("Stokes layer δ = %.3f m, Re_δ = %.0f, tidal period = %.0f s",
               δ, Re_δ, T_tide)

n_periods = 4
duration  = n_periods * T_tide
max_Δt    = 100.0

# ---------------- Grid (bottom-refined stretching) ----------------
refinement = 1.8
stretching = 10

h(k) = (Nz + 1 - k) / Nz
ζ(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))
z_faces(k) = -Lz * (ζ(k) * Σ(k) - 1)

grid = RectilinearGrid(topology = (Periodic, Flat, Bounded),
                       size = (Nx, Nz),
                       x = (0, Lx),
                       z = z_faces)

# For 3D (needed for real turbulence) and/or GPU, use instead:
# grid = RectilinearGrid(GPU(); topology = (Periodic, Periodic, Bounded),
#                        size = (Nx, Ny, Nz), x = (0, Lx), y = (0, Ly), z = z_faces)

Δz_bottom = zspacings(grid, Center())[1]
@info @sprintf("Bottom Δz = %.4f m (%.1f points across δ); Δx = %.3f m",
               Δz_bottom, δ / Δz_bottom, Lx / Nx)

# ---------------- Boundary conditions ----------------
# No-slip bottom for u and v; free-slip top (default).
u_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0))
v_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0))

# Insulating bottom (default anyway, but explicit), fixed gradient N² at top
# so the background stratification is maintained there.
b_bcs = FieldBoundaryConditions(top    = GradientBoundaryCondition(N²),
                                bottom = FluxBoundaryCondition(0))

# ---------------- Forcing ----------------
# Body force du/dt = U₀ ω cos(ωt) drives a free-stream velocity U₀ sin(ωt)
# (starting from rest at t = 0).
@inline tidal_forcing(x, z, t, p) = p.U₀ * p.ω * cos(p.ω * t)
u_tide = Forcing(tidal_forcing, parameters = (; U₀, ω))

# Sponge layer in the top ~5 m: damps internal waves radiated by the boundary
# layer so they don't reflect off the rigid lid and re-enter the domain.
sponge_width = 5.0
sponge_rate  = 1 / 2000               # ≈ 20 damping times per tidal period
@inline top_mask(x, z) = exp(-(z - Lz)^2 / (2 * sponge_width^2))

u_sponge = Relaxation(rate = sponge_rate, mask = top_mask,
                      target = (x, z, t) -> U₀ * sin(ω * t))
w_sponge = Relaxation(rate = sponge_rate, mask = top_mask)          # target 0
b_sponge = Relaxation(rate = sponge_rate, mask = top_mask,
                      target = (x, z, t) -> N² * z)

# ---------------- Model ----------------
model = NonhydrostaticModel(grid;
            advection   = WENO(order = 5),
            timestepper = :RungeKutta3,
            tracers     = (:b, :c),
            buoyancy    = BuoyancyTracer(),
            closure     = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(),
                                            ν = ν, κ = κ),
            boundary_conditions = (u = u_bcs, v = v_bcs, b = b_bcs),
            coriolis    = nothing,
            forcing     = (u = (u_tide, u_sponge), w = w_sponge, b = b_sponge))

# ---------------- Initial conditions ----------------
# Confine the random kick to the near-wall region (within a few Stokes layers)
# so we perturb the boundary layer without filling the stratified interior
# (and the sponge region) with noise.
kick = 0.1 * U₀
damped_noise(z) = kick * randn() * exp(-z / (4δ))

uᵢ(x, z) = damped_noise(z)
wᵢ(x, z) = damped_noise(z)
bᵢ(x, z) = N² * z
cᵢ(x, z) = exp(-((x - Lx/2) / (Lx/50))^2)   # thin dye streak at mid-domain

set!(model, u = uᵢ, w = wᵢ, b = bᵢ, c = cᵢ)

# ---------------- Simulation ----------------
simulation = Simulation(model, Δt = 1.0, stop_time = duration)

wizard = TimeStepWizard(cfl = 0.85, max_change = 1.1, max_Δt = max_Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

start_time = time_ns()
progress(sim) = @printf("i: %6d, t: %10.1f s (%.2f periods), wall: %10s, Δt: %6.2f, CFL: %.2e\n",
                        sim.model.clock.iteration,
                        sim.model.clock.time,
                        sim.model.clock.time / T_tide,
                        prettytime(1e-9 * (time_ns() - start_time)),
                        sim.Δt,
                        AdvectiveCFL(sim.Δt)(sim.model))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# ---------------- Output ----------------
u, v, w = model.velocities
b = model.tracers.b
c = model.tracers.c

filename = "TidalBoundaryLayer2D"

# (1) x-z slices for animation. 1200 frames ≈ 2.5 min of video at 8 fps;
# raise n_frames if you want smoother animation.
n_frames = 1200
simulation.output_writers[:xz_slices] =
    JLD2Writer(model, (; u, w, b, c),
               filename = filename * ".jld2",
               indices = (:, 1, :),
               schedule = TimeInterval(duration / n_frames),
               overwrite_existing = true,
               with_halos = false)

# (2) Horizontally averaged profiles + turbulence statistics.
# These are 1D so we can afford high output frequency (200 per tidal period).
U  = Field(Average(u, dims = (1, 2)))          # mean velocity profile
B  = Field(Average(b, dims = (1, 2)))          # mean buoyancy profile
dbdz = Field(∂z(B))                            # mean stratification profile
uw = Field(Average(w * u, dims = (1, 2)))      # vertical momentum flux ⟨uw⟩ ≈ u'w'
ww = Field(Average(w^2, dims = (1, 2)))        # vertical velocity variance

simulation.output_writers[:profiles] =
    JLD2Writer(model, (; U, B, dbdz, uw, ww),
               filename = filename * "_profiles.jld2",
               schedule = TimeInterval(T_tide / 200),
               overwrite_existing = true,
               with_halos = false)

run!(simulation)