using Oceananigans, Printf
using CUDA

# 3D Tidal (Stokes) boundary layer following Gayen et al. (2009)
# z runs from 0 (bottom wall) to Lz (top), grid refined near the bottom.
#
# Changes from the 2D version:
#   - (Periodic, Periodic, Bounded) topology with a y dimension
#   - forcing / sponge / IC functions take (x, y, z, ...) signatures
#   - v is also perturbed initially (spanwise noise is what lets 3D
#     turbulence develop; without it the flow stays effectively 2D)
#   - extra outputs: x-y slice at z ≈ δ, full 3D snapshots once per period,
#     spanwise statistics (V, vw, uu, vv)
#   - Checkpointer + wall_time_limit so an overnight run can't be lost
#
# Physical parameters are declared `const` so closures that capture them
# (sponge targets, masks) compile cleanly into GPU kernels.

# ---------------- Architecture ----------------
arch = CPU()          # <-- set to CPU() if you have no CUDA/ROCm GPU
                      #     (and start Julia with `julia -t auto`)

# ---------------- Physical & numerical parameters ----------------
const Lx = 20.0                # domain size (m)
const Ly = 20.0
const Lz = 30.0

Nx, Ny, Nz = 48, 48, 150

const ω  = 1.4075235e-4        # M2 tidal frequency (s⁻¹), period ≈ 12.4 h
const U₀ = 0.05                # tidal velocity amplitude (m s⁻¹)
const N² = 1e-7                # background stratification (s⁻²)
const ν  = 1.109e-5            # viscosity (m² s⁻¹)
const κ  = ν                   # diffusivity

const δ      = sqrt(2ν / ω)    # laminar Stokes layer thickness
const T_tide = 2π / ω
Re_δ = U₀ * δ / ν              # Stokes Reynolds number (transition ≈ 500-800)

@info @sprintf("Stokes layer δ = %.3f m, Re_δ = %.0f, tidal period = %.0f s",
               δ, Re_δ, T_tide)

n_periods = 6
duration  = n_periods * T_tide
max_Δt    = 100.0

# ---------------- Grid (bottom-refined stretching) ----------------
refinement = 1.8
stretching = 10

h(k) = (Nz + 1 - k) / Nz
ζ(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))
z_faces(k) = -Lz * (ζ(k) * Σ(k) - 1)

grid = RectilinearGrid(arch;
                       topology = (Periodic, Periodic, Bounded),
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_faces)

Δz_bottom = minimum(Array(zspacings(grid, Center())))
@info @sprintf("Bottom Δz = %.4f m (%.1f points across δ); Δx = %.3f m, Δy = %.3f m",
               Δz_bottom, δ / Δz_bottom, Lx / Nx, Ly / Ny)

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
# (starting from rest at t = 0). Note the (x, y, z, t, p) signature in 3D.
@inline tidal_forcing(x, y, z, t, p) = p.U₀ * p.ω * cos(p.ω * t)
u_tide = Forcing(tidal_forcing, parameters = (; U₀, ω))

# Sponge layer in the top ~5 m: damps internal waves radiated by the boundary
# layer so they don't reflect off the rigid lid and re-enter the domain.
const sponge_width = 5.0
sponge_rate = 1 / 2000                # ≈ 20 damping times per tidal period
@inline top_mask(x, y, z) = exp(-(z - Lz)^2 / (2 * sponge_width^2))

u_sponge = Relaxation(rate = sponge_rate, mask = top_mask,
                      target = (x, y, z, t) -> U₀ * sin(ω * t))
v_sponge = Relaxation(rate = sponge_rate, mask = top_mask)          # target 0
w_sponge = Relaxation(rate = sponge_rate, mask = top_mask)          # target 0
b_sponge = Relaxation(rate = sponge_rate, mask = top_mask,
                      target = (x, y, z, t) -> N² * z)

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
            forcing     = (u = (u_tide, u_sponge),
                           v = v_sponge,
                           w = w_sponge,
                           b = b_sponge))

# ---------------- Initial conditions ----------------
# Confine the random kick to the near-wall region (within a few Stokes layers)
# so we perturb the boundary layer without filling the stratified interior
# (and the sponge region) with noise. Perturbing v (not just u, w) is
# essential in 3D: it breaks spanwise symmetry so genuine 3D turbulence can
# develop instead of a 2D flow replicated in y.
kick = 0.1 * U₀
damped_noise(z) = kick * randn() * exp(-z / (4δ))

uᵢ(x, y, z) = damped_noise(z)
vᵢ(x, y, z) = damped_noise(z)
wᵢ(x, y, z) = damped_noise(z)
bᵢ(x, y, z) = N² * z
cᵢ(x, y, z) = exp(-((x - Lx/2) / (Lx/50))^2)   # thin dye sheet at mid-domain

set!(model, u = uᵢ, v = vᵢ, w = wᵢ, b = bᵢ, c = cᵢ)

# ---------------- Simulation ----------------
# wall_time_limit stops the run cleanly (all output flushed) before your
# 8 hours are up; combined with the Checkpointer below you can always
# resume from the last completed tidal period.
simulation = Simulation(model, Δt = 1.0, stop_time = duration,
                        wall_time_limit = 7.5 * 3600)

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

filename = "Stokes/TidalBoundaryLayer3D"

n_frames = 1200
slice_schedule = TimeInterval(duration / n_frames)

# (1) x-z slices at y = 0 for animation — same layout as the 2D run, so the
# existing animation script works after changing only `filename`.
simulation.output_writers[:xz_slices] =
    JLD2Writer(model, (; u, w, b, c),
               filename = filename * ".jld2",
               indices = (:, 1, :),
               schedule = slice_schedule,
               overwrite_existing = true,
               with_halos = false)

# (2) x-y slice at z ≈ δ: plan view of the near-wall turbulence (streaks,
# bursts) that simply doesn't exist in 2D — this is one of the main payoffs
# of going 3D.
zc_nodes = Array(znodes(grid, Center()))
k_δ = searchsortedfirst(zc_nodes, δ)
@info @sprintf("x-y slice at k = %d (z = %.3f m ≈ δ)", k_δ, zc_nodes[k_δ])

simulation.output_writers[:xy_slices] =
    JLD2Writer(model, (; u, v, w, b, c),
               filename = filename * "_xy.jld2",
               indices = (:, :, k_δ),
               schedule = slice_schedule,
               overwrite_existing = true,
               with_halos = false)

# (3) Horizontally averaged profiles + turbulence statistics.
# In 3D the averages are over both x and y, so ⟨uw⟩, variances etc. are
# proper turbulence statistics. Second moments are saved raw (⟨u²⟩, ⟨uw⟩...);
# subtract the mean in post-processing, e.g. u'w' = ⟨uw⟩ − U W.
U  = Field(Average(u, dims = (1, 2)))          # mean streamwise velocity
V  = Field(Average(v, dims = (1, 2)))          # mean spanwise velocity
B  = Field(Average(b, dims = (1, 2)))          # mean buoyancy
uw = Field(Average(u * w, dims = (1, 2)))      # vertical momentum flux
vw = Field(Average(v * w, dims = (1, 2)))
wb = Field(Average(w * b, dims = (1, 2)))      # turbulent buoyancy flux
uu = Field(Average(u^2,  dims = (1, 2)))
vv = Field(Average(v^2,  dims = (1, 2)))
ww = Field(Average(w^2,  dims = (1, 2)))

simulation.output_writers[:profiles] =
    JLD2Writer(model, (; U, V, B, uw, vw, wb, uu, vv, ww),
               filename = filename * "_profiles.jld2",
               schedule = TimeInterval(T_tide / 200),
               overwrite_existing = true,
               with_halos = false)

# (4) Full 3D snapshots once per tidal period, for volume rendering /
# later re-analysis. ~170 MB per snapshot at 128×128×256 with 5 fields.
simulation.output_writers[:fields3d] =
    JLD2Writer(model, (; u, v, w, b, c),
               filename = filename * "_fields.jld2",
               schedule = TimeInterval(T_tide / 2),
               overwrite_existing = true,
               with_halos = false)

# (5) Checkpointer: keeps only the most recent checkpoint (cleanup = true).
# If the run dies or hits the wall-time limit, restart with
    # run!(simulation, pickup = true)
# simulation.output_writers[:checkpoint] =
#     Checkpointer(model,
#                  schedule = TimeInterval(T_tide / 2),
#                  prefix = filename * "_checkpoint",
#                  cleanup = true)

run!(simulation)
# To resume an interrupted run, comment out the line above and use:
# run!(simulation, pickup = true)
include("Stokes/Tidal3Danimation.jl")
include("Stokes/Tidal3Dprofiles.jl")