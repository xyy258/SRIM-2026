using Printf
using Oceananigans

# 3D Tidal (Stokes) boundary layer following Gayen, Sarkar & Taylor (2010).
# Run one case with:   julia -t auto --project=. Tidal3D.jl Ri0
# (then Ri500, Ri2500 — those restart from the Ri0 turbulent state).
#
# Changes for fidelity to the paper (vs the earlier exploratory version):
#   - Cases defined by Ri = N²/ω² ∈ {0, 500, 2500} at Re_s = 1790
#     (previous N² = 1e-7 was Ri ≈ 5, effectively unstratified)
#   - Explicit SGS closure: AnisotropicMinimumDissipation alongside the
#     molecular ScalarDiffusivity (the paper uses a dynamic mixed model and
#     notes plain Smagorinsky fails for this flow; AMD is the closest
#     Oceananigans equivalent)
#   - Molecular Pr = 0.7 (κ = ν/Pr), paper value
#   - Ly = 10 m ≈ 25 δ_s (halves Δy⁺ toward the paper's ≈30)
#   - Sponge rate = 20ω (paper's peak damping), previously 6× weaker
#   - CFL = 0.72 (paper value)
#   - Paper protocol: stratified cases initialize u, v, w from the final
#     Ri=0 turbulent snapshot with a fresh linear b = N²z, mimicking
#     "turn on stratification after turbulent spin-up"
#
# Each case writes everything into output_<case>/ with labeled filenames.

include(joinpath(@__DIR__, "case_params.jl"))

# ---------------- Architecture ----------------
arch = CPU()          # start Julia with `julia -t auto`

Nx, Ny, Nz = 48, 48, 192

n_frames = 200 * n_periods          # animation frames (same cadence per period)
duration = n_periods * T_tide
max_Δt   = 100.0

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

# Adiabatic bottom (paper); fixed gradient N² at top so the background
# stratification is maintained there.
b_bcs = FieldBoundaryConditions(top    = GradientBoundaryCondition(N²),
                                bottom = FluxBoundaryCondition(0))

# ---------------- Forcing ----------------
# Body force du/dt = U₀ ω cos(ωt) drives a free-stream velocity U₀ sin(ωt).
@inline tidal_forcing(x, y, z, t, p) = p.U₀ * p.ω * cos(p.ω * t)
u_tide = Forcing(tidal_forcing, parameters = (; U₀, ω))

# Sponge layer in the top ~5 m: damps internal waves radiated by the boundary
# layer so they don't reflect off the rigid lid. Rate = 20ω is the paper's
# peak damping coefficient.
const sponge_width = 5.0
sponge_rate = 20ω
@inline top_mask(x, y, z) = exp(-(z - Lz)^2 / (2 * sponge_width^2))

u_sponge = Relaxation(rate = sponge_rate, mask = top_mask,
                      target = (x, y, z, t) -> U₀ * sin(ω * t))
v_sponge = Relaxation(rate = sponge_rate, mask = top_mask)          # target 0
w_sponge = Relaxation(rate = sponge_rate, mask = top_mask)          # target 0
b_sponge = Relaxation(rate = sponge_rate, mask = top_mask,
                      target = (x, y, z, t) -> N² * z)

# ---------------- Model ----------------
# AMD provides the explicit SGS stresses/fluxes; the ScalarDiffusivity carries
# the molecular ν and κ = ν/Pr on top of it.
model = NonhydrostaticModel(grid;
            advection   = WENO(order = 5),
            timestepper = :RungeKutta3,
            tracers     = (:b, :c),
            buoyancy    = BuoyancyTracer(),
            closure     = (AnisotropicMinimumDissipation(),
                           ScalarDiffusivity(VerticallyImplicitTimeDiscretization(),
                                             ν = ν, κ = κ)),
            boundary_conditions = (u = u_bcs, v = v_bcs, b = b_bcs),
            coriolis    = nothing,
            forcing     = (u = (u_tide, u_sponge),
                           v = v_sponge,
                           w = w_sponge,
                           b = b_sponge))

# ---------------- Initial conditions ----------------
bᵢ(x, y, z) = N² * z
cᵢ(x, y, z) = exp(-((x - Lx/2) / (Lx/50))^2)   # thin dye sheet at mid-domain

spinup_file = joinpath("output_Ri0", "TidalBL3D_Ri0_fields.jld2")

if Ri > 0 && isfile(spinup_file)
    # Paper protocol: turbulent unstratified spin-up, then stratification on.
    # The Ri=0 run ends at t = 6 T_tide where U∞ = 0, the same phase this run
    # starts from, so the restart is phase-consistent.
    @info "Initializing velocities from Ri=0 spin-up: $spinup_file"
    uts = FieldTimeSeries(spinup_file, "u"; backend = OnDisk())
    vts = FieldTimeSeries(spinup_file, "v"; backend = OnDisk())
    wts = FieldTimeSeries(spinup_file, "w"; backend = OnDisk())
    nlast = length(uts.times)
    @info @sprintf("Using snapshot %d/%d (t = %.2f periods of the spin-up)",
                   nlast, nlast, uts.times[nlast] / T_tide)
    set!(model, u = Array(interior(uts[nlast])),
                v = Array(interior(vts[nlast])),
                w = Array(interior(wts[nlast])))
    set!(model, b = bᵢ, c = cᵢ)
else
    Ri > 0 && @warn "No Ri=0 spin-up snapshot found — starting $case from rest with noise."
    # Near-wall random kick; perturbing v breaks spanwise symmetry so genuine
    # 3D turbulence develops.
    kick = 0.1 * U₀
    damped_noise(z) = kick * randn() * exp(-z / (4δ))
    uᵢ(x, y, z) = damped_noise(z)
    vᵢ(x, y, z) = damped_noise(z)
    wᵢ(x, y, z) = damped_noise(z)
    set!(model, u = uᵢ, v = vᵢ, w = wᵢ, b = bᵢ, c = cᵢ)
end

# ---------------- Simulation ----------------
simulation = Simulation(model, Δt = 1.0, stop_time = duration)

# Quick correctness check without the full run: TIDAL_SMOKE=1 julia Tidal3D.jl
if get(ENV, "TIDAL_SMOKE", "0") == "1"
    simulation.stop_iteration = 20
    @info "Smoke test: stopping after 20 iterations"
end

wizard = TimeStepWizard(cfl = 0.72, max_change = 1.1, max_Δt = max_Δt)
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

slice_schedule = TimeInterval(duration / n_frames)

# (1) x-z slices at y = 0 for animation.
simulation.output_writers[:xz_slices] =
    JLD2Writer(model, (; u, w, b, c),
               filename = filename * ".jld2",
               indices = (:, 1, :),
               schedule = slice_schedule,
               overwrite_existing = true,
               with_halos = false)

# (2) x-y slice at z ≈ δ: plan view of the near-wall streaks/bursts.
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
# Second moments are saved raw (⟨u²⟩, ⟨uw⟩...); subtract the mean in
# post-processing, e.g. u'w' = ⟨uw⟩ − U W.
U  = Field(Average(u, dims = (1, 2)))
V  = Field(Average(v, dims = (1, 2)))
B  = Field(Average(b, dims = (1, 2)))
uw = Field(Average(u * w, dims = (1, 2)))
vw = Field(Average(v * w, dims = (1, 2)))
wb = Field(Average(w * b, dims = (1, 2)))
uu = Field(Average(u^2,  dims = (1, 2)))
vv = Field(Average(v^2,  dims = (1, 2)))
ww = Field(Average(w^2,  dims = (1, 2)))

simulation.output_writers[:profiles] =
    JLD2Writer(model, (; U, V, B, uw, vw, wb, uu, vv, ww),
               filename = filename * "_profiles.jld2",
               schedule = TimeInterval(T_tide / 200),
               overwrite_existing = true,
               with_halos = false)

# (4) Full 3D snapshots twice per tidal period — used both for re-analysis
# and as the turbulent initial condition for the stratified cases.
simulation.output_writers[:fields3d] =
    JLD2Writer(model, (; u, v, w, b, c),
               filename = filename * "_fields.jld2",
               schedule = TimeInterval(T_tide / 2),
               overwrite_existing = true,
               with_halos = false)

# (5) Checkpointer: keeps only the most recent checkpoint.
simulation.output_writers[:checkpoint] =
    Checkpointer(model,
                 schedule = TimeInterval(T_tide / 2),
                 dir = outdir,
                 prefix = "TidalBL3D_" * case * "_checkpoint",
                 cleanup = true)

run!(simulation)
# To resume an interrupted run, use:  run!(simulation, pickup = true)
# (and set overwrite_existing = false in the writers first)
