using Printf
using Oceananigans
using CUDA

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
#   - Paper's own dimensional parameters and domain (see case_params.jl):
#     U₀ = 1.5 cm/s, ν = 10⁻⁶, δ_s = 0.119 m, 50 δ_s × 25 δ_s × 90 δ_s with
#     the sponge occupying 70 δ_s – 90 δ_s
#   - Sponge rate = 20ω (paper's peak damping), previously 6× weaker
#   - CFL = 0.72 (paper value)
#   - Ri = 0 carries the thermal field as a genuine passive scalar (buoyancy
#     term off, background gradient retained), as the paper does — so its
#     figure 4/5 panels contain real data
#   - Paper protocol: cases initialize u, v, w from a turbulent Ri=0 snapshot
#     with a fresh background b profile, mimicking "turn on stratification
#     after turbulent spin-up"
#   - Background stratification is exponential rather than uniform:
#     N²_bg(z) = N∞²[1 − exp(−z/L)] with L = 10 δ_s, so the seabed starts
#     unstratified and the far field is the paper's N∞² (see case_params.jl).
#     The uniform-background version is preserved in "Centered - Linear/".
#
# Each case writes everything into output_<case>/ with labeled filenames.

include(joinpath(@__DIR__, "case_params.jl"))

# ---------------- Architecture ----------------
arch = GPU()          # start Julia with `julia -t auto`

# OLD (Gayen reproduction): 64, 64, 256. NEW: reduced by default for the
# L_strat sweep — 8 runs at ~8 periods is only affordable at coarser resolution.
# The new regime (U₀ = 4 cm/s, ω = 1e-4) has Re_s ≈ 5640 > the paper's 1788, so
# this is a deliberate resolution compromise, not paper-fidelity. 
Nx = 48
Ny = 48
Nz = 192

n_frames = 200 * n_periods          # animation frames (same cadence per period)
duration = n_periods * T_tide
max_Δt   = 100.0

# ---------------- Grid (bottom-refined stretching) ----------------
# OLD: the two-parameter Oceananigans stretching below was effectively inert —
# it gave Δz = 0.0869 m at the wall growing only to 0.0925 m at z = 8 m, i.e. a
# uniform grid. That put the first cell face at Δz⁺ ≈ 11 against the paper's
# Δz_min⁺ = 2, leaving the wall layer unresolved: turbulence relaminarized
# during the accelerating phase (u_rms/U₀ ≈ 0.037 at φ = 0–30° vs the paper's
# ≈ 0.10 in their figure 3) and re-transitioned each half-cycle, which is what
# produced the staircase mixed-layer growth in Figure4_reproduction2.
#
# refinement = 1.8
# stretching = 10
#
# h(k) = (Nz + 1 - k) / Nz
# ζ(k) = 1 + (h(k) - 1) / refinement
# Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))
# z_faces(k) = -Lz * (ζ(k) * Σ(k) - 1)

# NEW: build the faces by integrating a prescribed Δz(z) so resolution can be
# placed per region rather than by fitting one analytic curve. Control points
# are given in Stokes thicknesses, (z/δ_s, Δz/δ_s), so the grid is defined in
# wall-relevant units and survives any change of dimensional scaling:
#   wall layer      Δz = 0.050 δ_s  (Δz⁺ ≈ 5, first centre z⁺ ≈ 2.5; paper: 2)
#   mixed layer     Δz = 0.24  δ_s at z/δ = 15  (Δz⁺ ≈ 24, paper's max: 20)
#   wave region     Δz = 0.70  δ_s at z/δ = 40
#   sponge          Δz = 2.0   δ_s above z/δ = 70 (damped, not analysed)
# With Nz = 192 the uniform rescale factor comes out ≈ 1.02, i.e. essentially
# the profile above: ~103 cells sit below z/δ = 15 and ~19 below z/δ = 1.
# Neighbouring cells differ by ≤ 4 %, keeping the second-order vertical
# truncation error small.
const dz_control_δ = [(0.0, 0.050), (1.25, 0.075), (5.0, 0.165),
                      (15.0, 0.240), (40.0, 0.700), (70.0, 2.000)]
const dz_control = [(zδ * δ, dδ * δ) for (zδ, dδ) in dz_control_δ]

function dz_target(z)
    for i in 1:length(dz_control)-1
        (z0, d0) = dz_control[i]
        (z1, d1) = dz_control[i+1]
        z <= z1 && return d0 + (d1 - d0) * (z - z0) / (z1 - z0)
    end
    return dz_control[end][2]
end

# March out Nz cells, then rescale Δz uniformly (bisection on the scale factor)
# so the last face lands exactly on Lz.
function build_z_faces(Nz, Lz)
    march(m) = (g = [0.0]; while length(g) - 1 < Nz
                    push!(g, g[end] + m * dz_target(g[end]))
                end; g)
    lo, hi = 0.5, 3.0
    for _ in 1:80
        m = (lo + hi) / 2
        march(m)[end] < Lz ? (lo = m) : (hi = m)
    end
    g = march((lo + hi) / 2)
    return g .* (Lz / g[end])
end

zf = build_z_faces(Nz, Lz)

grid = RectilinearGrid(arch;
                       topology = (Periodic, Periodic, Bounded),
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = zf)

Δz_bottom = minimum(abs.(diff(zf)))
n_test = count(z -> z <= Lz_test, zf) - 1
@info @sprintf("Bottom Δz = %.4f m = %.3f δ_s (%.1f points across δ_s); %d of %d cells below the sponge",
               Δz_bottom, Δz_bottom / δ, δ / Δz_bottom, n_test, Nz)
# Horizontal resolution, for the record: at u_τ/U₀ ≈ 0.056 the Stokes layer is
# δ_s⁺ ≈ 100, so Δx⁺ ≈ 104 and Δy⁺ ≈ 52 here against the paper's 60 and 30
# (they afford 64×64 with spectral horizontal accuracy). This is the main
# resolution compromise made to fit three cases into one night.
@info @sprintf("Δx = %.4f m = %.2f δ_s, Δy = %.4f m = %.2f δ_s",
               Lx / Nx, Lx / (Nx * δ), Ly / Ny, Ly / (Ny * δ))

# ---------------- Boundary conditions ----------------
# No-slip bottom for u and v; free-slip top (default).
u_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0))
v_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0))

# Adiabatic bottom (paper); fixed gradient at top so the background
# stratification is maintained there. N²_ref rather than N² so the Ri = 0
# passive scalar keeps its background gradient too.
#
# OLD (uniform background): the top gradient was the far-field value itself.
# b_bcs = FieldBoundaryConditions(top    = GradientBoundaryCondition(N²_ref),
#                                 bottom = FluxBoundaryCondition(0))
#
# NEW: with the exponential background the imposed gradient is N²_bg evaluated
# at the lid. At z = Lz = 90 δ_s with L = 10 δ_s this is 0.9999 N²_ref, so the
# change is numerically tiny — it is made so the BC follows the profile
# definition rather than coincidentally agreeing with it.
b_bcs = FieldBoundaryConditions(top    = GradientBoundaryCondition(N²_background(Lz)),
                                bottom = FluxBoundaryCondition(0))

# ---------------- Forcing ----------------
# Body force du/dt = U₀ ω cos(ωt) drives a free-stream velocity U₀ sin(ωt).
@inline tidal_forcing(x, y, z, t, p) = p.U₀ * p.ω * cos(p.ω * t)
u_tide = Forcing(tidal_forcing, parameters = (; U₀, ω))

# Sponge layer at the top of the domain: damps internal waves radiated by the
# boundary layer so they don't reflect off the rigid lid. Rate = 20ω is the
# paper's peak damping coefficient.
#
# The Gaussian shape is unchanged; only its width is rescaled to the paper's
# domain, where the sponge occupies 70 δ_s – 90 δ_s. σ = 8 δ_s puts the mask at
# 4.4 % on the sponge floor (z = Lz_test) and below 10⁻⁸ anywhere in the
# analysed region z < 40 δ_s, so the test section evolves freely.
const sponge_width = 8δ
sponge_rate = 20ω
@inline top_mask(x, y, z) = exp(-(z - Lz)^2 / (2 * sponge_width^2))

u_sponge = Relaxation(rate = sponge_rate, mask = top_mask,
                      target = (x, y, z, t) -> U₀ * sin(ω * t))
v_sponge = Relaxation(rate = sponge_rate, mask = top_mask)          # target 0
w_sponge = Relaxation(rate = sponge_rate, mask = top_mask)          # target 0
# OLD (uniform background): b_sponge relaxed towards the linear ramp.
# b_sponge = Relaxation(rate = sponge_rate, mask = top_mask,
#                       target = (x, y, z, t) -> N²_ref * z)
# NEW: relax towards the exponential background so the sponge does not fight the
# profile the interior is initialized with. Deep in the sponge the two targets
# differ by the constant N²_ref·L, which would otherwise be forced onto the
# solution as a spurious offset at the lid.
b_sponge = Relaxation(rate = sponge_rate, mask = top_mask,
                      target = (x, y, z, t) -> b_background(z))

# ---------------- Model ----------------
# AMD provides the explicit SGS stresses/fluxes; the ScalarDiffusivity carries
# the molecular ν and κ = ν/Pr on top of it.
model = NonhydrostaticModel(grid;
            # OLD: advection = WENO(order = 5) — WENO's upwind stencil adds
            # numerical dissipation on top of the AMD closure, so the SGS model
            # is no longer the only sink. With marginally turbulent phases that
            # double-damping helps kill the flow during acceleration. The paper
            # uses a non-dissipative pseudo-spectral/central scheme and lets the
            # dynamic mixed model carry all the dissipation.
            # advection   = WENO(order = 5),
            advection   = Centered(order = 2),
            timestepper = :RungeKutta3,
            tracers     = (:b, :c),
            # Ri = 0 is the paper's "passive scalar case": the thermal field is
            # advected and diffused with the same background gradient but exerts
            # no buoyancy force. Dropping the buoyancy term (rather than setting
            # N² = 0 and leaving b ≡ 0) is what makes the Ri = 0 panels of
            # figures 4 and 5 show an actual scalar field.
            buoyancy    = passive_scalar ? nothing : BuoyancyTracer(),
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
# OLD (uniform background): bᵢ(x, y, z) = N²_ref * z
# OLD (exponential background): bᵢ(x, y, z) = b_background(z)   [= N∞²(z + L(e^{−z/L}−1))]
# NEW: linear stratification with a bottom mixed layer of depth T pre-carved in
# (case_params.jl): bᵢ = N∞² z [1 − exp(−0.2 (z/T)^5)]. The background the sponge
# and top BC relax toward is now the plain linear ramp b_background(z) = N∞² z.
bᵢ(x, y, z) = b_initial(z)
cᵢ(x, y, z) = exp(-((x - Lx/2) / (Lx/50))^2)   # thin dye sheet at mid-domain

# The spin-up snapshot supplies velocities only. Ri = 0 carries b as a passive
# scalar, so its velocity field is independent of the thermal profile: a
# turbulent snapshot from a previous run is a valid restart state for every case
# here, Ri = 0 included, which is what lets the exponential-background run skip
# a fresh spin-up phase entirely.
#
# SPINUP_FILE pins the source (e.g. the archived linear run) so that a
# concurrently running Ri0 job cannot overwrite the file the stratified cases
# are reading from.
spinup_file = get(ENV, "SPINUP_FILE",
                  joinpath("output_Ri0", "TidalBL3D_Ri0_fields.jld2"))

if isfile(spinup_file)
    # Paper protocol: turbulent unstratified spin-up, then stratification on.
    # Snapshots are written every half period, i.e. always at U∞ = 0 — the same
    # phase this run starts from — so the restart is phase-consistent.
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
    @warn "No spin-up snapshot at $spinup_file — starting $case from rest with noise."
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

# OLD: cfl = 0.95 — contradicted the paper value quoted in the header, and is
# aggressive now that advection is centered (non-dissipative, so grid-scale
# noise is no longer damped by an upwind stencil).
# wizard = TimeStepWizard(cfl = 0.95, max_change = 1.2, max_Δt = max_Δt)
wizard = TimeStepWizard(cfl = 0.72, max_change = 1.2, max_Δt = max_Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

start_time = time_ns()
# flush: stdout is block-buffered when redirected to a file, so without this an
# overnight run's log stays empty for tens of minutes and the driver cannot read
# how far along the case is.
progress(sim) = (@printf("i: %6d, t: %9.1f s, ωt: %6.2f (%.2f periods), wall: %10s, Δt: %6.2f, CFL: %.2e\n",
                         sim.model.clock.iteration,
                         sim.model.clock.time,
                         ω * sim.model.clock.time,
                         sim.model.clock.time / T_tide,
                         prettytime(1e-9 * (time_ns() - start_time)),
                         sim.Δt,
                         AdvectiveCFL(sim.Δt)(sim.model));
                 flush(stdout))
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
