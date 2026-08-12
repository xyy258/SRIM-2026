using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Printf
using Oceananigans
using Oceananigans.AbstractOperations: ∂x, ∂z   # for spanwise-averaged vorticity
using CUDA
# 3D Tidal (Stokes) boundary layer after Gayen, Sarkar & Taylor (2010).
#
# Run one case with:
#   PROFILE=4 T_STRAT=10 julia -t auto --project=. Tidal3D.jl sqrtRi2
# Stratified cases restart from a turbulent Ri = 0 spin-up; produce that once with
#   RUN_TAG=spinup julia -t auto --project=. Tidal3D.jl Ri0
# and every case then picks it up via spinup_default (case_params.jl). This is the
# paper's protocol: spin up unstratified, then switch stratification on.
#
# Set-up notes:
#   - Explicit SGS closure: AnisotropicMinimumDissipation alongside the molecular
#     ScalarDiffusivity. The paper uses a dynamic mixed model and notes plain
#     Smagorinsky fails for this flow; AMD is the closest Oceananigans equivalent.
#   - Molecular Pr = 10 (κ = ν/Pr). NOTE: the paper's value is 0.7; Pr = 10
#     makes scalar structure √10 ≈ 3.2× finer than velocity structure, so on
#     this grid the b field is effectively SGS-controlled.
#   - Dimensional parameters and domain (see case_params.jl): U₀ = 4 cm/s,
#     ω = 1e-4, ν = 10⁻⁶, δ_s = 0.141 m, 10 × 10 × 40 m physical with a 10 m
#     sponge on top. This is Re_s ≈ 5657, NOT the paper's 1788.
#   - The sponge sits ON TOP of the physical domain rather than occupying its
#     upper part: the grid runs to Lz_total = Lz + L_sponge and the mask is
#     exactly zero for z ≤ Lz, so the whole physical domain is undamped.
#   - CFL = 0.72 (paper value).
#   - Ri = 0 carries the thermal field as a genuine passive scalar (buoyancy term
#     off, background gradient retained), as the paper does — so its figure 4/5
#     panels contain real data.
#   - Background stratification is selected by PROFILE (case_params.jl):
#     0 linear, 1 nonlinear, 2 exponential, 3 linear+exponential decay,
#     4 softplus (default; T sets the pycnocline height, sharp its width).
#     The uniform-background version is preserved in "-Centered - Linear/".
#
# Each case writes everything into outputs/<casetag>/ with labeled filenames.

include(joinpath(@__DIR__, "case_params.jl"))

# ---------------- Architecture ----------------
arch = GPU()          # start Julia with `julia -t auto`

# Below the paper's 64×64×256: at Re_s ≈ 5657 (vs their 1788) this is a
# deliberate resolution compromise made to keep a multi-case sweep affordable.
Nx = 48
Ny = 48
n_frames = 200 * n_periods          # animation frames (same cadence per period)
duration = n_periods * T_tide
max_Δt   = parse(Float64, get(ENV, "MAX_DT", "100.0"))

# ---------------- Grid (bottom-refined stretching) ----------------
# The faces come from integrating a prescribed Δz(z), so resolution can be placed
# per region rather than by fitting one analytic curve — a near-uniform grid
# leaves the wall layer unresolved and the flow relaminarizes each half-cycle.
#
# Control points are (z/δ_s, Δz/δ_s) — wall-relevant units that survive a change
# of dimensional scaling. These are the spacings the grid ACTUALLY gets:
#   wall layer      Δz = 0.086 δ_s  (Δz⁺ ≈ 9, first centre z⁺ ≈ 4)
#   mixed layer     Δz = 0.41  δ_s at z/δ = 15
#   wave region     Δz = 1.20  δ_s at z/δ = 40
#   far field       Δz = 3.44  δ_s above z/δ = 70 (wave field, coarse)
#
# NOTE these are 1.721× the values that stood here before. That is not a change
# of grid: the old code fixed Nz_phys = 192 and bisected a uniform scale factor
# to make the last face land on Lz, and that factor came out at 1.721 — so the
# nominal "0.050 δ_s at the wall, Δz⁺ ≈ 5" was never what ran. Nz is now DERIVED
# from these spacings instead (see build_z_faces), so a control point means what
# it says and adding cells in one region no longer coarsens every other region.
const dz_control_δ = [(0.0, 0.0861), (1.25, 0.1291), (5.0, 0.2840),
                      (15.0, 0.4130), (40.0, 1.2047), (70.0, 3.4420)]

# Targeted refinement at the pycnocline (profiles 1 and 4, which put a sharp
# feature at z = T high in the domain where the grid above is ~3.4 δ_s coarse).
# Refining globally is not affordable: Δz scales as 1/Nz everywhere, so ~10 cells
# across the transition would need Nz ≈ 1400, and since Δt is limited by w/Δz AT
# THE WALL that would cut the timestep ~8× as well — a ~64× cost. Adding a fine
# band only where the feature is leaves the wall spacing (hence Δt) untouched and
# costs roughly a doubling of cells.
const w_pycno  = 2log(9) / sharp          # 10–90 % width of the db/dz transition
const n_pycno  = parse(Int, get(ENV, "N_PYCNO", "10"))   # cells wanted across it
const dz_pycno = w_pycno / n_pycno
const hw_pycno = max(2.0, 3 * w_pycno)    # half-width of a refined band (m)
const ramp_pycno = 5.0                    # ramp back to the far field over this (m)

# Bands are placed at EVERY T in the sweep, not only at this run's T, so all runs
# share one grid. Two reasons. (1) A T-sweep in which each T had its own vertical
# grid cannot separate the physics from the discretization — which is the very
# confound the refinement exists to remove. (2) The spin-up snapshot is restarted
# by array shape, so one spin-up can only seed runs on an identical grid;
# per-T grids would need one spin-up each.
#
# Set T_SWEEP to this run's T alone to get the cheaper per-T grid instead
# (Nz ≈ 254–290 rather than 458), accepting both costs above.
const T_sweep = parse.(Float64, split(get(ENV, "T_SWEEP", "5 10 20 30")))

const dz_control = [(zδ * δ, dδ * δ) for (zδ, dδ) in dz_control_δ]

function dz_base(z)
    for i in 1:length(dz_control)-1
        (z0, d0) = dz_control[i]
        (z1, d1) = dz_control[i+1]
        z <= z1 && return d0 + (d1 - d0) * (z - z0) / (z1 - z0)
    end
    return dz_control[end][2]
end

# A band: dz_pycno within ±hw of the centre, ramping back to the far-field
# spacing over the next `ramp` metres. Taking the min over bands (and with the
# base profile) composes them without any splicing logic and handles overlap.
function dz_band(z, c, dz_far)
    d = abs(z - c)
    d <= hw_pycno            && return dz_pycno
    d >= hw_pycno + ramp_pycno && return dz_far
    return dz_pycno + (dz_far - dz_pycno) * (d - hw_pycno) / ramp_pycno
end

function dz_target(z)
    d = dz_base(z)
    profile in (1, 4) || return d
    dz_far = dz_control[end][2]
    for c in T_sweep
        d = min(d, dz_band(z, c, dz_far))
    end
    return d
end

# March the prescribed Δz(z) until the column is covered — Nz is whatever that
# takes — then rescale by the (tiny, ≤ one cell in Lz) factor that lands the last
# face exactly on Lz. NZ_PHYS forces a specific count instead, in which case a
# uniform scale factor is bisected as before to make it fit.
function build_z_faces(Lz)
    march(m) = (g = [0.0]; while g[end] < Lz
                    push!(g, g[end] + m * dz_target(g[end]))
                end; g)
    forced = get(ENV, "NZ_PHYS", "")
    if isempty(forced)
        g = march(1.0)
        return g .* (Lz / g[end])
    end
    Nz = parse(Int, forced)
    marchN(m) = (g = [0.0]; while length(g) - 1 < Nz
                     push!(g, g[end] + m * dz_target(g[end]))
                 end; g)
    lo, hi = 0.05, 10.0
    for _ in 1:100
        m = (lo + hi) / 2
        marchN(m)[end] < Lz ? (lo = m) : (hi = m)
    end
    g = marchN((lo + hi) / 2)
    return g .* (Lz / g[end])
end

# The sponge layer is stacked on top of the physical grid instead of being
# carved out of it, so the physical faces are exactly what they were before this
# change: build_z_faces still marches Nz_phys cells onto 0 – Lz. Sponge cells
# have constant Δz, inheriting the last physical Δz, so there is no resolution
# jump at z = Lz — a jump there would reflect the very waves the sponge exists
# to absorb.
function append_sponge_faces(zf, L_sponge)
    Δz_top = zf[end] - zf[end-1]
    n = max(1, round(Int, L_sponge / Δz_top))
    Δz = L_sponge / n
    return vcat(zf, zf[end] .+ Δz .* (1:n))
end

zf_phys  = build_z_faces(Lz)
Nz_phys  = length(zf_phys) - 1
zf       = append_sponge_faces(zf_phys, L_sponge)
Nz       = length(zf) - 1
Nz_spnge = Nz - Nz_phys

grid = RectilinearGrid(arch;
                       topology = (Periodic, Periodic, Bounded),
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = zf)

Δz_bottom = minimum(abs.(diff(zf)))
n_test = count(z -> z <= Lz_test, zf) - 1
@info @sprintf("Bottom Δz = %.4f m = %.3f δ_s (%.1f points across δ_s); %d cells below z = %.0f δ_s; %d physical + %d sponge cells (sponge Δz = %.3f δ_s)",
               Δz_bottom, Δz_bottom / δ, δ / Δz_bottom, n_test, Lz_test / δ,
               Nz_phys, Nz_spnge, (zf[end] - zf[end-1]) / δ)
# Horizontal resolution, for the record — coarser in wall units than the paper's
# Δx⁺ = 60, Δy⁺ = 30 (they afford 64×64 with spectral horizontal accuracy).
@info @sprintf("Δx = %.4f m = %.2f δ_s, Δy = %.4f m = %.2f δ_s",
               Lx / Nx, Lx / (Nx * δ), Ly / Ny, Ly / (Ny * δ))

# Did the targeted refinement actually land on the pycnocline? This is the check
# that the sweep's T-dependence is physical rather than a grid artefact.
if profile in (1, 4)
    kT = clamp(searchsortedlast(zf, T), 1, Nz)
    Δz_T = zf[kT+1] - zf[kT]
    @info @sprintf("Pycnocline z = T = %.1f m: Δz = %.4f m, transition width = %.3f m → %.1f cells across (band ±%.1f m)",
                   T, Δz_T, w_pycno, w_pycno / Δz_T, hw_pycno)
    w_pycno / Δz_T < 4 &&
        @warn "Pycnocline resolved by fewer than 4 cells — it will behave as a numerical step."
end

# ---------------- Boundary conditions ----------------
# No-slip bottom for u and v; free-slip top (default).
u_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0))
v_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0))

# Adiabatic bottom (paper); fixed gradient at top so the background
# stratification is maintained there. The imposed gradient is N²_bg evaluated at
# the lid — Lz_total, the top of the added sponge layer, not Lz. It must be the
# exact derivative of b_background or the lid fights the interior.
b_bcs = FieldBoundaryConditions(top    = GradientBoundaryCondition(N²_background(Lz_total)),
                                bottom = FluxBoundaryCondition(0))

# ---------------- Forcing ----------------
# Body force du/dt = U₀ ω cos(ωt) drives a free-stream velocity U₀ sin(ωt).
@inline tidal_forcing(x, y, z, t, p) = p.U₀ * p.ω * cos(p.ω * t)
u_tide = Forcing(tidal_forcing, parameters = (; U₀, ω))

# Sponge layer sitting ON TOP of the physical domain, spanning z = Lz – Lz_total:
# damps internal waves radiated by the boundary layer so they don't reflect off
# the rigid lid. The paper's peak damping coefficient is 20ω.
#
# The mask is a raised cosine confined to the added layer. The clamp makes it
# exactly zero for z ≤ Lz, so the physical domain is genuinely undamped; it
# reaches 1 at the lid and has zero slope at both ends, so waves entering the
# sponge meet no abrupt change in damping (which would itself reflect).
sponge_rate = 5ω
@inline top_mask(x, y, z) = sinpi(clamp((z - Lz) / L_sponge, 0, 1) / 2)^2

u_sponge = Relaxation(rate = sponge_rate, mask = top_mask,
                      target = (x, y, z, t) -> U₀ * sin(ω * t))
v_sponge = Relaxation(rate = sponge_rate, mask = top_mask)          # target 0
w_sponge = Relaxation(rate = sponge_rate, mask = top_mask)          # target 0
# b relaxes towards the same background the interior is initialized with, so the
# sponge does not fight the profile — relaxing to a different one (e.g. a linear
# ramp) forces the offset between them onto the solution at the lid.
b_sponge = Relaxation(rate = sponge_rate, mask = top_mask,
                      target = (x, y, z, t) -> b_background(z))

# ---------------- Model ----------------
# AMD provides the explicit SGS stresses/fluxes; the ScalarDiffusivity carries
# the molecular ν and κ = ν/Pr on top of it.
model = NonhydrostaticModel(grid;
            # Centered, not WENO: an upwind stencil adds numerical dissipation on
            # top of the AMD closure, so the SGS model is no longer the only sink,
            # and in the marginally turbulent phases that double-damping helps
            # kill the flow during acceleration. The paper uses a non-dissipative
            # pseudo-spectral/central scheme and lets its dynamic mixed model
            # carry all the dissipation.
            advection   = Centered(order = 2),
            timestepper = :RungeKutta3,
            tracers     = (:b),
            # Ri = 0 is the paper's "passive scalar case": the thermal field is
            # advected and diffused with the same background gradient but exerts
            # no buoyancy force. Dropping the buoyancy term (rather than setting
            # N² = 0 and leaving b ≡ 0) is what makes the Ri = 0 panels of
            # figures 4 and 5 show an actual scalar field.
            buoyancy    = passive_scalar ? nothing : BuoyancyTracer(),
            closure     = (AnisotropicMinimumDissipation(),
                           ScalarDiffusivity(ν = ν, κ = κ)),
            boundary_conditions = (u = u_bcs, v = v_bcs, b = b_bcs),
            coriolis    = nothing,
            forcing     = (u = (u_tide, u_sponge),
                           v = v_sponge,
                           w = w_sponge,
                           b = b_sponge))

# ---------------- Initial conditions ----------------
# The initial buoyancy IS the background selected by PROFILE, and the sponge and
# top BC relax toward that same profile (case_params.jl).
bᵢ(x, y, z) = b_background(z)

# The spin-up snapshot supplies velocities only. Ri = 0 carries b as a passive
# scalar, so its velocity field is independent of the thermal profile: a
# turbulent snapshot from a previous run is a valid restart state for every case
# here, Ri = 0 included, which is what lets the exponential-background run skip
# a fresh spin-up phase entirely.
#
# SPINUP_FILE pins the source so that a concurrently running Ri0 job cannot
# overwrite the file the stratified cases are reading from.
spinup_file = get(ENV, "SPINUP_FILE", spinup_default)

# Snapshots written before the sponge became an added layer stop at the old lid
# (Nz_phys levels for u/v, Nz_phys+1 for w) and so are shorter than this grid.
# Their physical cells are identical, so copy them in and repeat the topmost
# plane through the sponge cells; the relaxation flattens that onto its target
# in 1/sponge_rate ≈ 2000 s ≪ T_tide. Any other z-extent means a genuinely different
# physical grid, which is an error rather than something to pad.
function pad_into_sponge(a, nz_target, nz_old)
    n = size(a, 3)
    n == nz_target && return a
    n == nz_old || error("Spin-up snapshot has $n z-levels; expected $nz_target " *
                         "(this grid) or $nz_old (a pre-sponge-layer grid)")
    padded = similar(a, size(a, 1), size(a, 2), nz_target)
    padded[:, :, 1:n] .= a
    for k in n+1:nz_target
        padded[:, :, k] .= @view a[:, :, n]
    end
    return padded
end

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
    set!(model, u = pad_into_sponge(Array(interior(uts[nlast])), Nz, Nz_phys),
                v = pad_into_sponge(Array(interior(vts[nlast])), Nz, Nz_phys),
                w = pad_into_sponge(Array(interior(wts[nlast])), Nz + 1, Nz_phys + 1))
    set!(model, b = bᵢ)
else
    @warn "No spin-up snapshot at $spinup_file — starting $case from rest with noise."
    # Near-wall random kick; perturbing v breaks spanwise symmetry so genuine
    # 3D turbulence develops.
    kick = 0.01 * U₀
    damped_noise(z) = kick * randn() 
    uᵢ(x, y, z) = damped_noise(z)
    vᵢ(x, y, z) = damped_noise(z)
    wᵢ(x, y, z) = damped_noise(z)
    set!(model, u = uᵢ, v = vᵢ, w = wᵢ, b = bᵢ)
end

# ---------------- Simulation ----------------
simulation = Simulation(model, Δt = 1.0, stop_time = duration)

# Quick correctness check without the full run: TIDAL_SMOKE=1 julia Tidal3D.jl
# SMOKE_ITERS raises the iteration budget — with MAX_DT pinned near the Δt the
# wizard settles on in the turbulent state, this reproduces production cost per
# iteration (same grid, same output cadence) and is how the runs were timed.
if get(ENV, "TIDAL_SMOKE", "0") == "1"
    simulation.stop_iteration = parse(Int, get(ENV, "SMOKE_ITERS", "20"))
    @info "Smoke test: stopping after $(simulation.stop_iteration) iterations"
end

wizard = TimeStepWizard(cfl = 0.9, max_change = 1.2, max_Δt = max_Δt)
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
# What the sweep's deliverables actually read:
#   Figure 4 / Figure 5   ← *_profiles.jld2  (U, V, B)
#   Tidal3Danimation.jl   ← *.jld2 xz slices (u, b — not w)
#   restart of a later run← *_fields.jld2    (the spin-up's reason to exist)
# LIGHT_OUTPUT=1 writes only those, dropping the vorticity and x-y slice files
# (which figures 4/5 and the animation never open) and the unused w slice.
# FIELDS3D forces the 3D snapshot on or off independently; it defaults on for a
# normal run and off under LIGHT_OUTPUT, so the spin-up must set FIELDS3D=1 if
# it is run with LIGHT_OUTPUT.
const light_output  = get(ENV, "LIGHT_OUTPUT", "0") == "1"
const write_fields3d = get(ENV, "FIELDS3D", light_output ? "0" : "1") == "1"

u, v, w = model.velocities
b = model.tracers.b

slice_schedule = TimeInterval(duration / n_frames)

# (1) x-z slices at y = 0 for the animation, which plots u and b' only. w is
# carried in the full-output mode for ad-hoc cross-section work.
xz_fields = light_output ? (; u, b) : (; u, w, b)
simulation.output_writers[:xz_slices] =
    JLD2Writer(model, xz_fields,
               filename = filename * ".jld2",
               indices = (:, 1, :),
               schedule = slice_schedule,
               overwrite_existing = true,
               with_halos = false)

# (1b) Spanwise-averaged spanwise vorticity ⟨ω_y⟩_y(x,z), ω_y = ∂u/∂z − ∂w/∂x.
# Both ∂z(u) and ∂x(w) live at (Face, Center, Face), so their difference needs no
# interpolation; averaging over y (dims = 2) gives the x–z cross-section field.
# Read only by Vorticity.jl — not by figures 4/5 or the animation. This is the
# priciest diagnostic: the operand is evaluated over the full 3D field and
# reduced, 800 times per run.
if !light_output
    ω_y = ∂z(u) - ∂x(w)
    Ω_y = Field(Average(ω_y, dims = 2))
    simulation.output_writers[:vort_xz] =
        JLD2Writer(model, (; omega_y = Ω_y),
                   filename = filename * "_vortxz.jld2",
                   schedule = slice_schedule,
                   overwrite_existing = true,
                   with_halos = false)

    # (2) x-y slice at z ≈ δ: plan view of the near-wall streaks/bursts.
    # No script in the tree currently reads this file.
    zc_nodes = Array(znodes(grid, Center()))
    k_δ = searchsortedfirst(zc_nodes, δ)
    @info @sprintf("x-y slice at k = %d (z = %.3f m ≈ δ)", k_δ, zc_nodes[k_δ])

    simulation.output_writers[:xy_slices] =
        JLD2Writer(model, (; u, v, w, b),
                   filename = filename * "_xy.jld2",
                   indices = (:, :, k_δ),
                   schedule = slice_schedule,
                   overwrite_existing = true,
                   with_halos = false)
end

# (3) Horizontally averaged profiles. The second moments below are disabled —
# re-enable them to get turbulence statistics; they are saved raw (⟨u²⟩, ⟨uw⟩…),
# so subtract the mean in post-processing, e.g. u'w' = ⟨uw⟩ − U W.
U  = Field(Average(u, dims = (1, 2)))
V  = Field(Average(v, dims = (1, 2)))
B  = Field(Average(b, dims = (1, 2)))
#uw = Field(Average(u * w, dims = (1, 2)))
#vw = Field(Average(v * w, dims = (1, 2)))
#wb = Field(Average(w * b, dims = (1, 2)))
#uu = Field(Average(u^2,  dims = (1, 2)))
#vv = Field(Average(v^2,  dims = (1, 2)))
#ww = Field(Average(w^2,  dims = (1, 2)))

simulation.output_writers[:profiles] =
    JLD2Writer(model, (; U, V, B),
               filename = filename * "_profiles.jld2",
               schedule = TimeInterval(T_tide / 200),
               overwrite_existing = true,
               with_halos = false)

# (4) Full 3D snapshots twice per tidal period. This is what the spin-up exists
# to produce — every stratified run restarts from it — and the only 3D data kept,
# so it is also the only route to any re-analysis this run did not anticipate.
# The production sweep does not read it, hence FIELDS3D=0 to skip it (~0.27 GB
# per run), at the cost of making that run unrepeatable as a restart source.
if write_fields3d
    simulation.output_writers[:fields3d] =
        JLD2Writer(model, (; u, v, w, b),
                   filename = filename * "_fields.jld2",
                   schedule = TimeInterval(T_tide / 2),
                   overwrite_existing = true,
                   with_halos = false)
end

# (5) Checkpointer: keeps only the most recent checkpoint, so this costs one
# snapshot of disk, not one per period. Kept even under LIGHT_OUTPUT — a run is
# ~1.6 h and this is what makes an interrupted one resumable.
simulation.output_writers[:checkpoint] =
    Checkpointer(model,
                 schedule = TimeInterval(T_tide / 2),
                 dir = outdir,
                 prefix = "TidalBL3D_" * casetag * "_checkpoint",
                 cleanup = true)

run!(simulation)
# To resume an interrupted run, use:  run!(simulation, pickup = true)
# (and set overwrite_existing = false in the writers first)
