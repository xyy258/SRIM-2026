# 3D Tidal (Stokes) boundary layer after Gayen, Sarkar & Taylor (2010).
#
# One case:    PROFILE=4 T_STRAT=10 julia -t auto --project=. Tidal3D.jl sqrtRi2
# Its spin-up: RUN_TAG=spinup       julia -t auto --project=. Tidal3D.jl Ri0
#
# Stratified cases restart from a turbulent Ri = 0 spin-up, which is the paper's
# protocol: spin up unstratified, then switch stratification on. They find it via
# spinup_default (case_params.jl). Output goes to outputs/<casetag>/.
#
# Deliberate departures from the paper:
#   - Re_s ≈ 5657, not their 1788 (U₀ = 4 cm/s, ω = 1e-4, ν = 1e-6, δ_s = 0.141 m).
#   - AMD SGS closure; they use a dynamic mixed model and report that plain
#     Smagorinsky fails for this flow.
#   - Pr = 10, not 0.7, so b is effectively SGS-controlled on this grid.
#   - The sponge is stacked ON TOP of the physical domain (z = Lz to Lz_total)
#     rather than carved out of it, leaving the physical domain fully undamped.
#   - Ri = 0 carries b as a genuine passive scalar, as they do, so its figure 4/5
#     panels hold real data.
#
# PROFILE (case_params.jl) selects the background: 0 linear, 1 nonlinear,
# 2 exponential, 3 linear+exponential decay, 4 softplus (default; T is the
# pycnocline height, sharp its width).

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Printf
using Oceananigans
using CUDA

include(joinpath(@__DIR__, "case_params.jl"))

arch = GPU()          # start Julia with `julia -t auto`

# FIXED grid: the array handed to Oceananigans is (100, 100, 300) for every case,
# Nz_total counting the sponge cells. Nothing below derives a cell count — the
# stretching sets only where the 300 vertical cells go.
Nx = 100
Ny = 100
Nz_total = 300

n_frames = 200 * n_periods          # animation frames (same cadence per period)
duration = n_periods * T_tide
max_Δt   = parse(Float64, get(ENV, "MAX_DT", "100.0"))

# ---------------- Grid (bottom-refined stretching) ----------------
# Faces come from integrating a prescribed Δz(z): a near-uniform column leaves the
# wall layer unresolved and the flow relaminarizes each half-cycle.
#
# Control points are (z/δ_s, Δz/δ_s) and set the SHAPE only, since build_z_faces
# rescales the whole profile by one multiplier m to land exactly Nz_total cells.
# The shape wants ~192 cells unscaled, so m ≈ 0.71 and everything runs finer than
# the table below: the wall gets 0.061 δ_s, Δz⁺ ≈ 6.4. Read the "Bottom Δz" line
# the model logs at startup rather than this table.
#
# The grid depends on nothing but these numbers — not T, not PROFILE, not any run
# knob — so every case in a sweep and every spin-up is discretised identically,
# and one spin-up is a valid restart for any of them.
const dz_control_δ = [(0.0, 0.0861), (1.25, 0.1291), (5.0, 0.2840),
                      (15.0, 0.4130), (40.0, 1.2047), (70.0, 3.4420)]

const dz_control = [(zδ * δ, dδ * δ) for (zδ, dδ) in dz_control_δ]

# Linear between control points, constant above the last.
function dz_target(z)
    for i in 1:length(dz_control)-1
        (z0, d0) = dz_control[i]
        (z1, d1) = dz_control[i+1]
        z <= z1 && return d0 + (d1 - d0) * (z - z0) / (z1 - z0)
    end
    return dz_control[end][2]
end

# March Δz(z) for exactly Nz_phys steps, bisecting a uniform multiplier m so the
# last face lands on Lz, then rescale by the sub-cell residual to put it there.
function build_z_faces(Lz, Nz_phys)
    march(m) = (g = [0.0]; for _ in 1:Nz_phys
                    push!(g, g[end] + m * dz_target(g[end]))
                end; g)
    lo, hi = 0.05, 10.0
    for _ in 1:100
        m = (lo + hi) / 2
        march(m)[end] < Lz ? (lo = m) : (hi = m)
    end
    g = march((lo + hi) / 2)
    return g .* (Lz / g[end])
end

# Sponge cells have constant Δz inheriting the last physical Δz, so there is no
# resolution jump at z = Lz — a jump there would reflect the waves the sponge
# exists to absorb.
append_sponge_faces(zf, L_sponge, n) = vcat(zf, zf[end] .+ (L_sponge / n) .* (1:n))

# The physical/sponge split is circular: the sponge count follows from the top
# physical Δz, which follows from the physical count. Iterate to a fixed point;
# the total is pinned by construction whether or not it converges.
function split_column(Nz_total)
    n = clamp(round(Int, L_sponge / dz_control[end][2]), 1, Nz_total - 2)
    zf = build_z_faces(Lz, Nz_total - n)
    for _ in 1:10
        n_new = clamp(round(Int, L_sponge / (zf[end] - zf[end-1])), 1, Nz_total - 2)
        n_new == n && break
        n = n_new
        zf = build_z_faces(Lz, Nz_total - n)
    end
    return zf, n
end

zf_phys, Nz_spnge = split_column(Nz_total)
Nz_phys = length(zf_phys) - 1
zf      = append_sponge_faces(zf_phys, L_sponge, Nz_spnge)
Nz      = length(zf) - 1
@assert Nz == Nz_total "vertical grid came out at $Nz cells, not the fixed $Nz_total"

grid = RectilinearGrid(arch;
                       topology = (Periodic, Periodic, Bounded),
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = zf)

Δz_bottom = minimum(diff(zf))
n_test = count(z -> z <= Lz_test, zf) - 1
@info @sprintf("Bottom Δz = %.4f m = %.3f δ_s (%.1f points across δ_s); %d cells below z = %.0f δ_s; %d physical + %d sponge cells (sponge Δz = %.3f δ_s)",
               Δz_bottom, Δz_bottom / δ, δ / Δz_bottom, n_test, Lz_test / δ,
               Nz_phys, Nz_spnge, (zf[end] - zf[end-1]) / δ)
# At 100×100 this is Δx = Δy = 0.71 δ_s, i.e. Δx⁺ ≈ 74 against the paper's 60 —
# comparable streamwise, ~2.5× coarse spanwise where they use Δy⁺ = 30.
@info @sprintf("Δx = %.4f m = %.2f δ_s, Δy = %.4f m = %.2f δ_s",
               Lx / Nx, Lx / (Nx * δ), Ly / Ny, Ly / (Ny * δ))

# Profiles 1 and 4 put a thin feature at z = T, well above where the grid has
# coarsened, so record how well it is resolved. EXPECT THE WARNING for T ≳ 10 at
# the default sharp = 6: the pycnocline spans ~2 cells there against ~7 at T = 5,
# so early erosion is partly grid-controlled and not equally so across a T-sweep.
# Lowering SHARP widens the transition (2ln9/sharp metres) and fixes it.
if profile in (1, 4)
    w_pycno = 2log(9) / sharp        # 10–90 % width of the db/dz transition
    kT = clamp(searchsortedlast(zf, T), 1, Nz)
    Δz_T = zf[kT+1] - zf[kT]
    @info @sprintf("Pycnocline z = T = %.1f m: Δz = %.4f m, transition width = %.3f m → %.1f cells across",
                   T, Δz_T, w_pycno, w_pycno / Δz_T)
    w_pycno / Δz_T < 4 &&
        @warn "Pycnocline resolved by fewer than 4 cells — it will behave as a numerical step."
end

# ---------------- Boundary conditions ----------------
# Quadratic bulk drag at the bottom, free-slip top (default), matching the Ekman
# 3D case. NOTE this replaces the no-slip bottom every run in outputs/ was made
# with, so a drag run is not directly comparable with them and needs its own
# spin-up (the no-slip snapshot is still a legal restart, but its wall layer is
# not the one drag sustains).

# Drag coefficient from the log law, evaluated at the REFERENCE HEIGHT rather
# than at this grid's first cell centre — see the long note in case_params.jl.
# κ_vk, z₀, z_drag_ref and cᴰ_ref come from there; NOT `κ`, which there is the
# molecular diffusivity ν/Pr.
z₁ = abs(Array(znodes(grid, Center()))[1])      # first cell centre above the wall
cᴰ = parse(Float64, get(ENV, "CD", string(cᴰ_ref)))

@info @sprintf("Bottom drag: cᴰ = %.4g from the log law at z_ref = %.4f m = %.1f z₀ (Ekman 3D: cᴰ = 0.0121 at z₁ = 0.0667 m); this grid's first cell is z₁ = %.5f m = %.2f z₀%s",
               cᴰ, z_drag_ref, z_drag_ref / z₀, z₁, z₁ / z₀,
               haskey(ENV, "CD") ? " [cᴰ OVERRIDDEN by CD]" : "")

# The reference height must stay AT OR ABOVE the first cell centre: below it the
# drag law would be reading a velocity from a height the grid never resolves.
z₁ > z_drag_ref &&
    @warn @sprintf("z_drag_ref = %.4f m is BELOW the first cell centre z₁ = %.4f m. The grid has coarsened past the reference height, so evaluate the law at z₁ instead: Z_DRAG_REF=%.4f.",
                   z_drag_ref, z₁, z₁)

# Quadratic drag. BulkDrag infers its direction from the field location, so the
# same constructor serves u and v.
drag_bc_u = BulkDrag(coefficient = cᴰ)
drag_bc_v = BulkDrag(coefficient = cᴰ)

# No-slip bottom, for reference — what every run currently in outputs/ used:
# drag_bc_u = ValueBoundaryCondition(0)
# drag_bc_v = ValueBoundaryCondition(0)

u_bcs = FieldBoundaryConditions(bottom = drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom = drag_bc_v)

# Adiabatic bottom (paper); fixed gradient at top to maintain the background
# stratification there. The gradient is N²_bg at the lid — Lz_total, not Lz — and
# must be the exact derivative of b_background or the lid fights the interior.
b_bcs = FieldBoundaryConditions(top    = GradientBoundaryCondition(N²_background(Lz_total)),
                                bottom = FluxBoundaryCondition(0))

# ---------------- Forcing ----------------
# Body force du/dt = U₀ ω cos(ωt) drives a free-stream velocity U₀ sin(ωt).
@inline tidal_forcing(x, y, z, t, p) = p.U₀ * p.ω * cos(p.ω * t)
u_tide = Forcing(tidal_forcing, parameters = (; U₀, ω))

# Sponge over z = Lz – Lz_total, damping internal waves radiated by the boundary
# layer so they do not reflect off the rigid lid. The mask is a raised cosine
# clamped to exactly zero for z ≤ Lz, reaching 1 at the lid with zero slope at
# both ends, so waves entering it meet no abrupt change in damping.
sponge_rate = 5ω
@inline top_mask(x, y, z) = sinpi(clamp((z - Lz) / L_sponge, 0, 1) / 2)^2

u_sponge = Relaxation(rate = sponge_rate, mask = top_mask,
                      target = (x, y, z, t) -> U₀ * sin(ω * t))
v_sponge = Relaxation(rate = sponge_rate, mask = top_mask)          # target 0
w_sponge = Relaxation(rate = sponge_rate, mask = top_mask)          # target 0
# b relaxes to the same background the interior is initialized with; relaxing to a
# different profile would force the offset between them onto the lid.
b_sponge = Relaxation(rate = sponge_rate, mask = top_mask,
                      target = (x, y, z, t) -> b_background(z))

# ---------------- Model ----------------
model = NonhydrostaticModel(grid;
            # Centered, not WENO: an upwind stencil adds numerical dissipation on
            # top of the AMD closure, and in the marginally turbulent phases that
            # double-damping kills the flow during acceleration.
            advection   = Centered(order = 2),
            timestepper = :RungeKutta3,
            tracers     = :b,
            # Ri = 0 is the paper's passive-scalar case: b is advected and diffused
            # with the same background gradient but exerts no buoyancy force.
            buoyancy    = passive_scalar ? nothing : BuoyancyTracer(),
            # AMD gives the SGS stresses/fluxes; ScalarDiffusivity carries the
            # molecular ν and κ = ν/Pr on top.
            closure     = (AnisotropicMinimumDissipation(),
                           ScalarDiffusivity(ν = ν, κ = κ)),
            boundary_conditions = (u = u_bcs, v = v_bcs, b = b_bcs),
            coriolis    = nothing,
            forcing     = (u = (u_tide, u_sponge),
                           v = v_sponge,
                           w = w_sponge,
                           b = b_sponge))

# ---------------- Initial conditions ----------------
bᵢ(x, y, z) = b_background(z)

# The spin-up supplies velocities only. Ri = 0 carries b passively, so its
# velocity field is independent of the thermal profile and one snapshot is a valid
# restart for every case here. SPINUP_FILE pins the source so a concurrently
# running Ri0 job cannot overwrite what the stratified cases are reading.
spinup_file = get(ENV, "SPINUP_FILE", spinup_default)

# The snapshot must come off this grid. It used to be padded when shorter (from
# before the sponge became an added layer); with a fixed grid every snapshot has
# the full Nz, so a mismatch is an error rather than something to fix up.
function check_snapshot(a, nz_expected)
    size(a)[1:2] == (Nx, Ny) && size(a, 3) == nz_expected ||
        error("Spin-up snapshot is $(join(size(a), "×")); this grid needs " *
              "$(Nx)×$(Ny)×$nz_expected. Re-run the spin-up on the current grid.")
    return a
end

if isfile(spinup_file)
    # Snapshots are written every half period, i.e. always at U∞ = 0 — the phase
    # this run starts from — so the restart is phase-consistent.
    @info "Initializing velocities from Ri=0 spin-up: $spinup_file"
    uts = FieldTimeSeries(spinup_file, "u"; backend = OnDisk())
    vts = FieldTimeSeries(spinup_file, "v"; backend = OnDisk())
    wts = FieldTimeSeries(spinup_file, "w"; backend = OnDisk())
    nlast = length(uts.times)
    @info @sprintf("Using snapshot %d/%d (t = %.2f periods of the spin-up)",
                   nlast, nlast, uts.times[nlast] / T_tide)
    set!(model, u = check_snapshot(Array(interior(uts[nlast])), Nz),
                v = check_snapshot(Array(interior(vts[nlast])), Nz),
                w = check_snapshot(Array(interior(wts[nlast])), Nz + 1))
    set!(model, b = bᵢ)
else
    @warn "No spin-up snapshot at $spinup_file — starting $case from rest with noise."
    # Perturbing v as well breaks spanwise symmetry so genuine 3D turbulence
    # develops rather than a 2D roll.
    kick = 0.01 * U₀
    noise(x, y, z) = kick * randn()
    set!(model, u = noise, v = noise, w = noise, b = bᵢ)
end

# ---------------- Simulation ----------------
simulation = Simulation(model, Δt = 1.0, stop_time = duration)

# Correctness check without the full run: TIDAL_SMOKE=1 julia Tidal3D.jl. With
# MAX_DT pinned near the Δt the wizard settles on, this also reproduces production
# cost per iteration, which is how the runs were timed.
if get(ENV, "TIDAL_SMOKE", "0") == "1"
    simulation.stop_iteration = parse(Int, get(ENV, "SMOKE_ITERS", "20"))
    @info "Smoke test: stopping after $(simulation.stop_iteration) iterations"
end

wizard = TimeStepWizard(cfl = 0.9, max_change = 1.2, max_Δt = max_Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

start_time = time_ns()
# flush: stdout is block-buffered when redirected to a file, so without it an
# overnight log stays empty for tens of minutes and the driver cannot see progress.
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
# Who reads what:
#   Figure4_metres.jl, Figure5.jl ← *_profiles.jld2  (U, V, B)
#   VorticityProfileAnim.jl       ← *_profiles.jld2  (dU/dz, dV/dz)
#   Tidal3Danimation.jl           ← *.jld2 xz slices (u, b)
#   restart of a later run        ← *_fields.jld2    (the spin-up's whole purpose)
# LIGHT_OUTPUT=1 now only drops w from the xz slices; FIELDS3D forces the 3D
# snapshot on or off independently — it defaults on, and off under LIGHT_OUTPUT,
# so a spin-up run with LIGHT_OUTPUT must set FIELDS3D=1.
const light_output   = get(ENV, "LIGHT_OUTPUT", "0") == "1"
const write_fields3d = get(ENV, "FIELDS3D", light_output ? "0" : "1") == "1"

u, v, w = model.velocities
b = model.tracers.b

slice_schedule = TimeInterval(duration / n_frames)

# (1) x-z slices at y = 0. The animation plots u and b′ only; w is carried in full
# output for ad-hoc cross-section work.
xz_fields = light_output ? (; u, b) : (; u, w, b)
simulation.output_writers[:xz_slices] =
    JLD2Writer(model, xz_fields,
               filename = filename * ".jld2",
               indices = (:, 1, :),
               schedule = slice_schedule,
               overwrite_existing = true,
               with_halos = false)

# (2) Horizontally averaged profiles — what figures 4 and 5 read. Second moments
# are not written; add Average(u * w, dims = (1, 2)) and friends here if
# turbulence statistics are wanted, remembering they save raw (u'w' = ⟨uw⟩ − U W).
U = Field(Average(u, dims = (1, 2)))
V = Field(Average(v, dims = (1, 2)))
B = Field(Average(b, dims = (1, 2)))

simulation.output_writers[:profiles] =
    JLD2Writer(model, (; U, V, B),
               filename = filename * "_profiles.jld2",
               schedule = TimeInterval(T_tide / 200),
               overwrite_existing = true,
               with_halos = false)

# (3) Full 3D snapshots twice per period — what the spin-up exists to produce, and
# the only 3D data kept, hence the only route to re-analysis this run did not
# anticipate. ~1.5 GB per run on this grid, so FIELDS3D=0 skips it for sweep cases
# at the cost of making them unusable as a restart source.
if write_fields3d
    simulation.output_writers[:fields3d] =
        JLD2Writer(model, (; u, v, w, b),
                   filename = filename * "_fields.jld2",
                   schedule = TimeInterval(T_tide / 2),
                   overwrite_existing = true,
                   with_halos = false)
end

# (4) Checkpointer, keeping only the most recent — one snapshot of disk, not one
# per period. Kept even under LIGHT_OUTPUT: it is what makes a multi-hour run
# resumable if the job is interrupted.
simulation.output_writers[:checkpoint] =
    Checkpointer(model,
                 schedule = TimeInterval(T_tide / 2),
                 dir = outdir,
                 prefix = "TidalBL3D_" * casetag * "_checkpoint",
                 cleanup = true)

# (5) Second moments — TKE, ⟨w′b′⟩, ⟨u′w′⟩ and the SGS buoyancy flux, written to
# *_moments.jld2 for MixedLayerDiffusivity.jl. Registers one extra writer and
# changes no existing filename or schedule, so every reader above is unaffected.
# MOMENTS=0 turns it off (the spin-up has no use for it).
if get(ENV, "MOMENTS", "1") == "1"
    include(joinpath(@__DIR__, "Moments.jl"))
else
    @info "MOMENTS=0 — no second-moment output; K_T cannot be measured from this run"
end

run!(simulation)
# To resume: run!(simulation, pickup = true), with overwrite_existing = false in
# the writers first.
