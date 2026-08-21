# 3D tidal (Stokes) boundary layer, after Gayen, Sarkar & Taylor (2010).
#
# One case:    PROFILE=4 T_STRAT=10 julia -t auto --project=. Tidal3D.jl sqrtRi2
# Its spin-up: RUN_TAG=spinup       julia -t auto --project=. Tidal3D.jl Ri0
#
# Following the paper, stratified cases restart from a turbulent Ri = 0 spin-up:
# spin up unstratified, then switch the stratification on. Output goes to
# outputs/<casetag>/.
#
# Differences from the paper:
#   - Re_s ≈ 5657 rather than 1788 (U₀ = 4 cm/s, ω = 1e-4, ν = 1e-6).
#   - AMD subgrid closure; they use a dynamic mixed model.
#   - Pr = 10 rather than 0.7.
#   - The sponge sits on top of the domain rather than inside it, so the
#     physical domain is undamped.
#
# PROFILE (case_params.jl) selects the background: 0 linear, 1 nonlinear,
# 2 exponential, 3 linear + exponential decay, 4 softplus (default; T is the
# pycnocline height and sharp its width).

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Printf
using Oceananigans
using CUDA

include(joinpath(@__DIR__, "case_params.jl"))

arch = GPU()          # start Julia with `julia -t auto`

# The grid is the same (100, 100, 300) for every case, Nz_total including the
# sponge cells. The stretching below only decides where the 300 vertical cells
# go, not how many there are.
Nx = 100
Ny = 100
Nz_total = 300

n_frames = 200 * n_periods          # animation frames (same cadence per period)
duration = n_periods * T_tide
max_Δt   = parse(Float64, get(ENV, "MAX_DT", "100.0"))

# ---------------- Grid (bottom-refined stretching) ----------------
# The faces come from integrating a prescribed spacing Δz(z), since a nearly
# uniform column leaves the wall layer unresolved and the flow relaminarizes
# each half cycle.
#
# The control points are (z/δ_s, Δz/δ_s) and set the shape only: build_z_faces
# rescales the whole profile so that it lands on exactly Nz_total cells. The
# actual spacing is finer than the table — see the "Bottom Δz" line the model
# logs at startup.
#
# The grid depends on nothing else, so every case in a sweep is discretised
# identically and one spin-up is a valid restart for any of them.
const dz_control_δ = [(0.0, 0.0861), (1.25, 0.1291), (5.0, 0.2840),
                      (15.0, 0.4130), (40.0, 1.2047), (70.0, 3.4420)]

const dz_control = [(zδ * δ, dδ * δ) for (zδ, dδ) in dz_control_δ]

# Linear between the control points, constant above the last one.
function dz_target(z)
    for i in 1:length(dz_control)-1
        (z0, d0) = dz_control[i]
        (z1, d1) = dz_control[i+1]
        z <= z1 && return d0 + (d1 - d0) * (z - z0) / (z1 - z0)
    end
    return dz_control[end][2]
end

# Step through Δz(z) Nz_phys times, bisecting on a multiplier m so the last face
# lands on Lz, then rescale to remove the remaining sub-cell error.
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

# The sponge cells take the spacing of the last physical cell, so there is no
# jump in resolution at z = Lz to reflect the waves the sponge should absorb.
append_sponge_faces(zf, L_sponge, n) = vcat(zf, zf[end] .+ (L_sponge / n) .* (1:n))

# The split between physical and sponge cells is circular — the sponge count
# depends on the top physical spacing, which depends on the physical count — so
# iterate to a fixed point. The total is fixed either way.
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
# At 100×100 the horizontal spacing is comparable to the paper's streamwise
# resolution and about 2.5 times coarser in the spanwise direction.
@info @sprintf("Δx = %.4f m = %.2f δ_s, Δy = %.4f m = %.2f δ_s",
               Lx / Nx, Lx / (Nx * δ), Ly / Ny, Ly / (Ny * δ))

# Profiles 1 and 4 put a thin pycnocline at z = T, above where the grid has
# coarsened, so record how well it is resolved. Lowering SHARP widens the
# transition (2ln9/sharp metres) if it is too thin.
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
# Quadratic bulk drag at the bottom and a free-slip top, as in the Ekman 3D case.
# This replaces the no-slip bottom used by the older runs in outputs/, so a drag
# run needs its own spin-up and is not directly comparable with them.

# Drag coefficient from the log law at the reference height, not at this grid's
# first cell centre (see case_params.jl). κ_vk, z₀, z_drag_ref and cᴰ_ref come
# from there; `κ` there is the molecular diffusivity.
z₁ = abs(Array(znodes(grid, Center()))[1])      # first cell centre above the wall
cᴰ = parse(Float64, get(ENV, "CD", string(cᴰ_ref)))

@info @sprintf("Bottom drag: cᴰ = %.4g from the log law at z_ref = %.4f m = %.1f z₀ (Ekman 3D: cᴰ = 0.0121 at z₁ = 0.0667 m); this grid's first cell is z₁ = %.5f m = %.2f z₀%s",
               cᴰ, z_drag_ref, z_drag_ref / z₀, z₁, z₁ / z₀,
               haskey(ENV, "CD") ? " [cᴰ OVERRIDDEN by CD]" : "")

# The reference height must stay at or above the first cell centre, or the drag
# law reads a velocity from a height the grid does not resolve.
z₁ > z_drag_ref &&
    @warn @sprintf("z_drag_ref = %.4f m is BELOW the first cell centre z₁ = %.4f m. The grid has coarsened past the reference height, so evaluate the law at z₁ instead: Z_DRAG_REF=%.4f.",
                   z_drag_ref, z₁, z₁)

# BulkDrag takes its direction from the field it is applied to, so the same
# constructor serves u and v.
drag_bc_u = BulkDrag(coefficient = cᴰ)
drag_bc_v = BulkDrag(coefficient = cᴰ)

# No-slip bottom, as used by the older runs in outputs/:
# drag_bc_u = ValueBoundaryCondition(0)
# drag_bc_v = ValueBoundaryCondition(0)

u_bcs = FieldBoundaryConditions(bottom = drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom = drag_bc_v)

# Insulating bottom, as in the paper, and a fixed gradient at the top to hold
# the background stratification there. The gradient is taken at the lid,
# Lz_total rather than Lz.
b_bcs = FieldBoundaryConditions(top    = GradientBoundaryCondition(N²_background(Lz_total)),
                                bottom = FluxBoundaryCondition(0))

# ---------------- Forcing ----------------
# Body force du/dt = U₀ ω cos(ωt) drives a free-stream velocity U₀ sin(ωt).
@inline tidal_forcing(x, y, z, t, p) = p.U₀ * p.ω * cos(p.ω * t)
u_tide = Forcing(tidal_forcing, parameters = (; U₀, ω))

# Sponge over z = Lz to Lz_total, damping the internal waves radiated by the
# boundary layer so they do not reflect off the rigid lid. The mask is a raised
# cosine, zero below z = Lz and 1 at the lid, so the damping turns on smoothly.
sponge_rate = 5ω
@inline top_mask(x, y, z) = sinpi(clamp((z - Lz) / L_sponge, 0, 1) / 2)^2

u_sponge = Relaxation(rate = sponge_rate, mask = top_mask,
                      target = (x, y, z, t) -> U₀ * sin(ω * t))
v_sponge = Relaxation(rate = sponge_rate, mask = top_mask)          # target 0
w_sponge = Relaxation(rate = sponge_rate, mask = top_mask)          # target 0
# b relaxes to the same background the interior starts from; any other target
# would force the difference between the two onto the lid.
b_sponge = Relaxation(rate = sponge_rate, mask = top_mask,
                      target = (x, y, z, t) -> b_background(z))

# ---------------- Model ----------------
model = NonhydrostaticModel(grid;
            # Centered rather than WENO: an upwind scheme adds numerical
            # dissipation on top of the closure, which kills the flow during
            # the marginally turbulent phases of the cycle.
            advection   = Centered(order = 2),
            timestepper = :RungeKutta3,
            tracers     = :b,
            # At Ri = 0, b is advected and diffused as a passive scalar and
            # exerts no buoyancy force.
            buoyancy    = passive_scalar ? nothing : BuoyancyTracer(),
            # AMD supplies the subgrid stresses and fluxes; ScalarDiffusivity
            # adds the molecular ν and κ.
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

# The spin-up supplies velocities only. Its buoyancy is passive, so its velocity
# field does not depend on the thermal profile and one snapshot restarts every
# case. SPINUP_FILE names the source explicitly.
spinup_file = get(ENV, "SPINUP_FILE", spinup_default)

# The snapshot must come from this grid; a size mismatch is an error rather
# than something to pad out.
function check_snapshot(a, nz_expected)
    size(a)[1:2] == (Nx, Ny) && size(a, 3) == nz_expected ||
        error("Spin-up snapshot is $(join(size(a), "×")); this grid needs " *
              "$(Nx)×$(Ny)×$nz_expected. Re-run the spin-up on the current grid.")
    return a
end

if isfile(spinup_file)
    # Snapshots are written every half period, so always at U∞ = 0 — the same
    # phase this run starts from.
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
    # Perturbing v as well breaks the spanwise symmetry, so the flow becomes
    # genuinely 3D rather than a 2D roll.
    kick = 0.01 * U₀
    noise(x, y, z) = kick * randn()
    set!(model, u = noise, v = noise, w = noise, b = bᵢ)
end

# ---------------- Simulation ----------------
simulation = Simulation(model, Δt = 1.0, stop_time = duration)

# Quick check without the full run: TIDAL_SMOKE=1 julia Tidal3D.jl. With MAX_DT
# set near the time step the wizard settles on, it also gives the cost per
# iteration of a production run.
if get(ENV, "TIDAL_SMOKE", "0") == "1"
    simulation.stop_iteration = parse(Int, get(ENV, "SMOKE_ITERS", "20"))
    @info "Smoke test: stopping after $(simulation.stop_iteration) iterations"
end

wizard = TimeStepWizard(cfl = 0.9, max_change = 1.2, max_Δt = max_Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

start_time = time_ns()
# flush is needed because stdout is buffered when redirected to a file, and the
# log would otherwise stay empty for tens of minutes at a time.
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
# Which script reads which file:
#   Figure4_metres.jl, Figure5.jl ← *_profiles.jld2  (U, V, B)
#   VorticityProfileAnim.jl       ← *_profiles.jld2  (dU/dz, dV/dz)
#   Tidal3Danimation.jl           ← *.jld2 x-z slices (u, b)
#   a restart of a later run      ← *_fields.jld2    (what the spin-up is for)
# LIGHT_OUTPUT=1 drops w from the x-z slices. FIELDS3D switches the 3D snapshot
# on or off; it is off under LIGHT_OUTPUT, so a spin-up run must set FIELDS3D=1.
const light_output   = get(ENV, "LIGHT_OUTPUT", "0") == "1"
const write_fields3d = get(ENV, "FIELDS3D", light_output ? "0" : "1") == "1"

u, v, w = model.velocities
b = model.tracers.b

slice_schedule = TimeInterval(duration / n_frames)

# (1) x-z slices at y = 0. The animation uses u and b′ only; w is kept in the
# full output for cross-section work.
xz_fields = light_output ? (; u, b) : (; u, w, b)
simulation.output_writers[:xz_slices] =
    JLD2Writer(model, xz_fields,
               filename = filename * ".jld2",
               indices = (:, 1, :),
               schedule = slice_schedule,
               overwrite_existing = true,
               with_halos = false)

# (2) Horizontally averaged profiles, which figures 4 and 5 read. Second moments
# are written separately by Moments.jl below.
U = Field(Average(u, dims = (1, 2)))
V = Field(Average(v, dims = (1, 2)))
B = Field(Average(b, dims = (1, 2)))

simulation.output_writers[:profiles] =
    JLD2Writer(model, (; U, V, B),
               filename = filename * "_profiles.jld2",
               schedule = TimeInterval(T_tide / 200),
               overwrite_existing = true,
               with_halos = false)

# (3) Full 3D snapshots twice per period. These are what the spin-up produces
# and the only 3D data kept, so the only route to analysis this run did not
# plan for. About 1.5 GB per run, so FIELDS3D=0 skips them for sweep cases,
# which then cannot be restarted from.
if write_fields3d
    simulation.output_writers[:fields3d] =
        JLD2Writer(model, (; u, v, w, b),
                   filename = filename * "_fields.jld2",
                   schedule = TimeInterval(T_tide / 2),
                   overwrite_existing = true,
                   with_halos = false)
end

# (4) Checkpointer, keeping only the most recent snapshot. This is what makes a
# multi-hour run resumable if the job is interrupted.
simulation.output_writers[:checkpoint] =
    Checkpointer(model,
                 schedule = TimeInterval(T_tide / 2),
                 dir = outdir,
                 prefix = "TidalBL3D_" * casetag * "_checkpoint",
                 cleanup = true)

# (5) Second moments — TKE, ⟨w′b′⟩, ⟨u′w′⟩ and the subgrid buoyancy flux —
# written to *_moments.jld2 for MixedLayerDiffusivity.jl. This adds one writer
# and changes nothing above it. MOMENTS=0 turns it off, as for the spin-up.
if get(ENV, "MOMENTS", "1") == "1"
    include(joinpath(@__DIR__, "Moments.jl"))
else
    @info "MOMENTS=0 — no second-moment output; K_T cannot be measured from this run"
end

run!(simulation)
# To resume a run: set overwrite_existing = false in the writers, then call
# run!(simulation, pickup = true).
