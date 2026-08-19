# Second-moment output for Tidal3D.jl — the raw material for TKE and K_T.
#
# Included by Tidal3D.jl immediately before run!(simulation). It adds ONE output
# writer, :moments, writing plane-averaged profiles to <filename>_moments.jld2.
# It touches no existing writer, filename or schedule.
#
#   means:   U, V, W, B, dBdz
#   moments: uu, vv, ww (Centers);  uw, vw, wb (Faces)
#   SGS:     kappa_sgs, F_sgs (Faces)
#
# Post-processed by MixedLayerDiffusivity.jl.
#
# ---------------------------------------------------------------------------
# WHY RAW MOMENTS, AND WHY NOT AveragedTimeInterval
# ---------------------------------------------------------------------------
# The domain is (Periodic, Periodic, Bounded) with horizontally uniform forcing,
# so the PLANE AVERAGE IS THE REYNOLDS AVERAGE, exactly:
#
#     U(z,t) = ⟨u⟩_xy        u′ = u − U(z,t)
#
# The tidal (Stokes) flow is horizontally uniform and therefore lives entirely in
# U(z,t). Subtracting the plane average removes the tidal component exactly and
# instantaneously — no time filtering is involved in DEFINING the fluctuations.
# (The 30-minute window in the handwritten notes is for single-point mooring data,
# a different problem, and must not be used as the decomposition here.)
#
# ORDERING TRAP — the single most important thing in this file. Time-averaging is
# still wanted, for noise reduction, but it MUST come after the decomposition. If
# raw moments are time-averaged first,
#
#     time_avg(⟨uu⟩) − (time_avg(U))²  =  time_avg(TKE) + Var_t(U)
#
# and Var_t(U) — the variance of the TIDAL MEAN FLOW over the averaging window —
# leaks straight back in as "TKE". It scales as (U₀ ω Δt)²/12, which at U₀ = 0.04,
# ω = 1e-4, Δt = 1800 s is ~4e-6 m² s⁻², the same order as u*² itself. A 100 %
# error. Oceananigans' AveragedTimeInterval schedule would do exactly this, so it
# is NOT used: every field below is written instantaneously on TimeInterval, the
# means are subtracted per sample offline, and only THEN is the resulting TKE
# smoothed (MixedLayerDiffusivity.jl).
#
# Fluctuation fields are also not built inside AbstractOperations. u − ⟨u⟩ would
# require the mean to be computed before the operation is evaluated, and the
# output writer does not guarantee that ordering. Raw moments + offline
# subtraction is the safe pattern, and is what the comment in Tidal3D.jl's output
# section already recommends.
#
# ---------------------------------------------------------------------------
# WHAT NEEDS A MEAN SUBTRACTED, AND WHAT DOES NOT
# ---------------------------------------------------------------------------
# With a rigid lid, an impermeable bottom and incompressibility, ⟨w⟩_xy ≡ 0 at
# every z, exactly. Therefore
#
#     ⟨w′b′⟩ = ⟨wb⟩_xy      and      ⟨u′w′⟩ = ⟨uw⟩_xy
#
# with NO mean subtraction. Only the variances need it: ⟨u′u′⟩ = ⟨uu⟩ − U²,
# ⟨v′v′⟩ = ⟨vv⟩ − V², ⟨w′w′⟩ = ⟨ww⟩.
#
# W is written anyway as a correctness check — it should come out at ~1e-18, and
# if it does not, every flux above is wrong. This is verification step 1.
#
# ---------------------------------------------------------------------------
# LOCATIONS
# ---------------------------------------------------------------------------
# Variances at Centers, so TKE is one Center-located profile. Fluxes at Faces, so
# K_T = −F/(dB/dz) is a pointwise ratio of two Face quantities with no
# interpolation in the division. `@at` pins each location explicitly rather than
# leaving it to the default placement of binary operations.
#
# dBdz comes from the model's own ∂z(b) operator, not from an offline
# finite difference, so it uses the same stretched-grid operator the solver does.

using Oceananigans.AbstractOperations: ∂z, @at

# ---------------------------------------------------------------------------
# Locating the SGS diffusivity
# ---------------------------------------------------------------------------
# The closure is a TUPLE, (AnisotropicMinimumDissipation(), ScalarDiffusivity()),
# so the container of closure auxiliary fields is a tuple in the same order: AMD
# is entry 1 and carries νₑ plus a κₑ NamedTuple keyed by tracer name; the
# ScalarDiffusivity entry has no auxiliary fields and is `nothing`.
#
# THE SPELLING HAS MOVED BETWEEN OCEANANIGANS VERSIONS, so nothing here is
# hardcoded. Verified against the pinned Manifest.toml (Oceananigans v0.110.11):
# the model property is `closure_fields` (nonhydrostatic_model.jl:51) — the older
# `diffusivity_fields` no longer exists anywhere in that source tree — and the AMD
# entry is `(; νₑ, κₑ)` with `κₑ` a NamedTuple over tracer names
# (anisotropic_minimum_dissipation.jl:372-375). Both spellings are searched for,
# and the resolver logs what it found.
#
# CLIPPING: this version DOES clip κₑ at zero —
#   anisotropic_minimum_dissipation.jl:203  @inbounds κₑ[i,j,k] = max(zero(FT), κˢᵍˢ)
# — so AMD backscatter cannot put negative κₑ into isolated cells and produce
# nonsense K_T there. The runtime check registered at the bottom of this file
# re-tests that on the actual run rather than trusting the source read.

const _CLOSURE_FIELD_PROPERTIES = (:closure_fields, :diffusivity_fields)
const _KAPPA_NAMES = (:κₑ, :κ_e, :κₜ, :kappa_e, :kappaₑ, :κ)

_is_field(x) = x isa Oceananigans.Fields.AbstractField

"""
    find_sgs_diffusivity(model, tracer_name)

Return the AMD subgrid diffusivity field for `tracer_name`, searching the model's
closure-auxiliary-field container rather than assuming a spelling. Logs where it
was found; errors loudly, with instructions, if it cannot find it.
"""
function find_sgs_diffusivity(model, tracer_name::Symbol)
    prop = findfirst(p -> hasproperty(model, p), _CLOSURE_FIELD_PROPERTIES)
    prop === nothing && error("""
        Moments.jl: could not find the closure auxiliary fields on the model.
        Tried $(join(string.("model.", _CLOSURE_FIELD_PROPERTIES), ", ")).
        Available properties: $(join(string.(propertynames(model)), ", ")).
        Find the one holding the SGS diffusivity in this Oceananigans version and
        add its name to _CLOSURE_FIELD_PROPERTIES at the top of Moments.jl.""")

    propname = _CLOSURE_FIELD_PROPERTIES[prop]
    container = getproperty(model, propname)
    @info "Moments.jl: closure auxiliary fields found at `model.$propname` ($(typeof(container).name.name))"

    # A tuple closure gives a tuple of containers, one per closure, with `nothing`
    # for closures that need no auxiliary fields (ScalarDiffusivity). A single
    # closure gives its NamedTuple directly. NamedTuple is not a subtype of Tuple,
    # so this test separates the two cleanly.
    candidates = container isa Tuple ? collect(container) : Any[container]

    tracers = keys(model.tracers)
    tidx = findfirst(==(tracer_name), collect(tracers))
    tidx === nothing && error("Moments.jl: tracer :$tracer_name is not among $(tracers)")

    for (i, c) in enumerate(candidates)
        c === nothing && continue
        c isa NamedTuple || continue

        # Preferred path: a property with a known κ spelling.
        for name in propertynames(c)
            name in _KAPPA_NAMES || continue
            κ_entry = getproperty(c, name)
            κ_field = κ_entry isa NamedTuple ? get(κ_entry, tracer_name, nothing) :
                      κ_entry isa Tuple      ? (tidx <= length(κ_entry) ? κ_entry[tidx] : nothing) :
                      κ_entry
            if _is_field(κ_field)
                @info "Moments.jl: SGS diffusivity resolved as `model.$propname[$i].$name" *
                      (κ_entry isa Union{NamedTuple,Tuple} ? "[:$tracer_name]`" : "`") *
                      " — $(summary(κ_field))"
                return κ_field
            end
        end

        # Fallback: any property that is a NamedTuple keyed by tracer name whose
        # entry is a Field. Catches a rename this file has not been told about.
        for name in propertynames(c)
            κ_entry = getproperty(c, name)
            κ_entry isa NamedTuple || continue
            κ_field = get(κ_entry, tracer_name, nothing)
            if _is_field(κ_field)
                @warn "Moments.jl: SGS diffusivity found by FALLBACK search at " *
                      "`model.$propname[$i].$name[:$tracer_name]` — the spelling has changed. " *
                      "Add :$name to _KAPPA_NAMES in Moments.jl."
                return κ_field
            end
        end
    end

    error("""
        Moments.jl: found `model.$propname` but no SGS diffusivity for tracer :$tracer_name inside it.
        Container: $(typeof(container))
        Entries:   $(join(string.(typeof.(candidates)), ", "))
        Entry properties: $(join([c === nothing ? "nothing" : string(propertynames(c)) for c in candidates], " | "))
        Expected an AnisotropicMinimumDissipation entry carrying νₑ and a κₑ NamedTuple
        keyed by tracer name. If the spelling has moved, add it to _KAPPA_NAMES
        (or the container name to _CLOSURE_FIELD_PROPERTIES) at the top of Moments.jl.
        K_T CANNOT BE MEASURED WITHOUT THIS — with Pr = $Pr the buoyancy field is
        effectively SGS-controlled on this grid, so dropping the subgrid flux would
        underestimate K_T most badly inside the pycnocline, the exact region the
        whole calculation depends on.""")
end

κₑ_b = find_sgs_diffusivity(model, :b)

# ---------------------------------------------------------------------------
# The profiles
# ---------------------------------------------------------------------------
# `u`, `v`, `w`, `b` are already bound by Tidal3D.jl's output section, which runs
# before this file is included.
plane_avg(op) = Field(Average(op, dims = (1, 2)))

# --- means -----------------------------------------------------------------
# U, V, B duplicate the :profiles writer deliberately: the moments file must be
# self-contained, since subtracting a mean read from a DIFFERENT file at a
# DIFFERENT sample time is precisely the ordering error described above. (The two
# writers share a schedule, so the values agree.)
M_U    = plane_avg(u)                                   # (Nothing, Nothing, Center)
M_V    = plane_avg(v)                                   # (Nothing, Nothing, Center)
M_W    = plane_avg(w)                                   # (Nothing, Nothing, Face) — the ⟨w⟩ ≈ 1e-18 check
M_B    = plane_avg(b)                                   # (Nothing, Nothing, Center)
M_dBdz = plane_avg(∂z(b))                               # (Nothing, Nothing, Face)

# --- raw second moments ----------------------------------------------------
# Variances at Centers → one Center-located TKE profile.
M_uu = plane_avg(@at (Center, Center, Center) u * u)
M_vv = plane_avg(@at (Center, Center, Center) v * v)
M_ww = plane_avg(@at (Center, Center, Center) w * w)

# Vertical fluxes at Faces → K_T = −F/(dB/dz) is a ratio of two Face profiles.
M_uw = plane_avg(@at (Center, Center, Face) u * w)
M_vw = plane_avg(@at (Center, Center, Face) v * w)
M_wb = plane_avg(@at (Center, Center, Face) w * b)

# --- SGS -------------------------------------------------------------------
# In LES the buoyancy flux is split at the filter scale. K_T is the diffusivity a
# 1D model with no resolved turbulence would need, so it must carry BOTH halves:
#
#     F_b = ⟨w′b′⟩_res + F_sgs        (positive upward)
#     F_b = −K_T dB/dz
#
# F_sgs is written already signed, so post-processing forms F_b = wb + F_sgs with
# no sign to get wrong.
#
# THE AVERAGE OF THE PRODUCT, NOT THE PRODUCT OF THE AVERAGES. κₑ and ∂b/∂z are
# correlated — AMD keys off local strain, which peaks exactly where the gradient
# is sharp — so ⟨κₑ⟩⟨∂b/∂z⟩ underestimates the flux precisely where it matters.
M_F_sgs = plane_avg(@at (Center, Center, Face) -(κₑ_b + κ) * ∂z(b))

# ⟨κₑ + κ_mol⟩ on its own, at Faces so it sits on the same nodes as F_sgs and K_T.
# Two uses in post-processing:
#   (1) the SGS share  K_sgs/K_T  with  K_sgs = −F_sgs/(dB/dz). Where that ratio
#       approaches 1, K_T is a property of the AMD closure rather than a
#       measurement of the flow, and must be reported alongside every K_T value.
#   (2) K_sgs vs this profile measures the κₑ–∂b/∂z correlation directly: they
#       agree only where the two are uncorrelated.
# Note this INCLUDES the molecular κ = ν/Pr = $(κ); subtract that constant to
# recover ⟨κₑ⟩ alone.
M_kappa_sgs = plane_avg(@at (Center, Center, Face) κₑ_b + κ)

# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------
# Its own "_moments" suffix, so it cannot collide with .jld2 (xz slices),
# _profiles.jld2 or _fields.jld2 — the writers use overwrite_existing = true and a
# tag collision destroys data silently.
#
# TimeInterval, matching the :profiles writer's cadence. NOT AveragedTimeInterval:
# see the ordering trap at the top of this file.
simulation.output_writers[:moments] =
    JLD2Writer(model,
               (U = M_U, V = M_V, W = M_W, B = M_B, dBdz = M_dBdz,
                uu = M_uu, vv = M_vv, ww = M_ww,
                uw = M_uw, vw = M_vw, wb = M_wb,
                kappa_sgs = M_kappa_sgs, F_sgs = M_F_sgs),
               filename = filename * "_moments.jld2",
               schedule = TimeInterval(T_tide / 200),
               overwrite_existing = true,
               with_halos = false)

@info "Moments.jl: writing 13 plane-averaged profiles to $(filename)_moments.jld2 " *
      @sprintf("every %.0f s (T_tide/200)", T_tide / 200)

# ---------------------------------------------------------------------------
# Runtime checks
# ---------------------------------------------------------------------------
# (a) ⟨w⟩_xy ≈ 0 — verification step 1. If this is not ~1e-18 relative to U₀ then
#     the Reynolds decomposition assumed above is wrong and so is every flux.
# (b) min(κₑ) ≥ 0 — confirms on this run what the source read says about clipping,
#     since an unclipped negative κₑ would give nonsense K_T in isolated cells.
# Cheap: two reductions over profiles/3D fields at a coarse interval.
function moments_health_check(sim)
    compute!(M_W)
    w_max = maximum(abs, interior(M_W))
    κ_min, κ_max = extrema(interior(κₑ_b))
    @info @sprintf("Moments check @ %.2f periods: max|⟨w⟩_xy| = %.3e m/s (= %.1e U₀); κₑ ∈ [%.3e, %.3e] m²/s",
                   sim.model.clock.time / T_tide, w_max, w_max / U₀, κ_min, κ_max)
    w_max / U₀ > 1e-10 &&
        @warn @sprintf("⟨w⟩_xy is %.2e U₀, not ~1e-18 — the plane average is NOT a clean Reynolds average and every flux in the moments file is suspect.", w_max / U₀)
    κ_min < 0 &&
        @warn @sprintf("κₑ went negative (min %.3e): this Oceananigans version does NOT clip AMD backscatter. K_T will be nonsense wherever that happens — mask it in post-processing.", κ_min)
    return nothing
end
# Under TIDAL_SMOKE a 20-iteration run never reaches T_tide/20, and this check is
# exactly what the smoke test exists to exercise, so it runs per-iteration there.
simulation.callbacks[:moments_check] =
    Callback(moments_health_check,
             get(ENV, "TIDAL_SMOKE", "0") == "1" ? IterationInterval(5) :
                                                   TimeInterval(T_tide / 20))
