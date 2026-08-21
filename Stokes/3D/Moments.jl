# Second-moment output for Tidal3D.jl, from which TKE and K_T are measured.
#
# Included by Tidal3D.jl just before run!(simulation). It adds one output writer,
# :moments, which writes plane-averaged profiles to <filename>_moments.jld2 and
# leaves every other writer alone.
#
#   means:   U, V, W, B, dBdz
#   moments: uu, vv, ww (Centers);  uw, vw, wb (Faces)
#   subgrid: kappa_sgs, F_sgs (Faces)
#
# Post-processed by MixedLayerDiffusivity.jl.
#
# ---------------- Why raw moments ----------------
# The domain is periodic in x and y and the forcing is horizontally uniform, so
# the plane average is exactly the Reynolds average:
#
#     U(z,t) = ⟨u⟩_xy        u′ = u − U(z,t)
#
# The tidal flow is horizontally uniform, so it lives entirely in U(z,t) and
# subtracting the plane average removes it exactly, with no time filtering.
#
# Time-averaging is still useful to reduce noise, but it must come after the
# decomposition. Averaging the raw moments first gives
#
#     time_avg(⟨uu⟩) − (time_avg(U))²  =  time_avg(TKE) + Var_t(U)
#
# where Var_t(U), the variance of the tidal mean flow over the window, leaks back
# in as "TKE" — an error of the same size as the quantity itself. Oceananigans'
# AveragedTimeInterval would do this, so every field below is written
# instantaneously on TimeInterval, the means are subtracted per sample offline,
# and only then is TKE smoothed (in MixedLayerDiffusivity.jl).
#
# The fluctuations are not built with AbstractOperations either: u − ⟨u⟩ needs
# the mean computed before the operation is evaluated, and the output writer does
# not guarantee that order.
#
# ---------------- Which terms need a mean subtracted ----------------
# With a rigid lid, an impermeable bottom and incompressibility, ⟨w⟩_xy = 0 at
# every height, so
#
#     ⟨w′b′⟩ = ⟨wb⟩_xy      and      ⟨u′w′⟩ = ⟨uw⟩_xy
#
# need no subtraction. Only the variances do: ⟨u′u′⟩ = ⟨uu⟩ − U²,
# ⟨v′v′⟩ = ⟨vv⟩ − V², ⟨w′w′⟩ = ⟨ww⟩.
#
# W is written anyway as a check: it should come out at about 1e-18, and if it
# does not then every flux here is wrong.
#
# ---------------- Where each quantity lives ----------------
# Variances at Centers, so TKE is a single Center profile; fluxes at Faces, so
# K_T = −F/(dB/dz) is a ratio of two Face quantities and needs no interpolation.
# `@at` sets each location explicitly.
#
# dBdz uses the model's own ∂z(b) rather than an offline difference, so it is the
# same stretched-grid operator the solver uses.

using Oceananigans.AbstractOperations: ∂z, @at

# ---------------- Finding the subgrid diffusivity ----------------
# The closure is a tuple, (AnisotropicMinimumDissipation(), ScalarDiffusivity()),
# so the closure auxiliary fields are a tuple in the same order: AMD is entry 1
# and carries νₑ and a κₑ NamedTuple keyed by tracer name, while the
# ScalarDiffusivity entry has none and is `nothing`.
#
# The name of that container has changed between Oceananigans versions, so
# nothing here is hardcoded: both spellings are searched for and the resolver
# logs what it found. This version also clips κₑ at zero, so backscatter cannot
# leave negative diffusivities behind; the runtime check at the bottom of this
# file confirms that on the actual run.

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

    # A tuple of closures gives a tuple of containers, one per closure, with
    # `nothing` where a closure needs no auxiliary fields. A single closure gives
    # its NamedTuple directly, and NamedTuple is not a subtype of Tuple, so this
    # test separates the two.
    candidates = container isa Tuple ? collect(container) : Any[container]

    tracers = keys(model.tracers)
    tidx = findfirst(==(tracer_name), collect(tracers))
    tidx === nothing && error("Moments.jl: tracer :$tracer_name is not among $(tracers)")

    for (i, c) in enumerate(candidates)
        c === nothing && continue
        c isa NamedTuple || continue

        # Preferred route: a property with one of the known names for κ.
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

        # Fallback: any property that is a NamedTuple keyed by tracer name and
        # holding a Field. This catches a rename this file does not know about.
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

# ---------------- The profiles ----------------
# u, v, w and b are already defined by Tidal3D.jl's output section, which runs
# before this file is included.
plane_avg(op) = Field(Average(op, dims = (1, 2)))

# --- means -----------------------------------------------------------------
# U, V and B repeat what the :profiles writer saves, so that the moments file is
# self-contained: subtracting a mean taken from another file at another sample
# time is exactly the ordering error described above.
M_U    = plane_avg(u)                                   # (Nothing, Nothing, Center)
M_V    = plane_avg(v)                                   # (Nothing, Nothing, Center)
M_W    = plane_avg(w)                                   # (Nothing, Nothing, Face) — the ⟨w⟩ ≈ 1e-18 check
M_B    = plane_avg(b)                                   # (Nothing, Nothing, Center)
M_dBdz = plane_avg(∂z(b))                               # (Nothing, Nothing, Face)

# --- raw second moments ----------------------------------------------------
# Variances at Centers, so TKE is a single Center profile.
M_uu = plane_avg(@at (Center, Center, Center) u * u)
M_vv = plane_avg(@at (Center, Center, Center) v * v)
M_ww = plane_avg(@at (Center, Center, Center) w * w)

# Vertical fluxes at Faces, so K_T = −F/(dB/dz) is a ratio of two Face profiles.
M_uw = plane_avg(@at (Center, Center, Face) u * w)
M_vw = plane_avg(@at (Center, Center, Face) v * w)
M_wb = plane_avg(@at (Center, Center, Face) w * b)

# --- subgrid ---------------------------------------------------------------
# In LES the buoyancy flux is split at the filter scale, and K_T is the
# diffusivity a 1D model with no resolved turbulence would need, so it must carry
# both halves:
#
#     F_b = ⟨w′b′⟩_resolved + F_sgs        (positive upward)
#     F_b = −K_T dB/dz
#
# F_sgs is written already signed, so post-processing simply adds the two.
#
# This is the average of the product, not the product of the averages: κₑ and
# ∂b/∂z are correlated, since AMD responds to local strain, which peaks where the
# gradient is sharp.
M_F_sgs = plane_avg(@at (Center, Center, Face) -(κₑ_b + κ) * ∂z(b))

# The subgrid plus molecular diffusivity on its own, at Faces so that it sits on
# the same nodes as F_sgs and K_T. Post-processing uses it for two things:
#   (1) the subgrid share K_sgs/K_T, with K_sgs = −F_sgs/(dB/dz). Where that
#       approaches 1, K_T describes the closure rather than the flow.
#   (2) comparing K_sgs with this profile, which measures how strongly κₑ and
#       ∂b/∂z are correlated.
# It includes the molecular κ; subtract that constant to recover ⟨κₑ⟩ alone.
M_kappa_sgs = plane_avg(@at (Center, Center, Face) κₑ_b + κ)

# ---------------- The writer ----------------
# Its own "_moments" suffix, so it cannot collide with the x-z slices, the
# profiles or the 3D fields.
#
# TimeInterval, matching the :profiles writer, and not AveragedTimeInterval — see
# the note on ordering at the top of this file.
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

# ---------------- Runtime checks ----------------
# (a) ⟨w⟩_xy should be about 1e-18 relative to U₀. If it is not, the Reynolds
#     decomposition assumed above is wrong and so is every flux.
# (b) κₑ should never go negative, which would give nonsense K_T in isolated
#     cells.
# Both are two cheap reductions at a coarse interval.
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
# A short test run never reaches T_tide/20, and this check is one of the things
# such a run exists to exercise, so it runs every few iterations there.
simulation.callbacks[:moments_check] =
    Callback(moments_health_check,
             get(ENV, "TIDAL_SMOKE", "0") == "1" ? IterationInterval(5) :
                                                   TimeInterval(T_tide / 20))
