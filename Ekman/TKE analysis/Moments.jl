# ==============================================================================
# Second-Moment Diagnostic Writer (Moments.jl)
#
# Generates plane-averaged 1D profiles for TKE and K_T post-processing.
# Output is saved to `<filename>Moments.jld2` at fixed instantaneous intervals.
#
# CRITICAL PARAMETER WARNING:
#   In Parameters.jl, `κ` = 0.41 (von Kármán constant).
#   Molecular diffusivity is strictly `κ₀` (ν₀/Pr = 1e-7 m²/s).
#   Always use `κ₀` in flux calculations.
# ==============================================================================

using Oceananigans.AbstractOperations: ∂z, @at

# ------------------------------------------------------------------------------
# SGS Diffusivity Resolver
# ------------------------------------------------------------------------------

const _CLOSURE_FIELD_PROPERTIES = (:closure_fields, :diffusivity_fields)
const _KAPPA_NAMES = (:κₑ, :κ_e, :κₜ, :kappa_e, :kappaₑ, :κ)

_is_field(x) = x isa Oceananigans.Fields.AbstractField

"""
    find_sgs_diffusivity(model, tracer_name::Symbol)

Dynamically locates the subgrid diffusivity field (`κₑ`) for `tracer_name` within 
the model closure container, supporting tuple closures and version differences.
"""
function find_sgs_diffusivity(model, tracer_name::Symbol)
    prop = findfirst(p -> hasproperty(model, p), _CLOSURE_FIELD_PROPERTIES)
    prop === nothing && error("""
        Moments.jl: Could not find closure auxiliary fields on model.
        Searched: $(join(string.("model.", _CLOSURE_FIELD_PROPERTIES), ", "))
        Properties: $(join(string.(propertynames(model)), ", "))""")

    propname = _CLOSURE_FIELD_PROPERTIES[prop]
    container = getproperty(model, propname)
    @info "Moments.jl: Closure fields located at `model.$propname` ($(typeof(container).name.name))"

    candidates = container isa Tuple ? collect(container) : Any[container]
    tracers = keys(model.tracers)
    tidx = findfirst(==(tracer_name), collect(tracers))
    tidx === nothing && error("Moments.jl: Tracer :$tracer_name not found in model tracers: $(tracers)")

    for (i, c) in enumerate(candidates)
        (c === nothing || !(c isa NamedTuple)) && continue

        # Preferred search: matching known κ property names
        for name in propertynames(c)
            name in _KAPPA_NAMES || continue
            κ_entry = getproperty(c, name)
            κ_field = κ_entry isa NamedTuple ? get(κ_entry, tracer_name, nothing) :
                      κ_entry isa Tuple      ? (tidx <= length(κ_entry) ? κ_entry[tidx] : nothing) :
                      κ_entry
            if _is_field(κ_field)
                @info "Moments.jl: SGS diffusivity resolved as `model.$propname[$i].$name` — $(summary(κ_field))"
                return κ_field
            end
        end

        # Fallback search: any NamedTuple containing fields mapped by tracer
        for name in propertynames(c)
            κ_entry = getproperty(c, name)
            κ_entry isa NamedTuple || continue
            κ_field = get(κ_entry, tracer_name, nothing)
            if _is_field(κ_field)
                @warn "Moments.jl: SGS diffusivity found via fallback at `model.$propname[$i].$name[:$tracer_name]`"
                return κ_field
            end
        end
    end

    error("""
        Moments.jl: Could not find SGS diffusivity for tracer :$tracer_name.
        Ensure an AMD closure exporting κₑ is active.""")
end

# Locate subgrid buoyancy diffusivity
κₑ_b = find_sgs_diffusivity(model, :b)

# ------------------------------------------------------------------------------
# Profile Averaging & Moment Definitions
# ------------------------------------------------------------------------------

# Define horizontal plane-average operator (reduces x and y dimensions)
plane_avg(op) = Field(Average(op, dims = (1, 2)))

# First Moments (Means)
# Note: Sampled instantaneously alongside second moments to prevent Reynolds decomposition phase mismatches.
M_U    = plane_avg(u)         # Mean U velocity (Center)
M_V    = plane_avg(v)         # Mean V velocity (Center)
M_W    = plane_avg(w)         # Mean W velocity (Face) — expected to be ~0 (impermeable wall check)
M_B    = plane_avg(b)         # Mean buoyancy (Center)
M_dBdz = plane_avg(∂z(b))     # Mean buoyancy gradient (Face)

# Second Moments: Variances (at Cell Centers)
# Offline TKE post-processing computes: <u'u'> = <uu> - U², <v'v'> = <vv> - V², <w'w'> = <ww>
M_uu = plane_avg(@at (Center, Center, Center) u * u)
M_vv = plane_avg(@at (Center, Center, Center) v * v)
M_ww = plane_avg(@at (Center, Center, Center) w * w)

# Second Moments: Vertical Turbulent Fluxes (at Cell Faces)
# Note: <w> = 0 implies <u'w'> = <uw> and <w'b'> = <wb> without offline mean subtraction.
M_uw = plane_avg(@at (Center, Center, Face) u * w)
M_vw = plane_avg(@at (Center, Center, Face) v * w)
M_wb = plane_avg(@at (Center, Center, Face) w * b)

# Subgrid-Scale (SGS) Diagnostics (at Cell Faces)
# Total buoyancy flux F_b = <w'b'>_res + F_sgs, where F_sgs = -(κₑ + κ₀) * ∂b/∂z.
# Uses molecular diffusivity κ₀ (NOT von Kármán constant κ). Averaged before product evaluation.
M_F_sgs     = plane_avg(@at (Center, Center, Face) -(κₑ_b + κ₀) * ∂z(b))
M_kappa_sgs = plane_avg(@at (Center, Center, Face) κₑ_b + κ₀)

# ------------------------------------------------------------------------------
# Output Writer Setup
# ------------------------------------------------------------------------------

# Instantaneous output (TimeInterval) is required instead of time-averaging (AveragedTimeInterval)
# to avoid leaking mean-flow variance (e.g., inertial oscillations) into TKE calculations.
simulation.output_writers[:moments] =
    JLD2Writer(model,
               (U = M_U, V = M_V, W = M_W, B = M_B, dBdz = M_dBdz,
                uu = M_uu, vv = M_vv, ww = M_ww,
                uw = M_uw, vw = M_vw, wb = M_wb,
                kappa_sgs = M_kappa_sgs, F_sgs = M_F_sgs),
               filename = filename * "Moments.jld2",
               schedule = TimeInterval(T_f / 200),  # 200 samples per inertial period
               overwrite_existing = true,
               with_halos = false)

@info "Moments.jl: Writing 13 plane-averaged profiles to $(filename)Moments.jld2 " *
      @sprintf("every %.0f s (T_f/200); %.0f total samples", T_f / 200, duration / (T_f / 200))

# ------------------------------------------------------------------------------
# Health Checks & Callbacks
# ------------------------------------------------------------------------------

"""
    moments_health_check(sim)

Validates kinematic consistency (<w> ≈ 0) and verifies SGS diffusivity non-negativity (κₑ ≥ 0).
"""
function moments_health_check(sim)
    compute!(M_W)
    w_max = maximum(abs, interior(M_W))
    κ_min, κ_max = extrema(interior(κₑ_b))

    @info @sprintf("Moments check @ %.2f T_f: max|⟨w⟩| = %.3e m/s (%.1e U∞); κₑ ∈ [%.3e, %.3e] m²/s",
                   sim.model.clock.time / T_f, w_max, w_max / U∞, κ_min, κ_max)

    if w_max / U∞ > 1e-10
        @warn @sprintf("⟨w⟩ is %.2e U∞ — domain average w ≠ 0 violates clean Reynolds decomposition.", w_max / U∞)
    end
    if κ_min < 0
        @warn @sprintf("Negative κₑ detected (min %.3e). AMD backscatter clipping may be inactive.", κ_min)
    end
    return nothing
end

# Register callback: frequent checking during smoke testing, standard checking otherwise
simulation.callbacks[:moments_check] =
    Callback(moments_health_check,
             get(ENV, "EKMAN_SMOKE", "0") == "1" ? IterationInterval(5) : TimeInterval(T_f / 20))