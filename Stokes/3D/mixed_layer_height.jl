# One definition of the mixed-layer height h, included by every script that
# needs it, so that the definition can be changed in one place.
#
# ---------------- The three definitions ----------------
#   peak (default)  h is the height at which the plane-averaged buoyancy
#                   gradient ∂⟨b⟩/∂z is largest, i.e. the middle of the
#                   pycnocline rather than its base.
#   crossing        h is the lowest height at which ∂⟨b⟩/∂z / N²_ref recovers
#                   to H_LEVEL (0.1) of the background value — the foot of the
#                   interface, 2–4 m below the peak in these runs.
#   flux            h is the height at which the downgradient buoyancy flux
#                   −F_b is largest, i.e. where the mixing is happening. This
#                   sits between the other two, inside the turbulent layer.
#
# These are not small variations on each other: anything read "at z = h", such
# as K_T, TKE or the mixing length l = K_T/√TKE, is being taken at a genuinely
# different place.
#
# The peak is the default because it needs no threshold and no reference
# gradient, and because it puts h where ∂b/∂z is largest, which is where
# K_T = −F_b/(∂b/∂z) is best conditioned.
#
# A peak is only meaningful if there is one. At N/ω = 0 the buoyancy is passive,
# nothing restores the profile, and the gradient above the layer is noise with no
# single interface, so the largest value is whichever noise spike happens to win.
# `peak_upcrossings` below measures this — one upcrossing means a clean
# interface — and h from an ambiguous peak should not be used as a length scale.
#
# ENV
#   H_DEF     peak | crossing | flux   (default peak)
#   H_LEVEL   0.1               crossing only: the fraction of N²_ref

const H_DEF   = get(ENV, "H_DEF", "peak")
const H_LEVEL = parse(Float64, get(ENV, "H_LEVEL", "0.1"))
H_DEF in ("peak", "crossing", "flux") ||
    error("H_DEF must be peak, crossing or flux — got \"$H_DEF\"")

# Lowest height at which fv crosses `level` from below, linearly interpolated.
# Also used by Figure5.jl for the Ri_g = 0.25 crossing, which is a different
# quantity and is unaffected by the choice of h above.
function first_crossing(z, fv, level; zmin = -Inf)
    for i in 1:length(fv)-1
        z[i] < zmin && continue
        if fv[i] < level <= fv[i+1]
            return z[i] + (level - fv[i]) * (z[i+1] - z[i]) / (fv[i+1] - fv[i])
        end
    end
    return NaN
end

# Height of the largest value of G, refined below the grid spacing by fitting a
# parabola through the peak node and its two neighbours.
#
# Without the refinement h could only take the values of the grid, which is
# stretched to Δz = 0.34 m at these heights, so h would jump in 0.34 m steps and
# that step would alias into every time series built from it. The parabola is
# written with divided differences because the grid is non-uniform; the
# equal-spacing formula would bias h downward.
function peak_height(z, G; zmax = Inf)
    kmax, gmax = 0, -Inf
    for k in eachindex(G)
        z[k] > zmax && break
        (isfinite(G[k]) && G[k] > gmax) && (gmax = G[k]; kmax = k)
    end
    kmax == 0 && return NaN
    (kmax == 1 || kmax == length(G) || z[kmax+1] > zmax) && return z[kmax]

    z1, z2, z3 = z[kmax-1], z[kmax], z[kmax+1]
    g1, g2, g3 = G[kmax-1], G[kmax], G[kmax+1]
    all(isfinite, (g1, g2, g3)) || return z2
    d1 = (g2 - g1) / (z2 - z1)
    d2 = ((g3 - g2) / (z3 - z2) - d1) / (z3 - z1)
    d2 < 0 || return z2                    # not concave: keep the node
    return clamp((z1 + z2) / 2 - d1 / (2d2), z1, z3)
end

# How many times G rises through `frac` of its maximum below zmax. One means a
# single coherent interface and a trustworthy peak; more means the profile has
# several comparable bumps and the largest of them is arbitrary.
function peak_upcrossings(z, G; zmax = Inf, frac = 0.5)
    gmax = -Inf
    for k in eachindex(G)
        z[k] > zmax && break
        isfinite(G[k]) && (gmax = max(gmax, G[k]))
    end
    isfinite(gmax) && gmax > 0 || return 0
    lev, n, below = frac * gmax, 0, true
    for k in eachindex(G)
        z[k] > zmax && break
        isfinite(G[k]) || continue
        if below && G[k] >= lev
            n += 1; below = false
        elseif !below && G[k] < lev
            below = true
        end
    end
    return n
end

# Which profile the peak is taken of: the gradient, or minus the flux so that
# downgradient mixing is a maximum. Using |F_b| instead would let a
# counter-gradient noise spike win.
function h_profile(G, F)
    H_DEF == "flux" || return G
    F === nothing && error("H_DEF=flux needs the buoyancy flux — pass F = F_b")
    return .-F
end

# The entry point. G is the gradient on its own grid z, N²_ref is used only by
# the crossing definition, and F is the buoyancy flux on the same grid, used only
# by H_DEF = flux. Both are always accepted, so a caller does not need to know
# which definition is in use.
mixed_layer_height(z, G, N²_ref; zmax = Inf, F = nothing) =
    H_DEF == "crossing" ? first_crossing(z, G ./ N²_ref, H_LEVEL) :
                          peak_height(z, h_profile(G, F); zmax = zmax)

# The ambiguity check, on whichever profile the definition takes its peak of.
h_upcrossings(z, G; zmax = Inf, F = nothing) =
    H_DEF == "crossing" ? 1 : peak_upcrossings(z, h_profile(G, F); zmax = zmax)
