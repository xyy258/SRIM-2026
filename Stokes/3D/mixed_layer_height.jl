# =============================================================================
# mixed_layer_height.jl — ONE definition of the mixed-layer height h.
#
# Every script that needs h includes THIS file. It replaces the three verbatim
# copies of `first_crossing` that used to live in MixedLayerDiffusivity.jl,
# swirlesrun3.jl and Figure5.jl, each carrying a comment telling the reader not
# to improve one copy alone. That warning was the right instinct and the wrong
# mechanism: the definition has now changed, and a single file is the only way
# to change it once.
#
# ---------------------------------------------------------------------------
# THE DEFINITION (H_DEF = peak, the default)
# ---------------------------------------------------------------------------
#     h = the height at which the plane-averaged buoyancy gradient ∂⟨b⟩/∂z
#         is LARGEST.
#
# That is the middle of the pycnocline — the sharpened interface the mixed layer
# has pushed ahead of itself — rather than its base.
#
# PREVIOUSLY (H_DEF = crossing, still available):
#     h = the lowest height, searching upward, at which ∂⟨b⟩/∂z / N²_ref first
#         recovers to H_LEVEL = 0.1 of the background value.
#
# The two are NOT small variations on each other. The crossing sits at the FOOT
# of the interface, where the gradient is only just measurable; the peak sits
# 2–4 m higher in these runs, at the top of the eroded layer. Anything computed
# "at z = h" — K_T, TKE, the mixing length l = K_T/√TKE — is being read at a
# genuinely different place, and the numbers move accordingly.
#
# WHY THE PEAK IS THE BETTER CHOICE HERE
#   · It needs no threshold and no N²_ref. The crossing has two arbitrary
#     constants (0.1, and what to normalise by); the peak has none, and is
#     invariant under any rescaling of b, which matters because the N/ω = 0 case
#     has no physical N²_ref at all and had to be given ω² by hand.
#   · It is where the gradient is LARGEST, so K_T = −F_b/(∂b/∂z) is evaluated
#     where its denominator is best conditioned. The crossing put h at
#     ∂b/∂z = 0.1 N², barely above the GRAD_FLOOR mask (0.05 N²) that K_T is
#     already censored by — the old h sat right on the edge of the masked region.
#
# WHERE IT FAILS, AND IT DOES FAIL
#   A peak is only meaningful if there IS one. In the N/ω = 0 cases the buoyancy
#   is a PASSIVE scalar: nothing restores the profile, turbulence and internal
#   motions stir the whole column, and ∂⟨b⟩/∂z above the layer is noise
#   fluctuating between ~0.4 and ~3 N²_ref with no single interface. The global
#   maximum then picks whichever noise spike is tallest — it wanders over
#   18–31 m between samples, far above the height the turbulence reaches
#   (h₀ ≈ 7 m). `peak_upcrossings` below measures this: a clean interface gives
#   ONE upcrossing of half the peak value, noise gives several. Callers should
#   report it, and h from an ambiguous peak should not be used as a length scale.
#
# A THIRD OPTION, H_DEF = flux: the height where the DOWNGRADIENT buoyancy flux
# −F_b is largest, F_b = ⟨w′b′⟩ + F_sgs. Also threshold-free, and it answers a
# different question again: not "where is the interface" but "where is the
# mixing actually happening". It sits between the other two — inside the
# turbulent layer, below the gradient peak — which matters because everything
# sampled "at z = h" (K_T, TKE, l = K_T/√TKE) is a statement about mixing.
# It is the noisiest of the three: F_b is a small difference of fluctuating
# quantities, and above the layer the internal-wave flux can rival the
# interfacial one, so `peak_upcrossings` matters most here.
#
# ENV
#   H_DEF     peak | crossing | flux   (default peak)
#   H_LEVEL   0.1               crossing only: the fraction of N²_ref
# =============================================================================

const H_DEF   = get(ENV, "H_DEF", "peak")
const H_LEVEL = parse(Float64, get(ENV, "H_LEVEL", "0.1"))
H_DEF in ("peak", "crossing", "flux") ||
    error("H_DEF must be peak, crossing or flux — got \"$H_DEF\"")

# Lowest height at which fv crosses `level` from below, linearly interpolated.
# The old h, kept because Figure5.jl also uses it for the Ri_g = 0.25 crossing,
# which is a different quantity and is NOT affected by the change above.
function first_crossing(z, fv, level; zmin = -Inf)
    for i in 1:length(fv)-1
        z[i] < zmin && continue
        if fv[i] < level <= fv[i+1]
            return z[i] + (level - fv[i]) * (z[i+1] - z[i]) / (fv[i+1] - fv[i])
        end
    end
    return NaN
end

# Height of the largest value of G, refined to sub-grid resolution by fitting a
# parabola through the peak node and its two neighbours.
#
# THE REFINEMENT IS NOT COSMETIC. Without it h can only take the ~300 values of
# the grid, and the grid is stretched to Δz = 0.34 m up there, so h would jump in
# 0.34 m steps — a 3% quantisation on a 10 m layer, aliasing into every time
# series that h feeds. The parabola is exact for a quadratic peak and costs three
# points. Written for a NON-UNIFORM grid via divided differences: the equal-Δz
# formula is wrong here and would bias h downward, toward the finer spacing.
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

# How many times G rises through `frac` of its maximum, below zmax. ONE means a
# single coherent interface and a trustworthy peak; more means the profile has
# several comparable bumps and the global maximum is a lottery between them.
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
# DOWNGRADIENT mixing is a maximum. Taking |F_b| instead would let a
# counter-gradient noise spike win, which is not what "peak flux" means.
function h_profile(G, F)
    H_DEF == "flux" || return G
    F === nothing && error("H_DEF=flux needs the buoyancy flux — pass F = F_b")
    return .-F
end

# The one entry point. G is the gradient on its own grid z; N²_ref is used only
# by the crossing definition; F is the buoyancy flux on the same grid and is
# used only by H_DEF = flux. Both are accepted always so that a call site does
# not have to know which definition is in force.
mixed_layer_height(z, G, N²_ref; zmax = Inf, F = nothing) =
    H_DEF == "crossing" ? first_crossing(z, G ./ N²_ref, H_LEVEL) :
                          peak_height(z, h_profile(G, F); zmax = zmax)

# The ambiguity check on whichever profile the definition actually peaks.
h_upcrossings(z, G; zmax = Inf, F = nothing) =
    H_DEF == "crossing" ? 1 : peak_upcrossings(z, h_profile(G, F); zmax = zmax)
