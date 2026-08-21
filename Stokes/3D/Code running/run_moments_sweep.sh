#!/bin/bash
# Staged driver for the K_T / TKE study: second moments, turbulent diffusivity,
# and the K_T(TKE) exponent.
#
#   ./run_moments_sweep.sh spinup    turbulent spin-up with the drag bottom
#   ./run_moments_sweep.sh stage0    the column's first case, N/ω = 0 → h0   ~2.4 h
#   ./run_moments_sweep.sh stage1    the column: T = 10, N/ω ∈ {0,1,2,10}, 8p ~9.6 h
#   ./run_moments_sweep.sh stage2    geometry: N/ω = 2 across a T list (optional)
#   ./run_moments_sweep.sh stage3    fill: every (T, N/ω) not yet done (optional)
#   ./run_moments_sweep.sh post      re-run MixedLayerDiffusivity.jl only
#   ./run_moments_sweep.sh h0        re-print h0 from the stage-0 run
#
# Each stage can be run on its own and each is resumable: a case that finished
# leaves outputs/<tag>/.done_moments behind and is skipped next time.
#
# ---------------- Before running stages 1 to 3 ----------------
# The default is one column: T = 10 m, N/ω ∈ {0,1,2,10}, 8 periods, which is
# 11.1 h of GPU time including the spin-up.
#
# T = 10 m comes from the previous sweep's figure 5 panels, archived under
# "-Softplus T sweep, no-slip bottom/figures/": the mixed-layer top reaches
# 12.5 m at N/ω = 2 and 10.4 m at N/ω = 10 by period 8, so the turbulent layer
# reaches the pycnocline in every case and how far it entrains past it varies
# strongly with N/ω. That contrast is what K_T is measured from. Those runs had a
# no-slip bottom, and drag raises u* and deepens the layer, so stage 0 measures
# the height again.
#
# Stage 0 is the column's own N/ω = 0 case rather than a throwaway probe. It
# measures h0, the height at which TKE falls to about 1 % of its near-wall peak,
# and is then kept and marked so the column skips it. A pycnocline above the
# turbulent layer is never reached, and the background is unstratified below
# z = T, so such a case measures nothing; measure_h0.jl prints the T to use
# instead. The usable window is roughly 0.3 h0 to 2 h0.
#
# Do not start stage 3 until panel (c) of stage 1 shows the two K_T estimates
# agreeing. K_T_bulk (from wb and F_sgs) and K_T_pe (from B alone) share no code,
# so if they disagree the subgrid flux is wrong and stage 3 would measure the
# closure rather than the flow. `check_gate` below enforces this, and
# STAGE3_FORCE=1 overrides it once the disagreement is understood.
#
# The bottom boundary is quadratic drag rather than the no-slip every run
# currently in outputs/ used, so those runs are not comparable with these and
# their spin-up is not the right restart. `spinup` builds a fresh one under its
# own tag and leaves the no-slip snapshots alone.
#
# SHARP = 2 for every case, the default in case_params.jl. Do not vary it per
# case to equalise the pycnocline cell count: that would make the initial
# pycnocline width depend on T and confound it with T itself.
#
# The periods and the case list are cut from the original plan to fit the 12 h
# allocation: 8 periods rather than 16, and N/ω ∈ {0,1,2,10} rather than
# {0,1,2,5,10}. The full list runs with PERIODS=16 SQRT_RI="0 1 2 5 10" at about
# five times the wall clock. See the header of swirlesrun.jl.
#
# ENV: PERIODS, SPIN_PERIODS, T_VALUES, SQRT_RI, N_PERIODS_STAGE0, CD,
#      Z_DRAG_REF, GRAD_FLOOR, SMOOTH, STAGE3_FORCE, SKIP_POST, DRY_RUN.

set -uo pipefail
cd "$(dirname "$0")" || exit 1
export GKSwstype=100                       # headless GR

STAGE="${1:-help}"

PERIODS="${PERIODS:-8}"                    # 7 survive SKIP_PERIODS=1; 16 was the
                                           # original number and costs twice the
                                           # wall clock.
SPIN_PERIODS="${SPIN_PERIODS:-5}"
STAGE0_PERIODS="${N_PERIODS_STAGE0:-$PERIODS}"   # the probe is a full case
T_VALUES="${T_VALUES:-10}"                 # one column; see the T = 10 note above
SQRT_RI="${SQRT_RI:-0 1 2 10}"             # 0.5 dropped: Δb is tiny there and
                                           # K_T = −∫F/Δb is badly conditioned.
                                           # 5 dropped for the budget, since it
                                           # sits between 2 and 10 on a log axis.
                                           # 0 kept as the unstratified control.
STAGE1_T=3
STAGE2_S=2
STAGE0_T=5
STAGE0_S=0

# The grid is unchanged from the earlier sweeps but the bottom boundary condition
# is not, so the tag carries "drag" and cannot be confused with a no-slip run.
GRID_TAG="${GRID_TAG:-100x100x300_drag}"
SPIN_TAG="spinup_${GRID_TAG}"
SPIN_FIELDS="outputs/${SPIN_TAG}/TidalBL3D_${SPIN_TAG}_fields.jld2"

# One case keeps its 3D snapshots, so there is a route to analysis this sweep did
# not plan for; every other case runs FIELDS3D=0 LIGHT_OUTPUT=1 and saves about
# 1.5 GB. It is the case both stage 1 and stage 2 pass through.
FIELDS3D_CASE="${FIELDS3D_CASE:-P4_T3_sqrtRi2}"

JULIA="${JULIA:-julia}"
JLFLAGS=(--project=. -t auto)

mkdir -p logs outputs
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

t_lbl() { python3 -c "import sys;x=float(sys.argv[1]);print(int(x) if x==int(x) else str(x).replace('.','p'))" "$1"; }
tag_of() { echo "P4_T$(t_lbl "$1")_sqrtRi$(t_lbl "$2")"; }
marker_of() { echo "outputs/$1/.done_moments_${GRID_TAG}"; }
periods_of() { grep -oE '\([0-9.]+ periods\)' "logs/$1.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+'; }

# ---------------- One case ----------------
run_case() {
    local T="$1" S="$2" NP="$3"
    local tag; tag="$(tag_of "$T" "$S")"
    local mark; mark="$(marker_of "$tag")"

    if [ -f "$mark" ]; then
        log "  $tag already complete — skipping"
        return 0
    fi
    # FIELDS3D=0 LIGHT_OUTPUT=1 everywhere except that one case.
    local fields3d=0
    [ "$tag" = "$FIELDS3D_CASE" ] && fields3d=1

    log "  run $tag: T = $T m, N/ω = $S, $NP periods, FIELDS3D=$fields3d"
    # DRY_RUN reports the plan, so it is checked before the spin-up is required;
    # otherwise the plan could not be inspected until the spin-up existed.
    if [ "${DRY_RUN:-0}" = "1" ]; then log "    (DRY_RUN — not launched)"; return 0; fi

    if [ ! -f "$SPIN_FIELDS" ]; then
        log "  FATAL: no drag spin-up at $SPIN_FIELDS — run './run_moments_sweep.sh spinup' first"
        return 1
    fi

    local case_name; case_name="sqrtRi$(t_lbl "$S")"
    PROFILE=4 T_STRAT="$T" N_PERIODS="$NP" \
        LIGHT_OUTPUT=1 FIELDS3D="$fields3d" MOMENTS=1 \
        SPINUP_FILE="$SPIN_FIELDS" \
        ${CD:+CD="$CD"} ${Z_DRAG_REF:+Z_DRAG_REF="$Z_DRAG_REF"} \
        "$JULIA" "${JLFLAGS[@]}" Tidal3D.jl "$case_name" >> "logs/${tag}.log" 2>&1

    if [ -f "outputs/${tag}/TidalBL3D_${tag}_moments.jld2" ]; then
        # The marker, not the file, is the test for completion: an interrupted
        # run still leaves a valid but truncated moments file behind.
        date > "$mark"
        rm -f outputs/"${tag}"/*_checkpoint_iteration*.jld2
        log "  $tag done at $(periods_of "$tag") periods"
    else
        log "  $tag FAILED — see logs/${tag}.log"
        return 1
    fi
}

post_case() {
    local T="$1" S="$2"
    local tag; tag="$(tag_of "$T" "$S")"
    [ -f "outputs/${tag}/TidalBL3D_${tag}_moments.jld2" ] || return 0
    [ "${SKIP_POST:-0}" = "1" ] && return 0
    T_VALUES="$T" N_OVER_OMEGA="$S" \
        ${GRAD_FLOOR:+GRAD_FLOOR="$GRAD_FLOOR"} ${SMOOTH:+SMOOTH="$SMOOTH"} \
        "$JULIA" --project=. MixedLayerDiffusivity.jl >> "logs/post_${tag}.log" 2>&1 \
        || log "  post-processing $tag failed — see logs/post_${tag}.log"
    # The verification block is the result, so echo it rather than bury it.
    sed -n '/^VERIFICATION/,/^  → /p' "logs/post_${tag}.log" | tail -20
}

# ---------------- The stage-3 gate: do the two K_T routes agree? ------------
check_gate() {
    [ "${STAGE3_FORCE:-0}" = "1" ] && { log "STAGE3_FORCE=1 — gate bypassed"; return 0; }
    "$JULIA" --project=. -e '
        using JLD2, Printf
        # Wrapped in a function because at top level the body of a `for` is
        # soft scope, and assigning to worst or nseen there creates locals.
        function gate(tags)
            worst, nseen = 0.0, 0
            for tag in tags
                f = joinpath("outputs", tag, "mixing_$(tag).jld2")
                isfile(f) || continue
                c = jldopen(d -> d["checks"], f)
                r = get(c, "pe_rel_diff", NaN)
                @printf("  %-20s K_T_bulk vs K_T_pe: %6.1f %%   K_sgs/K_T: %.2f\n",
                        tag, 100r, get(c, "K_sgs_over_K_T", NaN))
                if isfinite(r)
                    worst = max(worst, r)
                    nseen += 1
                end
            end
            if nseen == 0
                println("  no stage-1 results found — run stage 1 and its post-processing first")
                return 1
            end
            @printf("  worst disagreement: %.1f %% over %d case(s)\n", 100worst, nseen)
            return worst < 0.3 ? 0 : 1
        end
        exit(gate(ARGS))' "$@"
}

# ---------------- Stages ----------------
case "$STAGE" in

spinup)
    if [ -f "$SPIN_FIELDS" ]; then
        log "drag spin-up already present at $SPIN_FIELDS — nothing to do"
        exit 0
    fi
    log "=== spin-up: Ri = 0 from rest, $SPIN_PERIODS periods, DRAG bottom ==="
    log "    (the no-slip snapshots under outputs/spinup_* are a different wall"
    log "     layer and are left untouched)"
    # FIELDS3D=1 is required, since the 3D snapshot is the product. MOMENTS=0
    # because the spin-up is unstratified and its second moments are of no use.
    PROFILE=4 RUN_TAG="$SPIN_TAG" T_STRAT="$STAGE0_T" N_PERIODS="$SPIN_PERIODS" \
        FIELDS3D=1 LIGHT_OUTPUT=1 MOMENTS=0 ${CD:+CD="$CD"} ${Z_DRAG_REF:+Z_DRAG_REF="$Z_DRAG_REF"} \
        "$JULIA" "${JLFLAGS[@]}" Tidal3D.jl Ri0 >> "logs/${SPIN_TAG}.log" 2>&1
    [ -f "$SPIN_FIELDS" ] || { log "FATAL: spin-up produced no snapshot — see logs/${SPIN_TAG}.log"; exit 1; }
    rm -f outputs/"${SPIN_TAG}"/*_checkpoint_iteration*.jld2
    log "spin-up done at $(periods_of "$SPIN_TAG") periods"
    ;;

stage0)
    log "=== STAGE 0 (probe): N/ω = $STAGE0_S, T = $STAGE0_T m, $STAGE0_PERIODS periods (~1.2 h) ==="
    log "    This run exists to measure h0. Stages 1-3 should be re-parameterised on it."
    run_case "$STAGE0_T" "$STAGE0_S" "$STAGE0_PERIODS" || exit 1
    post_case "$STAGE0_T" "$STAGE0_S"
    "$0" h0
    ;;

h0)
    tag="$(tag_of "$STAGE0_T" "$STAGE0_S")"
    f="outputs/${tag}/TidalBL3D_${tag}_moments.jld2"
    [ -f "$f" ] || { log "no stage-0 moments file at $f — run stage0 first"; exit 1; }
    "$JULIA" --project=. measure_h0.jl "$f" 2>&1 | tee "logs/h0_${tag}.log"
    ;;

stage1)
    log "=== STAGE 1 (closure): T = $STAGE1_T m, N/ω ∈ {$SQRT_RI}, $PERIODS periods (~12 h) ==="
    log "    This is the stage the K_T(TKE) exponent comes from."
    for S in $SQRT_RI; do run_case "$STAGE1_T" "$S" "$PERIODS"; done
    for S in $SQRT_RI; do post_case "$STAGE1_T" "$S"; done
    log "--- stage 1 gate check (this is what stage 3 waits on) ---"
    gate_tags=(); for S in $SQRT_RI; do gate_tags+=("$(tag_of "$STAGE1_T" "$S")"); done
    check_gate "${gate_tags[@]}" \
        && log "GATE PASS — panel (c) agrees; stage 3 may be launched" \
        || log "GATE FAIL — the two K_T routes disagree. Fix the SGS flux before stage 3."
    ;;

stage2)
    log "=== STAGE 2 (geometry): N/ω = $STAGE2_S, T ∈ {$T_VALUES}, $PERIODS periods (~9 h) ==="
    log "    This is the PE / geometry test: does delta_eff track h across T?"
    for T in $T_VALUES; do run_case "$T" "$STAGE2_S" "$PERIODS"; done
    for T in $T_VALUES; do post_case "$T" "$STAGE2_S"; done
    ;;

stage3)
    log "=== STAGE 3 (fill): the remaining 12 of the 4×5 grid (~28 h) ==="
    gate_tags=(); for S in $SQRT_RI; do gate_tags+=("$(tag_of "$STAGE1_T" "$S")"); done
    if ! check_gate "${gate_tags[@]}"; then
        log "GATE FAIL: stage 1's K_T_bulk and K_T_pe do not agree, so the subgrid"
        log "  flux is not trustworthy and 28 h of stage 3 would measure the AMD"
        log "  closure rather than the flow. Fix that first, or set STAGE3_FORCE=1"
        log "  if you have decided the disagreement is understood."
        exit 1
    fi
    log "gate passed — filling the grid"
    for T in $T_VALUES; do for S in $SQRT_RI; do run_case "$T" "$S" "$PERIODS"; done; done
    for T in $T_VALUES; do for S in $SQRT_RI; do post_case "$T" "$S"; done; done
    ;;

post)
    log "=== post-processing every case with a moments file on disk ==="
    T_VALUES="$T_VALUES" N_OVER_OMEGA="$SQRT_RI" \
        ${GRAD_FLOOR:+GRAD_FLOOR="$GRAD_FLOOR"} ${SMOOTH:+SMOOTH="$SMOOTH"} \
        "$JULIA" --project=. MixedLayerDiffusivity.jl 2>&1 | tee logs/post_moments.log
    ;;

*)
    sed -n '2,16p' "$0"
    echo
    echo "Case status on grid tag $GRID_TAG  (checkmark = complete):"
    for T in $T_VALUES; do for S in $SQRT_RI; do
        tag="$(tag_of "$T" "$S")"
        [ -f "$(marker_of "$tag")" ] && echo "  ✓ $tag" || echo "    $tag"
    done; done
    ;;
esac

log "=== $STAGE done ==="
