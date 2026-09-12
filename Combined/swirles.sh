#!/bin/bash
#SBATCH --job-name=lowN
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/cephfs/store/damtp/tll46/logs/%x_%j.out
#SBATCH --error=/cephfs/store/damtp/tll46/logs/%x_%j.err

# ---------------------------------------------------------------------------
# The weakly stratified end of BOTH columns, T = 10 m, into one folder.
#
# WHY. L_s/L_N = N/S = sqrt(Ri), so the crossing L_s = L_N on
# figures/L_N_L_s_vs_r_T10.png is Ri = 1 — the point where the shear scale stops
# being the larger of the two. The medians put it at r ~ 1.4 for Stokes and,
# extrapolated below the sweep, r ~ 0.4 for Ekman. Neither is resolved:
#
#   Stokes  the crossing is bracketed by r = 1 and r = 2 only, and those are the
#           two cases whose medians discard 28 % of the tidal cycle to
#           counter-gradient flux. r = 0.5 exists in outputs/ but PREDATES the
#           moments pipeline — no *_moments.jld2, no mixing_*, no drag marker —
#           so it cannot be used and is re-run here.
#   Ekman   r = 0.5 is the lowest case and is still on the stratification-limited
#           side (Ri = 1.6, shear-limited only 23 % of the time). r = 0.2 puts a
#           point on the far side of the extrapolated crossing.
#
# Both flows get the same r so that N matches case for case, which is what the
# whole comparison rests on.
#
# ---------------------------------------------------------------------------
# HOW TO RUN
#
#   1. once, on a login node — checks the environment, prints the plan, runs no
#      GPU work:
#
#        cd /cephfs/store/damtp/tll46/SRIM-2026
#        MODE=preflight bash Combined/swirles.sh
#
#   2. then
#
#        sbatch Combined/swirles.sh                 # everything, serially
#        sbatch --array=0-2 Combined/swirles.sh     # one case per task
#
#   3. when it comes back, before pulling anything home:
#
#        MODE=status bash Combined/swirles.sh
#
# MODE = preflight | stokes | ekman | all (default) | status
#
# ---------------------------------------------------------------------------
# WHAT COMES BACK, AND HOW TO SCP IT
#
# Everything lands under ONE directory, which is the point:
#
#   Combined/Data/lowN/
#     stokes/P4_T10_sqrtRi0p5/    TidalBL3D_*_moments.jld2, mixing_*_hcross.jld2
#     stokes/P4_T10_sqrtRi0p2/
#     ekman/r=0.2, T=10.0/        Moments.jld2, Avg_*.jld2
#     logs/                       one log a case, plus this script's own
#
#   scp -r <host>:/cephfs/store/damtp/tll46/SRIM-2026/Combined/Data/lowN \
#          ~/SRIM-2026/Combined/Data/
#
# That is ~900 MB. If the link is slow, the analysis only ever reads two files a
# case, so this is enough:
#
#   scp -r --include='*_moments.jld2' --include='mixing_*_hcross.jld2' ...
#   (or:  rsync -av --include='*/' --include='*_moments.jld2' \
#           --include='mixing_*_hcross.jld2' --include='Moments.jld2' \
#           --exclude='*' <host>:.../Combined/Data/lowN ~/SRIM-2026/Combined/Data/)
#
# which is ~60 MB a Stokes case and ~220 MB for the Ekman one.
#
# Nothing under Ekman/, Stokes/3D/outputs/ or Combined/Data/Ekman_moments/ is
# written to. The Stokes spin-up under outputs/ is READ, never modified.
#
# ---------------------------------------------------------------------------
# THE BUDGET
#
#   Stokes  8 tidal periods, 100x100x300, measured 1.95 h a case on this
#           partition (Stokes/3D/logs/P4_T10_sqrtRi*.log). Two cases ~4 h.
#   Ekman   1.6e6 s at max_Dt = 8 s is 200 000 steps on 100x100x400 = 5.0 Mcell
#           at 0.0058 s/step/Mcell, so ~1.6 h. One case.
#
# ~5.6 h of the 12 h wall. The per-case markers mean a re-submission continues
# rather than restarts, so an overrun costs one more queue wait and nothing else.
#
# EKMAN DURATION IS DOUBLED, deliberately. The default column ran 8e5 s and the
# four lowest-N/f cases still had h creeping upward at the end — r = 0.2 will be
# worse, not better. 1.6e6 s is the same reduction window (the last 4 inertial
# periods) taken further into the run, so it is strictly better and still fits.
# It does mean this case is not duration-matched to the existing seven; set
# EKMAN_DURATION=8e5 if you would rather it were.
#
# ---------------------------------------------------------------------------
# ENV
#   MODE              preflight | stokes | ekman | all | status
#   STOKES_RATIOS     default "0.5 0.2"
#   EKMAN_RATIOS      default "0.2"   (r = 0.5 already has a complete moments run
#                                      under Data/Ekman_moments/4; add it here to
#                                      duplicate it as a cross-check)
#   STOKES_PERIODS    default 8, matching the existing column
#   EKMAN_DURATION    default 1.6e6 s
#   LOWN_ROOT         default Combined/Data/lowN
#   DRY_RUN=1         print the commands, launch nothing
# ---------------------------------------------------------------------------

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/cephfs/store/damtp/tll46/SRIM-2026}"
cd "$PROJECT_DIR" || { echo "FATAL: no $PROJECT_DIR"; exit 1; }

MODE="${MODE:-all}"
STOKES_RATIOS="${STOKES_RATIOS:-0.5 0.2}"
EKMAN_RATIOS="${EKMAN_RATIOS:-0.2}"
STOKES_PERIODS="${STOKES_PERIODS:-8}"
EKMAN_DURATION="${EKMAN_DURATION:-1.6e6}"
T_STRAT="${T_STRAT:-10}"
GRID_TAG="${GRID_TAG:-100x100x300_drag}"
LOWN_ROOT="${LOWN_ROOT:-$PROJECT_DIR/Combined/Data/lowN}"

STOKES_OUT="$LOWN_ROOT/stokes"
EKMAN_OUT="$LOWN_ROOT/ekman"
LOWN_LOGS="$LOWN_ROOT/logs"
mkdir -p "$STOKES_OUT" "$EKMAN_OUT" "$LOWN_LOGS" "$PROJECT_DIR/logs"

SPIN_FIELDS="$PROJECT_DIR/Stokes/3D/outputs/spinup_${GRID_TAG}/TidalBL3D_spinup_${GRID_TAG}_fields.jld2"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOWN_LOGS/swirles.log"; }

# 0.5 -> 0p5, 0.25 -> 0p25, 10 -> 10.  Matches case_params.jl's naming, which
# takes 'p' for the decimal point so the case is a safe directory name.
t_lbl() { awk -v x="$1" 'BEGIN{ if (x==int(x)) printf "%d", x;
                                else { s=sprintf("%g",x); gsub(/\./,"p",s); printf "%s", s } }'; }

# ekmanrun.jl names case folders with @sprintf("r=%.1f, T=%.1f"), so an r needing
# two decimals would be silently truncated — 0.25 becomes the folder "r=0.2" and
# collides with a genuine r = 0.2. Refuse rather than write into the wrong place.
ek_check() {
    local r="$1"
    local back; back="$(awk -v r="$r" 'BEGIN{printf "%.1f", r}')"
    if [ "$(awk -v a="$r" -v b="$back" 'BEGIN{print (a==b)?1:0}')" != "1" ]; then
        log "FATAL: Ekman r = $r needs more than one decimal, and ekmanrun.jl's"
        log "  case_dir() formats r with %.1f — it would write into \"r=$back\"."
        log "  Use a one-decimal r (0.1, 0.2, 0.3, 0.4), or fix case_dir() in"
        log "  Combined/ekmanrun.jl and the matching readers in"
        log "  Combined/reduce_ekman_moments_T10.jl first."
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
if [ "$MODE" != "status" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    module purge
    module load julia/1.12.4
    module load cuda/12.6.3
    export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
fi
# The depot goes on the project store: precompiling CUDA and Oceananigans writes
# several GB, which the home quota cannot hold, and it fails part way through as
# a broken environment rather than as an out-of-space message. $HOME/.julia stays
# second so already-downloaded packages are reused.
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/cephfs/store/damtp/tll46/.julia:$HOME/.julia}"
# Only on the cluster: off it, /cephfs does not exist and the mkdir is noise.
[ -d /cephfs/store/damtp/tll46 ] && mkdir -p /cephfs/store/damtp/tll46/.julia
export GKSwstype=100
JULIA="${JULIA:-julia}"

# ---------------------------------------------------------------------------
# One Stokes case: the run, then MixedLayerDiffusivity.jl for h and K_T.
# ---------------------------------------------------------------------------
run_stokes() {
    local r="$1"
    local lbl; lbl="$(t_lbl "$r")"
    local tag="P4_T$(t_lbl "$T_STRAT")_sqrtRi${lbl}"
    local mark="$STOKES_OUT/$tag/.done_moments_${GRID_TAG}"

    if [ -f "$mark" ]; then log "  $tag already complete — skipping"; return 0; fi
    log "  Stokes N/omega = $r  ->  $tag  ($STOKES_PERIODS periods)"
    if [ "${DRY_RUN:-0}" = "1" ]; then log "    (DRY_RUN — not launched)"; return 0; fi
    if [ ! -f "$SPIN_FIELDS" ]; then
        log "  FATAL: no drag spin-up at $SPIN_FIELDS"
        log "    build it with:  cd Stokes/3D && ./'Code running'/run_moments_sweep.sh spinup"
        return 1
    fi

    # LIGHT_OUTPUT=1 FIELDS3D=0 as the rest of the column used. MOMENTS=1 is what
    # writes TidalBL3D_*_moments.jld2, which is the file the Combined analysis
    # actually reads — the r = 0.5 run already in outputs/ predates it.
    ( cd "$PROJECT_DIR/Stokes/3D" && \
      PROFILE=4 T_STRAT="$T_STRAT" N_PERIODS="$STOKES_PERIODS" \
      LIGHT_OUTPUT=1 FIELDS3D=0 MOMENTS=1 \
      SPINUP_FILE="$SPIN_FIELDS" OUT_ROOT="$STOKES_OUT" \
      "$JULIA" --project=. -t auto Tidal3D.jl "sqrtRi${lbl}" ) \
      >> "$LOWN_LOGS/${tag}.log" 2>&1

    if [ ! -f "$STOKES_OUT/$tag/TidalBL3D_${tag}_moments.jld2" ]; then
        log "  $tag FAILED — see $LOWN_LOGS/${tag}.log"
        return 1
    fi
    # The marker, not the file, is the test: an interrupted run still leaves a
    # valid but truncated moments file behind.
    date > "$mark"
    rm -f "$STOKES_OUT/$tag"/*_checkpoint_iteration*.jld2

    # h and K_T. H_DEF/MIX_SUFFIX must match what the Combined scripts read,
    # which is mixing_<tag>_hcross.jld2 from the crossing definition at 0.1 N2.
    log "  $tag: post-processing (H_DEF=crossing)"
    ( cd "$PROJECT_DIR/Stokes/3D" && \
      OUT_ROOT="$STOKES_OUT" T_VALUES="$T_STRAT" N_OVER_OMEGA="$r" \
      H_DEF=crossing H_LEVEL=0.1 MIX_SUFFIX=_hcross \
      FIG_DIR="$LOWN_ROOT/figures" \
      "$JULIA" --project=. MixedLayerDiffusivity.jl ) \
      >> "$LOWN_LOGS/post_${tag}.log" 2>&1 \
      || log "  post-processing $tag failed — see $LOWN_LOGS/post_${tag}.log"

    # K_T_bulk and K_T_pe share no code, so their disagreement is the honest test
    # of whether K_T means anything here. It matters at low N: run_moments_sweep.sh
    # dropped r = 0.5 precisely because delta_b is small and K_T = -F/delta_b is
    # badly conditioned. Echo it rather than bury it in the log.
    sed -n '/^VERIFICATION/,/^  . /p' "$LOWN_LOGS/post_${tag}.log" | tail -20 | tee -a "$LOWN_LOGS/swirles.log"
    log "  $tag done"
}

# ---------------------------------------------------------------------------
# One Ekman case, handed to the existing driver with the output redirected.
# ---------------------------------------------------------------------------
run_ekman() {
    local r="$1"
    ek_check "$r" || return 1
    local dir; dir="$EKMAN_OUT/$(awk -v r="$r" -v t="$T_STRAT" 'BEGIN{printf "r=%.1f, T=%.1f", r, t}')"
    if compgen -G "$dir/.done_moments_*" > /dev/null; then
        log "  Ekman r = $r already complete — skipping"; return 0
    fi
    log "  Ekman N/f = $r  ->  $dir  (duration $EKMAN_DURATION s)"
    if [ "${DRY_RUN:-0}" = "1" ]; then log "    (DRY_RUN — not launched)"; return 0; fi

    OUT_ROOT="$EKMAN_OUT" RATIOS="$r" T_STRAT="$T_STRAT" DURATION="$EKMAN_DURATION" \
    SWEEP_STAGE=cases SKIP_PREFLIGHT=1 WALL_HOURS="${WALL_HOURS:-11.0}" \
        "$JULIA" --project="$PROJECT_DIR" "$PROJECT_DIR/Combined/ekmanrun.jl" \
        >> "$LOWN_LOGS/ekman_r${r}.log" 2>&1
    if compgen -G "$dir/.done_moments_*" > /dev/null; then
        log "  Ekman r = $r done"
    else
        log "  Ekman r = $r FAILED — see $LOWN_LOGS/ekman_r${r}.log"; return 1
    fi
}

# ---------------------------------------------------------------------------
# Array mode: one case per task, across both flows. The cases share nothing —
# no restart, no ordering — so the split is free. Stokes first, since those are
# the two that bracket the crossing.
# ---------------------------------------------------------------------------
TASKS=()
for r in $STOKES_RATIOS; do TASKS+=("stokes $r"); done
for r in $EKMAN_RATIOS;  do TASKS+=("ekman $r");  done

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASKS[@]}" ]; then
        log "array task $SLURM_ARRAY_TASK_ID: nothing to do (${#TASKS[@]} cases)"; exit 0
    fi
    read -r flow r <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
    log "=== array task $SLURM_ARRAY_TASK_ID: $flow r = $r ==="
    case "$flow" in
        stokes) run_stokes "$r"; exit $? ;;
        ekman)  run_ekman  "$r"; exit $? ;;
    esac
fi

# ---------------------------------------------------------------------------
# Serial modes
# ---------------------------------------------------------------------------
case "$MODE" in

preflight)
    log "=== preflight ==="
    log "  project     $PROJECT_DIR"
    log "  output      $LOWN_ROOT"
    log "  Stokes r    $STOKES_RATIOS   ($STOKES_PERIODS periods each)"
    log "  Ekman  r    $EKMAN_RATIOS    (duration $EKMAN_DURATION s)"
    log "  cases       ${#TASKS[@]}  ->  sbatch --array=0-$(( ${#TASKS[@]} - 1 ))"
    [ -f "$SPIN_FIELDS" ] && log "  spin-up     found: $SPIN_FIELDS" \
                          || log "  spin-up     MISSING: $SPIN_FIELDS"
    for r in $EKMAN_RATIOS; do ek_check "$r" || exit 1; done
    log "  checking the Ekman driver's own preflight..."
    OUT_ROOT="$EKMAN_OUT" RATIOS="$EKMAN_RATIOS" T_STRAT="$T_STRAT" \
    DURATION="$EKMAN_DURATION" SWEEP_STAGE=preflight \
        "$JULIA" --project="$PROJECT_DIR" "$PROJECT_DIR/Combined/ekmanrun.jl" 2>&1 | tail -20
    log "  now:  DRY_RUN=1 MODE=all bash Combined/swirles.sh"
    ;;

status)
    log "=== status under $LOWN_ROOT ==="
    for r in $STOKES_RATIOS; do
        tag="P4_T$(t_lbl "$T_STRAT")_sqrtRi$(t_lbl "$r")"
        if [ -f "$STOKES_OUT/$tag/.done_moments_${GRID_TAG}" ]; then
            log "  [done] $tag"
        else
            log "  [ -- ] $tag"
        fi
    done
    for r in $EKMAN_RATIOS; do
        dir="$EKMAN_OUT/$(awk -v r="$r" -v t="$T_STRAT" 'BEGIN{printf "r=%.1f, T=%.1f", r, t}')"
        if compgen -G "$dir/.done_moments_*" > /dev/null; then
            log "  [done] ekman r = $r"
        else
            log "  [ -- ] ekman r = $r"
        fi
    done
    du -sh "$LOWN_ROOT" 2>/dev/null | tee -a "$LOWN_LOGS/swirles.log"
    ;;

stokes) log "=== Stokes, r = $STOKES_RATIOS ==="
        for r in $STOKES_RATIOS; do run_stokes "$r" || exit 1; done ;;

ekman)  log "=== Ekman, r = $EKMAN_RATIOS ==="
        for r in $EKMAN_RATIOS; do run_ekman "$r" || exit 1; done ;;

all)    log "=== both flows: Stokes $STOKES_RATIOS, Ekman $EKMAN_RATIOS ==="
        for r in $STOKES_RATIOS; do run_stokes "$r" || exit 1; done
        for r in $EKMAN_RATIOS;  do run_ekman  "$r" || exit 1; done ;;

*)      log "unknown MODE \"$MODE\" — use preflight | stokes | ekman | all | status"
        exit 1 ;;
esac

log "=== $MODE done ==="
