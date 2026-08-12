#!/bin/bash
# Softplus sweep driver: profile 4, T ∈ {5,10,20,30} m × √Ri ∈ {0,0.5,1,2,5,10}
# = 24 runs, 4 tidal periods each, then figures and animations.
# All run data lands under outputs/<tag>/; figures under figures/.
#
# PER-T GRIDS. Each T gets a vertical grid refined only at its own pycnocline
# (T_SWEEP = that T), so Nz is 254–290 rather than the 458 a shared grid needs.
# Two consequences, both accepted deliberately to fit a 43 h budget:
#   * a spin-up per T, since a snapshot only restarts onto an identical grid;
#   * T and the discretization vary together, so a difference between two T
#     values is not purely physical. Wall spacing (0.0121 m) and pycnocline
#     resolution (10 cells) ARE matched across all four, so the grids differ
#     only in how coarsely they cover regions with no feature in them.
# Set T_SWEEP="5 10 20 30" below to go back to one shared grid (~42 h).
#
# Measured: 21.3–24.3 ms/iteration, ~150k iterations per 4-period run
#   → 6.4–7.3 h per T block (6 runs + its spin-up), ~28 h total + ~1.5 h figures.
#
# Runs are SEQUENTIAL — one job already saturates the GPU; two concurrent jobs
# measured slightly worse combined throughput than one at a time.
#
# Usage:  ./run_softplus_sweep.sh
#         T_VALUES="10 20" ./run_softplus_sweep.sh        (subset / resume)
#         SKIP_ANIM=1 ./run_softplus_sweep.sh             (no animations)
# Completed cases and existing spin-ups are skipped, so re-running resumes.

set -u
cd /home/tll46/SRIM-2026/Stokes/3D || exit 1
export GKSwstype=100                       # headless GR for plotting

T_VALUES="${T_VALUES:-5 10 20 30}"
SQRT_RI="${SQRT_RI:-0 0.5 1 2 5 10}"
SPIN_PERIODS="${SPIN_PERIODS:-5}"
SKIP_ANIM="${SKIP_ANIM:-0}"

export PROFILE=4

mkdir -p logs
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# case label -> directory suffix: 0.5 -> sqrtRi0p5, 2 -> sqrtRi2
ri_case() { printf 'sqrtRi%s' "$(echo "$1" | sed 's/\./p/')"; }
t_lbl()   { printf '%s' "$(echo "$1" | sed 's/\.0$//;s/\./p/')"; }

started=$(date +%s)
elapsed() { printf '%.1f h' "$(echo "scale=3; ($(date +%s) - $started) / 3600" | bc)"; }

log "=== softplus sweep (per-T grids): T ∈ {$T_VALUES}, √Ri ∈ {$SQRT_RI} ==="

# ---------------- The four T blocks ----------------
# T outer is forced by the per-T grid: every run in a block shares that block's
# grid and spin-up, so interleaving T would mean redoing spin-ups. If the budget
# runs out mid-sweep you therefore end up with complete T rows and missing ones,
# which the figure scripts handle (missing cases become "no data" panels).
for T in $T_VALUES; do
  TL=$(t_lbl "$T")
  # This block's grid: refined at THIS T only.
  export T_SWEEP="$T"

  SPIN_TAG="spinup_T${TL}"
  SPIN_FIELDS="outputs/${SPIN_TAG}/TidalBL3D_${SPIN_TAG}_fields.jld2"
  if [ ! -f "$SPIN_FIELDS" ]; then
      log "[$(elapsed)] T=${T}: spin-up (Ri0 from rest, ${SPIN_PERIODS} periods)"
      # FIELDS3D=1 is mandatory — the 3D snapshot IS the spin-up's product.
      RUN_TAG="$SPIN_TAG" T_STRAT=$T N_PERIODS=$SPIN_PERIODS FIELDS3D=1 \
          julia --project=. -t auto Tidal3D.jl Ri0 > "logs/${SPIN_TAG}.log" 2>&1
  fi
  if [ ! -f "$SPIN_FIELDS" ]; then
      log "  T=${T}: spin-up produced no snapshot — skipping this block"; continue
  fi

  for S in $SQRT_RI; do
    RI=$(ri_case "$S"); TAG="P4_T${TL}_${RI}"
    if [ -f "outputs/${TAG}/TidalBL3D_${TAG}_profiles.jld2" ]; then
        log "  ${TAG} already present — skipping"; continue
    fi
    log "[$(elapsed)] run ${TAG}"
    # LIGHT_OUTPUT writes only what figures 4/5 and the animation read (~0.31 GB
    # instead of ~0.64 GB per run, at no cost in runtime). Set LIGHT_OUTPUT=0 to
    # also keep the vorticity/x-y slices and the 3D snapshots.
    LIGHT_OUTPUT="${LIGHT_OUTPUT:-1}" T_STRAT=$T SPINUP_FILE="$SPIN_FIELDS" \
        julia --project=. -t auto Tidal3D.jl "$RI" > "logs/${TAG}.log" 2>&1 \
        || log "  ${TAG} FAILED (see logs/${TAG}.log)"
    # The resume checkpoint is ~0.15 GB and is dead weight once a run finishes.
    if [ "${KEEP_CHECKPOINTS:-0}" != "1" ] && \
       [ -f "outputs/${TAG}/TidalBL3D_${TAG}_profiles.jld2" ]; then
        rm -f "outputs/${TAG}/TidalBL3D_${TAG}_checkpoint_iteration"*.jld2
    fi
  done
done
log "[$(elapsed)] all simulations done"

# ---------------- 3. Animations (one per case) ----------------
if [ "$SKIP_ANIM" != "1" ]; then
  for T in $T_VALUES; do
    for S in $SQRT_RI; do
      RI=$(ri_case "$S"); TAG="P4_T$(t_lbl "$T")_${RI}"
      [ -f "outputs/${TAG}/TidalBL3D_${TAG}.jld2" ] || continue
      log "  animation ${TAG}"
      T_STRAT=$T timeout 3600 julia --project=. Tidal3Danimation.jl "$RI" \
          >> "logs/post_${TAG}.log" 2>&1 || log "  ${TAG} animation failed"
    done
  done
fi

# ---------------- 4. Figures ----------------
log "figure 4 sweep + figure 5 per case"
T_VALUES="$T_VALUES" SQRT_RI="$SQRT_RI" timeout 3600 \
    julia --project=. Figure4_metres.jl >> logs/post_figures.log 2>&1 || log "  Figure4 failed"
T_VALUES="$T_VALUES" SQRT_RI="$SQRT_RI" timeout 3600 \
    julia --project=. Figure5.jl        >> logs/post_figures.log 2>&1 || log "  Figure5 failed"

log "[$(elapsed)] === DONE ==="
NDONE=$(ls -d outputs/P4_*/ 2>/dev/null | wc -l)
log "completed cases: ${NDONE} / $(( $(echo $T_VALUES | wc -w) * $(echo $SQRT_RI | wc -w) ))"
du -sh outputs 2>/dev/null | tail -1
