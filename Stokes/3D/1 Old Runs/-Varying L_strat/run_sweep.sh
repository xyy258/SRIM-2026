#!/bin/bash
# L_strat sweep driver (deviates from Gayen et al. — see case_params.jl).
#
# New regime shared with a colleague's Ekman set-up: ω = f = 1e-4, U₀ = 4 cm/s,
# Pr = 10, exponential background N²_bg = N∞²[1 − exp(−z/L)] with L in METRES.
# Sweeps L ∈ {2,4,6,8} m, each for Ri ∈ {500, 2500} → 8 runs, ~8 periods each.
#
# Timing facts measured on 2026-07-30 (RTX 4000 Ada, this grid):
#   * ~59,000 iterations per tidal period, ~independent of grid resolution
#     (CFL is pinned by the near-wall grid + peak-flow velocity, not Δx).
#   * cost per iteration ∝ cell count; at 32×32×128 ≈ 9 ms → ~9 min/period.
#   * concurrency does NOT help — one job already saturates the GPU, two run
#     ~3× slower each. So every run is SEQUENTIAL.
# Full sweep therefore ≈ 8 runs × 8 periods × ~9 min + spin-up + figures ≈ 10–11 h.
#
# Protocol: one fresh turbulent spin-up (unstratified Ri0, from rest, new regime
# and resolution) seeds all eight stratified runs — their velocity IC is shared
# because Ri0 carries b as a passive scalar, so a single snapshot is a valid
# restart for every (L, Ri). Stratification is then switched on, matching the
# paper's "turbulent spin-up, then stratify" approach.
#
# Usage:  ./run_sweep.sh
#         PERIODS=6 SPIN_PERIODS=4 ./run_sweep.sh      (shorter)
#         L_VALUES="4" RI_VALUES="Ri500" ./run_sweep.sh (subset)

set -u
cd /home/tll46/SRIM-2026/Stokes/3D || exit 1
export GKSwstype=100                 # headless GR for plotting
# OLD: export NX=32 NY=32 NZ=128     # reduced resolution for the sweep
# The grid is now fixed at 48×48×192 in Tidal3D.jl and no longer reads NX/NY/NZ,
# so exporting them here would have no effect while still being logged below.
GRID_DESC="48x48x192 (fixed in Tidal3D.jl)"

SPIN_PERIODS="${SPIN_PERIODS:-5}"
PERIODS="${PERIODS:-8}"
L_VALUES="${L_VALUES:-2 4 6 8}"
RI_VALUES="${RI_VALUES:-Ri500 Ri2500}"
SPIN_CASE=Ri0
SPIN_L=4

mkdir -p logs
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

periods_of() { grep -oE '\([0-9.]+ periods\)' "logs/$1.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+'; }

log "=== L_strat sweep start: L ∈ {$L_VALUES} m, Ri ∈ {$RI_VALUES}, ${PERIODS} periods, grid ${GRID_DESC} ==="

# ---------------- 1. Spin-up (fresh turbulent velocities) ----------------
SPIN_OUT="output_L${SPIN_L}_${SPIN_CASE}"
SPIN_FIELDS="${SPIN_OUT}/TidalBL3D_L${SPIN_L}_${SPIN_CASE}_fields.jld2"
log "spin-up: ${SPIN_CASE} from rest, ${SPIN_PERIODS} periods → $SPIN_FIELDS"
L_STRAT_M=$SPIN_L N_PERIODS=$SPIN_PERIODS \
    julia --project=.. -t 16 Tidal3D.jl $SPIN_CASE > logs/spinup.log 2>&1
if [ ! -f "$SPIN_FIELDS" ]; then
    log "FATAL: spin-up produced no fields snapshot — aborting"; exit 1
fi
log "spin-up done at $(periods_of spinup) periods; seeding stratified runs"

# ---------------- 2. The eight stratified runs (sequential) ----------------
for L in $L_VALUES; do
  for RI in $RI_VALUES; do
    TAG="L${L}_${RI}"
    log "run ${TAG}: ${PERIODS} periods"
    L_STRAT_M=$L N_PERIODS=$PERIODS SPINUP_FILE="$SPIN_FIELDS" \
        julia --project=.. -t 16 Tidal3D.jl $RI > logs/${TAG}.log 2>&1 \
        || log "  ${TAG} FAILED (see logs/${TAG}.log)"
    log "  ${TAG} reached $(periods_of $TAG) periods"
  done
done
log "all simulations done"

# ---------------- 3. Post-processing: animation per case, then figures ----------------
# Only figures 4, 5 and the per-case animation are wanted (user), so the other
# diagnostic scripts are not run.
for L in $L_VALUES; do
  for RI in $RI_VALUES; do
    TAG="L${L}_${RI}"
    [ -f "output_${TAG}/TidalBL3D_${TAG}.jld2" ] || { log "  ${TAG}: no slice data, skip animation"; continue; }
    log "  animation ${TAG}"
    L_STRAT_M=$L timeout 2400 julia --project=.. Tidal3Danimation.jl $RI \
        >> logs/post_${TAG}.log 2>&1 || log "  ${TAG} animation failed"
  done
done

log "figures 4 & 5 (all cases)"
timeout 1800 julia --project=.. Figure4.jl >> logs/post_figures.log 2>&1 || log "  Figure4 failed"
timeout 1800 julia --project=.. Figure5.jl >> logs/post_figures.log 2>&1 || log "  Figure5 failed"

log "=== DONE ==="
for L in $L_VALUES; do for RI in $RI_VALUES; do
  log "  L${L}_${RI}: $(periods_of L${L}_${RI}) periods"
done; done
du -sh output_L*_Ri* 2>/dev/null
