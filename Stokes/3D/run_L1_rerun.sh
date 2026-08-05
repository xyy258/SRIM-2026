#!/bin/bash
# Rerun of L = 1·Lz on the NEW taller domain (case_params.jl: Lz = 150 δ_s with a
# 20 δ_s sponge stacked ON TOP; dye tracer removed). Two cases, Ri ∈ {500, 2500}.
#
# The archived spin-up is on the OLD 90 δ_s / 128-level grid, so it can no longer
# be index-copied onto this grid (different physical heights, different Nz). We
# therefore make a FRESH Ri0 spin-up from rest on the new grid — exactly how the
# archived one was made — and branch both stratified cases from its final
# (turbulent, U∞ = 0) snapshot as an EXACT same-grid restart.
#
# Per case, immediately after the sim (so a completed case keeps its outputs even
# if a later one is interrupted):
#   - Tidal3Danimation.jl        u & b' x–z animation (dye panel removed)
#   - Vorticity.jl               spanwise-averaged ⟨ω_y⟩_y(x,z) at 4 phases
#   - VorticityProfileAnim.jl    plane-averaged ⟨ω_x⟩,⟨ω_y⟩ profiles vs t, /ω
# Then Figures 4 & 5 (mixed sweep: L=1 new-domain panels + retained 90δ panels).
#
# Usage:  ./run_L1_rerun.sh

set -u
cd /home/tll46/SRIM-2026/Stokes/3D || exit 1
export GKSwstype=100                 # headless GR

SPIN_PERIODS="${SPIN_PERIODS:-6}"    # Ri0 spin-up length (archived one was ~5)
PERIODS="${PERIODS:-8}"              # stratified-case length (ωt ≈ 50)
FR=1.0                               # L = 1·Lz
RI_VALUES="${RI_VALUES:-Ri500 Ri2500}"

mkdir -p logs
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
periods_of() { grep -oE '\([0-9.]+ periods\)' "logs/$1.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+'; }

SPIN_TAG="L1Lz_Ri0"
SPIN_FILE="output_${SPIN_TAG}/TidalBL3D_${SPIN_TAG}_fields.jld2"

log "=== L=1·Lz rerun on new 150δ domain: fresh Ri0 spin-up (${SPIN_PERIODS}p) → Ri ∈ {$RI_VALUES} (${PERIODS}p) ==="

# ---------------- 1. Fresh Ri0 spin-up from rest on the new grid ----------------
log "spin-up ${SPIN_TAG} from rest: ${SPIN_PERIODS} periods"
L_STRAT_LZ=$FR N_PERIODS=$SPIN_PERIODS SPINUP_FILE="__from_rest__" \
    julia --project=.. -t 16 Tidal3D.jl Ri0 > logs/${SPIN_TAG}.log 2>&1 \
    || { log "  spin-up FAILED (see logs/${SPIN_TAG}.log)"; exit 1; }
log "  spin-up reached $(periods_of $SPIN_TAG) periods"
[ -f "$SPIN_FILE" ] || { log "  FATAL: no spin-up fields at $SPIN_FILE"; exit 1; }

# ---------------- 2. Stratified cases, branch from the spin-up ------------------
for RI in $RI_VALUES; do
  TAG="L1Lz_${RI}"
  log "run ${TAG}: ${PERIODS} periods (restart from ${SPIN_FILE})"
  L_STRAT_LZ=$FR N_PERIODS=$PERIODS SPINUP_FILE="$SPIN_FILE" \
      julia --project=.. -t 16 Tidal3D.jl $RI > logs/${TAG}.log 2>&1 \
      || { log "  ${TAG} FAILED (see logs/${TAG}.log)"; continue; }
  log "  ${TAG} reached $(periods_of $TAG) periods"

  if [ -f "output_${TAG}/TidalBL3D_${TAG}.jld2" ]; then
    log "  animation ${TAG}"
    L_STRAT_LZ=$FR timeout 1800 julia --project=.. Tidal3Danimation.jl $RI \
        >> logs/post_${TAG}.log 2>&1 || log "  ${TAG} animation failed"
  fi
  if [ -f "output_${TAG}/TidalBL3D_${TAG}_vortxz.jld2" ]; then
    log "  vorticity (xz) ${TAG}"
    L_STRAT_LZ=$FR timeout 1200 julia --project=.. Vorticity.jl $RI \
        >> logs/post_${TAG}.log 2>&1 || log "  ${TAG} vorticity-xz failed"
  fi
  if [ -f "output_${TAG}/TidalBL3D_${TAG}_profiles.jld2" ]; then
    log "  vorticity-profile animation ${TAG}"
    L_STRAT_LZ=$FR timeout 1200 julia --project=.. VorticityProfileAnim.jl $RI \
        >> logs/post_${TAG}.log 2>&1 || log "  ${TAG} vorticity-profile-anim failed"
  fi
done
log "all simulations + per-case post-processing done"

# ---------------- 3. Figures 4 & 5 (mixed sweep) -------------------------------
log "figures 4 & 5 (mixed sweep)"
timeout 1800 julia --project=.. Figure4_metres.jl >> logs/post_figures.log 2>&1 || log "  Figure4_metres failed"
timeout 1800 julia --project=.. Figure5.jl        >> logs/post_figures.log 2>&1 || log "  Figure5 failed"

log "=== DONE ==="
log "  ${SPIN_TAG}: $(periods_of $SPIN_TAG) periods"
for RI in $RI_VALUES; do log "  L1Lz_${RI}: $(periods_of L1Lz_${RI}) periods"; done
du -sh output_L1Lz_Ri* 2>/dev/null
