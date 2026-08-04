#!/bin/bash
# T sweep driver (deviates from Gayen et al. — see case_params.jl).
#
# Set-up: LINEAR background stratification (N²_bg = N∞², uniform) with the initial
# buoyancy carrying a pre-formed bottom mixed layer of depth T:
#     bᵢ(z) = N∞² z [1 − exp(−0.2 (z/T)^5)]
# T is the height of peak departure from linear. Sweeps T ∈ {3, 4.5, 6} m, each
# for Ri ∈ {500, 2500} → 6 runs, ~8 periods each. Same regime as the previous
# exponential sweep: ω = f = 1e-4, U₀ = 4 cm/s, Pr = 10, grid 32×32×128.
#
# Spin-up: NOT re-run. The velocity IC is independent of the buoyancy profile
# (Ri0 carries b as a passive scalar), so the archived turbulent snapshot from
# the previous sweep seeds every case. This skips the ~3 h spin-up.
#
# Concurrency does NOT help (one job saturates the GPU), so runs are SEQUENTIAL.
#
# Usage:  ./run_T_sweep.sh
#         T_VALUES="4.5" RI_VALUES="Ri500" ./run_T_sweep.sh   (subset)

set -u
cd /home/tll46/SRIM-2026/Stokes/3D || exit 1
export GKSwstype=100                 # headless GR for plotting
export NX=32 NY=32 NZ=128            # reduced resolution for the sweep

PERIODS="${PERIODS:-8}"
T_VALUES="${T_VALUES:-3 4.5 6}"
RI_VALUES="${RI_VALUES:-Ri500 Ri2500}"

# Turbulent velocity snapshot from the previous (exponential) sweep's Ri0 spin-up,
# now archived. Same regime, domain and grid, so it is a valid restart here.
SPIN_FIELDS="-Varying L_strat/output_L4_Ri0/TidalBL3D_L4_Ri0_fields.jld2"

mkdir -p logs
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
periods_of() { grep -oE '\([0-9.]+ periods\)' "logs/$1.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+'; }
tag_t() { echo "$1" | sed 's/\./p/'; }   # 4.5 → 4p5 (matches case_params.jl)

if [ ! -f "$SPIN_FIELDS" ]; then
    log "FATAL: archived spin-up not found at $SPIN_FIELDS — aborting"; exit 1
fi
log "=== T sweep start: T ∈ {$T_VALUES} m, Ri ∈ {$RI_VALUES}, ${PERIODS} periods, grid ${NX}×${NY}×${NZ} ==="
log "reusing archived spin-up: $SPIN_FIELDS"

# ---------------- 1. The six stratified runs (sequential) ----------------
for T in $T_VALUES; do
  for RI in $RI_VALUES; do
    TAG="T$(tag_t $T)_${RI}"
    log "run ${TAG}: ${PERIODS} periods"
    T_STRAT_M=$T N_PERIODS=$PERIODS SPINUP_FILE="$SPIN_FIELDS" \
        julia --project=.. -t 16 Tidal3D.jl $RI > logs/${TAG}.log 2>&1 \
        || log "  ${TAG} FAILED (see logs/${TAG}.log)"
    log "  ${TAG} reached $(periods_of $TAG) periods"
  done
done
log "all simulations done"

# ---------------- 2. Post-processing: animation per case, then figures ----------------
# Only figures 4, 5 and the per-case animation are wanted (user), in dimensional
# units (metres), so the other diagnostic scripts and the z/δ figures are skipped.
for T in $T_VALUES; do
  for RI in $RI_VALUES; do
    TAG="T$(tag_t $T)_${RI}"
    [ -f "output_${TAG}/TidalBL3D_${TAG}.jld2" ] || { log "  ${TAG}: no slice data, skip animation"; continue; }
    log "  animation ${TAG}"
    T_STRAT_M=$T timeout 2400 julia --project=.. Tidal3Danimation.jl $RI \
        >> logs/post_${TAG}.log 2>&1 || log "  ${TAG} animation failed"
  done
done

log "figures 4 & 5 (all cases, dimensional units)"
timeout 1800 julia --project=.. Figure4_metres.jl >> logs/post_figures.log 2>&1 || log "  Figure4_metres failed"
timeout 1800 julia --project=.. Figure5.jl        >> logs/post_figures.log 2>&1 || log "  Figure5 failed"

log "=== DONE ==="
for T in $T_VALUES; do for RI in $RI_VALUES; do
  TAG="T$(tag_t $T)_${RI}"
  log "  ${TAG}: $(periods_of $TAG) periods"
done; done
du -sh output_T*_Ri* 2>/dev/null
