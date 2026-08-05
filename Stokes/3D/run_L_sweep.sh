#!/bin/bash
# L sweep driver (deviates from Gayen et al. — see case_params.jl).
#
# Set-up: EXPONENTIAL background N²_bg = N∞²[1 − exp(−z/L)], the initial buoyancy
# equal to that background (the -Varying L_strat configuration). L is swept as a
# fraction of the domain height Lz = 90 δ_s ≈ 12.73 m: {0.2, 0.5, 1.0, 1.5} Lz,
# each for Ri ∈ {500, 2500} → 8 runs. Regime: ω = f = 1e-4, U₀ = 4 cm/s, Pr = 10,
# grid 48×48×192, 8 periods (ωt ≈ 50).
#
# Timing (measured): ~30 ms/iter, ~61.5k iters/period → ~4.1 h per 8-period run.
# 8 runs ≈ 33 h of simulation; with a 34 h budget the post-processing must be
# cheap and robust. So each case's animation + vorticity are made IMMEDIATELY
# after that case finishes — if the deadline cuts the last case off, every
# completed case already has its figures. Figures 4 & 5 (all cases) run at the end.
#
# Spin-up is reused from the archived Ri0 snapshot (velocity IC is independent of
# the buoyancy profile), skipping the ~3 h spin-up. Runs are SEQUENTIAL (one job
# saturates the GPU).
#
# Usage:  ./run_L_sweep.sh
#         L_FRACS="0.5" RI_VALUES="Ri500" ./run_L_sweep.sh   (subset)

set -u
cd /home/tll46/SRIM-2026/Stokes/3D || exit 1
export GKSwstype=100                 # headless GR for plotting

PERIODS="${PERIODS:-8}"
L_FRACS="${L_FRACS:-0.2 0.5 1 1.5}"
RI_VALUES="${RI_VALUES:-Ri500 Ri2500}"

# Turbulent velocity snapshot from the archived Ri0 spin-up (same regime, domain
# and grid), a valid restart for every case.
SPIN_FIELDS="-Varying L_strat/output_L4_Ri0/TidalBL3D_L4_Ri0_fields.jld2"

mkdir -p logs
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
periods_of() { grep -oE '\([0-9.]+ periods\)' "logs/$1.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+'; }
tag_l() { echo "$1" | sed 's/\./p/'; }   # 0.2 → 0p2, 1 → 1, 1.5 → 1p5 (matches case_params.jl)

if [ ! -f "$SPIN_FIELDS" ]; then
    log "FATAL: archived spin-up not found at $SPIN_FIELDS — aborting"; exit 1
fi
log "=== L sweep start: L ∈ {$L_FRACS} Lz, Ri ∈ {$RI_VALUES}, ${PERIODS} periods, grid 48×48×192 ==="
log "reusing archived spin-up: $SPIN_FIELDS"

# ---------------- Per case: simulate, then animate + vorticity immediately ------
for FR in $L_FRACS; do
  for RI in $RI_VALUES; do
    TAG="L$(tag_l $FR)Lz_${RI}"
    log "run ${TAG}: ${PERIODS} periods"
    L_STRAT_LZ=$FR N_PERIODS=$PERIODS SPINUP_FILE="$SPIN_FIELDS" \
        julia --project=.. -t 16 Tidal3D.jl $RI > logs/${TAG}.log 2>&1 \
        || { log "  ${TAG} FAILED (see logs/${TAG}.log)"; continue; }
    log "  ${TAG} reached $(periods_of $TAG) periods"

    if [ -f "output_${TAG}/TidalBL3D_${TAG}.jld2" ]; then
      log "  animation ${TAG}"
      L_STRAT_LZ=$FR timeout 1800 julia --project=.. Tidal3Danimation.jl $RI \
          >> logs/post_${TAG}.log 2>&1 || log "  ${TAG} animation failed"
    fi
    if [ -f "output_${TAG}/TidalBL3D_${TAG}_vortxz.jld2" ]; then
      log "  vorticity ${TAG}"
      L_STRAT_LZ=$FR timeout 1200 julia --project=.. Vorticity.jl $RI \
          >> logs/post_${TAG}.log 2>&1 || log "  ${TAG} vorticity failed"
    fi
  done
done
log "all simulations + per-case post-processing done"

# ---------------- Figures 4 & 5 (all cases, dimensional units) ------------------
log "figures 4 & 5 (all cases)"
timeout 1800 julia --project=.. Figure4_metres.jl >> logs/post_figures.log 2>&1 || log "  Figure4_metres failed"
timeout 1800 julia --project=.. Figure5.jl        >> logs/post_figures.log 2>&1 || log "  Figure5 failed"

log "=== DONE ==="
for FR in $L_FRACS; do for RI in $RI_VALUES; do
  TAG="L$(tag_l $FR)Lz_${RI}"
  log "  ${TAG}: $(periods_of $TAG) periods"
done; done
du -sh output_L*Lz_Ri* 2>/dev/null
