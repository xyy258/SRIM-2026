#!/bin/bash
# Re-run L=0.2·Lz and L=0.5·Lz at Ri=2500 on the NEW 150δ grid, branched from the
# existing L1Lz_Ri0 turbulent spin-up (velocities reused; buoyancy re-initialized
# to each case's background via set!(model, b=bᵢ) in Tidal3D.jl). Old 90δ data for
# these two cases was archived to archive_90delta_Ri2500/ beforehand.
# Then regenerate Figure 4 sweep and Figure 5. Uses --project=. (NOT .. — that's an
# empty env; only Tidal3D.jl self-activates, the figure scripts do not).
set -u
cd /home/tll46/SRIM-2026/Stokes/3D || exit 1
export GKSwstype=100
PERIODS=8
SPIN="output_L1Lz_Ri0/TidalBL3D_L1Lz_Ri0_fields.jld2"
mkdir -p logs
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
periods_of() { grep -oE '\([0-9.]+ periods\)' "logs/$1.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+'; }

[ -f "$SPIN" ] || { log "FATAL: no spin-up at $SPIN"; exit 1; }
log "=== rerun L0p2Lz_Ri2500 & L0p5Lz_Ri2500 on 150δ grid (${PERIODS}p each) ==="

# fr → L_STRAT_LZ, tag suffix must match case_params flbl (0.2→0p2, 0.5→0p5)
for FR in 0.2 0.5; do
  LBL=$(echo "$FR" | sed 's/\./p/')
  TAG="L${LBL}Lz_Ri2500"
  log "run ${TAG}: L_STRAT_LZ=${FR}, ${PERIODS} periods (restart from spin-up)"
  L_STRAT_LZ=$FR N_PERIODS=$PERIODS SPINUP_FILE="$SPIN" \
      julia --project=. -t 16 Tidal3D.jl Ri2500 > logs/${TAG}.log 2>&1 \
      || { log "  ${TAG} FAILED (see logs/${TAG}.log)"; continue; }
  log "  ${TAG} reached $(periods_of $TAG) periods"
done

log "figures 4 & 5"
julia --project=. Figure4_metres.jl >> logs/post_figures.log 2>&1 && log "  Figure4 OK" || log "  Figure4 FAILED"
julia --project=. Figure5.jl         >> logs/post_figures.log 2>&1 && log "  Figure5 OK" || log "  Figure5 FAILED"
log "=== DONE ==="
