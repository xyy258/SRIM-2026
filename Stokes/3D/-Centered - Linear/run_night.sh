#!/bin/bash
# Overnight driver for the three Gayen et al. (2010) cases, on the paper's own
# dimensional parameters (U₀ = 1.5 cm/s, ν = 1e-6, Re_s = 1788).
#
# Schedule (the "full protocol, fewer periods" plan):
#   Phase 1  Ri0 alone until 8 tidal periods, or the phase-1 deadline.
#            8 periods is ωt = 50.3, the late time in the paper's figure 5.
#   Phase 2  Ri500 and Ri2500 concurrently, each restarting from the final Ri0
#            snapshot, until 12 periods or the simulation deadline. Each job
#            needs ~320 MiB of GPU memory; if the device is too full for the
#            second one it is retried on its own after the first finishes.
#   Phase 3  Post-processing: per-case profiles, mean-velocity/mixed-layer,
#            vorticity, animation, then the shared figures 4 and 5.
#
# Every half period is a U∞ = 0 snapshot, so cutting a run at a deadline still
# leaves a phase-consistent restart state and complete output up to that point.
#
# Usage:  ./run_night.sh              (8 h budget from now)
#         HOURS=6 ./run_night.sh      (different budget)

set -u
cd /home/tll46/SRIM-2026/Stokes/3D || exit 1
export GKSwstype=100          # headless GR for the plotting scripts

HOURS="${HOURS:-8}"
RI0_PERIODS="${RI0_PERIODS:-8}"
STRAT_PERIODS="${STRAT_PERIODS:-12}"

START=$(date +%s)
BUDGET=$(awk "BEGIN{printf \"%d\", $HOURS*3600}")
# 3 h for the spin-up, 4 h of two-up running, 1 h of plotting. Measured rates on
# this machine (contended GPU) are ~0.55 h per period for Ri0 alone and ~0.5 h
# per period for the two stratified cases run concurrently — stratification
# damps the turbulence, so those take larger time steps despite sharing the GPU.
# Ri0 will therefore be cut by its deadline before reaching RI0_PERIODS; that is
# intended, since its only jobs are to supply a turbulent restart state and one
# figure panel, while figures 4 and 5 are about the stratified cases.
PHASE1_DEADLINE=$((START + BUDGET * 3 / 8))
PHASE2_DEADLINE=$((START + BUDGET * 7 / 8))

mkdir -p logs figures

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

periods_of() {   # current model time of a case, in periods, from its own log
    grep -oE '\([0-9.]+ periods\)' "logs/$1.log" 2>/dev/null \
        | tail -1 | grep -oE '[0-9.]+'
}

# Wait for $1 (pid) to exit, or kill it once the wall clock passes $2.
# The 20 s pause before the signal avoids landing mid-JLD2-write.
wait_or_deadline() {
    local pid=$1 deadline=$2 label=$3
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            log "deadline reached with $label still running at $(periods_of "$label") periods — stopping it"
            sleep 20
            kill "$pid" 2>/dev/null
            sleep 30
            kill -9 "$pid" 2>/dev/null
            return 1
        fi
        sleep 30
    done
    return 0
}

log "budget ${HOURS} h; phase 1 ends $(date -d "@$PHASE1_DEADLINE" '+%T'), phase 2 ends $(date -d "@$PHASE2_DEADLINE" '+%T')"

# ---------------- Phase 1: Ri0 spin-up ----------------
log "phase 1: Ri0 to ${RI0_PERIODS} periods"
N_PERIODS=$RI0_PERIODS nohup julia --project=.. -t 16 Tidal3D.jl Ri0 \
    > logs/Ri0.log 2>&1 < /dev/null &
RI0_PID=$!
wait_or_deadline $RI0_PID $PHASE1_DEADLINE Ri0
log "Ri0 stopped at $(periods_of Ri0) periods"

if [ ! -f output_Ri0/TidalBL3D_Ri0_fields.jld2 ]; then
    log "FATAL: no Ri0 fields snapshot — the stratified cases cannot start"
    exit 1
fi

# ---------------- Phase 2: Ri500 + Ri2500 ----------------
# 8 threads each so the two jobs do not oversubscribe the 16 cores.
log "phase 2: Ri500 and Ri2500 to ${STRAT_PERIODS} periods"
N_PERIODS=$STRAT_PERIODS nohup julia --project=.. -t 8 Tidal3D.jl Ri500 \
    > logs/Ri500.log 2>&1 < /dev/null &
P500=$!
sleep 120                     # let Ri500 claim its GPU memory before crowding it
N_PERIODS=$STRAT_PERIODS nohup julia --project=.. -t 8 Tidal3D.jl Ri2500 \
    > logs/Ri2500.log 2>&1 < /dev/null &
P2500=$!

sleep 120
if ! kill -0 $P2500 2>/dev/null && [ -z "$(periods_of Ri2500)" ]; then
    log "Ri2500 failed to start (likely GPU memory) — will retry after Ri500"
    RETRY_2500=1
else
    RETRY_2500=0
fi

for pid in $P500 $P2500; do
    kill -0 "$pid" 2>/dev/null && wait_or_deadline "$pid" $PHASE2_DEADLINE \
        "$([ "$pid" = "$P500" ] && echo Ri500 || echo Ri2500)"
done

# A case that failed to start gets its own slot rather than being squeezed into
# what is left of phase 2. On 2026-07-22 Ri2500 died on a full GPU four minutes
# in, Ri500 then ran right up to PHASE2_DEADLINE, and the retry was skipped
# because the deadline had already passed — so the case was simply never run.
# The retry now takes the plotting hour and pushes post-processing back, and it
# waits for device memory instead of assuming it is available.
if [ "$RETRY_2500" = "1" ]; then
    RETRY_DEADLINE=$((START + BUDGET))
    log "retrying Ri2500 on its own until $(date -d "@$RETRY_DEADLINE" '+%T')"
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 700 ]; do
        [ "$(date +%s)" -ge "$RETRY_DEADLINE" ] && break
        log "  waiting for GPU memory"
        sleep 300
    done
    if [ "$(date +%s)" -lt "$RETRY_DEADLINE" ]; then
        N_PERIODS=$STRAT_PERIODS nohup julia --project=.. -t 16 Tidal3D.jl Ri2500 \
            > logs/Ri2500.log 2>&1 < /dev/null &
        wait_or_deadline $! $RETRY_DEADLINE Ri2500
    else
        log "  no time left for Ri2500 — run ./run_ri2500.sh separately"
    fi
fi

log "Ri500 reached $(periods_of Ri500), Ri2500 reached $(periods_of Ri2500) periods"

# ---------------- Phase 3: post-processing (never fatal) ----------------
log "phase 3: post-processing"
for c in Ri0 Ri500 Ri2500; do
    [ -f "output_${c}/TidalBL3D_${c}_profiles.jld2" ] || { log "  $c: no data, skipping"; continue; }
    for script in Tidal3Dprofiles.jl MeanVelocity.jl Vorticity.jl Tidal3Danimation.jl; do
        log "  $c: $script"
        timeout 2400 julia --project=.. "$script" "$c" >> "logs/post_${c}.log" 2>&1 \
            || log "  $c: $script failed (simulation data is intact)"
    done
done

for script in Figure4.jl Figure5.jl; do
    log "  $script"
    timeout 1200 julia --project=.. "$script" >> logs/post_figures.log 2>&1 \
        || log "  $script failed"
done

log "ALL DONE"
for c in Ri0 Ri500 Ri2500; do log "  $c reached $(periods_of $c) periods"; done
du -sh output_Ri0 output_Ri500 output_Ri2500 2>/dev/null
