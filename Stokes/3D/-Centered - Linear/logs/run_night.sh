#!/bin/bash
# Autonomous overnight driver for the three Gayen et al. cases.
#
#   1. Waits for the already-running Ri0 job to reach N_TARGET periods, then
#      stops it (every half period is a U_inf = 0 snapshot, so stopping at any
#      half-period boundary keeps the restart phase-consistent).
#   2. Runs Ri500 and Ri2500 concurrently -- they both depend only on the Ri0
#      final snapshot, not on each other, and each job uses < 500 MB of GPU
#      memory, so overlapping them costs nothing in memory.
#   3. Runs the profile/animation post-processing for each case (non-fatal).
#
# Deliberately does not restart Ri0: it is already several periods in.

set -u
cd /home/tll46/SRIM-2026/Stokes/3D || exit 1
export GKSwstype=100          # headless GR for the plotting scripts

N_TARGET="${N_TARGET:-15}"    # periods for Ri0 before we cut it over
DEADLINE_EPOCH="${DEADLINE_EPOCH:-0}"   # absolute wall-clock stop; 0 = none

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

past_deadline() {
    [ "$DEADLINE_EPOCH" -gt 0 ] && [ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]
}

# ---- current model time of a case, in periods, from the probe log ----
periods_of() {
    local case=$1
    grep -oE "PROBE $case  t=[0-9.]+" "logs/probe_${case}.log" 2>/dev/null \
        | tail -1 | grep -oE '[0-9.]+$'
}

# ---------------- Phase 1: let Ri0 reach N_TARGET ----------------
log "Phase 1: waiting for Ri0 to reach ${N_TARGET} periods"
while true; do
    sim_pid=$(pgrep -u "$USER" -f 'Tidal3D.jl Ri0' | head -1)
    if [ -z "$sim_pid" ]; then
        log "Ri0 process is gone (finished or died) -- moving on"
        break
    fi
    t=$(periods_of Ri0)
    if [ -n "$t" ] && awk "BEGIN{exit !($t >= $N_TARGET)}"; then
        log "Ri0 reached $t periods >= $N_TARGET -- stopping it (pid $sim_pid)"
        # Pause briefly so we are not killing mid-write of a JLD2 snapshot.
        sleep 20
        kill "$sim_pid"
        sleep 30
        kill -9 "$sim_pid" 2>/dev/null
        break
    fi
    if past_deadline; then
        log "Deadline hit during Ri0 (at ${t:-?} periods) -- stopping it"
        sleep 20; kill "$sim_pid"; sleep 30; kill -9 "$sim_pid" 2>/dev/null
        break
    fi
    sleep 60
done

pkill -u "$USER" -f 'logs/probe.jl Ri0' 2>/dev/null
final_t=$(periods_of Ri0)
log "Ri0 stopped at ${final_t:-unknown} periods"

if [ ! -f output_Ri0/TidalBL3D_Ri0_fields.jld2 ]; then
    log "FATAL: no Ri0 fields snapshot -- stratified cases cannot start"
    exit 1
fi

# ---------------- Phase 2: Ri500 and Ri2500 concurrently ----------------
# 8 threads each so the two jobs do not oversubscribe the 16 cores.
log "Phase 2: launching Ri500 and Ri2500 concurrently"
for c in Ri500 Ri2500; do
    setsid nohup julia --project=.. -t 8 Tidal3D.jl "$c" \
        > "logs/${c}.log" 2>&1 < /dev/null &
    log "  launched $c (pid $!)"
    sleep 5
    setsid nohup julia --project=.. logs/probe.jl "$c" 360 \
        > "logs/probe_${c}.log" 2>&1 < /dev/null &
done

# Wait for both simulations to exit, with a deadline backstop.
while true; do
    running=$(pgrep -u "$USER" -f 'Tidal3D.jl Ri(500|2500)' | wc -l)
    [ "$running" -eq 0 ] && { log "Both stratified cases finished"; break; }
    if past_deadline; then
        log "Deadline hit with $running stratified job(s) still running -- stopping them"
        log "  Ri500 at $(periods_of Ri500) periods, Ri2500 at $(periods_of Ri2500) periods"
        sleep 20
        pkill -u "$USER" -f 'Tidal3D.jl Ri(500|2500)'
        sleep 30
        pkill -9 -u "$USER" -f 'Tidal3D.jl Ri(500|2500)' 2>/dev/null
        break
    fi
    sleep 120
done

pkill -u "$USER" -f 'logs/probe.jl' 2>/dev/null

# ---------------- Phase 3: post-processing (never fatal) ----------------
log "Phase 3: post-processing"
for c in Ri0 Ri500 Ri2500; do
    [ -f "output_${c}/TidalBL3D_${c}_profiles.jld2" ] || { log "  $c: no data, skipping"; continue; }
    log "  $c: profiles"
    timeout 3600 julia --project=.. Tidal3Dprofiles.jl "$c" \
        >> "logs/post_${c}.log" 2>&1 || log "  $c: profile plots failed (data intact)"
    log "  $c: animation"
    timeout 3600 julia --project=.. Tidal3Danimation.jl "$c" \
        >> "logs/post_${c}.log" 2>&1 || log "  $c: animation failed (data intact)"
done

log "ALL DONE"
for c in Ri0 Ri500 Ri2500; do
    log "  $c reached $(periods_of $c) periods"
done
du -sh output_Ri0 output_Ri500 output_Ri2500 2>/dev/null
