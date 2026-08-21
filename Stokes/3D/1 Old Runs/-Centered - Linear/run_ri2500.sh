#!/bin/bash
# Runs the Ri2500 case that the overnight driver could not start.
#
# On the night of 2026-07-22/23 Ri2500 died at model construction with
#   ERROR: Out of GPU memory trying to allocate 6.750 MiB
#   Effective GPU memory usage: 99.78% (19.502 GiB/19.545 GiB)
# because another user's job had grown to fill the shared RTX 4000 Ada. This
# script therefore waits for the device to have room before launching, and goes
# back to waiting if the launch still fails.
#
# It restarts from output_Ri0/TidalBL3D_Ri0_fields.jld2, exactly as Ri500 did,
# so the two stratified cases share the same turbulent initial condition.
#
# Usage:  ./run_ri2500.sh                 (wait up to 24 h for the GPU, run 12 periods)
#         HOURS=5 ./run_ri2500.sh         (different simulation time budget)

set -u
cd /home/tll46/SRIM-2026/Stokes/3D || exit 1
export GKSwstype=100

PERIODS="${PERIODS:-12}"
HOURS="${HOURS:-5}"                 # wall-clock budget once it actually starts
NEED_MIB="${NEED_MIB:-700}"         # the job settles at ~320 MiB; leave headroom
MAX_WAIT_H="${MAX_WAIT_H:-24}"      # give up waiting for the GPU after this

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

periods_of() {
    grep -oE '\([0-9.]+ periods\)' logs/Ri2500.log 2>/dev/null \
        | tail -1 | grep -oE '[0-9.]+'
}

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

[ -f output_Ri0/TidalBL3D_Ri0_fields.jld2 ] || {
    log "FATAL: no Ri0 spin-up snapshot to restart from"; exit 1; }

GIVE_UP=$(( $(date +%s) + MAX_WAIT_H * 3600 ))

while true; do
    # ---- wait for room on the device ----
    while [ "$(free_mib)" -lt "$NEED_MIB" ]; do
        if [ "$(date +%s)" -ge "$GIVE_UP" ]; then
            log "gave up: GPU still full after ${MAX_WAIT_H} h ($(free_mib) MiB free)"
            exit 1
        fi
        sleep 120
    done
    log "GPU has $(free_mib) MiB free — launching Ri2500 for ${PERIODS} periods"

    N_PERIODS=$PERIODS nohup julia --project=.. -t 16 Tidal3D.jl Ri2500 \
        > logs/Ri2500.log 2>&1 < /dev/null &
    PID=$!
    DEADLINE=$(( $(date +%s) + $(awk "BEGIN{printf \"%d\", $HOURS*3600}") ))

    # ---- did it survive construction? ----
    sleep 180
    if ! kill -0 $PID 2>/dev/null && [ -z "$(periods_of)" ]; then
        log "Ri2500 failed again (GPU was $(free_mib) MiB free) — back to waiting"
        sleep 600
        continue
    fi

    # ---- run it out, with a wall-clock backstop ----
    while kill -0 $PID 2>/dev/null; do
        if [ "$(date +%s)" -ge "$DEADLINE" ]; then
            log "budget reached with Ri2500 at $(periods_of) periods — stopping it"
            sleep 20; kill $PID; sleep 30; kill -9 $PID 2>/dev/null
            break
        fi
        sleep 60
    done
    break
done

log "Ri2500 reached $(periods_of) periods"

# ---- post-processing for Ri2500, then the shared figures ----
for script in Tidal3Dprofiles.jl MeanVelocity.jl Vorticity.jl Tidal3Danimation.jl; do
    log "  Ri2500: $script"
    timeout 2400 julia --project=.. "$script" Ri2500 >> logs/post_Ri2500.log 2>&1 \
        || log "  Ri2500: $script failed (simulation data is intact)"
done

# Figures 4 and 5 span all three cases, so they are rebuilt now that Ri2500 exists.
for script in Figure4.jl Figure5.jl; do
    log "  $script"
    timeout 1200 julia --project=.. "$script" >> logs/post_figures.log 2>&1 \
        || log "  $script failed"
done

log "DONE — Ri2500 at $(periods_of) periods"
du -sh output_Ri2500
