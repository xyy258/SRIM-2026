#!/bin/bash
# Overnight driver for the three Gayen et al. (2010) cases, on the paper's own
# dimensional parameters (U₀ = 1.5 cm/s, ν = 1e-6, Re_s = 1788), with the
# exponential background stratification N²_bg(z) = N∞²[1 − exp(−z/L)], L = 10 δ_s.
#
# Schedule (differs from the linear run's driver, which is kept in
# "Centered - Linear/run_night.sh"):
#   Phase 1  ALL THREE cases concurrently, from the start.
#            The previous driver spent its first 3 h spinning up Ri0 to supply a
#            turbulent restart state. That is not needed here: Ri = 0 carries the
#            thermal field as a passive scalar, so the velocity field of the
#            archived linear run is independent of the buoyancy profile and is a
#            valid restart state for every case, Ri = 0 included. All three
#            therefore start from the same turbulent snapshot at the same tidal
#            phase, which is also the paper's protocol ("stratification switched
#            on after turbulent spin-up") and makes the three directly
#            comparable at equal t.
#   Phase 2  Post-processing: per-case profiles, mean-velocity/mixed-layer,
#            vorticity, animation, then the shared figures 4 and 5.
#
# Every half period is a U∞ = 0 snapshot, so cutting a run at a deadline still
# leaves a phase-consistent restart state and complete output up to that point.
#
# Usage:  ./run_night.sh              (7.5 h budget from now)
#         HOURS=6 ./run_night.sh      (different budget)

set -u
cd /home/tll46/SRIM-2026/Stokes/3D || exit 1
export GKSwstype=100          # headless GR for the plotting scripts

HOURS="${HOURS:-7.5}"
N_PERIODS_TARGET="${N_PERIODS_TARGET:-12}"

# Velocity restart state. Pinned to the archived linear run so that the Ri0 job
# running right now cannot overwrite the file the other two are reading.
export SPINUP_FILE="${SPINUP_FILE:-Centered - Linear/output_Ri0/TidalBL3D_Ri0_fields.jld2}"

START=$(date +%s)
BUDGET=$(awk "BEGIN{printf \"%d\", $HOURS*3600}")
# Post-processing measured at ~10 min for three cases on the linear run; 45 min
# is a comfortable reserve that still leaves the bulk of the night to the solver.
POST_RESERVE=2700
SIM_DEADLINE=$((START + BUDGET - POST_RESERVE))

mkdir -p logs figures

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

periods_of() {   # current model time of a case, in periods, from its own log
    grep -oE '\([0-9.]+ periods\)' "logs/$1.log" 2>/dev/null \
        | tail -1 | grep -oE '[0-9.]+'
}

launch() {       # launch $1, echo its pid
    N_PERIODS=$N_PERIODS_TARGET nohup julia --project=.. -t 5 Tidal3D.jl "$1" \
        > "logs/$1.log" 2>&1 < /dev/null &
    echo $!
}

if [ ! -f "$SPINUP_FILE" ]; then
    log "FATAL: no spin-up snapshot at '$SPINUP_FILE'"
    exit 1
fi

log "budget ${HOURS} h; simulations end $(date -d "@$SIM_DEADLINE" '+%T'), then post-processing"
log "restarting velocities from '$SPINUP_FILE'"

# ---------------- Phase 1: all three cases ----------------
# 5 threads each so the three jobs do not oversubscribe the 16 cores. Each needs
# ~320 MiB of device memory; the GPU is shared, so a job can still die on a full
# device — hence the retry below rather than assuming all three survive.
CASES="Ri0 Ri500 Ri2500"
declare -A PID RETRIED
for c in $CASES; do
    PID[$c]=$(launch "$c")
    log "launched $c (pid ${PID[$c]})"
    sleep 90                  # stagger so each claims its GPU memory in turn
    RETRIED[$c]=0
done

# Supervise until every case has finished or the deadline passes. A case that
# dies without reaching its target is relaunched once — on 2026-07-23 Ri2500
# died four minutes in on a GPU another user had filled, and with no retry path
# that case was simply never run.
while :; do
    now=$(date +%s)
    running=0
    for c in $CASES; do
        if kill -0 "${PID[$c]}" 2>/dev/null; then
            running=$((running + 1))
        elif [ "${RETRIED[$c]}" = "0" ] && [ "$now" -lt "$((SIM_DEADLINE - 3600))" ]; then
            reached=$(periods_of "$c")
            case "$reached" in
                ''|0*|1.*|2.*|3.*)
                    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
                    log "$c exited early at ${reached:-0} periods (GPU free: ${free} MiB) — relaunching once"
                    RETRIED[$c]=1
                    PID[$c]=$(launch "$c")
                    running=$((running + 1))
                    ;;
            esac
        fi
    done

    [ "$running" = "0" ] && { log "all cases finished on their own"; break; }

    if [ "$now" -ge "$SIM_DEADLINE" ]; then
        log "simulation deadline reached — stopping what is still running"
        for c in $CASES; do
            kill -0 "${PID[$c]}" 2>/dev/null || continue
            log "  $c at $(periods_of "$c") periods"
        done
        sleep 20              # avoid landing mid-JLD2-write
        for c in $CASES; do kill "${PID[$c]}" 2>/dev/null; done
        sleep 30
        for c in $CASES; do kill -9 "${PID[$c]}" 2>/dev/null; done
        break
    fi
    sleep 30
done

for c in $CASES; do log "$c reached $(periods_of "$c") periods"; done

# ---------------- Phase 2: post-processing (never fatal) ----------------
log "post-processing"
for c in $CASES; do
    [ -f "output_${c}/TidalBL3D_${c}_profiles.jld2" ] || { log "  $c: no data, skipping"; continue; }
    for script in Tidal3Dprofiles.jl MeanVelocity.jl Vorticity.jl Tidal3Danimation.jl; do
        log "  $c: $script"
        timeout 1800 julia --project=.. "$script" "$c" >> "logs/post_${c}.log" 2>&1 \
            || log "  $c: $script failed (simulation data is intact)"
    done
done

for script in Figure4.jl Figure5.jl; do
    log "  $script"
    timeout 1200 julia --project=.. "$script" >> logs/post_figures.log 2>&1 \
        || log "  $script failed"
done

log "ALL DONE"
for c in $CASES; do log "  $c reached $(periods_of $c) periods"; done
du -sh output_Ri0 output_Ri500 output_Ri2500 2>/dev/null
