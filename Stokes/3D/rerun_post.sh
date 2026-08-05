#!/bin/bash
# Re-run ONLY the post-processing for the L=1·Lz cases. The original driver called
# these scripts with `julia --project=..`, which resolves to Stokes/ (no
# Project.toml → empty env), so every post step died with "Oceananigans not
# found". The sims survived only because Tidal3D.jl self-activates via
# Pkg.activate("."). Fix: activate the 3D project explicitly with --project=.
set -u
cd /home/tll46/SRIM-2026/Stokes/3D || exit 1
export GKSwstype=100
FR=1.0
mkdir -p logs
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

for TAG in L1Lz_Ri500 L1Lz_Ri2500; do
  RI=${TAG#L1Lz_}
  : > logs/post_${TAG}.log
  log "animation $TAG"
  L_STRAT_LZ=$FR julia --project=. Tidal3Danimation.jl $RI     >> logs/post_${TAG}.log 2>&1 && log "  animation OK"          || log "  animation FAILED"
  log "vorticity-xz $TAG"
  L_STRAT_LZ=$FR julia --project=. Vorticity.jl $RI            >> logs/post_${TAG}.log 2>&1 && log "  vorticity-xz OK"       || log "  vorticity-xz FAILED"
  log "vorticity-profile-anim $TAG"
  L_STRAT_LZ=$FR julia --project=. VorticityProfileAnim.jl $RI >> logs/post_${TAG}.log 2>&1 && log "  vorticity-profile OK"  || log "  vorticity-profile FAILED"
done

: > logs/post_figures.log
log "Figure 4"
julia --project=. Figure4_metres.jl >> logs/post_figures.log 2>&1 && log "  Figure4 OK" || log "  Figure4 FAILED"
log "Figure 5"
julia --project=. Figure5.jl         >> logs/post_figures.log 2>&1 && log "  Figure5 OK" || log "  Figure5 FAILED"
log "=== POST DONE ==="
