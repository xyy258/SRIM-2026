#!/bin/zsh
# Runs the three Gayen et al. cases in sequence: Ri0 first (it provides the
# turbulent spin-up state), then Ri500 and Ri2500 restarting from it.
# Each case's simulation output, plots, and animation land in output_<case>/.
#
# Usage:  ./run_all.sh            (all three cases)
#         ./run_all.sh Ri500      (just one case)

cd "$(dirname "$0")" || exit 1
export GKSwstype=100    # headless GR so plotting works without a display

cases=("$@")
[ ${#cases[@]} -eq 0 ] && cases=(Ri0 Ri500 Ri2500)

for c in "${cases[@]}"; do
    echo "================ Case $c : simulation ================"
    caffeinate -i julia --project=. -t auto Tidal3D.jl "$c" || {
        echo "Simulation $c FAILED — stopping (later cases depend on Ri0)."
        exit 1
    }
    echo "================ Case $c : profiles ================"
    caffeinate -i julia --project=. Tidal3Dprofiles.jl "$c" \
        || echo "Profile plots failed for $c (simulation data is intact)"
    echo "================ Case $c : animation ================"
    caffeinate -i julia --project=. Tidal3Danimation.jl "$c" \
        || echo "Animation failed for $c (simulation data is intact)"
done

echo "================ ALL CASES COMPLETE ================"
