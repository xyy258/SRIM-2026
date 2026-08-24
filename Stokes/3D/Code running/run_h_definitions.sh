#!/bin/bash
# Run the whole K_T / TKE analysis three times, once per definition of the
# mixed-layer height h, into three self-contained folders.
#
#   h_defs/crossing_0p1/    h = lowest z where ∂⟨b⟩/∂z reaches 0.1 N²_ref
#                           (the original definition, the foot of the interface)
#   h_defs/peak_gradient/   h = z of max ∂⟨b⟩/∂z   (the middle of the interface)
#   h_defs/peak_flux/       h = z of max −F_b      (where the mixing is)
#
# Each folder gets figures/ and logs/ with the same filenames, so the three can be
# compared by switching between folders. The mixing_*.jld2 data files share
# outputs/<tag>/ and are told apart by a suffix (_hcross, _hgrad, _hflux) and by
# the h_def field inside them. Every script downstream reads only the files
# matching its own H_DEF, so the three passes cannot mix whatever order they run
# in.
#
#   ./run_h_definitions.sh              all three
#   ./run_h_definitions.sh peak_flux    just one
#
# ENV: T_VALUES ("5 10"), SQRT_RI ("0 1 2 10"), JULIA, SKIP_PERIODS (3)
set -u
cd "$(dirname "$0")"

JULIA="${JULIA:-julia}"
T_VALUES="${T_VALUES:-5 10}"
SQRT_RI="${SQRT_RI:-0 1 2 10}"
ROOT="${H_DEFS_ROOT:-h_defs}"

run_one() {
    local name="$1" hdef="$2" mixsfx="$3"
    local fig="$ROOT/$name/figures" log="$ROOT/$name/logs"
    mkdir -p "$fig" "$log"
    echo "=============================================================="
    echo "==  $name   (H_DEF=$hdef)  ->  $ROOT/$name"
    echo "=============================================================="

    # 1. K_T and h themselves, one case at a time. Writes
    #    mixing_<tag><MIX_SUFFIX>.jld2 next to the moments file, and
    #    K_T_<tag>.png into this definition's folder.
    H_DEF="$hdef" MIX_SUFFIX="$mixsfx" FIG_DIR="$fig" \
        T_VALUES="$T_VALUES" N_OVER_OMEGA="$SQRT_RI" \
        "$JULIA" --project=. MixedLayerDiffusivity.jl > "$log/post_moments.log" 2>&1 \
        || { echo "  !! MixedLayerDiffusivity failed — see $log/post_moments.log"; return 1; }
    grep -E "^  (1|2|3|3b|4|5)\.|^VERIFICATION" "$log/post_moments.log" | sed 's/^/  /'

    # 2. everything that reads those files back in
    H_DEF="$hdef" FIG_DIR="$fig" LOG_DIR="$log" \
        "$JULIA" --project=. swirlesrun2.jl > "$log/panelD.stdout" 2>&1 \
        || echo "  !! swirlesrun2 failed"
    H_DEF="$hdef" FIG_DIR="$fig" LOG_DIR="$log" \
        "$JULIA" --project=. swirlesrun3.jl > "$log/gamma.stdout" 2>&1 \
        || echo "  !! swirlesrun3 failed"
    H_DEF="$hdef" FIG_DIR="$fig" LOG_DIR="$log" \
        "$JULIA" --project=. swirlesrun4.jl > "$log/length_scales.stdout" 2>&1 \
        || echo "  !! swirlesrun4 failed"
    H_DEF="$hdef" FIG_DIR="$fig" LOG_DIR="$log" K_SOURCE=at_h RESULT_SUFFIX=_ath \
        "$JULIA" --project=. swirlesrun4.jl > "$log/length_scales_ath.stdout" 2>&1 \
        || echo "  !! swirlesrun4 (at_h) failed"

    echo "  $(ls "$fig" | wc -l) figures in $fig"
}

case "${1:-all}" in
    crossing_0p1)  run_one crossing_0p1  crossing _hcross ;;
    peak_gradient) run_one peak_gradient peak     _hgrad  ;;
    peak_flux)     run_one peak_flux     flux     _hflux  ;;
    all)
        run_one crossing_0p1  crossing _hcross
        run_one peak_gradient peak     _hgrad
        run_one peak_flux     flux     _hflux
        ;;
    *) sed -n '2,20p' "$0"; exit 1 ;;
esac
