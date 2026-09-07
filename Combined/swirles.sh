#!/bin/bash
#SBATCH --job-name=ekmanmoments
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/cephfs/store/damtp/tll46/logs/%x_%j.out   # Standard output log
#SBATCH --error=/cephfs/store/damtp/tll46/logs/%x_%j.err    # Standard error log

# ---------------------------------------------------------------------------
# The Ekman N/f column at T = 10 m, re-run with the subgrid buoyancy flux saved.
# Combined/ekmanrun.jl carries the reasoning; the practical points are:
#
#   1. once, on a login node -- checks the environment and prints the plan
#
#        cd /cephfs/store/damtp/tll46/SRIM-2026
#        SWEEP_STAGE=preflight julia --project=. Combined/ekmanrun.jl
#        DRY_RUN=1             julia --project=. Combined/ekmanrun.jl
#
#   2. then
#
#        sbatch Combined/swirles.sh
#
# All seven cases, one job, about 3 h inside the 12 h wall.
#
# ---------------------------------------------------------------------------
# THE BUDGET. A case is 50 000 steps -- 40e4 s of model time with the step pinned
# at max_Dt = 8 s, the advective CFL sitting at 0.8, so the ceiling sets the step
# and N does not change it. The Stokes column measured 0.0058 s per step per
# Mcell on this partition (logs/P4_T10_sqrtRi*.log: 403 000 steps in 1.95 h on
# 100x100x300), and this grid is 5.0 Mcell, so
#
#        50 000 x 0.0058 x 5.0 = 1450 s = 0.40 h a case, 2.8 h for the column.
#
# CASE_HOURS holds 1.0 h a case as margin, which still fits all seven. The
# wall-clock guard and the per-case markers are kept for the case that overruns:
# re-submitting continues rather than restarts.
#
# IF YOU WANT IT FASTER. The seven cases share nothing -- no spin-up, no restart
# file, no ordering -- so they can go as a seven-task array instead:
#
#        sbatch --array=0-6 Combined/swirles.sh
#
# That turns 3 h of compute into ~25 min, at the cost of seven queue waits, and
# the markers mean the two modes are interchangeable. It is a convenience, not a
# requirement: nothing about the column needs an array to fit 12 h.
#
# WHY IT IS BEING RE-RUN. "Ekman 3D.jl" has its diffusivity_fields writer
# commented out, so the runs on disk have no F_sgs and their K_T is the resolved
# flux alone. On the Stokes side the subgrid share at z = h goes from 0.03 at
# N/omega = 1 to 0.59 at N/omega = 50, so at the strong end that is a factor of
# two. It cannot be recovered offline -- kappa_e needs the full 3D fields and
# only a y slice was saved -- hence this column.
#
# Other stages:  SWEEP_STAGE=preflight | cases | check | auto (default)
#   check   re-reads the finished Moments.jld2 files and reports <w>_xy, the sign
#           of kappa_sgs and the subgrid share -- run it on a login node when the
#           job comes back, before pulling 0.8 GB home over scp:
#
#             SWEEP_STAGE=check julia --project=. Combined/ekmanrun.jl
#
# Output:   Combined/Data/Ekman_moments/4/r=<r>, T=10.0/   ~110 MB a case
# Logs:     Combined/logs/P4_T10_r<r>.log, Combined/logs/params_*.txt
# Nothing under Ekman/ or Combined/Data/Ekman/ is written to.
# ---------------------------------------------------------------------------

# Exit immediately if any command fails
set -eo pipefail

# Project directory
PROJECT_DIR="/cephfs/store/damtp/tll46/SRIM-2026"
mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

# Load modules
module purge
module load julia/1.12.4
module load cuda/12.6.3

# Pass allocated Slurm CPUs to Julia multi-threading
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Put the Julia depot on the project store rather than in $HOME. Precompiling
# CUDA and Oceananigans writes several GB of cache and artifacts, which the home
# quota cannot hold; it then fails part way through precompilation and shows up
# as a broken environment rather than as an out-of-space message.
#
# $HOME/.julia stays on the path as a second entry: Julia writes only to the
# first depot but reads from the rest, so packages already downloaded under home
# are reused rather than fetched again.
export JULIA_DEPOT_PATH="/cephfs/store/damtp/tll46/.julia:$HOME/.julia"
mkdir -p /cephfs/store/damtp/tll46/.julia

# The wall-clock guard needs to know the wall it is working against. 11.5 h of
# the 12 h above leaves half an hour for module loading and the final check.
export WALL_HOURS="${WALL_HOURS:-11.5}"

# ---------------------------------------------------------------------------
# Array mode, only when --array is given. One stratification per task, strongest
# first, so that if only a few slots come free they are the cases where the
# subgrid flux actually moves the points. Without --array this block is skipped
# and the job runs the whole column serially, which is the default.
#
# SKIP_PREFLIGHT because seven tasks precompiling into one shared depot at the
# same moment serialise on Julia's precompile lock at best, and race at worst.
# Step 1 in the header does it once instead; drop the export if you would rather
# each task checked for itself.
# ---------------------------------------------------------------------------
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ALL_RATIOS=(50 25 10 5 2 1 0.5)
    export RATIOS="${ALL_RATIOS[$SLURM_ARRAY_TASK_ID]}"
    export SWEEP_STAGE="cases"
    export SKIP_PREFLIGHT=1
    echo "array task $SLURM_ARRAY_TASK_ID of ${SLURM_ARRAY_JOB_ID:-?}: N/f = $RATIOS"
fi

# Run the Julia script with project activation. The root project is the one the
# existing Ekman runs used, and ekmanrun.jl passes it on to each case it spawns.
srun julia --project="$PROJECT_DIR" "Combined/ekmanrun.jl"
