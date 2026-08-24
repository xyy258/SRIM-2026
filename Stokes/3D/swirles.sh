#!/bin/bash
#SBATCH --job-name=stokestidal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/cephfs/store/damtp/tll46/logs/%x_%j.out   # Standard output log
#SBATCH --error=/cephfs/store/damtp/tll46/logs/%x_%j.err    # Standard error log

# ---------------------------------------------------------------------------
# The N/omega column: T = 10 m, N/omega in {0, 1, 2, 5, 10, 25, 50}, 8 periods.
# swirlesrun7.jl carries the reasoning; the practical points are:
#
#   sbatch swirles.sh          run it
#
# The four cases at N/omega = 0, 1, 2 and 10 are already complete from the
# previous column and are skipped, so this is 3 new cases at ~1.95 h = 5.9 h in
# the 12 h wall. If those four are NOT on this filesystem all seven run, which is
# 13.6 h and does not fit: the wall-clock guard then declines the cases it cannot
# finish and re-submitting the same script continues rather than restarts, since
# every case writes its own completion marker.
#
# Check which of the two it will be BEFORE submitting, on a login node -- this
# prints the plan and exits without touching the GPU:
#
#   cd /cephfs/store/damtp/tll46/SRIM-2026/Stokes/3D
#   DRY_RUN=1 julia --project=. swirlesrun7.jl
#
# Other stages:  SWEEP_STAGE=spinup | cases | post | auto (default)
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

# Run the Julia script with project activation
srun julia --project="$PROJECT_DIR" "Stokes/3D/swirlesrun7.jl"