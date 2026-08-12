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

# Julia depot on the project store, NOT in $HOME. Precompiling CUDA and
# Oceananigans writes several GB of cache files and artifacts, and the cephfs
# home quota cannot hold it: it fails mid-precompile with
#   LLVM ERROR: IO failure on output stream: Disk quota exceeded
# which then surfaces as a broken environment (a zero-byte artifact download, or
# a pidfile error) rather than as an out-of-space message.
#
# $HOME/.julia stays on the path as a SECOND entry: Julia writes only to the
# first depot, but still reads from the rest, so the package sources already
# downloaded under home are reused instead of fetched again.
export JULIA_DEPOT_PATH="/cephfs/store/damtp/tll46/.julia:$HOME/.julia"
mkdir -p /cephfs/store/damtp/tll46/.julia

# Run the Julia script with project activation
srun julia --project="$PROJECT_DIR" "Stokes/3D/swirlesrun4.jl"