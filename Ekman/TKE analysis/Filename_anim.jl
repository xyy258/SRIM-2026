# Sets the save folder for new files, and the root file name of the data file.
#
# COPIED FROM "Ekman/3D Simulation/Filename_anim.jl". The if-chain over profile
# is gone — this folder runs one case — and BOTH PATHS ARE ANCHORED TO @__DIR__
# rather than being relative to the repo root. Two reasons:
#
#   * the originals write into "Ekman/3D Simulation/...", and this run must not
#     put anything inside the folder it was copied from;
#   * anchoring to the script means the result does not depend on where julia was
#     started, which a batch job cannot be relied on to control.
#
# The data itself goes in Data/ beside this file, NOT in the shared Ekman/Data
# tree, so a run here can never overwrite one from "3D Simulation".

save_folder = joinpath(@__DIR__, "Animations") * "/"
root        = joinpath(@__DIR__, "Data", @sprintf("r=%.1f, T=%.1f", r, T)) * "/"
mkpath(save_folder)
