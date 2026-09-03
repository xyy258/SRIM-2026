# Comparison of both models

We combine the results of both simulations for each model.

## l against √TKE/N at z = h, T = 10 m

The Stokes (tidal) and Ekman (rotating) columns on one axis, for the softplus
background with the pycnocline at T = 10 m.

```
cd Combined
GKSwstype=100 julia --project=. reduce_ekman_T10.jl          # ~25 min, reads ~10 GB
GKSwstype=100 julia --project=. plot_l_vs_qN_T10_combined.jl # seconds
```

| file | what it is |
|---|---|
| `reduce_ekman_T10.jl` | reduces the Ekman runs under `Data/Ekman/4/r=*, T=10.0/` to `h(t)`, `K_at_h(t)`, `TKE_at_h(t)` |
| `plot_l_vs_qN_T10_combined.jl` | draws `figures/l_vs_q_over_N_ath_T10_combined.png` |
| `mixed_layer_height.jl` | copy of `Stokes/3D/mixed_layer_height.jl`, unmodified |
| `Project.toml`, `Manifest.toml` | copies of `Stokes/3D`'s, the environment known to work |
| `Data/ekman_lengthscales_T10.jld2` | the reduction's output, ~200 kB |
| `ekmanrun.jl` | re-runs the Ekman T = 10 column with `F_sgs` saved — see below |
| `swirles.sh` | the Slurm script that launches `ekmanrun.jl` on the cluster |

Nothing under `Ekman/` is read for code and nothing there is modified. The
Ekman side is read only as `.jld2` data under `Data/`.

### Both sides use the same definitions

`l = K_T(z=h)/√TKE(z=h)`, `x = √TKE(z=h)/N`, `h` from the crossing definition at
0.1 N²_ref, a time boxcar of one twentieth of a forcing period before any ratio
is formed, and a gradient floor of 0.05 N²_ref. The Stokes tidal frequency ω and
the Ekman Coriolis parameter f are both 1e-4 s⁻¹, so N/ω and N/f label the same N
and the colour ramp means the same thing on both sides.

### The one asymmetry — read this before using the figure

`Ekman 3D.jl` writes no subgrid buoyancy flux (its `diffusivity_fields` writer is
commented out), so the Ekman `K_T` is built from the resolved flux `⟨w'b'⟩` alone,
while the Stokes `K_T` uses `⟨w'b'⟩ + F_sgs`. On the Stokes side the subgrid share
at z = h is 0.03 at N/ω = 1 and 0.59 at N/ω = 50, so the omission is negligible at
weak stratification and a factor of two at the strong end.

The figure therefore carries the Stokes medians twice — full `K_T` (circles, what
the fit is made against) and resolved-only `K_T − K_sgs` (squares). **The Ekman
crosses should be read against the squares.**

A second, smaller asymmetry: `Velocity.jld2` and `Buoyancy.jld2` are written with
`indices = (:, 1, :)`, so the Ekman averages are over 100 x points at one y rather
than the full plane. The means subtracted are the true horizontal averages from
`Avg_vel.jld2` and `Avg_b.jld2`, so the fluctuations are unbiased, just noisier.

## Closing both asymmetries — `ekmanrun.jl`

Neither can be fixed offline: AMD sets κₑ from the full 3D gradients, and only a
y slice was saved, so there is nothing on disk to rebuild either quantity from.
`ekmanrun.jl` re-runs the T = 10 column with identical physics — it *includes*
`Ekman/3D Simulation/Parameters.jl` rather than copying it, so the two cannot
drift — and writes the same thirteen plane-averaged profiles `Stokes/3D/Moments.jl`
writes: `U V W B dBdz`, `uu vv ww`, `uw vw wb`, `kappa_sgs F_sgs`. Full-plane
averages, so the y-slice asymmetry goes with the subgrid one.

```
DRY_RUN=1 julia --project=. Combined/ekmanrun.jl   # from the repo root: the plan, no GPU
sbatch Combined/swirles.sh                         # on the cluster
SWEEP_STAGE=check julia --project=. Combined/ekmanrun.jl   # health of what has finished
```

**About 3 h for all seven, in one 12 h job.** A case is 50 000 steps — 40e4 s of
model time with Δt pinned at `max_Δt` = 8 s and the advective CFL at 0.8, so the
ceiling sets the step and N does not change it. The Stokes column measured
0.0058 s per step per Mcell on the same partition (`Stokes/3D/logs/P4_T10_sqrtRi*.log`:
403 000 steps in 1.95 h on 100×100×300), and this grid is 5.0 Mcell, giving
0.40 h a case. Do not calibrate from a desktop GPU — the same case benchmarks at
0.21–0.43 s/step on a shared RTX 4000 Ada, about 10× the cluster rate.

Output costs nothing measurable: all thirteen moments plus the four `Avg_*`
writers came to 0.4320 s/step against 0.4294 s/step for the moments alone,
A/B'd on the same card minutes apart. So the `Avg_*` files stay on and the
existing Ekman scripts run unchanged on this output. Disk is ~110 MB a case into
`Data/Ekman_moments/`, a sibling of `Data/Ekman/` so the existing runs are
untouched; the x-z slices are off by default since nothing here reads them.

If 3 h is still too long, `sbatch --array=0-6 Combined/swirles.sh` runs one
stratification per task and finishes in ~25 min — the cases share nothing, so
there is no ordering to respect. Each case writes its own completion marker, so
the serial and array modes are interchangeable and a re-submission runs only
what is missing.

Once the column is back, `reduce_ekman_T10.jl` should be pointed at the moments
files instead of the slices — `TKE = ½(uu − U² + vv − V² + ww)` and
`F_b = wb + F_sgs` straight from the file, which is what
`Stokes/3D/MixedLayerDiffusivity.jl` already does — and the Ekman crosses then
belong on the circles rather than the squares.

### A caveat on the weakly stratified cases

`h(t)` from `Data/Ekman/4/r=*, T=10.0/Avg_grad_b.jld2`, crossing definition,
over the full 6.37 inertial periods of the existing runs:

| N/f | h at the end | settles to within 5 % of that from |
|---|---|---|
| 50 | 8.34 m | 1.16 T_f |
| 10 | 11.83 m | 4.20 T_f |
| 2 | 17.42 m | never — still growing at 6.37 T_f |
| 0.5 | 15.79 m | never — still growing at 6.37 T_f |

So the run length cannot be cut to save time: at N/f ≤ 2 it is already too short.
It also means the low-N/f Ekman crosses in the figure — the ones sitting well
above the Stokes plateau at large √TKE/N — are measured on a layer that has not
finished deepening, and should be read as a lower bound on `l` rather than as a
converged value.

### First attempt, 2026-09-02: all seven cases failed

Worth recording, because the failure was silent. Every case wrote its `t = 0`
snapshot, stopped at iteration 100 with `NaN found in field u`, exited 0, and was
stamped complete.

Cause: `Ekman 3D.jl` writes the profile-4 softplus as
`log(1 + exp(sharp*(z - T)))`, which overflows to `Inf` above

    z = T + 709.78/sharp = 128.3 m     (T = 10, sharp = 6)

Commit `e278f55` ("Updated Ekman", 2026-09-01) took `Lz` from 100 to 150, so the
grid top `H = Lz + S` went from 120 m — just under that threshold — to 170 m,
just over. On the real grid the naive form is non-finite in 28 of 400 cells,
first at z = 129.2 m; `dBdz` and `κₑ` go `NaN` with it and the `NaN` reaches `u`
through the sponge target.

Two fixes, both in place: `ekmanrun.jl` uses `max(x,0) + log1p(exp(-|x|))`, which
agrees with the naive form to 1.7e-21 where that is finite and tends to
`N²(z - T)` above it; and the driver no longer trusts a clean exit — it checks
that the moments file reached the stop time with finite values before writing a
marker.

**The overflow is still in `Ekman 3D.jl`.** It affects every profile-4 run at the
current domain height, at every `T` in the sweep (threshold `T + 118.3` m, and
even `T = 50` stays below the 170 m top).
