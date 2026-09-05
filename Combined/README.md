# Comparison of both models

We combine the results of both simulations for each model.

## l against √TKE/N at z = h, T = 10 m

The Stokes (tidal) and Ekman (rotating) columns on one axis, for the softplus
background with the pycnocline at T = 10 m.

```
cd Combined
GKSwstype=100 julia --project=. reduce_ekman_moments_T10.jl  # ~80 s, reads 845 MB
GKSwstype=100 julia --project=. plot_l_vs_qN_T10_combined.jl # seconds
```

| file | what it is |
|---|---|
| `reduce_ekman_moments_T10.jl` | **current**: reduces the moment files under `Data/Ekman_moments/4/r=*, T=10.0/` to `h(t)`, `K_at_h(t)`, `TKE_at_h(t)`, with `K_T` from the whole flux |
| `reduce_ekman_T10.jl` | superseded: the same reduction from the old slice output under `Data/Ekman/4/`, resolved flux only. Kept because it is the only thing that reads those runs |
| `plot_l_vs_qN_T10_combined.jl` | draws `figures/l_vs_q_over_N_ath_T10_combined.png`; prefers the moments reduction and falls back to the old one |
| `mixed_layer_height.jl` | copy of `Stokes/3D/mixed_layer_height.jl`, unmodified |
| `Project.toml`, `Manifest.toml` | copies of `Stokes/3D`'s, the environment known to work |
| `Data/ekman_lengthscales_T10_moments.jld2` | the current reduction's output, ~300 kB |
| `Data/ekman_lengthscales_T10.jld2` | the superseded one |
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

### The two asymmetries, and that they are now closed

Until the 2026-09-04 re-run the two sides were not comparable, for two reasons.

`Ekman 3D.jl` writes no subgrid buoyancy flux (its `diffusivity_fields` writer is
commented out), so the Ekman `K_T` was built from the resolved flux `⟨w'b'⟩` alone,
while the Stokes `K_T` uses `⟨w'b'⟩ + F_sgs`. And `Velocity.jld2` / `Buoyancy.jld2`
are written with `indices = (:, 1, :)`, so the Ekman averages were over 100 x
points at one y rather than the full plane.

`ekmanrun.jl` closed both at once by re-running the column with full-plane moments
including `F_sgs`. The current figure uses the whole flux on both sides, so the
crosses and the circles are the same quantity. The resolved-only Stokes medians
are drawn only if the reduction falls back to the old data.

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

This was done on 2026-09-04 and the column came back complete: 2548 samples per
case over the full 12.73 inertial periods, no non-finite values anywhere, and
`min κₑ ≈ 1e-7 m²/s` so the subgrid closure is live everywhere.
`reduce_ekman_moments_T10.jl` reads it.

### What the subgrid flux turned out to be worth

The share of `K_T` at `z = h` that the subgrid flux carries, against the Stokes
side measured the same way:

| N/f = N/ω | 0.5 | 1 | 2 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|---|
| Ekman | 0.04 | 0.05 | 0.06 | 0.10 | 0.17 | 0.32 | 0.56 |
| Stokes | — | 0.03 | 0.03 | 0.06 | 0.10 | 0.30 | 0.59 |

Two different flows, the same subgrid share as a function of stratification.
This is what the re-run existed to establish: reading the old resolved-only
Ekman `K_T` against the full Stokes curve understated `l` by about a factor of
two at `N/f = 50`, which is why those points sat off the curve.

With both sides on the whole flux, the three Ekman cases that have equilibrated
land on the Stokes fit — `l/l_fit` = 0.91, 0.81, 0.98 at `N/f` = 10, 25, 50.

### A caveat on the weakly stratified cases

Doubling the duration to 12.73 inertial periods helped but did not finish the
job. Over the 4 T_f averaging window at the end of the record:

| N/f | h | drift in h | drift in l | |
|---|---|---|---|---|
| 50 | 6.24 m | −1.4 % | +0.3 % | equilibrated |
| 25 | 8.31 m | +0.9 % | −4.4 % | equilibrated |
| 10 | 11.45 m | +4.1 % | +3.1 % | equilibrated |
| 5 | 14.46 m | +8.1 % | −20.8 % | not |
| 2 | 18.66 m | +6.7 % | −13.9 % | not |
| 1 | 20.66 m | +7.4 % | −11.4 % | not |
| 0.5 | 20.52 m | +7.2 % | −10.1 % | not |

**The direction matters and is the opposite of what was assumed here before.**
`l` is *falling* in the four unconverged cases, not rising, so those points are
upper bounds on the converged `l`, not lower bounds. They are exactly the points
sitting above the Stokes curve at large √TKE/N, and they are still moving towards
it. Whether the excess survives to equilibrium cannot be settled from this record.

The figure draws them as a bar spanning the median of `l` over the first quarter
of the window to the median over the last, with ▼ at the latest value, rather
than as a single symbol.

Because only three cases have equilibrated, no independent Ekman fit is drawn: a
two-parameter saturating curve through three points is not a fit, and the attempt
pinned `L∞` to the edge of the search grid.

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
