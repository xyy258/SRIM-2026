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
| `plot_l_vs_qN_T10_combined.jl` | draws both figures below; prefers the moments reduction and falls back to the old one |
| `mixed_layer_height.jl` | copy of `Stokes/3D/mixed_layer_height.jl`, unmodified |
| `Project.toml`, `Manifest.toml` | copies of `Stokes/3D`'s, the environment known to work |
| `Data/ekman_lengthscales_T10_moments.jld2` | the current reduction's output, ~300 kB |
| `Data/ekman_lengthscales_T10.jld2` | the superseded one |
| `ekmanrun.jl` | re-runs the Ekman T = 10 column with `F_sgs` saved — see below |
| `swirles.sh` | the Slurm script that launches `ekmanrun.jl` on the cluster |

Nothing under `Ekman/` is read for code and nothing there is modified. The
Ekman side is read only as `.jld2` data under `Data/`.

### Two versions of the figure

`STYLE` selects them; the default `both` writes both in one run.

| `STYLE` | file | what it shows |
|---|---|---|
| `cloud` | `figures/l_vs_q_over_N_ath_T10_combined.png` | every retained time sample, medians on top |
| `errorbars` | `figures/l_vs_q_over_N_ath_T10_combined_errorbars.png` | medians only, with the interquartile range in both `x` and `l` |

Neither summarises the other. The clouds are worth keeping because they are
*trajectories*, not scatter — the loops are the forcing cycle — and that is
invisible once the case is reduced to a bar.

**The bars are not error bars in the usual sense.** They are quartiles of a
strongly autocorrelated time series, so they say what range each case visits
over a forcing cycle, not how uncertain its median is. The uncertainty on the
median is much smaller than the bar; the honest caveat on these points is the
non-equilibration below, not the spread.

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

### The fits

`l = L∞(1 − e^(−x/x₀))`, fitted to the case medians in log `l`. Three are drawn:
each flow alone, and one overall curve through all thirteen cases.

| fit | cases | `L∞` | `x₀` | rms in `l` |
|---|---|---|---|---|
| Stokes | 6 | 0.60 m | 1.48 m | 7.5 % |
| Ekman | 7 | 1.79 m | 5.40 m | 7.5 % |
| **Overall, both flows** | 13 | **1.46 m** | **4.17 m** | **14.9 %** |

Each flow on its own is described well by the saturating form, to 7.5 %. One
curve through both is twice as bad, and the reason is visible in the figure: the
overall fit is dragged up by the Ekman points and then misses the two largest
Stokes cases by about a factor of two. **The two flows do not collapse onto a
single mixing-length law at T = 10 m** — they agree closely below
`√TKE/N ≈ 1` and separate above it, differing mainly in `L∞`, 0.60 m against
1.79 m.

Two things temper that. The Ekman `L∞` is set by the four cases whose `l` has
not equilibrated and is still falling, so it is probably too high and the true
separation is smaller than a factor of three. And neither dataset reaches its
plateau — the largest case is at 86 % (Stokes) and 84 % (Ekman) of its fitted
`L∞` — so both `L∞` values are extrapolations and carry more uncertainty than
the rms suggests.

`fit_sat` searches `L∞ ∈ [0.05, 4.0]` and `x₀ ∈ [0.02, 20]` by brute force and
reports whether the best point landed on a grid edge. It does that because an
earlier version searched only `L∞ ≤ 1.60` and silently returned the lower edge,
0.300 m, for a fit that had too few points to be constrained at all.

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

The figures no longer mark these cases — the drift is reported here and in
`logs/plot_l_vs_qN_T10_combined.log` instead. It still matters for reading the
fits below: the four drifting cases are the ones at large `√TKE/N`, so they are
what pins the Ekman `L∞`, and since their `l` is still falling that `L∞` is
likely an overestimate.

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


## Is the mixing length set by stratification or by shear?

```
cd Combined
GKSwstype=100 julia --project=. plot_shear_scales_T10.jl   # ~3 min
```

Two length scales built from the same TKE at the same height `z = h`, and the
mixing length itself:

    L_K = K_T/√TKE      L_N = √TKE/N      L_s = √TKE/S

with `N = rω` or `rf` the background stratification and
`S = |∂⟨u_h⟩/∂z| = √((∂U/∂z)² + (∂V/∂z)²)` the mean shear. Dividing each by
`√TKE` turns them into times — `τ_K = K_T/TKE`, `τ_N = 1/N`, `τ_s = 1/S` — which
is the dimensionally honest way to plot them.

Neither pipeline stored the shear. Both store plane-averaged `U` and `V`, so `S`
is differenced from those: the Ekman side inside
`reduce_ekman_moments_T10.jl`, the Stokes side inside `plot_shear_scales_T10.jl`
from `TidalBL3D_*_moments.jld2`, with the same time boxcar in both cases.

| figure | what it shows |
|---|---|
| `figures/tau_K_vs_timescales_T10.png` | `τ_K` against `τ_N` and against `τ_s` |
| `figures/L_N_L_s_vs_r_T10.png` | `L_N` and `L_s` against `r`, and their ratio |

No fit lines — this is for looking at before choosing a functional form.

### What they show

| flow | r | L_K (m) | L_N (m) | L_s (m) | L_s/L_N |
|---|---|---|---|---|---|
| Stokes | 1 | 0.478 | 2.896 | 2.151 | **0.74** |
| Stokes | 2 | 0.423 | 1.431 | 2.081 | 1.45 |
| Stokes | 5 | 0.180 | 0.511 | 1.816 | 3.55 |
| Stokes | 10 | 0.096 | 0.282 | 1.369 | 4.86 |
| Stokes | 25 | 0.038 | 0.103 | 0.630 | 6.14 |
| Stokes | 50 | 0.018 | 0.044 | 0.685 | 15.62 |
| Ekman | 0.5 | 1.476 | 9.847 | 12.508 | 1.27 |
| Ekman | 1 | 1.057 | 4.578 | 13.037 | 2.85 |
| Ekman | 2 | 0.777 | 2.960 | 10.651 | 3.60 |
| Ekman | 5 | 0.425 | 1.448 | 4.847 | 3.35 |
| Ekman | 10 | 0.238 | 0.851 | 3.048 | 3.58 |
| Ekman | 25 | 0.109 | 0.376 | 1.187 | 3.16 |
| Ekman | 50 | 0.064 | 0.171 | 0.487 | 2.85 |

**`L_N` is the smaller scale nearly everywhere.** The one exception in the whole
set is Stokes `r = 1`, at `L_s/L_N = 0.74`. So over the range covered these runs
are stratification-limited, and the shear-limited regime the search was aimed at
lies below `r ≈ 1` on the Stokes side and is not reached at all on the Ekman side.

The two flows behave quite differently in the ratio. `L_s/L_N` climbs steeply
with `r` for Stokes, 0.74 to 15.6, but is nearly flat for Ekman, 2.9 to 3.6 with
no trend. In the Ekman runs the shear and the stratification scale together.

`τ_K` follows `τ_N` closely and, in this normalisation, the two flows very nearly
collapse onto one another — considerably better than `L_K` against `L_N` did.
Against `τ_s` the same points scatter and the two flows separate.

### The weighted scale

Testing `L_c = 1/(1/L_N + 1/L_s)` with `c_N = c_s = 1`, so nothing is tuned.
The numbers are the slope of `log L_K` against `log(scale)` and the rms residual
about that straight line — a slope of 1 means straight proportionality:

| set | vs `L_N` | vs `L_s` | vs `L_c` |
|---|---|---|---|
| Stokes | slope 0.82, rms 15.8 % | slope 2.26, rms 35.8 % | **slope 1.01, rms 9.1 %** |
| Ekman | slope 0.82, rms 12.1 % | slope 0.92, rms 17.7 % | slope 0.87, rms 6.9 % |
| both | slope 0.83, rms 14.2 % | slope 1.08, rms 50.9 % | slope 0.91, rms 12.7 % |

The harmonic combination helps both flows, and for Stokes it takes the slope to
1.01 — `L_K ∝ L_c` with no curvature left — while nearly halving the scatter.
That is the result the weighted form was hoped to give.

Three cautions before leaning on it. `L_c` is a monotone function of `L_N` and
`L_s`, so some improvement from adding a second scale is expected with only six
or seven points; the slope moving to 1 is stronger evidence than the rms falling.
Combining both flows still does not work (12.7 %), so `L_c` does not unify them.
And the Stokes `r = 1` and `r = 2` medians — the two that most influence the
slope at the large-`L_N` end — are taken over samples with 28 % of the cycle
discarded for counter-gradient flux, `K_T ≤ 0`. That exclusion is inherent to
the existing Stokes reduction, not new here, but it biases those two medians
upward. The mean shear never vanishes at `z = h` in either flow, so `1/S` needs
no such exclusion.


## Fits, and the weighted length scale

```
cd Combined
GKSwstype=100 julia --project=. plot_shear_scales_T10.jl    # ~3 min, also writes the cache
GKSwstype=100 julia --project=. plot_Lc_candidates_T10.jl   # seconds, reads the cache
```

`plot_shear_scales_T10.jl` writes `Data/shear_scales_T10.jld2`, the per-sample
`L_K`, `L_N`, `L_s` for every case, so the candidate search does not have to
walk 1601 Stokes snapshots per case again.

### τ_K against the two candidate times

Fitted to the case medians, per flow and to both together. Two families are
tried and the better one is drawn per panel — never a pinned one.

| x | set | power law | saturating |
|---|---|---|---|
| `τ_N` | Stokes | b = 0.81, rms 17.0 % | Y∞ = 2156 s, x₀ = 5328 s, **rms 11.4 %** |
| `τ_N` | Ekman | b = 0.86, rms 12.4 % | Y∞ = 4013 s, x₀ = 12593 s, **rms 9.1 %** |
| `τ_N` | both | b = 0.83, rms 16.1 % | Y∞ = 3359 s, x₀ = 9645 s, **rms 14.0 %** |
| `τ_s` | Stokes | **b = 2.30, rms 47.0 %** | pinned |
| `τ_s` | Ekman | **b = 0.94, rms 18.8 %** | rms 20.4 % |
| `τ_s` | both | **b = 1.04, rms 52.5 %** | pinned |

Against `τ_s` the saturating form pins: there is no knee in that data, so `x₀`
runs off the top of its grid and the curve degenerates into a straight line with
two redundant parameters. Better rms, no meaning — hence the pinned check.

The exponents say it plainly. Against `τ_N` the two flows agree, b = 0.81 and
0.86. Against `τ_s` they do not: 2.30 against 0.94.

### Candidate weighted scales

All formed per sample, then reduced to a case median. Free parameters chosen on
both flows together, since a scale needing a different weight per flow has
unified nothing. `b = 1` is the one-parameter proportionality `L_K = A·L_c`,
which is the form a closure would want; `b` free is the same fit with the
exponent released, as the honesty check.

| candidate | param | b=1 rms (both) | b free slope | **saturating on `L_c`, both** |
|---|---|---|---|---|
| `L_N` alone | — | 29.5 % | 0.83 | 14.9 % |
| `L_s` alone | — | 51.7 % | 1.08 | 51.7 % |
| `min(L_N, L_s)` | — | 25.6 % | 0.85 | **10.2 %** |
| harmonic, equal | — | 18.3 % | 0.91 | **10.4 %** |
| harmonic, weighted | β = 1.40 | 17.5 % | 0.92 | 11.7 % |
| p-norm | p = 0.60 | 17.2 % | 0.93 | 11.9 % |
| geometric | α = 0.65 | 17.6 % | 0.94 | 13.2 % |

**No candidate makes `L_K` proportional to `L_c`.** The best `b = 1` fit is
17.2 %, and every combined scale still has a natural exponent near 0.9. The
curvature that the saturating form captures is real and no reweighting removes
it.

**But combining the scales does unify the two flows, once the saturating form is
kept.** One curve through all thirteen cases goes from 14.9 % on `L_N` alone to
10.2 % on `min(L_N, L_s)` and 10.4 % on the equal-weight harmonic — close to the
7.5 % that a single flow reaches on its own. The equal-weight harmonic is the
one to prefer: it ties the minimum to within noise, is smooth rather than kinked,
and has no free parameter to justify.
