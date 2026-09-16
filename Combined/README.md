# Comparison of both models

We combine the results of both simulations for each model.

## Notation, and how the figures are drawn

Length scales, all evaluated at `z = h` and formed per time sample before any
median is taken:

| symbol | definition | where |
|---|---|---|
| `ℓ`, `L_K` | `K_T/√TKE`, the mixing length | every figure |
| `L_N` | `√TKE/N`, the stratification scale | |
| `L_s` | `√TKE/S`, the shear scale | |
| **`L_C`** | **`(ε/S³)^(1/2)`, the Corrsin shear scale** | `plot_l_vs_corrsin_T10.jl` only |
| `L_harm` | `1/L_harm = 1/L_N + 1/L_s`, equal-weight harmonic | `plot_L_K_vs_Lharm_T10.jl` |
| `L_β` | `1/L_β = 1/L_N + β/L_s`, weighted harmonic | `plot_Lcomb_candidates_T10.jl` |
| `L_comb` | any candidate combination of `L_N` and `L_s` | `plot_Lcomb_candidates_T10.jl` |

**`L_C` means the Corrsin scale and nothing else.** The combined scales were
called `L_c` up to 2026-09-07, which collided with it; they are `L_harm`, `L_β`
and `L_comb` now, and the scripts and figures were renamed to match.

Every plotting script here opens with

```julia
default(dpi = 600, fontfamily = "DejaVu Sans")
```

and writes its axis labels, titles and legends as `LaTeXStrings`. The maths is
set by GR's own mathtext, so it follows the TeX shapes whatever `fontfamily`
says; `fontfamily` fixes the surrounding text. Two things GR's mathtext does not
have, learned the hard way: `\text{...}` (so no hyphenated words inside
`\mathrm`, they come out as minus signs), and a line break inside a label. And
`%g` inside a label prints `2.16e + 03`, with the exponent's sign set as a
binary operator — `texnum` in `plot_shear_scales_T10.jl` is there for that.

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
each flow alone, and one overall curve through all sixteen cases.

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
job. Over the 4 T_f averaging window at the end of the record (`r = 0.2` ran
25.46 T_f, the others 12.73):

| N/f | h | drift in h | drift in l | |
|---|---|---|---|---|
| 50 | 6.24 m | −1.4 % | +0.3 % | equilibrated |
| 25 | 8.31 m | +0.9 % | −4.4 % | equilibrated |
| 10 | 11.45 m | +4.1 % | +3.1 % | equilibrated |
| 5 | 14.46 m | +8.1 % | −20.8 % | not |
| 2 | 18.66 m | +6.7 % | −13.9 % | not |
| 1 | 20.66 m | +7.4 % | −11.4 % | not |
| 0.5 | 20.52 m | +7.2 % | −10.1 % | not |
| 0.2 | 22.61 m | +0.9 % | −20.0 % | not |

The `r = 0.2` row is new (2026-09-15). It is the odd one: `h` has settled to
+0.9 %, as well converged as `r = 25`, while `l` is still falling at −20 %. So
`h` equilibrating does not mean `l` has, and the two have to be checked
separately. Its `h` = 22.6 m is the thickest layer in either column — that case
has eaten well past the `T = 10 m` pycnocline.

**The direction matters and is the opposite of what was assumed here before.**
`l` is *falling* in the five unconverged cases, not rising, so those points are
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

Testing `L_harm = 1/(1/L_N + 1/L_s)` with `c_N = c_s = 1`, so nothing is tuned.
The numbers are the slope of `log L_K` against `log(scale)` and the rms residual
about that straight line — a slope of 1 means straight proportionality:

| set | vs `L_N` | vs `L_s` | vs `L_harm` |
|---|---|---|---|
| Stokes | slope 0.82, rms 15.8 % | slope 2.26, rms 35.8 % | **slope 1.01, rms 9.1 %** |
| Ekman | slope 0.82, rms 12.1 % | slope 0.92, rms 17.7 % | slope 0.87, rms 6.9 % |
| both | slope 0.83, rms 14.2 % | slope 1.08, rms 50.9 % | slope 0.91, rms 12.7 % |

The harmonic combination helps both flows, and for Stokes it takes the slope to
1.01 — `L_K ∝ L_harm` with no curvature left — while nearly halving the scatter.
That is the result the weighted form was hoped to give.

Three cautions before leaning on it. `L_harm` is a monotone function of `L_N` and
`L_s`, so some improvement from adding a second scale is expected with only six
or seven points; the slope moving to 1 is stronger evidence than the rms falling.
Combining both flows still does not work (12.7 %), so `L_harm` does not unify them.
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
GKSwstype=100 julia --project=. plot_Lcomb_candidates_T10.jl   # seconds, reads the cache
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
unified nothing. `b = 1` is the one-parameter proportionality `L_K = A·L_comb`,
which is the form a closure would want; `b` free is the same fit with the
exponent released, as the honesty check.

| candidate | param | `A` in `L_K = A L_comb` | b=1 rms (both) | b free slope | **saturating on `L_comb`, both** |
|---|---|---|---|---|---|
| `L_N` alone | — | 0.202 | 79.8 % | 0.65 | 31.9 % |
| `L_s` alone | — | 0.109 | 62.4 % | 1.03 | 62.2 % |
| `min(L_N, L_s)` | — | 0.282 | 31.4 % | 0.82 | **10.0 %** |
| harmonic, equal | — | 0.377 | 23.1 % | 0.88 | **10.8 %** |
| harmonic, weighted | β = 1.42 | 0.433 | 21.2 % | 0.90 | 11.9 % |
| p-norm | p = 0.62 | 0.532 | 22.7 % | 0.89 | 13.4 % |
| geometric | α = 0.42 | 0.140 | 27.2 % | 0.96 | 25.3 % |

`A` is the geometric mean of `L_K/L_comb` and is what the black line in each
panel of `L_K_vs_Lcomb_candidates_T10.png` is drawn with; it is printed on the
panel next to the rms. It is **not** a quality measure — it says where a
candidate sits, not how well it collapses — and it is only comparable between
candidates that put `L_comb` on the same footing. It rises from 0.109 (`L_s`,
the largest scale, so the smallest ratio) to 0.532 (p-norm), simply because
each combination is a smaller length than the one before it.

*(16 cases, 2026-09-15. The 13-case version of this table had `L_N` alone at
14.9 % and the harmonic at 10.4 %.)*

**No candidate makes `L_K` proportional to `L_comb`.** The best `b = 1` fit is
21.2 %, and every combined scale still has a natural exponent near 0.9. The
curvature that the saturating form captures is real and no reweighting removes
it.

**Combining the scales unifies the two flows, and the low-`N` cases are what
show it.** `L_N` alone went from 14.9 % to **31.9 %** when `r = 0.2` and `0.5`
were added — those three points flatten off completely in the `L_N` panel,
because at low `N` the stratification scale is no longer what limits the mixing.
`min(L_N, L_s)` holds them at 10.0 % and the equal-weight harmonic at 10.8 %.
The case for a *combined* scale rather than a stratification one is much
stronger than it was on the r ≥ 1 sweep.

**Which combination, though, is now open.** On 13 cases the harmonic beat the
minimum, 10.4 % against 10.2 %, close enough to prefer the harmonic for being
smooth and parameter-free. On 16 the order reverses, 10.8 % against 10.0 %. The
two are inside each other's noise either way and nothing here separates them;
the harmonic is kept as the written form for the same reasons as before, but it
should no longer be called the winner.


### The harmonic law read nondimensionally

`1/L_K = 1/L_N + 1/L_s` is a statement about times as much as lengths. Putting
`L_K = K_T/√TKE`, `L_N = √TKE/N`, `L_s = √TKE/S` into it and clearing `√TKE`:

    TKE / K_T = (N + S) / A        i.e.       K_T* ≡ K_T N / TKE = A √Ri/(1 + √Ri)

since `√Ri = N/S` here, so `S/N = 1/√Ri`. `K_T*` is `τ_K N = τ_K/τ_N`, the
mixing time in buoyancy times — the left panel of `tau_K_vs_timescales_T10.png`
normalised. The limits are the two regimes: `Ri → ∞` gives `K_T → TKE/N`
(stratification-limited) and `Ri → 0` gives `K_T → TKE/S` (shear-limited).

**This is a change of variables, not a change of model.** Fitting
`K_T* = A√Ri/(1+√Ri)` has the same log residuals as fitting `L_K = A L_harm` —
measured 24.0 % against 23.1 %, the gap being only the per-sample median
convention. It cannot improve on the dimensional fit and does not.

What it makes visible is where the dimensions actually are. `A` was always
dimensionless; the dimensional constants are in the **saturating** form,
`L_∞ = 2.004 m` and `x₀ = 4.498 m`, and those have no nondimensional
expression. So the choice is 23–24 % with a clean law or 10.7 % with two
unexplained lengths — see the 2026-09-16 entry in `LOG.txt` for why a freer
`f(Ri)` does not close that gap (m free gives 23.3 %), why `f` is not universal
across the two flows (~21 % offset), and why the Ekman column cannot span `Ri`.

### Built: `K_T*` against `Ri`, Stokes only

```
cd Combined
GKSwstype=100 julia --project=. plot_KTstar_Ri_stokes_T10.jl
```

`figures/KTstar_vs_Ri_stokes_T10.png`. One panel: the pure harmonic form with
one free constant, `K_T* = A√Ri/(1+√Ri)`, at `z = h` on the background
`N = r·ω`.

`K_T* = 0.412 √Ri/(1+√Ri)`, **rms 14.7 %** on eight cases, `Ri` from 0.0043 to
263 (×61 000).

**The background version works.** 14.7 % on eight cases with one dimensionless
constant and no length scale anywhere, and both asymptotes are populated: the
low-`Ri` cases lie on `K_T* = A√Ri` (`K_T → TKE/S`, shear-limited) and the
high-`Ri` ones flatten onto `K_T* = A` (`K_T → TKE/N`, stratification-limited),
with the knee at `Ri ≈ 1` where it belongs. It is the same model as
`L_K = A L_harm`, so 14.7 % is the Stokes-only proportionality rms rather than
an improvement — the gain is interpretive, and it is a real one.

**A local gradient `Ri` was tried and dropped.** It cannot be taken at `z = h`:
`h` is *defined* as the height where `∂b/∂z` crosses `0.1 N²_ref`, so the
gradient there is identically `0.1 N²_bg` (measured, `0.100000`, min = max =
median over every sample of every case) and a "local" `Ri` is exactly
`0.1 × Ri_bg`. At a fixed height the gradient is free but no single height sits
at the same place in the flow for every case — median `h` runs 8.6 to 11.3 m —
so it fits worse: 26.3 % at the best height (`z = 10 m = T`), and 30–47 % over
9–18 m. Inside the layer the gradient falls under the mask and `K_T` is
discarded; above it `S` dies and `√Ri` runs away. The numbers are in `LOG.txt`.

The panel carries a note that `K_T* ∝ N` and `Ri ∝ N²` share `N`, so some
correlation is built in, exactly as `L_K` vs `L_harm` shares `√TKE`.

### Why this is Stokes only

`S` at `z = h`, fitted over the equilibrated range `r ≥ 1`:

| flow | `S ∝ N^p` | `S` spread over a ×250 change in `N` | so `Ri ∝ N^(2(1−p))` |
|---|---|---|---|
| Stokes | `p = 0.281` | ×3.3 | `Ri ∝ N^1.44` — spans 5 decades |
| Ekman | **`p = 1.018`** | ×58 | `Ri ∝ N^(−0.04)` — **pinned near 10** |

**The tidal layer has an externally imposed clock and the rotating layer does
not.** `h = 1.05 u_*/ω (1+N²/ω²)^(−0.020)` is almost stratification-blind: `ω`
sets the thickness, `N` gets no time to act, `S` stays put and `Ri` tracks `N`.
The steady rotating layer has no such clock, so it equilibrates — and
equilibration means adjusting until `Ri` at the layer top reaches a marginal
value. That is self-regulation, a result rather than a defect.

### The height scan — `plot_KTstar_Ri_zscan_T10.jl`

```
cd Combined
GKSwstype=100 julia --project=. plot_KTstar_Ri_zscan_T10.jl
```

`figures/KTstar_vs_Ri_zscan_T10.png`. One point per (case, height) rather than
one per case, everything local: `N = √(⟨∂b/∂z⟩)` at `z`, `S` at `z`, `K_T` and
`TKE` at `z`. Heights are fractions of each case's median `h` (0.6 → 3.0), so
the scan sits at the same place in the flow for every case; a point is kept only
where `∂b/∂z` clears the `0.05 N²_ref` mask in at least half its samples.

| set | `A` | rms | `n` | `Ri` range |
|---|---|---|---|---|
| both | 0.217 | 65.0 % | 95 | 5.5×10⁻⁴ → 7.0×10⁶ |
| Stokes | 0.331 | 50.3 % | 45 | 5.5×10⁻⁴ → 7.0×10⁶ |
| Ekman | 0.149 | 52.3 % | 50 | 8.8×10⁻³ → 4.5×10⁵ |

**The range problem is solved** — Ekman goes from `Ri` 0.44–12.8 at `z = h` to
eight decades, so an Ekman version of the figure does exist.

**But the plateau is not universal**: `A` = 0.331 for Stokes against 0.149 for
Ekman, a factor 2.2, with the Ekman points sitting below the Stokes ones
throughout. Whatever `K_T*` saturates at, it is not the same number in the two
flows.

**And the collapse is much looser**, 50–52 % per flow against 14.7 %. The scan
buys range by mixing two regions — the turbulent layer interior and the
quiescent fluid above it, where `S` has died and `Ri` runs to 10⁵–10⁶. Those are
not the same physics, and at the top end the points keep climbing rather than
flattening, so the plateau is not clean.

What stays out of reach either way is the **well-mixed interior**, where `Ri` is
genuinely small: there is no gradient there, so `K_T = −F_b/⟨∂b/∂z⟩` does not
exist, and no choice of height changes that.

## The one-figure summary

```
cd Combined
GKSwstype=100 julia --project=. plot_L_K_vs_Lharm_T10.jl   # seconds, reads the cache
```

`figures/L_K_vs_Lharm_T10.png` — the preferred candidate only, and the regime
question, on one page.

**Left**: `L_K` against `L_harm = 1/(1/L_N + 1/L_s)` for all sixteen cases, with
the saturating curve through both flows, `L_K = 2.004(1 − e^(−L_harm/4.498))`,
rms 10.7 % (Stokes alone 11.7 %, Ekman alone 8.4 %). The pure proportionality
`L_K = 0.377 L_harm` (rms 23.1 %) is dotted alongside so the residual curvature
is visible rather than asserted. On 13 cases this was
`2.18(1 − e^(−L_harm/4.89))` at 10.4 % — the constants moved by about 8 % and
the quality of the collapse did not.

**Right**: written as `1/L_harm = 1/L_N + 1/L_s` the two terms are resistances in
series and their shares add to one,

    w_N = L_harm/L_N       w_s = L_harm/L_s       w_N + w_s = 1

so `w_s` is the fraction of `1/L_harm` the shear contributes. `w_s > 1/2` is
shear-limited, `w_s < 1/2` stratification-limited, `w_s = 1/2` is `L_s = L_N`.
Only `w_s` is drawn — `w_N` is `1 − w_s`. It is also what colours the markers on
the left, so a point's colour there says which regime it came from. The weight
is arithmetic on `L_N` and `L_s`, not a fitted quantity.

| flow | r | 0.5 | 1 | 2 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|---|---|
| Stokes | `w_s` | — | **0.59** | 0.40 | 0.23 | 0.18 | 0.15 | 0.06 |
| Ekman | `w_s` | 0.44 | 0.26 | 0.22 | 0.23 | 0.22 | 0.24 | 0.26 |

Exactly one case in the set is shear-limited, Stokes `r = 1`. The Stokes
sequence marches monotonically into the stratification-limited corner as `N`
rises, 0.59 → 0.06; the Ekman sequence does not march anywhere, sitting at
0.22–0.26 for `r ≥ 1` and only creeping to 0.44 at the weakest stratification.
So the shear-limited regime is approached from the Stokes side at low `N` and
never reached on the Ekman side. The shear still earns its place in `L_harm` at
`w_s ≈ 0.25` — that is where the collapse of the two flows onto one curve comes
from.

## `HANDOVER.md`

The prompt to paste when starting a new Claude session on this work: the
standing rules, where things stand, the next step, and the gotchas. It points at
`LOG.txt` and `README.md` rather than repeating them, and should be updated
whenever the next step changes.

## `LOG.txt`

A short chronological record of what this folder has been for, in the comment
style of `Ekman 3D.jl` — one entry per episode, giving what was done and what
came of it, plus the mistakes made and what is still open. **The detail and the
reasoning live in this README; `LOG.txt` is the narrative.** Keep it that way
when appending: newest entries at the bottom, and append when something is
learned, not when something is run.


## l against the Corrsin shear length scale

```
cd Combined
GKSwstype=100 julia --project=. reduce_ekman_moments_T10.jl   # ~2 min, now also does ε
GKSwstype=100 julia --project=. plot_l_vs_corrsin_T10.jl      # ~4 min first time, then cached
```

Same figure as `l` against `√TKE/N`, same two styles, with the abscissa changed
to the Corrsin scale

    L_C = (ε/S³)^(1/2)

the scale at which the eddy turnover rate matches the mean shear rate. It is
the shear analogue of the Ozmidov scale, and a different question from
`L_s = √TKE/S` — `L_s` is built from the energy, `L_C` from its flux.

| file | what it shows |
|---|---|
| `figures/l_vs_corrsin_ath_T10_combined.png` | every retained sample, medians on top |
| `figures/l_vs_corrsin_ath_T10_combined_errorbars.png` | medians with the interquartile range |

### ε is not stored, and had to be estimated

Neither pipeline wrote the dissipation rate, and like `F_sgs` it cannot be
rebuilt from stored averages. What the moments do carry is every other term of
the TKE budget, so ε comes from local equilibrium:

    ε ≈ P + B      P = −⟨u′w′⟩ ∂U/∂z − ⟨v′w′⟩ ∂V/∂z + νₑ S²
                   B = F_b = ⟨w′b′⟩ + F_sgs        (negative when stable)

`reduce_ekman_moments_T10.jl` now writes `P_at_h` and `eps_at_h` alongside
`S_at_h`; the Stokes side is built the same way inside the plot script and
cached in `Data/corrsin_T10.jld2`. Three assumptions ride on the abscissa that
do not ride on the ordinate: transport is dropped and cannot be checked (the
worst of them, and it is exactly what dominates at the top of a mixed layer);
storage is dropped, and is checked below; and `νₑ ≈ κₑ`, since only the
buoyancy diffusivity was written.

### Where it works, and where it does not

**Local equilibrium fails at `z = h` itself.** The fraction of samples with
`ε > 0`:

| flow, r | 0.25h | 0.50h | 0.75h | 1.00h |
|---|---|---|---|---|
| Stokes 1 | 0.96 | 0.98 | 0.94 | 0.90 |
| Stokes 5 | 0.99 | 0.96 | 0.97 | 0.60 |
| Stokes 25 | 1.00 | 0.96 | 0.87 | 0.19 |
| Stokes 50 | 0.99 | 0.97 | 0.83 | **0.03** |
| Ekman, all r | 1.00 | 1.00 | 1.00 | 0.73–1.00 |

Everywhere inside the layer the estimate is fine. It collapses only on the top
face, and only for the strongly stratified Stokes cases, where `P ≈ 0` — the
mean shear has nothing left to do at `z = h` — and the budget there is
transport against buoyancy destruction, which is precisely the balance local
equilibrium throws away. This is a statement about the height, not about the
method.

So three cases are unusable at `z = h`: Stokes `r` = 10, 25, 50, with `ε > 0`
in 44 %, 19 % and 3 % of samples. They are drawn hollow and kept out of every
fit — their surviving samples are selected on the sign of a budget residual,
which is the kind of selection that manufactures a trend.

### What the usable cases show

13 of the 16 cases are usable — Stokes `r` = 10, 25 and 50 have a median `ε`
that is negative, so `L_C = (ε/S³)^(1/2)` does not exist there. **The two new
low-`N` Stokes cases have the best-conditioned `ε` in that column**,
`ε/(P+|B|)` = 1.00 at `r = 0.2` and 0.99 at `r = 0.5` against 0.88 at `r = 1`
and 0.21 at `r = 5`: the residual estimate degrades with stratification, not
with the lack of it.

| flow | r | ε (m²/s³) | S (1/s) | L_C (m) | l (m) | ε/(P+\|B\|) |
|---|---|---|---|---|---|---|
| Stokes | 0.2 | 1.40e−11 | 3.02e−04 | 0.743 | 0.481 | 1.00 |
| Stokes | 0.5 | 6.50e−12 | 2.27e−04 | 0.645 | 0.513 | 0.99 |
| Stokes | 1 | 1.66e−12 | 1.47e−04 | 0.790 | 0.473 | 0.88 |
| Stokes | 2 | 1.01e−12 | 1.30e−04 | 0.676 | 0.432 | 0.61 |
| Stokes | 5 | 4.30e−13 | 1.51e−04 | 0.509 | 0.189 | 0.21 |
| Ekman | 0.5 | 1.07e−12 | 3.94e−05 | 4.132 | 1.483 | 0.73 |
| Ekman | 1 | 4.28e−13 | 3.44e−05 | 3.128 | 1.002 | 0.33 |
| Ekman | 2 | 1.09e−12 | 5.59e−05 | 2.729 | 0.752 | 0.23 |
| Ekman | 5 | 7.07e−12 | 1.51e−04 | 1.444 | 0.426 | 0.35 |
| Ekman | 10 | 1.56e−11 | 2.89e−04 | 0.840 | 0.239 | 0.28 |
| Ekman | 25 | 5.02e−11 | 8.00e−04 | 0.310 | 0.109 | 0.28 |
| Ekman | 50 | 1.10e−10 | 1.76e−03 | 0.144 | 0.064 | 0.29 |

`ε/(P+|B|)` sits near 0.3, so ε is a healthy fraction of the budget rather than
the small difference of two large terms — the failure mode that would have made
this hopeless.

| fit | form | rms |
|---|---|---|
| Ekman, 7 cases | `l = 0.331 L_C^0.92` | 12.4 % |
| Stokes, 3 cases | `l = 0.874 L_C^2.19` | 10.9 % |
| Overall, 10 cases | `l = 0.375 L_C^0.89` | 25.7 % |

The saturating form pins on every set that has enough cases to try it: over
this range `l` against `L_C` has no knee, so a power law is what gets drawn.

**The Ekman column is close to `l ∝ L_C`** — exponent 0.92, and `l ≈ L_C/3`
across a factor of thirty in `L_C`. That is a cleaner statement than anything
`√TKE/N` gave on that side. **The two flows still do not share it**: the Stokes
exponent is 2.19, and one line through both leaves 25.7 %. But the Stokes fit
rests on three points spanning less than a factor of two in `L_C`, over exactly
the part of the sweep where the `ε` estimate is starting to degrade (`ε/(P+|B|)`
falling 0.88 → 0.21), so it is the weakest number in the table and should not be
read as a contradiction of the Ekman result.


## The δ model: does a boundary-layer thickness explain the mixing?

```
cd Combined
GKSwstype=100 julia --project=. reduce_profiles_T10.jl     # ~6 min, walks both columns
GKSwstype=100 julia --project=. plot_delta_T10.jl          # stage 1
GKSwstype=100 julia --project=. plot_tke_profiles_T10.jl   # stage 2
GKSwstype=100 julia --project=. plot_KT_model_T10.jl       # stage 3
```

Testing

    TKE(z) = A₁ exp(−A₂ z/δ)
    K_T    = B₁ δ √TKE (B₂ + exp(B₃ L_harm/δ))

Divided by `√TKE` the second is the saturating form already fitted, with one new
claim: **that the plateau and the knee both scale with δ** instead of being
fitted per flow. (As written the bracket only gives a saturating curve with
`B₁ < 0, B₂ = −1, B₃ < 0`; everything below uses the equivalent
`L_K/δ = C₁(1 − e^(−C₂ L_harm/δ))`.)

### u_* is measured, not assumed

`reduce_profiles_T10.jl` takes `u_*² = max over 0 < z ≤ h of |τ|` with
`τ = (⟨uw⟩ − ⟨u⟩⟨w⟩) − νₑ ∂U/∂z` and `νₑ ≈ κₑ`, per sample then reduced.

| | measured `u_*` | `√c_D·U∞` | `√(c_D·|U₁|²)` |
|---|---|---|---|
| Stokes | 1.11e−3 | 4.40e−3 | 1.00e−3 |
| Ekman | 2.63e−3 | 3.81e−3 | 1.94e−3 |

`u_*` is flat to under 3 % across each sweep — stratification does not change
the bed stress. The free-stream estimate is **4× too large** for Stokes, so
this was worth measuring rather than assuming.

### Stage 1 — which δ tracks h

`figures/delta_vs_h_T10.png`. The leading constant is not what is being
tested — a later fit absorbs it — so a candidate is good if `h/δ` is **flat**.
Ratio of max to min across each sweep:

| δ | Stokes | Ekman |
|---|---|---|
| plain, no stratification | ×1.32 | ×3.52 |
| Weatherly & Martin, `(1+N²/Ω²)^(−1/4)` | ×6.72 | ×2.09 |
| **fitted exponent p** | **×1.28 (p = 0.020)** | **×1.13 (p = 0.155)** |
| `u_*/√(ΩN)` | ×15.16 | ×4.49 |

**This stage works, but less well on the tidal side than the six-case sweep
suggested.** The same functional form fits both flows:

    h = 1.05 u_*/ω (1 + N²/ω²)^(−0.020)     flat to 7.3 %   (tidal)
    h = 0.85 u_*/f (1 + N²/f²)^(−0.155)     flat to 4.2 %   (rotating)

**The tidal numbers moved and the rotating ones did not.** Adding `r = 0.2` and
`0.5` took the Stokes exponent from 0.040 to 0.020 and the flatness from 1.6 %
to 7.3 %; `h` runs from 8.97 to 11.30 m across the eight cases where it ran from
8.60 to 11.30 m across six. The Ekman law absorbed its new point without
moving — p = 0.155 unchanged, 4.4 % to 4.2 % — which is the stronger result of
the two now. **"Flat to 1.6 %" was a six-case number and should not be quoted.**

Three things worth noting. The tidal layer is **almost stratification-blind**
(p = 0.02, even more so than before): its thickness is set by ω, and `N` gets no
time to act. WM's `p = 1/4` is **too steep** for the rotating column here —
though the five low-`N/f` cases have not equilibrated and their `h` is still
rising, so the fitted 0.155 is a lower bound. And the two prefactors, 1.05 and
0.85, agree to 24 %, which is closer than the published 0.4 and 1.3 would
suggest.

**Not a pycnocline artefact — that was checked.** `h` sitting near `z = T = 10 m`
looks like it might be geometry rather than physics, so `reduce_profiles_T10.jl`
reports `h_pin`, the fraction of samples with `h` inside ±5 % of `T`. It is
24 % at `r = 0.2` and 26 % at `r = 0.5` against **45 % at `r = 5` and 94 % at
`r = 10`**. The whole Stokes column sits near the pycnocline, so there is no
basis for treating the low-`N` cases differently and all eight are fitted. Ekman
is nowhere near it: `h` is 6.2 to 22.6 m and `h_pin` is 0 % at every `r`.

### Stage 2 — is TKE exponential, and does it collapse

`figures/tke_profiles_T10.png`, plotted log-`TKE` against linear `z/h` so an
exponential is a straight line. Fitted from the TKE peak to `z = h`; fitting
from the wall would fold the near-wall rise into `A₂`.

| | `A₂` | `A₂ h/δ` (decay over one `h`) | `A₁/u_*²` | fit rms |
|---|---|---|---|---|
| Stokes | 1.38 – 1.52 (×1.10) | 2.94 – 3.77 (×1.28) | 3.29 – 3.55 (×1.08) | 13–19 % |
| Ekman | 4.01 – 6.88 (×1.72) | 2.60 – 4.53 (×1.75) | 1.73 – 2.23 (×1.29) | 2–13 % |

**Half works, and the extra cases did not change which half.** The eight Stokes
profiles collapse onto a single exponential with no stratification dependence at
all — `A₁/u_*²` spans only ×1.08 over the whole sweep, `A₂` only ×1.10. The
Ekman profiles **do not collapse**: they fan out by ×1.72 in decay rate, and
they flatten to a floor of 1–3 % of `u_*²` above `z ≈ h`. So `h/δ` being flat
(stage 1) does **not** imply the profiles collapse on δ.

The one thing that loosened is `A₂h/δ` on the Stokes side, ×1.11 → ×1.28, which
is the stage-1 `h` spread showing through — `δ` is unchanged but `h` now runs
further.

`A₂` is quoted per δ as the model defines it, but the published 0.4 and 1.3 put
`h/δ` at 2.62 and 0.66, so `A₂` alone is not comparable between flows; `A₂h/δ`
is, and on that measure both flows decay by about `e^(−3.4)` over one `h`.

### Stage 3 — the δ model for K_T

`figures/KT_delta_model_T10.png`. Does `L_K/δ` against `L_harm/δ` collapse both
flows better than the unscaled fit's 10.7 %?

| δ | both | Stokes | Ekman |
|---|---|---|---|
| published (0.4, 1.3) | 21.4 % | 11.9 % | 7.4 % |
| one common constant | 11.8 % | 11.9 % | 7.4 % |
| `δ = h`, the most favourable | 10.8 % | 11.0 % | 7.8 % |
| **unscaled, no δ at all** | **10.7 %** | 11.7 % | 8.4 % |

*(16 cases. Both scripts now compute the unscaled reference rather than quoting
it from each other — it used to be written in as the literal 10.4 %.)*

**This stage does not work.** No thickness improves the two-flow collapse, and
the published constants make it much worse by putting the two δ a factor of
five apart. `δ = h` is the best case available and is still no better than not
scaling. Within the Ekman column alone δ helps slightly, 8.9 % → 7.9 %, which
is a hint that its knee moves with thickness but is well inside the noise on
eight points.

Two caveats on the verdict. The Stokes column **cannot test this**: its largest
`L_harm` is `0.23 x₀`, so `exp(−L_harm/x₀) ≥ 0.80` throughout and it never
reaches the bend — its 9.1 % is the same straight line however δ is chosen.
And the one thing that does survive is the **initial slope**, 0.451 (Stokes)
against 0.454 (Ekman) — but `C₁C₂ = dL_K/dL_harm` contains no δ, so that
agreement is evidence for `L_harm`, not for the thickness.

**Where this leaves the model.** The `h` law of stage 1 is a genuine result and
worth keeping, with the tidal flatness read as 7.3 % rather than 1.6 %. The
`K_T` law is better written without δ, as it already was:
`L_K = 2.004(1 − e^(−L_harm/4.498))`, rms 10.7 %. The margin narrowed — `δ = h`
is now 10.8 % against 10.7 % rather than 11.1 % against 10.4 % — but the verdict
is the same, and it is the same verdict for the same reason: nothing is bought.


## The same study with the Corrsin scale — `plot_corrsin_scales_T10.jl`

```
cd Combined
GKSwstype=100 julia --project=. plot_corrsin_scales_T10.jl   # reads both caches
```

The `L_N`/`L_s` study redone with `L_C = (ε/S³)^(1/2)` in place of
`L_s = √TKE/S`, to ask whether the Corrsin scale is the better shear-side
partner for `L_N`. Three figures, mirroring the `L_s` ones:

| figure | the `L_s` original |
|---|---|
| `L_N_L_C_vs_r_T10.png` | `L_N_L_s_vs_r_T10.png` |
| `L_K_vs_Lcomb_corrsin_T10.png` | `L_K_vs_Lcomb_candidates_T10.png` |
| `L_K_vs_Lbest_corrsin_T10.png` | `L_K_vs_Lharm_T10.png` |

Everything comes from the two existing caches — `Data/corrsin_T10.jld2` for the
Stokes per-sample `S`, `ε` and `Data/ekman_lengthscales_T10_moments.jld2` for
the Ekman ones — so `L_K`, `L_N`, `L_s` and `L_C` are all built from the same
`K_T`, `TKE`, `S`, `ε` on one sample basis. A case is used only if `ε > 0` in at
least half its samples, the same rule as `plot_l_vs_corrsin_T10.jl`, which
leaves **13 of 16**: Stokes `r` = 10, 25 and 50 are out.

### The answer: `L_C` does not give the two-regime structure

The idea being tested is that `L_N` limits the mixing at strong stratification
and a shear scale limits it at weak. With `L_s` that is exactly what the data
show — `L_s/L_N = √Ri` crosses 1 cleanly and monotonically, and keeps going, to
×15.6 by Stokes `r = 50`. With `L_C` it does not happen:

| flow | `L_C/L_N` across the reliable sweep | `L_s/L_N` |
|---|---|---|
| Stokes (`r` ≤ 5) | 0.031 → 0.997, never above 1 | 0.068 → 3.5, crosses at `r ≈ 1.3` |
| Ekman (all) | 0.216 → 0.997, peak 0.997, back to 0.840 | 0.663 → 3.6, crosses at `r ≈ 0.35` |

**`L_C` is the smaller of the two scales in every reliable case.** Since
`w_C = L_harm/L_C = 1/(1 + L_C/L_N)`, that puts the Corrsin share at or above ½
everywhere — there is no `r` at which `L_N` takes over. The two Stokes points
that do exceed 1 (`r` = 10 at 1.13 and `r` = 50 at 1.28) are ε-unusable cases,
and they are not monotonic (`r` = 25 falls back to 0.55), so they are the ε
estimate failing, not a crossing.

**And the Stokes column cannot test it anyway.** Over the range where ε is
usable (`r` ≤ 5), the scales move by:

| | `L_N` | `L_s` | `L_C` | `L_K` |
|---|---|---|---|---|
| Stokes, `r` ≤ 5 | ×46.0 | ×1.31 | ×1.52 | ×2.9 |
| Stokes, all 8 | ×535 | ×3.27 | ×13.8 | ×28.0 |
| Ekman, all 8 | ×121 | ×28.1 | ×31.0 | ×26.8 |

`L_C` changes by half as much as `L_K` does across the usable Stokes range, and
the ε cut removed exactly the three cases where it would have varied. That is
the cluster of Stokes circles sitting off the line in the `L_C alone` panel.

### What the candidate table says

| candidate | `A` | b=1 rms both | power `b` | **saturating, both** | Stokes | Ekman |
|---|---|---|---|---|---|---|
| `L_N` alone | 0.175 | 82.0 % | 0.54 | 35.2 % | 6.8 % | 7.1 % |
| `L_C` alone | 0.418 | 33.5 % | 0.88 | 31.0 % | 27.1 % | 14.9 % |
| `min(L_N, L_C)` | 0.422 | 32.6 % | 0.89 | 29.9 % | 25.4 % | 13.8 % |
| harmonic, equal | 0.643 | 20.3 % | 0.87 | 13.8 % | 12.6 % | 9.5 % |
| harmonic, weighted (β = 0.88) | 0.591 | 20.0 % | 0.87 | **12.8 %** | 12.4 % | 9.3 % |
| p-norm (p = 0.82) | 0.729 | 19.9 % | 0.87 | **12.8 %** | 13.4 % | 9.4 % |
| geometric (α = 0.25) | 0.331 | 20.9 % | 0.88 | 16.9 % | 18.7 % | 11.0 % |

`A` here is a third of the way to being a result on its own: `L_C` alone gives
`A` = 0.418 against 0.109 for `L_s` alone, so `L_K` is about 0.4 `L_C` but only
0.1 `L_s`. The Corrsin scale sits much closer to the mixing length than the
shear scale does, which is the other half of why it looks better one-on-one.

Two things to read off it.

**`L_C` alone really is a better shear-side scale than `L_s` alone** — 33.5 %
against 62.4 % on the proportionality, and its natural exponent is 0.88 rather
than 1.03. `L_C` carries information `L_s` does not.

**But combining with `L_C` is worse than combining with `L_s`, on fewer cases**:
12.8 % on 13 cases against 10.0 % (`min`) and 10.8 % (harmonic) on 16. And
`min(L_N, L_C)` at 29.9 % is barely different from `L_C` alone at 31.0 %, which
is the same statement as above — the minimum is `L_C` almost everywhere, so
taking it adds nothing.

**The combination does still beat `L_N` alone** (35.2 % → 12.8 %), so `L_C` is
not useless. But the mechanism is the reverse of the one being tested: `L_C` is
the smaller scale everywhere yet nearly flat across the Stokes column, so it is
`L_N` that supplies the case-to-case variation there. That is not "stratification
limits at high `N`, shear limits at low `N`" — it is two scales that happen to
combine well for a different reason.

**Where this leaves the approach.** The two-regime idea is well supported, but
by `L_s`, not `L_C`: `L_s/L_N = √Ri` is the quantity with a clean crossing, a
physical interpretation, and the better collapse on the full 16 cases. `L_C` is
worth keeping as the independent check it already was in the Corrsin section —
it is not the shear scale to build the model on.


## Re-running the weakly stratified end — `swirles.sh`

`L_s/L_N = N/S = √Ri`, so the `L_s = L_N` crossing on
`figures/L_N_L_s_vs_r_T10.png` is `Ri = 1`. The medians put it at `r ≈ 1.4`
(Stokes) and, extrapolated below the sweep, `r ≈ 0.4` (Ekman) — neither
resolved. `swirles.sh` re-runs both columns at low `r` to fix that.

**Resolved, 2026-09-15.** With the new cases in, both crossings are bracketed by
data rather than extrapolated, and both original estimates stand:

| flow | crossing sits between | `w_s` either side |
|---|---|---|
| Stokes | `r = 1` and `r = 2` | 0.591 → 0.395 |
| Ekman | `r = 0.2` and `r = 0.5` | 0.602 → 0.441 |

Four of the sixteen cases are now shear-limited (`w_s > ½`): Stokes `r` = 0.2,
0.5, 1 and Ekman `r` = 0.2. It was one in thirteen.

```
cd /cephfs/store/damtp/tll46/SRIM-2026
MODE=preflight bash Combined/swirles.sh     # login node, no GPU work
sbatch Combined/swirles.sh                  # ~5.6 h serially
sbatch --array=0-2 Combined/swirles.sh      # or one case per task, ~2 h
MODE=status bash Combined/swirles.sh        # before pulling anything home
```

| flow | default `r` | why |
|---|---|---|
| Stokes | 0.5, 0.2 | `r = 0.5` exists in `outputs/` but **predates the moments pipeline** — no `*_moments.jld2`, no `mixing_*`, no drag marker — so it is re-run, not reused |
| Ekman | 0.2 | `r = 0.5` already has a complete moments run; `0.2` crosses to the far side of `Ri = 1` |

Both flows take the same `r` so `N` matches case for case, which is what the
comparison rests on.

### One folder, for scp

```
Combined/Data/lowN/
  stokes/P4_T10_sqrtRi0p5/    TidalBL3D_*_moments.jld2, mixing_*_hcross.jld2
  stokes/P4_T10_sqrtRi0p2/
  ekman/r=0.2, T=10.0/        Moments.jld2, Avg_*.jld2
  logs/
```

```
scp -r <host>:/cephfs/store/damtp/tll46/SRIM-2026/Combined/Data/lowN \
       ~/SRIM-2026/Combined/Data/
```

~900 MB. The analysis only ever reads two files a case, so an `rsync` filtered
to `*_moments.jld2`, `mixing_*_hcross.jld2` and `Moments.jld2` brings it to
~60 MB a Stokes case — the command is in the script header.

Nothing under `Ekman/`, `Stokes/3D/outputs/` or `Data/Ekman_moments/` is written
to; the Stokes drag spin-up under `outputs/` is read, never modified.

### Two things to know before running it

**The Ekman driver cannot take an `r` with two decimals.** `case_dir()` in
`ekmanrun.jl` formats it as `%.1f`, so `r = 0.25` would silently write into the
folder `r=0.2` and collide with a genuine `r = 0.2`. `swirles.sh` refuses such
an `r` with a message rather than letting it happen. Fixing it properly means
changing `case_dir()` **and** the matching readers in
`reduce_ekman_moments_T10.jl` and the `r=%.1f` group keys in the plot scripts.

**`K_T` is badly conditioned at low `N`, and that is why `r = 0.5` was dropped
from the Stokes column in the first place** (`run_moments_sweep.sh` says so).
`Δb` is small there and `K_T = −F_b/⟨∂b/∂z⟩` divides by it. The acceptance test
is the `VERIFICATION` block that `swirles.sh` echoes after each Stokes case:
`K_T_bulk` and `K_T_pe` share no code, so if they disagree by more than ~30 %
the diffusivity at that `r` is not measuring the flow. Check that before adding
these points to any figure.

### What came back                                          (run 2026-09-10/11)

All three cases ran to their full stop time and passed. `logs/swirles.log` has
the driver's own account; the numbers below are from the per-case logs.

| case | wall clock | span reached | `K_T_bulk` vs `K_T_pe` | `K_sgs/K_T` at `h` | `δ_eff` |
|---|---|---|---|---|---|
| Stokes `r = 0.5` | 2.13 h | 8.00 periods (5.027e5 s) | rms 0.1 %, bias −0.0 % | 0.03 | 6.66 ± 2.88 m (CV 0.43) |
| Stokes `r = 0.2` | 2.15 h | 8.00 periods (5.027e5 s) | rms 0.1 %, bias −0.0 % | 0.04 | 8.50 ± 1.66 m (CV 0.19) |
| Ekman `r = 0.2` | 2.06 h | 25.46 `T_f` (1.5998e6 s) | — (no Ekman post-step yet) | — | — |

**The conditioning check passes, decisively.** `K_T_bulk` and `K_T_pe` agree to
0.1 % rms at both `r = 0.5` and `r = 0.2` — they share no code, so this is the
strong form of the test, not the ~30 % acceptance threshold. Panel (c) of
`Data/lowN/figures/K_T_P4_T10_sqrtRi0p*.png` shows the two curves lying on top
of each other for the whole record. **The original reason for dropping Stokes
`r = 0.5` does not apply to these runs.** The `GRAD_FLOOR = 0.05` mask is doing
more work than at high `r` — panel (b) is visibly speckled above `z ≈ 10` m —
but that is the mask working, not `K_T` failing.

Both columns are clean: `max|⟨w⟩_xy| ~ 1e-19 U₀` throughout, no NaNs, and every
field finite in the last snapshot of all three files. The only log warnings are
CUDA loading `libcusparse`/`libnvJitLink` from a system path, which is the
cluster's module setup and is present in every run in this folder.

**One caveat, and one that turned out not to be one.**

~~`h` reaches the pycnocline, so these points must not go into the `h` fit.~~
**Withdrawn, 2026-09-15.** `h(t)` does climb to `z = T = 10 m` and sit there in
both cases, and that reads as geometry limiting a layer that should be limited
by `u_*` and `ω`. But it is not special to low `N`: measured as `h_pin`, the
fraction of samples with `h` inside ±5 % of `T`, it is 24 % at `r = 0.2` and
26 % at `r = 0.5` against **45 % at `r = 5` and 94 % at `r = 10`**. `h` is 8.6
to 11.3 m across the whole Stokes sweep while `T` = 10 m, so every case sits on
the pycnocline and the low-`N` ones sit on it *less* than most. All eight are
fitted. See the stage 1 section for what that did to the numbers.

`δ_eff` is noisy at `r = 0.5`, CV 0.43 against 0.19 at `r = 0.2`, and goes
negative for a few tidal phases (`ωt ≈ 18, 26, 32, 41`) where `F_b` changes
sign. `δ_eff = ∫F_b dz / F_b|peak` is undefined through a sign change. This does
not touch `K_T`, and the δ model was abandoned at stage 3 anyway, but any
`δ_eff` median at `r = 0.5` must be formed with those phases dropped.

The panel (d) slope is +0.40 (`r = 0.5`) and +0.41 (`r = 0.2`), against ½ for
`K_T ~ √TKE·l` and 1 for `K_T ~ TKE/N`. Both sit nearer ½, same as the rest of
the sweep — the weakly stratified end has not switched regime by this diagnostic.

### Wiring it into the figures — `sweep.jl`                  (done 2026-09-15)

The three new cases are not beside the originals: they were run into
`Data/lowN` as one folder for `scp`. Six scripts had the case root, the
`sqrtRi0p5` tag spelling and the `r`-colour ramp open-coded, so all three moved
into **`sweep.jl`**, which every reduction and plot script now includes:

| from `sweep.jl` | what it gives |
|---|---|
| `SVALS`, `RATIOS` | the sweep — `0.2 0.5 1 2 5 10 25 50` in both columns |
| `stokes_case(s)` | searches `Data/lowN/stokes` first, then `Stokes/3D/outputs` |
| `ekman_case(r)` | searches `Data/lowN/ekman` first, then `Data/Ekman_moments/4` |
| `sqrtRi_tag(s)` | `"P4_T10_sqrtRi0p5"` — the decimal point is written `p` |
| `RAMP`, `ramp_colour` | two anchors added below `r = 1`; `r ≥ 1` unchanged |

`Data/lowN` is searched **first**, and has to be:
`Stokes/3D/outputs/P4_T10_sqrtRi0p5` exists but predates the moments pipeline
and holds no `*_moments.jld2` and no `mixing_*`, which is why `r = 0.5` was
dropped from the original sweep in the first place.

Adding a case is now one edit to `SVALS`/`RATIOS` rather than eight.

To rebuild everything from the raw data:

```
cd Combined
GKSwstype=100 julia --project=. reduce_ekman_moments_T10.jl   # ~4 min, 8 cases
GKSwstype=100 julia --project=. reduce_profiles_T10.jl        # then the plots
GKSwstype=100 julia --project=. plot_shear_scales_T10.jl      # writes the cache
GKSwstype=100 julia --project=. plot_delta_T10.jl
GKSwstype=100 julia --project=. plot_tke_profiles_T10.jl
GKSwstype=100 julia --project=. plot_L_K_vs_Lharm_T10.jl
GKSwstype=100 julia --project=. plot_Lcomb_candidates_T10.jl
GKSwstype=100 julia --project=. plot_KT_model_T10.jl
REBUILD=1 GKSwstype=100 julia --project=. plot_l_vs_corrsin_T10.jl
GKSwstype=100 julia --project=. plot_l_vs_qN_T10_combined.jl
```

**`REBUILD=1` is not optional on the Corrsin script.** It caches the Stokes
walk in `Data/corrsin_T10.jld2` and reuses it silently, so without the flag it
would have rebuilt the figure from the old six-case Stokes set without saying
so.
