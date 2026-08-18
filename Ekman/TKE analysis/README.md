# Ekman TKE / K_T analysis

One stratified Ekman case, instrumented with the second-moment output and the
turbulent-diffusivity analysis built for the tidal (Stokes) study, so the two can
be compared directly.

The question both answer: does the effective turbulent diffusivity scale as
`K_T ~ √TKE · l` (log–log slope ½) or `K_T ~ TKE / N` (slope 1)?

## The case

| | |
|---|---|
| `r = N/f` | **1** |
| background | softplus (`profile = 4`), `sharp = 6` |
| pycnocline height | `T = 20 m` |
| grid | 100 × 100 × 500 over 75 × 75 × 120 m (100 m physical + 20 m sponge) |
| `max_Δt` | 7.5 s |
| duration | 4 × 10⁵ s = **6.37 inertial periods** |
| bottom | quadratic drag, `cᴰ = (0.41/log(z₁/z₀))² ≈ 0.0121` |

**Every physical parameter, the grid, the duration and the timestep are exactly
those of `Ekman/3D Simulation/Parameters.jl` and `Ekman 3D.jl`.** Nothing in this
folder changes the physics of the original.

## Relationship to `Ekman/3D Simulation/`

Nothing here writes into that folder, and nothing there was edited. Files were
copied in, and fall into three groups:

| file | status |
|---|---|
| `Ekman_plot.jl`, `Ekman_anim.jl`, `TKE.jl` | **verbatim copies** |
| `Parameters.jl` | copy; `r`, `T`, `profile` pinned to this case, `sharp` made `const`, inertial period added |
| `Filename_plot.jl`, `Filename_anim.jl` | copy; paths anchored to this folder instead of `Ekman/3D Simulation/` and `Ekman/Data/` |
| `Ekman3D.jl` | copy of `Ekman 3D.jl`; moments hook, paths, opt-in figures |
| `Moments.jl`, `MixedLayerDiffusivity.jl`, `measure_h0.jl` | copies of the Stokes versions, names bridged |
| `Figures.jl`, `swirlesrun.jl`, `swirles.sh`, this README | new |

Data goes in `Data/`, figures in `Plots/` and `Animations/`, job logs in `logs/`
— all inside this folder.

## How to run

On the cluster:

```bash
sbatch swirles.sh                          # simulation → K_T analysis → figures
EKMAN_STAGE=post    sbatch swirles.sh      # re-analyse data already on disk
EKMAN_STAGE=figures sbatch swirles.sh      # redraw figures only
```

Interactively, from the repo root:

```bash
julia --project=. "Ekman/TKE analysis/Ekman3D.jl"                 # the simulation
julia --project=. "Ekman/TKE analysis/measure_h0.jl"              # did turbulence reach z = T?
julia --project=. "Ekman/TKE analysis/MixedLayerDiffusivity.jl"   # K_T + the checks
julia --project=. "Ekman/TKE analysis/Figures.jl"                 # the existing Ekman figures
```

Smoke test before committing an allocation:

```bash
EKMAN_SMOKE=1 SMOKE_ITERS=20 julia --project=. "Ekman/TKE analysis/Ekman3D.jl"
```

## Cost

Roughly **1 hour** against the 12 h wall, and it is worth seeing why this case is
so much cheaper than a tidal one. Scaling from the Stokes study's measured
17.5 ms/iteration at 100 × 100 × 300 gives ~29 ms here for 5.0 M cells — but the
Stokes runs needed 434,800 iterations because a 0.0086 m wall cell held the CFL
timestep near 3 s, whereas this grid's first cell is 0.133 m and `max_Δt = 7.5 s`
binds instead:

```
4e5 s / 7.5 s = 53,300 iterations × 29 ms ≈ 0.43 h,  call it ~1 h with output
```

That is an estimate transplanted from a different case. The job prints its own
wall time; the first submission is what turns it into a measurement.

## Reading the result

`measure_h0.jl` runs **first**, and answers the question that decides whether the
run measured anything: did the turbulent layer reach the pycnocline at `z = 20 m`?
If `h0 ≪ 20 m`, the buoyancy flux there is negligible, `Δb` never changes, and
`K_T` is being formed from noise — nothing below is meaningful.

Then the VERIFICATION block, in this order — an earlier failure invalidates
everything after it:

1. **`⟨w⟩_xy ≈ 1e-18`.** Exact by incompressibility with an impermeable bottom.
   This is what makes `⟨w′b′⟩ = ⟨wb⟩` with no mean subtraction. If it fails, stop.
2. **`K_T_bulk` vs `K_T_pe` (panel c).** One route uses `wb` and `F_sgs`, the
   other only `B`; they share no code path. The single most valuable check.
3. **`K_sgs/K_T ≪ 1`.** Above ~0.5 the number characterises the AMD closure
   rather than the flow, and must be quoted alongside every `K_T`.
4. **`delta_eff` steady.** Tests, rather than assumes, the "peak = mean" shorthand.
5. **Only then, the panel (d) slope.** ½ ⇒ `√TKE·l`; 1 ⇒ `TKE/N`.

## Things worth knowing

- **`κ` means different things in the two projects.** In `Stokes/3D/case_params.jl`
  it is the molecular diffusivity `ν/Pr = 1e-7`; in the Ekman `Parameters.jl` it
  is the **von Kármán constant, 0.41**, and the diffusivity is `κ₀`. Writing `κ`
  in the subgrid flux here would add 0.41 m² s⁻¹ of "molecular" diffusivity to
  every cell — six million times too large, larger than any turbulent diffusivity
  in the domain — and would silently make `K_T` nonsense without erroring.
  `Moments.jl` uses `κ₀` everywhere and carries a banner about it.
- **The closure tuple is in the opposite order** from the Stokes case:
  `(ScalarDiffusivity, AnisotropicMinimumDissipation)` rather than the reverse, so
  AMD is entry 2 of `model.closure_fields`. `Moments.jl` searches rather than
  indexes, so neither order is assumed.
- **The drag coefficient is genuinely valid here.** `cᴰ = (κ_vk/log(z₁/z₀))²`
  needs the first cell inside a log layer. This grid's is at `z₁ = 0.0667 m
  = 41.7 z₀`, comfortably so — unlike the wall-resolving Stokes grid, where the
  same formula had to be evaluated at a fixed reference height instead.
- **The inertial period is the clock.** `2π/f₀ = 62832 s`, numerically identical
  to the Stokes tidal period because `ω` there was set to `1e-4` to match this
  `f₀`. So both studies use the same output cadence and the same smoothing
  windows, and their `K_T` results are comparable without adjustment.
- **`SKIP_PERIODS` defaults to 2, not 1.** This run starts from `u = U∞ + noise`
  and has to trip, become turbulent and grow a boundary layer. That transient is
  not the physics being measured. 2 of the 6.37 periods are discarded, leaving
  ~4.4 usable. For the panel (d) slope prefer `SMOOTH=tide` (the full-period
  envelope) over the default `tide20`.
- **`sharp = 6` is fine on this grid.** The Stokes study had to drop its default
  from 6 to 2, because its grid coarsens to `Δz ≈ 0.34 m` above 10 m and a 0.73 m
  transition is only ~2 cells there — the pycnocline starts life as a numerical
  step. This grid is almost uniform at `Δz = 0.135 m` from the wall past 30 m, so
  the same `sharp = 6` gives **5.4 cells** across the transition at `z = 20 m`.
  Nothing needed changing.
- **`TKE.jl` is kept as an independent check.** It computes TKE the original way —
  from the `(:, 1, :)` x–z slice, averaging fluctuations over a single y-plane —
  while `MixedLayerDiffusivity.jl` uses the full plane averages `Moments.jl` forms
  on the GPU at every sample. They should agree to within the sampling noise of
  one slice. If they do not, one of them is wrong.
- **`GRAD_FLOOR` dominates the error bar on `K_T`.** `K_T = −F_b/(dB/dz)` is 0/0
  inside the mixed layer, so cells with `|dB/dz| < GRAD_FLOOR·N²_ref` are masked
  (default 0.05). Sweep it and quote the spread:
  ```bash
  for g in 0.02 0.05 0.10; do
      GRAD_FLOOR=$g RESULT_SUFFIX=_g$g julia --project=. \
          "Ekman/TKE analysis/MixedLayerDiffusivity.jl"
  done
  ```
