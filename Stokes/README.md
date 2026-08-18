# Stokes/tidal model (tll46)

3D LES of an oscillating (tidal) bottom boundary layer on Oceananigans, after
Gayen, Sarkar & Taylor (2010). Everything lives in `3D/`; `2D/` is the earlier
two-dimensional version.

## TKE and turbulent diffusivity (`K_T`)

Measures an effective turbulent diffusivity from the simulations and tests how it
scales with turbulent kinetic energy — specifically whether `K_T ~ √TKE · l`
(log–log slope ½) or `K_T ~ TKE / N` (slope 1).

### The new files

| file | what it does |
|---|---|
| `3D/Moments.jl` | Included by `Tidal3D.jl` just before `run!`. Registers one extra output writer producing 13 plane-averaged profiles to `<filename>_moments.jld2`: means `U V W B dBdz`, raw second moments `uu vv ww` (Centers) and `uw vw wb` (Faces), and the subgrid pair `kappa_sgs F_sgs` (Faces). Changes no existing writer, filename or schedule. `MOMENTS=0` turns it off. |
| `3D/MixedLayerDiffusivity.jl` | Post-processing. Reads `*_moments.jld2`, writes `outputs/<tag>/mixing_<tag>.jld2` and `figures/K_T_<tag>.png`, and prints a VERIFICATION block. |
| `3D/measure_h0.jl` | Measures `h0`, the height at which TKE falls to ~1 % of its near-wall peak at peak phase. This is what stage 0 exists to produce. |
| `3D/run_moments_sweep.sh` | Staged driver, for interactive use. |
| `3D/swirlesrun.jl` | The same sweep for a Slurm allocation: preflight, resumable per-case markers, wall-clock budget. This is what `swirles.sh` runs — the file **replaced** the earlier figure-4/5 column driver of the same name, which is archived with the sweep it drove. |

`3D/case_params.jl` also changed: the default `SHARP` is now **2**, not 6 (see
below). `3D/Tidal3D.jl` now uses a **quadratic drag** bottom instead of no-slip.

The previous sweep — softplus, `T ∈ {5,10,15,20,30}`, no-slip bottom, and the
figure-4/5 driver that produced it — is archived whole under
`3D/-Softplus T sweep, no-slip bottom/`, data and scripts together, so
`3D/outputs/`, `3D/figures/` and `3D/logs/` start empty for this study.

### How to run

The default is **one column**: `T = 10 m`, `N/ω ∈ {0, 1, 2, 10}`, 8 tidal
periods. On the cluster `swirles.sh` is unchanged and takes no arguments:

```bash
sbatch swirles.sh          # spin-up → stage 0 (+h0, +the T check) → the rest of the column
```

**To see one result quickly**, run only the unstratified control — spin-up plus
the `N/ω = 0` case, ~3.9 h:

```bash
MOMENTS_STAGE=stage0 sbatch swirles.sh
```

That case is kept and marked, so a later bare `sbatch swirles.sh` skips it and
runs only the three stratified cases (~7.4 h). Nothing is paid twice. Note what
the control can and cannot tell you: at `N/ω = 0` the `b` field is a passive
scalar, so `K_T` is a genuine tracer diffusivity and checks 1–4 all apply — but
`K_T ~ TKE/N` is undefined at `N = 0`, so the panel (d) slope there tests the
measurement chain against the `√TKE·l` branch rather than discriminating between
the two hypotheses. Add `FIELDS3D_CASE=P4_T10_sqrtRi0` if you want its 3D fields
kept for animation.

If the wall-clock guard defers the last case, re-submitting the same line
finishes it — every case carries its own marker.

Interactively, the same thing in stages:

```bash
cd Stokes/3D
./run_moments_sweep.sh spinup     # drag spin-up, 5 periods           ~1.3 h
./run_moments_sweep.sh stage0     # the column's N/ω=0 case → h0      ~2.4 h
./run_moments_sweep.sh stage1     # the column, T=10, N/ω∈{0,1,2,10}  ~9.6 h
./run_moments_sweep.sh post       # re-run the post-processing only
./run_moments_sweep.sh            # no argument: status of every case
```

A second column later, if you want the T dependence:

```bash
MOMENTS_STAGE=stage1 STAGE1_T=5 sbatch swirles.sh
```

### Why `T = 10 m`

Not a guess — it comes from the previous sweep's own figure 5 panels, archived
under `3D/-Softplus T sweep, no-slip bottom/figures/`. By period 8 the
mixed-layer top sits at **12.5 m at `N/ω = 2`** and **10.4 m at `N/ω = 10`**. So
at `T = 10` the turbulent layer reaches the pycnocline in *every* case, and how
far it entrains past it varies strongly with stratification. That contrast is
what `K_T` is measured from: a pycnocline placed above the layer is never
reached and measures nothing, and one buried deep inside it is wiped out in the
first period or two, collapsing `Δb` and making `K_T = −∫F dz/Δb`
ill-conditioned.

Those runs had a **no-slip** bottom. Drag raises `u*` and deepens the layer, so
stage 0 re-measures `h0` from the column's own `N/ω = 0` case and checks
`0.3 h0 ≲ T ≲ 2 h0` before the other three cases are spent. Outside that window
the job stops and prints the `T` to use instead, having paid for one case rather
than four. `T_H0_FORCE=1` overrides.

### The 12 h budget

`swirles.sh` asks for `--time=12:00:00`. Measured throughput on this grid — the
previous sweep's own `logs/P4_T5_sqrtRi*.log`, now archived — is **1.98–2.11 h
for 8 tidal periods**, i.e. 0.26 h/period; the moments writer adds roughly 15 %,
so budget **0.30 h/period**.

| | |
|---|---|
| drag spin-up, 5 periods (no moments writer) | 1.3 h |
| 4 cases × 8 periods | 9.6 h |
| post-processing | 0.2 h |
| **total** | **11.1 h** in a 12 h wall |

It fits, with about an hour of margin. The job measures its own throughput after
the first case and replaces the estimate, so the guard tracks reality rather than
this table; if the node runs slow the last case is deferred to a short second
submission rather than being lost.

What that drops from the brief:

| | brief | now | what it costs |
|---|---|---|---|
| `T_VALUES` | {2,3,5,8} | **{10}** | one column instead of a T-sweep. The exponent comes from varying `N` at fixed `T`, which is what a column is; the geometry test (does `delta_eff` track `h` across `T`) goes, and stage 3 with it. |
| `N_PERIODS` | 16 | **8** | 7 usable periods after `SKIP_PERIODS=1` — enough for the `T_tide` boxcar, the phase bins and the log–log fit. Statistics only. |
| `SQRT_RI` | {0,1,2,5,10} | **{0,1,2,10}** | drops 5, which sits between 2 and 10 on a log axis. The full decade in `N` and the `N = 0` control both survive. |

The brief's full version is `T_VALUES="2 3 5 8" N_PERIODS=16 SQRT_RI="0 1 2 5 10"`
— 20 cases, ~96 h, eight or nine submissions.

Post-processing on its own, following the `Figure4_metres.jl` / `Figure5.jl`
conventions:

```bash
T_VALUES="10" N_OVER_OMEGA="0 1 2 10" julia --project=. MixedLayerDiffusivity.jl
```

### Interpretation order — read the diagnostics in this order

An earlier failure invalidates everything after it. The VERIFICATION block prints
them numbered and says which fail.

1. **`⟨w⟩_xy ≈ 1e-18.**  With a rigid lid, an impermeable bottom and
   incompressibility this is exact, which is what makes `⟨w′b′⟩ = ⟨wb⟩` and
   `⟨u′w′⟩ = ⟨uw⟩` with no mean subtraction. If it fails, stop — every flux below
   it is wrong.
2. **`K_T_bulk` vs `K_T_pe` (panel c).**  One route uses `wb` and `F_sgs`, the
   other only `B`; they share no code path. Disagreement means the subgrid flux is
   wrong, usually under-counted, which shows as the flux route reading low. This
   is the most valuable single test in the project, and stage 3 is gated on it.
3. **`K_sgs/K_T ≪ 1`.**  Above ~0.5 the number characterises the AMD closure
   rather than the flow, and must be reported alongside every `K_T` value.
4. **`delta_eff` steady, ideally tracking `h`.**  This tests, rather than assumes,
   the "peak = mean" shorthand — replacing the flux integral by its peak value is
   only legitimate if the flux profile's effective width is comparable to the
   depth it is spread over.
5. **Only then, the panel (d) slope.**  ½ ⇒ `√TKE·l`; 1 ⇒ `TKE/N`.

### Things worth knowing before trusting a number

- **`GRAD_FLOOR` dominates the error bar on `K_T`.**  `K_T = −F_b/(dB/dz)` is 0/0
  inside the mixed layer, so cells with `|dB/dz| < GRAD_FLOOR · N²_ref` are masked
  (default 0.05). Sweep it and quote the spread:
  ```bash
  for g in 0.02 0.05 0.10; do
      GRAD_FLOOR=$g RESULT_SUFFIX=_g$g julia --project=. MixedLayerDiffusivity.jl
  done
  ```
- **`T_tide = 2π/ω = 62832 s = 17.45 h`**, not the 12.4 h of a real M2 tide: `ω`
  was set to `1e-4` to match a colleague's Coriolis parameter. Smoothing windows
  scale with that. `SMOOTH=tide20` (default, ~52 min) keeps the intra-cycle burst;
  `SMOOTH=tide` gives the slowly evolving envelope, which is what mixed-layer
  growth should be read from; `SMOOTH=phase` bins by `mod(ωt, 2π)` and ensembles
  over periods.
- **The decomposition is instantaneous, and the smoothing comes second.**  The
  plane average *is* the Reynolds average here, so subtracting it removes the
  tidal flow exactly. Averaging the raw moments first instead would fold the
  variance of the tidal mean flow into "TKE" — a 100 % error. Neither script uses
  `AveragedTimeInterval`; see the header of `Moments.jl`.
- **Stage 0 is what makes stages 1–3 meaningful.**  Its only job is to measure
  `h0`. The T values in the drivers ({2, 3, 5, 8} m) are provisional and should be
  replaced by roughly {0.4, 0.7, 1.2, 2.0} × `h0`. Because the softplus background
  is unstratified below `z = T`, a pycnocline above the turbulent layer is never
  reached and such a case measures nothing at all.
- **`SHARP = 2`, fixed for every case.**  At the old default of 6 the pycnocline
  spanned ~6.9 cells at T = 5 but only ~2.1 at T = 10, 20, 30, so it began life as
  a numerical step at the large-T end and the T-sweep confounded stratification
  height with initial pycnocline resolution. At `sharp = 2` the transition is
  `2ln9/2 = 2.20 m` — the same *physical* width at every T. Do not vary `SHARP`
  per case to equalise the cell count; that re-confounds the two.
- **`*_moments.jld2` and `mixing_*.jld2` are gitignored.**  `3D/.gitignore`
  excludes everything under `outputs/` except the animations, so nothing new needs
  adding there. That is the right outcome anyway: a 16-period moments file is
  ~200 MB, over GitHub's per-file limit. The figures and this analysis are the
  deliverables; the data stays on the cluster.

### Departures from the brief that this code makes deliberately

- **`model.diffusivity_fields` does not exist** in the pinned Oceananigans
  (v0.110.11); it is `model.closure_fields`. `Moments.jl` resolves it by search
  rather than by name and logs what it found.
- **AMD's `κₑ` is clipped at zero** in this version
  (`anisotropic_minimum_dissipation.jl:203`), so the backscatter concern does not
  apply. `Moments.jl` re-checks it at runtime anyway.
- **`K_T_pe` needs a boundary term the brief drops.**  Integrating
  `∂B/∂t = −∂F_b/∂z` by parts leaves
  `d/dt ∫₀^H zB dz = −H·F_b(H) + ∫₀^H F_b dz`, so
  `K_T_bulk = K_T_pe − H·F_b(H)/Δb`. The brief assumes `F_b(zref) = 0`, but the
  softplus puts `N² = N∞²` at *every* level above `z = T`, so there is background
  subgrid and molecular flux at every candidate `zref` and no height where the
  flux vanishes. Measured, the dropped term was larger than `K_T_pe` itself, and
  restoring it brought the two routes from 171 % apart to 2.8 %. Both versions are
  computed, plotted and saved; check 2 is taken on the corrected one.
- **The drag coefficient.**  An earlier note in this file claimed the Ekman case
  used `cᴰ = 2e-3` with its log-law line commented out. That was wrong: in
  `Ekman 3D.jl` the log law is *live* and the no-slip line is the commented one,
  and it evaluates to **`cᴰ = 0.0121`**, from `z₁ = 0.0667 m = 41.7 z₀` on that
  much coarser near-wall grid. The Stokes grid resolves the wall (`z₁ = 0.0043 m
  = 2.7 z₀`), which is inside the roughness sublayer where a rough-wall log law
  is undefined, and evaluating it there returns `cᴰ = 0.172`. Since the two cases
  share a physical bottom they must share a physical drag coefficient, so the law
  is now evaluated at a fixed reference height `Z_DRAG_REF = 0.0667 m` — Ekman's
  own first cell — giving `cᴰ = 0.0121` on both. `CD` still overrides it
  outright. On this grid `z₁⁺ ≈ 8`, so the *molecular* stress `ν·U₁/z₁` is the
  larger term anyway and the drag law is a roughness correction on top of it,
  which is the right physics for a wall-resolving LES; `cᴰ = 0.17` would have
  swamped it by an order of magnitude.
- **The sweep is one column, not a 4 × 5 grid** — `T = 10` only, 8 periods
  rather than 16, `N/ω ∈ {0,1,2,10}` rather than `{0,1,2,5,10}` — because the
  brief's version is 96 h of GPU against a 12 h allocation. See the budget table
  above for what each cut costs and how to restore it.
- **Stage 0 is no longer a throwaway probe.** The brief had it run at its own
  `(T, N/ω)` and be discarded. It is now the column's own `N/ω = 0` case, kept
  and marked, so the h0 measurement is free rather than costing 2.4 h.
- **`Figure4_metres.jl` and `Figure5.jl` now default `SHARP` to 2**, tracking
  `case_params.jl`. Everything archived under
  `3D/-Softplus T sweep, no-slip bottom/` was made at `sharp = 6`, so redrawing
  those figures needs `SHARP=6` passed explicitly.
