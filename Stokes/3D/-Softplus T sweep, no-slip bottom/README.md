# Softplus T sweep, no-slip bottom

The sweep that produced `figures/Figure4_softplus_*.png` and
`figures/Figure5_P4_T*_sqrtRi*.png`. Archived when the K_T / TKE study changed
the bottom boundary condition, so nothing here is directly comparable with what
comes after it.

## What it was

| | |
|---|---|
| background | softplus (`PROFILE=4`), `sharp = 6` |
| pycnocline height | `T ∈ {5, 10, 15, 20, 30}` m |
| stratification | `N/ω = √Ri ∈ {0, 0.5, 1, 2, 5, 10}` |
| bottom BC | **no-slip** (`ValueBoundaryCondition(0)`) |
| duration | 8 tidal periods per case, from a shared `Ri = 0` spin-up |
| grid | two of them — see below |

**Two grids are mixed in here, and that matters.** `outputs/P4_T5_*` is the
re-run at 100 × 100 × 300 (Δz = 0.0086 m at the wall, Δx = 0.10 m); every other
column — T = 10, 15, 20, 30 — is the earlier 48 × 48 × 254 data (Δz = 0.0121 m,
Δx = 0.208 m). `Figure4_softplus_sweep.png` stacks them as rows, so a difference
between the T = 5 row and the rest is partly discretisation and not only physics.
That is why the driver ran `Figure4_metres.jl` with `SKIP_SWEEP=1`.

## Reproducing it

The `.jl`, `.sh` and `.toml` files here are the exact versions from commit
`9a40c44`, which is what generated the data. They are a snapshot, not live code:
run them from a checkout of that commit, not against the current `Stokes/3D/`.

Note `SHARP`: `case_params.jl` in the working tree now defaults to 2, so
redrawing any of these figures with the current scripts needs `SHARP=6` set
explicitly or the t = 0 overlay will not be the initial condition these runs
actually started from.

Measured throughput, quoted by the current driver's budget note:
`logs/P4_T5_sqrtRi*.log` give 1.98–2.11 h for 8 periods on 100 × 100 × 300.
