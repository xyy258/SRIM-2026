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
crosses should be read against the squares.** Getting the circles for Ekman needs
an `F_sgs` output writer added to a copy of `Ekman 3D.jl` and the T = 10 column
re-run.

A second, smaller asymmetry: `Velocity.jld2` and `Buoyancy.jld2` are written with
`indices = (:, 1, :)`, so the Ekman averages are over 100 x points at one y rather
than the full plane. The means subtracted are the true horizontal averages from
`Avg_vel.jld2` and `Avg_b.jld2`, so the fluctuations are unbiased, just noisier.
