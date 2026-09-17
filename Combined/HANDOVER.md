# Starting a new Claude session on this work

Paste the block below as the first message of a new conversation. It gives the
working rules, points at the two documents that carry the history, and says what
the next step is.

**Keep it current.** It deliberately holds only what a fresh session cannot
infer from the repo: the rules, where things stand, and the gotchas. The
history itself lives in `LOG.txt` and the reasoning in `README.md` — do not
copy those in here, or there will be two versions to keep in sync.

---

```
I'm working on LES of the ocean bottom mixed layer (Oceananigans.jl) in
~/SRIM-2026, branch Stekman. Two simulations of the same physics: "Stokes"
(oscillating tidal boundary layer, Stokes/3D, mine) and "Ekman" (steady
rotating, Ekman/, a collaborator's). ω = f = 1e-4 deliberately, so the sweep
parameter r = N/ω = N/f means the same background N in both. All the
comparison work lives in Combined/, at T = 10 m.

START BY READING, in this order:
  Combined/LOG.txt    ~400 lines, the narrative: what was done, in order, what
                      came of it, mistakes made, and what's still open
  Combined/README.md  the detail and the reasoning behind each figure

Those two are kept deliberately split — LOG.txt is the story, README.md is the
justification. Keep it that way if you add to them.

HOW THE WORK ACTUALLY RUNS:
This machine is for writing and checking code, not for running the simulations.
The long runs go on a Slurm HPC cluster that cannot run Claude. The loop is:
write/verify the script here -> I commit and push it -> I run it on the cluster
-> I scp the data files back into Combined/Data/ and commit the small artefacts
(logs, figures, .done markers; the .jld2 are gitignored) -> I ask you to check
it landed correctly. So: batch scripts must be right before they leave, because
the feedback loop is hours long, and "run it and see" is not available to you.

STANDING RULES:
- Work in Combined/. Do NOT change any code in Ekman/ — it's the collaborator's.
  Copy anything that needs changing into Combined/ first. `git status
  --porcelain Ekman/` must stay empty.
- Do NOT stage, commit or push — not `git add`, not `git commit` — unless I ask
  in so many words. Leave everything in the working tree and tell me what
  changed; I review and commit myself.
- No Co-Authored-By: Claude or Claude-Session trailers in any commit message.
- After EVERY task, update all three documents: HANDOVER.md (where things stand
  / next step / new gotchas), README.md (the reasoning and the detail) and
  LOG.txt (a dated entry in the narrative). Not at the end of a session — after
  each task, as part of it.
- Keep code CONCISE with MINIMAL comments — aim for the style of
  Ekman/3D Simulation/Ekman 3D.jl: short single-line comments where something
  is not obvious, and nothing else. Do not write essay headers. (The older
  Combined/ scripts are heavily commented; that is not the target, and new code
  should not copy it.)
- Run Julia from inside Combined/ with --project=. (--project=.. is empty).
  Scripts live one level down and are invoked with their folder, e.g.
  `GKSwstype=100 julia --project=. plot/plot_delta_T10.jl`. The layout is
      Combined/sweep.jl, mixed_layer_height.jl   shared, included by everything
      Combined/plot/     the 14 plot_* scripts
      Combined/reduce/   the 3 reduce_* scripts
      Combined/run/      ekmanrun.jl, swirles.sh
      Combined/Data/raw/      Ekman, Ekman_moments, lowN, rough_ekman
      Combined/Data/cache/    the derived .jld2 every plot script reads
      Combined/figures/, Combined/logs/
  Moved scripts use `const HERE = dirname(@__DIR__)`, so HERE is still
  Combined/ and every joinpath(HERE, ...) is unchanged. A NEW script in plot/
  must do the same, not @__DIR__.
- One figure per question. Extra diagnostics go in the log file, not more panels.
- Every plotting script opens with default(dpi = 600, fontfamily = "DejaVu
  Sans") and uses LaTeXStrings for all labels.

WHERE THINGS STAND:
The sweep is 16 cases — Stokes and Ekman each at r = 0.2, 0.5, 1, 2, 5, 10, 25,
50 — and every figure in Combined/figures was rebuilt on all of them on
2026-09-15. The low-N re-runs (swirles.sh) are done, checked and plotted.

The main result HOLDS on 16 cases: the mixing length L_K = K_T/√TKE collapses
onto a combined stratification/shear scale, 1/L_harm = 1/L_N + 1/L_s, as
L_K = 2.004(1 − e^(−L_harm/4.498)), rms 10.7 % (Stokes 11.7 %, Ekman 8.4 %) —
far better than either scale alone, and the low-N points are what make that
case: against L_N alone the both-flows rms went from 14.9 % to 31.9 % when they
were added, because those three points flatten off entirely.

The question the re-runs were for is ANSWERED. The shear-to-stratification
crossing (w_s = ½, i.e. Ri = 1) is now bracketed by data in both flows: Stokes
between r = 1 and 2, Ekman between r = 0.2 and 0.5. Four of sixteen cases are
shear-limited where it used to be one of thirteen.

THREE THINGS MOVED, and they are the ones to know:
  - The Stokes h law got WEAKER. Refitting on eight cases gives
    h = 1.05 u_*/ω (1+N²/ω²)^(−0.020), flat to 7.3 %. It was (−0.040) and
    "flat to 1.6 %" on six cases — do not quote 1.6 % any more. The Ekman law
    h = 0.85 u_*/f (1+N²/f²)^(−0.155) took its new point without moving, 4.2 %.
  - min(L_N, L_s) now edges out the harmonic as the saturating fit, 10.0 % vs
    10.8 %; on 13 cases it was the other way round. They are inside each other's
    noise. The harmonic is still the written form, but it is no longer a winner.
  - The δ model still does NOT work (δ = h gives 10.8 % against 10.7 %
    unscaled). Verdict unchanged, margin narrower. See stage 3 in LOG.txt.

The Corrsin scale was tried as the shear-side partner for L_N on 2026-09-16
(plot_corrsin_scales_T10.jl, three figures). IT DOES NOT WORK and the reason is
worth keeping: L_C/L_N never crosses 1 in any reliable case, so the Corrsin
scale is always the smaller and there is no regime where L_N takes over. On top
of that the eps>0 rule drops Stokes r = 10, 25, 50, and over what is left
(r <= 5) L_C moves only x1.52 while L_K moves x2.9. Combined fit 12.8 % on 13
cases against 10.0-10.8 % on 16 with L_s. Keep L_s = sqrt(TKE)/S; L_s/L_N is
sqrt(Ri) and that is the quantity with the crossing.

A nondimensional reformulation K_T* = K_T N/TKE = f(Ri) was assessed on
2026-09-16 (numbers in LOG.txt, no figure yet). It is algebraically the same
model as L_K = A L_harm, so it reproduces 23-24 %, not the saturating form's
10.7 % — its value is clarity about what the good fit costs, not accuracy. The
blocking problem is that Ri is an output, not a control: the Ekman S grows with
N so its Ri is pinned near 10 for r = 1..50 (six of eight cases) while K_T*
still moves. The tidal column spans Ri over five decades and can carry such a
law; the rotating one cannot without a run that varies S independently of N.

Built 2026-09-16: plot_KTstar_Ri_stokes_T10.jl, the harmonic law
nondimensionalised for the Stokes column, K_T* = K_T N/TKE = A sqrt(Ri)/(1+sqrt(Ri))
at z = h on the background N. A = 0.412, rms 14.7 % over five decades of Ri, both
asymptotes populated, no length scale anywhere — the cleanest form of the
two-regime result so far. A local-gradient Ri was tried and dropped: it is
degenerate at z = h (h IS the 0.1 N2_ref gradient crossing, so N_loc = sqrt(0.1)
N_bg exactly) and no fixed height is fair either (26-47 % over z = 9-18 m).

THERE IS NO EKMAN VERSION AND THERE CANNOT EASILY BE ONE. S ~ N^1.018 at z = h
in the rotating column against N^0.281 in the tidal one, so Ekman Ri is pinned
near 10 while Stokes Ri spans five decades. The tidal layer's thickness is set
by omega and cannot respond to N; the steady rotating layer equilibrates to a
marginal Ri at its top. A z-scan (plot_KTstar_Ri_zscan_T10.jl, built
2026-09-16) DOES fix the range — Ekman spans Ri 8.8e-3 to 4.5e5 — but the
plateau is NOT universal, A = 0.331 Stokes against 0.149 Ekman, and the collapse
loosens to 50-52 % because the scan mixes the turbulent interior with the
quiescent fluid above the layer. Inside the mixed layer Ri is genuinely small
but db/dz vanishes, so K_T does not exist there and no height fixes that.

DONE 2026-09-17: the rough-bed Ekman run (r = 25, z0 0.0016 -> 0.0137 m, c_D
x4, everything else identical) finished locally in 8.76 h and ANSWERS ITS
QUESTION: Ri at the Ekman layer top IS self-regulated. sqrt(Ri) at z = h moved
3.127 -> 3.195, +2 %, while h went 8.31 -> 10.26 m. As elasticities against the
measured u_*: d ln h/d ln u_* = 1.27 but d ln sqrt(Ri)/d ln u_* = 0.13, against
about -1 if Ri were slaved to the forcing. The layer answers extra bottom
friction by thickening, not by shearing harder. So the Ri ~ 10 pinning across
r >= 1 is a property of the steady rotating layer, not a coincidence — and there
is no second sweep axis here for an Ekman K_T*(Ri) curve.
Data: Combined/Data/rough/ekman. Figure: figures/rough_vs_smooth_Ri_T10.png.
CAVEAT: c_D x4 gave only u_* x1.18, because a rougher bed slows the first-cell
velocity and eats the gain. A sharper test would raise U_inf, at roughly double
the runtime.

The Ekman counterpart figure exists (plot_KTstar_Ri_ekman_T10.jl,
figures/KTstar_vs_Ri_ekman_T10.png): A = 0.330, rms 26.2 %, but six of eight
cases sit at Ri = 8-13 and the fit rests entirely on r = 0.2 and 0.5. Keep it as
the visual argument for why the rotating column cannot carry the relation, NOT
as a result — and do not quote its 26.2 % against the Stokes 14.7 % as if the
two were comparable measurements.

NEXT STEP: not chosen. The open list at the end of LOG.txt is the menu; the
strongest item is a second pycnocline depth (T = 15 or 20, which already exist
on the Stokes side), because it is the only way to tell whether the Stokes h
law's 7.3 % is the low-N cases behaving badly or the layer interacting with the
T = 10 m pycnocline.

GOTCHAS THAT COST TIME TO FIND:
- Combined/sweep.jl is the single place that says what the sweep is (SVALS,
  RATIOS), where a case lives (stokes_case, ekman_case — Data/lowN is searched
  BEFORE the original roots) and what colour an r gets. Every reduction and plot
  script includes it. Adding a case is one edit there, not eight.
- REBUILD=1 is required on plot_l_vs_corrsin_T10.jl. It caches the Stokes walk
  and reuses it silently, so without the flag it rebuilds the figure from
  whatever case set was current when the cache was written.
- ekmanrun.jl's case_dir() formats r as %.1f, so an Ekman r = 0.25 silently
  becomes the folder "r=0.2". swirles.sh guards against it; fixing it properly
  means changing the readers too.
- K_T = −F_b/⟨∂b/∂z⟩ is badly conditioned at low N — that's why Stokes r = 0.5
  was dropped originally. Check the K_T_bulk vs K_T_pe agreement before
  trusting any new low-r point. (For the 2026-09 lowN runs it agrees to 0.1 %,
  so those two are fine; the check still applies to anything new.)
- The Stokes h(t) sits on the z = T = 10 m pycnocline at EVERY r, not just the
  low-N ones. h_pin (fraction of samples within 5 % of T, in profiles_T10.jld2)
  is 24 % at r = 0.2 and 94 % at r = 10. This looked like a reason to exclude
  the low-N cases from the h fit; it is not. All eight are fitted.
- Ekman r = 0.2 has h settled to +0.9 % drift but l still falling at −20 %.
  h equilibrating does not mean l has — check both.
- δ_eff = ∫F_b dz / F_b|peak goes negative wherever F_b changes sign, which it
  does several times per record at Stokes r = 0.5 (CV 0.43). Drop those phases
  before taking a median.
- Always smooth in time before forming a ratio, and form ratios per sample
  before taking the median — never median/median.
- L_C is reserved for the Corrsin scale. Combined scales are L_harm, L_β, L_comb.
- A saturating fit is not a fit when its knee is outside the data, even if no
  parameter hit a grid edge. plot_l_vs_Ls_T10.jl calls x0 >= 2*max(x) "no knee"
  and refuses to draw it; without that a straight line gets drawn as a
  saturating curve. Worth applying wherever fit_sat is used.
- GR's mathtext has no \text or \mbox, no line breaks in labels, swallows
  spaces inside \mathrm{} unless escaped, and prints %g as "2.16e + 03".
- Piping a Julia run into `head` SIGPIPE-kills it before it writes its figures.
```
