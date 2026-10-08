# From mixed-layer growth to a closure for $K_T$

The theoretical side of the project: what we are trying to derive, and the
reasoning behind each step. How the simulations are run is not covered here;
see `README.md` and `LOG.txt` for that.

## 1. The goal: a growth time scale

A bottom mixed layer of thickness $h$ deepens into the stratified fluid above it
by turbulent diffusion. Treated as diffusive spreading, $h^2 \sim 4 K_T t$, so

$$
\frac{dh}{dt} = \frac{2 K_T}{h}.
$$

If $K_T$ were constant this would give $h \propto t^{1/2}$ with time scale
$t_h = h^2 / 4K_T$. In practice $K_T$ depends on the state of the layer. If it
can be written as a function of $h$ alone, the growth law separates:

$$
t(h) = \int^{h} \frac{h'}{2 K_T(h')}\, dh' .
$$

**The whole task is therefore to find $K_T(h)$.** Everything below serves that.

## 2. What $K_T$ is

$K_T$ is the eddy diffusivity of buoyancy, defined by the flux–gradient relation

$$
K_T = -\frac{\langle w'b' \rangle}{\partial \langle b \rangle / \partial z},
$$

and evaluated at $z = h$, where the deepening happens. It is something we
*measure*, not something we know how to predict. We need a model for it in
terms of quantities that we *can* predict as functions of $h$.

## 3. Step one: $K_T$ in terms of TKE, using a mixing length

A diffusivity has units of m²/s, i.e. a velocity times a length. Turbulence
supplies the velocity, $\sqrt{\mathrm{TKE}}$. Write

$$
K_T = \sqrt{\mathrm{TKE}} \; L_K ,
\qquad\text{i.e.}\qquad
L_K \equiv \frac{K_T}{\sqrt{\mathrm{TKE}}} .
$$

This is Prandtl's mixing-length idea. On its own it is **a definition, not an
assumption**: $L_K$ is just $K_T$ rewritten. The point of rewriting it is that
it turns a question about a diffusivity into a question about a *length*, the
typical size of the eddies doing the mixing. There are physical arguments for
what limits an eddy's size, but none that apply to a diffusivity directly.

So if $\mathrm{TKE}(z)$ is known (step 5), the problem reduces to finding $L_K$.

## 4. Step two: what limits the eddy size — $L_N$ and $L_s$

There are two candidate physical limits, both built from the same
$\sqrt{\mathrm{TKE}}$:

| scale | definition | physical meaning |
|---|---|---|
| $L_N$ | $\sqrt{\mathrm{TKE}}/N$ | **Stratification limit.** An eddy with vertical velocity $\sim\sqrt{\mathrm{TKE}}$ can only lift fluid a distance $L$ before its kinetic energy is used up against buoyancy: $\tfrac12 w^2 \sim \tfrac12 N^2 L^2$. |
| $L_s$ | $\sqrt{\mathrm{TKE}}/S$ | **Shear limit.** The mean shear $S = \lvert\partial \mathbf{u}_h/\partial z\rvert$ creates and deforms eddies on a time $1/S$, so eddies cannot be coherent over much more than $\sqrt{\mathrm{TKE}}/S$. |

Whichever constraint is tighter, i.e. whichever scale is smaller, should win. So the
combined scale is

$$
\frac{1}{L_\mathrm{harm}} = \frac{1}{L_N} + \frac{1}{L_s}
$$

(or $\min(L_N, L_s)$, its non-smooth version). Each term dominates in its own
regime: strong stratification gives $L_\mathrm{harm} \to L_N$, and strong shear
gives $L_\mathrm{harm} \to L_s$.

Their ratio is a familiar number:

$$
\frac{L_s}{L_N} = \frac{N}{S} = \sqrt{Ri},
$$

so the switch between the two regimes is at $Ri \approx 1$. **The reason for
introducing both scales** is that one alone covers only one regime. A
stratification-only model ($L_K \propto L_N$) breaks down where the shear is
strong and the stratification weak.

## 5. Step three: fitting — finding the form and the constants

Section 4 gives the variables, but not the constants or the exact functional
shape. Those come from data, by plotting $L_K$ against a candidate scale and
fitting:

| form | what it tests |
|---|---|
| $L_K = A\,L$ | the simplest closure: eddy size proportional to the limiting scale, with one dimensionless constant |
| $L_K = A\,L^{b}$ | whether proportionality is right, i.e. whether $b \approx 1$ |
| $L_K = L_\infty\left(1 - e^{-L/x_0}\right)$ | whether there is also a cap: eddies cannot grow without limit, so $L_K$ saturates at large $L$ |

Each candidate scale ($L_N$, $L_s$, $L_\mathrm{harm}$, $\min$, weighted
variants…) is fitted the same way, and they are compared by the rms scatter
about the fit. The comparison answers two questions:

1. **Which scale** collapses the data? A good scale puts different stratifications
   and different flows on one curve.
2. **Which form** is needed? The saturating form fits better, but it introduces two
   *dimensional* constants ($L_\infty$, $x_0$, in metres) that the theory does
   not explain. The proportional form has one dimensionless constant and is the
   one a closure would want.

## 6. Step four: non-dimensionalising

Substitute the definitions into the proportional law $L_K = A\,L_\mathrm{harm}$:

$$
\frac{\sqrt{\mathrm{TKE}}}{K_T} = \frac{1}{A}\left(\frac{N}{\sqrt{\mathrm{TKE}}} + \frac{S}{\sqrt{\mathrm{TKE}}}\right)
\;\;\Longrightarrow\;\;
K_T = \frac{A\,\mathrm{TKE}}{N + S},
$$

and multiply by $N/\mathrm{TKE}$:

$$
K_T^{*} \equiv \frac{K_T\,N}{\mathrm{TKE}} = A\,\frac{\sqrt{Ri}}{1 + \sqrt{Ri}} .
$$

**This is the same model written differently, so it cannot fit the data any
better.** It is done for interpretation:

- **It removes every dimensional constant.** What is left is one dimensionless
  number $A$ and one dimensionless function of one dimensionless variable. That
  is the form a law must take if it is to carry over between flows, depths and
  parameters. It also shows that any extra dimensional constants (the
  saturating fit's $L_\infty$, $x_0$) are not explained by the theory.
- **It exposes the single control parameter, $Ri$.** In length form you have to
  track two scales. Here a single axis carries the whole regime structure.
- **It makes the two limits explicit:**
  - $Ri \to \infty$: $K_T^* \to A$, i.e. $K_T \to A\,\mathrm{TKE}/N$ (stratification-limited)
  - $Ri \to 0$: $K_T^* \to A\sqrt{Ri}$, i.e. $K_T \to A\,\mathrm{TKE}/S$ (shear-limited)

  A plot of $K_T^*$ against $\sqrt{Ri}$ should show a rising branch, a knee at
  $Ri \approx 1$ and a plateau. That is a direct visual test of the two-regime idea.
- **It shows what a single flow can and cannot test.** The relation can only be
  checked if the data span $Ri$ on both sides of 1. A flow whose $Ri$ at the
  layer top stays roughly constant (for example because the layer adjusts to a
  marginal $Ri$) cannot test it, however many runs there are.

Two cautions:

- $K_T^* \propto N$ and $Ri \propto N^2$ both contain $N$, so some correlation
  between the axes is built in. The same is true of $L_K$ against $L_\mathrm{harm}$,
  which share $\sqrt{\mathrm{TKE}}$.
- $Ri$ is an *output* of the flow, not a control parameter.

## 7. Closing the loop: back to $K_T(h)$

The closure gives $K_T = A\,\mathrm{TKE}/(N+S)$ at $z = h$. To use it in
$dh/dt = 2K_T/h$, each ingredient has to be a function of $h$:

| ingredient | how it becomes a function of $h$ |
|---|---|
| $N$ | the background stratification. It is known. |
| $\mathrm{TKE}(h)$ | from a vertical profile of TKE (time-averaged over the forcing cycle in the oscillating case), e.g. $\mathrm{TKE}(z) = A_1 e^{-A_2 z/\delta}$ with $A_1 \propto u_*^2$ and $\delta$ a boundary-layer thickness, evaluated at $z = h$ |
| $S(h)$ | **still needed.** The shear at the layer top also needs a model in terms of $h$ (and $u_*$, $N$). Without one, $K_T$ is a function of $h$ *and* $S$, not of $h$ alone. |

With all three in place,

$$
\frac{dh}{dt} = \frac{2A}{h}\,\frac{\mathrm{TKE}(h)}{N + S(h)},
$$

which can be integrated for $t(h)$, and the time scale for growth read off.

A related idea, the "$\delta$ model", tested whether the fitted constants
themselves scale with the boundary-layer thickness. The aim was to replace the
dimensional constants of the saturating form with multiples of $\delta$, so that
they become explained rather than fitted.

## Summary of the logic

1. Growth law: $dh/dt = 2K_T/h$. We need $K_T(h)$.
2. Rewrite $K_T = \sqrt{\mathrm{TKE}}\,L_K$. Now we need an eddy size $L_K$.
3. Eddy size is limited by stratification ($L_N$) or by shear ($L_s$),
   whichever is smaller: $1/L_\mathrm{harm} = 1/L_N + 1/L_s$.
4. Fit $L_K$ against the candidate scales to find the form and the constants.
5. Non-dimensionalise: $K_T N/\mathrm{TKE} = A\sqrt{Ri}/(1+\sqrt{Ri})$. This is the
   same model, but it shows the single parameter ($Ri$), the two regimes and
   which constants are dimensional.
6. Close with $\mathrm{TKE}(h)$ and $S(h)$, and integrate the growth law.
