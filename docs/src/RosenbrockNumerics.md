# Rosenbrock-average microphysics substepping

The [`RosenbrockAverage`](@ref CloudMicrophysics.BulkMicrophysicsTendencies.RosenbrockAverage) tendency mode
returns time-averaged microphysics tendencies over a time step `Δt` by taking `nsub` linearized-implicit
(Rosenbrock-Euler) substeps. Each substep solves

```math
\left(\frac{I}{h} - J\right)\, \Delta = f(x), \qquad x \leftarrow \max(x + \Delta,\, 0), \qquad h = \Delta t / n_\mathrm{sub},
```

where `f` is the raw pointwise tendency, `x` the species state, and `J` a matrix that approximates the tendency
Jacobian. The averaged tendency returned is `(x_final - x_initial) / Δt`. Temperature is advanced between
substeps from the latent heat of the realized increment.

## Options

`RosenbrockAverage` is parameterized by three independent option families:

- **`Jacobian`** — the matrix `J` used in the substep solve.
  - `DonorJacobian` — the donor-based linearization `M`: each transfer is linearized in its donor species,
    vapor sources enter as a constant, and rates are floored by `max(q_min, q_donor)`. This is the matrix the
    operational `LinearizedAverage` mode uses.
  - `CoupledDonorJacobian` — the donor-based matrix with the vapor-competition (Wegener–Bergeron–Findeisen)
    coupling added. The donor-based linearization keeps only donor-species slopes; restoring the dependence of
    each rate on the shared vapor specific content recovers the cross-species coupling and corrects the sign of
    the snow-from-cloud-liquid entry. The direct condensate dependence of the rates (rain ventilation, the
    availability terms) is not recovered; use `ExactJacobian` for the full derivative.
  - `ExactJacobian` — the exact tendency derivative, formed with `ForwardDiff`.
  - `ManualJacobian` — a hand-built approximation for the two-moment + P3 model, closed-form on the
    phase-change and number-adjustment couplings and donor-linearized on the remaining transfers. See
    [The folded and temperature-coupled Jacobians on the 2M+P3 model](@ref) below.
  - `TemperatureCoupledJacobian` — the two-moment + P3 model with the substep temperature promoted to a
    ninth prognostic variable, so the phase-change couplings are represented without folding the
    psychrometric feedback into the relaxation timescale. See the same section.

- **`GrowthTreatment`** — how the positive (growth) diagonal of `J` enters the implicit operator.
  - `ImplicitGrowth` — leave `J` unchanged.
  - `ExplicitGrowthDiagonal` — zero the positive diagonal entries of `J`, so a growth mode is taken explicitly
    and only the decay diagonal remains in the implicit operator.

- **`TendencyLimiter`** — a limiter applied to the realized substep increment.
  - `NoLimiter`.
  - `EndStateSaturationAdjustment` — scale the increment so the latent-heated end state does not cross saturation
    over its more-supersaturated phase, `max(S_ice, S_liq)` (see below).

Three preset configurations are supported:

| preset | Jacobian | growth | limiter |
|---|---|---|---|
| `rosenbrock_donor()` | `DonorJacobian` | `ImplicitGrowth` | `NoLimiter` |
| `rosenbrock_coupled()` | `CoupledDonorJacobian` | `ImplicitGrowth` | `NoLimiter` |
| `rosenbrock_exact()` | `ExactJacobian` | `ExplicitGrowthDiagonal` | `EndStateSaturationAdjustment` |
| `rosenbrock_manual()` | `ManualJacobian` | `ExplicitGrowthDiagonal` | `EndStateSaturationAdjustment` |
| `rosenbrock_manual_temperature()` | `TemperatureCoupledJacobian` | `ExplicitGrowthDiagonal` | `NoLimiter` |

`rosenbrock_donor()` reproduces `LinearizedAverage` (the operational donor-based scheme), now expressed within
the unified framework; in `Float64` the two agree to round-off. The donor-based matrices are not available on
the two-moment + P3 model; use `rosenbrock_exact()`, `rosenbrock_manual()`, or `rosenbrock_manual_temperature()`
there.

The `Verbose(mode)` wrapper additionally returns the per-process tendencies realized by the implicit solve,
attributed through the same substep factorization so that they sum to the net of the unlimited solve.

### Extending the framework

To add a new Jacobian, define `struct MyJacobian <: Jacobian end` and the methods `_jacobian_provider(::MyJacobian)`
(returning a `(g, x) -> J` provider) and `_species_mask(::MyJacobian, ::GrowthTreatment)`. A new growth treatment
is a `GrowthTreatment` subtype plus an `_apply_growth(::MyGrowth, J)` method; a new limiter is a `TendencyLimiter`
subtype plus an `_apply_limiter(::MyLimiter, x, Δ, ...)` method. The substep driver dispatches on the option types
at compile time, so a configured mode resolves with no run-time branch.

## The folded and temperature-coupled Jacobians on the 2M+P3 model

The two-moment + P3 model exposes two closures of the same phase-change (condensation/evaporation,
deposition/sublimation) physics: the folded 8×8 (`ManualJacobian`, preset `rosenbrock_manual()`) and the
temperature-coupled 9×9 (`TemperatureCoupledJacobian`, preset `rosenbrock_manual_temperature()`). Both
linearize the same relaxation-to-saturation process; they differ in whether the substep temperature is
carried as an explicit prognostic variable or eliminated analytically, and the two linearizations are not
numerically equivalent. This section states what each closure computes, an exact identity relating a
temperature-coupled step to a reduced species-only step in general, and why that general identity does not
connect the two closures as implemented.

### Two closures of the same relaxation

The phase-change tendency in both closures is a relaxation toward saturation, with capacitance-integral
timescale `τ` and supersaturation `s`,

```math
s = q_v - q_{v,\mathrm{sat}}(T).
```

`Instantaneous2MP3Tendency` ([`_condevap_derivs`](@ref CloudMicrophysics.BulkMicrophysicsTendencies._condevap_derivs)
gives its closed-form derivative) folds the psychrometric feedback of the released latent heat on
`q_v,sat` into the timescale,

```math
\partial_t q = \frac{s}{\tau\,\Gamma}, \qquad
\Gamma = 1 + \frac{L}{c_p}\frac{\partial q_{v,\mathrm{sat}}}{\partial T} \quad (\texttt{CMNonEq.gamma\_helper}),
```

a closure valid when the substep temperature co-adjusts with the condensate within the same relaxation rather
than being tracked as a separate state (a limited branch, active when the relaxation would exceed the donor's
own mass, replaces `s` with the donor-capped `-min(-s, max(0, q_limit))`; both closures carry it identically).
`Temperature2MP3Tendency` instead evaluates the bare relaxation `s/τ` and represents the feedback through the
coupled temperature. The two are related exactly: `_per_process_2mp3`'s breakdown `pp`, shared by both
closures, already contains the folded rate, and
[`_temperature_2mp3_tendency`](@ref CloudMicrophysics.BulkMicrophysicsTendencies._temperature_2mp3_tendency)
adds `(Γ - 1)` times it back to recover the bare rate,

```math
\frac{s}{\tau\Gamma} + (\Gamma - 1)\frac{s}{\tau\Gamma} = \Gamma \cdot \frac{s}{\tau\Gamma} = \frac{s}{\tau},
```

so the bare rate is `Γ` times the folded rate, both evaluating the same physical rate through different
algebra. [`_jacobian_2mp3t_manual`](@ref CloudMicrophysics.BulkMicrophysicsTendencies._jacobian_2mp3t_manual)
builds its species block the same way, replacing the folded closed-form entries with their bare counterparts
before appending the temperature row and column; the shared thermodynamic quantities (`τ`, `Γ`, latent heats,
the moist heat capacity) come from
[`_phase_relaxation_context`](@ref CloudMicrophysics.BulkMicrophysicsTendencies._phase_relaxation_context),
evaluated once per substep and consumed by both.

### Why the two Jacobians differ by Γ², not a residual

Write `s`'s dependence on the condensed mass `q` along the path both closures generate: condensing an
increment of `q` draws `q_v` down directly and, through the latent-heat response of the coupled temperature
(weight `c = L/c_p` on the mass species, `0` on number and rime species - the same vector
`Temperature2MP3Tendency`'s temperature tendency uses), draws `q_v,sat` up in proportion. The total
draw-down of `s` per unit `q` condensed is

```math
\frac{\mathrm{d}s}{\mathrm{d}q} = -\Bigl(1 + \frac{L}{c_p}\frac{\partial q_{v,\mathrm{sat}}}{\partial T}\Bigr) = -\Gamma,
```

the same `Γ` named above. Both closures share this slope - it is what fixes the equilibrium `q_eq - q = s/Γ`
common to both - but they represent the *rate* of approach to that equilibrium differently. On the
temperature-coupled system, `s` evolves continuously as `T` responds to the bare relaxation `∂ₜq = s/τ`:

```math
\dot s = \frac{\mathrm{d}s}{\mathrm{d}q}\,\dot q = -\Gamma \cdot \frac{s}{\tau} = -\frac{\Gamma}{\tau}\,s,
```

a feedback-*accelerated* decay. The folded closure instead posts the rate `s/(τΓ)` directly; read through the
same `ds/dq = -Γ` slope, it corresponds to `ṡ = -Γ·s/(τΓ) = -s/τ` - the *bare, unaccelerated* decay. Both
trajectories reach the identical equilibrium, but at rates that differ by `Γ`, and since the condensate
Jacobian is one further factor of `ds/dq = -Γ` removed from the supersaturation rate, the two closures'
condensate Jacobians differ by `Γ²`:

```math
\frac{\partial_q\bigl(s/(\tau\Gamma)\bigr)}{\partial_q\bigl(\text{eliminated bare rate}\bigr)}
 = \frac{-1/(\tau\Gamma)}{-\Gamma/\tau} = \frac{1}{\Gamma^2}.
```

This is a leading-order structural difference between the two closures, not a small residual: at a
representative mixed-phase state (`ρ = 0.85` kg m⁻³, `T = 265` K, cloud liquid, rain, and rimed ice all
present and supersaturated), the ice channel has `Γ_ice = 1.69` and the measured ratio of the two ice-channel
Jacobian entries is `2.86`, matching `Γ_ice² = 2.85` (see Numerical verification below).

### The general block-elimination identity

Independent of which closure `f(q, T)` computes, the temperature-coupled system's structure admits an exact
reduction. Write the eight-species tendency as `f(q, T)` with Jacobian blocks `f_q = ∂f/∂q`, `f_T = ∂f/∂T`.
`MicroState2MP3T`'s temperature equation is exactly the latent-heat combination of the species tendency,
`Ṫ = cᵀf(q, T)` with the constant `c` above, so its own Jacobian blocks are `cᵀ` applied to the species
block: the temperature row is `cᵀf_q` and the temperature corner is `cᵀf_T`. Both hold to floating-point
precision by construction (verified below) - this is how `_temperature_2mp3_tendency` and
`_jacobian_2mp3t_manual` build the temperature row and column, not an independent physical assumption.

The 9×9 Rosenbrock substep solves `(I/h - J₉)Δy = f(y)` for `y = (q, T)`, with

```math
J_9 = \begin{pmatrix} f_q & f_T \\ c^\mathsf{T}f_q & c^\mathsf{T}f_T \end{pmatrix}.
```

Impose the ansatz `ΔT = cᵀΔq` - the temperature increment tracks the species increment through the same
weights as the tendency itself - and substitute into the species-block row:

```math
\Bigl(\frac{I}{h} - f_q - f_T c^\mathsf{T}\Bigr)\Delta q = f_q(y).
```

Left-multiplying this 8×8 equation by `cᵀ` reproduces the temperature-row equation under the same ansatz
exactly, so the temperature-row equation is satisfied automatically whenever the species-block equation is.
The 9×9 step therefore reduces exactly to an 8×8 step with effective Jacobian

```math
J_8^{\mathrm{elim}} = f_q + f_T\,c^\mathsf{T}, \qquad \Delta T = c^\mathsf{T}\Delta q,
```

for any `f` satisfying `Ṫ = cᵀf` with `c` constant. The identity needs two conditions: `c` constant (in the
code, `L` and `c_p` are evaluated once per substep and not differentiated; the folded closure's own `∂Γ/∂q`
chain through the moist heat capacity's condensate dependence is, in this respect, a refinement beyond a
constant-`c` treatment, not a term the identity above predicts), and `T` staying on the latent-heat manifold
`dT = cᵀdq` - true for the phase-change and freezing/melting sources here, since every temperature change in
this substep is the latent heat of a resolved species tendency, but not true of a temperature tendency from
any other source (radiative heating, advection, mixing), which the 8×8's analytic elimination has no state to
represent and the 9×9 does.

`J₈ᵉˡⁱᵐ` is the exact reduction of the *bare* closure's own 9×9 (block-eliminating `Temperature2MP3Tendency`'s
Jacobian): the Γ² relation above is exactly the statement that `J₈ᵉˡⁱᵐ` does not equal `ManualJacobian`'s
folded closure. `ManualJacobian` is not, and is not intended as, the general block-elimination result applied
to the bare closure - it is a separately designed closure that represents the same feedback through the
Γ-scaled relaxation timescale instead.

### Consequences for substep behavior

The two closures' condensate Jacobians differ in magnitude by Γ² (previous section), and for the physically
expected sign (`Γ > 1`, condensation warming that raises `q_v,sat` and throttles further condensation) the
folded closure is the smaller-magnitude, less stiff one: `1/(τΓ)` is smaller than `Γ/τ`. A smaller Jacobian
magnitude is expected to tolerate a coarser substep at a given accuracy target, so the folded 8×8 may sustain
larger substeps than the temperature-coupled 9×9 before the linearization error becomes visible. The 9×9
carries the feedback-accelerated transient - arguably the more faithful one if `τ` is read as the
fixed-temperature microphysical relaxation timescale it is defined at - at the cost of representing a
genuinely faster, stiffer process. The `dt`-ladder comparison planned between `rosenbrock_manual()` and
`rosenbrock_manual_temperature()` therefore compares two closures of the phase-change physics that differ
already at the raw-tendency level, not two numerical treatments of one shared model; which transient a
production configuration should use is a modeling question this section does not settle, and remains open for
review.

### Numerical verification

Verified directly against the code (script and instructions preserved at
`p3-2mp3-stability/20260723-combined/verify_9x9_8x8_gamma_mismatch.jl` in the campaign scratch directory), at
the representative state above:

- `Ṫ = cᵀf`: `f₉[9]` and `c·f₉[1:8]` agree to `1e-14` relative.
- The temperature row and corner: `J₉` row 9 vs. `cᵀf_q`, and `J₉[9,9]` vs. `cᵀf_T`, agree to floating-point
  roundoff (`~1e-9`, `~1e-11`).
- `J₈ᵉˡⁱᵐ = f_q + f_T·cᵀ` does not match `ManualJacobian`'s actual output: the maximum absolute difference
  between the two matrices is `~10²`, against diagonal entries of order `10²`-`10⁴`.
- Isolated on the ice channel, where the condensation/deposition entry dominates the diagonal, the ratio
  `J₈ᵉˡⁱᵐ[ice,ice] / J₈[ice,ice]` is `2.86`, against `Γ_ice² = 2.85`. The cloud-liquid channel's full diagonal
  entry looks deceptively close (ratio `1.00001`) because autoconversion, accretion, and Bigg immersion -
  donor-linearized terms identical in both matrices, since they do not involve `Γ` - dominate the same
  diagonal entry at this state and dilute the same underlying Γ² mismatch; isolating the Tier-1 condensation
  entry alone reproduces the same structure as the ice channel.

The invariant the [`_per_process_2mp3`](@ref CloudMicrophysics.BulkMicrophysicsTendencies._per_process_2mp3)
breakdown itself satisfies - that summing it reproduces the raw entry tendency bit for bit - is checked by the
"verbose instantaneous parts sum to total" test (`rosenbrock_verbose_tests.jl`); no automated test yet checks
the block-elimination identity above.

## The coarse-step ice-growth instability

At a cold, ice-supersaturated state carrying supercooled cloud liquid, the ice-growth tendency has an
autocatalytic mode: rime mass grows by collecting cloud droplets, and denser rimed particles fall faster and
sweep out more liquid, so the rime-mass tendency increases with rime mass. The exact Jacobian carries this as a
positive diagonal in the rime-mass (`q_rim`) row. At a representative state — `ρ = 1.0` kg m⁻³, `T = 263` K,
`q_tot = 10⁻²`, `q_lcl = 2 × 10⁻³`, `n_lcl = 10⁸` m⁻³, `q_ice = 2 × 10⁻³`, `n_ice = 10⁴` m⁻³, unrimed, ice
supersaturation `S_ice ≈ 1.8` — the diagonal is `+5 × 10⁻² s⁻¹` (time scale ≈ 20 s, identical in `Float32` and
`Float64`), and it grows past `10⁻¹ s⁻¹` at colder, more liquid-rich states. The mode requires supercooled
liquid: with the same ice state but no cloud liquid the rime-mass diagonal falls to order `10⁻⁵ s⁻¹`, and the
pure-deposition diagonal is negative there (vapor depletion opposes further deposition). With the exact Jacobian
and `ImplicitGrowth`, the implicit operator `I/h − J` loses positive-definiteness once the growth eigenvalue
exceeds `1/h`, i.e. once the substep is coarse relative to the growth time scale. The single substep then
overshoots the nonlinear limit the linear operator does not see: ice is over-grown past the available condensate,
the latent heating drives a spurious temperature excursion, and the state goes non-physical. This crash is a
property of the single-column convective configuration, not of an isolated cell; the growth diagonal above is an
isolated-cell measurement, but the crash itself appears only in the coupled single-column run.

### What resolves it

The exact preset removes the growth mode from the implicit operator and bounds the now-explicit growth by the
physical saturation limit:

- **`ExplicitGrowthDiagonal`** zeros the positive diagonal, so the implicit operator carries only non-positive
  modes and is well-conditioned at any substep size. The exact off-diagonal couplings (which the donor-based
  matrix drops) are retained, so accuracy at cold, supersaturated cells is better than the donor scheme.
- **`EndStateSaturationAdjustment`** scales the substep increment by the largest `s ∈ [0, 1]` for which the
  latent-heated end state stays at or above saturation over its more-supersaturated phase (`max(S_ice, S_liq)`;
  see the next section). It acts only on cells that begin at or above saturation (a subsaturated, evaporating or
  sublimating cell cannot over-deposit, so its increment is returned unchanged). It is a no-op at fine substeps
  and engages only when the full step would cross saturation. The bisection count is set from the float
  precision.

Both pieces are required: zeroing the growth diagonal alone leaves the explicit growth unbounded at the coarsest
single-substep steps, and the saturation adjustment supplies the missing nonlinear bound. Together they make the
exact scheme robust across the resolved time-step envelope.

!!! note "Use two or more substeps for accurate climate"
    At a single substep the explicit growth is bounded only by the saturation adjustment, which over-produces
    precipitation at coarse time steps. Two or more substeps recover accurate precipitation; the saturation
    adjustment is then rarely active.

## The mixed-phase saturation criterion

A cell has two saturation thresholds, over liquid and over ice, and the condensation and deposition processes draw
on a single shared vapor reservoir. `EndStateSaturationAdjustment` limits the increment on the more-supersaturated
phase, `max(S_ice, S_liq)`: it keeps the latent-heated end state at or above the saturation of whichever phase
carries the larger supersaturation. Equivalently the vapor floor is the lower of the two saturation specific
humidities, since the smaller `q_sat` is the larger supersaturation.

The two saturation curves cross at the freezing point (panel (c) below). Below freezing `q_sat_ice < q_sat_liq`, so
ice carries the larger supersaturation and the criterion binds on ice — reducing exactly to the ice-saturation
limit that bounds the ice-growth instability, so the cold behaviour is unchanged. Above freezing the curves swap
and the criterion binds on liquid; an ice-only limit there would stop vapor depletion early and leave the cell
supersaturated over liquid, suppressing warm cloud, which binding on the more-supersaturated phase avoids.

A single 0-D parcel that exchanges vapor with cloud liquid and cloud ice only (no collection or precipitation)
isolates the mixed-phase physics behind the two thresholds. Cloud liquid is kinetically fast and cloud ice slow, so
while liquid is present it pins the vapor near water saturation (panel (a), `S_liq ≈ 0`); the parcel stays
supersaturated over ice (`S_ice > 0`) and ice deposits, drawing mass from the evaporating liquid (panel (b)) — the
Wegener–Bergeron–Findeisen transfer. Only once the liquid is exhausted does the vapor relax to ice saturation
(`S_ice → 0`). The drawdown from water saturation to ice saturation is therefore inherently a multi-step process at
the resolved time step.

```@example
include("plots/SatAdjustmentWBF_plots.jl")
```
![](SatAdjustmentWBF.svg)

Binding on `max(S_ice, S_liq)` is a single shared scalar on the whole increment, which is what keeps the limiter
stable: scaling processes independently breaks the coupling between paired transfers and the shared vapor draw. The
criterion binds the correct phase wherever a single phase grows from vapor — ice only (the cold deposition cells,
where it reduces to the ice limit) or liquid only (warm cells). When both phases are supersaturated below freezing,
a vigorous mixed-phase updraft core, the criterion binds on the ice floor (the more-supersaturated phase there) and
so condenses the co-present, still-growing liquid past its own saturation in a single step rather than transferring
it gradually. A fully per-phase floor — binding on the first growing phase to saturate and letting the slower phase
relax over subsequent steps — would represent the gradual transfer more faithfully and is a candidate refinement.
The distinction is inactive at fine substeps, where the limiter rarely engages, and does not affect the cold
instability resolution, which is set by the ice floor.

## Approaches that did not resolve the instability

These were tried and are not part of the supported framework.

- **Field-of-values growth clamp** (a uniform diagonal shift bringing the operator's rightmost eigenvalue to
  `α/h`). It stabilizes the linear operator but does not bound the single-step explicit overshoot of the
  nonlinear source as it approaches saturation: the crash is a saturation overshoot, not an operator
  amplification, so the shift only delays it. With `α` near one the near-singular resolvent it leaves amplifies
  the growth into a larger overshoot.
- **Diagonal growth clamp** (limit each positive diagonal to `α/h`). It shares the same limitation, and limiting
  a positive diagonal balanced by off-diagonal structure can itself destabilize an otherwise-stable step.
- **Smooth species mask** (a differentiable replacement for the near-empty species mask). At coarse single
  substeps it routes activating ice and liquid species to a forward-Euler step, which itself overshoots the fast
  growth.
- **Implicit temperature** (promoting `T` into the implicitly solved state). It removes the error of the
  operator-split between-substep temperature update, but the dominant brake on the growth — the nonlinear
  condensate depletion — is not linear, so a linear implicit temperature feedback does not bound the growth
  overshoot at fixed coarse substeps.
