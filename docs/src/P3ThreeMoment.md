# [Three-moment ice](@id P3-three-moment-ice)

The three-moment closure adds the sixth moment of the ice size distribution,
``M_6``, as a third prognostic moment alongside the number ``N`` and the mass
content ``L``.
Predicting ``M_6`` lets the shape parameter ``\mu`` evolve with the flow rather
than follow a diagnostic slope law, which relaxes the mean-size limiter that
two-moment ice needs to suppress excessive size sorting [Milbrandt2021](@cite).

The closure is selected with the `moments = :three_moment` option on
[`ParametersP3`](@ref), which stores the shape bound ``\mu_{max}``, the
initiation shape parameter ``\mu_{init}``, and the mean-mass band used to bound
the reflectivity-tendency coefficients.
The two-moment path is unchanged.

## Prognostic moment and the size distribution

The number-size distribution is the generalized gamma

```math
N'(D) = N_0 \, D^\mu \, e^{-\lambda D},
```

with ``N = M_0`` and ``M_6`` complete moments (no mass-regime truncation),
so both are exact gamma integrals.
The volumetric sixth moment is ``\rho z_\mathrm{ice} = M_6`` [m``^6`` m``^{-3}``].
It is zero under two-moment ice.

## Shape solve: the one-dimensional reduction

Because ``N`` and ``M_6`` are complete moments,

```math
\frac{Z}{N} = \frac{M_6}{M_0}
            = \frac{\Gamma(\mu+7)}{\Gamma(\mu+1)}\, \lambda^{-6},
```

which pins the slope in closed form for any candidate ``\mu``:

```math
\log \lambda(\mu) = \tfrac{1}{6}\left[\log\Gamma(\mu+7) - \log\Gamma(\mu+1) - \log(Z/N)\right].
```

Substituting ``\log\lambda(\mu)`` into the mass target reduces the two-variable
inversion to a one-dimensional root find in ``\mu``,

```math
r(\mu) = \log\!\frac{L}{N}\Big(\mu, \log\lambda(\mu)\Big) - \log\!\frac{L}{N}\Big|_\text{target} = 0,
```

where the model ``\log(L/N)`` uses the piecewise mass moment
([`logmass_gamma_moment`](@ref)) and the complete number moment
([`loggamma_moment`](@ref)).
The mass target is regularised exactly as in the two-moment solve
([`get_distribution_logλ`](@ref)): the mass and number are floored inside the
logarithm so the target is finite and continuous across ice onset.

The residual ``r(\mu)`` is strictly increasing in ``\mu`` on ``[0, \mu_{max}]``
across the sampled rime states and ``Z/N`` targets, in both `Float64` and
`Float32`, so the root is unique.
The sweep that establishes this is part of the test suite
(`test/p3_three_moment_tests.jl`).
The solve reuses the branchless Brent method with a fixed iteration budget
(8 in `Float32`, 10 in `Float64`) and a deterministic bracket ``[0, \mu_{max}]``
with no warm start, so the iteration count is input-independent.

The shape closure ``G(\mu) = M_0 M_6 / M_3^2`` is an identity here, not a
separate equation:

```math
G(\mu) = \frac{\Gamma(\mu+7)\Gamma(\mu+1)}{\Gamma(\mu+4)^2}
       = \frac{(\mu+6)(\mu+5)(\mu+4)}{(\mu+3)(\mu+2)(\mu+1)},
```

decreasing monotonically from ``G(0) = 20`` toward ``G(\infty) = 1``
([`G_of_μ`](@ref)).
The third moment ``M_3`` is the exact analytic third moment of the solved gamma,
``M_3 = N \,\Gamma(\mu+4)/\Gamma(\mu+1)\,\lambda^{-3}``, so no bulk-density
estimate enters the closure.

### The log(λ) bound clamp inside the residual

The size bounds ``\log\lambda \in [`` `LOGλ_MIN` ``,`` `LOGλ_MAX` ``] = [2, 17]``
(mean size ``1/\lambda`` from 0.04 µm to 135 mm) are enforced by clamping
``\log\lambda(\mu)`` inside the residual and returning the same clamped value.
Clamping only the returned ``\log\lambda`` while solving ``\mu`` against the
unclamped slope would leave the ``(\mu, \log\lambda)`` pair inconsistent once a
bound binds, missing the mass target.
Clamping inside the residual keeps the returned pair self-consistent, preserves
monotonicity, and keeps the full ``\mu \in [0, \mu_{max}]`` range as the shape
limiter.
For physical ice sizes the bounds are interior (``\log\lambda \approx 4.6`` to
``11.5``), so this clamp changes nothing in the physical regime.

## The reflectivity limiter

The admissible window ``\mu \in [0, \mu_{max}]`` is enforced by the ``\mu``-clamp
inside the one-dimensional solve, applied before any integrand is built (the
shape is solved first, and all integrands take the solved shape).
No separate clamp of ``Z`` against an estimated ``M_3`` is needed.

Two regularisations keep the solve well posed:

  - The ratio ``Z/N`` is floored as a single quantity into a physical window,
    never the components ([`reflectivity_number_ratio`](@ref)).
    The window ``[zn_{lo}, zn_{hi}]`` is the ``Z/N`` range spanned by the
    ``\mu`` and ``\log\lambda`` bounds, cached on the `ThreeMoment` closure,
    with the lower bound floored by `floatmin` for `Float32` finiteness.
    At the window's lower bound, ``\log\lambda(\mu)`` exceeds the maximum size
    bound for every ``\mu``, so the residual clamp saturates ``\log\lambda``
    there and ``\mu`` follows the mass target; the clamp keeps the map from the
    targets continuous.
  - A construction-time admissibility clamp bounds the recovered
    ``\rho z_\mathrm{ice}`` to physical magnitudes ([`apply_z_bounds`](@ref)),
    the analogue of the Fortran `apply_mui_bounds_to_zi`.
    The tight ``[G(\mu_{max}), G(0)]\, M_3^2/N`` window requires the slope
    ``\lambda`` (hence the shape), which is not available at construction.
    A solid-ice ``M_3`` estimate would systematically clamp valid low-density
    ice, so the construction clamp instead bounds ``\rho z_\mathrm{ice}`` into
    ``[zn_{lo}, zn_{hi}] \cdot \rho n_\mathrm{ice}``, the same window the solve
    uses, and the exact per-state ``\mu \in [0, \mu_{max}]`` limiting is left to
    the ``\mu``-clamp in the shape solve.

At exactly ``\rho n_\mathrm{ice} = 0`` the construction clamp stores
``\rho z_\mathrm{ice} = 0`` and the ratio evaluates at the window's lower bound,
while the limit ``\rho n_\mathrm{ice} \to 0^+`` with a saturated sixth moment
approaches the upper bound.
The shape (and with it the shape-only ``V_z``) therefore changes discontinuously
at the empty state, but every extensive quantity, including the sedimentation
flux ``V_z \cdot \rho z_\mathrm{ice}``, vanishes continuously with
``N_0 \to 0``; the decoupled onset sweep in `test/p3_three_moment_tests.jl`
documents this behavior.

## Per-process reflectivity tendencies

The default closure holds ``\mu`` fixed across growth and decay and adds an
initiation term for each new-ice process, following [Milbrandt2021](@cite).
Processes are grouped by how they change the moments.

| Process group | Ice changes | Reflectivity form |
|:---|:---|:---|
| Growth/decay (riming, collection, deposition, sublimation, melt, aggregation, number adjustment) | net ``dL``, ``dN`` | constant-``\mu`` (Eq. 10) |
| Deposition/heterogeneous nucleation, ice multiplication | ``dN > 0`` at ``D_{nuc}`` | monodisperse initiation (Eq. 9), ``\mu_{init}`` |
| Cloud/rain drop freezing | ``dN > 0`` from a drop PSD | drop-freezing initiation (Eq. 9), ``\mu`` of the source PSD |

The growth term is linear in the net rates, so it is one post-pass over the
summed Group-2 ``(dL, dN)`` with frozen coefficients
([`reflectivity_growth_tendency`](@ref)):

```math
\frac{dZ}{dt}\bigg|_\text{growth}
   = G(\mu)\left[2\,\frac{M_3}{N}\frac{M_3}{L}\,dL - \left(\frac{M_3}{N}\right)^2 dN\right].
```

The three coefficients ``(G, M_3/N, M_3/L)`` are computed once per substep by
[`reflectivity_growth_coefficients`](@ref) from the pre-substep state and shape
and held fixed, exactly as the shape itself is frozen; a Jacobian therefore
differentiates the growth term only through the rates ``(dL, dN)``.
``M_3/N`` is the analytic ratio ``\Gamma(\mu+4)/\Gamma(\mu+1)\,\lambda^{-3}``;
the ``M_3/L`` factor uses the mean particle mass ``L/N`` clamped into the
``[`` `mean_mass_min` ``,`` `mean_mass_max` ``]`` band, so the coefficients stay
bounded as the category empties.

The initiation terms conserve the three moments of the transferred ice
([`reflectivity_initiation_monodisperse`](@ref),
[`reflectivity_initiation_freezing`](@ref)).
For monodisperse nucleation at ``D_{nuc}`` the form is division-free,
``dZ_\mathrm{init} = G(\mu_{init})\, D_{nuc}^6\, dN``.
For drop freezing the frozen mean drop mass ``\bar m = dq/dN`` is clamped into
the mean-mass band,
``dZ_\mathrm{init} = G(\mu_\text{source})\,(6/\pi\rho_\text{new})^2\,\bar m\, dq``.

The [`ZContribution`](@ref) accumulator stores the growth rates and the summed
initiation rate, plus a reserved term for the moment-6-weighted rates of an
evolving-``\mu`` closure that is not implemented here.
[`reflectivity_tendency`](@ref) assembles the total ``d\rho z/dt`` from an
accumulated contribution.

## Sedimentation

The reflectivity moment sediments with its own fall speed, the sixth-moment
weighted mean

```math
V_z = \frac{\int D^6\, v(D)\, N'(D)\, dD}{\int D^6\, N'(D)\, dD},
```

computed as one fused quadrature pass over a numerator and denominator sharing
the same nodes ([`ice_terminal_velocity_reflectivity_weighted`](@ref)).
The ``D^6`` weight is folded into the log-exponent of the integrand,
``\exp(\log N_0 + (\mu+6)\log D - \lambda D)\, v(D)``, so the weighted number does
not underflow in `Float32`.
The integration bounds use ``\text{moment\_order} = 6``, which shifts the tail
quantiles and places the integrand mode ``(\mu+6)/\lambda`` on a subinterval
boundary.
The mode breakpoint generalisation applies only to weighted integrals; the
``\text{moment\_order} = 0`` bounds keep the decay-scale breakpoint ``3/\lambda``
unchanged.
The denominator is floored by `floatmin` for finiteness only, and the ratio is a
bounded weighted average of ``v`` on the integration window, continuous as the
category empties.

``V_z`` weights the largest particles, so it is the largest of the three ice
fall speeds and sets the sedimentation stability limit.
The reflectivity integrals use Gauss-Legendre order 12, selected by the
reflectivity level of the quadrature error study
(`run_reflectivity_error_study` in `test/p3_quadrature_error_study.jl`), which
sweeps the order against a Gauss-Legendre(128) reference across three-moment
states derived from the study's column and hail-core states with
``\mu \in \{0, 5, 10, 20\}``.
The sweep shows order 6 (the mass/number default) insufficient for a transport
quantity (maximum relative error ``3 \times 10^{-4}``), the `Float32` error
saturating at its precision limit (``\approx 10^{-6}``) from order 10, and a
`Float64` maximum error of ``4 \times 10^{-10}`` at order 12.
The order is the module constant `REFLECTIVITY_QUADRATURE_ORDER`, overridable
per call through the `order` keyword argument.

## Advection transform

The moment ratios, and hence ``\mu``, are best preserved under transport by
advecting ``\sqrt{\rho n_\mathrm{ice}\, \rho z_\mathrm{ice}}`` rather than
``\rho z_\mathrm{ice}`` directly [Milbrandt2021](@cite).
The host holds the advected variable as its prognostic; the host-model coupling
transforms it at the boundaries:

```math
\rho z_\mathrm{adv} = \sqrt{\rho n_\mathrm{ice}\, \rho z_\mathrm{ice}},
\qquad
\rho z_\mathrm{ice} = \frac{\rho z_\mathrm{adv}^2}{\max(\rho n_\mathrm{ice}, n_\text{presence})}.
```

The forward transform ([`advected_reflectivity`](@ref)) is continuous as either
moment goes to zero.
The recovery ([`reflectivity_from_advected`](@ref)) floors the number by the
presence scale ``n_\text{presence}`` so the ``0/0`` at vanishing number is finite.
The state construction re-imposes the admissibility window on the recovered
``\rho z_\mathrm{ice}`` immediately, which bounds the value that decoupled
transport of the number and the advected variable can produce.
Sedimentation acts on the true ``\rho z_\mathrm{ice}`` with ``V_z``, while the
resolved transport acts on ``\rho z_\mathrm{adv}``.

## Host wiring contract

The three-moment scheme is selected with
`Microphysics2MParams(FT; with_ice = true, moments = :three_moment)` and enters
the bulk-tendency interface through the packed 2M+P3 entry; there is no
positional form (the `logλ`-based positional wrappers require a slope law and
remain two-moment only).
Per grid cell and substep, the host:

 1. recovers the volumetric sixth moment from its advected prognostic,
    ``\rho z_\mathrm{ice} = `` [`reflectivity_from_advected`](@ref)`(ρz_adv, ρn_ice, n_presence)`,
    with `n_presence = moments.n_presence`, the number presence scale on the
    `ThreeMoment` closure (ClimaParams `P3_ice_number_presence_concentration`,
    ``10^{-3}`` m``^{-3}``);
 2. builds the state and solves the shape once,
    `state = state_from_prognostic(params, ρq_ice, ρn_ice, ρq_rim, ρb_rim, ρz_ice)`
    (the construction clamps ``\rho z_\mathrm{ice}`` into the admissible window)
    and `shape = get_distribution_shape(state)`;
 3. calls the packed entry with the per-category input
    `ice = ((; q_ice, n_ice, q_rim, b_rim, z_ice),)` where
    `z_ice = ρz_ice / ρ` [m⁶/kg], and `shapes = (shape,)`.

The returned tendency `NamedTuple` gains a `dz_ice_dt` field [m⁶/kg/s] after
the rime-volume field, both in the instantaneous entry and in the
`RosenbrockAverage` modes (`ExactJacobian` and `ManualJacobian`; the `Verbose`
diagnostic path does not support three-moment ice).
The sixth-moment sedimentation velocity is
[`ice_terminal_velocity_reflectivity_weighted`](@ref)`(velocity_params, ρ, state, shape)`,
and the outbound transform is [`advected_reflectivity`](@ref)`(ρn_ice, ρz_ice)`.

Inside the entry, the reflectivity tendency assembles as the constant-μ growth
term over the net growth/decay ice rates plus the initiation terms:
deposition nucleation (monodisperse at ``D_{nuc}`` with ``\mu_{init}``), cloud
immersion freezing (drop form; the SB2006 cloud PSD is a generalized gamma in
mass, not a gamma in diameter, so its shape parameter falls back to
``\mu_{init}``), and rain freezing (drop form with ``\mu = 0``, the exponential
SB2006 rain PSD).
The growth coefficients are frozen per entry call from the input state and
shape, so within a Rosenbrock substep the reflectivity row of the Jacobian is
the frozen linear combination ``c_q \cdot (q_\mathrm{ice}\ \text{row}) + c_n
\cdot (n_\mathrm{ice}\ \text{row})`` and the reflectivity column is zero
(receiver-only; verified by a ForwardDiff test through the entry).

Under three-moment ice the number-adjustment mean-size limiter reads the
closure's ``[`` `mean_mass_min` ``,`` `mean_mass_max` ``]`` band, whose relaxed
upper bound (the 400 mm mean-size equivalent, `P3_ice_mean_mass_max`) leaves
size sorting to the prognostic ``\mu``; the two-moment scheme keeps its fixed
``[10^{-12}, 10^{-5}]`` kg band.

For box and column runs, use `rosenbrock_manual()`: the hand-built Jacobian is
finite on every tested state.
The `ForwardDiff` Jacobian of the liquid-ice collision quadrature can produce
non-finite entries in `Float32` at some (state, shape) combinations, which
routes those substeps to the forward-Euler fallback; three-moment shapes reach
such combinations more often than the two-moment slope-law shapes.
The trigger fraction is quantified in `test/p3_three_moment_tests.jl`
(`test_3m_exact_jacobian_f32_fragility`); resolving the `Float32`
differentiation sensitivity of the collision quadrature is open follow-up work.

## Behavior under two-moment ice

Under the default two-moment closure ``\rho z_\mathrm{ice}`` is zero,
[`apply_z_bounds`](@ref) returns zero, and no reflectivity tendency, fall speed,
or transform is evaluated.
The shape comes from the slope law as before, so the two-moment path reproduces
its previous output byte for byte.
